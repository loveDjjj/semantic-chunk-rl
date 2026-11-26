"""Bounded asynchronous decoding and frozen reward workers in isolated environments."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from scrl.runtime.transport import atomic_json
from scrl.plan import digest
from scrl.runtime.transport import receive_message, send_message
from scrl.rl.reward import RewardBundle, ScopeScores, UnscorableReward



class ResidentProcess:
    """Single request at a time per model; bounded socket waits, no filesystem polling."""

    def __init__(self, kind, settings, directory, timeout=1200):
        self.directory, self.timeout = Path(directory).resolve(), timeout
        self.directory.mkdir(parents=True, exist_ok=False)
        atomic_json(self.directory / "config.json", settings)
        self.log = (self.directory / "worker.log").open("a")
        from scrl.config import SOURCE
        root = SOURCE
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        self.socket, child = socket.socketpair()
        self.socket.settimeout(timeout)
        self.channel = self.socket.makefile("rwb")
        self.lock, self.ready_value, self.broken = threading.Lock(), None, False
        module = "scrl.models.scoring.vae_worker" if kind == "vae" else "scrl.models.scoring.worker"
        command = [settings["python"], "-m", module, "--directory", str(self.directory),
                   "--device", settings["device"], "--socket-fd", str(child.fileno())]
        if kind != "vae":
            command.extend(["--kind", kind])
        try:
            self.process = subprocess.Popen(command, cwd=root, env=env, pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.channel.close()
            self.socket.close()
            self.log.close()
            raise
        finally:
            child.close()

    def check(self):
        if self.broken or self.process.poll() is not None:
            raise RuntimeError(f"Service failed; see {self.directory / 'worker.log'}")

    def _receive(self):
        try:
            value = receive_message(self.channel)
            if value.get("state") == "failed":
                raise RuntimeError(value["error"])
            return value
        except BaseException as error:
            self.broken = True
            raise RuntimeError(f"Service failed: {self.directory / 'worker.log'}: {error}") from error

    def ready(self):
        with self.lock:
            if self.ready_value is None:
                self.check()
                self.ready_value = self._receive()
            return self.ready_value

    def score(self, request):
        self.ready()
        with self.lock:
            self.check()
            try:
                send_message(self.channel, {"op": "score", "request": request})
                reply = self._receive()
                if reply.get("state") == "unscorable":
                    raise UnscorableReward(reply["error"])
                return reply["result"]
            except UnscorableReward:
                raise
            except BaseException:
                self.broken = True
                raise

    def close(self):
        if self.process.poll() is None:
            if self.ready_value is not None and not self.broken:
                try:
                    send_message(self.channel, {"op": "stop"})
                except (OSError, ValueError):
                    pass
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.channel.close()
        self.socket.close()
        self.log.close()


class RewardProcess(ResidentProcess):
    pass


def _collect_scores(jobs):
    # Drain mmap readers and never let unscorable evidence mask a broken worker.
    values, error = {}, None
    for key, future in jobs:
        try:
            values[key] = future.result()
        except Exception as caught:
            if error is None or isinstance(error, UnscorableReward):
                error = caught
    if error is not None:
        raise error
    return values


def write_rgb(path, pixels, fps=16):
    """Optional lossless export; training scores raw RGB without calling this function."""
    import imageio_ffmpeg
    height, width = pixels.shape[1:3]
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps,
        codec="libx264rgb", pix_fmt_in="rgb24", pix_fmt_out="rgb24", macro_block_size=1,
        output_params=["-crf", "0", "-threads", "1"])
    writer.send(None)
    try:
        for frame in pixels.contiguous().numpy():
            writer.send(frame)
    finally:
        writer.close()


class VideoRewardService:
    def __init__(self, backend, settings, directory, protocol, *, options=None):
        options = options or {}
        self.global_only = options.get("global_only", False)
        self.protocol = protocol
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workers, self.pool, self.io_pool, self.judge_pool = {}, None, None, None
        self.failure = None
        self.closed = False
        limit = options.get("max_inflight", 2)
        if type(limit) is not int or limit < 1:
            raise ValueError("max_inflight must be a positive integer")
        self.slots = threading.BoundedSemaphore(limit)
        self.scratch = options.get("scratch_directory", "/dev/shm")
        self.structured = 'judge' in settings
        if protocol.split(":", 1)[0] == "event-reward-v2" and not self.structured:
            raise ValueError("event-reward-v2 requires a video judge")
        if not Path(self.scratch).is_dir():
            raise ValueError("Scratch directory must exist on this node")
        vae = {"python": sys.executable, "device": options.get("vae_device", "cuda:3"),
               "upstream": backend.paths.upstream, "wan": backend.paths.wan}
        vae.update(options.get("vae", {}))
        try:
            for kind in (("quality", "judge") if self.structured else ("quality", "alignment")):
                if kind == 'judge' and options.get('judge_port'):
                    from scrl.models.scoring.http_judge import HttpJudge
                    self.workers[kind] = HttpJudge(settings[kind], options['judge_port'], limit)
                    self.judge_pool = ThreadPoolExecutor(limit, thread_name_prefix='vllm-judge')
                else:
                    self.workers[kind] = RewardProcess(kind, settings[kind], self.directory / kind)
            self.workers["vae"] = ResidentProcess("vae", vae, self.directory / "vae")
            for worker in self.workers.values():
                worker.ready()
            self.pool = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="evaluation")
            self.io_pool = ThreadPoolExecutor(max_workers=2 * limit, thread_name_prefix="reward-rpc")
        except BaseException:
            self.close()
            raise

    def submit(self, original, rollout):
        if self.closed:
            raise RuntimeError("Evaluation service is closed")
        self.slots.acquire()
        if self.failure is not None:
            self.slots.release()
            raise RuntimeError("A previous evaluation failed; abort this group") from self.failure
        try:
            # Keep only CPU latents and metadata in the queue, never the replay KV cache/tape.
            future = self.pool.submit(self._evaluate, original, rollout.plan, rollout.final.latents,
                                      rollout.final.prefix_id, rollout.target_segment, time.monotonic())
        except BaseException:
            self.slots.release()
            raise
        def finished(done):
            error = None if done.cancelled() else done.exception()
            if error is not None and not isinstance(error, UnscorableReward):
                self.failure = error
            self.slots.release()
        future.add_done_callback(finished)
        return future

    def __call__(self, original, rollout):
        return self.submit(original, rollout).result()

    def _evaluate(self, original, plan, latents, prefix_id, target_segment, submitted):
        import torch
        key = digest({"prefix": prefix_id, "prompt": original,
                      "plan": plan.to_dict(), "target": target_segment, "protocol": self.protocol})
        folder = self.directory / "videos" / key
        receipt = folder / "reward.json"
        if receipt.exists():
            data = json.loads(receipt.read_text())
            return RewardBundle(ScopeScores(**data["global_scores"]),
                ScopeScores(**data["segment_scores"]), data["protocol"], data["boundary_quality"])
        folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="scrl-rgb-", dir=self.scratch) as temporary:
            latent_path, rgb_path = Path(temporary) / "latents.pt", Path(temporary) / "rgb.npy"
            if isinstance(latents, Path):
                # ExecutionWorker retains the published file until this RPC returns.
                latent_path = latents
            else:
                torch.save(latents, latent_path)
            decoded = self.workers["vae"].score({"latents": str(latent_path), "rgb": str(rgb_path)})
            n = decoded["frames"]
            if n != 12 * plan.total_chunks - 3:
                raise RuntimeError("VAE output differs from plan extent")
            span = plan.spans()[target_segment]
            if self.structured:
                return self._structured_scores(original, plan, span, target_segment, rgb_path,
                    n, decoded, folder, receipt, started, submitted)
            scopes = {"global": (0, n, original),
                      "segment": (span.pixel_start, span.pixel_end, plan.segments[target_segment].caption)}
            if span.pixel_start:
                scopes["boundary"] = (max(0, span.pixel_start - 4), min(n, span.pixel_start + 4), "")
            jobs = []
            score_started = time.monotonic()
            for scope, (begin, end, caption) in scopes.items():
                payload = {"rgb": str(rgb_path), "start": begin, "end": end,
                           "prompt": caption, "frames": end - begin, "fps": 16}
                jobs.append(((scope, "quality"), self.io_pool.submit(self.workers["quality"].score, payload)))
                if scope != "boundary":
                    jobs.append(((scope, "alignment"), self.io_pool.submit(self.workers["alignment"].score, payload)))
            values = _collect_scores(jobs)
            scoring_seconds = time.monotonic() - score_started
        def scores(scope):
            return ScopeScores(values[scope, "quality"]["quality"], values[scope, "alignment"]["semantic"])
        boundary = None if "boundary" not in scopes else {
            "consistency": values["boundary", "quality"]["quality"]["consistency"]}
        result = RewardBundle(scores("global"), scores("segment"), self.protocol, boundary)
        atomic_json(folder / "raw.json", {f"{scope}/{kind}": row for (scope, kind), row in values.items()})
        atomic_json(folder / "timings.json", {"queue_seconds": started - submitted,
                    "vae": decoded, "scoring_seconds": scoring_seconds,
                    "service_seconds": time.monotonic() - started, "transport": "socket+mmap-rgb"})
        atomic_json(receipt, asdict(result))
        return result

    def _structured_scores(self, original, plan, span, target, rgb_path, n, decoded,
                           folder, receipt, started, submitted):
        payload = {'rgb': str(rgb_path), 'start': 0, 'end': n, 'frames': n, 'fps': 16,
                   'original': original, 'caption': plan.segments[target].caption,
                   'segment_start': span.pixel_start, 'segment_end': span.pixel_end}
        event_mode = self.protocol.split(":", 1)[0] == "event-reward-v2"
        if event_mode:
            payload.update(reward_mode="event-v2", target=target, captions=[s.caption for s in plan.segments])
        whole = getattr(self, "global_only", False)
        if whole:
            if not event_mode:
                raise ValueError("Global S/Q requires event evidence")
            payload['global_only'] = True
        jobs = [('judge', (self.judge_pool or self.io_pool).submit(self.workers['judge'].score, payload))]
        scopes = [('global', 0, n)]
        if not whole:
            scopes.append(('segment', span.pixel_start, span.pixel_end))
        for name, begin, end in scopes:
            jobs.append((name, self.io_pool.submit(self.workers['quality'].score,
                dict(payload, start=begin, end=end, frames=end-begin))))
        boundaries = [] if whole else [frame for frame in (span.pixel_start, span.pixel_end) if 0 < frame < n]
        if boundaries:
            jobs.append(('boundaries', self.io_pool.submit(self.workers['quality'].score,
                dict(payload, operation='boundaries', boundaries=boundaries))))
        values = _collect_scores(jobs)
        def scores(scope):
            judged = values['judge']['scopes'][scope]
            return ScopeScores(dict(values[scope]['quality'], **judged['quality']), judged['semantic'])
        boundary = values['boundaries']['boundary_quality'] if boundaries else None
        if event_mode:
            from scrl.models.scoring.events import VERSION
            judged = values["judge"]
            if judged.get("protocol") != VERSION:
                raise ValueError("Expected sampled event evidence")
            if boundaries:
                exempt = {plan.spans()[i].pixel_start for i in judged["assignment"]["cut_before"] if i is not None}
                rows = [r for r in values["boundaries"]["boundaries"] if r["frame"] not in exempt]
                boundary = ({key: sum(r[key] for r in rows) / len(rows) for key in boundary}
                            if rows else None)
                values["boundaries"]["applied_frames"] = [r["frame"] for r in rows]
        result = RewardBundle(scores('global'), ScopeScores({}, {}) if whole else scores('segment'), self.protocol, boundary)
        atomic_json(folder/'raw.json', values)
        atomic_json(folder/'timings.json', {'queue_seconds': started-submitted, 'vae': decoded,
            'quality_seconds': sum(values[k]['seconds'] for k, _, _ in scopes),
            'boundary_seconds': values.get('boundaries', {}).get('seconds', 0.),
            'judge_seconds': values['judge']['seconds'], 'service_seconds': time.monotonic()-started,
            'transport': 'shared-latent+local-rgb', 'decoded_once': True})
        atomic_json(receipt, asdict(result))
        return result

    def close(self):
        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.io_pool is not None:
            self.io_pool.shutdown(wait=True, cancel_futures=True)
        if self.judge_pool is not None:
            self.judge_pool.shutdown(wait=True, cancel_futures=True)
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()


class TimedLock:
    def __init__(self, lock):
        self.lock, self.local = lock, threading.local()

    def __enter__(self):
        start = time.monotonic()
        self.lock.acquire()
        self.local.wait = getattr(self.local, 'wait', 0.) + time.monotonic() - start
        return self

    def __exit__(self, *args):
        self.lock.release()


class TimedWorker:
    def __init__(self, worker, stage, emit):
        self.worker, self.stage, self.emit = worker, stage, emit
        self.clock = None
        if stage in ('quality', 'vae'):
            self.clock = worker.lock = TimedLock(worker.lock)

    def score(self, request):
        if self.clock is not None:self.clock.local.wait = 0.
        start, clock = time.time(), time.monotonic()
        result = self.worker.score(request)
        span = time.monotonic() - clock
        self.emit(dict(video=Path(request['rgb']).parent.name, stage=self.stage, start=start,
                       end=start+span, seconds=span, child_seconds=result.get('seconds'),
                       lock_wait=self.clock.local.wait if self.clock is not None else None))
        return result

    def close(self):
        return self.worker.close()


def trace_workers(service):
    path = service.directory/'stage-spans.jsonl'
    lock = threading.Lock()
    def emit(row):
        with lock, path.open('a') as out:
            out.write(json.dumps(row, separators=(',',':'))+'\n')
    for kind in ('vae','quality','judge'):
        service.workers[kind] = TimedWorker(service.workers[kind],kind,emit)
