"""Wait for one arm, score it once and publish only a complete 16-dimension summary."""

import argparse
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from scrl.evaluation.io import atomic_json, read
from scrl.evaluation.protocol import DIMENSIONS, metric_ready
from scrl.plan import digest


def shard_progress(run, arm, shards, indices):
    if (run / 'STOP').exists() or (run / ('STOP-' + arm)).exists() or (run / 'FAIL.json').exists():
        raise RuntimeError('Generation stopped')
    completed, failed = 0, []
    for i in (range(shards) if indices is None else indices):
        f = run / 'workers' / arm / f'generate-{i}' / 'status.json'
        if f.exists():
            state = read(f)
            if state['shard'] != i:
                raise RuntimeError(f'Generation shard {i} has the wrong identity; inspect {f}')
            if state['state'] == 'failed':
                failed.append(i)
            completed += state['state'] == 'complete'
    return completed, failed


def ready(run, arm, shards, expected, *, indices=None):
    completed, failed = shard_progress(run, arm, shards, indices)
    if failed:
        raise RuntimeError(f'Generation shards failed: {failed}')
    if completed == shards:
        actual = {p.stem for p in (run / 'records' / arm).glob('*.json')}
        if actual != expected:
            raise ValueError(f'Generation ended with {len(actual)} records, not the expected task set')
    return completed


def verify(report, arm):
    raw, agg = report['raw_dimensions'], report['aggregate']
    if (report['arm'] != arm or set(raw) != set(DIMENSIONS) or agg['state'] != 'complete'
            or not all(type(v) in (float, int) and math.isfinite(v)
                       for v in [*raw.values(), *(agg[k] for k in ('quality','semantic','total'))])):
        raise ValueError('Scoring did not produce a complete finite 16-dimension report')


def incremental(run, arm, cfg, tasks, shards, deadline, env, status, *, indices=None, yield_idle=False):
    """One controller, one subprocess per visible device; only published inputs may be scored."""
    devices = [d.strip() for d in env.get('CUDA_VISIBLE_DEVICES', '').split(',')]
    if len(set(devices)) != len(devices) or not all(re.fullmatch(r'[0-9]+|(?:GPU|MIG)-[A-Za-z0-9/-]+',d) for d in devices):
        raise ValueError('Incremental scoring needs an explicit, nonempty CUDA_VISIBLE_DEVICES list')
    folder = run/'evaluation'/arm
    stages = ('static_filter', *DIMENSIONS)
    def valid_receipt(d):
        receipt = folder/(d + '.json')
        if not receipt.is_file():
            return False
        try:
            r = read(receipt)
        except ValueError:
            return False
        return (isinstance(r,dict) and r.get('state') == 'complete' and r.get('dimension') == d and r.get('arm') == arm
                and r.get('protocol_sha256') == cfg['protocol_sha256']
                and isinstance(r.get('inputs_sha256'),str)
                and re.fullmatch(r'[a-f0-9]{64}', r['inputs_sha256']) is not None)

    # Old, unbound receipts are not resumable. Finalization checks binding to current video bytes.
    done = {d for d in stages if valid_receipt(d)}
    if 'static_filter' not in done:
        done.discard('temporal_flickering')
    expected = {row['id'] for row in tasks}
    cmd = [sys.executable, '-m', 'scrl.evaluation.vbench', str(run), '--arm', arm]
    active, failures = {}, {}
    try:
        while True:
            completed, failed_shards = shard_progress(run, arm, shards, indices)
            published = {p.stem for p in (run/'records'/arm).glob('*.json')}
            if published - expected:
                raise ValueError('Found record IDs outside the prepared task set')
            for dev, (p, d, started) in list(active.items()):
                rc = p.poll()
                if rc is None:
                    if time.monotonic() - started > 3*86400:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        rc = p.wait()
                        failures[d] = 'timed out'
                    else:
                        continue
                if rc != 0 or not valid_receipt(d):
                    failures.setdefault(d,f'exit={rc}, no valid complete result; inspect {d}.log')
                elif d not in failures:
                    done.add(d)
                del active[dev]
            if completed == shards and published == expected and len(done) == len(stages):
                return True
            running = {v[1] for v in active.values()}
            eligible = [d for d in stages if d not in done and d not in running and d not in failures
                        and metric_ready(d, tasks, published, static_complete='static_filter' in done)]
            progress = dict(completed_shards=completed, failed_shards=failed_shards, shards=shards,
                            records=len(published), missing_records=len(expected-published),
                            completed_dimensions=sorted(done), failed_dimensions=failures)
            if not running and not eligible:
                if yield_idle:
                    status('yielded', **progress, active={}, blocked_dimensions=sorted(set(stages)-done))
                    return False
                if completed + len(failed_shards) == shards or time.monotonic() >= deadline:
                    raise RuntimeError(f'Cannot finalize; missing records={progress["missing_records"]}, '
                                       f'failed shards={failed_shards}, failed dimensions={failures}, '
                                       f'blocked dimensions={sorted(set(stages)-done)}')
            for dev,d in zip((d for d in devices if d not in active), eligible):
                try:
                    with (folder/(d+'.log')).open('a') as log:
                        p = subprocess.Popen([*cmd,'--dimension',d], stdout=log, stderr=subprocess.STDOUT,
                                             env=dict(env, CUDA_VISIBLE_DEVICES=dev), start_new_session=True)
                except OSError as exc:
                    failures[d] = f'could not launch: {exc}'
                    continue
                active[dev] = (p,d,time.monotonic())
            status('scoring' if active else 'waiting', **progress,
                   active={dev:{'dimension':d,'pid':p.pid} for dev,(p,d,_) in active.items()})
            time.sleep(10)
    finally:
        # A stopped watcher must not leave a metric using a card after its owner lock is removed.
        for p,_,_ in active.values():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--shard-indices', help='Comma-separated actual worker indices; count must equal --shards')
    parser.add_argument('--expected-tasks', type=int, required=True)
    parser.add_argument('--wait-seconds', type=int, default=90000)
    parser.add_argument('--incremental', action='store_true',
                        help='Score available dimensions as inputs arrive; one lane per CUDA_VISIBLE_DEVICES entry')
    parser.add_argument('--yield-when-idle', action='store_true',
                        help='Incremental only: release the owner and exit successfully when all remaining dimensions are blocked')
    args = parser.parse_args(argv)
    run, arm = args.run.absolute(), args.arm
    cfg = read(run / 'config.json')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', arm) or arm not in cfg['arms']:
        parser.error('unknown arm')
    if min(args.shards, args.expected_tasks, args.wait_seconds) < 1:
        parser.error('counts and timeout must be positive')
    if args.yield_when_idle and not args.incremental:
        parser.error('--yield-when-idle requires --incremental')
    indices = None
    if args.shard_indices is not None:
        if not re.fullmatch(r'(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*', args.shard_indices):
            parser.error('invalid --shard-indices; use comma-separated nonnegative integers')
        indices = tuple(map(int,args.shard_indices.split(',')))
        if len(set(indices)) != len(indices) or len(indices) != args.shards:
            parser.error('--shard-indices must contain exactly --shards distinct indices')
    tasks = read(run / 'inputs/tasks.json')
    expected = {row['id'] for row in tasks}
    if (len(tasks) != len(expected) or len(expected) != args.expected_tasks
            or digest(tasks) != cfg['scoring_protocol']['tasks_sha256']):
        raise ValueError('Task manifest/count does not match the prepared protocol')
    folder = run / 'evaluation' / arm
    folder.mkdir(parents=True, exist_ok=True)
    lock, summary = folder / '.score-owner', folder / 'summary.json'
    lock.mkdir()  # One owner across shared mounts. Inspect owner before clearing a stale lock.
    def status(state, **details):
        atomic_json(folder / 'watch-status.json', dict(state=state, host=socket.gethostname(),
                    pid=os.getpid(), checked_unix=time.time(), **details))
    try:
        if summary.exists() and not args.incremental:
            verify(read(summary), arm)
            status('complete')
            return
        deadline = time.monotonic() + args.wait_seconds
        weights = Path(cfg['vbench_weights'])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (cfg['vbench_source'], os.environ.get('PYTHONPATH')))),
                   VBENCH_CACHE_DIR=str(weights), VBENCH_BERT_PATH=str(weights / 'bert-base-uncased'),
                   TORCH_HOME=str(weights / 'torch_hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if args.incremental:
            if not incremental(run, arm, cfg, tasks, args.shards, deadline, env, status,
                               indices=indices, yield_idle=args.yield_when_idle):
                return
        else:
            while True:
                completed = ready(run, arm, args.shards, expected, indices=indices)
                if completed == args.shards:
                    break
                status('waiting', completed_shards=completed, shards=args.shards)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Timed out waiting for all generation shards')
                time.sleep(min(30, max(0, deadline - time.monotonic())))
        status('finalizing' if args.incremental else 'scoring')
        candidate = folder / f'.summary-{os.getpid()}-{time.time_ns()}.json'
        with (folder / 'score.log').open('a') as log:
            subprocess.run([sys.executable, '-m', 'scrl.evaluation.vbench', str(run), '--arm', arm,
                            *(['--finalize-only'] if args.incremental else []),
                            '--output', str(candidate)], stdout=log, stderr=subprocess.STDOUT,
                           env=env, check=True, timeout=3*86400)
        verify(read(candidate), arm)
        candidate.replace(summary)
        status('complete')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
    finally:
        lock.rmdir()


if __name__ == '__main__':
    def terminate(*_):
        raise InterruptedError('Score watcher terminated')
    signal.signal(signal.SIGTERM, terminate)
    main()
