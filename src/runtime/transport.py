"""Authenticated worker transport, bounded scoring admission and fail-closed run receipts."""

from collections import deque
from concurrent.futures import Future
from pathlib import Path
import errno
import hashlib
import json
import os
import shutil
import socket
import socketserver
import threading
import time
import traceback
import uuid

from scrl.rl.reward import UnscorableReward


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Peer:
    def __init__(self, address, port, token, timeout=1800):
        self.address, self.port, self.token, self.timeout = address, port, token, timeout

    def call(self, operation, **payload):
        with socket.create_connection((self.address, self.port), timeout=20) as sock:
            sock.settimeout(self.timeout)
            with sock.makefile('rwb') as channel:
                send_message(channel, {'token': self.token, 'operation': operation, **payload})
                result = receive_message(channel)
        if operation == 'evaluate' and result.get('state') == 'unscorable':
            raise UnscorableReward(result['error'])
        if result.get('state') == 'failed':
            raise RuntimeError(f'{self.address}:{self.port}: {result["error"]}')
        return result['result']

    def ready(self, timeout=1800, failure_path=None):
        deadline = time.monotonic() + timeout
        while True:
            if failure_path is not None and Path(failure_path).exists():
                raise RuntimeError(f'Another worker failed: {failure_path}')
            try:
                return self.call('health')
            except (ConnectionError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Worker not ready: {self.address}:{self.port}')
                time.sleep(1)


class ScoringPeer(Peer):
    def call(self, operation, **payload):
        if operation != 'evaluate':
            return super().call(operation, **payload)
        key = payload.pop('request_id', None) or uuid.uuid4().hex
        for attempt in range(3):
            try:
                return super().call(operation, request_id=key, **payload)
            except (EOFError, ConnectionError) as exc:
                print(f'score connection failure {self.address}:{self.port} id={key} attempt={attempt+1}: {exc!r}', flush=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def serve(port, token, handler, *, concurrent=False):
    class Request(socketserver.StreamRequestHandler):
        def handle(self):
            operation = None
            try:
                request = receive_message(self.rfile)
                if request.pop('token', None) != token:
                    raise PermissionError('Invalid worker token')
                operation = request.pop('operation')
                result = handler(operation, request)
                send_message(self.wfile, {'state': 'complete', 'result': result})
                if operation == 'stop':
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
            except BaseException as error:
                rejected = operation == 'evaluate' and isinstance(error, UnscorableReward)
                if not rejected:
                    traceback.print_exc()
                try:
                    send_message(self.wfile, {'state': 'unscorable', 'error': str(error)} if rejected
                                 else {'state': 'failed', 'error': repr(error)})
                except OSError:
                    pass
    cls = socketserver.ThreadingTCPServer if concurrent else socketserver.TCPServer
    cls.allow_reuse_address = True
    with cls(('0.0.0.0', port), Request) as server:
        server.serve_forever(poll_interval=.2)


def send_message(channel, value):
    channel.write(json.dumps(value, allow_nan=False).encode() + b"\n")
    channel.flush()


def receive_message(channel):
    line = channel.readline()
    if not line:
        raise EOFError("Service connection closed")
    return json.loads(line)


GIB = 1024**3
CAPACITY_ERRORS = {errno.ENOSPC, errno.EDQUOT}


def optional_write(fn, *args):
    """Only telemetry copies are optional; never apply this to a training artifact."""
    try:
        fn(*args)
    except OSError as exc:
        if exc.errno not in CAPACITY_ERRORS:
            raise
        return False
    return True


def local_event(run, name, row):
    root=Path(os.environ.get('SCRL_DIAGNOSTICS',str(Path.home()/'.cache/semantic-chunk-rl/diagnostics')))
    directory=root/hashlib.sha256(str(run).encode()).hexdigest()[:16]
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    atomic_json(directory/name,row)


def append_row(path, text):
    with Path(path).open('a') as stream:stream.write(text+'\n')


def wait_for_space(run, status, free=None, sleep=time.sleep):
    """Hysteresis at an update boundary; do not interrupt an in-flight update."""
    free=free or (lambda:shutil.disk_usage(run).free)
    available=free()
    if available>=6*GIB:return True
    while available<12*GIB:
        if any((run/k).exists() for k in ('FAIL.json','STOP','DRAIN')):return False
        status('waiting_for_storage',shared_free_bytes=available,resume_at_bytes=12*GIB)
        sleep(10);available=free()
    return True


def effective_state(status, failure, alive):
    previous=status.get('state','unknown')
    if failure:return 'failed'
    if previous in ('failed','complete','drained','budget_complete'):return previous
    if not alive:return 'coordinator_unavailable'
    return previous


def record_failure(run, value):
    run=Path(run)
    row=dict(value,recorded_unix=time.time())
    name=f'{time.time_ns()}-{uuid.uuid4().hex}.json'
    try:local_event(run,name,row)
    except OSError:pass
    optional_write(atomic_json,run/'failures'/name,row)
    try:
        with (run/'FAIL.json').open('x') as handle:
            json.dump(row,handle,indent=2,allow_nan=False)
            handle.write('\n')
    except FileExistsError:
        pass
    except OSError as exc:
        if exc.errno not in CAPACITY_ERRORS:raise


class EvaluationDispatcher:
    def __init__(self, cfg):
        self.event_cache = None
        if cfg.get('event_cache_backend') == 'dispatcher-v1':
            from scrl.runtime.contract_cache import EventCacheAuthority
            self.event_cache = EventCacheAuthority(cfg['run'])
        self.peers = [Peer(s['address'], s['port'], cfg['token']) for s in cfg['evaluations']]
        self.limit = cfg.get('services', {}).get('max_inflight', 2)
        self.condition = threading.Condition()
        self.active = [0]*len(self.peers)
        self.work = [0.]*len(self.peers)
        self.completed = [0]*len(self.peers)
        self.failed = [0]*len(self.peers)
        self.queued = self.queue_peak = 0
        self.waiting = deque()
        self.closed = False
        self.video_limit = cfg.get('frozen_score_queue_videos', 0)
        self.reserved = set()
        self.peak_reserved = 0
        self.requests = {}
        self.finished = deque()

    def __call__(self, op, request):
        if op in ('event_cache_acquire', 'event_cache_publish', 'event_cache_abandon'):
            if self.event_cache is None:
                raise ValueError('Dispatcher event cache is not enabled')
            if self.closed:
                raise RuntimeError('Scoring dispatcher closed')
            return self.event_cache(op, request)
        if op == 'health':
            with self.condition:
                return dict(role='dispatcher', active=list(self.active), estimated_work=list(self.work),
                            completed=list(self.completed), failed=list(self.failed), queued=self.queued, queue_peak=self.queue_peak,
                            reserved_videos=len(self.reserved), peak_reserved=self.peak_reserved)
        if op == 'reserve_videos':
            keys = set(request['keys'])
            if not 0 < len(keys) == len(request['keys']) <= self.video_limit:
                raise ValueError('Invalid video reservation')
            with self.condition:
                ok = self.condition.wait_for(lambda: self.closed or
                    len(self.reserved) + len(keys) <= self.video_limit, timeout=900)
                if not ok or self.closed or keys & self.reserved:
                    raise RuntimeError('Video reservation failed')
                self.reserved.update(keys)
                self.peak_reserved = max(self.peak_reserved, len(self.reserved))
            return {}
        if op == 'release_video':
            with self.condition:
                self.reserved.remove(request['key'])
                self.condition.notify_all()
            return {}
        if op == 'stop':
            self.close()
            return {}
        if op != 'evaluate':
            raise ValueError(op)
        key = request.get('request_id')
        if key is None:
            return self.dispatch(request)
        with self.condition:
            owner = key not in self.requests
            future = self.requests.setdefault(key, Future())
        if not owner:
            return future.result()
        try:
            print(f'score start id={key} attempt={request.get("attempt")} time={time.time()}', flush=True)
            result = self.dispatch({k:v for k,v in request.items() if k != 'request_id'})
            future.set_result(result)
            return result
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            print(f'score end id={key} time={time.time()}', flush=True)
            with self.condition:
                self.finished.append(key)
                if len(self.finished) > 256:
                    del self.requests[self.finished.popleft()]

    def dispatch(self, request):
        started = time.monotonic()
        cost = sum(s['num_chunks'] for s in request['plan']['segments'])
        with self.condition:
            ticket = object()
            self.waiting.append(ticket)
            self.queued += 1
            self.queue_peak = max(self.queue_peak, self.queued)
            try:
                while True:
                    if self.closed:
                        raise RuntimeError('Scoring dispatcher closed')
                    choices = [i for i, n in enumerate(self.active) if n < self.limit]
                    if choices and self.waiting[0] is ticket:
                        index = min(choices, key=lambda i: (self.work[i], self.completed[i], i))
                        self.active[index] += 1
                        self.work[index] += cost
                        break
                    self.condition.wait(timeout=10)
            finally:
                self.waiting.remove(ticket)
                self.queued -= 1
                self.condition.notify_all()
        admitted = time.monotonic()
        success=False
        try:
            result = self.peers[index].call('evaluate', **request)
            result['dispatch'] = dict(instance=index, queue_seconds=admitted-started,
                                      service_seconds=time.monotonic()-admitted)
            success=True
            return result
        except (EOFError,ConnectionError,TimeoutError) as exc:
            raise RuntimeError(f'Scorer {index} ({self.peers[index].address}:{self.peers[index].port}): {exc!r}') from exc
        finally:
            with self.condition:
                self.active[index] -= 1
                self.work[index] -= cost
                (self.completed if success else self.failed)[index] += 1
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
