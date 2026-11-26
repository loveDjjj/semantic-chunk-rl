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
from scrl.evaluation.protocol import DIMENSIONS
from scrl.plan import digest


def ready(run, arm, shards, expected):
    if (run / 'STOP').exists() or (run / ('STOP-' + arm)).exists() or (run / 'FAIL.json').exists():
        raise RuntimeError('Generation stopped')
    completed = 0
    for i in range(shards):
        f = run / 'workers' / arm / f'generate-{i}' / 'status.json'
        if f.exists():
            state = read(f)
            if state['shard'] != i or state['state'] == 'failed':
                raise RuntimeError(f'Generation shard {i} invalid or failed; inspect {f}')
            completed += state['state'] == 'complete'
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--expected-tasks', type=int, required=True)
    parser.add_argument('--wait-seconds', type=int, default=90000)
    args = parser.parse_args(argv)
    run, arm = args.run.absolute(), args.arm
    cfg = read(run / 'config.json')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', arm) or arm not in cfg['arms']:
        parser.error('unknown arm')
    if min(args.shards, args.expected_tasks, args.wait_seconds) < 1:
        parser.error('counts and timeout must be positive')
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
        if summary.exists():
            verify(read(summary), arm)
            status('complete')
            return
        deadline = time.monotonic() + args.wait_seconds
        while True:
            completed = ready(run, arm, args.shards, expected)
            if completed == args.shards:
                break
            status('waiting', completed_shards=completed, shards=args.shards)
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for all generation shards')
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        status('scoring')
        weights = Path(cfg['vbench_weights'])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([cfg['vbench_source'], os.environ['PYTHONPATH']]),
                   VBENCH_CACHE_DIR=str(weights), VBENCH_BERT_PATH=str(weights / 'bert-base-uncased'),
                   TORCH_HOME=str(weights / 'torch_hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        candidate = folder / f'.summary-{os.getpid()}.json'
        with (folder / 'score.log').open('a') as log:
            subprocess.run([sys.executable, '-m', 'scrl.evaluation.vbench', str(run), '--arm', arm,
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
