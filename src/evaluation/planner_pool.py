"""Run independent seeded Qwen planner lanes before loading the single video model."""
import json
import subprocess
import tempfile
import time


def plan_concurrently(command, tasks, lanes, timeout):
    if lanes not in (1, 2, 3):
        raise ValueError('SCRL_EVAL_PLANNER_LANES must be 1, 2, or 3')
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate planner task IDs')
    processes = []
    deadline = time.monotonic() + timeout
    try:
        for index in range(min(lanes, len(tasks))):
            # A private input file avoids blocking on a slow-loading model's stdin.
            with tempfile.TemporaryFile(mode='w+') as stream:
                json.dump(tasks[index::lanes], stream)
                stream.seek(0)
                processes.append(subprocess.Popen(command, stdin=stream))
        while processes:
            for process in processes[:]:
                code = process.poll()
                if code is not None:
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    processes.remove(process)
            if time.monotonic() > deadline and processes:
                raise subprocess.TimeoutExpired(command, timeout)
            if processes:
                time.sleep(0.2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
