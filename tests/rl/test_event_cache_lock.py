import errno
import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from scrl.models.scoring import events


def _request_contract(root, contention, results):
    real_flock = events.fcntl.flock

    def flock(*args):
        try:
            return real_flock(*args)
        except BlockingIOError:
            contention.set()
            raise

    def generate(*_):
        if not contention.wait(10):
            raise TimeoutError("No peer contended for the contract lock")
        return {"writer": os.getpid()}

    with patch.object(events.fcntl, "flock", flock), patch.object(events, "_generate", generate):
        results.put(events._cached(SimpleNamespace(cache=root), "test", {}, dict))


class EventCacheLockTests(unittest.TestCase):
    def test_peer_creation_can_precede_visibility(self):
        path = Mock(spec=Path)
        handle = object()
        path.open.side_effect = [FileNotFoundError(), FileExistsError(),
                                 FileNotFoundError(), FileExistsError(), handle]
        with patch.object(events.time, "sleep"), patch.object(events.time, "monotonic", return_value=0):
            self.assertIs(events._open_cache_lock(path, 1), handle)
        self.assertEqual(path.open.call_args_list, [call("r+b"), call("x+b")] * 2 + [call("r+b")])

    def test_persistent_visibility_failure_is_bounded(self):
        path = Mock(spec=Path)
        path.open.side_effect = [FileNotFoundError(), FileExistsError()]
        with self.assertRaisesRegex(TimeoutError, "opening event cache lock"):
            events._open_cache_lock(path, 0)

    def test_other_io_failures_propagate(self):
        for errors in ([PermissionError()], [FileNotFoundError(), OSError(errno.ENOSPC, "full")]):
            with self.subTest(errors=errors):
                path = Mock(spec=Path)
                path.open.side_effect = errors
                with self.assertRaises(OSError) as caught:
                    events._open_cache_lock(path, 0)
                self.assertIs(caught.exception, errors[-1])

    def test_contending_processes_publish_one_decision(self):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contention, results = ctx.Event(), ctx.Queue()
            workers = [ctx.Process(target=_request_contract, args=(root, contention, results))
                       for _ in range(2)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(20)
                    self.assertEqual(worker.exitcode, 0)
                self.assertTrue(contention.is_set())
                first, second = results.get(timeout=2), results.get(timeout=2)
                self.assertEqual(first, second)
                receipt, = (root / events.VERSION).glob("*.json")
                self.assertEqual(json.loads(receipt.read_bytes()), first)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                results.close()
