"""Unscorable evidence affects one video, not the resident service or its peers."""

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scrl.models.scoring import worker
from scrl.models.scoring.http_judge import HttpJudge, parallel_class
from scrl.models.scoring.service import ResidentProcess, VideoRewardService, _collect_scores
from scrl.rl.reward import UnscorableReward


class ScoreIsolationTests(unittest.TestCase):
    def test_resident_socket_survives_rejection_but_not_device_failure(self):
        client = ResidentProcess.__new__(ResidentProcess)
        gpu = Mock()
        gpu.max_memory_allocated.return_value = 1000
        judge = Mock()
        judge.score.side_effect = [UnscorableReward("missing event rows"), {"ok": True},
                                   RuntimeError("device out of memory")]
        with tempfile.TemporaryDirectory() as tmp:
            directory = client.directory = Path(tmp)
            (directory / "config.json").write_text("{}")
            parent, child = socket.socketpair()
            parent.settimeout(5)
            client.process = Mock()
            client.process.poll.return_value = None
            client.lock, client.ready_value, client.broken = threading.Lock(), None, False
            argv = ["worker", "--kind", "judge", "--directory", tmp, "--device", "cpu",
                    "--socket-fd", str(child.detach())]
            with (patch.object(sys, "argv", argv), patch.object(worker.torch, "cuda", gpu),
                  patch.object(worker.torch, "set_num_threads"),
                  patch("scrl.models.scoring.video_judge.VideoJudge", return_value=judge),
                  ThreadPoolExecutor(1) as pool, parent, parent.makefile("rwb") as client.channel):
                serving = pool.submit(worker.main)
                with self.assertRaisesRegex(UnscorableReward, "missing event rows"):
                    client.score({})
                self.assertFalse(client.broken)
                self.assertFalse((directory / "failure.json").exists())
                self.assertTrue(client.score({})["ok"])
                with self.assertRaisesRegex(RuntimeError, "device out of memory"):
                    client.score({})
                self.assertTrue(client.broken)
                with self.assertRaisesRegex(RuntimeError, "device out of memory"):
                    serving.result(timeout=5)
                self.assertIn("device out of memory", json.loads((directory / "failure.json").read_text())["error"])

    def test_queue_accepts_next_video_only_after_request_scoped_failure(self):
        service = VideoRewardService.__new__(VideoRewardService)
        service.closed, service.failure = False, None
        service.slots = threading.BoundedSemaphore(1)
        service._evaluate = Mock(side_effect=[UnscorableReward("bad evidence"), 42, OSError("disk")])
        rollout = SimpleNamespace(plan=None, final=SimpleNamespace(latents=None, prefix_id="p"),
                                  target_segment=0)
        with ThreadPoolExecutor(1) as service.pool:
            with self.assertRaises(UnscorableReward):
                service("one", rollout)
            self.assertEqual(service("two", rollout), 42)
            with self.assertRaises(OSError):
                service("three", rollout)
            with self.assertRaisesRegex(RuntimeError, "previous evaluation failed"):
                service("four", rollout)

    def test_parallel_scores_prioritize_fatal_errors_in_both_orders(self):
        for errors in [(UnscorableReward("bad"), OSError("disk")),
                       (OSError("disk"), UnscorableReward("bad"))]:
            futures = []
            for error in errors:
                future = Future()
                future.set_exception(error)
                futures.append(future)
            with self.assertRaisesRegex(OSError, "disk"):
                _collect_scores(enumerate(futures))

            def generate(_judge, texts, *args, **kwargs):
                raise errors[texts[0]]

            with tempfile.TemporaryDirectory() as tmp, patch.object(HttpJudge, "_generate", generate):
                judge = parallel_class()(dict(max_frames=16, max_new_tokens=32, pixels_per_frame=128,
                                              contract_cache=tmp), 8000, 2)
                try:
                    with self.assertRaisesRegex(OSError, "disk"):
                        judge._generate([0, 1], [0, 1], [0, 1])
                finally:
                    judge.close()
