from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

from scrl.runtime.transport import EvaluationDispatcher


class EventCacheAuthorityTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = {"evaluations": [], "event_cache_backend": "dispatcher-v1", "run": tmp.name}
        self.dispatcher = EvaluationDispatcher(self.cfg)
        self.key = "aa" * 32
        self.a, self.b = (uuid4().hex for _ in range(2))
        self.value = {"owners": [1, None]}

    def call(self, op, owner, key=None, **payload):
        return self.dispatcher("event_cache_" + op, dict(key=key or self.key, owner=owner, **payload))

    def test_contended_miss_grants_one_owner_and_independent_keys_make_progress(self):
        owners = [uuid4().hex for _ in range(12)]
        barrier = threading.Barrier(len(owners), timeout=10)

        def start(owner):
            barrier.wait()
            return self.call("acquire", owner)["state"]

        with ThreadPoolExecutor(max_workers=len(owners)) as pool:
            states = list(pool.map(start, owners))
        self.assertCountEqual(states, ["owner"] + ["wait"] * 11)
        primary = owners[states.index("owner")]
        self.assertEqual(self.call("acquire", primary)["state"], "owner")
        self.assertEqual(self.call("acquire", self.b, key="bb"*32)["state"], "owner")
        hit = self.call("publish", primary, value=self.value)
        self.assertEqual(hit, {"state": "hit", "value": self.value})
        for owner in owners:
            self.assertEqual(self.call("acquire", owner), hit)
        receipts = list(Path(self.cfg["run"]).rglob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertNotIn(primary, receipts[0].read_text())

    def test_abandon_releases_only_matching_claim_and_fences_late_publish(self):
        self.call("acquire", self.a)
        self.assertEqual(self.call("abandon", self.b)["state"], "abandoned")
        self.assertEqual(self.call("acquire", self.b)["state"], "wait")
        self.call("abandon", self.a)
        self.assertEqual(self.call("acquire", self.b)["state"], "owner")
        self.assertEqual(self.call("abandon", self.a)["state"], "abandoned")
        self.assertEqual(self.call("acquire", self.a)["state"], "wait")
        with self.assertRaises(ValueError):
            self.call("publish", self.a, value=self.value)
        self.call("publish", self.b, value=self.value)
        self.assertEqual(self.call("abandon", self.a)["state"], "abandoned")
        self.assertEqual(self.call("acquire", self.a), {"state": "hit", "value": self.value})

    def test_restart_drops_claim_but_not_durable_idempotence(self):
        self.call("acquire", self.a)
        self.dispatcher = EvaluationDispatcher(self.cfg)
        with self.assertRaises(ValueError):
            self.call("publish", self.a, value=self.value)
        self.call("acquire", self.b)
        canonical = self.call("publish", self.b, value=self.value)
        self.dispatcher = EvaluationDispatcher(self.cfg)
        self.assertEqual(self.call("acquire", self.a), canonical)
        self.assertEqual(self.call("publish", self.b, value=self.value), canonical)
        with self.assertRaises(ValueError):
            self.call("publish", self.a, value=self.value)
        with self.assertRaises(ValueError):
            self.call("publish", self.b, value={"owners": [True, None]})

    def test_failed_persistence_is_not_a_hit_and_receipt_corruption_is_fatal(self):
        self.call("acquire", self.a)
        with patch("scrl.runtime.contract_cache.atomic_json", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.call("publish", self.a, value=self.value)
        self.assertEqual(self.call("acquire", self.b)["state"], "wait")
        self.call("publish", self.a, value=self.value)
        next(Path(self.cfg["run"]).rglob("*.json")).write_text("null")
        with self.assertRaisesRegex(ValueError, "Invalid canonical"):
            self.call("acquire", self.b)

    def test_rejects_unconfigured_backend_bad_identifiers_and_closed_dispatcher(self):
        with self.assertRaisesRegex(ValueError, "not enabled"):
            EvaluationDispatcher({"evaluations": []})("event_cache_acquire", {})
        for key in ("../other", "AB"*32, "0"*63):
            with self.assertRaises(ValueError):
                self.call("acquire", self.a, key=key)
        for owner in (None, "abc", "../../invalid"):
            with self.assertRaises(ValueError):
                self.call("acquire", owner)
        self.dispatcher.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.call("acquire", self.a)
