import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scrl.models.scoring import events
from scrl.rl.reward import UnscorableReward


ORIGINAL = "A dial clicks,  then the lid opens. Cut to a blue light."
REJECTED = dict(events=["the dial makes a click", "the lid opens"],
                cuts=["a cut reveals the blue light"],
                order=[[0, 1, "after the dial clicks"]], motion_mode="subject")
SPANS = [[0, 2], [9, 14], [2, 4]]
RECOVERED = dict(events=["A dial clicks", "the lid opens"], cuts=["Cut to a blue light."],
                 order=[[0, 1, "clicks,  then"]], motion_mode="subject")


class EventContractSpanTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.judge = SimpleNamespace(cache=Path(tmp.name), event_decisions=events.BOUNDED_DECISIONS)
        self.directory = self.judge.cache / events.VERSION
        self.path = self.directory / (events.digest([events.CONTRACT, ORIGINAL, events.BOUNDED_DECISIONS]) + ".json")

    def contract(self):
        return events._cached(self.judge, events.CONTRACT, ORIGINAL,
                              lambda value: events.validate_contract(value, ORIGINAL))

    def test_repairs_only_missing_quotes_once_and_caches_validated_contract(self):
        answers = [REJECTED, REJECTED, {"spans": SPANS}]
        with patch.object(events, "_generate", side_effect=answers) as generate:
            self.assertEqual(self.contract(), RECOVERED)
            self.assertEqual(self.contract(), RECOVERED)
            self.assertEqual(generate.call_count, 3)
            request = generate.call_args.args[2]
            self.assertEqual(request["tokens"][9], [9, "Cut"])
            self.assertEqual([(q["field"], q["index"]) for q in request["quotes_to_locate"]],
                             [("events", 0), ("cuts", 0), ("order", 0)])
        self.assertEqual(json.loads(self.path.read_bytes()), RECOVERED)
        self.assertEqual(len(list(self.directory.glob("*-invalid-*.json"))), 2)

    def test_missing_unanchored_and_invalid_evidence_cannot_be_published(self):
        for spans in (SPANS[:-1], [None, *SPANS[1:]], [[False, 1], *SPANS[1:]],
                      [[2, 1], *SPANS[1:]], [[2, 55], *SPANS[1:]],
                      [[5, 7], *SPANS[1:]]):  # duplicated event rejected by original validator
            with self.subTest(spans=spans):
                with patch.object(events, "_generate", side_effect=[REJECTED, REJECTED, {"spans": spans}]) as gen:
                    with self.assertRaises(UnscorableReward):
                        self.contract()
                    self.assertEqual(gen.call_count, 3)
                self.assertFalse(self.path.exists())
                receipt, = self.directory.glob("*-span-repair.json")
                record = json.loads(receipt.read_bytes())
                self.assertTrue(record["error"])
                self.assertEqual(record["response"], {"spans": spans})

    def test_valid_text_answer_does_not_make_span_request(self):
        with patch.object(events, "_generate", side_effect=[RECOVERED]) as gen:
            self.assertEqual(self.contract(), RECOVERED)
            self.assertEqual(gen.call_count, 1)
        self.assertFalse(list(self.directory.glob("*-span-repair.json")))

    def test_legacy_protocol_rejects_after_two_ordinary_attempts(self):
        del self.judge.event_decisions
        with patch.object(events, "_generate", side_effect=[REJECTED, REJECTED]) as gen:
            with self.assertRaisesRegex(UnscorableReward, "original-prompt quotes"):
                self.contract()
            self.assertEqual(gen.call_count, 2)
        self.assertEqual(len(list(self.directory.glob("*-invalid-*.json"))), 2)
        self.assertFalse(list(self.directory.glob("*-span-repair.json")))
        legacy_key = events.digest([events.CONTRACT, ORIGINAL])
        self.assertFalse((self.directory / (legacy_key + ".json")).exists())
