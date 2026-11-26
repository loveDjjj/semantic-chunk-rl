import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scrl.models.scoring import events
from scrl.models.scoring.http_judge import HttpJudge, UserText
from scrl.rl.reward import UnscorableReward


ORIGINAL = "A dial turns. Cut to the lid."
CONTRACT = dict(events=["A dial turns"], order=[], cuts=["Cut to the lid"], motion_mode="subject")


class BoundedEventTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.judge = HttpJudge.__new__(HttpJudge)
        self.judge.cache = Path(tmp.name)
        self.judge.processor = UserText()
        self.judge.max_frames = 4
        self.judge.event_decisions = events.BOUNDED_DECISIONS
        self.judge.request_text = self.decide
        self.assignments, self.video_choices, self.observations = [], [], []
        self.unknown = 0

    def decide(self, text, video=None, metadata=None, image=None, limit=None, constraint=None):
        self.assertIsNotNone(constraint)
        if image is not None:
            self.assertIn("regex", constraint)
            self.observations.append(int(image[0, 0, 0]))
            return "A dial is visible."
        choices = constraint["choice"]
        if video is None:
            self.assignments.append(choices)
            return choices[1]
        if "unknown" in choices:
            self.video_choices.append(list(metadata.frames_indices))
            if self.unknown:
                self.unknown -= 1
                return "unknown"
            return "done 1"
        self.assertEqual(choices, [f"{a} {b}" for a in range(5) for b in range(5)])
        return "3 4"

    def score(self, length=24, global_only=False):
        request = dict(original=ORIGINAL, captions=["turning dial", "the lid"], target=0,
                       fps=16, segment_start=0, segment_end=length // 2)
        if global_only:
            request = dict(original=ORIGINAL, fps=16, global_only=True)
        pixels = np.broadcast_to(np.arange(length)[:, None, None, None], (length, 2, 2, 3))
        with patch.object(events, "_generate", side_effect=[CONTRACT]) as free:
            try:
                return events.score_events(self.judge, request, pixels, lambda *_: 1.0)
            finally:
                self.assertEqual(free.call_count, 1)
                self.assertEqual(free.call_args.args[1], events.CONTRACT)

    def test_choices_are_primary_and_unknown_gets_new_frames(self):
        self.unknown = 1
        result = self.score()
        self.assertEqual(self.assignments, [["null", "0", "1"], ["null", "1"]])
        self.assertEqual(result["assignment"], {"owners": [0], "cut_before": [1]})
        global_scope = result["scopes"]["global"]
        self.assertEqual(global_scope["semantic"]["completion"], 1.0)
        self.assertIn("unknown", global_scope["attempts"][0]["error"])
        self.assertTrue(set(self.video_choices[0]).isdisjoint(self.video_choices[1]))
        self.assertEqual(global_scope["attempts"][-1]["absolute_rows"][0][1], self.observations[0])
        self.assertEqual(global_scope["attempts"][-1]["method"], events.BOUNDED_DECISIONS)

    def test_inconclusive_evidence_is_recorded_not_zeroed(self):
        self.unknown = 5
        with self.assertRaisesRegex(UnscorableReward, "unknown"):
            self.score()
        self.assertEqual(len(self.video_choices), 2)
        receipt, = (self.judge.cache / events.VERSION).glob("unscorable-*.json")
        self.assertEqual(len(json.loads(receipt.read_bytes())["attempts"]), 2)

    def test_no_spare_frames_does_not_repeat_an_identical_judgment(self):
        self.unknown = 5
        with self.assertRaisesRegex(UnscorableReward, "unknown"):
            self.score(length=4)
        self.assertEqual(len(self.video_choices), 1)

    def test_invalid_response_from_http_choice_decoder_is_rejected(self):
        self.judge.request_text = lambda *a, **kw: "-1"
        with self.assertRaisesRegex(UnscorableReward, "outside the allowed choices"):
            self.score()

    def test_global_only_skips_ownership_and_local_judgments(self):
        result = self.score(global_only=True)
        self.assertEqual(set(result["scopes"]), {"global"})
        self.assertNotIn("assignment", result)
        self.assertEqual(self.assignments, [])
        self.assertEqual(len(self.video_choices), 1)
        self.assertEqual(result["scopes"]["global"]["semantic"]["completion"], 1.0)
