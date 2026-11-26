"""CPU tests for event protocol, actual scoring service and legacy isolation."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from scrl.models.scoring import events
from scrl.models.scoring.http_judge import HttpJudge, UserText
from scrl.models.scoring.service import VideoRewardService
from scrl.plan import FinitePlan, Segment
from scrl.rl.reward import RewardBundle, RewardComposer, RewardConfig, ScopeScores, UnscorableReward


PROMPT = "Press red, then press blue. Cut to a chair."
CONTRACT = dict(events=["Press red", "press blue", "a chair"], order=[[0, 1, "then"]],
                cuts=["Cut to"], motion_mode="unspecified")


class FakeJudge(HttpJudge):
    def __init__(self, root):
        self.cache, self.processor, self.max_frames = root, UserText(), 4
        self.text_calls, self.tasks, self.unknown = [], [], 0

    def _generate(self, texts, videos=None, metadata=None, **kwargs):
        text = texts[0].removesuffix("<|im_end|>")
        if text.startswith(events.CONTRACT):
            self.text_calls.append("contract")
            return [CONTRACT]
        if text.startswith(events.ASSIGN):
            self.text_calls.append("assignment")
            return [dict(owners=[0, 1, 2], cut_before=[2])]
        data = json.loads(text.split("TASK: ")[-1])
        self.tasks.append(dict(data,frames=metadata[0].frames_indices))
        assert data["allowed_slots"] == list(range(len(metadata[0].frames_indices)))
        rows = [["done", i, "verified state"] for i in range(len(data["events"]))]
        if self.unknown:
            self.unknown -= 1
            rows[0] = ["unknown", None, "occluded"]
        return [dict(rows=rows, dynamics=4, physics=4)]


class Quality:
    def score(self, request):
        if request.get("operation") != "boundaries":
            return {"quality": {"visual": 1.}, "seconds": 0}
        return {"boundaries": [{"frame": frame, "motion": float(i)}
                               for i, frame in enumerate(request["boundaries"])],
                "boundary_quality": {"motion": .5}, "seconds": 0}


class EventRewardTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.judge = FakeJudge(self.root)
        self.plan = FinitePlan(tuple(Segment(c, 1) for c in ("press red", "press blue", "a chair")))
        self.frames = self.plan.spans()[-1].pixel_end
        self.rgb = self.root / "rgb.npy"
        np.save(self.rgb, np.zeros((self.frames, 2, 2, 3), dtype=np.uint8))

    def service(self, target=1):
        service = VideoRewardService.__new__(VideoRewardService)
        service.protocol = events.VERSION + ":test"
        span = self.plan.spans()[target]
        with ThreadPoolExecutor(3) as pool:
            service.io_pool, service.judge_pool = pool, None
            service.workers = {"quality": Quality(), "judge": self.judge}
            return service._structured_scores(PROMPT, self.plan, span, target, self.rgb, self.frames,
                {}, self.root, self.root / "reward.json", time.monotonic(), time.monotonic())

    def test_source_only_cache_target_only_witnesses_and_requested_cut(self):
        bundle = self.service()
        self.assertEqual(bundle.global_scores.semantic, {"completion": 1., "order": 1.})
        self.assertEqual(bundle.segment_scores.semantic, {"completion": 1., "order": None, "owned": 1})
        self.assertEqual(bundle.boundary_quality, {"motion": 0.})  # exit cut exempt, entry retained
        first, second = self.judge.tasks
        self.assertNotIn("caption", first)
        self.assertEqual(second["events"], ["press blue"])
        self.assertTrue(all(9 <= f < 21 for f in second["frames"]))
        self.assertIsNone(self.service(target=2).boundary_quality)
        self.assertEqual(self.judge.text_calls, ["contract", "assignment"])
        self.plan = FinitePlan((Segment("different red", 1), *self.plan.segments[1:]))
        self.service()
        self.assertEqual(self.judge.text_calls, ["contract", "assignment", "assignment"])

    def test_unclear_resamples_new_witnesses_or_aborts_without_a_zero(self):
        self.judge.unknown = 1
        self.service()
        self.assertNotEqual(self.judge.tasks[0]["frames"], self.judge.tasks[1]["frames"])
        self.judge.unknown = 2
        with self.assertRaisesRegex(UnscorableReward, "Insufficient sampled"):
            self.service()
        self.assertTrue(list((self.root / events.VERSION).glob("unscorable-*.json")))

    def test_short_and_long_windows_get_maximum_distinct_new_frames(self):
        for end in (4, 5, 8, 9, 14, 40):
            first = set(events._indices(2, end + 2, 4, False))
            second = events._indices(2, end + 2, 4, True)
            self.assertEqual(len(set(second)), 4)
            self.assertEqual(len(set(second) - first), min(end - 4, 4))

    def test_source_graph_ownership_and_evidence_must_validate(self):
        base = dict(events=["the cloud continues"], order=[], cuts=[], motion_mode="subject")
        self.assertEqual(events.validate_contract(base, "The cloud continues.")["events"],
                         ["The cloud continues"])
        for source, quotes in [("The cloud continues. THE CLOUD CONTINUES.", ["the cloud continues"]),
                               ("The cloud continues.", ["The cloud continues", "the cloud continues"]),
                               ("The cloud continues.", ["the cloud stops"])]:
            with self.assertRaises(UnscorableReward):
                events.validate_contract(dict(base, events=quotes), source)
        for changes in (dict(events=["not in the original"]), dict(cuts=["pan right"]),
                        dict(order=[[0, 1, "then"], [1, 0, "then"]]), dict(order=[[0, 7, "then"]])):
            with self.assertRaises(UnscorableReward):
                events.validate_contract(CONTRACT | changes, PROMPT)
        for owners, cuts in (([[0, 1], 1, 2], [2]), ([0, 1, 2], [0]), ([0, 1, 2], [])):
            with self.assertRaises(UnscorableReward):
                events.validate_assignment(dict(owners=owners, cut_before=cuts), CONTRACT, 3)
        with self.assertRaises(UnscorableReward):
            events.evidence(dict(rows=[["done", 7, "neighbor"]], dynamics=4, physics=4), [8, 9], [0], [])

    def test_reversed_order_missing_action_no_order_and_transition(self):
        compose = RewardComposer(RewardConfig(mode="event-v2", quality_components={"q": 1.}), "test").compose
        def rewards(rows, order, q, owned=1):
            judgment = dict(rows=rows, dynamics=4, physics=4)
            g = ScopeScores({"q": q}, events.evidence(judgment, [0, 1], [0, 1], order))
            l = ScopeScores({"q": .5}, {"owned": owned, "completion": .5 if owned else None})
            return compose(RewardBundle(g, l, "test"))
        good = [["done", 0, "red depressed"], ["done", 1, "blue depressed"]]
        reversed_ = [["done", 1, "red depressed"], ["done", 0, "blue depressed"]]
        missing = [good[0], ["missing", 1, "blue untouched"]]
        self.assertAlmostEqual(rewards(good, CONTRACT["order"], .5)["planner"], .9)
        self.assertAlmostEqual(rewards(reversed_, CONTRACT["order"], 1.)["planner"], .6)
        self.assertAlmostEqual(rewards(missing, CONTRACT["order"], 1.)["planner"], .3)
        self.assertEqual(rewards(reversed_, [], 1.)["planner"], 1.)
        self.assertAlmostEqual(rewards(good, [], 1.)["executor"], .7 * .5 * .85 + .3)
        r = rewards(missing, CONTRACT["order"], 1., owned=0)
        self.assertAlmostEqual(r["planner"], r["executor"])
