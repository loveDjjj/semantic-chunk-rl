from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest

from scrl.config import DEFAULT_CONFIGS, read
from scrl.models.scoring.service import VideoRewardService
from scrl.rl.reward import RewardBundle, RewardComposer, RewardConfig, ScopeScores, UnscorableReward


class GlobalRewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RewardConfig(**read(DEFAULT_CONFIGS / "rl-whole24.json")["reward"])
        self.score = RewardComposer(self.cfg, "event-reward-v2")

    def test_exact_equal_weights_no_local_evidence_needed(self):
        for order, expected_s in ((None, .5), (0., .3)):
            full = ScopeScores(dict.fromkeys(self.cfg.quality_components, .8), dict(completion=.5, order=order))
            result = self.score.compose(RewardBundle(full, ScopeScores({}, {}), self.score.protocol))
            self.assertAlmostEqual(result["planner"], .5 * (.8 + expected_s))
            self.assertEqual(result["planner"], result["executor"])
            self.assertAlmostEqual(result["semantic_global"], expected_s)
            self.assertNotIn("semantic_local", result)
            full.semantic["completion"] = None
            with self.assertRaises(UnscorableReward):
                self.score.compose(RewardBundle(full, ScopeScores({}, {}), self.score.protocol))

    def test_no_unconsumed_model_calls(self):
        service = VideoRewardService.__new__(VideoRewardService)
        service.protocol, service.global_only, service.judge_pool = self.score.protocol, True, None
        calls = []
        def judge(req):
            self.assertTrue(req["global_only"])
            calls.append("judge")
            return dict(protocol=service.protocol, seconds=1., scopes={"global": {
                "quality": {"dynamics": 1., "physics": 1.}, "semantic": {"completion": 1., "order": None}}})
        def quality(req):
            self.assertEqual((req["start"], req["end"]), (0, 45))
            calls.append("quality")
            return dict(seconds=.5, quality=dict.fromkeys(("imaging","aesthetic","consistency","motion"), 1.))
        service.workers = {k: SimpleNamespace(score=v) for k, v in (("judge",judge),("quality",quality))}
        with TemporaryDirectory() as tmp, ThreadPoolExecutor(2) as service.io_pool:
            directory = Path(tmp)
            span = SimpleNamespace(pixel_start=10, pixel_end=30)
            plan = SimpleNamespace(segments=[SimpleNamespace(caption="a")])
            result = service._structured_scores("a", plan, span, 0, directory/"rgb", 45, {},
                                    directory, directory/"reward.json", time.monotonic(), time.monotonic())
            self.assertEqual(self.score.compose(result)["planner"], 1.)
            self.assertCountEqual(calls, ["judge", "quality"])
