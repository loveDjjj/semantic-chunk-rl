"""Contract and gradient checks executed on the explicitly selected CUDA device."""

from dataclasses import replace
import json
import os
import tempfile
import unittest

import torch

from scrl.plan import FinitePlan, PlanLimits, PolicyStamp, Segment
from scrl.models.segments import PrefixState, indexed_noise
from scrl.plan import DecisionTokens
from scrl.rl.reward import RewardBundle, RewardComposer, ScopeScores, UnscorableReward
from scrl.rl.checkpoint import JointCoordinator, Phase, load_checkpoint, save_checkpoint
from scrl.rl.objectives import ScoredRollout, executor_advantages, planner_advantages
from scrl.rl.objectives import consistency_loss, planner_loss


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get("SCRL_TEST_DEVICE", "cpu")
        if cls.device.startswith("cuda:"):
            torch.cuda.set_device(cls.device)

    def test_variable_plans_without_end_and_resource_truncation(self):
        for count in (1, 2, 5):
            plan = FinitePlan(tuple(Segment("a scene", 1) for _ in range(count)))
            restored = FinitePlan.parse(json.dumps(plan.to_output()))
            self.assertEqual(restored, plan)
            self.assertEqual(FinitePlan.from_dict(plan.to_dict()), plan)
            spans = plan.spans()
            self.assertEqual(spans[-1].pixel_end, 12 * count - 3)
            self.assertEqual([s.pixel_end for s in spans[:-1]], [s.pixel_start for s in spans[1:]])
        with self.assertRaises(ValueError):
            FinitePlan.parse(json.dumps(plan.to_output()), finish_reason="length")
        with self.assertRaises(ValueError):
            FinitePlan((Segment("scene", 3),)).validate(PlanLimits())
        with self.assertRaises(ValueError):
            FinitePlan.parse('[{"step_index":0,"prompt":"scene","num_chunks":1,"num_chunks":2}]')
        with self.assertRaises(ValueError):
            FinitePlan.parse('[{"step_index":1,"prompt":"scene","num_chunks":1}]')
        with self.assertRaises(ValueError):
            FinitePlan.parse('{"segments":[],"terminal":"END"}')
        exact = FinitePlan(tuple(Segment("a coherent action", 6) for _ in range(5)))
        self.assertEqual(FinitePlan.parse(json.dumps(exact.to_output())).total_chunks, 30)
        with self.assertRaises(ValueError):
            FinitePlan(tuple(Segment("action", 1) for _ in range(6))).validate(PlanLimits())

    def test_prefix_noise_independent_of_suffix_extent(self):
        options = dict(device=self.device, dtype=torch.float32)
        short = indexed_noise(4, 0, 3, (2, 2, 2), **options)
        long = indexed_noise(4, 0, 9, (2, 2, 2), **options)
        torch.testing.assert_close(short, long[:, :3], atol=0, rtol=0)
        prefix = PrefixState((Segment("scene", 1),), 0, "test", short)
        short.zero_()
        prefix.verify()
        prefix.latents.zero_()
        with self.assertRaises(RuntimeError):
            prefix.verify()

    def test_optional_grammar_accepts_variable_array_without_terminal(self):
        from scrl.models.planner import FinitePlanGrammar
        grammar = object.__new__(FinitePlanGrammar)
        grammar.limits = PlanLimits()
        for count in (1, 3, 5):
            plan = FinitePlan(tuple(Segment('A "red" cup stays still.', 1) for _ in range(count)))
            output = json.dumps(plan.to_output(), separators=(",", ":"))
            state = grammar.initial()
            self.assertTrue(grammar.consume(state, output))
            self.assertEqual(state["mode"], "done")
            self.assertEqual(state["lengths"], [1] * count)
            self.assertFalse(grammar.consume(state, 'END'))
        too_many = FinitePlan(tuple(Segment("action", 1) for _ in range(6)))
        self.assertFalse(grammar.consume(grammar.initial(), json.dumps(too_many.to_output(), separators=(",", ":"))))

    def test_grammar_keeps_fused_continuation_and_whitespace_tokens(self):
        from scrl.models.planner import FinitePlanGrammar
        pieces = ['<eos>', '1', '2', '4', '6', '}', '},\n  {', '}\n]', ' ', '\n  ',
                  '[{"step_index":0,"prompt":"', 'a', '", "num_chunks": ']
        class Tokenizer:
            eos_token_id = 0
            all_special_ids = [0]
            def get_vocab(self):
                return {piece: i for i, piece in enumerate(pieces)}
            def batch_decode(self, rows, **kwargs):
                return [''.join(pieces[i] for i in row) for row in rows]
            def encode(self, text, **kwargs):
                return [pieces.index(text)]
        grammar = FinitePlanGrammar(Tokenizer(), 0)
        state = grammar.initial()
        self.assertTrue(grammar.consume(state, '[\n { "step_index" : 0, "prompt" : "a", "num_chunks" : 1'))
        allowed = grammar.allowed(state)
        self.assertIn(6, allowed)  # One token can close a row and start the next row.
        self.assertIn(7, allowed)
        self.assertIn(9, allowed)
        plan = FinitePlan(tuple(Segment('a', 1) for _ in range(5)))
        at_limit = grammar.initial()
        self.assertTrue(grammar.consume(at_limit, json.dumps(plan.to_output(), separators=(',', ':'))[:-2]))
        self.assertNotIn(6, grammar.allowed(at_limit))
        self.assertIn(7, grammar.allowed(at_limit))
        spans = grammar.decision_spans([10, 11, 12, 1, 7, 0], 1)
        self.assertEqual(spans[0], DecisionTokens((1, 3), (3, 4)))

    def test_quality_primary_and_missing_is_not_zero(self):
        def score(q, s):
            return ScopeScores(dict.fromkeys(["imaging", "aesthetic", "consistency", "motion"], q),
                               {"alignment": s})
        composer = RewardComposer()
        a = composer.compose(RewardBundle(score(.9, .5), score(.9, .5), composer.protocol))
        b = composer.compose(RewardBundle(score(.5, .9), score(.5, .9), composer.protocol))
        self.assertGreater(a["planner"], b["planner"])
        broken = score(.9, .5)
        broken.quality["motion"] = None
        with self.assertRaises(UnscorableReward):
            composer.compose(RewardBundle(broken, broken, composer.protocol))

    def test_group_seed_averages_and_policy_isolation(self):
        stamp = PolicyStamp()
        a, b = FinitePlan((Segment("a", 1),)), FinitePlan((Segment("b", 2),))
        rows = [ScoredRollout("g", name, seed, "original", "prefix", stamp, plan, 0, value, value)
                for name, plan, values in [("a", a, [.8, .6]), ("b", b, [.2, .4])]
                for seed, value in enumerate(values)]
        adv = planner_advantages(rows, stamp, expected_candidates=["a", "b"], expected_seeds=[0, 1])
        self.assertAlmostEqual(adv["a"], 1)
        self.assertAlmostEqual(adv["b"], -1)
        with self.assertRaises(ValueError):
            executor_advantages(rows, stamp, expected_seeds=[0, 1])
        with self.assertRaises(ValueError):
            planner_advantages(rows[:-1], stamp, expected_candidates=["a", "b"], expected_seeds=[0, 1])
        with self.assertRaises(ValueError):
            planner_advantages(rows, replace(stamp, executor=1), expected_candidates=["a", "b"], expected_seeds=[0, 1])

    def test_focal_token_gradients_and_duration_weight(self):
        current = torch.full((10,), -1., device=self.device, requires_grad=True)
        old, ref = current.detach().clone(), current.detach().clone()
        loss = planner_loss(current, old, ref, 1., DecisionTokens((2, 6), (7, 8)),
                            kl_beta=0, duration_weight=.5)
        loss.backward()
        self.assertEqual(float(current.grad[[0, 1, 6, 8, 9]].abs().sum()), 0)
        self.assertAlmostEqual(float(current.grad[2:6].sum()), float(current.grad[7]), places=6)

    def test_consistency_gradient_agrees_with_gaussian_density(self):
        torch.manual_seed(12)
        prediction = torch.randn(2, 3, device=self.device, requires_grad=True)
        old = prediction.detach().clone()
        following = torch.randn_like(old)
        sigma = .6
        actual = consistency_loss(prediction, old, following, sigma, .7, old, 0)
        gradient = torch.autograd.grad(actual, prediction)[0]
        density_loss = .7 * (following - (1 - sigma) * prediction).square().mean() / (2 * sigma**2)
        expected = torch.autograd.grad(density_loss, prediction)[0]
        torch.testing.assert_close(gradient, expected)

    def test_joint_transaction_and_atomic_checkpoint(self):
        coordinator = JointCoordinator()
        stamp = coordinator.begin("p0", ["a", "b"])
        coordinator.complete("a", stamp)
        with self.assertRaises(RuntimeError):
            coordinator.commit(lambda *_: {"planner": {"updated": True}, "executor": {"updated": True}})
        coordinator.complete("b", stamp)
        coordinator.commit(lambda *_: {"planner": {"updated": True}, "executor": {"updated": True}})
        self.assertEqual(coordinator.stamp, replace(stamp, planner=1, executor=1))
        self.assertEqual(coordinator.phase, Phase.JOINT)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory + "/checkpoint", coordinator,
                                   {"planner": torch.ones(2, device=self.device), "executor": torch.zeros(2, device=self.device)})
            restored, payload = load_checkpoint(path, map_location=self.device)
            self.assertEqual(restored.stamp, coordinator.stamp)
            self.assertEqual(float(payload["planner"].sum()), 2)
            with self.assertRaises(ValueError):
                restored.begin("p0", ["new"])

    def test_verl_batch_preserves_original_token_ids_and_field_masks(self):
        from scrl.models.planner import PlannerSample
        from scrl.models.verl_policy import focal_batch
        plan = FinitePlan((Segment("test", 1),))
        samples = [PlannerSample(plan, torch.tensor([[3, 4]]), torch.tensor([[5, 6, 7, 8]]),
                    torch.ones(4) * -.2, torch.ones(4) * -.3,
                    {i: [int(t)] for i, t in enumerate([5, 6, 7, 8])},
                    {0: DecisionTokens((0, 2), (2, 3))}, 2),
                   PlannerSample(plan, torch.tensor([[9]]), torch.tensor([[11, 12, 13]]),
                    torch.ones(3) * -.2, torch.ones(3) * -.3,
                    {i: [int(t)] for i, t in enumerate([11, 12, 13])},
                    {0: DecisionTokens((0, 1), (1, 2))}, 2)]
        tensors, metadata = focal_batch(samples, [1., -1.], [0, 0], pad_token_id=0, device=self.device)
        self.assertEqual(tensors["prompts"].tolist(), [[3, 4], [0, 9]])
        self.assertEqual(tensors["responses"].tolist(), [[5, 6, 7, 8], [11, 12, 13, 0]])
        self.assertEqual(tensors["response_mask"].tolist(), [[True, True, True, False], [True, True, False, False]])
        self.assertTrue(metadata["requires_constrained_log_probs"])

    def test_reward_ties_remain_neutral_and_failed_update_requires_recovery(self):
        from scrl.rl.objectives import normalize
        self.assertEqual(normalize([.5, .5]), [0., 0.])
        self.assertLess(abs(normalize([.5, .500001])[0]), .001)
        c = JointCoordinator()
        stamp = c.begin("x", ["a"])
        c.complete("a", stamp)
        def fail(*_):
            raise RuntimeError("simulated optimizer failure")
        with self.assertRaises(RuntimeError):
            c.commit(fail)
        with self.assertRaises(RuntimeError):
            c.state_dict()

    def test_noncontiguous_decoded_frames_encode_losslessly(self):
        import imageio_ffmpeg
        import subprocess
        from pathlib import Path
        from scrl.models.scoring.service import write_rgb
        torch.manual_seed(8)
        source = torch.randint(0, 256, (5, 3, 32, 64), dtype=torch.uint8, device=self.device)
        pixels = source.permute(0, 2, 3, 1).cpu()
        self.assertFalse(pixels.is_contiguous())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            write_rgb(path, pixels)
            raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
                                  "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                                 check=True, capture_output=True, timeout=30).stdout
            decoded = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape_as(pixels)
            torch.testing.assert_close(decoded, pixels, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
