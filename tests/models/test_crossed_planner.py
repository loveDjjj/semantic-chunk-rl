import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from scrl.models.planner import (
    CrossedPlannerSample, FieldTrace, PlannerPolicy, crossed_terms, field_loss, token_log_probs,
)
from scrl.models.verl_policy import VerlPlannerPolicy, crossed_batch
from scrl.plan import FinitePlan, PlanLimits, Segment


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros((7, 7)))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits[input_ids])


def trace(first, responses, support=(1, 2)):
    response = torch.tensor([responses])
    constraints = {i: list(support) for i in range(len(responses))}
    logps = torch.full((len(responses),), -torch.tensor(len(support)).float().log())
    return FieldTrace(torch.tensor([[first]]), response, constraints, logps, logps.clone())


class CrossedPlannerTests(unittest.TestCase):
    def setUp(self):
        self.anchor = FinitePlan(tuple(Segment(c, n) for c, n in zip("abcd", (1, 2, 4, 1))))
        self.policy = object.__new__(PlannerPolicy)
        self.policy.device = torch.device("cpu")
        self.policy.activate = Mock()
        self.policy.limits = PlanLimits()
        self.policy.version = 0
        self.policy.tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([[1]])))

    def test_sample_is_a_product_with_a_fixed_pair_budget_and_suffix(self):
        caption, duration = [object(), object()], [object(), object()]
        self.policy._cross_caption = Mock(side_effect=[("x", caption[0]), ("y", caption[1])])
        self.policy._cross_durations = Mock(return_value=[(2, duration[0]), (4, duration[1])])
        cells = self.policy.sample_crossed("p", self.anchor, 1, 11)
        self.assertEqual([s.plan.segments[1].caption for s in cells], ["x", "x", "y", "y"])
        self.assertEqual([s.plan.segments[1].num_chunks for s in cells], [2, 4, 2, 4])
        self.assertEqual([s.plan.segments[2].num_chunks for s in cells], [4, 2, 4, 2])
        for i, s in enumerate(cells):
            self.assertEqual(s.plan.total_chunks, self.anchor.total_chunks)
            self.assertEqual(s.plan.segments[0], self.anchor.segments[0])
            self.assertEqual(s.plan.segments[-1], self.anchor.segments[-1])
            self.assertEqual(s.plan.segments[2].caption, "c")
            self.assertIs(s.caption_trace, caption[i // 2])
            self.assertIs(s.duration_trace, duration[i % 2])

    def test_last_segment_and_duplicate_caption_yield_no_field_traces(self):
        self.policy._cross_caption = Mock(return_value=("same", object()))
        self.policy._cross_durations = Mock()
        cells = self.policy.sample_crossed("p", self.anchor, 3, 4)
        self.policy._cross_durations.assert_not_called()
        self.assertTrue(all(s.caption_trace is None and s.duration_trace is None for s in cells))

    def test_duration_old_and_reference_use_only_the_pair_support(self):
        p = self.policy
        p.model = TinyLM()
        p.tokenizer.encode.side_effect = lambda x, **_: [int(x)]
        chosen = p._cross_durations(torch.tensor([[0]]), [2, 4], 1)
        for chunks, t in chosen:
            self.assertIn(chunks, (2, 4))
            self.assertEqual(t.constraints, {0: [2, 4]})
            new = token_log_probs(p.model, t.context, t.response, t.constraints)
            torch.testing.assert_close(new, t.old_log_probs)
            torch.testing.assert_close(new, t.reference_log_probs)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
    def test_transformers_caption_stops_at_field_and_replays_behavior(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        pieces = ["<eos>", '[{"step_index":0,"prompt":"', "1", "2", "4", "6", '",', "x"]
        tok = Mock(eos_token_id=0, pad_token_id=0, all_special_ids=[0])
        tok.get_vocab.return_value = {piece: i for i, piece in enumerate(pieces)}
        tok.batch_decode.side_effect = lambda rows, **_: ["".join(pieces[i] for i in row) for row in rows]
        tok.decode.side_effect = lambda ids, **_: "".join(pieces[int(i)] for i in ids)
        tok.encode.side_effect = lambda piece, **_: [pieces.index(piece)]
        p = self.policy
        p.tokenizer, p.grammar = tok, None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            p.model = GPT2LMHeadModel(GPT2Config(vocab_size=len(pieces), n_layer=1, n_head=1,
                                                  n_embd=4, n_positions=80)).eval()
            caption, t = p._cross_caption(torch.tensor([[7]]), torch.tensor([[1]]), 18, 64)
        self.assertTrue(caption.strip())
        self.assertEqual(t.response[0, -1].item(), 6)
        new = token_log_probs(p.model, t.context, t.response, t.constraints)
        torch.testing.assert_close(t.old_log_probs, new, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(t.reference_log_probs, new)

    def test_only_credited_field_moves_then_ties_skip_adam_momentum(self):
        p = self.policy
        cap, dur = trace(3, [1]), trace(4, [2])
        dur.reference_log_probs += .9
        for axis, active_token, inactive_token in [("caption", 3, 4), ("duration", 4, 3)]:
            p.model = TinyLM()
            p.parameters = list(p.model.parameters())
            p.optimizer = torch.optim.AdamW(p.parameters, lr=.05, weight_decay=0)
            sample = CrossedPlannerSample(self.anchor, cap, dur, p.version)
            credit = {"caption": 0., "duration": 0., axis: 1.}
            self.assertTrue(p.update([sample], [credit], [1])["updated"])
            self.assertGreater(p.model.logits[active_token].abs().sum().item(), 0.)
            self.assertEqual(p.model.logits[inactive_token].abs().sum().item(), 0.)
            snapshot = p.model.logits.detach().clone()
            version = sample.policy_version = p.version
            for actor in (PlannerPolicy, VerlPlannerPolicy):
                self.assertFalse(actor.update(p, [sample], [dict.fromkeys(credit, 0.)], [1])["updated"])
            self.assertEqual(p.version, version)
            torch.testing.assert_close(p.model.logits, snapshot, rtol=0, atol=0)

    def test_verl_field_rows_match_reference_objective_and_cell_denominator(self):
        cap, dur = trace(3, [1, 2]), trace(4, [2])
        sample = CrossedPlannerSample(self.anchor, cap, dur, 0)
        null = CrossedPlannerSample(self.anchor, None, None, 0)
        terms = crossed_terms([sample, sample, null], [{"caption": 1., "duration": -.2}] * 2
                              + [{"caption": 0., "duration": 0.}], [1] * 3, version=0, duration_weight=.3)
        self.assertEqual(len(terms), 2)
        batch = crossed_batch(terms, pad_token_id=0, device="cpu")
        self.assertFalse(batch["attention_mask"][:, 0].any())
        self.assertEqual(batch["position_ids"][:, 0].tolist(), [0, 1])
        weights = batch["response_mask"]
        self.assertAlmostEqual(weights.sum(1)[0].item() / 2, .7 * 2 / 3)
        self.assertAlmostEqual(weights.sum(1)[1].item() / 2, .3 * 2 / 3)
        self.assertEqual(weights[1, 1].item(), 0)
        m = TinyLM()
        ref_loss = sum(w * field_loss(token_log_probs(m, t.context, t.response, t.constraints),
            t.old_log_probs, t.reference_log_probs, a, .2, .01) for t, a, w in terms)
        self.assertAlmostEqual(ref_loss.item(), (-batch["advantages"] * weights).sum(1).mean().item())
        with self.assertRaisesRegex(ValueError, "absent"):
            crossed_terms([null], [{"caption": 1., "duration": 0.}], [0], version=0, duration_weight=.5)


if __name__ == "__main__":
    unittest.main()
