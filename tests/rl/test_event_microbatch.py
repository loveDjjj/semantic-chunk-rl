import unittest

import numpy as np
import torch

from scrl.models.scoring import events
from scrl.models.scoring.grounding import event_batch_size
from scrl.models.scoring.video_judge import VideoJudge
from scrl.rl.reward import UnscorableReward


class Tokenizer:
    eos_token_id, pad_token_id, padding_side = 1, 0, "left"

    def encode(self, text, **_):
        return list(map(ord, text))

    def decode(self, ids, **_):
        return "".join(map(chr, ids))


class Inputs(dict):
    def to(self, _device):
        return self


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **_):
        return messages[0]["content"][-1]["text"]

    def __call__(self, text, videos, videos_kwargs, **_):
        assert videos[0] is videos[-1]
        assert videos_kwargs["video_metadata"][0] is videos_kwargs["video_metadata"][-1]
        width = max(map(len,text))
        return Inputs(input_ids=torch.tensor([[0]*(width-len(s))+list(map(ord,s)) for s in text]))


class EachRowStops:
    def generate(self, input_ids, max_new_tokens, eos_token_id, pad_token_id, prefix_allowed_tokens_fn, **_):
        finished = [False] * len(input_ids)
        for _ in range(max_new_tokens):
            next_tokens = []
            for row, ids in enumerate(input_ids):
                allowed = prefix_allowed_tokens_fn(row, ids)
                if finished[row]:
                    assert allowed == [eos_token_id]
                    next_tokens.append(pad_token_id)
                else:
                    next_tokens.append(allowed[0])
                    finished[row] = allowed[0] == eos_token_id
            input_ids = torch.cat((input_ids,torch.tensor(next_tokens)[:,None]),dim=1)
            if all(finished):
                return input_ids
        raise AssertionError("Rows did not terminate")


class FakeJudge:
    event_decisions = events.BOUNDED_DECISIONS

    def __init__(self, size, unknown=False):
        self.event_choice_batch_size, self.unknown = size, unknown

    def event_choice(self, prompt, choices, *_):
        if self.unknown and 'single event: "dial".' in prompt:
            return "unknown"
        return "done 1" if "unknown" in choices else "2 3"

    def event_choices(self, batch, frames, meta):
        return [self.event_choice(p, choices, frames, meta) for p, choices in batch]

    def _observe_frame(self, frame):
        return "Observation of frame " + str(frame)


class EventMicrobatchTests(unittest.TestCase):
    def test_real_hf_adapter_with_row_specific_tries_left_padding_and_independent_eos(self):
        judge = VideoJudge.__new__(VideoJudge)
        judge.device, judge.max_frames, judge.pixel_budget = "cpu", 4, 2048
        judge.model, judge.processor = EachRowStops(), Processor()
        queries = [("p", ["a"]), ("long input", ["zeta"])]
        self.assertEqual(judge.event_choices(queries, object(), object()), ["a", "zeta"])
        judge.processor.tokenizer.padding_side = "right"
        with self.assertRaisesRegex(ValueError, "left padding"):
            judge.event_choices(queries, object(), object())

    def test_serial_and_batch_same_result_but_fewer_choice_calls(self):
        data = dict(original="dial", scope="global", events=["dial", "lid", "light"])
        receipts, results = [], []
        for size in (1, 2):
            receipt = {}
            results.append(events.rejudge_events(FakeJudge(size), data, np.arange(4), None, receipt))
            receipts.append(receipt)
        self.assertEqual(results[0], results[1])
        self.assertEqual([len(r["choice_calls"]) for r in receipts], [4,2])
        self.assertEqual([r["rows"] for r in receipts[1]["choice_calls"]], [[0,1],[2,"ratings"]])
        self.assertEqual(len(receipts[1]["observations"]), 1)
        self.assertTrue(all(c["seconds"] >= 0 for c in receipts[1]["choice_calls"]))

    def test_early_unknown_cannot_be_averaged_with_known_peer(self):
        data = dict(original="dial", scope="global", events=["dial", "lid"])
        receipt = {}
        with self.assertRaisesRegex(UnscorableReward, "event 0: unknown"):
            events.rejudge_events(FakeJudge(2, unknown=True), data, np.arange(4), None, receipt)
        self.assertEqual(receipt["choice_calls"][0]["answers"], ["unknown", "done 1"])
        self.assertFalse(receipt["observations"])

    def test_http_and_default_remain_serial(self):
        judge = FakeJudge(2)
        judge.request_text = object()
        receipt = {}
        events.rejudge_events(judge, dict(events=["lid"], original="lid", scope="segment"),
                              np.arange(4), None, receipt)
        self.assertEqual((receipt["backend"], receipt["effective_batch_size"]), ("http-serial",1))
        self.assertEqual(event_batch_size({}), 1)
        for size in (True, 0, 3, "2"):
            with self.assertRaises(ValueError):
                event_batch_size(dict(event_choice_batch_size=size))
