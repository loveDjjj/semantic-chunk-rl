"""Empty evidence must recover from pixels without changing or inventing scores."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scrl.rl.reward import UnscorableReward
from scrl.models.scoring.grounding import ObservationTokens, choice_constraint, rating_choices
from scrl.models.scoring.protocol import parse_object, validate_judgment_scores, validate_video_judgment
from scrl.models.scoring.video_judge import VideoJudge


class CharacterTokenizer:
    eos_token_id = 127
    all_special_ids = [127]

    def get_vocab(self):
        return {chr(i): i for i in range(32, 128)}

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row) for row in rows]


class GroundingTests(unittest.TestCase):
    def test_nonfinite_json_is_rejected_before_writing_an_audit(self):
        for number in ('NaN', 'Infinity', '-Infinity', '1e999'):
            with self.subTest(number=number), self.assertRaises(UnscorableReward):
                parse_object('{"evidence": [[' + number + ', "office"]]}')

    def test_successful_format_retry_cannot_raise_or_lower_valid_initial_ratings(self):
        judge = VideoJudge.__new__(VideoJudge)
        judge.max_frames, judge.batch_size = 16, 2
        judge.processor = Mock()
        judge.processor.apply_chat_template.return_value = 'task<|im_end|>'
        judge.contract = Mock(return_value={'motion_mode': 'unspecified', 'requirements': ['desk']})
        initial = dict(alignment=0, dynamics=0, physics=4, entry=None, exit=None, evidence=[])
        valid = dict(initial, evidence=[[.75, 'A desk is visible.']])
        retry = dict(valid, alignment=4, dynamics=4, physics=0)
        judge._generate = Mock(side_effect=[[valid, initial], [retry]])
        judge._observe_frame = Mock(side_effect=AssertionError('Valid retry needs no extra observation'))
        with tempfile.TemporaryDirectory() as folder:
            judge.cache = Path(folder)/'contracts'
            with patch('scrl.models.scoring.video_judge.read_rgb', return_value=np.zeros((21, 4, 4, 3), dtype=np.uint8)):
                result = judge.score(dict(original='desk', fps=16, segment_start=9,
                                          segment_end=21, caption='A desk.'))
            local = result['scopes']['segment']
            self.assertEqual(local['judgment'], valid)
            self.assertEqual(local['evidence_audit']['repair_count'], 1)
            audit = json.loads(next((Path(folder)/'invalid_judgments').glob('*.json')).read_text())
            self.assertEqual(audit['repaired_response'], retry)
            self.assertEqual(audit['final_response'], valid)

    def test_empty_evidence_uses_real_frame_and_preserves_first_valid_scores(self):
        judge = VideoJudge.__new__(VideoJudge)
        pixels = object()
        judge._observe_frame = Mock(return_value='A person is seated in front of a desk.')
        judge._constrained_scores = Mock(side_effect=AssertionError('Must not rejudge valid scores'))
        initial = dict(alignment=0, dynamics=0, physics=4, entry=None, exit=None, evidence=[])
        source = copy.deepcopy(initial)
        value, audit = judge._grounded_judgment({}, [('initial', initial),
            ('retry', dict(initial, alignment=4))], None, None, pixels, 239, 16)
        judge._observe_frame.assert_called_once_with(pixels)
        self.assertEqual(initial, source)
        self.assertEqual([value[k] for k in ('alignment', 'dynamics', 'physics')], [0, 0, 4])
        self.assertEqual(audit['score_source'], 'initial')
        self.assertEqual(value['evidence'][0][0], 239/16)
        validate_video_judgment(value, begin=205, end=261, fps=16)

    def test_invalid_scores_require_model_rejudgment_never_a_default(self):
        judge = VideoJudge.__new__(VideoJudge)
        selected = dict(alignment=1, dynamics=2, physics=3, entry=None, exit=None, evidence=[])
        judge._constrained_scores = Mock(return_value=selected)
        judge._observe_frame = Mock(return_value='A chair is visible beside a desk.')
        value, audit = judge._grounded_judgment({}, [('initial', {}), ('retry', {'evidence': []})],
                                               'video', 'metadata', 'pixels', 10, 16)
        judge._constrained_scores.assert_called_once_with({}, 'video', 'metadata')
        self.assertEqual(audit['score_source'], 'constrained-video-ratings')
        self.assertEqual([value[k] for k in ('alignment', 'dynamics', 'physics')], [1, 2, 3])

    def test_unreadable_frame_is_not_replaced_with_canned_evidence(self):
        judge = VideoJudge.__new__(VideoJudge)
        judge._observe_frame = Mock(side_effect=OSError('unreadable pixels'))
        scores = dict(alignment=0, dynamics=0, physics=4, entry=None, exit=None, evidence=[])
        with self.assertRaises(OSError):
            judge._grounded_judgment({}, [('initial', scores)], None, None, None, 0, 16)

    def test_invalid_observation_is_unscorable_on_both_backends_but_io_and_oom_propagate(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        judge = VideoJudge.__new__(VideoJudge)
        judge.processor = Mock()
        judge.request_text = Mock(return_value='A, , .')
        with self.assertRaises(UnscorableReward):
            judge._observe_frame(frame)
        judge.request_text.side_effect = OSError('disconnected')
        with self.assertRaises(OSError):
            judge._observe_frame(frame)
        del judge.request_text
        judge.model, judge.observation_tokens = Mock(), Mock()
        judge.device, judge.pixel_budget = 'cpu', 1024
        judge.processor.return_value.to.return_value = {'input_ids': torch.tensor([[1]])}
        judge.model.generate.return_value = torch.tensor([[1, 2]])
        judge.processor.tokenizer.decode.return_value = 'A, , .'
        with self.assertRaises(UnscorableReward):
            judge._observe_frame(frame)
        judge.model.generate.side_effect = torch.OutOfMemoryError('out of memory')
        with self.assertRaises(torch.OutOfMemoryError):
            judge._observe_frame(frame)

    def test_score_decoder_covers_all_independent_ratings_and_forbids_early_eos(self):
        tok = CharacterTokenizer()
        choices = rating_choices()
        self.assertEqual(len(set(choices)), 125)
        constraint, budget = choice_constraint(tok, choices, 1)
        for choice in choices:
            row = json.loads(choice)
            validate_judgment_scores(row)
            prefix = [0]
            for token in tok.encode(choice):
                allowed = constraint(0, torch.tensor(prefix))
                self.assertNotIn(tok.eos_token_id, allowed)
                self.assertIn(token, allowed)
                prefix.append(token)
            self.assertEqual(constraint(0, torch.tensor(prefix)), [tok.eos_token_id])
            self.assertLessEqual(len(prefix), budget)

    def test_observation_cannot_be_empty_whitespace_or_exceed_budget(self):
        tok = CharacterTokenizer()
        constraint = ObservationTokens(tok).constraint(0, max_chars=12)
        self.assertNotIn(tok.eos_token_id, constraint(0, torch.tensor([], dtype=torch.long)))
        self.assertNotIn(ord(' '), constraint(0, torch.tensor([], dtype=torch.long)))
        for i in range(8):
            allowed = constraint(0, torch.tensor(tok.encode('a'*i), dtype=torch.long))
            self.assertNotIn(tok.eos_token_id, allowed)
            self.assertIn(ord('a'), allowed)
        self.assertIn(tok.eos_token_id, constraint(0, torch.tensor(tok.encode('a'*8))))
        self.assertEqual(constraint(0, torch.tensor(tok.encode('a'*12))), [tok.eos_token_id])
        invalid = dict(alignment=0, dynamics=0, physics=4, entry=None, exit=None,
                       evidence=[[0, '   ']])
        with self.assertRaises(UnscorableReward):
            validate_video_judgment(invalid, begin=0, end=10, fps=16)


if __name__ == '__main__':
    unittest.main()
