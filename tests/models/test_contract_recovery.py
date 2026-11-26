"""Model schema errors and corrupt caches must not discard original requirements."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scrl.plan import digest
from scrl.rl.reward import UnscorableReward
from scrl.models.scoring.grounding import motion_choices, motion_mode_from_flags
from scrl.models.scoring.protocol import VERSION, source_requirements, validate_contract
from scrl.models.scoring.video_judge import VideoJudge


class ContractRecoveryTests(unittest.TestCase):
    def test_independent_motion_flags_cover_all_modes_without_suppressing_subject_motion(self):
        choices = [json.loads(x) for x in motion_choices()]
        self.assertEqual(len(choices), 8)
        self.assertEqual({motion_mode_from_flags(x) for x in choices},
                         {'subject', 'camera', 'mixed', 'static', 'unspecified'})
        for row in choices:
            if row['subject_motion'] and row['camera_motion']:
                self.assertEqual(motion_mode_from_flags(row), 'mixed')
            elif row['subject_motion']:
                self.assertEqual(motion_mode_from_flags(row), 'subject')
        with self.assertRaises(ValueError):
            motion_mode_from_flags(dict(subject_motion=1, camera_motion=False, explicit_stillness=False))

    def make_judge(self, folder, responses):
        judge = VideoJudge.__new__(VideoJudge)
        judge.cache = Path(folder)/'contracts'
        judge.cache.mkdir()
        judge.processor = Mock()
        judge.processor.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]['content'][0]['text']
        judge._generate = Mock(side_effect=[[value] for value in responses])
        judge._constrained_motion_mode = Mock(return_value='subject')
        return judge

    def test_all_json_types_reject_as_validation_errors(self):
        values = [None, [], ['bad'], 3, 'bad', {},
                  {'motion_mode': [], 'requirements': ['dog']},
                  {'motion_mode': {}, 'requirements': ['dog']},
                  {'motion_mode': True, 'requirements': ['dog']}]
        for value in values:
            with self.subTest(value=value), self.assertRaises(UnscorableReward):
                validate_contract(value)

    def test_schema_retry_succeeds_without_invoking_fallback(self):
        bad = dict(motion_mode='subject', requirements=['A dog runs.']*9)
        good = dict(motion_mode='subject', requirements=['A dog runs.'])
        with tempfile.TemporaryDirectory() as folder:
            judge = self.make_judge(folder, [bad, good])
            self.assertEqual(judge.contract('A dog runs.'), good)
            judge._constrained_motion_mode.assert_not_called()
            audit = json.loads(next((Path(folder)/'invalid_contracts').glob('*.json')).read_text())
            self.assertEqual(audit['state'], 'repaired')
            self.assertEqual(audit['attempts'][0]['response'], bad)
            self.assertEqual(audit['final_response'], good)
            self.assertEqual(judge.contract('A dog runs.'), good)
            self.assertEqual(judge._generate.call_count, 2)

    def test_exhausted_retries_preserve_the_complete_source_for_every_bad_list(self):
        original = 'A robot lifts a blue cube. '*14 + 'It places the cube on a red pad.'
        bad_values = [dict(motion_mode='subject', requirements=rows) for rows in
                      ([], ['action']*9, ['x'*241], [' '], 'action', [None], [3], [{'action': 'lift'}])]
        bad_values += [{'invalid_json': '{"motion_mode":', 'parse_error': 'truncated JSON'},
                       {'motion_mode': ['subject'], 'requirements': ['action']}]
        for bad in bad_values:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                judge = self.make_judge(folder, [bad, bad])
                result = judge.contract(original)
                self.assertEqual(''.join(result['requirements']), ' '.join(original.split()))
                self.assertEqual(result['motion_mode'], 'subject')
                validate_contract(result, original)
                judge._constrained_motion_mode.assert_called_once_with(original)
                audit = json.loads(next((Path(folder)/'invalid_contracts').glob('*.json')).read_text())
                self.assertEqual(audit['state'], 'source-backed')
                self.assertTrue(audit['complete_source_retained'])
                self.assertEqual(len(audit['attempts']), 2)

    def test_invalid_cache_is_audited_and_replaced_atomically(self):
        for raw in ('{broken', '[]', '{"motion_mode":[],"requirements":["dog"]}'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as folder:
                good = {'motion_mode': 'subject', 'requirements': ['A dog runs.']}
                judge = self.make_judge(folder, [good])
                path = judge.cache/(digest({'original': 'A dog runs.', 'version': VERSION})+'.json')
                path.write_text(raw)
                self.assertEqual(judge.contract('A dog runs.'), good)
                self.assertEqual(json.loads(path.read_text()), good)
                audit = json.loads(next((Path(folder)/'invalid_contracts').glob('*.json')).read_text())
                self.assertEqual(audit['attempts'][0]['raw'], raw)

    def test_fallback_keeps_no_invented_stillness_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            judge = self.make_judge(folder, [{}, {}])
            judge._constrained_motion_mode.return_value = 'static'
            self.assertEqual(judge.contract('a black bowl')['motion_mode'], 'unspecified')

    def test_source_capacity_is_explicit_and_whitespace_does_not_drop_words(self):
        original = '  A dog\n runs.\tThen it sits.  '
        self.assertEqual(''.join(source_requirements(original)), 'A dog runs. Then it sits.')
        for prompt in ('', ' ', None, 'x'*1921):
            with self.subTest(prompt=str(prompt)[:20]), self.assertRaises(ValueError):
                source_requirements(prompt)

    def test_model_execution_errors_are_not_disguised_as_contracts(self):
        with tempfile.TemporaryDirectory() as folder:
            judge = self.make_judge(folder, [])
            judge._generate.side_effect = RuntimeError('GPU execution failed')
            with self.assertRaisesRegex(RuntimeError, 'GPU execution'):
                judge.contract('A dog runs.')
            self.assertEqual(list(judge.cache.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
