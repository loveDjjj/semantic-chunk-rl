from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scrl.evaluation.io import atomic_json, read
from scrl.evaluation.protocol import DIMENSIONS
from scrl.evaluation.score_when_ready import main, ready
from scrl.plan import digest


class ScoreWatcherTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)
        self.folder = self.run/'evaluation/base'
        self.shard = self.run/'workers/base/generate-0/status.json'
        tasks = [{'id': 'a'}]
        atomic_json(self.run/'inputs/tasks.json', tasks)
        atomic_json(self.run/'config.json', dict(arms={'base': {}}, vbench_weights='/weights',
                    vbench_source='/upstream', scoring_protocol={'tasks_sha256': digest(tasks)}))
        self.args = [str(self.run), '--arm','base','--shards','1','--expected-tasks','1']

    def test_wait_fail_missing_records_and_lock(self):
        self.assertEqual(ready(self.run,'base',1,{'a'}), 0)
        atomic_json(self.shard, dict(shard=0, state='failed'))
        with self.assertRaises(RuntimeError):
            main(self.args)
        self.assertEqual(read(self.folder/'watch-status.json')['state'], 'failed')
        atomic_json(self.shard, dict(shard=0,state='complete'))
        with self.assertRaisesRegex(ValueError, 'records'):
            main(self.args)
        (self.folder/'.score-owner').mkdir()
        with self.assertRaises(FileExistsError):
            main(self.args)

    def test_publish_only_complete_and_never_resubmit(self):
        atomic_json(self.shard, dict(shard=0,state='complete'))
        atomic_json(self.run/'records/base/a.json', {})
        report = dict(arm='base', aggregate={'state':'complete','quality':.5,'semantic':.5,'total':.5},
                      raw_dimensions={d: .5 for d in DIMENSIONS})
        def scorer(cmd, **kwargs):
            self.assertNotIn('--dimension', cmd)
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
            atomic_json(Path(cmd[-1]), report)
        with patch('scrl.evaluation.score_when_ready.subprocess.run', side_effect=scorer) as sub:
            report['aggregate']['state'] = 'inconclusive_missing_dimension'
            with self.assertRaises(ValueError):
                main(self.args)
            self.assertFalse((self.folder/'summary.json').exists())
            report['aggregate']['state'] = 'complete'
            main(self.args)
            main(self.args)
            self.assertEqual(sub.call_count, 2)
        self.assertEqual(read(self.folder/'summary.json'), report)
