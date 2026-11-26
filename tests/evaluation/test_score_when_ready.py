from pathlib import Path
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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

    def test_incremental_uses_two_lanes_before_completion_and_finalizes_once(self):
        tasks = [dict(id='a',sample_index=0,dimensions=[d for d in DIMENSIONS if d!='temporal_flickering']),
                 dict(id='b',sample_index=0,dimensions=['temporal_flickering'])]
        cfg = read(self.run/'config.json')
        cfg.update(protocol_sha256='p', scoring_protocol={'tasks_sha256':digest(tasks)})
        atomic_json(self.run/'config.json', cfg)
        atomic_json(self.run/'inputs/tasks.json', tasks)
        atomic_json(self.run/'records/base/a.json', {})
        atomic_json(self.shard, dict(shard=0,state='generating'))
        def receipt(d):
            atomic_json(self.folder/(d+'.json'),dict(dimension=d,arm='base',state='complete',
                        protocol_sha256='p',inputs_sha256='a'*64))
        receipt('scene')  # A prior completed bound dimension should not be relaunched.
        atomic_json(self.folder/'color.json',dict(state='complete'))  # Old unbound receipt.
        jobs = []
        def launch(cmd, env, **kwargs):
            d = cmd[-1]
            self.assertEqual(cmd[-2], '--dimension')
            jobs.append((d,env['CUDA_VISIBLE_DEVICES'],read(self.shard)['state']))
            receipt(d)
            return SimpleNamespace(pid=123,poll=lambda:0)
        def tick(_):
            atomic_json(self.run/'records/base/b.json',{})
            atomic_json(self.shard,dict(shard=0,state='complete'))
        def finalize(cmd,**kwargs):
            self.assertIn('--finalize-only',cmd)
            self.assertEqual(read(self.shard)['state'],'complete')
            atomic_json(Path(cmd[-1]), dict(arm='base',raw_dimensions={d:1 for d in DIMENSIONS},
                        aggregate=dict(quality=1,semantic=1,total=1,state='complete')))
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='2,3'), \
                patch('scrl.evaluation.score_when_ready.subprocess.Popen',side_effect=launch), \
                patch('scrl.evaluation.score_when_ready.time.sleep',side_effect=tick), \
                patch('scrl.evaluation.score_when_ready.subprocess.run',side_effect=finalize) as end:
            main([*self.args[:-1],'2','--incremental'])
        dims = [j[0] for j in jobs]
        self.assertEqual(len(dims),len(set(dims)))
        self.assertEqual(set(dims),{'static_filter',*DIMENSIONS}-{'scene'})
        self.assertEqual([j[1:] for j in jobs[:2]], [('2','generating'),('3','generating')])
        self.assertLess(dims.index('static_filter'),dims.index('temporal_flickering'))
        self.assertEqual(end.call_count,1)
        self.assertEqual(read(self.folder/'watch-status.json')['state'],'complete')

    def test_sparse_indices_yield_without_summary_but_keep_full_id_gate(self):
        cfg = read(self.run/'config.json')
        tasks = [dict(id='a',sample_index=0,dimensions=list(DIMENSIONS))]
        cfg.update(protocol_sha256='p',scoring_protocol={'tasks_sha256':digest(tasks)})
        atomic_json(self.run/'config.json',cfg)
        atomic_json(self.run/'inputs/tasks.json',tasks)
        atomic_json(self.shard,dict(shard=0,state='failed'))  # Not a member of this arm.
        actual = self.run/'workers/base/generate-25/status.json'
        atomic_json(actual,dict(shard=25,state='generating'))
        flags = ['--incremental','--yield-when-idle','--shard-indices','25']
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='2'):
            main([*self.args,*flags])
            self.assertEqual(read(self.folder/'watch-status.json')['state'],'yielded')
            self.assertFalse((self.folder/'.score-owner').exists())
            self.assertFalse((self.folder/'summary.json').exists())
            atomic_json(actual,dict(shard=25,state='complete'))
            main([*self.args,*flags])
            self.assertEqual(read(self.folder/'watch-status.json')['missing_records'],1)
            with self.assertRaisesRegex(RuntimeError,'missing records=1'):
                main([*self.args,'--incremental','--shard-indices','25'])
            atomic_json(self.run/'records/base/unexpected.json',{})
            with self.assertRaisesRegex(ValueError,'outside'):
                main([*self.args,*flags])

    def test_terminal_missing_record_failed_shard_and_failed_metric_do_not_block_other_scores(self):
        tasks = [dict(id='a',sample_index=0,dimensions=[d for d in DIMENSIONS if d!='temporal_flickering']),
                 dict(id='b',sample_index=0,dimensions=['temporal_flickering','color'])]
        cfg = read(self.run/'config.json')
        cfg.update(protocol_sha256='p',scoring_protocol={'tasks_sha256':digest(tasks)})
        atomic_json(self.run/'config.json',cfg)
        atomic_json(self.run/'inputs/tasks.json',tasks)
        atomic_json(self.run/'records/base/a.json',{})
        for state in ('complete','failed'):
            atomic_json(self.shard,dict(shard=0,state=state))
            jobs = []
            def launch(cmd, **_):
                d = cmd[-1]
                jobs.append(d)
                if d != 'scene':
                    atomic_json(self.folder/(d+'.json'),dict(arm='base',dimension=d,state='complete',
                                protocol_sha256='p',inputs_sha256='b'*64))
                return SimpleNamespace(pid=321,poll=lambda: int(d == 'scene'))
            with self.subTest(state=state), patch.dict(os.environ,CUDA_VISIBLE_DEVICES='2,3'), \
                    patch('scrl.evaluation.score_when_ready.time.sleep'), \
                    patch('scrl.evaluation.score_when_ready.subprocess.Popen',side_effect=launch), \
                    patch('scrl.evaluation.score_when_ready.subprocess.run') as finalize:
                main([*self.args[:-1],'2','--incremental','--yield-when-idle'])
                finalize.assert_not_called()
                self.assertEqual(set(jobs),set(DIMENSIONS)-{'temporal_flickering','color'})
                self.assertEqual(len(jobs),len(set(jobs)))
                result = read(self.folder/'watch-status.json')
                self.assertEqual(result['state'],'yielded')
                self.assertEqual(result['missing_records'],1)
                self.assertEqual(set(result['failed_dimensions']),{'scene'})
                self.assertEqual(result['failed_shards'],[0] if state=='failed' else [])
                for d in result['completed_dimensions']:
                    (self.folder/(d+'.json')).unlink()
