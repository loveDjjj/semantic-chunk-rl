import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scrl.config import DEFAULT_CONFIGS,read
from scrl.evaluation.protocol import DIMENSIONS, make_tasks
from scrl.rl import cycle
from scrl.runtime.transport import atomic_json


class PeriodicVBenchTests(unittest.TestCase):
    def test_history_and_device_bindings(self):
        prompts=read(DEFAULT_CONFIGS/'vbench-periodic-prompts.json')
        self.assertEqual(len(prompts),95)
        self.assertEqual(len(make_tasks(prompts,read(DEFAULT_CONFIGS/'vbench.json'))),635)
        from scrl.config import reserve_evaluator
        inventory=read(DEFAULT_CONFIGS/'cluster.example.json')
        node,train=reserve_evaluator(inventory,5)
        self.assertEqual(node['physical_gpus'],[2,3])
        self.assertEqual(len(set(cycle.cards(train))),24)
        self.assertNotIn(node['address'],[a for a,_ in cycle.cards(train)])
        self.assertEqual(train['nodes'][5]['address'],inventory['nodes'][6]['address'])

    def test_watch_queue(self):
        from scrl.rl.watch import published
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in [60,40,22,20,80]:
                cp=root/f'checkpoint-{n:05d}'
                cp.mkdir()
                for f in ('planner.pt','state.json','executor.pt'):
                    if f!='executor.pt' or n!=80:
                        (cp/f).touch()
            self.assertEqual([n for n,_ in published(root,20)],[20,40,60])

    def test_failure_at_50_then_retry_before_51(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'meta').touch()
            cfg=dict(run=tmp,token='not-to-be-saved',nodes=[dict(index=0,address='a',gpus=2)],
                     source_sha256='test', assets={'rewards':{'quality':dict(source=tmp,weights=tmp)}},
                     resume=str(root/'checkpoint-00048'),validation_round_limit=3,repeat_dataset=True,
                     prompts=['test'],python='python',training={},adaptive_sampling=[],seed=1,reward={},
                     dataset_sha256='test',plan={},periodic_vbench=dict(every_steps=50,score_node=0,
                         prompts='vbench-periodic-prompts.json',recipe='vbench.json',
                         metadata_relative_to_vbench='meta',weights_relative_to_training_weights='.'))
            calls=[]
            def train(addr,argv,log,token=False):
                phase=Path(argv[-1]); child=read(phase/'config.json')
                self.assertNotIn('token',child)
                start=int(Path(child['resume']).name.split('-')[1]);end=start+child['validation_round_limit']
                calls.append(('train',start,end))
                atomic_json(phase/'status.json',dict(completed_rounds=end,cursor=1,state='drained'))
                atomic_json(phase/'nodes/node-0/stopped.json',{})
                atomic_json(phase/'checkpoints'/f'checkpoint-{end:05d}'/'state.json',
                            dict(format='split-policies-v1',files={'planner.pt':'p','executor.pt':'e'}))
            fleet=Mock()
            fleet.run.side_effect=train
            def evaluate(*args):
                calls.append(('eval',args[3]))
                if len(calls)==2:raise ValueError('missing dimension')
            with patch.object(cycle,'evaluate',side_effect=evaluate):
                with self.assertRaisesRegex(ValueError,'missing dimension'):
                    cycle._cycle(cfg,fleet)
                self.assertFalse((root/'windows/to-00051').exists())
                cycle._cycle(cfg,fleet)
            self.assertEqual(calls,[('train',48,50),('eval',50),('eval',50),('train',50,51)])

    def test_no_partial_or_nan_gate(self):
        report={'aggregate':dict(state='complete',quality=1.,semantic=1.,total=1.),
                'raw_dimensions':dict.fromkeys(DIMENSIONS,1.)}
        cycle.assert_gate(report)
        report['raw_dimensions'][DIMENSIONS[0]]=float('nan')
        with self.assertRaises(ValueError):cycle.assert_gate(report)
        del report['raw_dimensions'][DIMENSIONS[0]]
        with self.assertRaises(ValueError):cycle.assert_gate(report)
