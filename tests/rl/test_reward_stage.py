"""Regression checks for changed reward evidence, capacity and recovery semantics."""

from concurrent.futures import ThreadPoolExecutor
import json
import copy
import os
import shutil
from pathlib import Path
import tempfile
import threading
import unittest

import torch

from scrl.rl.reward import RewardComposer, RewardConfig, RewardBundle, ScopeScores, UnscorableReward
from scrl.models.scoring.protocol import QUALITY_WEIGHTS, dynamic_score, parse_object, validate_contract, validate_judgment
from scrl.runtime.transport import EvaluationDispatcher
from scrl.runtime.launch import explicit_layout
from scrl.rl.checkpoint import freeze_cpu, SnapshotWriter
from scrl.rl.checkpoint import JointCoordinator, publish_split_checkpoint, load_checkpoint
from scrl.rl.identity import training_fingerprint
from scrl.plan import PolicyStamp


class RewardStageTests(unittest.TestCase):
    def test_missing_invalid_and_duplicate_evidence_never_become_zero_rewards(self):
        for text in ('{} trailing', '{"alignment":1,"alignment":4}', '[1]'):
            with self.assertRaises(UnscorableReward):parse_object(text)
        row = dict(alignment=4, dynamics=3, physics=2, entry=None, exit=None,
                   evidence=[[.2, 'The person remains still.']])
        validate_judgment(row, entry=False, exit=False, start=0, end=1)
        validate_judgment(dict(row,evidence=[[i/10,'Visible action'] for i in range(4)]),
                          entry=False,exit=False,start=0,end=1)
        for update in ({'dynamics': True}, {'physics': 5}, {'entry': {}},
                       {'evidence': [[10., 'Outside the video']]}):
            with self.assertRaises(UnscorableReward):
                validate_judgment(dict(row, **update), entry=False, exit=False, start=0, end=1)

    def test_unrelated_flow_cannot_rescue_missing_action_and_static_needs_no_flow(self):
        self.assertEqual(dynamic_score({'dynamics': 0}, 'subject', 1.), 0.)
        self.assertEqual(dynamic_score({'dynamics': 4}, 'static'), 1.)
        self.assertEqual(dynamic_score({'dynamics': 4}, 'unspecified'), 1.)
        self.assertLess(dynamic_score({'dynamics': 4}, 'subject', 0.), 1.)
        with self.assertRaises(UnscorableReward):dynamic_score({'dynamics': 4}, 'subject')
        with self.assertRaises(UnscorableReward):validate_contract({'motion_mode':'none','requirements':['person']})
        contract = {'motion_mode':'static','requirements':['a black bowl']}
        self.assertEqual(validate_contract(contract, 'a black bowl')['motion_mode'], 'unspecified')
        self.assertEqual(validate_contract(contract, 'a black bowl stays still')['motion_mode'], 'static')

    def test_reward_is_quality_led_and_boundary_count_does_not_increase_return(self):
        cfg = RewardConfig(quality_components=QUALITY_WEIGHTS,
                           boundary_components={'appearance':.5, 'motion':.5})
        composer = RewardComposer(cfg, 'test-v2')
        scope = ScopeScores({k: .5 for k in QUALITY_WEIGHTS}, {'alignment': .5})
        result = composer.compose(RewardBundle(scope, scope, 'test-v2', {'appearance':.5, 'motion':.5}))
        self.assertAlmostEqual(result['planner'], .5)
        self.assertAlmostEqual(result['executor'], .5)
        incomplete = ScopeScores({k: .5 for k in QUALITY_WEIGHTS if k != 'dynamics'}, {'alignment':.5})
        with self.assertRaises(UnscorableReward):composer.compose(RewardBundle(incomplete, scope, 'test-v2'))

    def test_16_cards_include_12_executors_4_independent_services_and_remote_actor(self):
        nodes = [dict(index=i, address=f'node-{i}', gpus=4) for i in range(4)]
        cfg = explicit_layout(nodes, [(2,2),(2,3),(3,2),(3,3)], actor_node=2)
        self.assertEqual(cfg['executor_count'],12)
        roles = cfg['executors']+cfg['evaluations']+cfg['planners']+[cfg['dispatcher']]
        self.assertEqual(len({(r['node'],r['port']) for r in roles}),len(roles))
        self.assertEqual(len({(r['node'],r['device']) for r in cfg['executors']+cfg['evaluations']}),16)
        self.assertEqual(cfg['actor_node'],2)
        self.assertEqual(len({r['rank'] for r in cfg['evaluations']}),4)

    def test_optimizer_and_parameter_snapshots_cannot_change_in_background(self):
        parameter = torch.tensor([2.])
        optimizer = torch.optim.Adam([parameter.requires_grad_()])
        parameter.grad = torch.ones_like(parameter);optimizer.step()
        original = dict(weights=parameter, optimizer=optimizer.state_dict())
        snapshot = freeze_cpu(original)
        expected = snapshot['optimizer']['state'][0]['exp_avg'].clone()
        parameter.grad = torch.ones_like(parameter);optimizer.step()
        self.assertNotEqual(parameter.item(),snapshot['weights'].item())
        torch.testing.assert_close(snapshot['optimizer']['state'][0]['exp_avg'],expected)
        ready=threading.Event()
        writer=SnapshotWriter()
        try:
            writer.submit(lambda: ready.wait(2))
            self.assertIsNone(writer.check());ready.set();self.assertTrue(writer.wait())
        finally:ready.set();writer.close()

    def test_latest_and_every_fiftieth_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for step in (45,50,55,95,100,105):
                staging=root/f'.{step}';staging.mkdir()
                torch.save({'executor':torch.tensor([step])},staging/'executor.pt')
                publish_split_checkpoint(staging,root/f'checkpoint-{step:05d}',JointCoordinator(),
                                         {'job_index':step},keep_every=50)
            self.assertEqual({p.name for p in root.glob('checkpoint-*')},
                             {'checkpoint-00050','checkpoint-00100','checkpoint-00105'})

    def test_dispatcher_capacity_and_failure_release(self):
        dispatcher=EvaluationDispatcher(dict(evaluations=[{'address':'node','port':1234}], token='test',
                                             services={'max_inflight':1}))
        entered=threading.Event();release=threading.Event()
        class Peer:
            def call(self,*args,**kwargs):
                entered.set();release.wait(2);raise ValueError('bad evidence')
        dispatcher.peers=[Peer()]
        request={'plan':{'segments':[{'num_chunks':6}]}}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(dispatcher,'evaluate',request)
            self.assertTrue(entered.wait(1));self.assertEqual(dispatcher('health',{})['active'],[1])
            second=pool.submit(dispatcher,'evaluate',request);release.set()
            for f in (first,second):
                with self.assertRaises(ValueError):f.result(timeout=3)
        self.assertEqual(dispatcher('health',{})['active'],[0])
        self.assertEqual(dispatcher('health',{})['estimated_work'],[0.])



if __name__=='__main__':unittest.main()
