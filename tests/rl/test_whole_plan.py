import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from scrl.models.planner import FinitePlanGrammar, PlannerSample
from scrl.models.verl_policy import whole_batch
from scrl.plan import FinitePlan, PlanLimits, PolicyStamp, Segment, digest
from scrl.rl.objectives import ScoredRollout, WHOLE_CREDIT, planner_advantages
from scrl.runtime.worker import PlanningWorker


class WholePlanTests(unittest.TestCase):
    def test_execution_sampling_and_fixed_layout_are_restore_identity(self):
        from scrl.rl.identity import IDENTITY_KEYS, training_fingerprint
        cfg = dict.fromkeys(IDENTITY_KEYS,0)
        cfg.update(training={'planner_credit':WHOLE_CREDIT},sampling={'max_new_tokens':512},
                   fixed_sampling={'plans':4,'seeds':4})
        identity = training_fingerprint(cfg)
        self.assertNotEqual(identity,training_fingerprint(dict(cfg,sampling={'max_new_tokens':256})))
        self.assertNotEqual(identity,training_fingerprint(dict(cfg,fixed_sampling={'plans':4,'seeds':2})))

    def test_independent_plans_validate_paired_repeats_not_common_focal_prefix(self):
        plans = [FinitePlan((Segment(str(i),1), Segment("last",2))) for i in (0,0,2,3)]
        stamp = PolicyStamp()
        rows = [ScoredRollout('g',f'c{i}',seed,'prompt',str(i),stamp,plan,i % 2,i+seed,.7)
                for i,plan in enumerate(plans) for seed in range(4)]
        args = dict(expected_candidates=[f'c{i}' for i in range(4)],expected_seeds=range(4))
        def score(values):return planner_advantages(values,stamp,credit=WHOLE_CREDIT,**args)
        scores = score(rows)
        self.assertEqual(scores['c0'],scores['c1'])
        self.assertLess(scores['c0'],scores['c3'])
        self.assertAlmostEqual(sum(scores.values()),0.)
        with self.assertRaises(ValueError):planner_advantages(rows,stamp,**args)
        with self.assertRaises(ValueError):score(rows[:-1])
        with self.assertRaises(ValueError):score([replace(rows[0],group_id='other'),*rows[1:]])
        with self.assertRaises(ValueError):score([replace(rows[0],seed=10),*rows[1:]])
        same = [replace(r,plan=plans[0]) for r in rows]
        self.assertEqual(set(score(same).values()),{0.})

    def test_worker_samples_four_times_without_anchor_and_publishes_each_replay(self):
        worker = PlanningWorker.__new__(PlanningWorker)
        worker.snapshot='same'
        worker.cfg={'training':{'planner_credit':WHOLE_CREDIT}, 'sampling':{'max_new_tokens':100}}
        sample=SimpleNamespace(plan=FinitePlan((Segment('a',1),Segment('b',2))))
        worker.policy=Mock(version=0)
        worker.policy.sample.return_value=sample
        with TemporaryDirectory() as tmp:
            req=dict(prompt='p',seed=7,candidates=4,version=0,snapshot='same',
                     stream_dir=tmp,output=str(Path(tmp)/'sample.pt'))
            out=worker('plan',req)
            for i,(args,kw) in enumerate(worker.policy.sample.call_args_list):
                self.assertEqual(args,('p',7+i))
                self.assertIsNone(kw['text_prefix_ids'])
                self.assertTrue(kw['whole_plan'])
                self.assertNotIn('anchor_only',kw)
                item=json.loads((Path(tmp)/f'{i}.json').read_text())
                self.assertEqual(item,dict(plan=sample.plan.to_dict(),**out['replay_by_candidate'][i]))
                self.assertEqual(item['target_probability'],.5)
                self.assertEqual(item['prefix_segments'],[])
            self.assertEqual(worker.policy.sample.call_count,4)
            self.assertEqual(len(set(r['target'] for r in out['replay_by_candidate'])),1)
            with self.assertRaises(ValueError):worker('plan',dict(req,random_focal=True))

    def test_whole_behavior_mask_free_stop_continue_both_fields_and_equal_plan_weight(self):
        class Tokenizer:
            eos_token_id=127
            all_special_ids=[127]
            def get_vocab(self):return {chr(i):i for i in range(128)}
            def encode(self,text,**kwargs):return list(map(ord,text))
            def batch_decode(self,rows,**kwargs):return [''.join(map(chr,row)) for row in rows]
        grammar = FinitePlanGrammar(Tokenizer(),0,PlanLimits(max_segments=2))
        samples=[]
        for count in (1,2):
            plan=FinitePlan(tuple(Segment('ab',2) for _ in range(count)))
            wire=json.dumps(plan.to_output(),separators=(',',':'))
            ids=list(map(ord,wire))+[127]
            state=grammar.initial()
            grammar.constraints={}
            for i,tok in enumerate(ids):
                grammar.constraints[i]=grammar.allowed(state)
                self.assertIn(tok,grammar.constraints[i])
                if tok!=127:self.assertTrue(grammar.consume(state,chr(tok)))
            whole=[]
            decisions=grammar.decision_spans(ids,count,whole)
            self.assertEqual((len(wire)-1 in whole),count==1)
            if count==2:self.assertIn(wire.index('},{')+1,whole)
            for d in decisions.values():
                self.assertIn(d.caption[0],whole)
                self.assertIn(d.duration[0],whole)
            self.assertNotIn(0,whole)
            self.assertNotIn(len(wire),whole)
            response=torch.tensor([ids])
            samples.append(PlannerSample(plan,torch.tensor([[0]]),response,torch.zeros(len(ids)),
                torch.zeros(len(ids)),grammar.constraints,decisions,0,whole_positions=tuple(whole),
                whole_prompt_sha256=digest('p')))
        batch=whole_batch(samples,[-1.,1.],pad_token_id=0,device='cpu')
        torch.testing.assert_close(batch['response_mask'].sum(1),torch.ones(2))
        self.assertEqual(batch['position_ids'][:,0].tolist(),[0,1])
