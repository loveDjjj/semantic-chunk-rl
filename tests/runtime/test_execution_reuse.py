"""Reuse must preserve candidate credit, sample counts and globally normalized gradients."""

import copy
import tempfile
import threading
import unittest
import torch

from scrl.plan import FinitePlan,PolicyStamp,Segment
from scrl.rl.objectives import ScoredRollout,joint_advantages
from scrl.rl.rollout import coalesce_requests,execution_aliases,expand_scored_rows
from scrl.runtime.worker import ExecutionWorker
from . import test_campaign_pipeline as fixtures


class ExecutionReuseTests(unittest.TestCase):
    def requests(self):
        helper=fixtures.CampaignPipelineTests()
        requests=[]
        for i in range(4):
            request=helper.request(f'b10/p320/c{i}')
            request.update(group_id='p320',candidate=f'c{i}',total_videos=64*4*4)
            requests.append(request)
        return requests

    def test_only_identical_execution_inputs_share_work(self):
        original=self.requests();saved=copy.deepcopy(original)
        grouped=coalesce_requests(original)
        self.assertEqual(len(grouped),1)
        self.assertEqual(len(execution_aliases(grouped[0])),4)
        self.assertEqual(original,saved)
        variants={'seeds':[21,22,23,24],'prefix_seed':99,'target_probability':.5,
                  'target':1,'prompt':'different','group_id':'other','attempt':'later',
                  'stamp':dict(vars(PolicyStamp()),executor=1),'total_videos':512,
                  'plan':{'segments':[{'caption':'different','num_chunks':1}]},
                  'prefix_segments':[{'caption':'prefix','num_chunks':1}]}
        for key,value in variants.items():
            rows=self.requests()[:2];rows[1][key]=value
            with self.subTest(key=key):self.assertEqual(len(coalesce_requests(rows)),2)
        bad=self.requests()[:2];bad[1]['job_id']=bad[0]['job_id']
        with self.assertRaises(ValueError):coalesce_requests(bad)

    def test_reuse_preserves_planner_and_executor_advantages(self):
        requests=self.requests();requests[2]['plan']['segments'][0]['caption']='other'
        grouped=coalesce_requests(requests);stamp=PolicyStamp();expanded=[];baseline=[]
        for request in requests:
            plan=FinitePlan.from_dict(request['plan']);offset=2 if request['candidate']=='c2' else 0
            baseline.extend(ScoredRollout('p320',request['candidate'],seed,'prompt','prefix',stamp,plan,0,
                                         offset+seed/20,seed/20) for seed in request['seeds'])
        for request in grouped:
            physical=[r for r in baseline if r.candidate_id==request['candidate']]
            for value in expand_scored_rows(physical,execution_aliases(request)):
                value['plan']=FinitePlan.from_dict(value['plan']);value['stamp']=PolicyStamp(**value['stamp'])
                expanded.append(ScoredRollout(**value))
        kwargs=dict(expected_candidates=['c0','c1','c2','c3'],expected_seeds=[11,12,13,14])
        self.assertEqual(len(expanded),16)
        self.assertEqual(joint_advantages(expanded,stamp,**kwargs),joint_advantages(baseline,stamp,**kwargs))

    def test_real_accumulator_preserves_repeated_loss_gradient_and_saves_execution(self):
        helper=fixtures.CampaignPipelineTests();requests=self.requests()
        def run(reuse):
            ready=threading.Event();ready.set()
            with tempfile.TemporaryDirectory() as folder:
                w,generated,_=helper.fake_worker(folder,ready)
                parameter=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
                calls=[];w.cfg={'training':{'kl_beta':.01}}
                def backward(rollout,advantage,*,kl_beta,weight):
                    seed=float(rollout.final.latents.item());calls.append(seed)
                    loss=(advantage*(parameter*seed).square()+kl_beta*(parameter-.3).square())*weight
                    loss.backward()
                    return {'replay_max_error':0.}
                w.backend.backward=backward
                w.accumulate=ExecutionWorker.accumulate.__get__(w,ExecutionWorker)
                try:
                    outcomes=[w.generate(r) for r in (coalesce_requests(requests) if reuse else requests)]
                    return parameter.grad.clone(),len(generated),len(calls),outcomes
                finally:w.score_pool.shutdown(wait=True)
        original,generated,replayed,_=run(False)
        optimized,reused_generated,reused_replayed,outcomes=run(True)
        torch.testing.assert_close(original,optimized,atol=1e-12,rtol=1e-12)
        self.assertEqual((generated,replayed),(16,16))
        self.assertEqual((reused_generated,reused_replayed),(4,4))
        self.assertEqual(len(outcomes[0]['rows']),16)
        self.assertEqual(outcomes[0]['logical_videos'],16)
        self.assertEqual(outcomes[0]['physical_videos'],4)
        self.assertEqual(outcomes[0]['logical_job_ids'],[r['job_id'] for r in requests])


if __name__=='__main__':unittest.main()
