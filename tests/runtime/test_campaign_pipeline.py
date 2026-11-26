"""Scheduling, bounded asynchronous evaluation, and checkpoint retention contracts."""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import json
import unittest
import gc
import weakref

import torch

from scrl.plan import FinitePlan, PolicyStamp, Segment
from scrl.models.segments import PrefixState, parallel_tree_fingerprint, tree_fingerprint
from scrl.runtime.worker import ExecutionWorker
from scrl.rl.checkpoint import JointCoordinator, publish_split_checkpoint, load_checkpoint, retain_latest
from scrl.runtime.launch import WorkQueue, compact_layout, estimate_work, advance_dataset, prompt_seed
from scrl.rl.objectives import ScoredRollout, joint_advantages


class CampaignPipelineTests(unittest.TestCase):
    def test_hashing_releases_tensor_references_without_cyclic_gc(self):
        enabled=gc.isenabled()
        gc.collect();gc.disable()
        try:
            for elements in (8, 16*2**20):
                value=torch.zeros(elements)
                reference=weakref.ref(value)
                nested={'layer':[value,{'alias':value}]}
                self.assertEqual(parallel_tree_fingerprint(nested),tree_fingerprint(nested))
                del nested,value
                self.assertIsNone(reference(),'Hash traversal retained a cache through a reference cycle')
        finally:
            if enabled:gc.enable()
            gc.collect()

    def test_repeated_dataset_resumes_at_boundary_with_new_reproducible_seeds(self):
        self.assertEqual(advance_dataset(4992,0,5000,True),(4992,0))
        self.assertEqual(advance_dataset(5000,0,5000,True),(0,1))
        self.assertEqual(advance_dataset(5000,0,5000,False),(5000,0))
        self.assertEqual(advance_dataset(5000,1,5000,True),(0,2))
        old={prompt_seed(2027,0,i,5000) for i in range(5000)}
        new={prompt_seed(2027,1,i,5000) for i in range(5000)}
        self.assertTrue(old.isdisjoint(new))
        self.assertEqual(prompt_seed(2027,1,3,5000),prompt_seed(2027,1,3,5000))

    def request(self, name, lengths=(1,), target=0):
        return dict(job_id=name,group_id=name,candidate='c0',prompt='prompt',
            plan={'segments':[{'caption':f'caption {i}','num_chunks':n} for i,n in enumerate(lengths)]},
            target=target,target_probability=1.,prefix_segments=[],prefix_seed=7,
            stamp=vars(PolicyStamp()),seeds=[11,12,13,14],total_videos=16,
            prefetch_text=True,overlap_backward=True,attempt='test')

    def test_queue_prioritizes_total_work_and_keeps_all_jobs(self):
        q=WorkQueue(warmup=10)
        short=self.request('short')
        long=self.request('long',(6,6,6),2)
        q.put(short);q.put(long);q.close()
        self.assertGreater(estimate_work(long),estimate_work(short))
        self.assertIs(q.get(),long);self.assertIs(q.get(),short);self.assertIsNone(q.get())

    def test_late_worker_can_take_the_remaining_long_seed_pair_without_waiters(self):
        req=self.request('long',(6,6,6),1)
        q=WorkQueue(split_chunks=18,queue_seed_pairs=True)
        q.put(req)
        first=q.get()
        q.close()
        second=q.get()
        self.assertIsNone(q.get())
        self.assertEqual([x['job_id'] for x in (first,second)],['long/s0','long/s1'])
        self.assertEqual(first['seeds']+second['seeds'],req['seeds'])
        for piece in (first,second):
            self.assertEqual(piece['parent_job'],req['job_id'])
            self.assertEqual({k:piece[k] for k in req if k not in ('job_id','seeds')},
                             {k:req[k] for k in req if k not in ('job_id','seeds')})
        for request,enabled in [(req,False),(self.request('short',(6,6,4)),True)]:
            q=WorkQueue(split_chunks=18,queue_seed_pairs=enabled)
            q.put(request);q.close();self.assertIs(q.get(),request);self.assertIsNone(q.get())

    def test_queue_failure_wakes_consumers_and_cannot_be_cleared(self):
        q=WorkQueue()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(q.get)
            q.close(ValueError('planning failed'));q.close()
            with self.assertRaisesRegex(RuntimeError,'Another task failed'):future.result(timeout=2)

    def test_compact_layout_uses_38_executor_cards_and_no_port_collisions(self):
        nodes=[dict(index=i,address=f'node-{i}',gpus=4) for i in range(11)]
        c=compact_layout(nodes)
        self.assertEqual(len(c['executors']),38)
        self.assertEqual(len(c['evaluations']),6)
        self.assertEqual(len(c['planners']),11)
        cards={(r['node'],r['device']) for r in c['executors']+c['evaluations']}
        self.assertEqual(len(cards),44)
        services={(r['node'],r['device']) for r in c['evaluations']}
        self.assertTrue(all((r['node'],r['device']) in services for r in c['planners']))
        roles=c['executors']+c['evaluations']+c['planners']
        self.assertEqual(len({(r['node'],r['port']) for r in roles}),len(roles))

    def test_owned_snapshot_has_identical_hash_and_mutation_detection(self):
        source=torch.randn(1,3,1,2,2)
        cache={'k':torch.randn(1,4,2),'counter':torch.tensor([3])}
        args=((Segment('caption',1),),0,'test',source,cache)
        copied=PrefixState(*args)
        owned=PrefixState(*args,_copy_inputs=False)
        self.assertEqual(copied.prefix_id,owned.prefix_id)
        self.assertEqual(owned.cache['k'].data_ptr(),cache['k'].data_ptr())
        self.assertNotEqual(copied.cache['k'].data_ptr(),cache['k'].data_ptr())
        cache['k'].add_(1)
        copied.verify(full=True)
        with self.assertRaises(RuntimeError):owned.verify()

    def fake_worker(self, folder, ready, seeds_per_group=4):
        worker=ExecutionWorker.__new__(ExecutionWorker)
        worker.frozen=False
        worker.rank=0;worker.pending={};worker.completed=set();worker.inflight={}
        worker.pipeline_depth=2;worker.transport=Path(folder);worker.evaluation_index=0
        worker.timing=dict(text=0.,prefix=0.,generation=0.,score_wait=0.,backward=0.,replay_max_error=0.)
        worker.backend=SimpleNamespace(version=0,encode_many=lambda captions:None,
                                        pipe=SimpleNamespace(generator=SimpleNamespace(seq_len=0)))
        generated=[]
        def rollout(plan,seed,**kwargs):
            generated.append(seed)
            worker.backend.pipe.generator.seq_len = 10 if len(generated)<=seeds_per_group else 20
            if len(generated)>=2*seeds_per_group:ready.set()
            return SimpleNamespace(final=SimpleNamespace(latents=torch.tensor([seed]),prefix_id=str(seed)))
        worker.executor=SimpleNamespace(build_prefix=lambda *args:SimpleNamespace(prefix_id='prefix'),rollout=rollout)
        def evaluate(op,**request):
            if not ready.wait(3):raise RuntimeError('Scoring blocked later generation')
            seed=int(torch.load(request['latent_path'],weights_only=True)[0])
            return dict(global_scores={'quality':{'seed':seed},'semantic':{}},
                        segment_scores={'quality':{},'semantic':{}},protocol=PolicyStamp().reward,boundary_quality=None)
        worker.evaluators=[SimpleNamespace(call=evaluate)]
        worker.score_pool=ThreadPoolExecutor(max_workers=8)
        worker.composer=SimpleNamespace(compose=lambda b:{'planner':float(b.global_scores.quality['seed']),
                                                          'executor':float(b.global_scores.quality['seed'])})
        accumulated=[]
        def accumulate(job):
            records,rollouts,advantages,weight=worker.pending.pop(job)
            accumulated.append((job,len(records),dict(advantages),weight,worker.backend.pipe.generator.seq_len))
            worker.completed.add(job)
        worker.accumulate=accumulate
        return worker,generated,accumulated

    def test_frozen_executor_scores_without_backward(self):
        ready=threading.Event();ready.set()
        with tempfile.TemporaryDirectory() as folder:
            worker,generated,backward=self.fake_worker(folder,ready)
            worker.frozen=True
            try:
                result=worker.generate(self.request("frozen"))
                self.assertEqual(result["physical_videos"],4)
                self.assertEqual(worker.completed,{"frozen"})
                self.assertFalse(worker.pending or backward)
            finally:
                worker.score_pool.shutdown(wait=True)

    def test_frozen_generator_can_finish_three_groups_before_any_score(self):
        from scrl.runtime.transport import EvaluationDispatcher
        broker = EvaluationDispatcher({'evaluations':[], 'frozen_score_queue_videos':12})
        gate = threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            worker,generated,_ = self.fake_worker(folder,threading.Event())
            worker.frozen = True
            worker.score_admission = SimpleNamespace(call=lambda op,**p:broker(op,p))
            def slow(op,**p):
                if not gate.wait(8):
                    raise RuntimeError('Generation still waits for a score')
                value=int(torch.load(p['latent_path'],weights_only=True)[0])
                return dict(global_scores={'quality':{'seed':value},'semantic':{}},
                            segment_scores={'quality':{},'semantic':{}},protocol=PolicyStamp().reward,
                            boundary_quality=None)
            worker.evaluators=[SimpleNamespace(call=slow)]
            try:
                for name in ('a','b','c'):
                    self.assertFalse(worker.enqueue(self.request(name))['completed'])
                self.assertEqual(len(generated),12)
                self.assertEqual(broker('health',{})['reserved_videos'],12)
                gate.set()
                self.assertEqual({j['job_id'] for j in worker.drain()['completed']},{'a','b','c'})
                self.assertEqual(broker('health',{})['reserved_videos'],0)
                self.assertFalse(list(Path(folder).iterdir()))
            finally:
                gate.set()
                worker.score_pool.shutdown(wait=True)

    def test_failed_score_frees_admission_and_backpressure_releases(self):
        from scrl.runtime.transport import EvaluationDispatcher
        broker = EvaluationDispatcher({'evaluations':[], 'frozen_score_queue_videos':1})
        admission = SimpleNamespace(call=lambda op,**p:broker(op,p))
        broker('reserve_videos',{'keys':['a']})
        with ThreadPoolExecutor(1) as pool:
            waiter=pool.submit(broker,'reserve_videos',{'keys':['b']})
            self.assertFalse(waiter.done())
            def fail(*args,**kwargs):
                raise RuntimeError('judge failed')
            with self.assertRaisesRegex(RuntimeError,'judge failed'):
                ExecutionWorker._score_file(SimpleNamespace(call=fail),{},admission,'a')
            self.assertEqual(waiter.result(timeout=2),{})
        self.assertEqual(broker('health',{})['peak_reserved'],1)
        broker('release_video',{'key':'b'})

    def test_next_group_generates_before_previous_rewards_and_cleanup_is_bounded(self):
        ready=threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            w,generated,accumulated=self.fake_worker(folder,ready)
            try:
                first=w.enqueue(self.request('first'))
                self.assertEqual(first['completed'],[])
                self.assertEqual(len(w.inflight),1)
                second=w.enqueue(self.request('second'))
                results=first['completed']+second['completed']+w.drain()['completed']
                self.assertEqual(generated,[11,12,13,14]*2)
                self.assertEqual({r['job_id'] for r in results},{'first','second'})
                self.assertEqual([a[1] for a in accumulated],[4,4])
                self.assertTrue(all(a[2][11]<0<a[2][14] for a in accumulated))
                self.assertTrue(all(a[3]==1/16 for a in accumulated))
                self.assertEqual({a[0]:a[4] for a in accumulated},{'first':10,'second':20})
                self.assertFalse(w.pending);self.assertFalse(w.inflight)
                self.assertEqual(list(Path(folder).iterdir()),[])
            finally:
                ready.set();w.score_pool.shutdown(wait=True)

    def test_streamed_duplicate_preserves_all_seeds_and_double_weight(self):
        ready=threading.Event();ready.set()
        with tempfile.TemporaryDirectory() as folder:
            w,generated,accumulated=self.fake_worker(folder,ready)
            try:
                request=self.request('first')
                seal=Path(folder)/'credits.json'
                request['alias_seal']=str(seal)
                w._start_job(request)
                self.assertEqual(generated,[11,12,13,14])
                self.assertFalse(accumulated)
                seal.write_text(json.dumps({'first':[{'job_id':'first','candidate':'c0'},
                                                      {'job_id':'late','candidate':'c1'}]}))
                result=w._finish_job('first')
                self.assertEqual(result['logical_job_ids'],['first','late'])
                self.assertEqual(len(result['rows']),8)
                self.assertEqual(accumulated[0][3],2/16)
            finally:
                w.score_pool.shutdown(wait=True)

    def test_three_seed_groups_overlap_with_exact_global_video_weight(self):
        ready=threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            w,generated,accumulated=self.fake_worker(folder,ready,seeds_per_group=3)
            try:
                results=[]
                for name in ['first','second']:
                    request=self.request(name)
                    request.update(seeds=[11,12,13],total_videos=32*4*3)
                    results.extend(w.enqueue(request)['completed'])
                results.extend(w.drain()['completed'])
                self.assertEqual(generated,[11,12,13]*2)
                self.assertEqual(len(results),2)
                for _,count,advantages,weight,_ in accumulated:
                    self.assertEqual(count,3)
                    self.assertEqual(set(advantages),{11,12,13})
                    self.assertAlmostEqual(sum(advantages.values()),0.)
                    self.assertAlmostEqual(weight,1/384)
                self.assertFalse(w.inflight)
                self.assertEqual(list(Path(folder).iterdir()),[])
            finally:
                ready.set();w.score_pool.shutdown(wait=True)

    def test_depth_limit_drains_oldest_before_accepting_more_gpu_work(self):
        w=ExecutionWorker.__new__(ExecutionWorker)
        w.pipeline_depth=2;w.inflight={name:{'futures':{0:Future()}} for name in ('a','b')}
        events=[]
        def finish(job):
            events.append(('finish',job));w.inflight.pop(job);return {'job_id':job}
        def start(request):
            events.append(('start',request['job_id']))
            w.inflight[request['job_id']]={'futures':{0:Future()}}
        w._finish_job=finish;w._start_job=start
        w._finish_next=lambda:finish(next(iter(w.inflight)))
        w._cache_budget_exceeded=lambda request:False
        result=w.enqueue(self.request('c'))
        self.assertEqual(events,[('finish','a'),('start','c'),('finish','b')])
        self.assertEqual(result['pending'],1)
        self.assertEqual([r['job_id'] for r in result['completed']],['a','b'])

    def test_long_prefix_drains_older_group_but_short_groups_can_overlap(self):
        w=ExecutionWorker.__new__(ExecutionWorker)
        w.pipeline_depth=2;w.pipeline_cache_limit=8
        w.backend=SimpleNamespace(prefix_cache_bytes=lambda frames:frames)
        small=self.request('small');small['prefix_segments']=[{'caption':'p','num_chunks':1}]
        big=self.request('big');big['prefix_segments']=[{'caption':'p','num_chunks':2}]
        pending=Future()
        w.inflight={'small':{'request':small,'futures':{0:pending}}}
        self.assertFalse(w._cache_budget_exceeded(small))
        self.assertTrue(w._cache_budget_exceeded(big))
        events=[]
        def finish_next():
            events.append('drain');w.inflight.clear();return {'job_id':'small'}
        def start(request):
            events.append('generate');w.inflight['big']={'request':big,'futures':{0:Future()}}
        w._finish_next=finish_next;w._start_job=start
        result=w.enqueue(big)
        self.assertEqual(events,['drain','generate'])
        self.assertEqual(result['pending'],1)

    def test_complete_group_overtakes_slow_group_without_partial_credit(self):
        w=ExecutionWorker.__new__(ExecutionWorker)
        slow=[Future() for _ in range(4)];fast=[Future() for _ in range(4)]
        for f in slow[:3]+fast:f.set_result(1)
        w.inflight={'slow':{'futures':dict(enumerate(slow))},'fast':{'futures':dict(enumerate(fast))}}
        def finish(job,**kwargs):
            w.inflight.pop(job);return {'job_id':job}
        w._finish_job=finish
        self.assertEqual(w._drain_ready(),[{'job_id':'fast'}])
        self.assertEqual(list(w.inflight),['slow'])
        slow[-1].set_result(1)
        self.assertEqual(w.drain()['completed'],[{'job_id':'slow'}])

    def test_four_by_four_grouping_uses_all_scores_and_rejects_missing_seed(self):
        stamp=PolicyStamp();rows=[]
        candidates=[f'c{i}' for i in range(4)];seeds=[11,12,13,14]
        for candidate_index,candidate in enumerate(candidates):
            plan=FinitePlan((Segment(candidate,1),))
            for seed_index,seed in enumerate(seeds):
                value=100*candidate_index+seed_index
                rows.append(ScoredRollout('g',candidate,seed,'prompt','prefix',stamp,plan,0,value,value))
        adv=joint_advantages(rows,stamp,expected_candidates=candidates,expected_seeds=seeds)
        self.assertEqual(len(adv['planner']),4)
        self.assertLess(adv['planner']['c0'],adv['planner']['c3'])
        for candidate in candidates:
            values=adv['executor'][candidate]
            self.assertEqual(len(values),4);self.assertAlmostEqual(sum(values.values()),0.)
            self.assertEqual(values,adv['executor']['c0'])
        with self.assertRaises(ValueError):
            joint_advantages(rows[:-1],stamp,expected_candidates=candidates,expected_seeds=seeds)

    def test_scorer_failure_drains_other_readers_and_cleans_files(self):
        ready=threading.Event();ready.set()
        with tempfile.TemporaryDirectory() as folder:
            w,_,_=self.fake_worker(folder,ready)
            w._start_job(self.request('failed'))
            for future in w.inflight['failed']['futures'].values():future.result(timeout=3)
            failed=Future();failed.set_exception(ValueError('score failure'))
            w.inflight['failed']['futures'][11]=failed
            with self.assertRaisesRegex(ValueError,'score failure'):w._finish_job('failed')
            self.assertFalse(w.pending);self.assertFalse(w.completed);self.assertFalse(w.inflight)
            self.assertEqual(list(Path(folder).iterdir()),[])
            w.score_pool.shutdown(wait=True)

    def test_permanent_checkpoints_keep_earlier_milestones(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (50,100):
                staging=root/f'.staging-{step}';staging.mkdir()
                torch.save({'executor':torch.tensor([step])},staging/'executor.pt')
                publish_split_checkpoint(staging,root/f'checkpoint-{step:05d}',JointCoordinator(),
                    {'planner':torch.tensor([step]),'job_index':step},keep_latest_only=False)
            self.assertTrue((root/'checkpoint-00050').is_dir())
            self.assertEqual((root/'latest').resolve(),root/'checkpoint-00100')
            _,payload=load_checkpoint(root/'latest')
            self.assertEqual(payload['job_index'],100)

    def test_latest_save_and_fiftieth_round_milestones_survive_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (1,5,10,45,50,55,95,100,105):
                staging=root/f'.staging-{step}';staging.mkdir()
                torch.save({'executor':torch.tensor([step])},staging/'executor.pt')
                publish_split_checkpoint(staging,root/f'checkpoint-{step:05d}',JointCoordinator(),
                    {'planner':torch.tensor([step]),'job_index':step},keep_latest_only=True,keep_every=50)
                expected={f'checkpoint-{n:05d}' for n in (50,100) if n<=step}
                expected.add(f'checkpoint-{step:05d}')
                self.assertEqual({p.name for p in root.glob('checkpoint-*')},expected)
            self.assertEqual((root/'latest').resolve().name,'checkpoint-00105')
            for step in (50,100,105):
                _,payload=load_checkpoint(root/f'checkpoint-{step:05d}')
                self.assertEqual(payload['job_index'],step)
            (root/'checkpoint-00105/planner.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                retain_latest(root,root/'checkpoint-00105',keep_every=50)
            self.assertTrue((root/'checkpoint-00050').is_dir())
            self.assertTrue((root/'checkpoint-00100').is_dir())

    def test_every_second_save_retains_latest_and_twentieth_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (2,4,18,20,22,38,40,42):
                staging=root/f'.staging-{step}';staging.mkdir()
                torch.save({'executor':torch.tensor([step])},staging/'executor.pt')
                publish_split_checkpoint(staging,root/f'checkpoint-{step:05d}',JointCoordinator(),
                    {'planner':torch.tensor([step]),'job_index':step},keep_latest_only=True,keep_every=20)
            self.assertEqual({p.name for p in root.glob('checkpoint-*')},
                             {'checkpoint-00020','checkpoint-00040','checkpoint-00042'})
            self.assertEqual((root/'latest').resolve().name,'checkpoint-00042')
            _,payload=load_checkpoint(root/'latest')
            self.assertEqual(payload['job_index'],42)


if __name__=='__main__':unittest.main()
