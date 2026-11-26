"""Joint grouping, asynchronous collection, and transaction failure regression checks."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from types import SimpleNamespace
import threading
import unittest

from scrl.plan import FinitePlan, PolicyStamp, Segment
from scrl.rl.checkpoint import JointCoordinator
from scrl.rl.objectives import ScoredRollout, joint_advantages
from scrl.rl.rollout import GroupCollector


class JointTests(unittest.TestCase):
    def rows(self):
        stamp = PolicyStamp()
        plans = {"a": FinitePlan((Segment("one", 1),)), "b": FinitePlan((Segment("two", 4),))}
        # Between-plan offsets must never leak into within-plan Executor advantages.
        rows = [ScoredRollout("g", candidate, seed, "prompt", "prefix", stamp, plans[candidate],
                             0, value, value)
                for candidate, values in [("a", [.8, .6]), ("b", [.2, .4])]
                for seed, value in zip([11, 12], values)]
        return stamp, plans, rows

    def test_same_observations_have_distinct_credit_groups(self):
        stamp, _, rows = self.rows()
        advantages = joint_advantages(rows, stamp, expected_candidates=["a", "b"], expected_seeds=[11, 12])
        self.assertAlmostEqual(advantages["planner"]["a"], 1)
        self.assertAlmostEqual(advantages["executor"]["a"][11], 1)
        self.assertAlmostEqual(advantages["executor"]["b"][11], -1)
        # The lower-quality plan still has its own positive execution branch.
        self.assertAlmostEqual(advantages["executor"]["b"][12], 1)
        tied = [replace(r, executor_return=.5) for r in rows]
        advantages = joint_advantages(tied, stamp, expected_candidates=["a", "b"], expected_seeds=[11, 12])
        self.assertEqual(advantages["executor"], {"a": {11: 0., 12: 0.}, "b": {11: 0., 12: 0.}})
        with self.assertRaises(ValueError):
            joint_advantages(rows[:-1], stamp, expected_candidates=["a", "b"], expected_seeds=[11, 12])

    def test_joint_partial_failure_cannot_publish_or_checkpoint(self):
        coordinator = JointCoordinator()
        stamp = coordinator.begin("g", ["a"])
        coordinator.complete("a", stamp)
        changed = []
        def update(*_):
            changed.append("planner")
            raise RuntimeError("executor optimizer failed after planner update")
        with self.assertRaises(RuntimeError):
            coordinator.commit(update)
        self.assertEqual(changed, ["planner"])
        self.assertEqual(coordinator.stamp, stamp)
        with self.assertRaises(RuntimeError):
            coordinator.state_dict()
        with self.assertRaises(RuntimeError):
            coordinator.begin("next", ["x"])

    def test_joint_independent_skips_and_stale_completions(self):
        c = JointCoordinator()
        stamp = c.begin("g", ["a"])
        with self.assertRaises(ValueError):
            c.complete("a", replace(stamp, executor=1))
        c.complete("a", stamp)
        c.commit(lambda *_: {"planner": {"updated": False}, "executor": {"updated": True}})
        self.assertEqual(c.stamp, replace(stamp, executor=1))
        self.assertEqual(JointCoordinator.from_state_dict(c.state_dict()).stamp, c.stamp)

    def test_generation_continues_while_score_is_waiting_and_all_tapes_are_captured(self):
        stamp, plans, scored = self.rows()
        generated, captured = [], []
        second_generated = threading.Event()
        class Executor:
            version = 0
            def rollout(self, plan, seed, **kwargs):
                generated.append((plan, seed))
                captured.append(kwargs["capture"])
                if len(generated) >= 2:
                    second_generated.set()
                return SimpleNamespace(plan=plan, seed=seed)
        with ThreadPoolExecutor(max_workers=2) as pool:
            class Scorer:
                def submit(self, prompt, result):
                    def score():
                        if not second_generated.wait(timeout=3):
                            raise AssertionError("Generation blocked on previous score")
                        item = next(r for r in scored if r.plan == result.plan and r.seed == result.seed)
                        return {"planner": item.planner_return, "executor": item.executor_return}
                    return pool.submit(score)
            composer = SimpleNamespace(protocol=stamp.reward, compose=lambda x: x)
            collector = GroupCollector(Executor(), Scorer(), composer, stamp)
            rows, artifacts, advantages = collector.joint("prompt", "g", plans, [11, 12],
                                                          SimpleNamespace(segments=(), prefix_id="prefix"), 0)
        self.assertEqual(len(generated), 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(captured))
        self.assertEqual(set(artifacts), {(c, s) for c in plans for s in [11, 12]})
        self.assertEqual(set(advantages), {"planner", "executor"})

    def test_failed_reward_aborts_the_joint_group(self):
        from concurrent.futures import Future
        stamp, plans, _ = self.rows()
        class Executor:
            version = 0
            def rollout(self, plan, seed, **kwargs):
                return SimpleNamespace(plan=plan, seed=seed)
        class Scorer:
            def submit(self, prompt, result):
                future = Future()
                future.set_exception(RuntimeError("scorer disconnected"))
                return future
        composer = SimpleNamespace(protocol=stamp.reward, compose=lambda x: x)
        collector = GroupCollector(Executor(), Scorer(), composer, stamp)
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            collector.joint("prompt", "g", plans, [11, 12],
                            SimpleNamespace(segments=(), prefix_id="prefix"), 0)

    def test_rgb_scopes_reject_extent_mismatch(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from scrl.models.scoring.models import read_rgb
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.npy"
            pixels = np.arange(9*4*8*3, dtype=np.uint8).reshape(9, 4, 8, 3)
            np.save(path, pixels, allow_pickle=False)
            request = {"rgb": str(path), "start": 3, "end": 7, "frames": 4}
            np.testing.assert_array_equal(read_rgb(request), pixels[3:7])
            with self.assertRaises(ValueError):
                read_rgb(dict(request, frames=5))
            with self.assertRaises(ValueError):
                read_rgb(dict(request, end=10, frames=7))

    def test_service_queue_backpressure_releases_slot_after_failure(self):
        from scrl.models.scoring.service import VideoRewardService
        service = object.__new__(VideoRewardService)
        service.closed, service.failure = False, None
        service.slots = threading.BoundedSemaphore(1)
        entered, release = threading.Event(), threading.Event()
        def evaluate(*_):
            entered.set()
            if not release.wait(timeout=3):
                raise AssertionError("Test failed to release worker")
            raise RuntimeError("decode failed")
        service._evaluate = evaluate
        rollout = SimpleNamespace(plan=None, final=SimpleNamespace(latents=None, prefix_id="x"), target_segment=0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            service.pool = pool
            first = service.submit("prompt", rollout)
            self.assertTrue(entered.wait(timeout=3))
            self.assertFalse(service.slots.acquire(blocking=False))
            release.set()
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                first.result()
        with self.assertRaisesRegex(RuntimeError, "previous evaluation failed"):
            service.submit("prompt", rollout)
        self.assertTrue(service.slots.acquire(blocking=False))

    def test_final_cache_elision_preserves_latents_and_replay_prefix(self):
        import torch
        from scrl.models.segments import SegmentExecutor
        class Backend:
            backend_id='test';dtype=torch.float32;latent_shape=(1,2,2)
            def sample(self,prefix,segment,seed,capture,*,retain_cache=True):
                n=prefix.latents.shape[1]+segment.latent_frames
                latent=torch.full((1,segment.latent_frames,1,2,2),float(n))
                return latent, {'history':torch.tensor([n])} if retain_cache else None, {'start':prefix.prefix_id} if capture else None
        executor=SegmentExecutor(Backend())
        plan=FinitePlan((Segment('first',1),Segment('second',2)))
        a=executor.rollout(plan,1,target_segment=1)
        b=executor.rollout(plan,1,target_segment=1,retain_final_cache=False)
        torch.testing.assert_close(a.final.latents,b.final.latents,atol=0,rtol=0)
        self.assertEqual(a.target_prefix.prefix_id,b.target_prefix.prefix_id)
        self.assertEqual(a.tape,b.tape)
        self.assertIsNone(b.final.cache)
        self.assertIsNotNone(b.target_prefix.cache)
        b.final.verify()

    def test_cached_support_log_probs_and_gradients_match_dense_reference(self):
        import torch
        from scrl.models.planner import token_log_probs
        device=os.environ.get("SCRL_TEST_DEVICE", "cpu")
        logits=torch.randn(1,5,17,device=device,requires_grad=True)
        model=lambda **_:SimpleNamespace(logits=logits)
        context=torch.tensor([[1,2]],device=device)
        response=torch.tensor([[4,2,6]],device=device)
        support=[2,4,6,8]
        constraints={0:support,1:[1,2,3],2:support}
        actual=token_log_probs(model,context,response,constraints)
        mask=torch.zeros((1,3,17),dtype=torch.bool,device=device)
        for position,allowed in constraints.items():mask[:,position,allowed]=True
        expected=logits[:,1:-1].float().masked_fill(~mask,-torch.inf).log_softmax(-1).gather(-1,response[:,:,None])[0,:,0]
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        ga=torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0]
        ge=torch.autograd.grad(expected.sum(),logits)[0]
        torch.testing.assert_close(ga,ge,atol=0,rtol=0)

    def test_lora_autocast_without_activation_upcast_preserves_values_and_gradients(self):
        import torch
        device = os.environ.get("SCRL_TEST_DEVICE", "cpu")
        if not device.startswith("cuda:"):
            self.skipTest("Explicit CUDA device required for mixed-precision adapter comparison")
        from peft import LoraConfig, get_peft_model
        from peft.helpers import disable_input_dtype_casting
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.q_proj=torch.nn.Linear(64,128,bias=False);self.down_proj=torch.nn.Linear(128,64,bias=False)
            def forward(self,x):return self.down_proj(torch.nn.functional.gelu(self.q_proj(x)))
        torch.manual_seed(4)
        model=get_peft_model(Net().to(device=device,dtype=torch.bfloat16),
                            LoraConfig(r=8,target_modules=['q_proj','down_proj'],lora_dropout=0.))
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.data=p.data.float();p.data.normal_(0,.01)
        x=torch.randn(2,31,64,device=device,dtype=torch.bfloat16)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            before=model(x);before.float().square().mean().backward()
        gradients={name:p.grad.clone() for name,p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16),disable_input_dtype_casting(model):
            after=model(x);after.float().square().mean().backward()
        torch.testing.assert_close(after,before,atol=0,rtol=0)
        for name,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,gradients[name],atol=0,rtol=0)

    def test_split_checkpoint_resume_and_latest_retention_preserve_recovery(self):
        import json,tempfile
        from pathlib import Path
        import torch
        from scrl.rl.checkpoint import publish_split_checkpoint,load_checkpoint,retain_latest
        c=JointCoordinator()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for index in range(2):
                stamp=c.begin(str(index),['x']);c.complete('x',stamp)
                c.commit(lambda *_:{'planner':{'updated':True},'executor':{'updated':True}})
                stage=root/f'.staging-{index}';stage.mkdir()
                torch.save({'executor':{'weight':torch.tensor([index])},'executor_optimizer':{}},stage/'executor.pt')
                path=publish_split_checkpoint(stage,root/f'checkpoint-{index:05d}',c,
                    {'planner':{'weight':torch.tensor([index+1])},'planner_optimizer':{},'cursor':index+1})
                restored,payload=load_checkpoint(path)
                self.assertEqual(restored.stamp,c.stamp)
                self.assertEqual(payload['cursor'],index+1)
            self.assertEqual((root/'latest').resolve(),path)
            self.assertFalse((root/'checkpoint-00000').exists())
            bad=root/'checkpoint-bad';bad.mkdir()
            (bad/'policies.pt').write_bytes(b'broken')
            (bad/'state.json').write_text(json.dumps({'policies_sha256':'invalid'}))
            with self.assertRaises(ValueError):retain_latest(root,bad)
            self.assertTrue(path.exists())
            self.assertEqual((root/'latest').resolve(),path)

    def test_prefix_content_hash_unchanged_and_cache_mutation_detected(self):
        import hashlib,torch
        from scrl.models.segments import PrefixState,tensor_hash
        value=torch.arange(8,dtype=torch.float32).reshape(1,2,1,2,2)
        header=f'{value.dtype}:{tuple(value.shape)}:'.encode()
        expected=hashlib.sha256(header+value.view(torch.uint8).numpy().tobytes()).hexdigest()
        self.assertEqual(tensor_hash(value),expected)
        empty=torch.empty((1,0,1,2,2))
        prefix=PrefixState((),0,'test',empty,{'layer':torch.ones(2)})
        prefix.verify(full=True)
        prefix.cache['layer'][0]=2
        with self.assertRaises(RuntimeError):prefix.verify()
        prefix=PrefixState((),0,'test',empty,{'layer':torch.ones(2)})
        prefix.cache['layer']=torch.ones(2)
        with self.assertRaises(RuntimeError):prefix.verify()

    def test_grammar_gpu_support_cache_across_complete_and_prefixed_generations(self):
        import torch,json
        from scrl.models.planner import FinitePlanGrammar
        chars=sorted(set('[{"step_index":0,"prompt":"a","num_chunks":1},2346] \n'))
        pieces=['<eos>']+chars
        class Tokenizer:
            eos_token_id=0;all_special_ids=[0]
            def get_vocab(self):return {p:i for i,p in enumerate(pieces)}
            def batch_decode(self,rows,**kwargs):return [self.decode(row,**kwargs) for row in rows]
            def decode(self,row,**kwargs):return ''.join(pieces[int(i)] for i in row if int(i)!=0)
            def encode(self,text,**kwargs):return [pieces.index(c) for c in text]
        tokenizer=Tokenizer();grammar=FinitePlanGrammar(tokenizer,0)
        scores=torch.zeros(1,len(pieces))
        for count in [1,2,3,4,5,3,1,5,2,4]:
            grammar.constraints,grammar.fields={},{}
            text=json.dumps(FinitePlan(tuple(Segment('a',1) for _ in range(count))).to_output())
            ids=[]
            for token in tokenizer.encode(text)+[0]:
                state=grammar.state(ids)
                expected=set(grammar.allowed(state))
                masked=grammar(torch.tensor([ids],dtype=torch.long),scores)
                actual=set(torch.isfinite(masked[0]).nonzero().flatten().tolist())
                self.assertEqual(actual,expected)
                self.assertIn(token,actual)
                self.assertEqual(0 in actual,state['mode']=='done')
                ids.append(token)
