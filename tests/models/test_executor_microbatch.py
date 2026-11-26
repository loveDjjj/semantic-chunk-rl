from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import torch

from scrl.models.executor import LongLiveBackend
from scrl.models.segments import SegmentExecutor
from scrl.plan import FinitePlan, Segment


class Pipe:
    num_frame_per_block = 3
    denoising_step_list = [4., 3., 2., 1.]
    frame_seq_length = 1
    args = SimpleNamespace(context_noise=0)

    def __init__(self):
        self.generator = SimpleNamespace(model=torch.nn.Identity(), seq_len=0)
        self.scheduler = SimpleNamespace(add_noise=lambda x, n, t: x + n * t[:, None, None, None] / 4)

    def _initialize_kv_cache(self, b, dtype, device, kv_cache_size_override):
        self.kv_cache1 = [{k: torch.zeros((b, kv_cache_size_override, 1, 1), device=device, dtype=dtype)
                           for k in ('k','v')}]

    def _initialize_crossattn_cache(self, b, **kwargs):
        self.crossattn_cache = [torch.zeros((b,1))]

    def _set_all_modules_max_attention_size(self, total):
        pass


class Backend(LongLiveBackend):
    dtype = torch.float32
    latent_shape = (1,1,1)
    private_gpu_history = True

    def __init__(self):
        super().__init__(Pipe(), SimpleNamespace(), 'cpu')
        self.batches = []

    def autocast(self):
        return nullcontext()

    def _recache(self, history, condition):
        self.pipe.kv_cache1[0]['k'][:, :history.shape[1]] += history.flatten(2).unsqueeze(-1)

    def forward(self, latent, condition, timestep, start, **kw):
        self.batches.append(len(latent))
        k = self.pipe.kv_cache1[0]['k']
        h = k[:,:start].sum(1).reshape(-1,1,1,1,1)
        y = latent * .25 + condition['prompt_embeds'].reshape(-1,1,1,1,1) + h * .075
        k[:,start:start+latent.shape[1]] = y.flatten(2).unsqueeze(-1)
        return y


class TwoVideoTests(unittest.TestCase):
    def test_different_seeds_conditions_and_private_history_match_serial(self):
        b = Backend()
        plans = [FinitePlan((Segment('shared word',1),Segment(word,2))) for word in ('x','y')]
        maps = [{k:{'prompt_embeds':torch.tensor([[v]])} for k,v in row}
                for row in [(('shared word',1.),('x',3.)),(('shared word',2.),('y',-4.))]]
        seeds = [731,134]
        expected = []
        for p,s,m in zip(plans,seeds,maps):
            b.conditions = m
            expected.append(SegmentExecutor(b).rollout(p,s,capture=False,retain_final_cache=False).final.latents)
        b.batches.clear()
        got = b.rollout_eval_pair(plans,seeds,maps)
        self.assertEqual(b.batches, [2]*(2*5))
        for x,y in zip(expected,got):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertIsNone(b.pipe.kv_cache1)
        self.assertIsNone(b.pipe.crossattn_cache)

    def test_reject_different_schedule_and_release_cache_on_error(self):
        b = Backend()
        p = FinitePlan((Segment('a',1),Segment('z',1)))
        q = FinitePlan((Segment('a',2),))
        with self.assertRaisesRegex(ValueError,'identical'):
            b.rollout_eval_pair([p,q],[1,2],[{},{}])
        with self.assertRaises(KeyError):
            m = {'a': {'prompt_embeds': torch.tensor([[1.]])}}
            b.rollout_eval_pair([p,p],[1,2],[m,m])
        self.assertIsNone(b.pipe.kv_cache1)
        self.assertIsNone(b.pipe.crossattn_cache)
