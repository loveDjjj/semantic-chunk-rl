import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scrl.models.planner import SeededBatchGrammar
from scrl.evaluation._planner_lane import planner_batch_size


class TestSeededBatchGrammar(unittest.TestCase):
    def test_eval_defaults_to_sixteen_rows_per_model(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(planner_batch_size(),16)
        with patch.dict('os.environ', {'SCRL_EVAL_PLANNER_BATCH_SIZE':'1'}, clear=True):
            self.assertEqual(planner_batch_size(),1)

    class Grammar:
        def __init__(self):
            self.device_support={}
            self.tokenizer=SimpleNamespace(eos_token_id=5)
            self.choices=[[0,1,3],[2,4],[5]]

        def state(self, ids):
            return len(ids)%3

        def allowed(self, state):
            return self.choices[state]

    def sample(self, seeds):
        g=self.Grammar()
        proc=SeededBatchGrammar(g,2,seeds,'cpu')
        inputs=torch.tensor([[5,3]]*len(seeds))
        scores=torch.tensor([.5,.1,1.5,-.1,.3,1.]).expand(len(seeds),-1)
        for step in range(3):
            masked=proc(inputs,scores)
            self.assertTrue(torch.isfinite(masked).sum(dim=1).eq(1).all())
            inputs=torch.cat([inputs,masked.argmax(dim=-1,keepdim=True)],dim=1)
        return inputs[:,2:].tolist()

    def test_independent_seed_and_legal_multinomial_per_row(self):
        saved=torch.random.get_rng_state()
        seeds=[40,22,7,13,64,3,90,8]
        rows=self.sample(seeds)
        self.assertEqual(self.sample(list(reversed(seeds))),list(reversed(rows)))
        for seed,row in zip(seeds,rows):
            self.assertEqual(self.sample([seed])[0], row)
            rng=torch.Generator().manual_seed(seed)
            expected=[]
            for support in self.Grammar().choices:
                raw=torch.tensor([.5,.1,1.5,-.1,.3,1.])
                masked=torch.full_like(raw,-torch.inf)
                masked[support]=raw[support]
                expected.append(torch.multinomial(masked.softmax(-1),1,generator=rng).item())
            self.assertEqual(row,expected)
        self.assertTrue(torch.equal(saved,torch.random.get_rng_state()))

    def test_finished_rows_do_not_force_sibling_eos(self):
        proc=SeededBatchGrammar(self.Grammar(),1,[12,13],'cpu')
        ids=torch.tensor([[3,5],[3,1]])
        draws=proc(ids,torch.ones(2,6)).argmax(-1).tolist()
        self.assertEqual(draws[0],5)
        self.assertIn(draws[1],[2,4])
