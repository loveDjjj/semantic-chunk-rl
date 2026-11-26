"""Fast adapter switches preserve frozen parameters, values and checkpointed gradients."""

import copy
import os
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig,get_peft_model
from scrl.models.executor import LongLiveBackend


class AdapterActivationTests(unittest.TestCase):
    def backend(self):
        device=os.environ.get('SCRL_TEST_DEVICE','cpu')
        model=get_peft_model(nn.Sequential(nn.Linear(8,8),nn.SiLU(),nn.Linear(8,4)),
            LoraConfig(r=2,lora_alpha=4,target_modules=['0','2'],lora_dropout=0.,bias='none'))
        model.add_adapter('reference',LoraConfig(r=2,lora_alpha=4,target_modules=['0','2'],lora_dropout=0.,bias='none'))
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_B' in name:p.normal_()
        pipe=nn.Module();pipe.generator=nn.Module();pipe.generator.model=model
        pipe.text_encoder=nn.Linear(8,8);pipe.vae=nn.Linear(8,8);pipe.to(device)
        backend=LongLiveBackend.__new__(LongLiveBackend);backend.pipe=pipe
        return backend

    def test_outputs_gradients_and_freezing_match_general_switch(self):
        torch.manual_seed(7);fast=self.backend();old=copy.deepcopy(fast)
        def original(backend,adapter):
            model=backend.pipe.generator.model;model.set_adapter(adapter);backend.pipe.requires_grad_(False)
            for name,p in model.named_parameters():p.requires_grad_(adapter=='default' and 'lora_' in name and '.default.' in name)
        x=torch.randn(3,8,device=next(fast.pipe.parameters()).device)
        for adapter in ['default','default','reference','default','reference','default','default']:
            fast.activate(adapter);original(old,adapter)
            if adapter=='reference':
                with torch.no_grad():a=fast.pipe.generator.model(x);b=old.pipe.generator.model(x)
            else:
                a=checkpoint(fast.pipe.generator.model,x,use_reentrant=False)
                b=checkpoint(old.pipe.generator.model,x,use_reentrant=False)
                a.square().mean().backward();b.square().mean().backward()
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            for (name,p),(_,q) in zip(fast.pipe.named_parameters(),old.pipe.named_parameters()):
                self.assertEqual(p.requires_grad,q.requires_grad,name)
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        with patch.object(fast.pipe.generator.model,'set_adapter',wraps=fast.pipe.generator.model.set_adapter) as switch:
            fast.activate('default');fast.activate('default');self.assertEqual(switch.call_count,0)

    def test_added_adapter_invalidates_cached_parameter_list(self):
        b=self.backend();b.activate('default');m=b.pipe.generator.model
        m.add_adapter('extra',LoraConfig(r=2,lora_alpha=4,target_modules=['0','2'],lora_dropout=0.,bias='none'))
        b.activate('reference')
        self.assertFalse(any(p.requires_grad for p in b.pipe.parameters()))
        b.activate('default')
        self.assertTrue(all(not p.requires_grad for n,p in m.named_parameters() if '.extra.' in n))


if __name__=='__main__':unittest.main()
