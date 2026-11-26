"""CUDA cache transfers must complete before hashing and preserve restore ordering."""

import os
from types import SimpleNamespace
import unittest
import torch
from scrl.models.executor import LongLiveBackend
from scrl.models.segments import clone_tree,tree_fingerprint


@unittest.skipUnless(os.environ.get('SCRL_TEST_DEVICE','').startswith('cuda'), 'Explicit CUDA test device required')
class CacheTransferTests(unittest.TestCase):
    def test_copy_completion_alias_isolation_and_partial_restore(self):
        device=torch.device(os.environ['SCRL_TEST_DEVICE'])
        torch.cuda.set_device(device)
        backend=LongLiveBackend.__new__(LongLiveBackend);backend.device=device
        layers=[{'k':torch.randn(1,512,12,128,device=device,dtype=torch.bfloat16),
                 'v':torch.randn(1,512,12,128,device=device,dtype=torch.bfloat16),
                 'global_end_index':torch.tensor([256],device=device),
                 'local_end_index':torch.tensor([256],device=device)} for _ in range(4)]
        backend.pipe=SimpleNamespace(frame_seq_length=128,kv_cache1=layers)
        reference=[{k:clone_tree(v[:,:256] if k in ('k','v') else v) for k,v in layer.items()} for layer in layers]
        # A non-default stream catches accidental synchronization of another stream.
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            actual=backend._snapshot(2)
            self.assertFalse(any(v.is_pinned() for layer in actual for v in layer.values()))
            self.assertEqual(tree_fingerprint(actual),tree_fingerprint(reference))
            for layer in layers:
                for value in layer.values():value.zero_()
            backend._restore(actual,2)
            for layer,saved in zip(layers,reference):
                for key in ('k','v'):
                    torch.testing.assert_close(layer[key][:,:256].cpu(),saved[key],atol=0,rtol=0)
                    self.assertEqual(torch.count_nonzero(layer[key][:,256:]).item(),0)
                for key in ('global_end_index','local_end_index'):
                    torch.testing.assert_close(layer[key].cpu(),saved[key],atol=0,rtol=0)
            for layer in layers:
                for value in layer.values():value.add_(1)
            stream.synchronize()
            self.assertEqual(tree_fingerprint(actual),tree_fingerprint(reference))


if __name__=='__main__':unittest.main()
