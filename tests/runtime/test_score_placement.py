import threading
import unittest

from scrl.runtime.launch import assign_local_vae
from scrl.models.scoring.service import TimedWorker


class ScorePlacementTest(unittest.TestCase):
    def test_aux_only_borrows_distinct_frozen_gpus_on_same_host(self):
        cfg=dict(training={'mode':'planner-only'},executors=[dict(node=2,device='cuda:0'),dict(node=2,device='cuda:1')],
                 evaluations=[dict(node=2,device='cuda:2'),dict(node=2,device='cuda:3')])
        assign_local_vae(cfg)
        self.assertEqual([(x['device'],x['vae_device']) for x in cfg['evaluations']],
                         [('cuda:2','cuda:0'),('cuda:3','cuda:1')])
        cfg['evaluations'][1]['node']=1
        with self.assertRaises(ValueError):assign_local_vae(cfg)
        cfg['training']['mode']='joint'
        with self.assertRaises(ValueError):assign_local_vae(cfg)

    def test_trace_keeps_same_response_and_rgb_identity_across_parallel_readers(self):
        class Fake:
            def __init__(self):self.lock=threading.Lock()
            def score(self, request):
                with self.lock:return request['result']
        rows=[]; req=dict(rgb='/dev/shm/scrl-rgb-x/rgb.npy',result={'seconds':0.25})
        for role in ('vae','quality','judge'):
            stage=TimedWorker(Fake(),role,rows.append)
            self.assertIs(stage.score(req),req['result'])
        self.assertEqual([r['stage'] for r in rows],['vae','quality','judge'])
        self.assertTrue(all(r['video']=='scrl-rgb-x' and r['child_seconds']==.25 for r in rows))
        self.assertTrue(all(0<=r['lock_wait']<=r['seconds'] for r in rows[:2]))
        self.assertIsNone(rows[-1]['lock_wait'])
