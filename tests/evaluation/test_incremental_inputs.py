from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scrl.evaluation.io import atomic_json, file_sha256
from scrl.evaluation.protocol import DIMENSIONS, make_tasks, settings
from scrl.evaluation.vbench import checked_records, main, receipt_binding
from scrl.plan import digest


class IncrementalInputs(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run=Path(tmp.name)
        dims=[d for d in DIMENSIONS if d!='scene']
        rows=[dict(index=0,prompt='trees',id='a',dimensions=dims,seed_count=25),
              dict(index=1,prompt='park',id='b',dimensions=['scene'],seed_count=5)]
        self.tasks=make_tasks(rows)
        metadata=[dict(prompt_en=r['prompt'],dimension=r['dimensions']) for r in rows]
        formula=settings()['formula']
        score=dict(tasks_sha256=digest(self.tasks),metadata_sha256=digest(metadata),formula_sha256=digest(formula),
                   metric_modules_sha256={d:'f'*64 for d in DIMENSIONS})
        self.cfg=dict(arms={'base':{}},formula=formula,protocol=score,scoring_protocol=score,
                      metric_modules_sha256=score['metric_modules_sha256'],protocol_sha256=digest(score))
        atomic_json(self.run/'config.json',self.cfg)
        atomic_json(self.run/'inputs/tasks.json',self.tasks)
        atomic_json(self.run/'inputs/metadata.json',metadata)

    def publish(self,t):
        v=self.run/'videos/base'/(t['id']+'.mp4')
        v.parent.mkdir(parents=True,exist_ok=True)
        v.write_bytes(t['id'].encode())
        r=dict(t,arm='base',protocol_sha256=self.cfg['protocol_sha256'],video=str(v),video_sha256=file_sha256(v))
        atomic_json(self.run/'records/base'/(t['id']+'.json'),r)
        return r

    def test_only_complete_dimension_and_correct_arm_seeds(self):
        for t in self.tasks[25:]:r=self.publish(t)
        atomic_json(self.run/'records/base/a_00.json',{'not':'yet usable'})
        self.assertEqual(len(checked_records(self.run,'base',self.cfg,self.tasks,'scene')),5)
        with self.assertRaises((KeyError,FileNotFoundError)):
            checked_records(self.run,'base',self.cfg,self.tasks,'static_filter')
        for key, replacement in [('planner_seed',-1),('arm','other'),('video_sha256','0'*64)]:
            atomic_json(self.run/'records/base'/(r['id']+'.json'),dict(r,**{key:replacement}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_records(self.run,'base',self.cfg,self.tasks,'scene')

    def test_finalization_requires_every_video_and_exact_bound_receipts(self):
        rs={t['id']:self.publish(t) for t in self.tasks}
        folder=self.run/'evaluation/base'
        static=dict(receipt_binding(self.cfg,'base','static_filter',self.tasks,rs),state='complete')
        atomic_json(folder/'static_filter.json',static)
        for d in DIMENSIONS:
            item=dict(receipt_binding(self.cfg,'base',d,self.tasks,rs),state='complete',score=.5,coverage={})
            if d=='temporal_flickering':item['static_filter_sha256']=digest(static)
            atomic_json(folder/(d+'.json'),item)
        out=self.run/'summary.json'
        def finalize():
            main([str(self.run),'--arm','base','--finalize-only','--output',str(out)])
        finalize()
        out.unlink()
        record=rs[self.tasks[0]['id']]
        Path(record['video']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Video identity'):
            finalize()
        record['video_sha256']=file_sha256(record['video'])
        atomic_json(self.run/'records/base'/(record['id']+'.json'),record)
        with self.assertRaisesRegex(ValueError,'Receipt/input mismatch'):
            finalize()
        self.assertFalse(out.exists())
