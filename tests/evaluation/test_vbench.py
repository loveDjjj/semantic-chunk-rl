"""Paired benchmark identities, exact coverage and pipelined metric readiness."""

import copy
import json
from argparse import Namespace
import unittest
import tempfile
from pathlib import Path
from scrl.evaluation.protocol import DIMENSIONS,make_tasks,validate_subset,required_tasks,metric_ready,build_metric_metadata,aggregate_scores,human_action_video_aliases,shard_tasks


class VBenchTests(unittest.TestCase):
    def inputs(self):
        prompts=[{'id':'static','prompt':'A still scene','index':8,'seed_count':25,'dimensions':list(DIMENSIONS)},
                 {'id':'action','prompt':'A person runs','index':19,'seed_count':5,'dimensions':['human_action']}]
        metadata=[{'prompt_en':p['prompt'],'dimension':p['dimensions'],'auxiliary_info':{'color':'red'}} for p in prompts]
        return prompts,metadata

    def test_pairing_is_independent_of_worker_assignment(self):
        prompts,metadata=self.inputs();tasks=make_tasks(prompts)
        self.assertEqual(len(tasks),30);self.assertEqual(tasks,make_tasks(copy.deepcopy(prompts)))
        shards=[tasks[i::4] for i in range(4)]
        self.assertEqual({t['id'] for s in shards for t in s},{t['id'] for t in tasks})
        self.assertEqual(len({t['video_seed'] for t in tasks}),len(tasks))
        validate_subset(prompts,metadata,metadata+[{'prompt_en':'unselected','dimension':['scene']}])
        bad=copy.deepcopy(metadata);bad[0]['auxiliary_info']['color']='blue'
        with self.assertRaises(ValueError):validate_subset(prompts,bad,metadata)

    def test_metric_cannot_start_with_missing_inputs(self):
        prompts,_=self.inputs();tasks=make_tasks(prompts)
        need=required_tasks(tasks,'human_action');complete={t['id'] for t in need}
        self.assertTrue(metric_ready('human_action',tasks,complete))
        self.assertFalse(metric_ready('human_action',tasks,complete-{need[0]['id']}))
        self.assertFalse(metric_ready('static_filter',tasks,{tasks[0]['id']}))
        self.assertFalse(metric_ready('temporal_flickering',tasks,complete))
        self.assertFalse(metric_ready('temporal_flickering',tasks,complete,static_complete=True))
        all_static={t['id'] for t in required_tasks(tasks,'static_filter')}
        self.assertEqual(len(all_static),25)
        self.assertTrue(metric_ready('static_filter',tasks,all_static))
        self.assertTrue(metric_ready('temporal_flickering',tasks,all_static,static_complete=True))

    def test_two_storage_assignments_are_disjoint_and_keep_original_shards(self):
        tasks=make_tasks(self.inputs()[0]);remote=[t['id'] for i,t in enumerate(tasks) if i%3==0]
        local=[t['id'] for t in tasks if t['id'] not in remote]
        actual=[]
        for selected in [local,remote]:
            for shard in range(4):
                owned=shard_tasks(tasks,shard,4,selected)
                self.assertTrue(all(tasks.index(t)%4==shard for t in owned));actual.extend(t['id'] for t in owned)
        self.assertEqual(len(actual),len(tasks));self.assertEqual(len(set(actual)),len(tasks))
        with self.assertRaises(ValueError):shard_tasks(tasks,0,4,['unknown'])
        with self.assertRaises(ValueError):shard_tasks(tasks,0,4,[tasks[0]['id']]*2)
        explicit={'shards':[[tasks[5]['id']],[tasks[0]['id']],[],[]]}
        self.assertEqual(shard_tasks(tasks,0,4,explicit),[tasks[5]])
        self.assertEqual(shard_tasks(tasks,1,4,explicit),[tasks[0]])

    def test_explicit_video_mapping_preserves_auxiliary_metadata(self):
        prompts,metadata=self.inputs();tasks=make_tasks(prompts)
        records={t['id']:{'video':'/videos/'+t['id']+'.mp4'} for t in tasks}
        entries=build_metric_metadata(metadata,tasks,records,'color')
        self.assertEqual(len(entries),1);self.assertEqual(len(entries[0]['video_list']),5)
        self.assertEqual(entries[0]['auxiliary_info'],metadata[0]['auxiliary_info'])
        self.assertEqual(entries[0]['prompt_en'],'A still scene')
        selected=[records[tasks[3]['id']]['video'],records[tasks[7]['id']]['video']]
        static={'results':{'A still scene':{'static_path':selected}}}
        self.assertEqual(build_metric_metadata(metadata,tasks,records,'temporal_flickering',static)[0]['video_list'],selected)

    def test_missing_dimension_is_not_replaced_with_zero(self):
        names=[d.replace('_',' ') for d in DIMENSIONS]
        formula={'NORMALIZE_DIC':{d:{'Min':0.,'Max':1.} for d in names},'DIM_WEIGHT':{d:1. for d in names},
                 'QUALITY_LIST':names[:7],'SEMANTIC_LIST':names[7:],'QUALITY_WEIGHT':4,'SEMANTIC_WEIGHT':1}
        scores={d:.5 for d in DIMENSIONS};self.assertEqual(aggregate_scores(scores,formula)['total'],.5)
        scores['color']=None;self.assertIsNone(aggregate_scores(scores,formula)['total'])

    def test_human_action_uses_official_prompt_name_without_copying_video(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'hashed_id.mp4';source.write_bytes(b'video')
            entries=[{'prompt_en':'person is running','video_list':[str(source)]}]
            records={'x':{'video':str(source),'sample_index':2}}
            aliases=human_action_video_aliases(entries,records,root/'aliases')
            alias=Path(entries[0]['video_list'][0])
            self.assertEqual(alias.name,'person is running-2.mp4')
            self.assertTrue(alias.samefile(source));self.assertEqual(aliases[str(alias)],str(source))
            self.assertEqual(alias.name.lower().split('-')[0].split('person is ')[-1].split('_')[0],'running')


if __name__=='__main__':unittest.main()
