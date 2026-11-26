"""Explicit prompt/seed mapping and dependency checks for VBench subset evaluation."""

import copy
import json
import math
from pathlib import Path
from scrl.evaluation.io import settings

DIMENSIONS=('subject_consistency','background_consistency','temporal_flickering',
    'motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality',
    'object_class','multiple_objects','human_action','color','spatial_relationship',
    'scene','appearance_style','temporal_style','overall_consistency')


def validate_recipe(recipe):
    if not isinstance(recipe,dict) or recipe.get('suite')!='VBench' or recipe.get('scope')!='ordinary':
        raise ValueError('Expected an ordinary VBench recipe')
    for key, minimum in (('planner_seed_base',0),('video_seed_base',0),('seed_stride',25),('fps',1),('max_new_tokens',1)):
        if type(recipe.get(key)) is not int or recipe[key]<minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    fields={'planner_seed_base','video_seed_base','seed_stride','fps','max_new_tokens','suite','scope','formula'}
    if set(recipe)-fields:raise ValueError(f'Unknown VBench recipe fields: {sorted(set(recipe)-fields)}')
    formula=recipe.get('formula')
    fields={'QUALITY_LIST','SEMANTIC_LIST','QUALITY_WEIGHT','SEMANTIC_WEIGHT','DIM_WEIGHT','TASK_INFO','NORMALIZE_DIC'}
    if not isinstance(formula,dict) or fields-set(formula):
        raise ValueError('Missing VBench aggregation formula fields')
    names={d.replace('_',' ') for d in DIMENSIONS}
    for key in ('TASK_INFO','QUALITY_LIST','SEMANTIC_LIST'):
        v=formula[key]
        if not isinstance(v,list) or not v or any(not isinstance(n,str) for n in v) or len(set(v))!=len(v):
            raise ValueError(f'Invalid {key}')
    q,s=set(formula['QUALITY_LIST']),set(formula['SEMANTIC_LIST'])
    if q&s or q|s!=names or set(formula['TASK_INFO'])!=names:raise ValueError('Invalid formula dimension partition')
    def finite(v):return type(v) in (int,float) and math.isfinite(v)
    def positive(v):return finite(v) and v>0
    weights,bounds=formula['DIM_WEIGHT'],formula['NORMALIZE_DIC']
    if not isinstance(weights,dict) or set(weights)!=names or not all(positive(v) for v in weights.values()):
        raise ValueError('Invalid dimension weights')
    if not all(positive(formula[k]) for k in ('QUALITY_WEIGHT','SEMANTIC_WEIGHT')):
        raise ValueError('Invalid aggregate weights')
    if (not isinstance(bounds,dict) or set(bounds)!=names or
            any(not isinstance(v,dict) or not all(finite(v.get(k)) for k in ('Min','Max')) or
                v['Min']>=v['Max'] for v in bounds.values())):
        raise ValueError('Invalid dimension normalization bounds')
    return recipe


def make_tasks(prompts, defaults=None):
    if defaults is None:defaults=settings()
    tasks=[]
    for row in prompts:
        expected=25 if 'temporal_flickering' in row['dimensions'] else 5
        if row['seed_count']!=expected:raise ValueError('Unexpected official sampling count')
        for sample in range(expected):
            tasks.append({'id':f"{row['id']}_{sample:02d}",'prompt_id':row['id'],
                'prompt':row['prompt'],'dimensions':row['dimensions'],'sample_index':sample,
                'planner_seed':defaults['planner_seed_base']+row['index']*defaults['seed_stride']+sample,
                'video_seed':defaults['video_seed_base']+row['index']*defaults['seed_stride']+sample})
    if len({t['id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate task')
    if len({t['planner_seed'] for t in tasks})!=len(tasks):raise ValueError('Repeated planner seed')
    return tasks


def validate_subset(prompts,metadata,official):
    if len({r['prompt'] for r in prompts})!=len(prompts):raise ValueError('Duplicate prompt')
    lookup={r['prompt']:r for r in prompts}
    official_rows={json.dumps(r,sort_keys=True) for r in official}
    dimensions=set()
    for row in metadata:
        if json.dumps(row,sort_keys=True) not in official_rows:raise ValueError('Metadata is not an unchanged official row')
        if row['prompt_en'] not in lookup:raise ValueError('Metadata prompt has no tasks')
        if not set(row['dimension'])<=set(lookup[row['prompt_en']]['dimensions']):raise ValueError('Dimension mapping changed')
        dimensions.update(row['dimension'])
    if dimensions!=set(DIMENSIONS):raise ValueError('Subset must cover all 16 dimensions')
    if set(lookup)!={r['prompt_en'] for r in metadata}:raise ValueError('Prompt coverage mismatch')


def required_tasks(tasks,dimension):
    if dimension=='static_filter':return [t for t in tasks if 'temporal_flickering' in t['dimensions']]
    return [t for t in tasks if dimension in t['dimensions'] and t['sample_index']<5]


def metric_ready(dimension,tasks,completed,static_complete=False):
    if dimension=='temporal_flickering':return static_complete
    required=required_tasks(tasks,dimension)
    if dimension=='static_filter':return any(t['id'] in completed for t in required)
    return bool(required) and all(t['id'] in completed for t in required)


def build_metric_metadata(metadata,tasks,records,dimension,static=None):
    by_key={(t['prompt'],t['sample_index']):t for t in tasks};entries=[]
    for original in metadata:
        if dimension not in original['dimension']:continue
        row=copy.deepcopy(original);row['dimension']=[dimension]
        if dimension=='temporal_flickering':
            if static is None:raise ValueError('Static filtering must finish first')
            row['video_list']=static['results'][row['prompt_en']]['static_path']
        else:
            ids=[by_key[row['prompt_en'],i]['id'] for i in range(5)]
            if any(i not in records for i in ids):raise ValueError('Metric requested before all videos are published')
            row['video_list']=[records[i]['video'] for i in ids]
        if row['video_list']:entries.append(row)
    return entries


def aggregate_scores(scores,formula):
    if set(scores)!=set(DIMENSIONS) or any(v is None or not math.isfinite(v) for v in scores.values()):
        return {'quality':None,'semantic':None,'total':None,'state':'inconclusive_missing_dimension'}
    weighted={}
    for key,value in scores.items():
        name=key.replace('_',' ');bounds=formula['NORMALIZE_DIC'][name]
        weighted[name]=(value-bounds['Min'])/(bounds['Max']-bounds['Min'])*formula['DIM_WEIGHT'][name]
    quality=sum(weighted[k] for k in formula['QUALITY_LIST'])/sum(formula['DIM_WEIGHT'][k] for k in formula['QUALITY_LIST'])
    semantic=sum(weighted[k] for k in formula['SEMANTIC_LIST'])/sum(formula['DIM_WEIGHT'][k] for k in formula['SEMANTIC_LIST'])
    total=(quality*formula['QUALITY_WEIGHT']+semantic*formula['SEMANTIC_WEIGHT'])/(formula['QUALITY_WEIGHT']+formula['SEMANTIC_WEIGHT'])
    return {'quality':quality,'semantic':semantic,'total':total,'state':'complete'}


def load_records(run,arm):
    return {p.stem:json.loads(p.read_text()) for p in (Path(run)/'records'/arm).glob('*.json')}


def shard_tasks(tasks,shard,shards,selected_ids=None):
    if not 0<=shard<shards:raise ValueError('Invalid generation shard')
    if isinstance(selected_ids,dict):
        groups=selected_ids['shards']
        if len(groups)!=shards:raise ValueError('Assignment shard count changed')
        flattened=[i for group in groups for i in group]
        if len(set(flattened))!=len(flattened) or not set(flattened)<={t['id'] for t in tasks}:
            raise ValueError('Duplicate or unknown task assignment')
        selected=set(groups[shard])
        return [t for t in tasks if t['id'] in selected]
    selected={t['id'] for t in tasks} if selected_ids is None else set(selected_ids)
    if selected_ids is not None and len(selected)!=len(selected_ids):raise ValueError('Repeated assignment')
    if not selected<={t['id'] for t in tasks}:raise ValueError('Unknown assigned task')
    return [t for i,t in enumerate(tasks) if i%shards==shard and t['id'] in selected]


def human_action_video_aliases(entries,records,directory):
    """Official human_action reads its target label from the prompt-based filename."""
    import os
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    lookup={row['video']:row for row in records.values()};aliases={}
    for row in entries:
        paths=[]
        for original in row['video_list']:
            record=lookup[original]
            name=f"{row['prompt_en']}-{record['sample_index']}.mp4"
            if Path(name).name!=name or len(name.encode())>255:raise ValueError('Unsafe official metric filename')
            target=directory/name
            try:os.link(original,target)
            except FileExistsError:
                if not os.path.samefile(original,target):raise ValueError('Metric alias points to another video')
            paths.append(str(target));aliases[str(target)]=original
        row['video_list']=paths
    return aliases
