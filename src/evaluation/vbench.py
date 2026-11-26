"""Invoke the pinned official metric implementations on explicit video mappings."""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

from scrl.evaluation.protocol import (DIMENSIONS, aggregate_scores, build_metric_metadata,
                                      load_records, human_action_video_aliases)
from scrl.evaluation.io import atomic_json, file_sha256, read, settings


def static_filter(run,arm,cfg,tasks):
    from easydict import EasyDict
    from vbench.cli.static_filter import StaticFilter
    args=EasyDict(model=str(Path(cfg['vbench_weights'])/'raft_model/models/raft-things.pth'),
                  small=False,mixed_precision=False,alternate_corr=False)
    model=StaticFilter(args,'cuda')
    folder=run/'evaluation'/arm;partial=folder/'static_filter_partial.json'
    result=json.loads(partial.read_text()) if partial.exists() else {}
    prompts=sorted({t['prompt'] for t in tasks if 'temporal_flickering' in t['dimensions']})
    by_key={(t['prompt'],t['sample_index']):t for t in tasks}
    for prompt in prompts:result.setdefault(prompt,{'static_path':[],'tested':[],'complete':False})
    while not all(v['complete'] for v in result.values()):
        if (run/'STOP').exists() or (run/('STOP-'+arm)).exists():raise InterruptedError('Evaluation stopped')
        progressed=False
        for prompt,row in result.items():
            if row['complete']:continue
            index=len(row['tested']);task=by_key[prompt,index];record_path=run/'records'/arm/(task['id']+'.json')
            if not record_path.exists():continue
            record=json.loads(record_path.read_text());accepted=bool(model.infer(record['video']))
            row['tested'].append({'sample':index,'static':accepted})
            if accepted:row['static_path'].append(record['video'])
            row['static_count']=len(row['static_path']);row['complete']=row['static_count']>=5 or len(row['tested'])>=25
            atomic_json(partial,result);progressed=True
        if not progressed:time.sleep(1)
    coverage={'prompts':len(prompts),'prompts_with_five':sum(v['static_count']==5 for v in result.values()),
              'selected_videos':sum(v['static_count'] for v in result.values()),
              'below_target':[p for p,v in result.items() if v['static_count']<5]}
    value={'state':'complete','results':result,'coverage':coverage,
           'protocol':'Official StaticFilter, inspect seed indices in order, first 5 accepted among 25'}
    atomic_json(folder/'static_filter.json',value)
    return value


def evaluate(run,arm,dimension,cfg,tasks):
    import torch
    from vbench.utils import init_submodules
    folder=run/'evaluation'/arm
    static=json.loads((folder/'static_filter.json').read_text()) if dimension=='temporal_flickering' else None
    records=load_records(run,arm)
    entries=build_metric_metadata(json.loads((run/'inputs/metadata.json').read_text()),tasks,
                                  records,dimension,static)
    expected={p for entry in entries for p in entry['video_list']}
    aliases=human_action_video_aliases(entries,records,folder/'human_action_videos') if dimension=='human_action' else {}
    info=folder/(dimension+'_full_info.json');atomic_json(info,entries)
    if any(not Path(p).is_file() for p in expected):raise FileNotFoundError('A required benchmark video is missing')
    if not entries:
        result={'state':'unscorable','dimension':dimension,'score':None,'details':[],
                'reason':'No video passed the official static filter',
                'coverage':{'expected_videos':0,'scored_videos':0,'unscored_paths':[]}}
    else:
        module=importlib.import_module('vbench.'+dimension)
        actual=file_sha256(module.__file__)
        if actual!=cfg['metric_modules_sha256'][dimension]:raise RuntimeError('Metric implementation changed')
        submodules=init_submodules([dimension],local=True,read_frame=False)[dimension]
        tick=time.monotonic()
        try:
            score,details=getattr(module,'compute_'+dimension)(str(info),torch.device('cuda'),submodules,
                imaging_quality_preprocessing_mode='longer')
        except ZeroDivisionError as error:
            trace=error.__traceback__
            while trace.tb_next is not None:trace=trace.tb_next
            if dimension!='color' or trace.tb_frame.f_code.co_name!='color' or trace.tb_frame.f_locals.get('video_count')!=0:
                raise
            # The official color metric excludes videos without detected objects.
            # An empty denominator is unscorable, never a fabricated zero score.
            score,details=None,[]
        if score is not None and not math.isfinite(float(score)):raise RuntimeError('Nonfinite official metric')
        for row in details:row['video_path']=aliases.get(row['video_path'],row['video_path'])
        actual_paths={row['video_path'] for row in details}
        if not actual_paths<=expected:raise RuntimeError('Metric scored unexpected videos')
        if dimension!='color' and actual_paths!=expected:raise RuntimeError('Incomplete metric coverage')
        result={'state':'complete' if score is not None else 'unscorable','dimension':dimension,
                'score':float(score) if score is not None else None,'details':details,
                'coverage':{'metadata_rows':len(entries),'expected_videos':len(expected),
                            'scored_videos':len(actual_paths),'unscored_paths':sorted(expected-actual_paths)},
                'seconds':time.monotonic()-tick,'module_sha256':actual,'module_path':module.__file__,
                'metadata_sha256':file_sha256(info)}
        if aliases:result['metric_video_aliases']=aliases
    if static is not None:result['static_filter_coverage']=static['coverage']
    atomic_json(folder/(dimension+'.json'),result)
    return result


def summarize(run, arm, formula):
    metrics = {}
    for dimension in DIMENSIONS:
        item = read(run/'evaluation'/arm/(dimension+'.json'))
        if item['dimension'] != dimension or item['state'] not in ('unscorable', 'complete'):
            raise ValueError(f'Invalid metric receipt: {dimension}')
        if ((item['state']=='unscorable') != (item['score'] is None) or
                (item['score'] is not None and not math.isfinite(item['score']))):
            raise ValueError(f'Invalid metric score: {dimension}')
        metrics[dimension] = item
    scores = {d: m['score'] for d,m in metrics.items()}
    return {'arm':arm, 'aggregate':aggregate_scores(scores,formula), 'raw_dimensions':scores,
            'coverage':{d:m['coverage'] for d,m in metrics.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--arm', required=True)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--summary-only',action='store_true',help='Read saved metric receipts; no GPU or videos required')
    group.add_argument('--dimension',choices=('static_filter',*DIMENSIONS),help='Only run this GPU metric')
    parser.add_argument('--output',type=Path,help='Write summary to a new file; otherwise print JSON')
    args=parser.parse_args(argv)
    if args.dimension and args.output:parser.error('--output requires a summary')
    if args.output and args.output.exists():raise FileExistsError(args.output)
    run,arm=args.run,args.arm
    if arm in ('','.','..') or Path(arm).name!=arm:raise ValueError('Invalid arm')
    cfg=read(run/'config.json')
    if 'arms' in cfg and arm not in cfg['arms']:raise ValueError('Unknown evaluation arm')
    formula=cfg['formula'] if 'formula' in cfg else settings()['formula']
    if not args.summary_only:
        tasks=read(run/'inputs/tasks.json')
        records=load_records(run,arm)
        if not tasks or len(tasks)!=len(records) or set(records)!={t['id'] for t in tasks}:
            raise ValueError('All generated task records must be present before scoring')
        for task in tasks:
            record=records[task['id']]
            if (any(task[k]!=record[k] for k in ('prompt','prompt_id','video_seed','sample_index')) or
                    file_sha256(record['video'])!=record['video_sha256']):
                raise ValueError(f"Video identity mismatch: {task['id']}")
        os.environ.update(VBENCH_CACHE_DIR=cfg['vbench_weights'],VBENCH_BERT_PATH=str(Path(cfg['vbench_weights'])/'bert-base-uncased'),
                          TORCH_HOME=str(Path(cfg['vbench_weights'])/'torch_hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        sys.path.insert(0,cfg['vbench_source'])
        import torch
        torch.set_num_threads(2);torch.cuda.set_device(0)
        (run/'evaluation'/arm).mkdir(parents=True,exist_ok=True)
        for d in [args.dimension] if args.dimension else ('static_filter',*DIMENSIONS):
            if d=='static_filter':static_filter(run,arm,cfg,tasks)
            else:evaluate(run,arm,d,cfg,tasks)
    if not args.dimension:
        report=summarize(run,arm,formula)
        data=json.dumps(report,indent=2,allow_nan=False)
        if args.output:
            with args.output.open('x') as stream: print(data,file=stream)
        else: print(data)


if __name__=='__main__':main()
