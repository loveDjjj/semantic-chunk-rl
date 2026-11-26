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
                                      required_tasks, human_action_video_aliases)
from scrl.evaluation.io import atomic_json, file_sha256, read, settings
from scrl.plan import digest


def checked_records(run, arm, cfg, tasks, dimension=None):
    selected = required_tasks(tasks, dimension) if dimension else tasks
    directory = run/'records'/arm
    if not selected or (dimension is None and {p.stem for p in directory.glob('*.json')} != {t['id'] for t in tasks}):
        raise ValueError('Required generated task records are missing or unexpected')
    records = {}
    for task in selected:
        tid = task['id']
        r = read(directory/(tid+'.json'))
        if (any(r[k] != task[k] for k in ('id','prompt','prompt_id','dimensions','sample_index','planner_seed','video_seed'))
                or r['arm'] != arm or r['protocol_sha256'] != cfg['protocol_sha256']
                or not os.path.samefile(run/'videos'/arm/(tid+'.mp4'), r['video'])
                or file_sha256(r['video']) != r['video_sha256']):
            raise ValueError(f'Video identity mismatch: {tid}')
        records[tid] = r
    return records


def receipt_binding(cfg,arm,dimension,tasks,records):
    selected = required_tasks(tasks,dimension)
    return dict(protocol_sha256=cfg['protocol_sha256'], arm=arm, dimension=dimension,
                inputs_sha256=digest([[t['id'],records[t['id']]['video_sha256']] for t in selected]))


def check_receipt(item,binding):
    if any(item.get(k)!=v for k,v in binding.items()):
        raise ValueError(f"Receipt/input mismatch: {binding['dimension']}")


def static_filter(run,arm,cfg,tasks,records,binding):
    from easydict import EasyDict
    from vbench.cli.static_filter import StaticFilter
    args=EasyDict(model=str(Path(cfg['vbench_weights'])/'raft_model/models/raft-things.pth'),
                  small=False,mixed_precision=False,alternate_corr=False)
    model=StaticFilter(args,'cuda')
    folder=run/'evaluation'/arm;partial=folder/'static_filter_partial.json'
    prior=read(partial) if partial.exists() else {}
    result=prior['results'] if prior.get('binding')==binding else {}
    prompts=sorted({t['prompt'] for t in tasks if 'temporal_flickering' in t['dimensions']})
    by_key={(t['prompt'],t['sample_index']):t for t in tasks}
    for prompt in prompts:result.setdefault(prompt,{'static_path':[],'tested':[],'complete':False})
    while not all(v['complete'] for v in result.values()):
        if (run/'STOP').exists() or (run/('STOP-'+arm)).exists():raise InterruptedError('Evaluation stopped')
        for prompt,row in result.items():
            if row['complete']:continue
            index=len(row['tested']);task=by_key[prompt,index]
            record=records[task['id']];accepted=bool(model.infer(record['video']))
            row['tested'].append({'sample':index,'static':accepted})
            if accepted:row['static_path'].append(record['video'])
            row['static_count']=len(row['static_path']);row['complete']=row['static_count']>=5 or len(row['tested'])>=25
            atomic_json(partial,dict(binding=binding,results=result))
    coverage={'prompts':len(prompts),'prompts_with_five':sum(v['static_count']==5 for v in result.values()),
              'selected_videos':sum(v['static_count'] for v in result.values()),
              'below_target':[p for p,v in result.items() if v['static_count']<5]}
    value={**binding,'state':'complete','results':result,'coverage':coverage,
           'protocol':'Official StaticFilter, inspect seed indices in order, first 5 accepted among 25'}
    atomic_json(folder/'static_filter.json',value)
    return value


def evaluate(run,arm,dimension,cfg,tasks,records,binding,*,entries=None):
    import torch
    from vbench.utils import init_submodules
    folder=run/'evaluation'/arm
    static=json.loads((folder/'static_filter.json').read_text()) if dimension=='temporal_flickering' else None
    if static is not None:
        check_receipt(static, receipt_binding(cfg,arm,'static_filter',tasks,records))
        if static['state']!='complete': raise ValueError('Incomplete static filter')
    if entries is None:
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
    if static is not None:
        result.update(static_filter_coverage=static['coverage'],static_filter_sha256=digest(static))
    result.update(binding)
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
    group.add_argument('--finalize-only',action='store_true',help='Validate every video and bound metric receipt, then aggregate on CPU')
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
        score=cfg['scoring_protocol']
        if (digest(tasks)!=score['tasks_sha256'] or digest(read(run/'inputs/metadata.json'))!=score['metadata_sha256']
                or digest(formula)!=score['formula_sha256'] or score['metric_modules_sha256']!=cfg['metric_modules_sha256']
                or digest(cfg['protocol'])!=cfg['protocol_sha256'] or not all(cfg['protocol'].get(k)==v for k,v in score.items())):
            raise ValueError('Evaluation inputs differ from the prepared protocol')
        records=checked_records(run,arm,cfg,tasks,args.dimension)
    if not args.summary_only and not args.finalize_only:
        os.environ.update(VBENCH_CACHE_DIR=cfg['vbench_weights'],VBENCH_BERT_PATH=str(Path(cfg['vbench_weights'])/'bert-base-uncased'),
                          TORCH_HOME=str(Path(cfg['vbench_weights'])/'torch_hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        sys.path.insert(0,cfg['vbench_source'])
        import torch
        torch.set_num_threads(2);torch.cuda.set_device(0)
        (run/'evaluation'/arm).mkdir(parents=True,exist_ok=True)
        for d in [args.dimension] if args.dimension else ('static_filter',*DIMENSIONS):
            binding=receipt_binding(cfg,arm,d,tasks,records)
            if d=='static_filter':static_filter(run,arm,cfg,tasks,records,binding)
            else:evaluate(run,arm,d,cfg,tasks,records,binding)
    if not args.dimension:
        if not args.summary_only:
            directory=run/'evaluation'/arm
            static=read(directory/'static_filter.json')
            check_receipt(static,receipt_binding(cfg,arm,'static_filter',tasks,records))
            if static['state']!='complete':raise ValueError('Incomplete static filter')
            for d in DIMENSIONS:
                item=read(directory/(d+'.json'))
                check_receipt(item,receipt_binding(cfg,arm,d,tasks,records))
                if d=='temporal_flickering' and item.get('static_filter_sha256')!=digest(static):
                    raise ValueError('Temporal flickering used a different static filter')
        report=summarize(run,arm,formula)
        if args.finalize_only and report['aggregate']['state']!='complete':
            raise ValueError('Incomplete aggregate: no complete result for every official dimension')
        data=json.dumps(report,indent=2,allow_nan=False)
        if args.output:
            with args.output.open('x') as stream: print(data,file=stream)
        else: print(data)


if __name__=='__main__':main()
