"""One frozen model lane: plan and encode once, then generate with bounded video IO."""

import argparse
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import re

from scrl.plan import FinitePlan,PlanLimits,Segment,digest
from scrl.config import DEFAULT_CONFIGS
from scrl.evaluation.io import atomic_json, file_sha256, read
from scrl.evaluation.protocol import DIMENSIONS, make_tasks, shard_tasks, validate_recipe, validate_subset


def prepare_main(argv=None):
    p=argparse.ArgumentParser(description='Prepare a new independent VBench evaluation directory, without loading models.')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIGS/'vbench.json',help='Complete ordinary VBench recipe')
    p.add_argument('--arm',required=True)
    p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--official-metadata',type=Path,required=True)
    p.add_argument('--prompts',type=Path,help='Optional subset: list of strings or historical prompt records')
    p.add_argument('--vbench-source',type=Path,required=True)
    p.add_argument('--vbench-weights',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--planner-variant',choices=('pre_sft','sft'),
                   help='Evaluate the base planner or the configured SFT adapter without an RL checkpoint')
    p.add_argument('--base-executor',action='store_true',
                   help='Keep the original LongLive backbone and base LoRA; omit the trainable policy adapter')
    p.add_argument('--text-batch-size',type=int,default=8,choices=range(1,17),
                   help='Maximum number of distinct captions in one Wan T5 forward')
    p.add_argument('--video-batch-size',type=int,choices=(1,2),default=1,
                   help='Maximum number of identical segment shapes in a LongLive generation batch')
    p.add_argument('--planner-python',type=Path)
    p.add_argument('--denoising-mode',choices=('segment-joint','causal-chunkwise'))
    p.add_argument('--native-chunks',type=int,help='Disable the planner: denoise this many native chunks with one original prompt')
    a=p.parse_args(argv)
    native=a.native_chunks is not None
    if a.planner_variant and (native or a.checkpoint):
        p.error('--planner-variant is for planned evaluation without --native-chunks or --checkpoint')
    if a.base_executor and a.checkpoint:
        p.error('--base-executor cannot be combined with --checkpoint')
    if native and (not 1<=a.native_chunks<=PlanLimits().max_total_chunks or a.checkpoint
                   or a.planner_python or a.denoising_mode=='segment-joint'):
        p.error('Native mode requires a valid fixed chunk count and no checkpoint, planner or joint denoising')
    mode=a.denoising_mode or ('causal-chunkwise' if native else 'segment-joint')
    if a.video_batch_size == 2 and (mode != 'segment-joint' or not a.base_executor or a.checkpoint):
        p.error('Video pairs require the frozen base executor and segment-joint denoising')
    if a.run.exists():raise FileExistsError(a.run)
    defaults=validate_recipe(read(a.config))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',a.arm):raise ValueError('Unsafe arm name')
    paths=read(a.paths)
    root=Path(paths['asset_root'])
    if not root.is_absolute():raise ValueError('asset_root must be absolute')
    def asset(v):
        if v is None:return None
        return str((root/v).resolve())
    raw=paths['assets']
    if a.planner_variant=='sft' and not raw.get('planner_adapter'):
        raise ValueError('--planner-variant sft requires a configured planner_adapter')
    assets={'planner_model':None if native else asset(raw['planner_model']),
            'planner_adapter':None if native or a.planner_variant=='pre_sft' else asset(raw.get('planner_adapter')),
            'executor':{k:asset(v) for k,v in raw['executor'].items()}}
    if native or a.base_executor:assets['executor']['policy_adapter']=None
    required=('upstream','wan','backbone','base_adapter')
    for k in required:
        if not assets['executor'].get(k):raise ValueError('Missing executor asset: '+k)
    for path in [assets['planner_model'],assets['planner_adapter'],*assets['executor'].values()]:
        if path is not None and not Path(path).exists():raise FileNotFoundError(path)
    weights=a.vbench_weights.resolve()
    if not weights.is_dir():raise FileNotFoundError(weights)
    modules={d:file_sha256(a.vbench_source/'vbench'/(d+'.py')) for d in DIMENSIONS}
    official=read(a.official_metadata)
    available={}
    for r in official:
        dims=[d for d in r['dimension'] if d in DIMENSIONS]
        if dims:available.setdefault(r['prompt_en'],set()).update(dims)
    subset=read(a.prompts) if a.prompts else list(available)
    rows=[]
    for i,row in enumerate(subset):
        text=row if isinstance(row,str) else row['prompt']
        dims=sorted(available[text])
        if isinstance(row,dict):
            if set(row['dimensions'])!=set(dims):
                # Historical subsets can intentionally select fewer dimensions per prompt.
                if not set(row['dimensions'])<=set(dims):raise ValueError('Unofficial dimension')
                dims=row['dimensions']
            idx,ident=row['index'],row['id']
        else:idx,ident=i,digest(text)[:16]
        if type(idx) is not int or idx<0 or not re.fullmatch(r'[A-Za-z0-9_-]+',ident):
            raise ValueError('Invalid task identity')
        rows.append(dict(prompt=text,id=ident,index=idx,dimensions=dims,seed_count=25 if 'temporal_flickering' in dims else 5))
    selected={r['prompt']:set(r['dimensions']) for r in rows}
    meta=[r for r in official if r['prompt_en'] in selected and set(r['dimension'])<=selected[r['prompt_en']]]
    validate_subset(rows,meta,official)
    tasks=make_tasks(rows,defaults)
    limits=asdict(PlanLimits())
    if native:limits=dict(max_segments=1,max_total_chunks=a.native_chunks,allowed_chunks=[a.native_chunks])
    score=dict(tasks_sha256=digest(tasks),metadata_sha256=digest(meta),formula_sha256=digest(defaults['formula']),
               metric_modules_sha256=modules)
    cp=a.checkpoint.resolve() if a.checkpoint else None
    version=0
    model_hashes={}
    for role,path in [('backbone',assets['executor']['backbone']),('base_adapter',assets['executor']['base_adapter']),
                      ('planner_adapter',assets['planner_adapter']),('policy_adapter',assets['executor'].get('policy_adapter'))]:
        if path is not None:model_hashes[role]=file_sha256(Path(path)/'adapter_model.safetensors' if role in ('planner_adapter','policy_adapter') else path)
    if cp:
        state=read(cp/'state.json')
        version=state['stamp']['planner']
        wanted={'planner.pt','executor.pt'} if assets['executor'].get('policy_adapter') else {'planner.pt'}
        if set(state['files'])!=wanted:raise ValueError('Checkpoint model set does not match configured adapters')
        for name in wanted:
            actual=file_sha256(cp/name)
            if actual!=state['files'][name]:raise ValueError('Checkpoint digest mismatch')
            model_hashes[name]=actual
    if cp and not assets['planner_adapter']:raise ValueError('Checkpoint needs a configured planner adapter')
    arm={'version':version,'checkpoint':str(cp) if cp else None,'planner_variant':'native' if native else ('sft' if assets['planner_adapter'] else 'pre_sft')}
    if cp and assets['executor'].get('policy_adapter'):
        arm['executor_version'] = state['stamp']['executor']
    planner_python=None if native else (a.planner_python or paths.get('role_python',{}).get('planner'))
    planner_python=asset(planner_python) if planner_python else None
    gen=dict(assets=assets,weights_sha256=model_hashes,arm=arm,plan=limits,fps=defaults['fps'],max_new_tokens=defaults['max_new_tokens'],
             denoising_mode=mode,planner_python=planner_python,native_chunks=a.native_chunks,text_batch_size=a.text_batch_size)
    if a.video_batch_size > 1:
        gen['video_batch_size'] = a.video_batch_size
    protocol=dict(score,generation_sha256=digest(gen))
    cfg=dict(arms={a.arm:arm},assets=assets,formula=defaults['formula'],metric_modules_sha256=modules,
             recipe=defaults,recipe_sha256=digest(defaults),
             vbench_source=str(a.vbench_source.resolve()),vbench_weights=str(weights),fps=gen['fps'],max_new_tokens=gen['max_new_tokens'],
             plan=limits,planner_python=planner_python,denoising_mode=mode,native_chunks=a.native_chunks,wall_seconds=86400,
             protocol=protocol,protocol_sha256=digest(protocol),scoring_protocol=score,scoring_protocol_sha256=digest(score),
             generation_identity=gen,generation_sha256=digest(gen))
    a.run.mkdir(parents=True,exist_ok=False)
    for name,value in [('tasks',tasks),('metadata',meta),('prompts',rows)]:
        atomic_json(a.run/'inputs'/(name+'.json'),value)
    atomic_json(a.run/'config.json',cfg)
    print(json.dumps({'arm':a.arm,'prompts':len(rows),'tasks':len(tasks),'run':str(a.run.resolve())}))


def write_video(path,pixels,fps,ffmpeg_path=None):
    import imageio_ffmpeg
    import numpy as np
    array=np.ascontiguousarray(pixels)
    if array.dtype!=np.uint8 or array.ndim!=4 or array.shape[-1]!=3:raise ValueError('Expected uint8 THWC video')
    temporary=path.with_suffix('.tmp.mp4')
    cmd=[ffmpeg_path or imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-y',
         '-f','rawvideo','-pix_fmt','rgb24','-video_size',f'{array.shape[2]}x{array.shape[1]}',
         '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','medium',
         '-crf','18','-pix_fmt','yuv420p','-threads','1','-map_metadata','-1',
         '-flags:v','+bitexact','-fflags','+bitexact','-f','mp4',str(temporary)]
    subprocess.run(cmd,input=array.tobytes(),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=180)
    temporary.replace(path)


def load_adapter(model,state):
    from peft import set_peft_model_state_dict
    from peft import get_peft_model_state_dict
    expected=get_peft_model_state_dict(model,adapter_name='default')
    if set(state)!=set(expected) or any(state[k].shape!=expected[k].shape for k in state):
        raise ValueError('Adapter keys or shapes differ from the model')
    result=set_peft_model_state_dict(model,state,adapter_name='default')
    if result.unexpected_keys:raise ValueError('Unexpected checkpoint keys')


def native_plan(prompt, chunks):
    return FinitePlan((Segment(prompt, chunks),))


def encode_caption_window(plans, encoder, batch_size):
    """Encode distinct captions across a bounded window of independent tasks."""
    unique = list(dict.fromkeys(s.caption for plan in plans for s in plan.segments))
    tok = encoder.tokenizer
    by_caption = {}
    for start in range(0, len(unique), batch_size):
        group = unique[start:start+batch_size]
        if any(len(tok.tokenizer(tok._clean(c), truncation=False, add_special_tokens=True)['input_ids']) > tok.seq_len
               for c in group):
            raise ValueError('Caption would be truncated by T5')
        # Upstream Wan pads to a fixed length and masks padding per row.
        encoded = {k:v.detach().cpu() for k,v in encoder(text_prompts=group).items()}
        for i, caption in enumerate(group):
            # torch.save serializes an entire backing storage even for a one-row view.
            by_caption[caption] = {k:v[i:i+1].clone() for k,v in encoded.items()}
    return [{s.caption:by_caption[s.caption] for s in plan.segments} for plan in plans]


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--arm',required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--task-ids-file');a=p.parse_args(argv)
    # Preserve the caller's shared mount name in records read by other nodes.
    run=Path(a.run).absolute();cfg=json.loads((run/'config.json').read_text());arm=a.arm
    if arm not in cfg['arms'] or Path(arm).name!=arm or arm in ('.','..'):raise ValueError('Unknown or unsafe arm')
    for name in ('videos','records','plans'):(run/name/arm).mkdir(parents=True,exist_ok=True)
    cfg.setdefault('conditions_dir',os.environ.get('SCRL_EVAL_CONDITIONS',str(run/'conditions')))
    folder=run/'workers'/arm/f'generate-{a.shard}';folder.mkdir(parents=True,exist_ok=True)
    started=time.time()
    def status(state,**kw):atomic_json(folder/'status.json',{'state':state,'pid':os.getpid(),'shard':a.shard,'checked_unix':time.time(),**kw})
    def limits():
        if (run/'STOP').exists() or (run/('STOP-'+arm)).exists():raise InterruptedError('Evaluation STOP requested')
        if time.time()-started>cfg.get('wall_seconds',86400):raise TimeoutError('Evaluation time budget exceeded')
    try:
        import torch
        from safetensors.torch import load_file
        from scrl.models.executor import LongLivePaths,load_longlive
        from scrl.models.segments import SegmentExecutor,tensor_hash
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.set_grad_enabled(False)
        tasks=json.loads((run/'inputs/tasks.json').read_text())
        selected=json.loads(Path(a.task_ids_file).read_text()) if a.task_ids_file else None
        owned=shard_tasks(tasks,a.shard,a.shards,selected)
        fleet = os.environ.get('SCRL_VBENCH_FLEET') == '1'
        if fleet:
            owned=[t for t in owned if not (run/'records'/arm/(t['id']+'.json')).is_file()]
        assets=cfg['assets'];checkpoint=cfg['arms'][arm].get('checkpoint');version=cfg['arms'][arm]['version']
        status('loading_planner_and_text',tasks=len(owned))
        def needs_encoding(task):
            path=run/'plans'/arm/(task['id']+'.json')
            if not path.exists():return True
            if not fleet:return False
            saved=json.loads(path.read_bytes());condition=Path(saved['conditions'])
            return not condition.is_file() or file_sha256(condition)!=saved['conditions_sha256']
        need_plan=[t for t in owned if needs_encoding(t)]
        if need_plan:
            policy=None
            native=cfg.get('native_chunks') is not None
            if native:
                if (cfg['denoising_mode']!='causal-chunkwise' or checkpoint or assets['executor'].get('policy_adapter')
                    or cfg['arms'][arm].get('planner_variant')!='native'):raise ValueError('Invalid native arm')
            elif cfg.get('planner_python'):
                planned=folder/'sampled'
                planned.mkdir(exist_ok=True)
                pending_plans=[t for t in need_plan if not fleet or not (planned/(t['id']+'.json')).exists()]
                if pending_plans:
                    if cfg.get('paired_joint_run'):
                        raise RuntimeError('Missing paired plan; refusing to resample')
                    from scrl.evaluation.planner_pool import plan_concurrently
                    lanes=int(os.environ.get('SCRL_EVAL_PLANNER_LANES','1'))
                    status('planning',pending=len(pending_plans),planner_lanes=min(lanes,len(pending_plans)))
                    plan_concurrently([cfg['planner_python'],'-m','scrl.evaluation._planner_lane',
                                       str(run),arm,str(planned)],pending_plans,lanes,cfg['wall_seconds'])
            else:
                from scrl.models.planner import PlannerPolicy
                policy=PlannerPolicy(assets['planner_model'],assets['planner_adapter'],device='cuda:0',limits=PlanLimits(**cfg['plan']),
                                     base_inference=cfg['arms'][arm].get('planner_variant')=='pre_sft')
                policy.optimizer=None
                if checkpoint:
                    payload=torch.load(Path(checkpoint)/'planner.pt',map_location='cpu',weights_only=True,mmap=True)
                    load_adapter(policy.model,payload['planner']);del payload
                policy.version=version
            sys.path.insert(0,str(Path(assets['executor']['upstream']).resolve()))
            from self_forcing import longlive_wan_wrapper as wrapper
            wrapper.WAN2_1_CHECKPOINT=str(Path(assets['executor']['wan']).resolve())
            encoder=wrapper.WanTextEncoder().to(device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(False)
            def plan_task(task):
                if native:
                    return native_plan(task['prompt'],cfg['native_chunks']),[]
                if policy is None:
                    saved=json.loads((planned/(task['id']+'.json')).read_bytes())
                    return FinitePlan.from_dict(saved['plan'],PlanLimits(**cfg['plan'])),saved['ids']
                plan,response,_=policy.sample(task['prompt'],task['planner_seed'],max_new_tokens=cfg['max_new_tokens'],anchor_only=True)
                return plan,response[0].tolist()
            size=cfg.get('generation_identity',{}).get('text_batch_size')
            window=size or 1  # Old manifests encoded up to four captions from one task.
            for index in range(0,len(need_plan),window):
                limits();batch=need_plan[index:index+window]
                status('encoding',completed=index,tasks=len(need_plan),text_batch_size=size or 4)
                entries=[plan_task(task) for task in batch]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    mappings=encode_caption_window([plan for plan,_ in entries],encoder,size or 4)
                for task,(plan,ids),mapping in zip(batch,entries,mappings):
                    cp=Path(cfg['conditions_dir'])/arm/(task['id']+'.pt');cp.parent.mkdir(parents=True,exist_ok=True)
                    temporary=cp.with_suffix('.tmp.pt')
                    torch.save(mapping,temporary);temporary.replace(cp)
                    atomic_json(run/'plans'/arm/(task['id']+'.json'),dict(task,plan=plan.to_dict(),
                        conditions=str(cp),conditions_sha256=file_sha256(cp),planner_response_ids=ids,
                        policy_version=version,protocol_sha256=cfg['protocol_sha256']))
                del mappings,mapping
            del policy,encoder
            gc.collect();torch.cuda.empty_cache()
        jobs=[json.loads((run/'plans'/arm/(t['id']+'.json')).read_text()) for t in owned]
        jobs.sort(key=lambda j:(-sum(s['num_chunks'] for s in j['plan']['segments']),j['id']))
        status('loading_executor',tasks=len(jobs))
        class PrecomputedTextOnly(torch.nn.Module):
            def forward(self,*args,**kwargs):raise RuntimeError('Evaluation requires precomputed text conditions')
        backend=load_longlive(LongLivePaths(**assets['executor']),'cuda:0',text_encoder=PrecomputedTextOnly())
        backend.eval_chunkwise = cfg.get('denoising_mode') == 'causal-chunkwise'
        for name,param in backend.pipe.generator.model.named_parameters():
            if 'lora_' in name:param.data=param.data.float()
        frozen = assets['executor'].get('policy_adapter') is None
        if checkpoint and not frozen:
            payload=torch.load(Path(checkpoint)/'executor.pt',map_location='cpu',weights_only=True,mmap=True)
            load_adapter(backend.pipe.generator.model,payload['executor']);del payload
        elif not frozen:
            load_adapter(backend.pipe.generator.model,load_file(str(Path(assets['executor']['policy_adapter'])/'adapter_model.safetensors')))
        backend.pipe.eval().requires_grad_(False);backend.version=0 if frozen else cfg['arms'][arm].get('executor_version',version)
        executor=SegmentExecutor(backend,version=backend.version)
        atomic_json(folder/'identity.json',{'arm':arm,'policy_version':version,'checkpoint':checkpoint,
            'inference_only':True,'optimizer_loaded':False,'trainable_parameters':sum(p.requires_grad for p in backend.pipe.parameters()),
            'lora_dtypes':sorted({str(p.dtype) for n,p in backend.pipe.generator.model.named_parameters() if 'lora_' in n}),
            'torch':torch.__version__,'protocol_sha256':cfg['protocol_sha256']})
        def publish(job,pixels,record):
            video=run/'videos'/arm/(job['id']+'.mp4');write_video(video,pixels,cfg['fps'],cfg.get('ffmpeg_executable'))
            record=dict(record,video=str(video),video_sha256=file_sha256(video),published_unix=time.time(),generation_host=os.uname().nodename)
            expected=cfg.get('validation_tasks',{}).get(arm)
            if expected and expected['id']==job['id']:
                if record['latent_sha256']!=expected['latent_sha256'] or record['video_sha256']!=expected['video_sha256']:
                    raise RuntimeError('Cross-storage calibration did not exactly reproduce the original video')
                atomic_json(run/'validation'/f'{arm}.json',{'state':'passed','id':job['id'],
                    'latent_sha256':record['latent_sha256'],'video_sha256':record['video_sha256'],'checked_unix':time.time()})
            atomic_json(run/'records'/arm/(job['id']+'.json'),record)
        from scrl.evaluation.video_batches import shape_groups
        size = cfg.get('generation_identity', {}).get('video_batch_size', 1)
        if size != 1 and (not frozen or checkpoint or cfg['denoising_mode'] != 'segment-joint'):
            raise ValueError('Video pairs require frozen segment-joint LongLive')
        pending, oom_shapes = [], set()
        index = 0
        with ThreadPoolExecutor(max_workers=1) as encoders:
            for group in shape_groups(jobs, size):
                limits()
                ready = []
                for job in group:
                    existing = run/'records'/arm/(job['id']+'.json')
                    if existing.exists():
                        record = read(existing)
                        if record['protocol_sha256'] != cfg['protocol_sha256'] or file_sha256(record['video']) != record['video_sha256']:
                            raise ValueError('Existing video does not match this evaluation')
                        continue
                    if job['protocol_sha256'] != cfg['protocol_sha256'] or file_sha256(job['conditions']) != job['conditions_sha256']:
                        raise ValueError('Plan or text conditions changed')
                    plan = FinitePlan.from_dict(job['plan'], PlanLimits(**cfg['plan']))
                    mapping = torch.load(job['conditions'], map_location='cpu', weights_only=True)
                    if not {s.caption for s in plan.segments} <= mapping.keys():
                        raise ValueError('Precomputed caption missing')
                    ready.append((job, plan, mapping))
                if not ready:
                    continue
                shape = tuple(s.num_chunks for s in ready[0][1].segments)
                ids = [row[0]['id'] for row in ready]
                status('generating', submitted=index, tasks=len(jobs), task=ids[0], batch=ids, total_chunks=sum(shape))
                tick = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                latents = None
                if len(ready) == 2 and shape not in oom_shapes:
                    try:
                        latents = backend.rollout_eval_pair([r[1] for r in ready],
                                    [r[0]['video_seed'] for r in ready], [r[2] for r in ready])
                    except torch.cuda.OutOfMemoryError:
                        oom_shapes.add(shape)
                    if latents is None:
                        gc.collect()
                        torch.cuda.empty_cache()
                paired = latents is not None
                batch_seconds = time.monotonic() - tick
                for i, (job, plan, mapping) in enumerate(ready):
                    if paired:
                        latent = latents[i]
                        seconds = batch_seconds / len(ready)
                    else:
                        backend.conditions.clear()
                        backend.conditions.update(mapping)
                        backend.pipe.generator.seq_len = 0
                        tick = time.monotonic()
                        latent = executor.rollout(plan, job['video_seed'], capture=False, retain_final_cache=False).final.latents
                        seconds = time.monotonic() - tick
                    if len(pending) >= 2:
                        pending.pop(0).result()
                    tick = time.monotonic()
                    pixels = backend.decode(latent).numpy()
                    if len(pixels) != plan.total_chunks * 12 - 3:
                        raise ValueError('Generated frame extent differs from plan')
                    record = dict(job, arm=arm, frames=len(pixels), fps=cfg['fps'], height=int(pixels.shape[1]), width=int(pixels.shape[2]),
                        latent_sha256=tensor_hash(latent), generation_seconds=seconds,
                        generation_decode_seconds=seconds + time.monotonic()-tick, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        actual_video_batch_size=len(ready) if paired else 1, batch_task_ids=ids if paired else [job['id']],
                        batch_denoise_seconds=batch_seconds if paired else seconds, pair_oom_fallback=shape in oom_shapes)
                    pending.append(encoders.submit(publish, job, pixels, record))
                    del pixels, latent
                    index += 1
                del latents
            for f in pending:
                f.result()
        status('complete',completed=len(jobs),seconds=time.time()-started)
    except BaseException as error:
        status('failed',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
