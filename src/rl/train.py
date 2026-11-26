"""Full-data joint training on persistent workers, with bounded pipelines, a time budget and retained recovery points."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
import torch

from scrl.plan import PlanLimits, PolicyStamp
from scrl.models.verl_policy import VerlPlannerPolicy
from scrl.runtime.transport import Peer, atomic_json, record_failure
from scrl.runtime.worker import cpu_state, state_hash
from scrl.rl.checkpoint import (JointCoordinator, SnapshotWriter, freeze_cpu, load_checkpoint,
                                publish_planner_checkpoint, publish_split_checkpoint, wait_executor)
from scrl.rl.objectives import CROSSED_CREDIT, WHOLE_CREDIT, planner_advantages
from scrl.runtime.launch import advance_dataset, prompt_seed
from scrl.rl.identity import training_fingerprint, validate_training_mode, verify_campaign_resume
from scrl.rl.rollout import collect_batch


def planner_update_batch(groups, rows, indices, *, seed, epoch, total_prompts, k, s, stamp, credit):
    """按采样顺序组装更新输入，不在不同提示词间比较优势。"""
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row.group_id, []).append(row)
    candidates = [f'c{i}' for i in range(k)]
    samples, advantages, targets = [], [], []
    for index in indices:
        seeds = [prompt_seed(seed, epoch, index, total_prompts) + 100 + j for j in range(s)]
        scores = planner_advantages(by_prompt[f'p{index}'], stamp, expected_candidates=candidates,
                                    expected_seeds=seeds, credit=credit)
        values = torch.load(groups[index]['samples'], map_location='cpu', weights_only=False)
        if len(values) != k:
            raise ValueError("Planner sample count does not match the scored candidates")
        if credit in (CROSSED_CREDIT, WHOLE_CREDIT):
            group_rows = by_prompt[f'p{index}']
            scored = {r.candidate_id: r.plan for r in group_rows}
            if any(v.plan != scored[c] or v.policy_version != stamp.planner for c, v in zip(candidates, values)):
                raise ValueError("Policy traces do not match the scored plans or version")
            if credit == CROSSED_CREDIT and groups[index]['target'] != by_prompt[f'p{index}'][0].target_segment:
                raise ValueError("Crossed target does not match scored plans")
            if credit == WHOLE_CREDIT:
                from scrl.plan import digest
                source = digest(group_rows[0].original_prompt)
                replay = groups[index]['replay_by_candidate']
                if (len(replay) != k or any(s.whole_prompt_sha256 != source or not s.whole_positions
                                            for s in values) or any(
                        replay[int(r.candidate_id[1:])]['target'] != r.target_segment for r in group_rows)):
                    raise ValueError("Whole-plan traces do not match the prompt or replay targets")
        samples.extend(values)
        advantages.extend(scores[c] for c in candidates)
        targets.extend([None if credit == WHOLE_CREDIT else groups[index]['target']] * k)
    return samples, advantages, targets


def main():
    os.umask(0)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    from scrl.config import load_run
    cfg=load_run(args.config)
    planner_only = validate_training_mode(cfg['training'].get('mode')) == 'planner-only'
    planner_credit = cfg['training'].get('planner_credit','candidate-v1')
    run=Path(cfg['run']);token=cfg['token']
    executors=[Peer(r['address'],r['port'],token) for r in cfg['executors']]
    planners=[Peer(n['address'],n.get('port',cfg['ports']['planner']),token)
              for n in cfg.get('planners',cfg['nodes'])]
    evaluations=[Peer(n['address'],n.get('port',cfg['ports']['evaluation']),token)
                 for n in cfg.get('evaluations',cfg['nodes'])]
    fingerprint=training_fingerprint(cfg)
    dispatcher = ([Peer(cfg['dispatcher']['address'], cfg['dispatcher']['port'], token)]
                  if cfg.get('dispatcher') else [])
    total_prompts=len(cfg['prompts']);batch_size=cfg['batch_prompts']
    deadline=float('inf') if cfg.get('full_dataset') else cfg['created_unix']+cfg['wall_seconds']
    cursor=step=epoch=0
    repeat=cfg.get('repeat_dataset',False)
    actor_device=cfg.get('actor_device','cuda:0')
    torch.cuda.set_device(actor_device)
    policy=None;recent=[];last_saved=-1
    migration_metadata=cfg.get('migration_id')
    checkpoint_writer=SnapshotWriter()
    checkpoints=run/'checkpoints';checkpoints.mkdir(exist_ok=True)
    coordinator=JointCoordinator(PolicyStamp(**cfg['stamp']))
    durable=Path(cfg['durable_run']) if cfg.get('durable_run') else None
    def status(state,**values):
        from scrl.eventlog import event
        if state != 'collecting' or not values.get('completed_jobs'):
            event('coordinator', state, run=run, step=step)
        data={'state':state,'cursor':cursor,'total_prompts':total_prompts,
            'completed_rounds':step,'planned_rounds':None if repeat else step+math.ceil((total_prompts-cursor)/batch_size),
            'epoch':epoch+1,'completed_epochs':epoch+int(cursor==total_prompts),
            'prompts_processed':epoch*total_prompts+cursor,'repeat_dataset':repeat,
            'rounds_per_epoch':math.ceil(total_prompts/batch_size),
            'checked_unix':time.time(),'stamp':asdict(coordinator.stamp),**values}
        if durable:
            from scrl.runtime.transport import optional_write
            atomic_json(durable/'status.json',data)
            optional_write(atomic_json,run/'status.json',data)
        else:atomic_json(run/'status.json',data)
    def payload():
        return freeze_cpu({'planner':cpu_state(policy.model),'planner_optimizer':policy.optimizer.state_dict(),
                'cursor':cursor,'epoch':epoch,'job_index':step,'config_fingerprint':fingerprint,
                'torch_rng':torch.get_rng_state(),'cuda_rng':[torch.cuda.get_rng_state(actor_device)],
                'reward_migration':migration_metadata})
    def checkpoint_now(staging=None):
        nonlocal last_saved
        checkpoint_writer.wait()
        if planner_only:
            frozen_payload = payload()
            frozen_payload['policy_version'] = policy.version
            checkpoint_writer.submit(lambda: publish_planner_checkpoint(
                checkpoints, frozen_payload['job_index'], frozen_payload, cfg['initial_adapter_hashes']))
            last_saved=step
            return str(checkpoints/f'checkpoint-{step:05d}')
        if staging is None:
            staging=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=checkpoints))
            staging.chmod(0o777)
            executors[0].call('save',checkpoint=str(staging/'executor.pt'))
        frozen_coordinator=JointCoordinator.from_state_dict(coordinator.state_dict())
        frozen_payload=payload()
        path=checkpoints/f'checkpoint-{step:05d}'
        def publish():
            tick=time.monotonic()
            wait_executor(staging)
            published=publish_split_checkpoint(staging,path,frozen_coordinator,frozen_payload,
                keep_latest_only=cfg.get('keep_checkpoints',1)==1,
                keep_every=cfg.get('checkpoint_keep_every',0))
            atomic_json(published/'save_timings.json',{'publish_seconds':time.monotonic()-tick,
                'published_unix':time.time(), 'round':frozen_payload['job_index']})
            return str(published)
        checkpoint_writer.submit(publish)
        last_saved=step
        return str(path)
    try:
        torch.set_num_threads(2)
        status('loading')
        assets=cfg['assets']
        if planner_only:
            from scrl.models.planner import PlannerPolicy as actor_class
        else:
            actor_class=VerlPlannerPolicy
        policy=actor_class(assets['planner_model'],assets['planner_adapter'],device=actor_device,
                           limits=PlanLimits(**cfg['plan']),learning_rate=cfg['training']['planner_learning_rate'])
        if cfg.get('init_from'):
            from scrl.models.adapters import restore_adapter
            init_state=torch.load(Path(cfg['init_from'])/'planner.pt',map_location='cpu',weights_only=True,mmap=True)
            restore_adapter(policy.model,init_state['planner'])
            del init_state
        if cfg.get('resume'):
            from peft import set_peft_model_state_dict
            if planner_only:
                from scrl.config import file_hash,read
                cp=Path(cfg['resume'])
                state=read(cp/'state.json')
                if state['format']!='planner-only-v1' or file_hash(cp/'planner.pt')!=state['files']['planner.pt']:
                    raise ValueError('Invalid planner-only resume payload')
                saved=torch.load(cp/'planner.pt',map_location='cpu',weights_only=True,mmap=True)
                coordinator=JointCoordinator(PolicyStamp(**{**cfg['stamp'],**state['stamp']}))
            else:
                coordinator,saved=load_checkpoint(cfg['resume'])
            migration_metadata=saved.get('reward_migration',migration_metadata)
            if cfg.get('quote_transition'):
                from dataclasses import replace
                from scrl.config import read as read_config
                from scrl.rl.identity import quote_anchor_transition
                cp=Path(cfg['resume'])
                parent=read_config(cp.parent.parent/'config.json')
                verified=quote_anchor_transition(parent,cfg)
                if (verified!=cfg['quote_transition'] or training_fingerprint(parent)!=saved['config_fingerprint']
                        or coordinator.stamp.reward!=verified['old_reward']):
                    raise ValueError('Invalid quote migration receipt')
                coordinator.stamp=replace(coordinator.stamp,reward=verified['new_reward'])
                migration_metadata=dict(verified,checkpoint=str(cp),step=saved['job_index'])
            elif cfg.get('rating_transition'):
                from scrl.rl.identity import rating_format_transition
                verified=rating_format_transition(read(cp.parent.parent/'config.json'),cfg,saved['config_fingerprint'])
                if verified!=cfg['rating_transition']:raise ValueError('Judge transition receipt mismatch')
                migration_metadata=dict(verified,checkpoint=str(cp),step=saved['job_index'])
            else:
                verify_campaign_resume(cfg, saved['config_fingerprint'], cfg['resume'])
            if cfg.get('reward_protocol') and any(getattr(coordinator.stamp, key) != cfg['stamp'][key]
                    for key in ('reward', 'sampler', 'planner_reference', 'executor_reference')):
                raise ValueError('Resume policy/reward stamp differs from migrated checkpoint')
            loaded=set_peft_model_state_dict(policy.model,saved['planner'],adapter_name='default')
            if loaded.unexpected_keys or state_hash(cpu_state(policy.model)) != state_hash(saved['planner']):
                raise ValueError('Planner parameters changed during restoration')
            policy.optimizer.load_state_dict(saved['planner_optimizer']);policy.version=coordinator.stamp.planner
            cursor,step=saved['cursor'],saved['job_index'];epoch=saved.get('epoch',0);last_saved=step
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'][0],device=actor_device)
            del saved
        initial_step = step
        with ThreadPoolExecutor(max_workers=len(executors)+len(planners)+len(evaluations)+len(dispatcher)) as pool:
            for f in as_completed([pool.submit(peer.ready,failure_path=run/'FAIL.json') for peer in executors+planners+evaluations+dispatcher]):f.result()
        status('ready')
        control_dirs = [run] + ([Path(cfg['cycle_window'])] if cfg.get('cycle_window') else [])
        while (repeat or cursor<total_prompts) and time.time()<deadline and not any(
                (directory/name).exists() for directory in control_dirs for name in ('STOP','DRAIN')):
            checkpoint_writer.check()
            if (run/'FAIL.json').exists():raise RuntimeError('Worker failure recorded before next update')
            if durable:
                from scrl.runtime.transport import wait_for_space
                if not wait_for_space(run,status):
                    if (run/'FAIL.json').exists():raise RuntimeError('Worker failure while waiting for storage')
                    break
            cursor,epoch=advance_dataset(cursor,epoch,total_prompts,repeat)
            remaining_time=deadline-time.time()
            if recent and remaining_time < max(x['seconds'] for x in recent[-2:])*1.15:
                break
            count=min(batch_size,total_prompts-cursor)
            k,s=cfg.get('fixed_sampling',[2,2])
            if not cfg.get('fixed_sampling') and len(recent)>=2:
                rate=sum(x['videos'] for x in recent[-3:])/sum(x['seconds'] for x in recent[-3:])
                budget=rate*remaining_time*.85/(total_prompts-cursor)
                for candidate,repeats in cfg['adaptive_sampling']:
                    if candidate*repeats<=budget:k,s=candidate,repeats
            batch=run/'batches'/f'{step:05d}';batch.mkdir(parents=True,exist_ok=False)
            started=time.monotonic();atomic_json(batch/'started.json',{'unix':time.time()})
            with ThreadPoolExecutor(max_workers=len(executors)) as pool:
                versions=list(pool.map(lambda peer:peer.call('begin_batch'),executors))
            if any(v['version']!=coordinator.stamp.executor for v in versions):raise ValueError('Replica version mismatch')
            snapshot=batch/'planner_snapshot.pt';torch.save(cpu_state(policy.model),snapshot)
            indices=list(range(cursor,cursor+count));jobs_expected=[f'b{step}/p{i}/c{c}' for i in indices for c in range(k)]
            stamp=coordinator.begin(f'joint-{step:05d}',jobs_expected)
            groups,rows,assignments,expected_logical=collect_batch(
                cfg=cfg,batch=batch,snapshot=snapshot,step=step,indices=indices,epoch=epoch,
                total_prompts=total_prompts,count=count,k=k,s=s,stamp=stamp,coordinator=coordinator,
                executors=executors,planners=planners,status=status,jobs_expected=jobs_expected)
            samples,advantages,targets=planner_update_batch(
                groups,rows,indices,seed=cfg['seed'],epoch=epoch,total_prompts=total_prompts,
                k=k,s=s,stamp=stamp,credit=planner_credit)
            collection=time.monotonic()-started;status('updating')
            save=((step+1)%cfg['checkpoint_interval']==0 or
                  (not repeat and cursor+count==total_prompts))
            if save:checkpoint_writer.wait()
            staging=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=checkpoints)) if save and not planner_only else None
            if staging:staging.chmod(0o777)
            tick=time.monotonic()
            def update(*_):
                def planner_update():
                    torch.cuda.set_device(actor_device)
                    signals = (v for a in advantages for v in a.values()) if planner_credit == CROSSED_CREDIT else advantages
                    if all(abs(x)<=1e-8 for x in signals):return {'updated':False,'reason':'equal_rewards'}
                    return policy.update(samples,advantages,targets,clip_epsilon=cfg['training']['clip_epsilon'],kl_beta=cfg['training']['kl_beta'])
                with ThreadPoolExecutor(max_workers=len(executors)+1) as pool:
                    pf=pool.submit(planner_update)
                    fs=[pool.submit(peer.call,'finalize',expected_jobs=len(assignments[rank]),
                        checkpoint=str(staging/'executor.pt') if staging else None) for rank,peer in enumerate(executors)]
                    er=[f.result() for f in fs];pr=pf.result()
                if len({(r['updated'],r['replica_sha256']) for r in er})!=1:raise RuntimeError('Executor replicas diverged')
                if any(r['replay_max_error'] > 1e-5 for r in er):
                    raise RuntimeError('Executor replay consistency check failed')
                return {'planner':pr,'executor':{'updated':er[0]['updated'],'ranks':er}}
            updates=coordinator.commit(update)
            if policy.version!=coordinator.stamp.planner or any(
                    r['version']!=coordinator.stamp.executor for r in updates['executor']['ranks']):
                coordinator.failed=True
                raise RuntimeError('Committed policy versions differ from workers')
            update_seconds=time.monotonic()-tick;cursor+=count;step+=1
            if durable:
                state=payload();state['policy_version']=policy.version
                publish_planner_checkpoint(durable/'checkpoints',step,state,cfg['initial_adapter_hashes'])
                del state
            checkpoint=checkpoint_now(staging) if save else None
            report={'state':'passed','round':step,'cursor':cursor,'prompts':count,'videos':len(rows),
                'physical_execution_jobs':len(expected_logical),'logical_execution_jobs':count*k,
                'physical_videos':len(expected_logical)*s,
                'epoch':epoch+1,'prompts_processed':epoch*total_prompts+cursor,
                'plans_per_prompt':k,'seeds_per_plan':s,'planner_credit':planner_credit,'seconds':time.monotonic()-started,
                'collection_seconds':collection,'update_seconds':update_seconds,'finished_unix':time.time(),
                'stamp':asdict(coordinator.stamp),'updates':updates,'checkpoint':checkpoint,
                'checkpoint_async':bool(checkpoint),'dispatcher':dispatcher[0].call('health') if dispatcher else None,
                'rollouts':[asdict(r) for r in rows]}
            if planner_credit == CROSSED_CREDIT:
                report['planner_field_advantages'] = {
                    f'p{index}': {f'c{c}': advantages[i*k+c] for c in range(k)}
                    for i, index in enumerate(indices)}
            if durable:atomic_json(durable/'batches'/batch.name/'report.json',report)
            atomic_json(batch/'report.json',report);recent.append(report)
            recent=recent[-3:]
            # Original IDs, plans, rewards and timings stay in JSON; large sampling tensors are consumed.
            for path in batch.glob('*.pt'):path.unlink()
            status('running',videos_per_hour=len(rows)*3600/report['seconds'],
                   projected_remaining_hours=(total_prompts-cursor)*report['seconds']/count/3600)
            if cfg.get('validation_round_limit') and step - initial_step >= cfg['validation_round_limit']:
                (run/'DRAIN').write_text('Completed the configured validation rounds.\n')
        if step and last_saved!=step:checkpoint_now()
        checkpoint_writer.wait()
        status('drained' if (run/'DRAIN').exists() else 'complete' if cursor==total_prompts and not repeat else 'budget_complete',
               remaining_prompts=total_prompts-cursor,latest_checkpoint=str(checkpoints/'latest'))
    except BaseException as error:
        status('failed',error=repr(error),traceback=traceback.format_exc())
        record_failure(run,{'role':'coordinator','error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        try:
            checkpoint_writer.close()
        finally:
            if policy is not None and hasattr(policy,'close'):policy.close()
            with ThreadPoolExecutor(max_workers=len(executors)+len(planners)+len(evaluations)) as pool:
                def stop(peer):
                    try:peer.call('stop')
                    except Exception:pass
                list(pool.map(stop,executors+planners+evaluations+dispatcher))


if __name__=='__main__':main()
