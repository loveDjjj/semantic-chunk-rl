"""Collect version-aligned candidate plans and match repeated execution seeds."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import copy
import json
import queue
import threading
import time
from scrl.plan import FinitePlan, PlanLimits, PolicyStamp, digest
from scrl.rl.objectives import ScoredRollout, WHOLE_CREDIT, joint_advantages
from scrl.runtime.launch import WorkQueue, prompt_seed
from scrl.runtime.transport import atomic_json


class GroupCollector:
    def __init__(self, executor, scorer, composer, stamp: PolicyStamp, progress=None):
        self.executor, self.scorer, self.composer, self.stamp = executor, scorer, composer, stamp
        self.progress = progress or (lambda **values: None)
        if executor.version != stamp.executor or composer.protocol != stamp.reward:
            raise ValueError("Collector backend identity mismatch")
        self.timings = {}

    def joint(self, prompt, group_id, candidates, seeds, prefix, target_segment):
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or not prompt.strip() or len(candidates) < 2:
            raise ValueError("Joint groups need multiple plans and multiple unique seeds per plan")
        if target_segment != len(prefix.segments):
            raise ValueError("A GRPO group must fork immediately after its shared prefix")
        if self.executor.version != self.stamp.executor:
            raise ValueError("Executor changed after this collector was created")
        started = time.monotonic()
        rows, artifacts, pending = [], {}, []
        generation_seconds = backpressure_seconds = 0.
        try:
            for candidate_id, plan in candidates.items():
                for seed in seeds:
                    self.progress(state="generating", candidate=candidate_id, seed=seed,
                                  submitted=len(pending), total=len(candidates) * len(seeds))
                    tick = time.monotonic()
                    result = self.executor.rollout(plan, seed, prefix=prefix,
                                                  target_segment=target_segment, capture=True,
                                                  retain_final_cache=False)
                    generation_seconds += time.monotonic() - tick
                    tick = time.monotonic()
                    future = self.scorer.submit(prompt, result)
                    backpressure_seconds += time.monotonic() - tick
                    pending.append((candidate_id, seed, plan, future))
                    artifacts[candidate_id, seed] = result
            tick = time.monotonic()
            self.progress(state="draining_rewards", submitted=len(pending))
            for candidate_id, seed, plan, future in pending:
                score = self.composer.compose(future.result())
                rows.append(ScoredRollout(group_id, candidate_id, seed, prompt,
                            prefix.prefix_id, self.stamp, plan, target_segment,
                            score["planner"], score["executor"]))
            self.timings = {"generation_seconds": generation_seconds,
                            "backpressure_seconds": backpressure_seconds,
                            "reward_tail_seconds": time.monotonic() - tick,
                            "collection_seconds": time.monotonic() - started}
        except BaseException:
            for *_, future in pending:
                future.cancel()
            raise
        advantages = joint_advantages(rows, self.stamp, expected_candidates=list(candidates),
                                      expected_seeds=seeds)
        return rows, artifacts, advantages


def execution_aliases(request):
    primary = request.get('parent_job',request['job_id'])
    aliases=request.get('execution_aliases',[{'job_id':primary,'candidate':request['candidate']}])
    if not isinstance(aliases,list) or not aliases:
        raise ValueError('Empty execution aliases')
    if any(not isinstance(a,dict) or set(a)!={'job_id','candidate'} or
           not all(isinstance(v,str) and v for v in a.values()) for a in aliases):
        raise ValueError('Invalid execution aliases')
    if aliases[0]!={'job_id':primary,'candidate':request['candidate']}:
        raise ValueError('Primary execution alias differs from request')
    if len({a['job_id'] for a in aliases})!=len(aliases) or len({a['candidate'] for a in aliases})!=len(aliases):
        raise ValueError('Duplicate execution alias')
    return aliases


def coalesce_requests(requests):
    groups={};job_ids=set();candidates=set()
    for request in requests:
        if 'execution_aliases' in request:raise ValueError('Already coalesced request')
        pair=(request['group_id'],request['candidate'])
        if request['job_id'] in job_ids or pair in candidates:raise ValueError('Duplicate input candidate')
        job_ids.add(request['job_id']);candidates.add(pair)
        # Every execution-affecting field participates, including prompt/group,
        # complete plan, seed order, shared prefix, target, versions and weights.
        key=digest({k:v for k,v in request.items() if k not in ('job_id','candidate')})
        alias={'job_id':request['job_id'],'candidate':request['candidate']}
        if key not in groups:
            groups[key]=copy.deepcopy(request)
            groups[key]['execution_aliases']=[]
        groups[key]['execution_aliases'].append(alias)
    return list(groups.values())


def expand_scored_rows(records,aliases):
    return [dict(asdict(row),candidate_id=alias['candidate'],plan=row.plan.to_dict())
            for alias in aliases for row in records]


def collect_batch(*, cfg, batch, snapshot, step, indices, epoch, total_prompts, count, k, s,
                  stamp, coordinator, executors, planners, status, jobs_expected):
    limits=PlanLimits(**cfg['plan'])
    available_planners=queue.Queue()
    work=WorkQueue(warmup=min(len(executors),count*k,cfg.get('dispatch_warmup',32)),
                   split_chunks=cfg.get('split_seed_min_chunks',0) if cfg['training']['mode']=='planner-only' else 0,
                   queue_seed_pairs=cfg.get('queue_seed_pairs',False))
    completion_lock=threading.Lock()
    for peer in planners:available_planners.put(peer)
    assignments=[[] for _ in executors];groups={};rows=[];expected_logical={};physical_completed=set()
    status('collecting',batch_prompts=count,plans_per_prompt=k,seeds_per_plan=s,videos=count*k*s)
    streaming=cfg.get('stream_plans',False)
    def make_request(index,candidate,plan,group):
        if 'replay_by_candidate' in group:
            group = group['replay_by_candidate'][candidate]
        return dict(job_id=f'b{step}/p{index}/c{candidate}',group_id=f'p{index}',
            candidate=f'c{candidate}',prompt=cfg['prompts'][index],plan=plan,
            stamp=asdict(stamp),attempt=f'round-{step}',
            seeds=[prompt_seed(cfg['seed'],epoch,index,total_prompts)+100+j for j in range(s)],
            total_videos=count*k*s,overlap_backward=True,prefetch_text=True,
            **{key:group[key] for key in ('target','target_probability','prefix_segments','prefix_seed')})
    def plan_prompt(index):
        peer=available_planners.get()
        folder=batch/'stream'/str(index)
        args=dict(prompt=cfg['prompts'][index],seed=prompt_seed(cfg['seed'],epoch,index,total_prompts),
                  candidates=k,version=stamp.planner,snapshot=str(snapshot),
                  random_focal=cfg['training'].get('planner_credit') != WHOLE_CREDIT,
                  output=str(batch/f'planner-{index}.pt'))
        try:
            if not streaming:return peer.call('plan',**args)
            folder.mkdir(parents=True)
            requests=[];sent=set()
            with ThreadPoolExecutor(max_workers=1) as rpc:
                call=rpc.submit(peer.call,'plan',**args,stream_dir=str(folder))
                for candidate in range(k):
                    path=folder/f'{candidate}.json'
                    while not path.is_file():
                        if call.done():
                            call.result()
                            if not path.is_file():raise RuntimeError('Candidate publication missing')
                            break
                        time.sleep(.10)
                    row=json.loads(path.read_text())
                    requests.append(make_request(index,candidate,row['plan'],row))
                    unique=coalesce_requests(requests) if cfg.get('reuse_identical_executions') else requests
                    for req in unique:
                        if req['job_id'] not in sent:
                            sent.add(req['job_id'])
                            work.put(dict(req,alias_seal=str(folder/'credits.json')))
                group=call.result()
            final=[make_request(index,i,plan,group) for i,plan in enumerate(group['plans'])]
            if final!=requests:raise RuntimeError('Streamed candidate differs from final plan')
            aliases={req['job_id']:req.get('execution_aliases',[{'job_id':req['job_id'],
                      'candidate':req['candidate']}]) for req in unique}
            with completion_lock:
                expected_logical.update({j:[a['job_id'] for a in v] for j,v in aliases.items()})
            atomic_json(folder/'credits.json',aliases)
            return group
        except BaseException:
            if streaming:(folder/'aborted').touch()
            raise
        finally:available_planners.put(peer)
    seen_seeds = {}
    def complete_result(result):
        with completion_lock:
            parent=result.get('parent_job',result['job_id'])
            logical=expected_logical[parent]
            seeds={r['seed'] for r in result['rewards']}
            previous=seen_seeds.setdefault(parent,set())
            if (previous & seeds or len(seeds)!=result['physical_videos'] or
                    result.get('logical_job_ids',[parent])!=logical or len(result['rows'])!=len(logical)*len(seeds)):
                raise ValueError('Executor returned incomplete logical candidate groups')
            previous.update(seeds)
            if len(previous)>s:raise ValueError('Excess seeds for candidate')
            if len(previous)==s:
                for job_id in logical:coordinator.complete(job_id,stamp)
                physical_completed.add(parent)
            for value in result['rows']:
                value['plan']=FinitePlan.from_dict(value['plan'],limits);value['stamp']=PolicyStamp(**value['stamp'])
                rows.append(ScoredRollout(**value))
            with (batch/'jobs.jsonl').open('a') as handle:
                handle.write(json.dumps({key:value for key,value in result.items() if key!='rows'})+'\n')
            status('collecting',completed_jobs=len(coordinator.completed),jobs=len(jobs_expected),
                   completed_execution_jobs=len(physical_completed),
                   submitted_execution_jobs=len(expected_logical),
                   physical_videos_completed=sum(map(len,seen_seeds.values())),
                   logical_videos_completed=len(rows),
                   batch_prompts=count,plans_per_prompt=k,seeds_per_plan=s,videos=count*k*s)
    def execute_rank(rank):
        try:
            while (request:=work.get()) is not None:
                assignments[rank].append(request['job_id'])
                if cfg.get('pipeline_depth',1)>1:
                    result=executors[rank].call('enqueue',**request)
                    for completed in result['completed']:complete_result(completed)
                else:
                    complete_result(executors[rank].call('generate',**request))
            if cfg.get('pipeline_depth',1)>1:
                for completed in executors[rank].call('drain')['completed']:complete_result(completed)
        except BaseException as error:
            work.close(error)
            raise
    with ThreadPoolExecutor(max_workers=len(planners)) as planning,ThreadPoolExecutor(max_workers=len(executors)) as generation:
        futures={planning.submit(plan_prompt,i):i for i in indices}
        jobs=[generation.submit(execute_rank,rank) for rank in range(len(executors))]
        try:
            for future in as_completed(futures):
                index=futures[future];group=future.result();groups[index]=group
                atomic_json(batch/f'planning-{index}.json',group)
                if streaming:continue
                requests=[make_request(index,i,plan,group) for i,plan in enumerate(group['plans'])]
                if cfg.get('reuse_identical_executions',False):requests=coalesce_requests(requests)
                for request in requests:
                    expected_logical[request['job_id']]=[a['job_id'] for a in request.get(
                        'execution_aliases',[{'job_id':request['job_id']}])]
                    work.put(request)
        except BaseException as error:
            work.close(error)
            raise
        else:
            work.close()
        for future in as_completed(jobs):future.result()
    return groups, rows, assignments, expected_logical
