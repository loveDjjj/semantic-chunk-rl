"""Persistent planner, executor and evaluation roles with a single authenticated worker entry point."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import random
import tempfile
import time
import torch
import torch.distributed as dist
import traceback
import uuid
from scrl.models.executor import LongLivePaths, load_longlive
from scrl.models.planner import PlannerPolicy
from scrl.models.scoring.service import VideoRewardService
from scrl.models.segments import SegmentExecutor
from scrl.plan import FinitePlan, PlanLimits, PolicyStamp, digest
from scrl.rl.checkpoint import SnapshotWriter, freeze_cpu, write_executor
from scrl.rl.identity import validate_training_mode
from scrl.rl.objectives import CROSSED_CREDIT, WHOLE_CREDIT, ScoredRollout, executor_advantages, reduce_gradients
from scrl.rl.reward import RewardBundle, RewardComposer, RewardConfig, ScopeScores, UnscorableReward
from scrl.rl.rollout import execution_aliases, expand_scored_rows
from scrl.runtime.transport import Peer, ScoringPeer, atomic_json, record_failure, serve


def cpu_state(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model).items()}


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class EvaluationWorker:
    def __init__(self, cfg, node, device='cuda:0', instance=-1):
        validate_training_mode(cfg['training'].get('mode'))
        self.cfg, self.node = cfg, node
        self.limits = PlanLimits(**cfg['plan'])
        assets = cfg['assets']
        backend = SimpleNamespace(paths=SimpleNamespace(**assets['executor']))
        settings = {k: dict(v, device=device) for k,v in assets['rewards'].items()}
        if cfg.get('run_local_contract_cache') and 'judge' in settings:
            settings['judge']['contract_cache'] = str(Path(cfg['run'])/'prompt_contracts')
        if cfg.get('event_cache_backend') == 'dispatcher-v1':
            endpoint = cfg['dispatcher']
            settings['judge']['event_cache_endpoint'] = {k: endpoint[k] for k in ('address', 'port')}
        if decisions := cfg.get('reward_protocol', {}).get('event_decisions'):
            settings['judge']['event_decisions'] = decisions
            settings['judge']['event_choice_batch_size'] = cfg['reward_protocol'].get('event_choice_batch_size', 1)
        aux=next((s.get('vae_device') for s in cfg.get('evaluations',[]) if s.get('rank')==instance),None)
        self.scorer = VideoRewardService(backend, settings,
            Path(cfg['run'])/'services'/cfg.get('runtime_session','initial')/f'node-{node}-instance-{instance}',
            cfg['stamp']['reward'], options={'vae_device': 'cuda:1' if aux else device,
                                           'max_inflight': cfg.get('services',{}).get('max_inflight',2),
                                           'judge_port': cfg.get('services',{}).get('judge_ports',{}).get(str(instance)),
                                           'global_only': cfg['reward'].get('aggregation') == 'global-sq-v1',
                                           'scratch_directory': '/dev/shm'})
        if cfg.get('services',{}).get('parallel_judge_scopes'):
            from scrl.models.scoring.http_judge import parallel_class
            previous=self.scorer.workers['judge']
            self.scorer.workers['judge']=parallel_class()(settings['judge'],cfg['services']['judge_ports'][str(instance)],
                                                        cfg['services']['max_inflight'])
            previous.close()
        if identity:=cfg.get('reward_protocol',{}).get('rating_format'):
            from scrl.models.scoring.protocol import install
            install(self.scorer.workers['judge'],identity)
        if cfg.get('services',{}).get('trace_stages'):
            from scrl.models.scoring.service import trace_workers
            trace_workers(self.scorer)

    def __call__(self, op, request):
        if op == 'health':
            return {'node': self.node, 'role': 'evaluation'}
        if op == 'stop':
            return {}
        if op != 'evaluate':
            raise ValueError(op)
        path = Path(request['latent_path']).resolve()
        shared = (Path(self.cfg['run'])/'transport').resolve()
        if not (str(path).startswith('/dev/shm/scrl-cluster-') or
                (self.cfg.get('shared_latent_transport') and path.is_relative_to(shared))):
            raise ValueError('Evaluation tensor must belong to this run')
        plan = FinitePlan.from_dict(request['plan'], self.limits)
        result = SimpleNamespace(plan=plan, target_segment=request['target'],
            final=SimpleNamespace(latents=path, prefix_id=digest({'final':request['final_id'],
                                                                 'attempt':request['attempt']})))
        return asdict(self.scorer(request['prompt'], result))

    def close(self):
        self.scorer.close()


class PlanningWorker:
    def __init__(self, cfg, node, device='cuda:0'):
        validate_training_mode(cfg['training'].get('mode'))
        self.cfg, self.node = cfg, node
        assets = cfg['assets']
        self.policy = PlannerPolicy(assets['planner_model'], assets['planner_adapter'], device=device,
                                    limits=PlanLimits(**cfg['plan']))
        self.snapshot = None

    def __call__(self, op, request):
        if op == 'health':
            return {'node': self.node, 'role': 'planner'}
        if op == 'stop':
            return {}
        if op != 'plan':
            raise ValueError(op)
        if not request['snapshot'] or type(request['version']) is not int or request['version'] < 0:
            raise ValueError('Invalid planner snapshot identity')
        from scrl.models.adapters import restore_adapter
        if self.snapshot != request['snapshot']:
            state = torch.load(request['snapshot'], weights_only=True, map_location='cpu')
            # A backend load may fail after partially copying weights. Do not
            # serve the previous path from cache until a full reload succeeds.
            self.snapshot = None
            restore_adapter(self.policy.model, state)
            self.policy.version = request['version']
            self.snapshot = request['snapshot']
        elif self.policy.version != request['version']:
            raise ValueError('Cached planner snapshot cannot be relabeled with a different version')
        started = time.monotonic()
        target, prefix_ids, prefix_segments, probability = 0, None, [], 1.
        crossed = self.cfg['training'].get('planner_credit') == CROSSED_CREDIT
        whole = self.cfg['training'].get('planner_credit') == WHOLE_CREDIT
        if whole and (request.get('random_focal') or request['candidates'] != 4):
            raise ValueError('Whole-plan sampling requires four unprefixed planner draws')
        if crossed and (not request.get('random_focal') or request['candidates'] != 4):
            raise ValueError('Crossed planning needs a focal anchor and four cells')
        options = dict(max_new_tokens=self.cfg['sampling']['max_new_tokens'],
                       truncate_at_limit=self.cfg['sampling'].get('truncate_at_limit', False))
        if whole:
            options['whole_plan'] = True
        if request.get('random_focal', False):
            anchor, response, decisions = self.policy.sample(request['prompt'], request['seed']+9000,
                                                            anchor_only=True, **options)
            target = random.Random(request['seed']).randrange(len(anchor.segments))
            probability = 1./len(anchor.segments)
            if not crossed:
                prefix_ids = response[:,:decisions[target].caption[0]]
            prefix_segments = anchor.to_dict()['segments'][:target]
        if crossed:
            draws = self.policy.sample_crossed(request['prompt'], anchor, target, request['seed'],
                                               max_new_tokens=options['max_new_tokens'])
            if len(draws) != 4:
                raise ValueError('Incomplete crossed plan group')
        else:
            draws = (self.policy.sample(request['prompt'], request['seed'] + i,
                                        text_prefix_ids=prefix_ids, **options)
                     for i in range(request['candidates']))
        samples, replay = [], []
        for i, sample in enumerate(draws):
            samples.append(sample)
            if whole:
                # One uniform focal replay per plan; the entire video uses its own seed.
                rng = random.Random(request['seed'] + 8000)
                target = rng.randrange(len(sample.plan.segments))
                probability = 1. / len(sample.plan.segments)
            row = dict(target=target,target_probability=probability,prefix_segments=prefix_segments,
                       prefix_seed=request['seed']+8000)
            replay.append(row)
            if request.get('stream_dir'):
                atomic_json(Path(request['stream_dir'])/f'{i}.json', dict(plan=sample.plan.to_dict(), **row))
        path = Path(request['output'])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, path)
        return {**({'replay_by_candidate':replay} if whole else {}),
                'samples':str(path), 'plans':[s.plan.to_dict() for s in samples],
                'seconds':time.monotonic()-started, 'version':self.policy.version,
                'target':target,'target_probability':probability,'prefix_segments':prefix_segments,
                'prefix_seed':request['seed']+8000}

    def close(self):
        pass


class ExecutionWorker:
    limits = PlanLimits()

    def __init__(self, cfg, rank, node, device):
        self.frozen = validate_training_mode(cfg['training'].get('mode')) == 'planner-only'
        self.cfg, self.rank, self.node, self.device = cfg, rank, node, device
        self.limits = PlanLimits(**cfg['plan'])
        torch.cuda.set_device(device)
        assets = cfg['assets']
        if self.frozen and (assets['executor'].get('policy_adapter') or cfg.get('init_from')):
            raise ValueError('Planner-only training requires native LongLive, not a policy checkpoint')
        self.backend = load_longlive(LongLivePaths(**assets['executor']), device)
        self.backend.frozen_text_copy = cfg.get('frozen_text_copy', False)
        self.backend.private_gpu_history = self.frozen and cfg.get('private_gpu_history',False)
        self.optimizer = None if self.frozen else self.backend.enable_training(cfg['training']['executor_learning_rate'])
        self.executor = SegmentExecutor(self.backend)
        if cfg.get('init_from'):
            from scrl.models.adapters import restore_adapter
            payload=torch.load(Path(cfg['init_from'])/'executor.pt',map_location='cpu',weights_only=True,mmap=True)
            restore_adapter(self.backend.pipe.generator.model,payload['executor'])
            del payload
        if cfg.get('resume') and not self.frozen:
            from peft import set_peft_model_state_dict
            from scrl.rl.checkpoint import file_sha256
            checkpoint=Path(cfg['resume'])
            state=json.loads((checkpoint/'state.json').read_text())
            if file_sha256(checkpoint/'executor.pt')!=state['files']['executor.pt']:
                raise ValueError('Executor resume checksum mismatch')
            payload=torch.load(checkpoint/'executor.pt',map_location='cpu',weights_only=True,mmap=True)
            loaded=set_peft_model_state_dict(self.backend.pipe.generator.model,payload['executor'],adapter_name='default')
            if loaded.unexpected_keys:
                raise ValueError('Unexpected Executor checkpoint parameters')
            if state_hash(cpu_state(self.backend.pipe.generator.model)) != state_hash(payload['executor']):
                raise ValueError('Executor parameters changed during restoration')
            self.optimizer.load_state_dict(payload['executor_optimizer'])
            self.backend.version=self.executor.version=state['stamp']['executor']
        self.composer = RewardComposer(RewardConfig(**cfg['reward']), cfg['stamp']['reward'])
        self.score_attempts = cfg['training'].get('score_attempts', 1)
        self.review_dir = Path(cfg['run'])/'workers'/f'executor-{node}-{rank}'/'score_reviews'
        self.evaluators = [Peer(s['address'],s['port'],cfg['token']) for s in cfg.get('evaluations',[])]
        if cfg.get('dispatcher'):
            endpoint = cfg['dispatcher']
            self.evaluators = [ScoringPeer(endpoint['address'], endpoint['port'], cfg['token'])]
        if not self.evaluators:
            self.evaluators = [Peer('127.0.0.1', cfg['ports']['evaluation'], cfg['token'])]
        self.evaluation_index = rank
        self.inflight = {}
        self.snapshot_writer = SnapshotWriter()
        self.pipeline_depth = cfg.get('pipeline_depth',1)
        self.score_admission = None
        if self.frozen and cfg.get('frozen_score_queue_videos'):
            if (not cfg.get('dispatcher') or self.pipeline_depth != 2 or
                    cfg['frozen_score_queue_videos'] < cfg['sampling']['executions_per_candidate']):
                raise ValueError('Frozen score queue requires a dispatcher and sufficient capacity')
            self.score_admission = self.evaluators[0]
        self.pipeline_cache_limit = int(cfg.get('pipeline_cache_gib',8) * 2**30)
        if self.pipeline_depth not in (1,2):raise ValueError('Pipeline depth must be one or two')
        self.score_pool = ThreadPoolExecutor(
            max_workers=self.pipeline_depth*cfg['sampling']['executions_per_candidate'],
            thread_name_prefix='score-rpc')
        self.transport = Path(cfg['run'])/'transport' if cfg.get('shared_latent_transport') else Path('/dev/shm')
        self.transport.mkdir(parents=True,exist_ok=True)
        if not self.frozen:
            dist.init_process_group('nccl', init_method=cfg['rendezvous'], rank=rank,
                                    world_size=cfg['executor_count'], timeout=timedelta(minutes=20))
            probe = torch.ones(1,device=device)
            dist.all_reduce(probe)
            if float(probe) != cfg['executor_count']:
                raise RuntimeError('Executor collective failed')
            self.initial = cpu_state(self.backend.pipe.generator.model)
        self.pending, self.completed = {}, set()
        self.timing = {'generation':0.,'score_wait':0.,'backward':0.,'replay_max_error':0.}
        if self.optimizer is not None:self.optimizer.zero_grad(set_to_none=True)

    def begin_batch(self):
        self.snapshot_writer.check()
        if self.pending or self.inflight:
            raise RuntimeError('Previous gradient jobs remain')
        self.completed.clear()
        self.backend.conditions.clear()
        if self.optimizer is not None:self.optimizer.zero_grad(set_to_none=True)
        self.timing = {'generation':0.,'score_wait':0.,'backward':0.,'replay_max_error':0.,'prefix':0.,'text':0.}
        torch.cuda.reset_peak_memory_stats()
        return {'rank':self.rank,'version':self.backend.version}

    def reset(self):
        if self.frozen:
            self.backend.conditions.clear()
            return {'rank':self.rank,'hash':self.cfg['stamp']['executor_reference']}
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.backend.pipe.generator.model, self.initial, adapter_name='default')
        self.optimizer = self.backend.enable_training(self.cfg['training']['executor_learning_rate'])
        self.backend.version = self.executor.version = 0
        self.backend.conditions.clear()
        self.pending.clear()
        self.completed.clear()
        self.timing = {'generation':0.,'score_wait':0.,'backward':0.,'replay_max_error':0.}
        return {'rank':self.rank, 'hash':state_hash(self.initial)}

    def accumulate(self, job_id):
        if job_id in self.completed or job_id not in self.pending:
            raise ValueError('Duplicate or missing gradient job')
        records, rollouts, advantages, weight = self.pending.pop(job_id)
        start = time.monotonic()
        for seed, rollout in rollouts.items():
            advantage = advantages[seed]
            if abs(advantage) <= 1e-8:
                continue
            metrics = self.backend.backward(rollout, advantage,
                kl_beta=self.cfg['training']['kl_beta'], weight=weight)
            self.timing['replay_max_error'] = max(self.timing['replay_max_error'], metrics['replay_max_error'])
        self.timing['backward'] += time.monotonic()-start
        self.completed.add(job_id)

    def _start_job(self, request):
        execution_aliases(request)
        job_id = request['job_id']
        if job_id in self.pending or job_id in self.completed or job_id in self.inflight:
            raise ValueError('Repeated Executor job')
        before = dict(self.timing)
        job_started, unix_started = time.monotonic(), time.time()
        stamp = PolicyStamp(**request['stamp'])
        if stamp.executor != self.backend.version:
            raise ValueError('Stale Executor request')
        plan = FinitePlan.from_dict(request['plan'], self.limits)
        tick = time.monotonic()
        if request['prefetch_text']:
            self.backend.encode_many([s.caption for s in plan.segments])
        self.timing['text'] = self.timing.get('text',0.) + time.monotonic()-tick
        from scrl.plan import Segment
        # Make independently rebuilt common prefixes independent of a worker's previous job length.
        self.backend.pipe.generator.seq_len = 0
        target = request.get('target',0)
        prefix_segments = tuple(Segment(**s) for s in request.get('prefix_segments',[]))
        if getattr(self,'cfg',{}).get('training',{}).get('planner_credit') == WHOLE_CREDIT:
            if prefix_segments:raise ValueError('Whole-plan generation must start at frame zero')
        elif prefix_segments != plan.segments[:target]:raise ValueError('Planner prefix mismatch')
        tick=time.monotonic()
        prefix = self.executor.build_prefix(prefix_segments,request.get('prefix_seed',0))
        self.timing['prefix'] = self.timing.get('prefix',0.) + time.monotonic()-tick
        rollouts, futures = {}, {}
        scratch = tempfile.TemporaryDirectory(prefix='scrl-cluster-',dir=self.transport)
        Path(scratch.name).chmod(0o777)
        admission = getattr(self,'score_admission',None)
        reserved = {seed:f'{job_id}/{seed}' for seed in request['seeds']} if admission else {}
        admission_start = time.monotonic()
        if admission:
            admission.call('reserve_videos', keys=list(reserved.values()))
        admission_wait = time.monotonic() - admission_start if admission else 0.
        try:
            for seed in request['seeds']:
                tick = time.monotonic()
                rollout = self.executor.rollout(plan, seed, prefix=prefix, target_segment=target, capture=not self.frozen,
                                               retain_final_cache=not self.frozen and not request['overlap_backward'])
                self.timing['generation'] += time.monotonic()-tick
                if not self.frozen:rollouts[seed] = rollout
                path = Path(scratch.name)/f'{seed}.pt'
                torch.save(rollout.final.latents,path)
                evaluator = self.evaluators[self.evaluation_index % len(self.evaluators)]
                self.evaluation_index += 1
                payload = dict(latent_path=str(path), plan=plan.to_dict(),prompt=request['prompt'],
                    target=target, final_id=rollout.final.prefix_id,attempt=request['attempt'],
                    job_id=job_id, seed=seed)
                futures[seed] = self.score_pool.submit(self._score_file,evaluator,payload,
                    admission,reserved.get(seed), self.composer.compose,
                    getattr(self,'score_attempts',1), getattr(self,'review_dir',None))
                reserved.pop(seed,None)
                del rollout
        except BaseException:
            for key in reserved.values():
                admission.call('release_video',key=key)
            for future in futures.values():
                try:future.result()
                except Exception:pass
            scratch.cleanup()
            raise
        self.inflight[job_id] = dict(request=request,plan=plan,prefix_id=prefix.prefix_id,stamp=stamp,
            rollouts=rollouts,futures=futures,scratch=scratch,started=job_started,started_unix=unix_started,
            generation_finished_unix=time.time(), generator_seq_len=self.backend.pipe.generator.seq_len,
            stages=dict({key:self.timing.get(key,0.)-before.get(key,0.) for key in ['text','prefix','generation']},
                        admission_wait=admission_wait))

    @staticmethod
    def _score_file(evaluator, payload, admission, key, compose=None, attempts=1, directory=None):
        identity = {k:payload.get(k) for k in ('job_id','seed','attempt','final_id')}
        history = []
        receipt = directory / (digest(identity)+'.json') if directory is not None else None
        def record():
            if receipt is not None:
                atomic_json(receipt, dict(identity, attempts=history))
        try:
            if type(attempts) is not int or attempts not in (1,2):
                raise ValueError('score_attempts must be 1 or 2')
            for number in range(1, attempts+1):
                request = dict(payload, request_id=uuid.uuid4().hex)
                if number > 1:
                    # A definitive rejection (including invalid composed metrics) needs fresh evidence.
                    request['attempt'] = f"{payload['attempt']}:review-{number}"
                row = dict(number=number, request_id=request['request_id'], attempt=request.get('attempt'))
                history.append(row)
                try:
                    data = evaluator.call('evaluate', **request)
                    scores = compose(RewardBundle(ScopeScores(**data['global_scores']),
                        ScopeScores(**data['segment_scores']),data['protocol'],data['boundary_quality']))
                except UnscorableReward as exc:
                    row.update(state='unscorable',error=str(exc))
                    record()
                    if number == attempts:
                        raise UnscorableReward(f"Job {identity['job_id']} seed {identity['seed']} "
                            f"unscorable after {attempts} attempt(s): {exc}; receipt={receipt}") from exc
                except Exception as exc:
                    row.update(state='failed',error=repr(exc))
                    record()
                    raise
                else:
                    row.update(state='complete')
                    if number > 1:
                        record()
                    Path(payload['latent_path']).unlink(missing_ok=True)
                    return data, scores, history
        finally:
            if admission:
                admission.call('release_video',key=key)

    def _finish_job(self, job_id, *, pre_wait=0.):
        item = self.inflight[job_id]
        request, plan, stamp = item['request'], item['plan'], item['stamp']
        records, error, reward_details = [], None, []
        tick = time.monotonic()
        # Drain every RPC before releasing its latent file, including failures.
        for seed,future in item['futures'].items():
            try:
                data, scores, reviews = future.result()
                reward_details.append({'seed': seed, 'components': data, 'returns': scores,
                                       'score_attempts': reviews})
                records.append(ScoredRollout(request['group_id'],request['candidate'],seed,
                    request['prompt'],item['prefix_id'],stamp,plan,request.get('target',0),
                    scores['planner'],scores['executor']))
            except Exception as caught:
                if error is None or isinstance(error, UnscorableReward):
                    error = caught
        waited = time.monotonic()-tick+pre_wait
        self.timing['score_wait'] += waited
        item['scratch'].cleanup()
        del self.inflight[job_id]
        if error is not None:raise error
        target = request.get('target',0)
        if 'alias_seal' in request:
            seal = Path(request['alias_seal'])
            deadline = time.monotonic()+300
            while not seal.is_file():
                if time.monotonic()>deadline or (seal.parent/'aborted').exists():
                    raise RuntimeError('Missing final candidate credits for '+job_id)
                time.sleep(.05)
            request['execution_aliases'] = json.loads(seal.read_text())[request.get('parent_job',job_id)]
        aliases=execution_aliases(request)
        weight = (plan.segments[target].num_chunks / plan.total_chunks / request['total_videos']
                  / request.get('target_probability',1.) * len(aliases))
        if self.frozen:
            self.completed.add(job_id)
        else:
            advantages = executor_advantages(records,stamp,expected_seeds=request['seeds'])
            self.pending[job_id] = (records,item['rollouts'],advantages,weight)
        before_backward = self.timing['backward']
        if not self.frozen and request['overlap_backward']:
            # Deferred replay must see the same padding extent as its own rollout,
            # even when a different-length job generated while rewards were pending.
            self.backend.pipe.generator.seq_len = item['generator_seq_len']
            self.accumulate(job_id)
        return {'rank':self.rank,'rows':expand_scored_rows(records,aliases),
                'logical_job_ids':[a['job_id'] for a in aliases],
                'physical_videos':len(records),'logical_videos':len(records)*len(aliases),
                'rewards': reward_details, 'parent_job':request.get('parent_job',job_id),
                'seconds':time.monotonic()-item['started'],'job_id':job_id,'timing':dict(self.timing),
                'started_unix':item['started_unix'],'generation_finished_unix':item['generation_finished_unix'],
                'finished_unix':time.time(),
                'stages':dict(item['stages'],score_wait=waited,backward=self.timing['backward']-before_backward)}

    def generate(self, request):
        self._start_job(request)
        return self._finish_job(request['job_id'])

    def enqueue(self, request):
        if not request['overlap_backward']:raise ValueError('Pipeline requires immediate gradient accumulation')
        completed = self._drain_ready()
        if getattr(self, 'score_admission', None):
            # Frozen executors retain no rollout/tape or prefix cache after generation.
            # These slots bound outstanding CPU metadata and shared latent files only.
            self._start_job(request)
            return {'completed':completed,'pending':len(self.inflight)}
        while self.inflight and (len(self.inflight) >= self.pipeline_depth or self._cache_budget_exceeded(request)):
            completed.append(self._finish_next())
        self._start_job(request)
        completed.extend(self._drain_ready())
        # Returning grants this rank another pull from the shared work queue.
        # Release a pipeline slot first so a busy rank cannot reserve a third job.
        if len(self.inflight) >= self.pipeline_depth:
            completed.append(self._finish_next())
        return {'completed':completed,'pending':len(self.inflight)}

    def _cache_budget_exceeded(self, request):
        estimate=getattr(self.backend,'prefix_cache_bytes',None)
        if estimate is None:return False
        def size(value):
            frames=sum(s['num_chunks']*3 for s in value.get('prefix_segments',[]))
            return estimate(frames)
        retained=sum(size(item['request']) for item in self.inflight.values())
        return retained+size(request)>self.pipeline_cache_limit

    def _drain_ready(self):
        return [self._finish_job(job_id) for job_id in list(self.inflight)
                if all(f.done() for f in self.inflight[job_id]['futures'].values())]

    def _finish_next(self):
        tick=time.monotonic()
        while self.inflight:
            for job_id,item in self.inflight.items():
                if all(f.done() for f in item['futures'].values()):
                    return self._finish_job(job_id,pre_wait=time.monotonic()-tick)
            # Four-seed groups need all four scores. A later complete group may
            # progress even when an earlier group still has one slow score.
            wait([f for item in self.inflight.values() for f in item['futures'].values() if not f.done()],
                 return_when=FIRST_COMPLETED)
        raise RuntimeError('No group to finish')

    def drain(self):
        completed=[]
        while self.inflight:completed.append(self._finish_next())
        return {'completed':completed}

    def finalize(self, request):
        if self.inflight:raise RuntimeError('Unscored jobs remain before update')
        for job_id in list(self.pending):
            self.accumulate(job_id)
        if len(self.completed) != request['expected_jobs']:
            raise ValueError('Incomplete local gradient group')
        if self.frozen:
            if request.get('checkpoint'):
                raise ValueError('Cannot save an Executor checkpoint in planner-only mode')
            return {'rank':self.rank,'updated':False,'version':0,'gradient_norm':0.,
                    'replica_sha256':self.cfg['stamp']['executor_reference'],'sync_seconds':0.,
                    'completed_jobs':len(self.completed),'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,
                    **self.timing}
        params = [p for p in self.backend.pipe.generator.model.parameters() if p.requires_grad]
        tick = time.monotonic()
        # Losses already use the GLOBAL sample denominator: SUM, never rank-average.
        reduce_gradients(params)
        torch.cuda.synchronize()
        sync = time.monotonic()-tick
        norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        updated = bool(norm > 0)
        if updated:
            self.optimizer.step()
            self.backend.version += 1
            self.executor.version = self.backend.version
        self.optimizer.zero_grad(set_to_none=True)
        state = cpu_state(self.backend.pipe.generator.model)
        checksum = state_hash(state)
        hashes = [None]*self.cfg['executor_count']
        dist.all_gather_object(hashes,checksum)
        if len(set(hashes)) != 1:
            raise RuntimeError('Executor replicas diverged')
        if self.rank == 0 and request.get('checkpoint'):
            payload = freeze_cpu({'executor':state,'executor_optimizer':self.optimizer.state_dict()})
            self.snapshot_writer.submit(write_executor, request['checkpoint'], payload)
        return {'rank':self.rank,'updated':updated,'version':self.backend.version,
                'gradient_norm':float(norm),'replica_sha256':checksum,'sync_seconds':sync,
                'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,
                'completed_jobs':len(self.completed),**self.timing}

    def __call__(self, op, request):
        torch.cuda.set_device(self.device)
        if op == 'health':
            return {'role':'executor','rank':self.rank,'version':self.backend.version}
        if op == 'reset': return self.reset()
        if op == 'begin_batch': return self.begin_batch()
        if op == 'save':
            if self.frozen:raise ValueError('Frozen LongLive has no mutable checkpoint')
            if self.rank != 0 or self.pending or self.inflight:raise RuntimeError('Only committed rank zero saves')
            self.snapshot_writer.submit(write_executor, request['checkpoint'], freeze_cpu({
                'executor':cpu_state(self.backend.pipe.generator.model),
                'executor_optimizer':self.optimizer.state_dict()}))
            return {'rank':0,'version':self.backend.version}
        if op == 'generate': return self.generate(request)
        if op == 'enqueue': return self.enqueue(request)
        if op == 'drain': return self.drain()
        if op == 'finalize': return self.finalize(request)
        if op == 'stop': return {}
        raise ValueError(op)

    def close(self):
        try:
            self.snapshot_writer.close()
        finally:
            self.score_pool.shutdown(wait=True,cancel_futures=True)
            for item in self.inflight.values():item['scratch'].cleanup()
            self.inflight.clear()
            if dist.is_initialized(): dist.destroy_process_group()


def main():
    os.umask(0)
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--role',choices=['evaluation','planner','executor','dispatcher'],required=True)
    p.add_argument('--node',type=int,required=True)
    p.add_argument('--rank',type=int,default=-1)
    p.add_argument('--device',default='cuda:3')
    p.add_argument('--port',type=int,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    from scrl.config import load_run
    cfg=load_run(args.config)
    directory=Path(cfg['run'])/'workers'/f'{args.role}-{args.node}-{args.rank}'
    directory.mkdir(parents=True,exist_ok=True)
    worker=None
    started=time.monotonic()
    from scrl.eventlog import event
    def log_state(state, error=None):
        event(args.role, state, run=cfg['run'], node=args.node, rank=args.rank,
              error_type=None if error is None else type(error).__name__)
    log_state('loading')
    atomic_json(directory/'status.json',{'state':'loading'})
    try:
        if args.role=='evaluation':worker=EvaluationWorker(cfg,args.node,args.device,args.rank)
        elif args.role=='planner':worker=PlanningWorker(cfg,args.node,args.device)
        elif args.role=='dispatcher':
            from scrl.runtime.transport import EvaluationDispatcher
            worker=EvaluationDispatcher(cfg)
        else:worker=ExecutionWorker(cfg,args.rank,args.node,args.device)
        atomic_json(directory/'status.json',{'state':'ready','load_seconds':time.monotonic()-started})
        log_state('ready')
        def handle(op, request):
            try:
                return worker(op, request)
            except BaseException as error:
                if (args.role in ('evaluation','dispatcher') and op == 'evaluate' and
                        isinstance(error, UnscorableReward)):
                    raise
                record_failure(cfg['run'],{'role':args.role,'node':args.node,
                            'rank':args.rank,'error':repr(error),'traceback':traceback.format_exc()})
                raise
        serve(args.port,cfg['token'],handle,concurrent=args.role in ('evaluation','dispatcher'))
        atomic_json(directory/'status.json',{'state':'stopped'})
        log_state('stopped')
    except BaseException as error:
        log_state('failed', error)
        record_failure(cfg['run'],{'role':args.role,'node':args.node,'rank':args.rank,'error':repr(error)})
        atomic_json(directory/'status.json',{'state':'failed','error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        if worker is not None:worker.close()


if __name__=='__main__':main()
