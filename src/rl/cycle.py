"""Controller-side serial train/checkpoint/VBench/resume schedule."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import time

from scrl.config import DEFAULT_CONFIGS, file_hash, read
from scrl.evaluation.generate import prepare_main
from scrl.evaluation.protocol import DIMENSIONS
from scrl.plan import digest
from scrl.rl.identity import training_fingerprint
from scrl.runtime.transport import atomic_json


def cards(cfg):
    return [(n['address'], g) for n in cfg['nodes']
            for g in n.get('physical_gpus', range(n['gpus']))]


def parallel(jobs, action, on_error=None):
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(action, job) for job in jobs]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            if on_error:
                on_error()
            for future in futures:
                future.cancel()
            raise


class Fleet:
    def __init__(self, cfg, helper):
        self.cfg, self.helper = cfg, helper

    def run(self, address, argv, log, *, token=False, env=None):
        settings = dict(PYTHONPATH=self.cfg['source_root'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', **(env or {}))
        words = ['env', *(f'{k}={v}' for k,v in settings.items()), *map(str, argv)]
        cmd = 'umask 000; ' + ('IFS= read -r SCRL_RPC_TOKEN || exit 1; export SCRL_RPC_TOKEN; ' if token else '')
        cmd += 'exec '+shlex.join(words)
        with log.open('a') as output:
            subprocess.run([str(self.helper),address,shlex.join(['bash','-c',cmd])], check=True,
                           input=self.cfg['token']+'\n' if token else '', text=True,
                           stdout=output,stderr=subprocess.STDOUT)

    def idle(self, directory):
        def check(n):
            ids=n.get('physical_gpus',range(n['gpus']))
            cmd='set -e; p=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); test -z "$p"'
            self.run(n['address'],['bash','-c',cmd,'probe',','.join(map(str,ids))],
                     directory/f"idle-{n['index']}.log")
        parallel(self.cfg['nodes'],check)


def assert_gate(report):
    aggregate=report['aggregate']
    import math
    if (aggregate['state']!='complete' or set(report['raw_dimensions'])!=set(DIMENSIONS)
            or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in
                   [*report['raw_dimensions'].values(), *(aggregate[k] for k in ('total','semantic','quality'))])):
        raise ValueError('Incomplete VBench score; training remains stopped')


def evaluate(cfg, fleet, cp, step, settings, *, output_root=None):
    parent=(output_root or Path(cfg['run'])/'vbench')/f'step-{step:05d}'
    parent.mkdir(parents=True,exist_ok=True)
    dest=parent/'evaluation'
    report=parent/'report.json'
    if cp is None and (step != 0 or cfg.get('resume') or cfg.get('init_from')):
        raise ValueError('Step zero requires fresh initial policies')
    inputs={'checkpoint':str(cp) if cp is not None else None,
            'checkpoint_state':file_hash(cp/'state.json') if cp is not None else cfg['verified_inputs'],
            **{k:file_hash(settings[k]) for k in ('recipe','prompts','metadata')}}
    pin=parent/'identity.json'
    if pin.exists() and read(pin)!=inputs:
        raise ValueError('Evaluation inputs changed since the previous attempt')
    atomic_json(pin,inputs)
    if report.exists():
        saved=read(report)
        if saved.get('identity')!=inputs or saved.get('step')!=step:
            raise ValueError('Existing report does not belong to this checkpoint')
        assert_gate(saved)
        return
    if not (dest/'config.json').is_file():
        paths=parent/'assets.json'
        atomic_json(paths,dict(asset_root='/',assets=cfg['assets'],role_python=cfg['role_python']))
        prepare_main(['--run',str(dest),'--arm','joint','--paths',str(paths),
                      *(['--checkpoint',str(cp)] if cp is not None else []),
                      '--prompts',settings['prompts'],'--config',settings['recipe'],
                      '--official-metadata',settings['metadata'],'--vbench-source',settings['source'],
                      '--vbench-weights',settings['weights']])
    dest.chmod(0o777)  # Scores and generations are written by distinct mapped UIDs.
    devices=cards(cfg)
    fleet.idle(parent)
    def generate(job):
        i,(address,gpu)=job
        fleet.run(address,[cfg['python'],'-m','scrl','eval','generate','--run',dest,'--arm','joint',
                           '--shard',i,'--shards',len(devices)],parent/f'generate-{i}.log',
                  env=dict(CUDA_VISIBLE_DEVICES=str(gpu),SCRL_VBENCH_FLEET='1',
                           SCRL_EVAL_CONDITIONS=(str(dest/'conditions') if output_root else
                               '/tmp/scrl-vbench-'+digest([str(dest),inputs])[:20])))
    parallel(list(enumerate(devices)),generate)
    score_host=cfg['nodes'][settings['score_node']]['address']
    scoring=[g for addr,g in devices if addr==score_host]
    scorer=cfg['assets']['rewards']['quality']['python']
    def score(gpu,dimension):
        fleet.run(score_host,[scorer,'-m','scrl','eval','vbench',dest,'--arm','joint',
                              '--dimension',dimension],parent/f'score-{dimension}.log',
                  env=dict(CUDA_VISIBLE_DEVICES=str(gpu)))
    score(scoring[0],'static_filter')
    def lane(i):
        for dimension in DIMENSIONS[i::len(scoring)]:
            score(scoring[i],dimension)
    parallel(list(range(len(scoring))),lane)
    from scrl.evaluation.vbench import summarize
    result=summarize(dest,'joint',read(dest/'config.json')['formula'])
    assert_gate(result)
    atomic_json(report,dict(result,step=step,identity=inputs))


def campaign(cfg, helper):
    root=Path(cfg['run'])
    with (root/'cycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            _cycle(cfg,Fleet(cfg,helper))
        except BaseException as exc:
            atomic_json(root/'cycle-status.json',dict(state='blocked',error_type=type(exc).__name__,
                        detail=str(exc),unix=time.time()))
            raise


def _cycle(cfg,fleet):
    root=Path(cfg['run'])
    settings=dict(cfg['periodic_vbench'])
    period=settings['every_steps']
    if type(period) is not int or period<1 or not 0<=settings['score_node']<len(cfg['nodes']):
        raise ValueError('Invalid periodic VBench layout or interval')
    if cfg.get('durable_run'):
        raise ValueError('This schedule requires shared split checkpoints, not a separate durable run')
    quality=cfg['assets']['rewards']['quality']
    settings.update(recipe=str(DEFAULT_CONFIGS/settings['recipe']),prompts=str(DEFAULT_CONFIGS/settings['prompts']),
                    source=quality['source'],metadata=str(Path(quality['source'])/settings['metadata_relative_to_vbench']),
                    weights=str((Path(quality['weights'])/settings['weights_relative_to_training_weights']).resolve()))
    for key in ('recipe','prompts','source','metadata','weights'):
        if not Path(settings[key]).exists():raise FileNotFoundError(settings[key])
    cp=Path(cfg['resume']) if cfg.get('resume') else None
    step=int(cp.name.removeprefix('checkpoint-')) if cp else 0
    limit=step+cfg['validation_round_limit'] if cfg.get('validation_round_limit') else None
    def status(state,step):
        atomic_json(root/'cycle-status.json',dict(state=state,step=step,unix=time.time()))
    while True:
        if (root/'STOP').exists() or (root/'DRAIN').exists():
            status('stopped',step);return
        if step and not step%period:
            status('evaluating',step)
            evaluate(cfg,fleet,cp,step,settings)
        if (limit is not None and step>=limit) or (root/'STOP').exists() or (root/'DRAIN').exists():
            status('stopped',step);return
        target=step+period-step%period
        if limit is not None:target=min(target,limit)
        phase=root/'windows'/f'to-{target:05d}'
        phase.mkdir(parents=True,exist_ok=True)
        phase.chmod(0o777)
        child={k:v for k,v in cfg.items() if k not in ('token','periodic_vbench')}
        child.update(run=str(phase),resume=str(cp) if cp else None,cycle_window=str(root),
                     validation_round_limit=target-step)
        config=phase/'config.json'
        if config.exists() and read(config)!=child:raise ValueError('Training window identity changed')
        if not config.exists():
            atomic_json(config,child)
            atomic_json(phase/'receipt.json',dict(config_sha256=file_hash(config),
                        source_sha256=cfg['source_sha256'],training_fingerprint=training_fingerprint(child)))
            (phase/'logs').mkdir(exist_ok=True)
            (phase/'logs').chmod(0o1777)
        closed=phase/'closed.json'
        if not closed.exists():
            if (phase/'launch').exists():
                raise RuntimeError(f'Unsealed previous training launch in {phase}; verify workers and recover from a published checkpoint')
            fleet.idle(phase)
            status('training',step)
            def start(n):
                fleet.run(n['address'],[cfg['python'],'-m','scrl','rl','train','--node',n['index'],
                                         '--run',phase],phase/f"launch-{n['index']}.log",token=True)
            parallel(cfg['nodes'],start,on_error=lambda:atomic_json(phase/'FAIL.json',{'error':'Node launch failed'}))
            summary=read(phase/'status.json')
            end=summary['completed_rounds']
            if not step<=end<=target or summary['state'] not in ('drained','complete','budget_complete'):
                raise RuntimeError('Unconfirmed training boundary')
            for n in cfg['nodes']:
                read(phase/'nodes'/f"node-{n['index']}"/'stopped.json')
            atomic_json(closed,dict(step=end,state=summary['state'],cursor=summary['cursor']))
        end=read(closed)['step']
        if end<=step:
            status('stopped',step);return
        cp=phase/'checkpoints'/f'checkpoint-{end:05d}'
        state=read(cp/'state.json')
        if state.get('format')!='split-policies-v1' or set(state['files'])!={'executor.pt','planner.pt'}:
            raise ValueError('No published joint checkpoint at training boundary')
        if end<target or (not cfg.get('repeat_dataset') and read(closed)['cursor']>=len(cfg['prompts'])):
            if end%period==0:evaluate(cfg,fleet,cp,end,settings)
            status('complete',end);return
        step=end
