"""Assign worker roles, start per-node supervisors and monitor their owned processes."""

from collections import defaultdict,deque
from pathlib import Path
import argparse
import csv
import heapq
import itertools
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from scrl.config import DEFAULT_CONFIGS
from scrl.runtime.transport import GIB, Peer, append_row, atomic_json, effective_state, local_event, optional_write, record_failure

JUDGE_LAUNCHER = DEFAULT_CONFIGS.parent / 'scripts' / 'serve_video_judge.sh'


def advance_dataset(cursor, epoch, size, repeat):
    """Keep the completed pass checkpointable and advance before the next batch."""
    if size<1 or not 0<=cursor<=size or epoch<0:
        raise ValueError('Invalid dataset position')
    return (0,epoch+1) if repeat and cursor==size else (cursor,epoch)


def prompt_seed(seed, epoch, index, size):
    return seed+(epoch*size+index)*10000


def estimate_work(request):
    """Relative cost includes prefix construction, recaching, suffixes and focal replay.

    This is a scheduling heuristic, not an estimate of seconds. It never changes
    sampling, normalization, group membership or the gradient update boundary.
    """
    lengths = [segment['num_chunks'] for segment in request['plan']['segments']]
    target, repeats = request['target'], len(request['seeds'])
    history = prefix = suffix = 0
    for index, length in enumerate(lengths):
        # Attention sees history; switching captions also recaches all history.
        forward = length * (1 + (history + length) / 16)
        recache = history * (1 + history / 16)
        cost = 5 * forward + recache
        if index < target:
            prefix += cost
        else:
            suffix += cost
        history += length
    prior = sum(lengths[:target])
    focal = lengths[target]
    replay = 15 * focal * (1 + (prior + focal) / 16) + prior * (1 + prior / 16)
    return prefix + repeats * (suffix + replay)


class WorkQueue:
    def __init__(self, warmup=1, split_chunks=0, *, queue_seed_pairs=False):
        self.warmup = warmup
        self.condition = threading.Condition()
        self.heap, self.counter = [], itertools.count()
        self.started = self.closed = False
        self.error = None
        self.split_chunks = split_chunks
        self.queue_seed_pairs = queue_seed_pairs
        self.waiters = 0

    def put(self, request):
        with self.condition:
            if self.closed:
                raise RuntimeError('Scheduling queue is closed')
            heapq.heappush(self.heap, (-estimate_work(request), next(self.counter), request))
            self.started |= len(self.heap) >= self.warmup
            self.condition.notify_all()

    def close(self, error=None):
        with self.condition:
            self.closed = True
            self.error = self.error or error
            self.condition.notify_all()

    def get(self):
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError('Another task failed') from self.error
                if self.heap and (self.started or self.closed):
                    request = heapq.heappop(self.heap)[2]
                    if (self.split_chunks and len(request['seeds']) == 4 and
                            (self.queue_seed_pairs or self.waiters > len(self.heap)) and
                            sum(s['num_chunks'] for s in request['plan']['segments']) >= self.split_chunks):
                        # Leave the second pair claimable even if all workers are busy right now.
                        parent = request['job_id']
                        pieces = [dict(request, parent_job=parent, job_id=f'{parent}/s{i}',
                                       seeds=request['seeds'][2*i:2*i+2]) for i in range(2)]
                        heapq.heappush(self.heap,(-estimate_work(pieces[1]),next(self.counter),pieces[1]))
                        self.condition.notify_all()
                        return pieces[0]
                    return request
                if self.closed:
                    return None
                self.waiters += 1
                try:
                    self.condition.wait()
                finally:
                    self.waiters -= 1


def validate_nodes(nodes):
    if not nodes or any(type(n.get('gpus')) is not int or n['gpus'] < 1 for n in nodes):
        raise ValueError('Each node needs a positive integer GPU count')
    if [n.get('index') for n in nodes] != list(range(len(nodes))):
        raise ValueError('Node indices must be consecutive, starting at zero')
    if any(not isinstance(n.get('address'),str) or not n['address'] for n in nodes):
        raise ValueError('Each node needs a communication address')
    if len({n['address'] for n in nodes}) != len(nodes):
        raise ValueError('Repeated node address')
    for n in nodes:
        g = n.get('physical_gpus', list(range(n['gpus'])))
        if (not isinstance(g,list) or len(g)!=n['gpus'] or any(type(i) is not int or i<0 for i in g)
                or len(set(g))!=len(g)):
            raise ValueError('Invalid physical GPU allowlist')


def assign_local_vae(cfg):
    """Each decoder borrows one distinct local generator GPU; RGB stays in local shm."""
    if cfg['training']['mode'] != 'planner-only':
        raise ValueError('Cannot share a training executor GPU with a decoder')
    for node in {e['node'] for e in cfg['evaluations']}:
        decoders=[e for e in cfg['evaluations'] if e['node']==node]
        devices=[e['device'] for e in cfg['executors'] if e['node']==node]
        if len(devices)<len(decoders):raise ValueError('Not enough local generator GPUs for VAE')
        for e,device in zip(decoders,devices):e['vae_device']=device


def compact_layout(nodes, service_count=None, planner_count=None, base_port=31150):
    """Share the actor with one service card and give every other card to execution."""
    validate_nodes(nodes)
    if any(n['gpus'] != 4 or n.get('physical_gpus',list(range(4))) != list(range(4)) for n in nodes):
        raise ValueError('Compact layout requires four GPUs per node')
    # Keep the measured service/executor ratio close to the 44-card layout.
    if service_count is None:service_count=max(1,round(len(nodes)*4*14/96))
    if planner_count is None:planner_count=len(nodes)
    if not 1 <= service_count <= len(nodes) or not 1 <= planner_count <= 2*service_count:
        raise ValueError('Invalid service or planner count')
    if not 1024<=base_port<=65514:raise ValueError('Invalid cluster base port')
    selected = [nodes[i * len(nodes) // service_count] for i in range(service_count)]
    services = [{'node': n['index'], 'address': n['address'], 'device': 'cuda:3', 'port': base_port}
                for n in selected]
    planners = []
    for index in range(planner_count):
        endpoint = services[index % service_count]
        planners.append(dict(endpoint, rank=index, port=base_port+1 + 20 * (index // service_count)))
    service_nodes = {s['node'] for s in services}
    executors = []
    for node in nodes:
        for gpu in range(node['gpus']):
            if node['index'] in service_nodes and gpu == 3:
                continue
            executors.append({'rank': len(executors), 'node': node['index'],
                              'address': node['address'], 'device': f'cuda:{gpu}', 'port': base_port+10 + gpu})
    return {'evaluations': services, 'planners': planners, 'executors': executors,
            'executor_count': len(executors), 'actor_device': 'cuda:0', 'actor_physical_device': 'cuda:3',
            'shared_latent_transport': True, 'pipeline_depth': 2, 'pipeline_cache_gib': 8,
            'ports':{'evaluation':base_port,'planner':base_port+1},
            'rendezvous':f'tcp://{nodes[0]["address"]}:{base_port+2}'}


def explicit_layout(nodes, service_cards, *, actor_node, base_port=34150, dedicated=False,
                    extra_planner_cards=(), actor_samples=False):
    """Multiple isolated services per node; communication roles are explicit."""
    validate_nodes(nodes)
    cards = [tuple(pair) for pair in service_cards]
    extra = {tuple(pair) for pair in extra_planner_cards}
    if actor_samples and (not dedicated or extra):
        raise ValueError('Actor sampling requires the dedicated two-planner layout without extra planner cards')
    if extra and (not dedicated or not extra.issubset(cards) or any(n == actor_node for n,g in extra)):
        raise ValueError('Additional planners require reserved dedicated non-actor cards')
    if not cards or len(set(cards)) != len(cards):
        raise ValueError('Service cards must be unique')
    if any(not 0 <= n < len(nodes) or g not in nodes[n].get('physical_gpus',range(nodes[n]['gpus'])) for n, g in cards):
        raise ValueError('Unknown service GPU')
    if not any(n == actor_node for n, _ in cards) or not 1024 <= base_port <= 65400:
        raise ValueError('Actor must share a service card; valid ports required')
    services = [dict(node=n, address=nodes[n]['address'], device=f'cuda:{g}',
                     rank=i, port=base_port+g) for i, (n, g) in enumerate(cards)]
    planners = [dict(s, port=base_port+20+int(s['device'].split(':')[1])) for s in services]
    executors = []
    for n in nodes:
        for gpu in n.get('physical_gpus',range(n['gpus'])):
            if (n['index'], gpu) not in cards:
                executors.append(dict(rank=len(executors), node=n['index'], address=n['address'],
                                      device=f'cuda:{gpu}', port=base_port+40+gpu))
    if not executors:
        raise ValueError('At least one Executor required')
    actor = next(s for s in services if s['node'] == actor_node)
    if dedicated:
        if len(cards) < 3:
            raise ValueError('Dedicated layout needs separate actor, planner and judge cards')
        planner = next((s for s in planners if s['node'] == actor_node and s['device'] != actor['device']), None)
        if planner is None:
            raise ValueError('The actor node needs a second card for the sampling planner')
        services = [s for s in services if s['node'] != actor_node]
        if len(services) != len(cards)-2:
            raise ValueError('Exactly two service cards must be on the actor node')
        def card(s):
            return (s['node'], int(s['device'].split(':')[1]))
        additional = [s for s in planners if card(s) in extra]
        if actor_samples:
            additional.append(next(s for s in planners if card(s) == card(actor)))
        services = [dict(s,rank=i) for i,s in enumerate(s for s in services if card(s) not in extra)]
        if not services:
            raise ValueError('Dedicated layout requires at least one judge')
        planners = [dict(s,rank=i) for i,s in enumerate([planner]+additional)]
    return dict(evaluations=services, planners=planners, executors=executors,
        executor_count=len(executors), actor_node=actor_node, actor_device='cuda:0',
        actor_physical_device=actor['device'], shared_latent_transport=True,
        pipeline_depth=2, pipeline_cache_gib=8, dispatch_warmup=4,
        dispatcher=dict(node=actor_node, address=actor['address'], port=base_port+60,
                        device=actor['device']),
        rendezvous=f'tcp://{executors[0]["address"]}:{base_port+80}')


def baseline_pid(output, reservation=None):
    """Allow one pre-existing process only under an explicit memory and command policy."""
    if reservation is not None:
        if not isinstance(reservation, dict) or set(reservation) != {'commands', 'max_mib'}:
            raise ValueError('gpu_reservation requires commands and max_mib')
        commands, limit = reservation['commands'], reservation['max_mib']
        if (not isinstance(commands, list) or not commands or
                any(not isinstance(c, str) or not Path(c).is_absolute() for c in commands) or
                type(limit) is not int or limit <= 0):
            raise ValueError('GPU reservation requires absolute command paths and a positive MiB limit')
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    if not rows:
        return ''
    if reservation is not None and len(rows) == 1 and len(rows[0]) == 3:
        pid, mib, command = rows[0]
        if pid.isdigit() and mib.isdigit() and int(mib) <= limit and command in commands:
            return pid
    raise RuntimeError('Judge GPU is not exclusively available; existing processes left untouched')


def start_judge(cfg, service, directory, env):
    from scrl.config import file_hash
    script = JUDGE_LAUNCHER
    if file_hash(script) != cfg['services']['judge_launcher_sha256']:
        raise ValueError('Judge launcher changed after prepare')
    gpu = service['device'].split(':')[1]
    port = cfg['services']['judge_ports'][str(service['rank'])]
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    apps = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,used_memory,process_name',
                                    '--format=csv,noheader,nounits'],text=True)
    judge = cfg['assets']['rewards']['judge']
    services = cfg['services']
    env = dict(env, JUDGE_BASELINE_PID=baseline_pid(apps, services.get('gpu_reservation')),
               JUDGE_MODEL=judge['weights'], JUDGE_PYTHON=judge['python'],
               JUDGE_MIN_FREE_MIB=str(services.get('judge_min_free_mib',76000)),
               JUDGE_SEQUENCES=str(services['max_inflight']))
    for name in ('SCRL_RPC_TOKEN','PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(name, None)
    log = (directory/f"vllm-{service['rank']}.log").open('a')
    try:
        proc = subprocess.Popen(['bash',str(script),gpu,str(port)],env=env,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:
        log.close()
        raise
    return proc, log, port


def wait_ready(servers, run, timeout=900):
    pending = dict(servers)
    deadline = time.monotonic() + timeout
    while pending:
        if any((run/flag).exists() for flag in ('STOP','FAIL.json')):
            return False
        for port, proc in list(pending.items()):
            if proc.poll() is not None:
                raise RuntimeError(f'Owned judge on port {port} exited ({proc.returncode}); inspect vllm log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
                    del pending[port]
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Judge startup timed out: {sorted(pending)}')
            time.sleep(1)
    return True


def launch(config, node, *, wait=False):
    from scrl.config import load_run
    os.umask(0)
    path=Path(config).resolve();cfg=load_run(path);run=Path(cfg['run'])
    if node not in {n['index'] for n in cfg['nodes']}:
        raise ValueError('Unknown node')
    source=Path(cfg['source_root']);launch=run/'launch';launch.mkdir(exist_ok=True)
    if cfg.get('resume'):
        resume=Path(cfg['resume'])
        if not resume.is_absolute() or not (resume/'state.json').is_file():
            raise ValueError('Resume must name an existing absolute checkpoint before launching workers')
    if ('judge' in cfg.get('assets', {}).get('rewards', {}) and
            cfg.get('reward', {}).get('mode') != 'event-v2'):
        from scrl.models.scoring.protocol import source_requirements
        for prompt in cfg['prompts']:
            source_requirements(prompt)
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=entry.read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        if ('scrl.runtime.launch' in cmd or 'scrl.rl.train' in cmd) and str(path) in cmd:
            raise RuntimeError('This node already owns live processes for the run')
    if any((run/name).exists() for name in ('FAIL.json','STOP','DRAIN')):
        raise RuntimeError('Run has a stop/failure marker; prepare a verified resume first')
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
             HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    children = []
    def start(module,name,extra=(),environment=None,role=None):
        interpreter=cfg.get('role_python',{}).get('actor',cfg['python']) if module=='scrl.rl.train' else cfg['python']
        with (launch/(name+'.log')).open('a') as log:
            process=subprocess.Popen([interpreter,'-u','-m',module,*([role] if role else []),'--config',str(path),*extra],
                cwd=source,env=environment or env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        children.append(process)
        return process.pid
    result={'node':node,'supervisor_pid':start('scrl.runtime.launch',f'node-{node}',
                                                  ['--node',str(node)],role='node')}
    if node==cfg.get('actor_node',0):
        actor_env=dict(env,CUDA_VISIBLE_DEVICES=cfg['actor_physical_device'].split(':')[-1])
        pid=start('scrl.rl.train','coordinator',environment=actor_env)
        (run/'coordinator_pid.json').write_text(json.dumps(dict(pid=pid,node=node,started_unix=time.time()))+'\n')
        result.update(coordinator_pid=pid,monitor_pid=start('scrl.runtime.launch','monitor',['--interval','30'],role='monitor'))
    (launch/f'node-{node}.json').write_text(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    if wait:
        codes = [child.wait() for child in children]
        if any(codes) or (run/'FAIL.json').exists():
            raise RuntimeError(f'Training node {node} did not stop cleanly: {codes}')


def owned_descendants(pid):
    """Track only child processes of a worker owned by this supervisor."""
    pending=[pid];result={}
    while pending:
        parent=pending.pop()
        children=set()
        for child_file in (Path('/proc')/str(parent)/'task').glob('*/children'):
            try:children.update(child_file.read_text().split())
            except OSError:pass
        for child in children:
            try:
                value=(Path('/proc')/child/'stat').read_text().rsplit(')',1)[1].split()[19]
            except (OSError,IndexError):continue
            if int(child) not in result:
                result[int(child)]=value;pending.append(int(child))
    return result


def signal_owned(descendants, signum):
    for pid,started in descendants.items():
        try:
            now=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            if now==started:os.kill(pid,signum)
        except (OSError,IndexError):pass


def node_main(argv=None):
    from scrl.config import SOURCE, load_run
    os.umask(0)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--node',type=int,required=True)
    args=p.parse_args(argv);config_path=str(Path(args.config).resolve())
    cfg=load_run(config_path);root=SOURCE
    run=Path(cfg['run']);directory=run/'nodes'/f'node-{args.node}';directory.mkdir(parents=True,exist_ok=True)
    (directory/'stopped.json').unlink(missing_ok=True)
    env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
             HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
             NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
             MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.update(cfg.get('communication_env', {}))
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if 'evaluations' in cfg:
        jobs=[('evaluation',s.get('rank',-1),s['device'],s['port']) for s in cfg['evaluations'] if s['node']==args.node]
        jobs += [('planner',s['rank'],s['device'],s['port']) for s in cfg['planners'] if s['node']==args.node]
    else:
        jobs=[('evaluation',-1,'cuda:3',cfg['ports']['evaluation']),('planner',-1,'cuda:3',cfg['ports']['planner'])]
    jobs += [('executor',r['rank'],r['device'],r['port']) for r in cfg['executors'] if r['node']==args.node]
    broker = cfg.get('dispatcher')
    if broker and broker['node'] == args.node:
        jobs.append(('dispatcher', 0, broker['device'], broker['port']))
    processes=[]; owners=[]; servers={}
    try:
        # Reject a host-only address before loading models or initializing NCCL.
        with socket.socket() as probe:
            probe.bind((cfg['nodes'][args.node]['address'],0))
        if cfg.get('services',{}).get('judge_launcher_sha256'):
            for service in cfg['evaluations']:
                if service['node'] != args.node:continue
                proc,log,port=start_judge(cfg,service,directory,env)
                processes.append((proc,log));owners.append(('vllm',service['rank']));servers[port]=proc
            if not wait_ready(servers,run):return
        for role,rank,device,port in jobs:
            log=(directory/f'{role}-{rank}.log').open('a')
            python=cfg.get('role_python',{}).get(role,cfg['python'])
            cmd=[python,'-u','-m','scrl.runtime.worker','--config',config_path,
                 '--role',role,'--rank',str(rank),'--node',str(args.node),'--device','cuda:0','--port',str(port)]
            visible=device.split(':')[-1]
            if role=='evaluation':
                aux=next((e.get('vae_device') for e in cfg.get('evaluations',[]) if e.get('rank')==rank),None)
                if aux:visible+=','+aux.split(':')[-1]
            job_env=dict(env,CUDA_VISIBLE_DEVICES=visible)
            proc=subprocess.Popen(cmd,cwd=root,env=job_env,stdin=subprocess.DEVNULL,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((proc,log));owners.append((role,rank))
        atomic_json(directory/'processes.json',[dict(role=r,rank=i,pid=p.pid) for (r,i),(p,_) in zip(owners,processes)])
        from scrl.telemetry import SpeedSampler
        sampler = SpeedSampler(cfg['nodes'][args.node].get('physical_gpus', list(range(cfg['nodes'][args.node]['gpus']))))
        tick=0
        while True:
            if (run/'FAIL.json').exists() or (run/'STOP').exists():break
            failed=[proc for (role,_),(proc,_) in zip(owners,processes)
                    if proc.poll() not in (None,0) or (role=='vllm' and proc.returncode is not None)]
            if failed:
                failure={'node':args.node,'pids':[p.pid for p in failed],
                         'exit_codes':{str(p.pid):p.returncode for p in failed},
                         'error':'Worker process exited; inspect node logs'}
                memory_events=Path('/sys/fs/cgroup/memory.events')
                if memory_events.exists():failure['memory_events']=memory_events.read_text()
                record_failure(run,failure)
                optional_write(atomic_json,directory/'failure.json',failure)
                break
            if all(proc.poll() is not None for (role,_),(proc,_) in zip(owners,processes) if role!='vllm'):break
            if tick%2==0:
                value=sampler.sample(args.node)
                optional_write(append_row,directory/'utilization.jsonl',json.dumps(value))
                heartbeat={'unix':time.time(),'live':sum(p.poll() is None for p,_ in processes)}
                heartbeat['shared_free_bytes']=shutil.disk_usage(run).free
                for name in ['memory.current','memory.max','memory.events']:
                    path=Path('/sys/fs/cgroup')/name
                    if path.exists():heartbeat[name]=path.read_text().strip()
                try:local_event(run,f'node-{args.node}-heartbeat.json',heartbeat)
                except OSError:pass
                optional_write(atomic_json,directory/'heartbeat.json',heartbeat)
                optional_write(append_row,directory/'host_memory.jsonl',json.dumps(heartbeat))
            time.sleep(1);tick+=1
    except BaseException as exc:
        record_failure(run,dict(node=args.node,error=repr(exc)))
        raise
    finally:
        try:
            atomic_json(directory/'pre_shutdown.json', {
                'unix': time.time(), 'global_failure': (run/'FAIL.json').exists(),
                'processes': [dict(role=role, rank=rank, pid=p.pid, returncode=p.poll())
                              for (role,rank),(p,_) in zip(owners,processes)]})
        except OSError as exc:
            # stdout/stderr may also be on the full device. Reaping must still run.
            try:local_event(run,f'node-{args.node}-shutdown-error.json',dict(error=repr(exc),unix=time.time()))
            except OSError:pass
        children={}
        for proc,_ in processes:
            if proc.poll() is None:children.update(owned_descendants(proc.pid))
        for proc,_ in processes:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        signal_owned(children,signal.SIGTERM)
        for proc,log in processes:
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
            log.close()
        signal_owned(children,signal.SIGKILL)
        stopped={'unix':time.time()}
        try:local_event(run,f'node-{args.node}-stopped.json',stopped)
        except OSError:pass
        optional_write(atomic_json,directory/'stopped.json',stopped)


class JsonlTail:
    def __init__(self,path):
        self.path=Path(path);self.offset=0;self.inode=None

    def read(self):
        if not self.path.exists():return []
        stat=self.path.stat()
        if self.inode!=stat.st_ino or stat.st_size<self.offset:
            self.offset=0;self.inode=stat.st_ino
        rows=[]
        with self.path.open() as f:
            if self.offset==0 and stat.st_size>2*2**20:
                f.seek(stat.st_size-2*2**20);f.readline();self.offset=f.tell()
            f.seek(self.offset)
            while True:
                line=f.readline()
                if not line or not line.endswith('\n'):break
                self.offset=f.tell()
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
        return rows


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return default


def monitor_main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--interval',type=float,default=30);p.add_argument('--once',action='store_true');args=p.parse_args(argv)
    if args.interval<5:p.error('Monitoring interval must be at least five seconds')
    from scrl.config import load_run
    cfg=load_run(args.config);run=Path(cfg['run']);out=run/'monitor'
    optional_write(lambda:out.mkdir(exist_ok=True))
    durable=Path(cfg['durable_run']) if cfg.get('durable_run') else None
    executor_cards={(r['node'],int(r['device'].split(':')[-1])) for r in cfg['executors']}
    tails={n['index']:JsonlTail(run/'nodes'/f'node-{n["index"]}'/'utilization.jsonl') for n in cfg['nodes']}
    samples={i:deque(maxlen=1000) for i in tails};baselines={};last_report_step=None;last_report=None
    while True:
        now=time.time();status=(read_json(durable/'status.json') if durable else None) or read_json(run/'status.json',{})
        node_health=[];alerts=[];roles=defaultdict(list);cards=set()
        free=shutil.disk_usage(run).free
        if free<12*GIB:alerts.append(f'shared training disk low: {free/GIB:.2f} GiB')
        for node,tail in tails.items():
            samples[node].extend(tail.read())
            while samples[node] and samples[node][0]['unix']<now-300:samples[node].popleft()
            directory=tail.path.parent;h=read_json(directory/'heartbeat.json',{})
            if node not in baselines:
                try:
                    with (directory/'host_memory.jsonl').open() as f:baselines[node]=json.loads(next(f)).get('memory.events')
                except (OSError,StopIteration,json.JSONDecodeError):pass
            age=max(0,now-h.get('unix',0))
            memory=int(h['memory.current'])/2**30 if 'memory.current' in h else None
            changed=node in baselines and baselines[node]!=h.get('memory.events')
            stopped=(directory/'stopped.json').exists()
            node_health.append({'node':node,'heartbeat_age_seconds':round(age,1),'live_processes':h.get('live'),
                'memory_gib':round(memory,1) if memory is not None else None,'memory_event_change':changed,'stopped':stopped})
            if age>60 and not stopped:alerts.append(f'node {node}: stale heartbeat')
            if changed:alerts.append(f'node {node}: memory counters changed')
            for row in samples[node]:
                if not now-300<=row['unix']<=now:continue
                for gpu in row['gpus']:
                    key=(node,gpu['index']);role='executor' if key in executor_cards else 'services'
                    if gpu.get('utilization') is not None:
                        roles[role].append(gpu['utilization']);roles['all'].append(gpu['utilization']);cards.add(key)
        step=status.get('completed_rounds',0)
        if step!=last_report_step:
            report_path=Path('batches')/f'{step-1:05d}'/'report.json'
            report=((read_json(durable/report_path) if durable else None) or read_json(run/report_path)) if step else None
            if report is not None:
                last_report={k:report.get(k) for k in ['round','cursor','videos','physical_videos','seconds','checkpoint']}
                ranks=report['updates']['executor']['ranks']
                last_report.update(replay_max_error=max(r['replay_max_error'] for r in ranks),
                    replicas_equal=len({r['replica_sha256'] for r in ranks})==1)
                last_report_step=step
        failed=(run/'FAIL.json').exists()
        failure=read_json(run/'FAIL.json') or ({'error':'Failure marker present but unreadable'} if failed else None)
        pid_data=read_json(run/'coordinator_pid.json',{})
        try:
            command=Path(f'/proc/{pid_data["pid"]}/cmdline').read_bytes()
            coordinator_alive=b'scrl.rl.train' in command
        except (OSError,KeyError):coordinator_alive=False
        if failure is not None:alerts.append('training failure recorded')
        effective=effective_state(status,failed,coordinator_alive)
        terminal=effective in ['failed','drained','complete','budget_complete']
        if not coordinator_alive and not terminal:alerts.append('coordinator not running')
        result={'checked_unix':now,'status':dict(status,state=effective),'reported_state':status.get('state'),
            'effective_state':effective,'shared_free_bytes':free,'coordinator_alive':coordinator_alive,
            'status_age_seconds':round(now-status.get('checked_unix',now),1),
            'gpu_sample_window_seconds':300,'gpu_cards_sampled':len(cards),
            'gpu_utilization_pct':{k:round(statistics.mean(v),1) for k,v in roles.items()},
            'nodes':node_health,'last_complete_round':last_report,
            'checkpoints':[p.name for p in sorted((run/'checkpoints').glob('checkpoint-*'))],
            'alerts':alerts,'failure':failure}
        if cfg.get('dispatcher') and not terminal:
            endpoint=cfg['dispatcher']
            try:
                result['scoring_dispatcher']=Peer(endpoint['address'],endpoint['port'],cfg['token'],timeout=5).call('health')
            except (OSError,RuntimeError,EOFError) as error:
                result['scoring_dispatcher']={'unavailable':repr(error)}
        recoveries={}
        for name in ('invalid_contracts','invalid_judgments'):
            counts=defaultdict(int)
            for path in (run/name).glob('*.json'):
                counts[read_json(path,{}).get('state','unknown')]+=1
            recoveries[name]=dict(counts)
        result['format_recoveries']=recoveries
        if durable:
            atomic_json(durable/'monitor/latest.json',result)
            append_row(durable/'monitor/history.jsonl',json.dumps(result))
        optional_write(atomic_json,out/'latest.json',result)
        # The project-level pointer must not remain "running" after this run stops.
        current_path=run.parent/'current_run.json'
        current=read_json(current_path,{})
        if current.get('run')==cfg['run']:
            current.update(state=effective,completed_rounds=step,
                prompts_processed=status.get('prompts_processed'),checked_unix=now,
                active_round=step+1 if effective in ('collecting','updating') else None,
                latest_checkpoint=(run/'checkpoints/latest').resolve().name,
                alerts=alerts,failure=failure)
            optional_write(atomic_json,current_path,current)
        optional_write(append_row,out/'history.jsonl',json.dumps(result))
        if args.once or terminal:break
        time.sleep(args.interval)


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("node", "monitor"))
    role = parser.parse_args(argv[:1]).role
    return (node_main if role == "node" else monitor_main)(argv[1:])


if __name__ == "__main__":
    main()
