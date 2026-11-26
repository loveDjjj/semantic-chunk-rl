"""Freeze immutable update snapshots and publish verified, resumable checkpoints."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path
import copy
import json
import os
import shutil
import tempfile
import time
import torch
from scrl.config import file_hash as file_sha256
from scrl.plan import PolicyStamp
from scrl.runtime.transport import atomic_json


def retain_latest(directory, checkpoint, *, prune=True, keep_every=0):
    """Publish the latest link, then prune ordinary saves while retaining milestones."""
    if type(keep_every) is not int or keep_every < 0:
        raise ValueError('Checkpoint milestone interval must be a nonnegative integer')
    directory, checkpoint = Path(directory), Path(checkpoint)
    state = json.loads((checkpoint/'state.json').read_text())
    hashes = state.get('files', {'policies.pt': state.get('policies_sha256')})
    if any(expected is None or file_sha256(checkpoint/name) != expected for name,expected in hashes.items()):
        raise ValueError('Latest checkpoint is incomplete; older checkpoints are retained')
    temporary = directory/'.latest-new'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint.name, target_is_directory=True)
    temporary.replace(directory/'latest')
    if not prune:
        return
    for old in directory.glob('checkpoint-*'):
        if old != checkpoint and old.is_dir() and not old.is_symlink() and (old/'state.json').is_file():
            suffix=old.name.removeprefix('checkpoint-')
            if keep_every and suffix.isdigit() and int(suffix)>0 and int(suffix)%keep_every==0:
                continue
            shutil.rmtree(old)


def publish_split_checkpoint(staging, destination, coordinator, planner_payload, *, keep_latest_only=True,
                             keep_every=0):
    """Executor rank zero writes directly into staging; no duplicate 4-GiB payload."""
    staging, destination = Path(staging), Path(destination)
    if destination.exists() or not (staging/'executor.pt').is_file():
        raise ValueError('Invalid checkpoint staging directory')
    torch.save(planner_payload, staging/'planner.pt')
    state = coordinator.state_dict()
    state.update(format='split-policies-v1',files={name:file_sha256(staging/name)
                 for name in ['planner.pt','executor.pt']})
    (staging/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    for name in ['planner.pt','executor.pt','state.json']:
        with (staging/name).open('rb') as handle:os.fsync(handle.fileno())
    staging.rename(destination)
    fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    retain_latest(destination.parent,destination,prune=keep_latest_only,keep_every=keep_every)
    return destination


def publish_planner_checkpoint(directory, step, planner_payload, initial_hashes):
    """Publish a resumable planner update; the frozen executor has no optimizer state."""
    if type(step) is not int or step < 1 or planner_payload.get('job_index') != step:
        raise ValueError('Checkpoint step and training cursor disagree')
    required = {'planner', 'planner_optimizer', 'cursor', 'epoch', 'torch_rng', 'cuda_rng',
                'config_fingerprint'}
    if not required <= planner_payload.keys():
        raise ValueError('Incomplete planner resume payload')
    if set(initial_hashes) != {'planner_adapter', 'executor_base_adapter', 'executor_backbone'}:
        raise ValueError('Missing cold-start identity')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f'checkpoint-{step:05d}'
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=directory, prefix='.planner-') as tmp:
        staged = Path(tmp) / dest.name
        staged.mkdir()
        torch.save(planner_payload, staged / 'planner.pt')
        state = {'format':'planner-only-v1', 'stamp':{'planner':planner_payload.get('policy_version',step), 'executor':0},
                 'initial_adapter_hashes':initial_hashes,
                 'files':{'planner.pt':file_sha256(staged / 'planner.pt')}}
        (staged / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        staged.rename(dest)
    retain_latest(directory, dest, prune=False)
    return dest


class Phase(str, Enum):
    JOINT = "joint"


class JointCoordinator:
    def __init__(self, stamp=PolicyStamp(), phase=Phase.JOINT):
        self.stamp, self.phase = stamp, Phase(phase)
        self.pending = set()
        self.completed = set()
        self.committed_batches = set()
        self.batch_id = None
        self.failed = False

    def begin(self, batch_id, job_ids):
        if self.batch_id is not None or self.failed:
            raise RuntimeError("Another batch is active or recovery is required")
        ids = list(job_ids)
        if not batch_id or batch_id in self.committed_batches or not ids or len(set(ids)) != len(ids):
            raise ValueError("Empty or repeated batch/jobs")
        self.batch_id, self.pending, self.completed = batch_id, set(ids), set()
        return self.stamp

    def complete(self, job_id, stamp):
        if stamp != self.stamp or job_id not in self.pending:
            raise ValueError("Stale, duplicate or unregistered job")
        self.pending.remove(job_id)
        self.completed.add(job_id)

    def commit(self, update):
        """Callback must finish its optimizer step before the version is advanced."""
        if self.batch_id is None or self.pending or self.failed:
            raise RuntimeError("Cannot update an incomplete or failed batch")
        try:
            metrics = update(self.phase, self.stamp)
        except BaseException:
            self.failed = True  # Optimizer may have partially changed; restore a checkpoint.
            raise
        if (not isinstance(metrics, dict) or any(
                not isinstance(metrics.get(key), dict) or type(metrics[key].get("updated")) is not bool
                for key in ("planner", "executor"))):
            self.failed = True
            raise RuntimeError("Both trainers must explicitly report whether parameters changed")
        self.stamp = replace(self.stamp, **{
            key: getattr(self.stamp, key) + int(metrics[key]["updated"])
            for key in ("planner", "executor")})
        self.committed_batches.add(self.batch_id)
        self.batch_id, self.completed = None, set()
        return metrics

    def state_dict(self):
        if self.batch_id is not None or self.failed:
            raise RuntimeError("Only committed states can be checkpointed")
        return {"schema_version": 2, "stamp": asdict(self.stamp), "phase": self.phase.value,
                "committed_batches": sorted(self.committed_batches)}

    @classmethod
    def from_state_dict(cls, state):
        if state["schema_version"] != 2:
            raise ValueError("Joint training requires a joint checkpoint; alternating checkpoints cannot resume")
        obj = cls(PolicyStamp(**state["stamp"]), Phase(state["phase"]))
        obj.committed_batches = set(state["committed_batches"])
        return obj


def save_checkpoint(destination, coordinator, payload):
    """One atomic directory publication contains both policies and optimizer states."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    state = coordinator.state_dict()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=destination.parent) as temporary:
        folder = Path(temporary)
        torch.save(payload, folder / "policies.pt")
        checksum = file_sha256(folder / "policies.pt")
        state["policies_sha256"] = checksum
        (folder / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        for name in ("policies.pt", "state.json"):
            with (folder / name).open("rb") as handle:
                os.fsync(handle.fileno())
        folder.rename(destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return destination


def load_checkpoint(directory, *, map_location="cpu"):
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text())
    if state.get('format') == 'split-policies-v1':
        if any(file_sha256(directory/name) != sha for name,sha in state['files'].items()):
            raise ValueError('Checkpoint payload changed')
        planner = torch.load(directory/'planner.pt',map_location=map_location,weights_only=True)
        executor = torch.load(directory/'executor.pt',map_location=map_location,weights_only=True)
        return JointCoordinator.from_state_dict(state), {**planner, **executor}
    if file_sha256(directory / "policies.pt") != state["policies_sha256"]:
        raise ValueError("Checkpoint payload changed")
    return JointCoordinator.from_state_dict(state), torch.load(
        directory / "policies.pt", map_location=map_location, weights_only=True)


def freeze_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True).contiguous()
    if isinstance(value, dict):
        return {k: freeze_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [freeze_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(freeze_cpu(v) for v in value)
    return copy.deepcopy(value)


class SnapshotWriter:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint')
        self.future = None

    def wait(self):
        if self.future is not None:
            result = self.future.result()
            self.future = None
            return result

    def submit(self, function, *args, **kwargs):
        self.wait()
        self.future = self.pool.submit(function, *args, **kwargs)

    def check(self):
        if self.future is not None and self.future.done():
            return self.wait()

    def close(self):
        try:
            self.wait()
        finally:
            self.pool.shutdown(wait=True)


def write_executor(path, payload):
    path = Path(path)
    started = time.monotonic()
    try:
        torch.save(payload, path)
        atomic_json(path.parent/'executor-ready.json', {'seconds': time.monotonic()-started})
    except BaseException as error:
        atomic_json(path.parent/'executor-error.json', {'error': repr(error)})
        raise


def wait_executor(staging, timeout=1200):
    started = time.monotonic()
    while not (Path(staging)/'executor-ready.json').is_file():
        error = Path(staging)/'executor-error.json'
        if error.exists():
            raise RuntimeError(error.read_text())
        if time.monotonic()-started > timeout:
            raise TimeoutError('Executor checkpoint writer did not finish')
        time.sleep(.1)
