"""Backend-independent semantic-segment execution with immutable shared prefixes."""

from dataclasses import dataclass, field, InitVar
from concurrent.futures import ThreadPoolExecutor
import hashlib
from typing import Any, Protocol
import torch
from scrl.plan import FinitePlan, Segment, digest


_HASH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='prefix-hash')


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    result = hashlib.sha256(header)
    result.update(memoryview(value.view(torch.uint8).numpy()))
    return result.hexdigest()


def storage_signature(value):
    """Track PyTorch writes, aliases, replacement tensors and container changes without rereading GBs."""
    if isinstance(value, torch.Tensor):
        try:
            version = value._version
        except RuntimeError:  # Inference tensors lack version counters: retain content checking.
            version = tensor_hash(value)
        return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype),
                str(value.device), version)
    if isinstance(value, dict):
        return tuple((k, storage_signature(v)) for k,v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(storage_signature(v) for v in value)
    return (type(value).__name__, repr(value))


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    return value


def tree_fingerprint(value, hashes=None):
    if isinstance(value, torch.Tensor):
        return hashes[id(value)] if hashes is not None else tensor_hash(value)
    if isinstance(value, dict):
        return {k: tree_fingerprint(v, hashes) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_fingerprint(v, hashes) for v in value]
    return value


def parallel_tree_fingerprint(value):
    leaves = {}
    # A recursive nested function closes over itself and `leaves`, keeping
    # multi-GiB cache tensors alive until cyclic GC runs. Traverse iteratively
    # so the final reference is released as soon as hashing returns.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            leaves[id(item)] = item
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list,tuple)):
            pending.extend(item)
    if sum(t.numel()*t.element_size() for t in leaves.values()) < 64*2**20:
        return tree_fingerprint(value)
    # Independent tensor SHA256s retain the exact original fingerprint, while
    # hashlib releases the GIL. Four threads per Executor fit a 32-core node.
    hashes = dict(zip(leaves, _HASH_POOL.map(tensor_hash, leaves.values())))
    return tree_fingerprint(value, hashes)


@dataclass(frozen=True)
class PrefixState:
    segments: tuple[Segment, ...]
    executor_version: int
    backend_id: str
    latents: torch.Tensor = field(repr=False, compare=False)
    cache: Any = field(default=None, repr=False, compare=False)
    _copy_inputs: InitVar[bool] = True
    prefix_id: str = field(init=False)
    _storage_signature: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self, _copy_inputs):
        if self.latents.ndim != 5 or self.latents.shape[0] != 1:
            raise ValueError("Prefix latents must be [1, T, C, H, W]")
        if self.latents.shape[1] != sum(s.latent_frames for s in self.segments):
            raise ValueError("Prefix extent does not match committed segments")
        if self.executor_version < 0 or not self.backend_id:
            raise ValueError("Invalid prefix identity")
        if _copy_inputs:
            object.__setattr__(self, "latents", clone_tree(self.latents))
            object.__setattr__(self, "cache", clone_tree(self.cache))
        object.__setattr__(self, "prefix_id", self.current_digest())
        object.__setattr__(self, "_storage_signature", storage_signature((self.latents, self.cache)))

    def current_digest(self):
        return digest({"segments": [(s.caption, s.num_chunks) for s in self.segments],
                       "executor_version": self.executor_version, "backend": self.backend_id,
                       "latent": tensor_hash(self.latents), "cache": parallel_tree_fingerprint(self.cache)})

    def verify(self, *, full=False):
        if (self._storage_signature != storage_signature((self.latents, self.cache)) or
                (full and self.prefix_id != self.current_digest())):
            raise RuntimeError("Committed prefix was mutated")

    def __getstate__(self):
        return {k:v for k,v in vars(self).items() if k != '_storage_signature'}

    def __setstate__(self, state):
        for key,value in state.items():
            object.__setattr__(self,key,value)
        if self.prefix_id != self.current_digest():
            raise ValueError('Serialized prefix content changed')
        object.__setattr__(self,'_storage_signature',storage_signature((self.latents,self.cache)))


class SegmentBackend(Protocol):
    backend_id: str
    latent_shape: tuple[int, int, int]
    dtype: torch.dtype

    def sample(self, prefix: PrefixState, segment: Segment, seed: int, capture: bool, *, retain_cache=True):
        """Return segment latents, a neutral post-segment cache, and optional tape."""


@dataclass
class SegmentRollout:
    plan: FinitePlan
    prefix_id: str
    target_segment: int
    final: PrefixState
    target_prefix: PrefixState
    tape: Any


class SegmentExecutor:
    def __init__(self, backend: SegmentBackend, version=0):
        self.backend, self.version = backend, version

    def empty_prefix(self):
        return PrefixState((), self.version, self.backend.backend_id,
                           torch.empty((1, 0, *self.backend.latent_shape), dtype=self.backend.dtype))

    def append(self, prefix, segment, seed, *, capture=False, retain_cache=True):
        prefix.verify()
        if prefix.executor_version != self.version or prefix.backend_id != self.backend.backend_id:
            raise ValueError("Stale or incompatible prefix")
        latent, cache, tape = self.backend.sample(prefix, segment, seed, capture, retain_cache=retain_cache)
        expected = (1, segment.latent_frames, *self.backend.latent_shape)
        if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
            raise RuntimeError("Invalid segment output")
        prefix.verify()
        joined = torch.cat((prefix.latents, latent.detach().to("cpu")), dim=1)
        # LongLive has already created independent CPU snapshots and releases its
        # live cache before returning. Transfer those allocations instead of
        # copying every history tensor a second time; keep the full SHA256.
        return PrefixState(prefix.segments + (segment,), self.version,
                           self.backend.backend_id, joined, cache,
                           _copy_inputs=not getattr(self.backend, 'owns_sample_outputs', False)), tape

    def build_prefix(self, segments, seed, *, prefix=None):
        state = self.empty_prefix() if prefix is None else prefix
        for segment in segments:
            state, _ = self.append(state, segment, seed)
        return state

    def rollout(self, plan, seed, *, prefix=None, target_segment=0, capture=True, retain_final_cache=True):
        state = self.empty_prefix() if prefix is None else prefix
        start = len(state.segments)
        if plan.segments[:start] != state.segments or not start <= target_segment < len(plan.segments):
            raise ValueError("Plan prefix or focal segment mismatch")
        parent = state.prefix_id
        if (getattr(self.backend,'private_gpu_history',False) and not capture and
                not retain_final_cache and start == target_segment):
            from types import SimpleNamespace
            origin = state
            for index in range(start, len(plan.segments)):
                segment = plan.segments[index]
                more = index+1<len(plan.segments)
                latent,cache,_ = self.backend.sample(state,segment,seed,False,retain_cache=more,private=True)
                if tuple(latent.shape)!=(1,segment.latent_frames,*self.backend.latent_shape) or not torch.isfinite(latent).all():
                    raise RuntimeError('Invalid private segment output')
                joined = torch.cat((state.latents,latent),dim=1)
                state = SimpleNamespace(segments=plan.segments[:index+1], latents=joined,
                    private_gpu_cache=cache, cache=None, backend_id=origin.backend_id,
                    executor_version=self.version,verify=origin.verify)
            final=PrefixState(tuple(plan.segments),self.version,origin.backend_id,joined,None,_copy_inputs=False)
            origin.verify()
            return SegmentRollout(plan,parent,target_segment,final,origin,None)
        tape = target_prefix = None
        for index in range(start, len(plan.segments)):
            if index == target_segment:
                target_prefix = state
            state, observation = self.append(state, plan.segments[index], seed,
                                             capture=capture and index == target_segment,
                                             retain_cache=retain_final_cache or index+1<len(plan.segments))
            if index == target_segment:
                tape = observation
        return SegmentRollout(plan, parent, target_segment, state, target_prefix, tape)


def indexed_noise(seed, start, frames, shape, *, device, dtype, phase="initial"):
    """Absolute-frame noise is independent of future length and allocation size."""
    rows = []
    for frame in range(start, start + frames):
        key = hashlib.sha256(f"{seed}:{phase}:{frame}".encode()).digest()
        generator = torch.Generator(device=device).manual_seed(int.from_bytes(key[:8], "little") % 2**63)
        rows.append(torch.randn((1, 1, *shape), generator=generator, device=device, dtype=dtype))
    return torch.cat(rows, dim=1)
