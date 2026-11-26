"""Optional LongLive/Wan backend. Upstream source and weights are user supplied."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import torch

from scrl.plan import Segment, digest
from scrl.models.segments import PrefixState, clone_tree, indexed_noise


@dataclass(frozen=True)
class LongLivePaths:
    upstream: str
    wan: str
    backbone: str
    base_adapter: str
    policy_adapter: str | None = None

    def validate(self):
        for name, value in vars(self).items():
            if name == "policy_adapter" and value is None:
                continue
            if not Path(value).exists():
                raise FileNotFoundError(f"Missing {name}: {value}")


def load_longlive(paths: LongLivePaths, device="cuda:0", *, text_encoder=None):
    paths.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("The LongLive backend requires a CUDA device")
    torch.cuda.set_device(device)
    sys.path.insert(0, str(Path(paths.upstream).resolve()))
    from omegaconf import OmegaConf
    import peft
    from self_forcing import longlive_wan_wrapper as wrapper
    from self_forcing.interactive_causal_pipeline import InteractiveCausalInferencePipeline
    from self_forcing.lora_utils import configure_lora_for_model

    wrapper.WAN2_1_CHECKPOINT = str(Path(paths.wan).resolve())
    cfg_root = Path(paths.upstream) / "self_forcing/config"
    cfg = OmegaConf.merge(OmegaConf.load(cfg_root / "default_config.yaml"),
                          OmegaConf.load(cfg_root / "longlive_interactive_inference.yaml"))
    cfg.generator_ckpt, cfg.lora_ckpt = paths.backbone, paths.base_adapter
    pipe = InteractiveCausalInferencePipeline(cfg, device=device, text_encoder=text_encoder)
    checkpoint = torch.load(paths.backbone, map_location="cpu", weights_only=False, mmap=True)
    pipe.generator.load_state_dict(checkpoint["generator"], strict=True)
    del checkpoint
    model = configure_lora_for_model(pipe.generator.model, "generator", cfg.adapter,
                                     is_main_process=False)
    state = torch.load(paths.base_adapter, map_location="cpu", weights_only=False, mmap=True)
    loaded = peft.set_peft_model_state_dict(model, state["generator_lora"])
    if loaded.unexpected_keys or any("lora_" in k for k in loaded.missing_keys):
        raise RuntimeError("LongLive adapter does not match the backbone")
    del state
    model = model.merge_and_unload()
    pipe.generator.model = (peft.PeftModel.from_pretrained(model, paths.policy_adapter,
                            is_trainable=False) if paths.policy_adapter is not None else model)
    pipe.to(dtype=torch.bfloat16)
    pipe.generator.to(device)
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.eval().requires_grad_(False)
    return LongLiveBackend(pipe, paths, device)


@contextmanager
def immutable_history(model):
    """Keep checkpoint recomputation from observing a mutated attention cache."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    original = base.__dict__.get("_apply_cache_updates")
    owned = "_apply_cache_updates" in base.__dict__
    base._apply_cache_updates = types.MethodType(lambda self, *args, **kwargs: None, base)
    try:
        yield
    finally:
        if owned:
            base._apply_cache_updates = original
        else:
            delattr(base, "_apply_cache_updates")


class LongLiveBackend:
    dtype = torch.bfloat16
    latent_shape = (16, 60, 104)
    owns_sample_outputs = True

    def __init__(self, pipe, paths, device):
        self.pipe, self.paths, self.device = pipe, paths, torch.device(device)
        # Paths locate assets; a production run also records content hashes in its manifest.
        self.backend_id = "longlive:" + digest(vars(paths))
        self.conditions = {}
        self.frozen_text_copy = False
        self.reference_loaded = False
        self.version = 0
        if pipe.num_frame_per_block != 3 or len(pipe.denoising_step_list) != 4:
            raise ValueError("Expected native three-latent chunks and four CM steps")

    @contextmanager
    def autocast(self):
        # LoRA masters remain FP32, but autocast already supplies BF16 matmuls.
        # Avoid PEFT's redundant BF16 -> FP32 -> BF16 activation round-trip,
        # which allocates multi-GiB FFN inputs during full-history recaching.
        from peft.helpers import disable_input_dtype_casting
        with torch.autocast("cuda", dtype=self.dtype), disable_input_dtype_casting(self.pipe.generator.model):
            yield

    @torch.no_grad()
    def encode_many(self, captions):
        """Move T5 once per plan, then cache exact embeddings for all its segments."""
        missing = list(dict.fromkeys(c for c in captions if c not in self.conditions))
        if not missing:
            return
        encoder = self.pipe.text_encoder
        tokenizer = encoder.tokenizer
        for caption in missing:
            count = len(tokenizer.tokenizer(tokenizer._clean(caption), truncation=False,
                        add_special_tokens=True)["input_ids"])
            if count > tokenizer.seq_len:
                raise ValueError("Caption exceeds T5 context; silent truncation is disabled")
        with self.autocast():
            if self.frozen_text_copy:
                # T5 is immutable. Drop the temporary GPU weights rather than transfer
                # them back to CPU. Keep one-caption forwards for exact comparisons.
                state = {name: tensor.to(self.device) for name, tensor in
                         (*encoder.named_parameters(), *encoder.named_buffers())}
                for caption in missing:
                    out = torch.func.functional_call(encoder, state, (), {'text_prompts': [caption]})
                    self.conditions[caption] = clone_tree(out)
            else:
                encoder.to(self.device)
                try:
                    batch_size = 4 if self.paths.policy_adapter is None else 1
                    for offset in range(0, len(missing), batch_size):
                        group = missing[offset:offset + batch_size]
                        encoded = encoder(text_prompts=group)
                        for i, caption in enumerate(group):
                            self.conditions[caption] = {k: v[i:i+1].clone() for k,v in encoded.items()}
                finally:
                    encoder.to("cpu")

    @torch.no_grad()
    def encode(self, caption):
        if caption not in self.conditions:
            encoder = self.pipe.text_encoder
            tokenizer = encoder.tokenizer
            count = len(tokenizer.tokenizer(tokenizer._clean(caption), truncation=False,
                        add_special_tokens=True)["input_ids"])
            if count > tokenizer.seq_len:
                raise ValueError("Caption exceeds T5 context; silent truncation is disabled")
            encoder.to(self.device)
            try:
                with self.autocast():
                    self.conditions[caption] = clone_tree(encoder(text_prompts=[caption]))
            finally:
                encoder.to("cpu")
        return {k: v.to(self.device) for k, v in self.conditions[caption].items()}

    def _initialize(self, total, live=None):
        pipe = self.pipe
        pipe.kv_cache1 = pipe.crossattn_cache = None
        pipe.local_attn_size = total
        pipe.generator.seq_len = max(getattr(pipe.generator, "seq_len", 0),
                                      total * pipe.frame_seq_length)
        pipe._set_all_modules_max_attention_size(total)
        for module in pipe.generator.model.modules():
            if hasattr(module, "local_attn_size"):
                module.local_attn_size = total
        capacity = total * pipe.frame_seq_length
        if live is None:
            pipe._initialize_kv_cache(1, dtype=self.dtype, device=self.device, kv_cache_size_override=capacity)
        else:
            # Only private, single-seed history may be consumed in place. Grow one
            # tensor at a time rather than retaining two complete GPU caches.
            for layer in live:
                for key in ('k','v'):
                    old = layer.pop(key)
                    shape = (old.shape[0], capacity, *old.shape[2:])
                    enlarged = old.new_zeros(shape)
                    enlarged[:, :old.shape[1]].copy_(old)
                    layer[key] = enlarged
                    del old
            pipe.kv_cache1 = live
        pipe._initialize_crossattn_cache(1, dtype=self.dtype, device=self.device)

    def prefix_cache_bytes(self, frames):
        # The native Wan1.3B cache is [1, frames*1560, 12, 128], K and V,
        # for every layer. Four seeds share the same immutable focal prefix.
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return frames * self.pipe.frame_seq_length * 12 * 128 * 2 * element_size * self.pipe.num_transformer_blocks

    def _snapshot(self, frames):
        count = frames * self.pipe.frame_seq_length
        # Queue copies on the current stream and wait once. Waiting after every
        # layer's K/V and index tensor otherwise inserts dozens of host stalls.
        # .to('cpu', non_blocking=True) implicitly allocates pinned host output
        # and retains large, differently sized histories in its allocator.
        # Explicit pageable destinations avoid that unbounded retained pool.
        snapshot=[]
        for layer in self.pipe.kv_cache1:
            saved={}
            for key,value in layer.items():
                source=value[:, :count] if key in ('k','v') else value
                output=torch.empty_like(source,device='cpu',pin_memory=False)
                output.copy_(source.detach(),non_blocking=True)
                saved[key]=output
            snapshot.append(saved)
        torch.cuda.current_stream(self.device).synchronize()
        return snapshot

    def _restore(self, cache, frames):
        count = frames * self.pipe.frame_seq_length
        for layer, saved in zip(self.pipe.kv_cache1, cache, strict=True):
            for key in ("k", "v"):
                layer[key][:, :count].copy_(saved[key],non_blocking=True)
            for key in ("global_end_index", "local_end_index"):
                layer[key].copy_(saved[key],non_blocking=True)

    def _recache(self, prefix, condition):
        pipe = self.pipe
        frames = prefix.shape[1]
        if not frames:
            return
        # The KV path ignores block_mask. Avoid building a quadratic unused mask.
        model = pipe.generator.model
        name = "_prepare_blockwise_causal_attn_mask"
        owned, previous = name in model.__dict__, model.__dict__.get(name)
        setattr(model, name, lambda *args, **kwargs: None)
        try:
            pipe._recache_after_switch(prefix, frames, condition)
        finally:
            if owned:
                setattr(model, name, previous)
            else:
                delattr(model, name)

    @torch.no_grad()
    def prepare(self, prefix: PrefixState, segment: Segment):
        prefix.verify()
        if prefix.backend_id != self.backend_id or prefix.executor_version != self.version:
            raise ValueError("Prefix model/version mismatch")
        condition = self.encode(segment.caption)
        live = getattr(prefix,'private_gpu_cache',None)
        self._initialize(prefix.latents.shape[1] + segment.latent_frames,live)
        with self.autocast():
            if prefix.segments:
                if live is None and prefix.cache is None:
                    raise ValueError("This backend needs a committed history cache")
                if live is None:
                    self._restore(prefix.cache, prefix.latents.shape[1])
                self._recache(prefix.latents.to(self.device), condition)
        return condition

    def forward(self, latent, condition, timestep, start, *, text_cache=True):
        times = torch.ones(latent.shape[:2], device=self.device) * timestep
        cross = self.pipe.crossattn_cache if text_cache else [None] * len(self.pipe.kv_cache1)
        return self.pipe.generator(noisy_image_or_video=latent, conditional_dict=condition,
            timestep=times, kv_cache=self.pipe.kv_cache1, crossattn_cache=cross,
            current_start=start * self.pipe.frame_seq_length)[1]

    @torch.no_grad()
    def sample(self, prefix, segment, seed, capture, *, retain_cache=True, private=False):
        condition = self.prepare(prefix, segment)
        start, frames = prefix.latents.shape[1], segment.latent_frames
        transitions = []
        try:
            with self.autocast():
                chunkwise = getattr(self, 'eval_chunkwise', False)
                if chunkwise and capture:
                    raise ValueError('Chunkwise comparison is inference only')
                width = self.pipe.num_frame_per_block if chunkwise else frames
                blocks = []
                for offset in range(start, start + frames, width):
                    noisy = indexed_noise(seed, offset, width, self.latent_shape,
                                          device=self.device, dtype=self.dtype)
                    schedule = self.pipe.denoising_step_list
                    for index, timestep in enumerate(schedule):
                        prediction = self.forward(noisy, condition, timestep, offset)
                        if index + 1 < len(schedule):
                            next_t = schedule[index + 1]
                            innovation = indexed_noise(seed, offset, width, self.latent_shape,
                                device=self.device, dtype=self.dtype, phase=f"renoise:{float(next_t):.8g}")
                            following = self.pipe.scheduler.add_noise(prediction.flatten(0, 1),
                                innovation.flatten(0, 1), next_t * torch.ones(width, device=self.device))
                            following = following.unflatten(0, prediction.shape[:2])
                            if capture:
                                sigma = float(self.pipe.scheduler.get_sigmas(
                                    torch.as_tensor([next_t], device=self.device))[0])
                                transitions.append({"noisy": clone_tree(noisy),
                                    "next_noisy": clone_tree(following), "prediction": clone_tree(prediction),
                                    "sigma": sigma, "timestep": float(timestep)})
                            noisy = following
                    self.forward(prediction, condition, self.pipe.args.context_noise, offset)
                    blocks.append(prediction)
                prediction = blocks[0] if len(blocks)==1 else torch.cat(blocks,dim=1)
            cache = (self.pipe.kv_cache1 if private else self._snapshot(start + frames)) if retain_cache else None
            tape = {"transitions": transitions, "segment": segment,
                    "prefix_id": prefix.prefix_id, "start": start,
                    "executor_version": self.version} if capture else None
            return clone_tree(prediction), cache, tape
        finally:
            self.pipe.kv_cache1 = self.pipe.crossattn_cache = None

    @torch.no_grad()
    def decode(self, latents):
        self.pipe.vae.to(self.device)
        try:
            with self.autocast():
                video = self.pipe.vae.decode_to_pixel(latents.to(self.device), use_cache=False)
                pixels = ((video.float() * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            return pixels[0].permute(0, 2, 3, 1).cpu()
        finally:
            self.pipe.vae.model.clear_cache()
            self.pipe.vae.to("cpu")

    def enable_training(self, learning_rate=1e-6):
        if self.paths.policy_adapter is None:
            raise RuntimeError("The native LongLive executor is frozen")
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        model = self.pipe.generator.model
        if not self.reference_loaded:
            model.load_adapter(self.paths.policy_adapter, adapter_name="reference", is_trainable=False)
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.data = parameter.data.float()
            state = load_file(str(Path(self.paths.policy_adapter) / "adapter_model.safetensors"))
            for name in ("default", "reference"):
                result = set_peft_model_state_dict(model, state, adapter_name=name)
                if result.unexpected_keys:
                    raise RuntimeError("Reference adapter mismatch")
            self.reference_loaded = True
        self._adapter_parameter_cache = None
        self._active_adapter = None
        self.activate("default")
        model.get_base_model().gradient_checkpointing = True
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable Executor LoRA parameters")
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=0)

    def activate(self, adapter):
        model = self.pipe.generator.model
        # Adapters that save full modules retain the general freeze path.
        if getattr(model, 'modules_to_save', None):
            model.set_adapter(adapter)
            self.pipe.requires_grad_(False)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(adapter == 'default' and 'lora_' in name and '.default.' in name)
            self._adapter_parameter_cache = None
            self._active_adapter = None
            return
        key=(id(model),tuple(sorted(model.peft_config)))
        cache=getattr(self,'_adapter_parameter_cache',None)
        if cache is None or cache[0]!=key:
            self.pipe.requires_grad_(False)
            cache=(key,tuple((name,p) for name,p in model.named_parameters() if 'lora_' in name))
            self._adapter_parameter_cache=cache
            self._active_adapter=None
        if getattr(self,'_active_adapter',None)==adapter and model.active_adapters==[adapter]:
            return
        model.set_adapter(adapter)
        for name, parameter in cache[1]:
            parameter.requires_grad_(adapter == "default" and "lora_" in name and ".default." in name)
        self._active_adapter=adapter

    def backward(self, rollout, advantage, *, kl_beta=0.01, weight=1.0, replay_atol=0.02):
        from scrl.rl.objectives import consistency_loss
        tape, prefix = rollout.tape, rollout.target_prefix
        if not self.reference_loaded or tape is None:
            raise RuntimeError("Enable training and capture the focal segment first")
        if tape["executor_version"] != self.version or tape["prefix_id"] != prefix.prefix_id:
            raise ValueError("Stale or mismatched training tape")
        self.activate("default")
        condition = self.prepare(prefix, tape["segment"])
        errors, losses = [], []
        try:
            for transition in tape["transitions"]:
                noisy = transition["noisy"].to(self.device)
                timestep = transition["timestep"]
                with immutable_history(self.pipe.generator.model), torch.no_grad(), self.autocast():
                    self.activate("reference")
                    reference = self.forward(noisy, condition, timestep, tape["start"], text_cache=False)
                    self.activate("default")
                with immutable_history(self.pipe.generator.model), self.autocast():
                    prediction = self.forward(noisy, condition, timestep, tape["start"], text_cache=False)
                    old = transition["prediction"].to(self.device)
                    error = float((prediction.detach().float() - old.float()).abs().max())
                    if error > replay_atol:
                        raise RuntimeError(f"Rollout replay changed before update: {error}")
                    loss = consistency_loss(prediction, old, transition["next_noisy"].to(self.device),
                        transition["sigma"], advantage, reference, kl_beta)
                    (loss * weight / len(tape["transitions"])).backward()
                    errors.append(error)
                    losses.append(float(loss.detach()))
                with torch.no_grad(), self.autocast():
                    self.forward(noisy, condition, timestep, tape["start"])
                del prediction, reference, loss, old
            return {"replay_max_error": max(errors), "losses": losses}
        finally:
            self.activate("default")
            self.pipe.kv_cache1 = self.pipe.crossattn_cache = None
