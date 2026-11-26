"""verl PPO actor integration for finite-plan sampling.

Uses verl's microbatch/update loop and optimizer path, with a constrained forward
and weighted response mask. The rollout engine remains Transformers in this first
integration; it does not claim vLLM batching or distributed throughput gains.
"""

import importlib.metadata
import math
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from scrl.models.planner import CrossedPlannerSample, PlannerPolicy, crossed_terms, token_log_probs, whole_mask


def make_actor(policy):
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    class ConstrainedActor(DataParallelPPOActor):
        def __init__(self):
            config = OmegaConf.create({
                "use_remove_padding": False, "use_fused_kernels": False,
                "ulysses_sequence_parallel_size": 1, "entropy_from_logits_with_chunking": False,
                "use_torch_compile": False, "grad_clip": 1.0, "use_kl_loss": True,
                "kl_loss_coef": .01, "kl_loss_type": "low_var_kl", "entropy_coeff": 0.,
                "ppo_epochs": 1, "ppo_mini_batch_size": 1, "ppo_micro_batch_size_per_gpu": 1,
                "use_dynamic_bsz": False, "clip_ratio": .2, "clip_ratio_low": None,
                "clip_ratio_high": None, "clip_ratio_c": 3.,
                "loss_agg_mode": "seq-mean-token-sum", "policy_loss": {"loss_mode": "vanilla"},
            })
            super().__init__(config, policy.model, policy.optimizer)
            self.constraints_by_input = {}
            self.field_supports = None
            self.applied_update = False

        def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
            if temperature != 1.0 or calculate_entropy:
                raise ValueError("Constrained actor requires temperature=1 and disabled entropy bonus")
            self.actor_module.eval()  # Match rollout dropout semantics.
            width = micro_batch["responses"].shape[1]
            prompt_width = micro_batch["input_ids"].shape[1] - width
            outputs = []
            for i, (ids, attention) in enumerate(zip(micro_batch["input_ids"], micro_batch["attention_mask"])):
                selected = ids[attention.bool()]
                if self.field_supports is not None:
                    if attention[0]:
                        raise ValueError("Missing field replay index")
                    support = self.field_supports[int(micro_batch["position_ids"][i, 0])]
                else:
                    key = tuple(selected.detach().cpu().tolist())
                    if key not in self.constraints_by_input:
                        raise ValueError("Actor batch lost its original sampling grammar")
                    support = self.constraints_by_input[key]
                pc = int(attention[:prompt_width].sum())
                context, response = selected[:pc][None], selected[pc:][None]
                logps = token_log_probs(self.actor_module, context, response, support)
                outputs.append(torch.nn.functional.pad(logps, (0, width - logps.numel())))
            return None, torch.stack(outputs)

        def _optimizer_step(self):
            nonzero = any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                          for p in self.actor_module.parameters() if p.requires_grad)
            if not nonzero:
                self.actor_optimizer.zero_grad(set_to_none=True)
                return torch.zeros((), device=policy.device)
            norm = super()._optimizer_step()
            if not torch.isfinite(norm):
                raise RuntimeError("verl actor produced a nonfinite gradient")
            self.applied_update = True
            return norm

    return ConstrainedActor()


class VerlPlannerPolicy(PlannerPolicy):
    def __init__(self, *args, **kwargs):
        version = importlib.metadata.version("verl")
        if version != "0.5.0":
            raise RuntimeError(f"This actor adapter is pinned to verl 0.5.0, found {version}")
        super().__init__(*args, **kwargs)
        self._owns_process_group = not dist.is_initialized()
        self._rendezvous = None
        if self._owns_process_group:
            fd, self._rendezvous = tempfile.mkstemp(prefix="semantic-chunk-rl-dist-")
            os.close(fd)
            torch.cuda.set_device(self.device)
            dist.init_process_group("nccl", init_method="file://" + self._rendezvous,
                                    rank=0, world_size=1)
        elif dist.get_world_size() != 1:
            raise RuntimeError("Use a dedicated planner worker group; this adapter currently validates one rank")
        self.actor = make_actor(self)

    def update(self, samples, advantages, target_segments, *, clip_epsilon=.2, kl_beta=.01,
               duration_weight=.5, max_grad_norm=1.0):
        if not samples or not (len(samples) == len(advantages) == len(target_segments)):
            raise ValueError("Incomplete planner update")
        if any(isinstance(s, CrossedPlannerSample) for s in samples):
            terms = crossed_terms(samples, advantages, target_segments, version=self.version,
                                  duration_weight=duration_weight)
            if not terms:
                return {"updated": False, "backend": "verl", "gradient_norm": 0.0}
            tensors = crossed_batch(terms, pad_token_id=self.tokenizer.pad_token_id, device=self.device)
            metadata = {"policy_version": self.version}
            self.actor.field_supports = [t.constraints for t, _, _ in terms]
        elif all(t is None for t in target_segments):
            if any(s.policy_version != self.version or s.forced_termination for s in samples):
                raise ValueError("Stale or truncated whole-plan trace")
            tensors = whole_batch(samples, advantages, pad_token_id=self.tokenizer.pad_token_id, device=self.device)
            self.actor.field_supports = [s.constraints for s in samples]
            metadata = {"policy_version": self.version}
        else:
            if any(s.policy_version != self.version or s.forced_termination for s in samples):
                raise ValueError("Stale or resource-truncated plan")
            if not 0 <= duration_weight <= 1 or not all(math.isfinite(a) for a in advantages):
                raise ValueError("Invalid field weight or advantage")
            self.actor.constraints_by_input = {
                tuple(torch.cat((s.context, s.response), 1)[0].tolist()): s.constraints for s in samples}
            tensors, metadata = focal_batch(samples, advantages, target_segments,
                pad_token_id=self.tokenizer.pad_token_id, device=self.device)
            caption, duration = tensors["caption_mask"].float(), tensors["duration_mask"].float()
            tensors["response_mask"] = ((1 - duration_weight) * caption / caption.sum(1, keepdim=True)
                                        + duration_weight * duration / duration.sum(1, keepdim=True))
        from verl import DataProto
        self.activate("default")
        metadata.update(temperature=1., micro_batch_size=1, use_dynamic_bsz=False)
        batch = DataProto.from_dict(tensors=tensors, meta_info=metadata)
        config = self.actor.config
        config.ppo_mini_batch_size = tensors["responses"].shape[0]
        config.clip_ratio, config.kl_loss_coef, config.grad_clip = clip_epsilon, kl_beta, max_grad_norm
        self.actor.applied_update = False
        try:
            metrics = self.actor.update_policy(batch)
        finally:
            self.actor.field_supports = None
            self.actor.constraints_by_input = {}
        updated = self.actor.applied_update
        if updated:
            self.version += 1
        self.model.eval()
        return {"updated": updated, "backend": "verl", "verl_version": "0.5.0",
                "gradient_norm": float(metrics["actor/grad_norm"][-1]), "metrics": metrics}

    def close(self):
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        if self._rendezvous:
            Path(self._rendezvous).unlink(missing_ok=True)


def _replay_batch(samples, advantages, *, pad_token_id, device, field_indices=False):
    prompt_width = max(s.context.shape[1] for s in samples) + int(field_indices)
    response_width = max(s.response.shape[1] for s in samples)
    batch_size = len(samples)
    prompts = torch.full((batch_size, prompt_width), pad_token_id, dtype=torch.long, device=device)
    responses = torch.full((batch_size, response_width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((batch_size, prompt_width + response_width), dtype=torch.long, device=device)
    old = torch.zeros((batch_size, response_width), device=device)
    reference = old.clone()
    for i, sample in enumerate(samples):
        pc, rc = sample.context.shape[1], sample.response.shape[1]
        prompts[i, -pc:] = sample.context[0].to(device)
        responses[i, :rc] = sample.response[0].to(device)
        attention[i, prompt_width - pc:prompt_width + rc] = 1
        old[i, :rc] = sample.old_log_probs.to(device)
        reference[i, :rc] = sample.reference_log_probs.to(device)
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    if field_indices:
        # verl 0.5 selects a fixed set of tensors. A masked padding position carries
        # the replay index through microbatch splitting, never into the actual LM.
        # Identical token IDs can have different legal adjacent-duration budgets.
        positions[:, 0] = torch.arange(batch_size, device=device)
    return {"prompts": prompts, "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=1),
            "attention_mask": attention,
            "position_ids": positions, "old_log_probs": old,
            "ref_log_prob": reference,
            "advantages": torch.as_tensor(advantages, device=device)[:, None].expand_as(old)}


def crossed_batch(terms, *, pad_token_id, device):
    traces, advantages, weights = zip(*terms)
    batch = _replay_batch(traces, advantages, pad_token_id=pad_token_id, device=device, field_indices=True)
    active = batch["attention_mask"][:, -batch["responses"].shape[1]:].float()
    # verl averages field rows; convert back to the logical cell denominator.
    weights = torch.as_tensor(weights, device=device)[:, None] * len(terms)
    batch["response_mask"] = weights * active / active.sum(1, keepdim=True)
    return batch


def whole_batch(samples, advantages, *, pad_token_id, device):
    if len(samples) != len(advantages) or not all(math.isfinite(a) for a in advantages):
        raise ValueError("Invalid whole-plan advantages")
    batch = _replay_batch(samples, advantages, pad_token_id=pad_token_id, device=device, field_indices=True)
    masks = torch.zeros_like(batch["responses"], dtype=torch.float32)
    for i, sample in enumerate(samples):
        mask = whole_mask(sample, device)
        masks[i,:len(mask)] = mask.float() / mask.sum()
    batch["response_mask"] = masks
    return batch


def focal_batch(samples, advantages, targets, *, pad_token_id, device):
    if not samples or not (len(samples) == len(advantages) == len(targets)):
        raise ValueError("Incomplete focal batch")
    if len({s.policy_version for s in samples}) != 1:
        raise ValueError("Mixed planner policy versions")
    batch = _replay_batch(samples, advantages, pad_token_id=pad_token_id, device=device)
    caption = torch.zeros_like(batch["responses"], dtype=torch.bool)
    duration = torch.zeros_like(caption)
    for i, (sample, target) in enumerate(zip(samples, targets)):
        rc = sample.response.shape[1]
        caption[i, :rc], duration[i, :rc] = sample.decisions[target].masks(rc, device)
    batch.update(response_mask=caption | duration, caption_mask=caption, duration_mask=duration)
    return batch, {
            "sampling_constraints": [s.constraints for s in samples],
            "policy_version": samples[0].policy_version,
            "requires_constrained_log_probs": True,
            "target_segments": list(targets)}


def to_dataproto(samples, advantages, targets, *, pad_token_id, device):
    """Optional dependency import occurs only when the verl boundary is used."""
    from verl import DataProto
    batch, metadata = focal_batch(samples, advantages, targets, pad_token_id=pad_token_id, device=device)
    return DataProto.from_dict(tensors=batch, meta_info=metadata)


def reference_actor_loss(model, samples, advantages, targets, *, device, **loss_options):
    """Correct reference actor implementation for testing a distributed actor port."""
    from scrl.rl.objectives import planner_loss
    losses = []
    for sample, advantage, target in zip(samples, advantages, targets, strict=True):
        new = token_log_probs(model, sample.context.to(device), sample.response.to(device), sample.constraints)
        losses.append(planner_loss(new, sample.old_log_probs.to(device),
            sample.reference_log_probs.to(device), advantage, sample.decisions[target], **loss_options))
    if not losses:
        raise ValueError("Empty actor batch")
    return torch.stack(losses).mean()
