"""Validate rollout groups, assign advantages and optimize the planner and executor."""

from dataclasses import dataclass
import math
from scrl.plan import FinitePlan, PolicyStamp


CROSSED_CREDIT = "crossed-2x2-v1"
WHOLE_CREDIT = "whole-plan-v1"


@dataclass(frozen=True)
class ScoredRollout:
    group_id: str
    candidate_id: str
    seed: int
    original_prompt: str
    prefix_id: str
    stamp: PolicyStamp
    plan: FinitePlan
    target_segment: int
    planner_return: float
    executor_return: float


def normalize(values, floor=0.02, clip=3.0):
    if floor <= 0 or clip <= 0 or len(values) < 2:
        raise ValueError("Need at least two candidates and positive normalization limits")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite reward")
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((x - mean)**2 for x in values) / len(values))
    return [max(-clip, min(clip, (x - mean) / max(std, floor))) for x in values]


def validate_group(rows, expected_stamp, *, whole=False):
    if not rows:
        raise ValueError("Empty training group")
    identities = {(r.group_id, r.original_prompt, r.stamp,
                   None if whole else (r.prefix_id, r.target_segment)) for r in rows}
    if len(identities) != 1 or rows[0].stamp != expected_stamp:
        raise ValueError("Mixed group, prompt, prefix, target or policy version")
    target = rows[0].target_segment
    if any(not 0 <= r.target_segment < len(r.plan.segments) for r in rows):
        raise ValueError("Target is outside a plan")
    if not whole and len({r.plan.segments[:target] for r in rows}) != 1:
        raise ValueError("Candidates do not share the same semantic prefix")
    if len({(r.candidate_id, r.seed) for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate execution")
    if any(not math.isfinite(v) for r in rows for v in (r.planner_return, r.executor_return)):
        raise ValueError("Nonfinite return")


def validate_planner_credit(credit):
    if credit not in ("candidate-v1", "focal-mean-v1", CROSSED_CREDIT, WHOLE_CREDIT):
        raise ValueError(f"Unknown planner_credit: {credit!r}")
    return credit


def candidate_advantages(means, actions, *, credit="candidate-v1", floor=0.02):
    """A focal action is (caption, duration) within one fixed prompt and prefix."""
    validate_planner_credit(credit)
    if credit in (CROSSED_CREDIT, WHOLE_CREDIT):
        raise ValueError("This credit requires complete plans and paired seeds")
    if len(means) != len(actions):
        raise ValueError("Every candidate needs a focal action")
    if credit == "candidate-v1":
        return normalize(means, floor)
    if not all(math.isfinite(v) for v in means):
        raise ValueError("Nonfinite reward")
    by_action = {}
    for action, value in zip(actions, means):
        by_action.setdefault(action, []).append(value)
    pooled = {a: math.fsum(v) / len(v) for a, v in by_action.items()}
    # Preserve the original sampling frequencies and per-candidate gradient slots.
    return normalize([pooled[a] for a in actions], floor)


def _crossed_advantages(plans, target, means, floor):
    """Cells are [caption0/duration0, caption0/duration1, caption1/duration0, caption1/duration1]."""
    if len(plans) != 4:
        raise ValueError("Crossed credit needs exactly four cells")
    base = plans[0]
    for plan in plans:
        if (len(plan.segments) != len(base.segments) or plan.total_chunks != base.total_chunks
                or plan.segments[:target] != base.segments[:target]
                or plan.segments[target+2:] != base.segments[target+2:]
                or (target+1 < len(base.segments) and
                    plan.segments[target+1].caption != base.segments[target+1].caption)):
            raise ValueError("Crossed plans must fix the prefix, suffix, neighbor caption and total duration")
    captions = [p.segments[target].caption for p in plans]
    durations = [p.segments[target].num_chunks for p in plans]
    if captions[0] != captions[1] or captions[2] != captions[3] or durations[:2] != durations[2:]:
        raise ValueError("Plans are not a caption-major Cartesian product")
    def axis(pairs, identical):
        return [0., 0.] if identical else normalize([math.fsum(means[i] for i in pair)/2 for pair in pairs], floor)
    caption = axis(((0, 1), (2, 3)), captions[0] == captions[2])
    duration = axis(((0, 2), (1, 3)), durations[0] == durations[1])
    return [dict(caption=caption[i//2], duration=duration[i%2]) for i in range(4)]


def planner_advantages(rows, expected_stamp, *, expected_candidates, expected_seeds, floor=0.02,
                       credit="candidate-v1"):
    if len(expected_candidates) < 2 or len(set(expected_candidates)) != len(expected_candidates):
        raise ValueError("Need at least two unique planning candidates")
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Execution repeat seeds must be unique")
    whole = credit == WHOLE_CREDIT
    if whole and (len(expected_candidates) != 4 or len(expected_seeds) != 4):
        raise ValueError("Whole-plan credit needs four draws and four paired seeds")
    validate_group(rows, expected_stamp, whole=whole)
    grouped = {}
    for row in rows:
        grouped.setdefault(row.candidate_id, []).append(row)
    if set(grouped) != set(expected_candidates):
        raise ValueError("Incomplete planner candidate group")
    means, actions, plans = [], [], []
    for candidate in expected_candidates:
        items = grouped[candidate]
        if {x.seed for x in items} != set(expected_seeds):
            raise ValueError("Incomplete execution repeats")
        if len({x.plan.fingerprint for x in items}) != 1:
            raise ValueError("One candidate ID refers to different plans")
        if whole and len({(x.target_segment,x.prefix_id) for x in items}) != 1:
            raise ValueError("Inconsistent replay identity within one whole-plan candidate")
        means.append(math.fsum(x.planner_return for x in items) / len(items))
        plans.append(items[0].plan)
        segment = items[0].plan.segments[items[0].target_segment]
        actions.append((segment.caption, segment.num_chunks))
    if whole:
        by_plan = {}
        for plan, mean in zip(plans, means):
            by_plan.setdefault(plan.fingerprint, []).append(mean)
        scores = normalize([math.fsum(by_plan[p.fingerprint])/len(by_plan[p.fingerprint]) for p in plans], floor)
    elif credit == CROSSED_CREDIT:
        if len(expected_seeds) < 2:
            raise ValueError("Crossed groups require multiple paired execution seeds")
        scores = _crossed_advantages(plans, rows[0].target_segment, means, floor)
    else:
        scores = candidate_advantages(means, actions, floor=floor, credit=credit)
    return dict(zip(expected_candidates, scores))


def executor_advantages(rows, expected_stamp, *, expected_seeds, floor=0.02):
    if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Need at least two unique execution branches")
    validate_group(rows, expected_stamp)
    if len({r.plan.fingerprint for r in rows}) != 1 or len({r.candidate_id for r in rows}) != 1:
        raise ValueError("Executor branches must have identical captions, lengths and suffix plan")
    if {r.seed for r in rows} != set(expected_seeds):
        raise ValueError("Incomplete executor group")
    by_seed = {r.seed: r for r in rows}
    return dict(zip(expected_seeds, normalize([by_seed[s].executor_return for s in expected_seeds], floor)))


def joint_advantages(rows, expected_stamp, *, expected_candidates, expected_seeds, floor=0.02,
                     credit="candidate-v1"):
    """Reuse observations, never mix different plans in an Executor comparison."""
    planner = planner_advantages(rows, expected_stamp, expected_candidates=expected_candidates,
                                 expected_seeds=expected_seeds, floor=floor, credit=credit)
    executor = {candidate: executor_advantages(
        [row for row in rows if row.candidate_id == candidate], expected_stamp,
        expected_seeds=expected_seeds, floor=floor) for candidate in expected_candidates}
    return {"planner": planner, "executor": executor}


def planner_loss(new, old, reference, advantage, tokens, *, clip_epsilon=0.2,
                 kl_beta=0.01, duration_weight=0.5):
    import torch
    if new.ndim != 1 or new.shape != old.shape or new.shape != reference.shape:
        raise ValueError("Expected aligned original-token log probabilities")
    if not 0 <= duration_weight <= 1 or not 0 < clip_epsilon < 1 or kl_beta < 0:
        raise ValueError("Invalid planner loss configuration")
    caption, duration = tokens.masks(new.numel(), new.device)
    active = caption | duration
    if not all(torch.isfinite(v[active]).all() for v in (new, old, reference)):
        raise ValueError("Nonfinite active policy probability")
    ratio = (new - old.detach()).exp()
    a = torch.as_tensor(advantage, device=new.device, dtype=torch.float32)
    if not torch.isfinite(a).all():
        raise ValueError("Nonfinite advantage")
    policy = -torch.minimum(ratio * a, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * a)
    delta = reference.detach() - new
    kl = delta.exp() - delta - 1
    loss = ((1 - duration_weight) * (policy[caption] + kl_beta * kl[caption]).mean()
            + duration_weight * (policy[duration] + kl_beta * kl[duration]).mean())
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite planner objective")
    return loss


def consistency_loss(prediction, old_prediction, next_noisy, sigma, advantage,
                     reference_prediction, kl_beta=0.01):
    """Single-update on-policy Gaussian score gradient, averaged per latent element."""
    import torch
    if not 0 < sigma < 1 or kl_beta < 0:
        raise ValueError("Only nondegenerate consistency transitions are stochastic")
    if not (prediction.shape == old_prediction.shape == next_noisy.shape == reference_prediction.shape):
        raise ValueError("Transition shapes differ")
    current = prediction.float()
    old = old_prediction.detach().float()
    delta = current - old
    residual = next_noisy.detach().float() - (1 - sigma) * old
    log_ratio = ((1 - sigma) * residual * delta / sigma**2
                 - 0.5 * (1 - sigma)**2 * delta.square() / sigma**2).mean()
    kl = 0.5 * (1 - sigma)**2 / sigma**2 * (
        current - reference_prediction.detach().float()).square().mean()
    loss = -advantage * log_ratio + kl_beta * kl
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite consistency objective")
    return loss


def reduce_gradients(parameters, bucket_bytes=32*2**20):
    import torch
    import torch.distributed as dist
    def reduce_bucket(bucket):
        flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in bucket])
        dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        offset=0
        for p in bucket:
            p.grad=flat[offset:offset+p.numel()].view_as(p).clone()
            offset+=p.numel()
    bucket,size=[],0
    for p in parameters:
        n=p.numel()*p.element_size()
        if size+n>bucket_bytes and bucket:
            reduce_bucket(bucket);bucket=[];size=0
        bucket.append(p);size+=n
    if bucket:reduce_bucket(bucket)
