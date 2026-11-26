"""Versioned rewards, with separate global, segment and boundary scopes."""

from dataclasses import dataclass, field
import math
from scrl.plan import digest


class UnscorableReward(ValueError):
    """Missing evidence is not a zero-valued training target."""


@dataclass(frozen=True)
class ScopeScores:
    quality: dict[str, float | None]
    semantic: dict[str, float | None]


@dataclass(frozen=True)
class RewardBundle:
    global_scores: ScopeScores
    segment_scores: ScopeScores
    protocol: str
    boundary_quality: dict[str, float | None] | None = None


@dataclass(frozen=True)
class RewardConfig:
    quality_weight: float = 0.8
    executor_local_weight: float = 0.7
    boundary_weight: float = 0.2
    quality_components: dict[str, float] = field(default_factory=lambda: {
        "imaging": 1.0, "aesthetic": 1.0, "consistency": 1.0, "motion": 1.0})
    semantic_components: dict[str, float] = field(default_factory=lambda: {"alignment": 1.0})
    boundary_components: dict[str, float] = field(default_factory=lambda: {"consistency": 1.0})
    mode: str = "quality-first-v1"
    aggregation: str = "legacy"
    global_task_floor: float = .8
    local_task_floor: float = .7
    completion_weight: float = .6

    def __post_init__(self):
        if self.mode not in {"quality-first-v1","event-v2"} or any(
                not 0 <= x <= 1 for x in (self.global_task_floor,self.local_task_floor,self.completion_weight)):
            raise ValueError("Invalid reward mode or event coefficient")
        if self.aggregation == "global-sq-v1":
            if self.mode != "event-v2" or self.quality_weight != .5:
                raise ValueError("Global S/Q requires event evidence and equal weights")
        elif self.aggregation != "legacy" or not 0.5 < self.quality_weight <= 1:
            raise ValueError("Quality must be the principal positive reward")
        if not 0 <= self.executor_local_weight <= 1 or not 0 <= self.boundary_weight <= 1:
            raise ValueError("Invalid reward mixture")
        for weights in (self.quality_components, self.semantic_components, self.boundary_components):
            if not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
                raise ValueError("Metric weights must be finite and positive")

    @property
    def fingerprint(self):
        from dataclasses import asdict
        return digest(asdict(self))


def aggregate(values, weights):
    total = 0.0
    for name, weight in weights.items():
        value = values.get(name)
        if value is None:
            raise UnscorableReward(f"Missing metric: {name}")
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise UnscorableReward(f"Metric must be calibrated to [0, 1]: {name}")
        total += weight * value
    return total / sum(weights.values())


class RewardComposer:
    def __init__(self, config=RewardConfig(), protocol="quality-semantic-v1"):
        self.config, self.protocol = config, protocol

    def compose(self, bundle):
        if bundle.protocol != self.protocol:
            raise ValueError("Reward protocol changed")
        cfg = self.config
        qg = aggregate(bundle.global_scores.quality, cfg.quality_components)
        if cfg.aggregation == "global-sq-v1":
            semantic = bundle.global_scores.semantic
            sg = aggregate(semantic, {"completion": 1})
            if "order" not in semantic:
                raise UnscorableReward("Missing order applicability")
            if semantic["order"] is not None:
                sg = cfg.completion_weight * sg + (1-cfg.completion_weight) * aggregate(semantic, {"order": 1})
            value = .5 * sg + .5 * qg
            return dict(planner=value, executor=value, **{"global": value},
                        quality_global=qg, semantic_global=sg)
        ql = aggregate(bundle.segment_scores.quality, cfg.quality_components)
        if bundle.boundary_quality is not None:
            qb = aggregate(bundle.boundary_quality, cfg.boundary_components)
            ql = (1 - cfg.boundary_weight) * ql + cfg.boundary_weight * qb
        if cfg.mode == "event-v2":
            def task(values):
                c = aggregate(values, {"completion":1})
                if "order" not in values:
                    raise UnscorableReward("Missing order applicability")
                if values["order"] is None:
                    return c
                return cfg.completion_weight*c + (1-cfg.completion_weight)*aggregate(values, {"order":1})
            sg = task(bundle.global_scores.semantic)
            global_return = sg*(cfg.global_task_floor+(1-cfg.global_task_floor)*qg)
            owned = bundle.segment_scores.semantic.get("owned")
            if type(owned) is not int or owned not in (0,1):
                raise UnscorableReward("Missing local event responsibility")
            sl = aggregate(bundle.segment_scores.semantic, {"completion": 1}) if owned else sg
            local_return = sl*(cfg.local_task_floor+(1-cfg.local_task_floor)*ql) if owned else global_return
        else:
            sg = aggregate(bundle.global_scores.semantic, cfg.semantic_components)
            sl = aggregate(bundle.segment_scores.semantic, cfg.semantic_components)
            global_return = cfg.quality_weight * qg + (1 - cfg.quality_weight) * sg
            local_return = cfg.quality_weight * ql + (1 - cfg.quality_weight) * sl
        executor_return = (cfg.executor_local_weight * local_return
                           + (1 - cfg.executor_local_weight) * global_return)
        return {"planner": global_return, "executor": executor_return,
                "global": global_return, "local": local_return,
                "quality_global": qg, "semantic_global": sg,
                "quality_local": ql, "semantic_local": sl}
