"""Plan a VBench shard in the Qwen runtime, without importing the Wan runtime."""
import json
from pathlib import Path
import sys

from scrl.evaluation.io import read
from scrl.plan import PlanLimits
from scrl.evaluation.generate import load_adapter
from scrl.models.planner import PlannerPolicy
from scrl.evaluation.io import atomic_json


def main():
    import torch
    run, arm, folder = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    cfg = read(run / "config.json")
    assets, arm_cfg = cfg["assets"], cfg["arms"][arm]
    base = arm_cfg.get("planner_variant") == "pre_sft"
    if base and (arm_cfg["checkpoint"] or assets["planner_adapter"] is not None):
        raise ValueError("Pre-SFT comparison cannot use trained planner weights")
    planner = PlannerPolicy(assets["planner_model"], assets["planner_adapter"],
                            limits=PlanLimits(**cfg["plan"]), base_inference=base)
    if arm_cfg["checkpoint"]:
        payload = torch.load(Path(arm_cfg["checkpoint"]) / "planner.pt",
                             map_location="cpu", weights_only=True, mmap=True)
        load_adapter(planner.model, payload["planner"])
    for task in json.load(sys.stdin):
        if (run / "STOP").exists() or (run / ("STOP-" + arm)).exists():
            raise InterruptedError("Evaluation stopped")
        plan, response, _ = planner.sample(task["prompt"], task["planner_seed"],
            max_new_tokens=cfg["max_new_tokens"], anchor_only=True)
        atomic_json(folder / (task["id"] + ".json"),
                    {"plan":plan.to_dict(), "ids":response[0].tolist()})


if __name__ == "__main__":
    main()
