"""Plan a VBench shard in the Qwen runtime, without importing the Wan runtime."""
import json
import os
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
    tasks=json.load(sys.stdin)
    batch_size=int(os.environ.get('SCRL_EVAL_PLANNER_BATCH_SIZE','8'))
    if batch_size not in (1,4,8,16):
        raise ValueError('SCRL_EVAL_PLANNER_BATCH_SIZE must be 1, 4, 8, or 16')
    if batch_size>1 and int(os.environ.get('SCRL_EVAL_PLANNER_LANES','1'))!=1:
        raise ValueError('Use one Qwen model process per GPU for batched planning')
    print(f'planner_tasks={len(tasks)} planner_batch_size={batch_size}',flush=True)
    for start in range(0,len(tasks),batch_size):
        if (run / "STOP").exists() or (run / ("STOP-" + arm)).exists():
            raise InterruptedError("Evaluation stopped")
        group=tasks[start:start+batch_size]
        if batch_size==1:
            task=group[0]
            plan, response, _ = planner.sample(task["prompt"], task["planner_seed"],
                max_new_tokens=cfg["max_new_tokens"], anchor_only=True)
            results=[(plan,response[0].tolist())]
        else:
            results=planner.sample_eval_batch([t['prompt'] for t in group],[t['planner_seed'] for t in group],
                                              max_new_tokens=cfg['max_new_tokens'])
        for task,(plan,ids) in zip(group,results):
            atomic_json(folder / (task["id"] + ".json"),
                        {"plan":plan.to_dict(),"ids":ids,"planner_batch_size":len(group)})


if __name__ == "__main__":
    main()
