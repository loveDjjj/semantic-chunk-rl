"""Run a joint warm start through collection, checkpoint publication and reload on CPU."""

from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from scrl.plan import FinitePlan, PlanLimits, PolicyStamp
from scrl.runtime.worker import PlanningWorker
from scrl.rl import train as trainer
from scrl.rl.checkpoint import load_checkpoint
from scrl.rl.checkpoint import write_executor


PLAN = {"allowed_chunks": [3], "max_segments": 1, "max_total_chunks": 3}


class TrainingPlannerConfigTest(unittest.TestCase):
    def test_sampler_and_forced_plan_use_recipe_limits(self):
        cfg = {"assets": {"planner_model": "model", "planner_adapter": "sft"}, "plan": PLAN,
               "training": {"mode": "planner-only"}}
        with patch("scrl.runtime.worker.PlannerPolicy") as constructor:
            PlanningWorker(cfg, node=0, device="cpu")
        limits = constructor.call_args.kwargs["limits"]
        self.assertEqual(limits, PlanLimits((3,), 1, 3))
        row = {"segments": [{"caption": "scene", "num_chunks": 3}]}
        self.assertEqual(FinitePlan.from_dict(row, limits).total_chunks, 3)
        with self.assertRaises(ValueError):
            FinitePlan.from_dict(row)
        with self.assertRaises(ValueError):
            FinitePlan.from_dict({"segments": row["segments"] * 2}, limits)
        with self.assertRaises(ValueError):
            PlanLimits([4], 1, 3)

    def test_joint_init_collect_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as patches:
            root = Path(tmp)
            init, run = root / "init", root / "run"
            init.mkdir()
            run.mkdir()
            weight = torch.tensor([[7.]])
            torch.save({"planner": {"weight": weight}, "job_index": 12}, init / "planner.pt")
            stamp = asdict(PolicyStamp())
            cfg = dict(run=str(run), token="test", nodes=[{"address": "local"}], ports={"planner": 2, "evaluation": 3},
                       executors=[{"address": "local", "port": 1}], prompts=["scene"], batch_prompts=1, seed=10,
                       assets={"planner_model": "model", "planner_adapter": "sft"}, plan=PLAN, reward={},
                       training=dict(mode="joint", planner_learning_rate=.1, clip_epsilon=.2, kl_beta=.01),
                       stamp=stamp, full_dataset=True, adaptive_sampling=[], fixed_sampling=[2, 1],
                       dataset_sha256="test", init_from=str(init), checkpoint_interval=1)
            actor = Mock(version=0)
            actor.model = torch.nn.Linear(1, 1, bias=False)
            actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=.1, momentum=.9)

            def update(*args, **kwargs):
                actor.model.weight.sum().backward()
                actor.optimizer.step()
                actor.version += 1
                return {"updated": True}

            actor.update.side_effect = update

            def rpc(op, **req):
                if op == "begin_batch":
                    return {"version": 0}
                if op == "plan":
                    plans = [{"segments": [{"caption": f"scene {i}", "num_chunks": 3}]} for i in range(2)]
                    torch.save([0, 1], req["output"])
                    return dict(samples=req["output"], plans=plans, target=0, target_probability=1.,
                                prefix_segments=[], prefix_seed=10)
                if op == "generate":
                    row = dict(group_id=req["group_id"], candidate_id=req["candidate"], seed=req["seeds"][0],
                               original_prompt=req["prompt"], prefix_id="same", stamp=stamp, plan=req["plan"],
                               target_segment=0, planner_return=float(req["candidate"] == "c1"), executor_return=0.)
                    return dict(job_id=req["job_id"], physical_videos=1, rewards=[{"seed": row["seed"]}], rows=[row])
                if op == "finalize":
                    write_executor(req["checkpoint"], {"executor": {}})
                    return dict(updated=False, replica_sha256="same", replay_max_error=0., version=0)
                if op == "stop":
                    return {}
                raise AssertionError(op)

            peer = Mock()
            peer.call.side_effect = rpc
            ctor = patches.enter_context(patch.object(trainer, "VerlPlannerPolicy", return_value=actor))
            patches.enter_context(patch.object(trainer, "Peer", return_value=peer))
            patches.enter_context(patch.object(trainer, "cpu_state", side_effect=lambda model: model.state_dict()))
            patches.enter_context(patch("scrl.config.load_run", return_value=cfg))
            patches.enter_context(patch("scrl.models.adapters.restore_adapter", side_effect=lambda m, s: m.load_state_dict(s)))
            patches.enter_context(patch("sys.argv", ["train", "--config", "test"]))
            patches.enter_context(patch("os.umask"))
            patches.enter_context(patch.object(torch, "set_num_threads"))
            patches.enter_context(patch.object(torch.cuda, "set_device"))
            patches.enter_context(patch.object(torch.cuda, "get_rng_state", return_value=torch.get_rng_state()))
            trainer.main()

            coordinator, saved = load_checkpoint(run / "checkpoints" / "latest")
            self.assertEqual(ctor.call_args.kwargs["limits"], PlanLimits((3,), 1, 3))
            self.assertEqual((saved["job_index"], saved["cursor"], coordinator.stamp.planner), (1, 1, 1))
            torch.testing.assert_close(saved["planner"]["weight"], weight - .1)
            self.assertTrue(saved["planner_optimizer"]["state"])
