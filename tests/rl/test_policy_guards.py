"""CPU policy identity checks, independent of training or generation GPUs."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scrl.plan import FinitePlan, Segment
from scrl.runtime.worker import EvaluationWorker, ExecutionWorker, PlanningWorker
from scrl.rl import train as trainer
from scrl.rl.identity import validate_training_mode


class PolicyGuardTests(unittest.TestCase):
    def test_unknown_mode_fails_before_loading_models_or_allocating_devices(self):
        for name in ("planner-only", "joint"):
            self.assertEqual(validate_training_mode(name), name)
        for name in ("joint ", "frozen", "executor-only", None):
            cfg = {"training": {"mode": name}}
            for worker, args in ((PlanningWorker,(0,)), (ExecutionWorker,(0,0,"cpu")),
                                 (EvaluationWorker,(0,))):
                with self.subTest(name=name, worker=worker), self.assertRaisesRegex(ValueError,"Unsupported training mode"):
                    worker(cfg, *args)
            with patch("scrl.config.load_run", return_value=cfg), patch("os.umask"), \
                 patch("sys.argv", ["train","--config","invalid"]), self.assertRaisesRegex(ValueError,"Unsupported training mode"):
                trainer.main()

    def test_partial_snapshots_never_advance_version_or_sample(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2,3)), LoraConfig(target_modules=["0"],r=1))
        state = {k: torch.ones_like(v) for k,v in get_peft_model_state_dict(model).items()}
        with patch("scrl.runtime.worker.PlannerPolicy") as ctor, tempfile.TemporaryDirectory() as tmp:
            worker = PlanningWorker(dict(assets=dict(planner_model="unused", planner_adapter="unused"),
                                         plan={}, sampling=dict(max_new_tokens=128),training=dict(mode="joint")),0,"cpu")
            policy = ctor.return_value
            policy.model = model
            plan = FinitePlan((Segment("scene",1),))
            policy.sample = Mock(return_value=SimpleNamespace(plan=plan))
            good, bad = (str(Path(tmp)/p) for p in ("good.pt","bad.pt"))
            torch.save(state, good)
            args = dict(snapshot=good, version=1, candidates=1, prompt="scene",seed=1,output=str(Path(tmp)/"out.pt"))
            worker("plan", args)
            key = next(iter(state))
            for invalid in ({k:v for k,v in state.items() if k!=key}, {**state, "extra": torch.ones(1)},
                            {**state,key: torch.ones(4,4)}):
                torch.save(invalid,bad)
                with self.assertRaisesRegex(ValueError,"keys or shapes"):
                    worker("plan",dict(args,snapshot=bad,version=2))
                self.assertEqual(policy.version,1)
                self.assertIsNone(worker.snapshot)
                policy.sample.assert_called_once()
                for k,v in get_peft_model_state_dict(model).items():
                    torch.testing.assert_close(v,state[k])
            worker("plan",args)
            with self.assertRaisesRegex(ValueError,"relabeled"):
                worker("plan",dict(args,version=2))
            def partial_copy(*args, **kwargs):
                with torch.no_grad():
                    next(p for n,p in model.named_parameters() if 'lora_' in n).fill_(9)
                raise RuntimeError("interrupted copy")
            torch.save(state,bad)
            with patch("peft.set_peft_model_state_dict",side_effect=partial_copy), self.assertRaisesRegex(RuntimeError,"interrupted"):
                worker("plan",dict(args,snapshot=bad,version=2))
            self.assertEqual(policy.version,1)
            self.assertIsNone(worker.snapshot)
            worker("plan",args)
            for k,v in get_peft_model_state_dict(model).items():
                torch.testing.assert_close(v,state[k])
