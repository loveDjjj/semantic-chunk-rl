import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scrl.config import DEFAULT_CONFIGS, compose, file_hash, load_run, prepare, read, runtime_cluster, stage_cluster
from scrl.rl.identity import training_fingerprint


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        prompts = self.write("prompts.json", [{"prompt": "A dancer turns."}])
        recipe = read(DEFAULT_CONFIGS / "rl.json")
        recipe["dataset_sha256"] = file_hash(prompts)
        paths = read(DEFAULT_CONFIGS / "paths.example.json")
        paths.update(asset_root=str(self.root), dataset=prompts.name)
        self.args = SimpleNamespace(recipe=self.write("rl.json", recipe), paths=self.write("paths.json", paths),
                                    cluster=DEFAULT_CONFIGS / "cluster.example.json", approvals=self.root / "pins.json",
                                    run=self.root / "run", rounds=1, resume=None)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def approved_inputs(self):
        # 真实的教师清单及模型输入核验另有测试，此处只替换大权重的核验结果。
        return patch("scrl.rl.preflight.audit", return_value={
            "input_blockers": {}, "spec_sha256": "a" * 64, "checked_sha256": {
                "fresh_origin": "b" * 64, "reviewed_teacher_manifest": "c" * 64,
                "teacher_sft_planner": "d" * 64, "trainable_executor": "e" * 64}})

    def test_missing_inputs_cannot_publish_run(self):
        self.write("pins.json", {})
        with self.assertRaisesRegex(ValueError, "teacher_sft_planner"):
            prepare(self.args)
        self.assertFalse(self.args.run.exists())

    def test_unsupported_scoring_settings_fail_before_prepare(self):
        for target, key, value in (("training", "score_attempts", True),
                                   ("training", "score_attempts", 3),
                                   ("reward_protocol", "event_decisions", "typo"),
                                   (None, "event_cache_backend", "typo"),
                                   (None, "run_local_contract_cache", False)):
            recipe = read(DEFAULT_CONFIGS / "rl.json")
            (recipe[target] if target else recipe)[key] = value
            self.write(self.args.recipe, recipe)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "score_attempts|event"):
                prepare(self.args)
            self.assertFalse(self.args.run.exists())

    def test_crossed_shape_and_event_protocol_fail_before_prepare(self):
        recipe = read(self.args.recipe)
        recipe["fixed_sampling"] = [2, 4]
        recipe["sampling"]["planner_candidates"] = 2
        self.write("rl.json", recipe)
        with self.assertRaisesRegex(ValueError, "four cells"):
            prepare(self.args)
        recipe["fixed_sampling"] = [4, 4]
        recipe["sampling"]["planner_candidates"] = 4
        recipe["reward_protocol"]["version"] = "legacy"
        self.write("rl.json", recipe)
        with self.assertRaisesRegex(ValueError, "must match"):
            prepare(self.args)
        self.assertFalse(self.args.run.exists())

    def test_published_config_names_both_policies_without_credentials(self):
        private = read(self.args.cluster)
        private.update(controller_password="dummy-controller-value")
        for node in private["nodes"]:
            node.update(username="dummy-node-user", password="dummy-node-password", future_secret="dummy-future-secret")
        self.args.cluster = self.write("private-cluster.json", private)
        with self.approved_inputs():
            prepare(self.args)
        path = self.args.run / "config.json"
        cfg = read(path)
        self.assertEqual(cfg["training"]["mode"], "joint")
        self.assertEqual(cfg["executor_count"], 14)
        self.assertEqual({(s['node'], s['device']) for s in cfg["evaluations"]},
                         {(n, f'cuda:{g}') for n in (3, 4) for g in range(4)})
        self.assertEqual(len(cfg["planners"]), 2)
        self.assertEqual(cfg["evaluation_node"]["physical_gpus"], [2, 3])
        self.assertEqual([r['device'] for r in cfg['executors'] if r['node'] == 5],
                         ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
        self.assertEqual(cfg["stamp"]["planner_reference"], "d" * 64)
        self.assertEqual(cfg["stamp"]["executor_reference"], "longlive:" + "e" * 64)
        self.assertNotIn("token", cfg)
        self.assertTrue(all(set(n) <= {"index", "address", "gpus", "physical_gpus"} for n in cfg["nodes"]))
        self.assertNotIn("dummy-", json.dumps(runtime_cluster(private)))
        for artifact in self.args.run.iterdir():
            if artifact.is_file():
                self.assertNotIn("dummy-", artifact.read_text())
        with patch.dict(os.environ, SCRL_RPC_TOKEN="x" * 32):
            self.assertEqual(load_run(path)["token"], "x" * 32)
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_run(path)

    def test_preparing_without_watcher_never_reclaims_reserved_cards(self):
        recipe = read(self.args.recipe)
        recipe.pop("vbench_watch", None)
        self.write("rl.json", recipe)
        with self.approved_inputs():
            cfg = compose(self.args)
        self.assertEqual(cfg["evaluation_node"]["index"], 5)
        self.assertNotIn(cfg["evaluation_node"]["address"], [r["address"] for r in cfg["nodes"]])
        self.assertEqual((cfg["executor_count"], len(cfg["evaluations"]), len(cfg["planners"])), (14, 8, 2))

    def test_stage_layout_respects_inventory_whitelist_and_explicit_legacy_layout(self):
        inventory = read(self.args.cluster)
        template = read(self.args.cluster)
        inventory["nodes"][3]["address"] = "198.51.100.4"
        inventory["service_cards"] = [[2,2],[2,3],[3,2],[3,3],[4,2],[4,3]]
        inventory.pop("reserved_inventory_node")
        public = stage_cluster(inventory, template)
        self.assertEqual(public["nodes"][3]["address"], "198.51.100.4")
        self.assertEqual((public["service_cards"], public["reserved_inventory_node"]), (template["service_cards"], 5))
        self.assertEqual(stage_cluster(inventory, None), runtime_cluster(inventory))
        inventory["nodes"][3]["physical_gpus"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "devices differ"):
            stage_cluster(inventory, template)

    def test_rejected_cluster_env_does_not_echo_private_values(self):
        cluster = read(self.args.cluster)
        cluster["communication_env"]["NCCL_SECRET"] = "private-sentinel"
        with self.assertRaisesRegex(ValueError, "unapproved") as raised:
            runtime_cluster(cluster)
        self.assertNotIn("private-sentinel", str(raised.exception))

    def test_resume_requires_both_files_and_identical_reward(self):
        with self.approved_inputs():
            cfg = compose(self.args)
            self.write("parent/config.json", cfg)
            cp = self.root / "parent/checkpoints/checkpoint-00001"
            files = {name: file_hash(self.write(cp / name, name)) for name in ("planner.pt", "executor.pt")}
            self.write(cp / "state.json", {"format": "split-policies-v1", "stamp": cfg["stamp"], "files": files})
            self.args.resume = cp
            self.assertEqual(training_fingerprint(compose(self.args)), training_fingerprint(cfg))
            recipe = read(self.args.recipe)
            recipe["reward"]["global_task_floor"] -= .1
            self.write(self.args.recipe, recipe)
            with self.assertRaisesRegex(ValueError, "identity"):
                compose(self.args)
            (cp / "executor.pt").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "Damaged"):
                compose(self.args)
