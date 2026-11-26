"""CPU-only acceptance tests for the documented shell/profile interface."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


class ShellEntrypointsTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cwd = self.root / "outside repo"
        self.cwd.mkdir()
        self.profile = self.root / "profile with spaces" / "commands.json"
        self.profile.parent.mkdir()
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        self.env.pop("SCRL_PROFILE", None)
        self.env.pop("PYTHONPATH", None)

    def write_profile(self, commands):
        self.profile.write_text(json.dumps({"commands": commands}), encoding="utf-8")

    def shell(self, stage, *args, **env):
        result = subprocess.run(
            ["sh", str(SCRIPTS / (stage + ".sh")), *map(str, args)], cwd=self.cwd,
            env={**self.env, **env}, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
        return result

    def test_sft_torchrun_print_preserves_argument_boundaries_and_overrides(self):
        self.write_profile({"sft.train": {
            "python": "/custom/python", "torchrun": ["--standalone", "--nproc-per-node=2"],
            "args": ["--dataset", "dir with spaces"],
        }})
        receipt = json.loads(self.shell("sft", "train", "--profile", self.profile,
                                        "--print-command", "--dataset", "override dir").stdout)
        self.assertEqual(Path(receipt["cwd"]), self.profile.parent)
        self.assertEqual(receipt["argv"], [
            "/custom/python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
            "--module", "scrl", "sft", "train", "--dataset", "dir with spaces", "--dataset", "override dir",
        ])

    def test_environment_profile_for_eval_and_explicit_profile_precedence(self):
        self.write_profile({"eval.vbench": {"args": ["run", "--arm", "base"]}})
        alternate = self.cwd / "other.json"
        alternate.write_text(json.dumps({"commands": {"eval.vbench": {"args": ["different run"]}}}))
        via_env = json.loads(self.shell("vbench", "vbench", "--print-command",
                                        SCRL_PROFILE=str(self.profile)).stdout)
        self.assertEqual(Path(via_env["cwd"]), self.profile.parent)
        self.assertEqual(via_env["argv"][1:], ["-m", "scrl", "eval", "vbench", "run", "--arm", "base"])
        explicit = json.loads(self.shell("vbench", "vbench", "--profile", alternate, "--print-command",
                                         SCRL_PROFILE=str(self.profile)).stdout)
        self.assertEqual(Path(explicit["cwd"]), alternate.parent)
        self.assertEqual(explicit["argv"][1:], ["-m", "scrl", "eval", "vbench", "different run"])

    def test_rl_exec_uses_profile_workdir_and_exact_argv(self):
        # This interpreter only reports argv/cwd; it cannot run a trainer.
        executable = self.profile.parent / "capture python"
        executable.write_text(
            f"#!{sys.executable}\nimport json, os, sys\n"
            "print(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd()}))\n", encoding="utf-8")
        executable.chmod(0o700)
        self.write_profile({"rl.check": {"python": str(executable), "args": ["--candidate", "sample $(literal)"]}})
        receipt = json.loads(self.shell("rl", "check", "--profile", self.profile,
                                        "--approvals", "review file.json").stdout)
        self.assertEqual(Path(receipt["cwd"]), self.profile.parent)
        self.assertEqual(receipt["argv"], [
            "-m", "scrl", "rl", "check", "--candidate", "sample $(literal)", "--approvals", "review file.json",
        ])

    def test_step_help_without_profile_does_not_import_torch(self):
        for wrapper, step in (("sft", "train"), ("rl", "train"), ("vbench", "vbench")):
            with self.subTest(wrapper=wrapper, step=step):
                result = self.shell(wrapper, step, "--help", PYTHONPROFILEIMPORTTIME="1")
                self.assertIn("--help", result.stdout)
                self.assertNotRegex(result.stderr, r"(?m)^import time:.*\|\s*torch(?:\.\S+)?$")

    def test_rl_commands_require_explicit_recipe_before_reading_inputs(self):
        for step, args in (
            ("prepare", ["--paths", "p", "--cluster", "c", "--approvals", "a", "--run", "out"]),
            ("check", ["--candidate", "candidate", "--approvals", "a"]),
        ):
            with self.subTest(step=step):
                result = subprocess.run(
                    ["sh", str(SCRIPTS / "rl.sh"), step, *args],
                    cwd=self.cwd, env=self.env, capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("required: --recipe", result.stderr)
                self.assertFalse((self.cwd / "out").exists())

    def test_example_rl_profile_selects_current_recipe(self):
        profile = SCRIPTS.parent / "configs" / "launch.example.json"
        for step in ("prepare", "check"):
            with self.subTest(step=step):
                cmd = json.loads(self.shell("rl", step, "--profile", profile, "--print-command").stdout)
                index = cmd["argv"].index("--recipe")
                recipe = Path(cmd["cwd"]) / cmd["argv"][index + 1]
                self.assertEqual(recipe.name, "rl-whole24.json")
                self.assertTrue(recipe.is_file())
