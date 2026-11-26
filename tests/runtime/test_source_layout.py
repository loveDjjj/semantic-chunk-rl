"""Public import names must survive source relocation, wheel and editable installs."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile
from importlib import metadata

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CONFIGS = {
    "sft.json", "sft-selection.json", "rl.json", "rl-whole24.json", "vbench.json",
    "vbench-periodic-prompts.json", "paths.example.json", "cluster.example.json", "launch.example.json",
}
PROBE = """
import pathlib, pickle, scrl.config
from scrl.plan import FinitePlan
from scrl.runtime.launch import JUDGE_LAUNCHER
from scrl.sft.data import prompt_key
assert pickle.loads(pickle.dumps(FinitePlan)) is FinitePlan
assert prompt_key.__module__ == 'scrl.sft.data'
assert (scrl.config.DEFAULT_CONFIGS / 'rl.json').is_file()
assert JUDGE_LAUNCHER.is_file()
print(pathlib.Path(scrl.config.__file__).resolve())
"""


class SourceLayoutTests(unittest.TestCase):
    def invoke(self, cmd, cwd, source=None):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(PYTHONDONTWRITEBYTECODE="1", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        if source is not None:
            env["PYTHONPATH"] = str(source)
        result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_source_from_foreign_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = ROOT/"src"
            self.assertFalse((source/"scrl").exists())
            self.assertIn("usage: scrl", self.invoke([sys.executable,"-m","scrl","--help"], tmp, source))
            self.assertEqual(Path(self.invoke([sys.executable,"-c",PROBE], tmp, source).strip()), source/"config.py")

    def test_clean_wheel_and_editable(self):
        try:
            for package in ("setuptools","wheel","pip"):
                metadata.version(package)
            if int(metadata.version("setuptools").split(".")[0]) < 68:
                self.skipTest("The project's offline build requires setuptools>=68")
        except metadata.PackageNotFoundError as e:
            self.skipTest(f"Offline build dependency not installed: {e}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root/"project"
            project.mkdir()
            for name in ("pyproject.toml","README.md"):
                shutil.copy2(ROOT/name, project/name)
            shutil.copytree(ROOT/"src", project/"src", ignore=shutil.ignore_patterns("__pycache__","*.egg-info"))
            (project/"configs").mkdir()
            for name in PUBLIC_CONFIGS:
                shutil.copy2(ROOT/"configs"/name, project/"configs"/name)
            (project/"configs/cluster.json").write_text('{"private_test_marker":true}')
            (project/"scripts").mkdir()
            shutil.copy2(ROOT/"scripts/serve_video_judge.sh", project/"scripts/serve_video_judge.sh")
            venv.EnvBuilder(system_site_packages=True, with_pip=False).create(root/"venv")
            python = root/"venv/bin/python"
            pip = [str(python),"-m","pip"]
            self.invoke([*pip,"wheel","--no-build-isolation","--no-deps","-w",str(root),str(project)], root)
            wheel, = root.glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                expected = {"scrl/"+p.relative_to(project/"src").as_posix() for p in
                            (project/"src").rglob("*.py") if p != project/"src/scrl.py"}
                self.assertEqual({p for p in archive.namelist() if p.endswith(".py")}, expected)
                config_names = {Path(p).name for p in archive.namelist() if "/configs/" in p}
                self.assertEqual(config_names, PUBLIC_CONFIGS)
                script, = (p for p in archive.namelist() if p.endswith("/scripts/serve_video_judge.sh"))
                self.assertEqual(archive.read(script), (ROOT/"scripts/serve_video_judge.sh").read_bytes())
            self.invoke([*pip,"install","--no-deps",str(wheel)], root)
            module = Path(self.invoke([str(python),"-I","-c",PROBE], root).strip())
            self.assertTrue(module.is_relative_to(root/"venv"))
            self.assertIn("usage: scrl", self.invoke([str(root/"venv/bin/scrl"),"--help"], root))
            self.invoke([*pip,"install","--no-build-isolation","--no-deps","--editable",str(project)], root)
            self.assertEqual(Path(self.invoke([str(python),"-I","-c",PROBE],root).strip()),project/"src/config.py")
            self.assertIn("usage: scrl", self.invoke([str(python),"-I","-m","scrl","--help"], root))
