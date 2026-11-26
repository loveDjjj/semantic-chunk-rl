from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GitSyncTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.call("git", "init", "-q")
        self.put(".gitignore", (ROOT / ".gitignore").read_bytes())

    def call(self, *args, check=True):
        return subprocess.run(args, cwd=self.repo, env=self.env, capture_output=True, check=check)

    def put(self, name, data):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def scan(self):
        return self.call(sys.executable, str(ROOT / "tools/check_git_sync.py"), check=False)

    def test_clean_snapshot_and_private_untracked_inputs(self):
        self.put("datasets/README.md", b"dataset catalog")
        self.put("configs/cluster.example.json", b'{"nodes":[]}')
        self.put("datasets/sft/private.jsonl", b"data")
        self.put("configs/cluster.runtime.json", b"{}")
        self.call("git", "add", ".")
        result = self.scan()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"3 indexed", result.stdout)

    def test_forced_paths_even_if_working_files_are_removed(self):
        names = ["configs/pins.json", "runs/checkpoint", "known_hosts", "models/m.pt"]
        for name in names:
            self.put(name, b"private")
        self.call("git", "add", "-f", "--", *names)
        for name in names:
            (self.repo / name).unlink()
        result = self.scan()
        self.assertNotEqual(result.returncode, 0)
        for name in names:
            self.assertIn(name.encode(), result.stderr)

    def test_staged_credentials_are_not_disclosed(self):
        self.put("config.json", b'{"nodes":[{"password":"not-for-display"}]}')
        self.call("git", "add", "config.json")
        self.put("config.json", b"{}")
        result = self.scan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"credential field", result.stderr)
        self.assertNotIn(b"not-for-display", result.stderr)

    def test_private_key_and_oversized_blob(self):
        self.put("notes", b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----")
        self.put("payload", bytes(2 * 1024 * 1024 + 1))
        self.call("git", "add", ".")
        result = self.scan()
        self.assertNotEqual(result.returncode, 0)
        for reason in (b"credential signature", b"exceeds 2 MiB"):
            self.assertIn(reason, result.stderr)
