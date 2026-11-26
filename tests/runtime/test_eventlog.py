import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scrl import cli
from scrl.eventlog import event


class EventLogTests(unittest.TestCase):
    def test_cli_failure_is_recorded_without_arguments_or_credentials(self):
        def fail(argv):
            raise ValueError("private input")
        with TemporaryDirectory() as tmp, patch.dict(os.environ, SCRL_LOG_DIR=tmp), \
                patch.dict(cli.COMMANDS["rl"], check=(fail, "")):
            with self.assertRaises(ValueError):
                cli.main(["rl", "check", "--secret", "private input"])
            text = next(Path(tmp).glob("*.jsonl")).read_text()
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual([r["state"] for r in rows], ["started", "failed"])
            self.assertEqual(rows[-1]["error_type"], "ValueError")
            self.assertNotIn("private", text)

    def test_worker_uses_run_logs_without_override(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, SCRL_LOG_DIR=""):
            event("executor", "ready", run=tmp, node=2, rank=1)
            row = json.loads(next((Path(tmp)/"logs").glob("*.jsonl")).read_bytes())
            self.assertEqual((row["component"],row["node"],row["rank"]), ("executor",2,1))
