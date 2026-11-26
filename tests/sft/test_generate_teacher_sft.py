"""Black-box CLI tests using a loopback-only Responses server and disposable runs."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Thread
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLAN = [{"step_index": 0, "prompt": "A stationary view.", "num_chunks": 2}]
ROWS = [{"prompt": "A red umbrella opens.", "group": "group0"},
        {"prompt": "A dancer takes a bow.", "group": "group3"}]
TEST_KEY = "local-test-key-not-a-credential"


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class TeacherCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)
        self.run_dir = self.tmp / "new-run"
        self.prompts = self.tmp / "prompts.jsonl"
        self.exclusions = self.tmp / "excluded.json"
        jsonl(self.prompts, ROWS)
        self.exclusions.write_text(json.dumps([{"prompt_en": "Evaluation prompt.", "group": "eval group"}]))
        self.requests = []
        self.fail_text = None
        self.refuse_text = None
        case = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                case.requests.append((self.path, self.headers.get("Authorization"), body))
                # Malformed planner output must exercise validation, not become a cached label.
                fail = case.fail_text and case.fail_text in str(body["input"])
                text = "not a plan" if fail else json.dumps(PLAN)
                refused = case.refuse_text and case.refuse_text in str(body["input"])
                part = {"type": "refusal", "refusal": "refused"} if refused else {"type": "output_text", "text": text}
                data = json.dumps({"id": "loopback-response", "status": "completed", "model": body["model"],
                                   "output": [{"type": "message", "role": "assistant", "content": [part]}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{server.server_port}/v1"
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)

    def invoke(self, tool, *args, key=TEST_KEY, ok=True):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("OPENAI_API_KEY", None)
        if key is not None:
            env["OPENAI_API_KEY"] = key
        result = subprocess.run([sys.executable, "-m", "scrl", "sft", tool, *map(str, args)],
                                env=env, cwd=self.tmp, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode == 0, ok, msg=result.stderr + result.stdout)
        self.assertNotIn(TEST_KEY, result.stderr + result.stdout)
        return result

    def generate(self, *args, **kwargs):
        return self.invoke("teacher", "--prompts", self.prompts,
                           "--exclude-metadata", self.exclusions, "--run", self.run_dir,
                           "--base-url", self.base, *args, **kwargs)

    def test_recipe_effort_and_missing_field(self):
        path = self.tmp / "teacher recipe.json"
        recipe = {"teacher":{"model":"a-teacher", "workers":1, "attempts":1, "effort":"medium"}}
        path.write_text(json.dumps(recipe))
        receipt = json.loads(self.generate("--config", path, "--dry-run", key=None).stdout)
        self.assertEqual((receipt["model"], receipt["effort"]), ("a-teacher", "medium"))
        del recipe["teacher"]["effort"]
        path.write_text(json.dumps(recipe))
        self.assertIn("effort", self.generate("--config", path, "--dry-run", key=None, ok=False).stderr)
        self.assertFalse(self.requests)

    def test_end_to_end_unreviewed_cache_resume_and_approved_export(self):
        self.generate("--dry-run", key=None)
        self.assertFalse(self.run_dir.exists())
        self.assertFalse(self.requests)
        self.generate()
        self.assertEqual(len(self.requests), 2)
        for path, auth, body in self.requests:
            self.assertEqual((path, auth), ("/v1/responses", "Bearer " + TEST_KEY))
            self.assertEqual((body["model"], body["reasoning"]), ("gpt-6.2-sol", {"effort":"low"}))
        labels = self.run_dir / "review.jsonl"
        raw = labels.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual({(r["prompt"], r["group"]) for r in rows}, {(r["prompt"], r["group"]) for r in ROWS})
        for r in rows:
            self.assertIs(r["approved"], False)
            self.assertIsNone(r["reviewer"])
            self.assertEqual((r["teacher_model"], r["plan"]), ("gpt-6.2-sol", PLAN))
            self.assertRegex(r["template_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue((self.run_dir / "manifest.json").is_file())
        self.assertTrue(any(p.is_file() for p in (self.run_dir / "cache").rglob("*")))
        self.generate()
        self.assertEqual(labels.read_bytes(),raw)
        self.generate("--model", "a-different-model", ok=False)
        self.generate("--effort", "medium", ok=False)
        jsonl(self.prompts, [*ROWS, {"prompt": "A chair turns.", "group": "group7"}])
        self.generate(ok=False)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(labels.read_bytes(),raw)
        output = self.tmp / "dataset"
        def package(path, ok):
            self.invoke("prepare", "--labels", path, "--exclude-metadata", self.exclusions,
                        "--output", output, ok=ok)
        package(labels, False)
        self.assertFalse(output.exists())
        approved = self.tmp / "approved.jsonl"
        jsonl(approved, [{**r,"reviewer":"test-reviewer","approved":True} for r in rows])
        package(approved, True)
        self.assertEqual(json.loads((output/"manifest.json").read_bytes())["counts"], {"train":1,"validation":1})

    def test_resume_after_invalid_responses_and_exhausted_attempts(self):
        self.fail_text = ROWS[1]["prompt"]
        self.generate("--attempts","2",ok=False)
        attempted = [r[2]["input"] for r in self.requests]
        self.assertEqual(sum(ROWS[1]["prompt"] in str(p) for p in attempted), 2)
        self.assertEqual(sum(ROWS[0]["prompt"] in str(p) for p in attempted), 1)
        self.assertFalse((self.run_dir / "review.jsonl").exists())
        self.fail_text = None
        self.generate("--attempts","2")
        self.assertEqual(len(self.requests), 4)
        self.assertIn(ROWS[1]["prompt"], str(self.requests[-1][2]["input"]))
        self.assertEqual(len((self.run_dir/"review.jsonl").read_text().splitlines()), 2)

    def test_preflight_blocks_all_requests(self):
        self.generate(key=None, ok=False)
        self.assertFalse(self.requests)
        second = self.tmp / "holdout.json"
        second.write_text(json.dumps([{"prompt_en":"Another holdout prompt."}]))
        invalid = [{"prompt":"AnOtHeR  Holdout Prompt.", "group":"safe-group"},
                   {"prompt":"A fresh moment.", "group":"EVAL  GROUP"},
                   {**ROWS[0], "prompt":"A RED  UMBRELLA OPENS.", "group":"different-group"}]
        for bad in invalid:
            with self.subTest(bad=bad):
                jsonl(self.prompts, [ROWS[0],bad])
                self.generate("--exclude-metadata",second,ok=False)
                self.assertFalse(self.requests)

    def test_explicit_official_training_and_frozen_wire_template(self):
        official = self.tmp / "official.json"
        official.write_text(json.dumps([{"prompt_en": ROWS[0]["prompt"]}]))
        self.generate("--include-official-metadata", official)
        manifest = json.loads((self.run_dir/"manifest.json").read_bytes())
        self.assertEqual(manifest["official_training_prompt_count"], 1)
        template = json.loads((self.run_dir/"teacher_request_template.json").read_bytes())
        for _, _, payload in self.requests:
            self.assertEqual(payload["instructions"], template["instructions"])
            self.assertEqual(payload["instructions"].count("Example input:"), 5)
            prompt = next(r["prompt"] for r in ROWS if payload["input"].endswith(r["prompt"]))
            self.assertEqual(template["input"].replace("{original_prompt}", prompt), payload["input"])
        output = self.tmp / "dataset"
        labels = self.tmp / "approved.jsonl"
        jsonl(labels, [dict(json.loads(line), approved=True, reviewer="test")
                       for line in (self.run_dir/"review.jsonl").read_text().splitlines()])
        args = ("prepare", "--labels", labels, "--exclude-metadata", self.exclusions, "--output", output)
        self.invoke(*args, ok=False)
        self.invoke(*args, "--include-official-metadata", official)
        receipt = json.loads((output/"manifest.json").read_bytes())
        self.assertEqual(receipt["official_training_prompt_count"], 1)
        self.assertEqual(receipt["official_training_metadata_sha256"], manifest["official_training_metadata_sha256"])
        self.assertNotEqual(receipt["evaluation_metadata_sha256"], receipt["official_training_metadata_sha256"])
        self.generate("--include-official-metadata", self.exclusions, ok=False)
        self.assertEqual(len(self.requests), 2)

    def test_continue_invalid_partial_and_no_refusal_resubmission(self):
        self.refuse_text = ROWS[0]["prompt"]
        self.generate("--workers", 1, "--continue-invalid", ok=False)
        self.assertEqual(len(self.requests), 2)
        self.assertFalse((self.run_dir/"review.jsonl").exists())
        self.assertEqual(json.loads((self.run_dir/"review.partial.jsonl").read_bytes())["prompt"], ROWS[1]["prompt"])
        self.assertEqual(json.loads((self.run_dir/"failures.jsonl").read_bytes())["reason"], "refused")
        self.refuse_text = None
        self.generate("--continue-invalid", key=None, ok=False)
        self.assertEqual(len(self.requests), 2)
