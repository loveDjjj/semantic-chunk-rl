import hashlib
from contextlib import redirect_stderr
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scrl.sft.data import load_splits, prompt_key
from scrl.sft.train import claim_output, main as train

class SftDataTests(unittest.TestCase):
    def rejected(self, argv, expected):
        message = io.StringIO()
        with redirect_stderr(message), self.assertRaises(SystemExit) as error:
            train(argv)
        self.assertEqual(error.exception.code, 2)
        self.assertIn(expected, message.getvalue())

    def test_train_requires_named_paths_and_reviewed_format(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.rejected([str(root/"run"), str(root)], "--model-path")
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"format":"legacy-winners-v1"}))
            args = ["--run",str(root/"run"),"--model-path",tmp,"--dataset",tmp]
            self.rejected(args, "--manifest-sha256")
            args += ["--manifest-sha256",hashlib.sha256(manifest.read_bytes()).hexdigest()]
            self.rejected(args, "reviewed-teacher-sft-v1")
            self.assertFalse((root/"run").exists())
            (root/"run").mkdir()
            (root/"run"/"result").write_text("previous run")
            self.rejected(args, "empty directory")
            self.assertEqual((root/"run"/"result").read_text(), "previous run")

    def test_one_job_claims_shared_output(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"run"
            def claim(i):
                try:
                    claim_output(path, {"job":i})
                except (ValueError, FileExistsError):
                    return None
                return i
            with ThreadPoolExecutor(max_workers=2) as pool:
                winners = [i for i in pool.map(claim, [1,2]) if i is not None]
            self.assertEqual(winners, [json.loads((path/"training-inputs.json").read_text())["job"]])

    def test_verified_splits_reject_tampering_and_overlap(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            for repeated in (False, True):
                manifest = {"sha256": {}, "counts": {"train": 1, "validation": 1}}
                for split in manifest["counts"]:
                    path = dataset / (split + ".jsonl")
                    raw = json.dumps({"prompt_id": "train" if repeated else split}).encode()
                    path.write_bytes(raw)
                    manifest["sha256"][path.name] = hashlib.sha256(raw).hexdigest()
                raw = json.dumps(manifest).encode()
                (dataset / "manifest.json").write_bytes(raw)
                with patch("scrl.sft.data.MANIFEST_SHA256", hashlib.sha256(raw).hexdigest()):
                    if repeated:
                        with self.assertRaisesRegex(ValueError, "repeated prompt"):
                            load_splits(dataset)
                    else:
                        self.assertEqual(load_splits(dataset)[1][0]["prompt_id"], "validation")
                    (dataset / "train.jsonl").write_bytes(b"{}")
                    with self.assertRaisesRegex(ValueError, "Digest mismatch"):
                        load_splits(dataset)

    def test_split_identity(self):
        self.assertEqual(prompt_key("Scene  ONE"), prompt_key("Ｓcene\none"))
