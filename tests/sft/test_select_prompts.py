import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scrl.sft.select_prompts import candidates, select, sha


class SelectionTests(unittest.TestCase):
    def test_raw_text_deduplication_sources_and_temporal_cues(self):
        official = [{"prompt_en":"A red vase.", "dimension":["color"]}]
        rows = [{"prompt":s} for s in ("a red vase.", "A cloud gradually moves.", "A gun.")]
        pools = candidates(official, rows, [3, 6])
        self.assertEqual(pools["official"][0]["sources"], ["VBench", "VBench-Long"])
        self.assertEqual([r["prompt"] for r in pools["rl"]], ["A cloud gradually moves."])
        self.assertEqual(pools["rl"][0]["cues"], ["gradual"])
        self.assertEqual(pools["rl"][0]["stratum"], "medium")

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Requires optional selection dependencies")
    def test_frozen_roster_deterministic_and_rejects_mismatched_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vb, rl, cfg = (root/f"{n}.json" for n in ("vb", "rl", "cfg"))
            vb.write_text(json.dumps([{"prompt_en":s, "dimension":["color"]}
                                      for s in ("A blue plate", "A red vase")]))
            rl.write_text(json.dumps([{"id":i, "prompt":s} for i,s in enumerate(
                ("A toy car rolls.", "A toy car stops.", "A lantern sways."))]))
            cfg.write_text(json.dumps(dict(seed=1, official_sha256=sha(vb.read_bytes()),
                rl_sha256=sha(rl.read_bytes()), official_by_primary_dimension={"color":1},
                rl_by_length={"short":3}, rl_word_cutoffs=[5,8], near_duplicate_cosine=.70)))
            a, b = (select(cfg,vb,vb,rl,root/n) for n in ("first","second"))
            self.assertEqual(a["roster_sha256"], b["roster_sha256"])
            self.assertEqual(a["count"], 4)
            rows = [json.loads(line) for line in (root/"first/prompts.jsonl").read_bytes().splitlines()]
            self.assertEqual(len({r["group"] for r in rows if "toy car" in r["prompt"]}), 1)
            with self.assertRaisesRegex(ValueError, "hashes"):
                select(cfg, vb, rl, rl, root/"bad")
