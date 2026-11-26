import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scrl.sft.data import epoch_batches, load_splits, prompt_key
from scrl.sft.data import prepare


def row(prompt, group):
    return dict(prompt=prompt, group=group, reviewer="review-1", approved=True,
                teacher_model="teacher-version", template_sha256="a" * 64,
                plan=[dict(step_index=0, prompt="A quiet room", num_chunks=1)])


class TeacherTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.metadata = self.root/"vbench.json"
        self.metadata.write_text(json.dumps([dict(prompt_en="A held-out title", group="Reserved")]))
        self.src = self.root/"labels.jsonl"
        self.dest = self.root/"out"
        find_group = lambda held: next(f"g{i}" for i in range(100)
                                        if (int(prompt_key(f"g{i}")[:8], 16) % 10 == 0) == held)
        self.rows = [row("First", find_group(False)), row("Second", find_group(False)),
                     row("Third", find_group(True))]

    def build(self, rows):
        self.src.write_text("".join(json.dumps(r)+"\n" for r in rows))
        return prepare(self.src, [self.metadata], self.dest)

    def test_split_groups_pin_and_no_overwrite(self):
        pin = self.build(self.rows)
        train, val = load_splits(self.dest, manifest_sha256=pin)
        self.assertEqual([len(train), len(val)], [2, 1])
        self.assertTrue({r["group_id"] for r in train}.isdisjoint(r["group_id"] for r in val))
        with self.assertRaises(ValueError):
            load_splits(self.dest)
        with self.assertRaises(FileExistsError):
            self.build(self.rows)
        (self.dest/"train.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Digest mismatch"):
            load_splits(self.dest, manifest_sha256=pin)

    def test_holdout_duplicate_review_and_format_fail_closed(self):
        for patch in [dict(prompt=" A  HELD-out title"), dict(group="reserved"),
                      dict(prompt="second"), dict(approved=False), dict(plan=[]),
                      dict(template_sha256="unknown"), dict(reviewer=" ")]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.build([{**self.rows[0], **patch}, *self.rows[1:]])
            self.assertFalse(self.dest.exists())
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.build(self.rows[:2])

    def test_loader_rejects_repinned_invalid_or_overlapping_teacher_groups(self):
        self.build(self.rows)
        path = self.dest/"validation.jsonl"
        val = json.loads(path.read_text())
        manifest_path = self.dest/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for group, error in [(prompt_key(self.rows[0]["group"]), "both train and validation"),
                             ("", "Invalid teacher group")]:
            with self.subTest(group=group):
                path.write_text(json.dumps(dict(val, group_id=group))+"\n")
                manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, error):
                    load_splits(self.dest, manifest_sha256=pin)

    def test_batches_keep_short_tails_without_duplicate_or_empty_ranks(self):
        for world in (1, 2, 4, 8):
            for n in range(world, 5 * world + 1):
                order = list(range(n))
                batches = list(epoch_batches(order, world))
                self.assertTrue(all(world <= len(b) <= world * 2 for b in batches))
                self.assertEqual(sum(batches, []), order)
            with self.assertRaises(ValueError):
                list(epoch_batches([], world))
