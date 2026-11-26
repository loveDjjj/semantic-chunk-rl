import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scrl.evaluation.paired import contribution, metric_stats, paired_report, summarize_draws
from scrl.evaluation.protocol import DIMENSIONS, aggregate_scores


def formula():
    names = [d.replace("_", " ") for d in DIMENSIONS]
    return {"QUALITY_LIST": names[:7], "SEMANTIC_LIST": names[7:],
            "DIM_WEIGHT": {n: i + 1 for i, n in enumerate(names)},
            "NORMALIZE_DIC": {n: {"Min": -.5, "Max": 1.5} for n in names},
            "QUALITY_WEIGHT": 4, "SEMANTIC_WEIGHT": 1}


class PairedVBenchTests(unittest.TestCase):
    def test_official_denominators(self):
        self.assertEqual(contribution("imaging_quality", {"video_results": 80}, {}), (.8, 1))
        self.assertEqual(contribution("spatial_relationship", {"frame_results": [.1, .2, .2]}, {}), (.5, 3))
        self.assertEqual(contribution("multiple_objects", {"success_frame_count": 2, "frame_count": 8}, {}), (2, 8))
        self.assertEqual(contribution("background_consistency", {"video_sim": 7, "cnt_per_video": 8}, {}), (7, 8))
        metric = {"dimension": "subject_consistency", "score": .75, "coverage": {"scored_videos": 2},
                  "details": [{"video_path": "x", "video_results": 1.}, {"video_path": "y", "video_results": 0.}]}
        records = {"a": {"prompt_id": "p", "frames": 4}, "b": {"prompt_id": "p", "frames": 2}}
        args = (metric, records, {"x": "a", "y": "b"}, {"p": 0})
        stats, _, meta = metric_stats(*args)
        self.assertEqual(stats.tolist(), [[3., 4.]])  # Retains both seeds and their unequal weights.
        self.assertEqual(meta["denominator"], 4)
        metric["score"] = .5
        self.assertTrue(metric_stats(*args)[2]["unavailable_reason"])
        metric["details"][1]["video_path"] = "x"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            metric_stats(*args)

    def test_paired_normalization_and_partial_coverage(self):
        f = formula()
        rng = np.random.default_rng(2)
        data = rng.random((3, len(DIMENSIONS)))
        stats = np.ones((2, 3, len(DIMENSIONS), 2))
        # Strong correlated prompt effects cancel only if both arms use identical clusters.
        stats[0, :, :, 0] = data
        delta = np.arange(len(DIMENSIONS)) / 50
        stats[1, :, :, 0] = data + delta
        before = aggregate_scores(dict(zip(DIMENSIONS, data.mean(axis=0))), f)
        after = aggregate_scores(dict(zip(DIMENSIONS, data.mean(axis=0) + delta)), f)
        out = summarize_draws(stats, f, 120, 11)
        for key, result in out.items():
            np.testing.assert_allclose(result["ci95_pp"], [100*(after[key]-before[key])] * 2)
            self.assertEqual(result["valid_draws"], 120)
        # One unsupported semantic score suppresses semantic/total; quality is unaffected.
        stats[0, :, -1] = 0
        partial = summarize_draws(stats, f, 120, 11)
        self.assertIsNone(partial["total"]["ci95_pp"])
        self.assertIsNone(partial["semantic"]["ci95_pp"])
        self.assertEqual(partial["quality"], out["quality"])

    def test_reject_changed_plan_and_report_missing_details(self):
        tasks = [{"id": key, "prompt_id": key} for key in ("a", "b")]
        config = {"formula": formula(), "protocol": {}, "protocol_sha256": "test", "assets": {},
                  "arms": {"joint": {}, "chunk": {}}}
        records = {t["id"]: dict(t, plan={}, frames=9, height=10, width=11, fps=16, video=t["id"]) for t in tasks}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "analysis").mkdir()
            summary = {"raw_dimensions": dict.fromkeys(DIMENSIONS, .5),
                       "coverage": dict.fromkeys(DIMENSIONS, {"scored_videos": 2})}
            for name in ("joint", "chunk"):
                (root / "analysis" / (name + ".json")).write_text(
                    json.dumps({**summary, "arm": name, "protocol_sha256": "test"}))
            arm = (config, tasks, records)
            with patch("scrl.evaluation.paired.load_arm", return_value=arm):
                result = paired_report(root, root, "joint", "chunk", 10)
            self.assertEqual(result["paired"]["total"]["official_delta_pp"], 0)
            self.assertIsNone(result["paired"]["total"]["ci95_pp"])
            self.assertIsNone(result["dimensions"]["color"]["joint_only"])
            changed = (config, tasks, {**records, "b": {**records["b"], "plan": {"different": True}}})
            with patch("scrl.evaluation.paired.load_arm", side_effect=(arm, changed)):
                with self.assertRaisesRegex(ValueError, "Plan"):
                    paired_report(root, root, "joint", "chunk", 10)
