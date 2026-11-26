"""CPU-only post-hoc prompt-cluster bootstrap for fixed-plan VBench denoising pairs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scrl.plan import digest
from scrl.evaluation.protocol import DIMENSIONS, aggregate_scores, load_records


ARM_NAMES = ("joint", "chunk")
SCORES = ("quality", "semantic", "total")
WEIGHTS = {
    "subject_consistency": "decoded adjacent frame pairs",
    "background_consistency": "cnt_per_video",
    "appearance_style": "scored frames",
    "spatial_relationship": "scored frames",
    **dict.fromkeys(("scene", "object_class", "multiple_objects"), "frame_count"),
}
POINT_ATOL = 1e-6  # Single-precision scorer reductions can differ from CPU float64.


def read(path):
    return json.loads(path.read_bytes())


def contribution(dimension, detail, record):
    """Sufficient statistics for the single-device official scorer, not video means."""
    if dimension == "subject_consistency":
        n = record["frames"] - 1
        value = detail["video_results"] * n
    elif dimension == "background_consistency":
        value, n = detail["video_sim"], detail["cnt_per_video"]
    elif dimension in ("scene", "object_class", "multiple_objects"):
        value, n = detail["success_frame_count"], detail["frame_count"]
    elif dimension in ("appearance_style", "spatial_relationship"):
        frames = detail["frame_results"]
        value, n = sum(frames), len(frames)
    else:
        value, n = detail["video_results"], 1
        if dimension == "imaging_quality":
            value /= 100
    if not math.isfinite(value) or type(n) is not int or n <= 0:
        raise ValueError("Invalid scorer numerator/denominator")
    return value, n


def load_arm(run, arm):
    config = read(run / "config.json")
    if arm not in config["arms"]:
        raise ValueError(f"Unknown arm {arm} in {run}")
    tasks = read(run / "inputs/tasks.json")
    records = load_records(run, arm)
    if len(tasks) != len(records) or {t["id"] for t in tasks} != records.keys():
        raise ValueError("Incomplete task records")
    for t in tasks:
        r = records[t["id"]]
        if r["arm"] != arm or any(r[k] != t[k] for k in
                                   ("id", "prompt", "prompt_id", "sample_index", "planner_seed", "video_seed")):
            raise ValueError(f"Task identity mismatch: {t['id']}")
        if r["protocol_sha256"] != config["protocol_sha256"]:
            raise ValueError(f"Protocol mismatch: {t['id']}")
    return config, tasks, records


def metric_stats(metric, records, by_path, prompt_index):
    d = metric["dimension"]
    values = np.zeros((len(prompt_index), 2))
    seen = set()
    reason = None
    for detail in metric["details"]:
        path = Path(detail["video_path"])
        task = by_path.get(str(path))
        if task is None:
            # Historical archives relocated the mount prefix; never join by bare ID.
            task = path.stem
            if (task not in records or len(path.parts) < 4 or path.parts[-3] != "videos" or
                    path.parts[-4:] != Path(records[task]["video"]).parts[-4:]):
                raise ValueError(f"Unrecognized {d} video path: {path}")
        if task in seen:
            raise ValueError(f"Duplicate {d} video detail: {task}")
        seen.add(task)
        try:
            values[prompt_index[records[task]["prompt_id"]]] += contribution(d, detail, records[task])
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"Insufficient scorer detail: {exc}"
    if metric["coverage"]["scored_videos"] != len(seen):
        raise ValueError(f"Coverage mismatch: {d}")
    num, den = values.sum(axis=0)
    score = metric["score"]
    recovered = float(num / den) if den > 0 and reason is None else None
    if score is None or recovered is None or not math.isclose(recovered, score, rel_tol=0, abs_tol=POINT_ATOL):
        reason = reason or "Per-video reconstruction does not reproduce the saved official score"
    result = {"official": score, "reconstructed": recovered, "denominator": None if reason else int(den),
              "denominator_unit": WEIGHTS.get(d, "scored videos"), "coverage": metric["coverage"],
              "unavailable_reason": reason}
    if reason:
        values[:] = 0
    return values, seen, result


def summarize_draws(stats, formula, draws, seed):
    """Use the identical multinomial prompt weights for both arms and all dimensions."""
    if draws < 2:
        raise ValueError("Need at least two bootstrap draws")
    rng = np.random.default_rng(seed)
    n = stats.shape[1]
    if n < 2:
        raise ValueError("Need at least two prompt clusters")
    results = {k: [] for k in SCORES}
    # Obtain the affine coefficients from the same official aggregator used for
    # the point, without treating any unavailable measurement as an observation.
    zero = dict.fromkeys(DIMENSIONS, 0.)
    origin = aggregate_scores(zero, formula)
    basis = [aggregate_scores({**zero, d: 1.}, formula) for d in DIMENSIONS]
    coeff = {k: np.array([b[k] - origin[k] for b in basis]) for k in SCORES}
    for start in range(0, draws, 512):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=min(512, draws-start))
        # arm x draw x dimension x (numerator, denominator)
        totals = np.einsum("bp,apdk->abdk", weights, stats)
        raw = np.divide(totals[..., 0], totals[..., 1], out=np.full(totals.shape[:-1], np.nan),
                        where=totals[..., 1] > 0)
        for k in SCORES:
            use = coeff[k] != 0
            diff = (raw[1] - raw[0])[:, use]
            valid = np.isfinite(diff).all(axis=1)
            results[k].extend(100 * np.einsum("bd,d->b", diff[valid], coeff[k][use]))
    return {k: {"ci95_pp": np.quantile(v, [.025, .975]).tolist() if len(v) >= 2 else None,
                "valid_draws": len(v), "excluded_draws": draws-len(v)} for k, v in results.items()}


def paired_report(joint_run, chunk_run, joint_arm, chunk_arm, draws=10000, seed=20261125):
    paths = (Path(joint_run), Path(chunk_run))
    arms = (joint_arm, chunk_arm)
    loaded = [load_arm(path, arm) for path, arm in zip(paths, arms)]
    cfg, tasks, records = zip(*loaded)
    protocols = []
    for c in cfg:
        if 'generation_identity' in c:
            gen, score = c['generation_identity'], c['scoring_protocol']
            if (digest(gen) != c['generation_sha256'] or digest(score) != c['scoring_protocol_sha256']
                    or c['protocol'] != dict(score, generation_sha256=digest(gen))
                    or digest(c['protocol']) != c['protocol_sha256']
                    or gen['denoising_mode'] != c.get('denoising_mode', 'segment-joint')):
                raise ValueError('Inconsistent generation/scoring identity')
            protocols.append((score, {k:v for k,v in gen.items() if k != 'denoising_mode'}))
        else:
            protocols.append({k: v for k, v in c["protocol"].items() if k != "denoising_mode"})
    if tasks[0] != tasks[1] or cfg[0]["formula"] != cfg[1]["formula"] or protocols[0] != protocols[1]:
        raise ValueError("Unpaired task manifest, scorer protocol or aggregation formula")
    if cfg[0]["arms"][joint_arm] != cfg[1]["arms"][chunk_arm] or cfg[0]["assets"] != cfg[1]["assets"]:
        raise ValueError("Model assets or checkpoint changed")
    historical_pair = paths[0].resolve() == paths[1].resolve() and arms == ("joint", "chunk")
    if not historical_pair and (cfg[0].get("denoising_mode") == "causal-chunkwise" or
                                cfg[1].get("denoising_mode") != "causal-chunkwise"):
        raise ValueError("Expected a joint reference and a causal-chunkwise candidate")
    for task, left in records[0].items():
        right = records[1][task]
        if any(left[k] != right[k] for k in ("plan", "frames", "fps", "height", "width")):
            raise ValueError(f"Plan, duration or geometry changed: {task}")
    prompt_index = {p: i for i, p in enumerate(sorted({t["prompt_id"] for t in tasks[0]}))}
    paths_to_ids = [{r["video"]: t for t, r in arm.items()} for arm in records]
    if any(len(m) != len(tasks[0]) for m in paths_to_ids):
        raise ValueError("Repeated video paths")
    stats = np.zeros((2, len(prompt_index), len(DIMENSIONS), 2))
    archives = []
    for i, (path, arm) in enumerate(zip(paths, arms)):
        saved = path / "analysis" / (arm + ".json")
        archive = read(saved) if saved.exists() else None
        if archive and (archive["arm"] != arm or archive["protocol_sha256"] != cfg[i]["protocol_sha256"]):
            raise ValueError("Archived score arm/protocol mismatch")
        archives.append(archive)
    dimensions = {}
    for j, d in enumerate(DIMENSIONS):
        results, ids = {}, []
        for i, name in enumerate(ARM_NAMES):
            path = paths[i] / "evaluation" / arms[i] / (d + ".json")
            if not path.exists() and archives[i] is not None:
                results[name] = {"official": archives[i]["raw_dimensions"][d], "reconstructed": None,
                                 "denominator": None, "denominator_unit": WEIGHTS.get(d, "scored videos"),
                                 "coverage": archives[i]["coverage"][d],
                                 "unavailable_reason": "No per-video metric file in this archive"}
                ids.append(None)
                continue
            metric = read(path)
            if metric["dimension"] != d or (metric["score"] is not None and
                    metric["module_sha256"] != cfg[i]["protocol"]["metric_modules_sha256"][d]):
                raise ValueError(f"Wrong metric or scorer version: {d}")
            stats[i, :, j], seen, results[name] = metric_stats(metric, records[i], paths_to_ids[i], prompt_index)
            ids.append(seen)
        if None in ids:
            results.update(shared_tasks=None, joint_only=None, chunk_only=None)
        else:
            results.update(shared_tasks=len(ids[0] & ids[1]), joint_only=sorted(ids[0] - ids[1]),
                           chunk_only=sorted(ids[1] - ids[0]))
        dimensions[d] = results
    formula = cfg[0]["formula"]
    points = {name: aggregate_scores({d: s[name]["official"] for d, s in dimensions.items()}, formula)
              for name in ARM_NAMES}
    summary = summarize_draws(stats, formula, draws, seed)
    for k in SCORES:
        l, r = points["joint"][k], points["chunk"][k]
        summary[k]["official_delta_pp"] = 100 * (r-l) if r is not None and l is not None else None
    return {"reference": {"run": str(paths[0]), "arm": joint_arm}, "candidate": {"run": str(paths[1]), "arm": chunk_arm},
            "tasks": len(tasks[0]), "unique_prompts": len(prompt_index), "draws": draws, "seed": seed,
            "method": "Paired percentile bootstrap, sampling original prompts with replacement; all seeds, "
                      "dimension overlaps and each arm's actual static-filter/color inclusion travel with the prompt. "
                      "Chunk minus joint, percentage points. Not a common-scored-video-only estimand. "
                      "Draws with an empty required denominator are omitted for that component. "
                      "No independent holdout, multiple-comparison adjustment or video regeneration.",
            "point_absolute_tolerance": POINT_ATOL, "official": points, "paired": summary, "dimensions": dimensions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joint_run", type=Path)
    parser.add_argument("chunk_run", type=Path)
    parser.add_argument("--joint-arm", required=True)
    parser.add_argument("--chunk-arm", required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--output", type=Path, help="Create new report file; defaults to stdout")
    args = parser.parse_args()
    result = paired_report(args.joint_run, args.chunk_run, args.joint_arm, args.chunk_arm, args.draws, args.seed)
    data = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x") as stream:
            print(data, file=stream)
    else:
        print(data)


if __name__ == "__main__":
    main()
