"""Load digest-pinned SFT splits without weakening the legacy release pin."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata

from scrl.plan import FinitePlan, instruction


MANIFEST_SHA256 = "6a93d203738cd4235f1ab549812a5921aac523bc71768e9037ccc9b8c2a00e9c"


def prompt_key(prompt):
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {field}")
    return value.strip()


def evaluation_exclusions(paths):
    """Use the same holdout check before sending prompts and before packaging labels."""
    blocked, digests = set(), []
    for path in paths:
        raw = path.read_bytes()
        records = json.loads(raw)
        if not isinstance(records, list) or not records:
            raise ValueError("Evaluation exclusions must be a non-empty metadata array")
        for record in records:
            blocked.add(prompt_key(nonempty(record["prompt_en"], "prompt_en")))
            if "group" in record:
                blocked.add(prompt_key(nonempty(record["group"], "group")))
        digests.append(hashlib.sha256(raw).hexdigest())
    if not blocked:
        raise ValueError("Evaluation exclusions are required")
    return blocked, sorted(digests)


def corpus_policy(exclusions, official):
    if not exclusions and not official:
        raise ValueError("Specify --exclude-metadata or explicitly --include-official-metadata")
    blocked, excluded_hashes = evaluation_exclusions(exclusions) if exclusions else (set(), [])
    allowed, included_hashes = evaluation_exclusions(official) if official else (set(), [])
    if blocked & allowed:
        raise ValueError("Official training metadata overlaps an explicit exclusion")
    return blocked, excluded_hashes, allowed, included_hashes


def load_splits(dataset, *, manifest_sha256=None, required_format=None):
    raw = (dataset / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (MANIFEST_SHA256 if manifest_sha256 is None else manifest_sha256):
        raise ValueError("SFT manifest does not match the requested pin")
    manifest = json.loads(raw)
    if required_format is not None and manifest.get("format") != required_format:
        raise ValueError(f"Training requires manifest format {required_format}")
    splits = []
    seen = set()
    seen_groups = set()
    for split in ("train", "validation"):
        name = split + ".jsonl"
        raw = (dataset / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"Digest mismatch: {name}")
        rows = [json.loads(line) for line in raw.splitlines()]
        ids = {row["prompt_id"] for row in rows}
        if not rows or len(rows) != manifest["counts"][split] or len(ids) != len(rows) or seen & ids:
            raise ValueError(f"Invalid count or repeated prompt in {name}")
        if manifest.get("format") == "reviewed-teacher-sft-v1":
            groups = [row.get("group_id") for row in rows]
            if any(not isinstance(g, str) or len(g) != 64 or
                   any(c not in "0123456789abcdef" for c in g) for g in groups):
                raise ValueError(f"Invalid teacher group ID in {name}")
            if seen_groups.intersection(groups):
                raise ValueError("Teacher group appears in both train and validation")
            seen_groups.update(groups)
        seen.update(ids)
        splits.append(rows)
    return tuple(splits)


def epoch_batches(order, world):
    """Visit every example once, with at least one item per DDP rank per step."""
    if world < 1 or len(order) < world:
        raise ValueError("Training needs at least one example per rank")
    start = 0
    while start < len(order):
        end = min(start + world * 2, len(order))
        if 0 < len(order) - end < world:
            end = len(order) - world
        yield order[start:end]
        start = end


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(labels, exclusions, output, *, official=()):
    if output.exists():
        raise FileExistsError(output)
    blocked, excluded_hashes, allowed, included_hashes = corpus_policy(exclusions, official)
    raw = labels.read_bytes()
    splits = {"train": [], "validation": []}
    seen, teachers = set(), set()
    official_count = 0
    for line, encoded in enumerate(raw.splitlines(), 1):
        row = json.loads(encoded)
        prompt = nonempty(row["prompt"], "prompt")
        key = prompt_key(prompt)
        group = prompt_key(nonempty(row["group"], "group"))
        if key in blocked or group in blocked or key in seen:
            raise ValueError(f"Evaluation overlap or duplicate prompt at line {line}")
        if sorted(row.get("official_training_metadata_sha256", [])) != included_hashes:
            raise ValueError("Official training metadata must match the draft policy")
        official_count += key in allowed
        if row.get("approved") is not True:
            raise ValueError(f"Unreviewed label at line {line}")
        reviewer = nonempty(row["reviewer"], "reviewer")
        model = nonempty(row["teacher_model"], "teacher_model")
        template = nonempty(row["template_sha256"], "template_sha256")
        if len(template) != 64 or any(c not in "0123456789abcdef" for c in template):
            raise ValueError(f"Invalid template SHA-256 at line {line}")
        plan = FinitePlan.parse(json.dumps(row["plan"]))
        split = "validation" if int(group[:8], 16) % 10 == 0 else "train"
        splits[split].append(dict(prompt_id=key, group_id=group, reviewer=reviewer,
            teacher_model=model, template_sha256=template, messages=[
                dict(role="user", content=instruction(prompt)),
                dict(role="assistant", content=json.dumps(plan.to_output(), ensure_ascii=False))]))
        teachers.add((model, template))
        seen.add(key)
    if not all(splits.values()):
        raise ValueError("Teacher data must produce nonempty train and validation splits")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        dest = Path(tmp)/"dataset"
        dest.mkdir()
        hashes = {}
        for name, rows in splits.items():
            path = dest/(name+".jsonl")
            with path.open("w", encoding="utf-8") as stream:
                for row in sorted(rows, key=lambda r: r["prompt_id"]):
                    print(json.dumps(row, ensure_ascii=False), file=stream)
            hashes[path.name] = sha(path.read_bytes())
        manifest = dict(format="reviewed-teacher-sft-v1", source_sha256=sha(raw),
            evaluation_metadata_sha256=sorted(excluded_hashes), sha256=hashes,
            counts={k: len(v) for k, v in splits.items()},
            teachers=[dict(model=m, template_sha256=t) for m, t in sorted(teachers)],
            exclusion_scope="Normalized exact prompt/group keys; no automatic semantic deduplication.")
        if included_hashes:
            manifest.update(official_training_metadata_sha256=included_hashes,
                            official_training_prompt_count=official_count)
        path = dest/"manifest.json"
        path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        pin = sha(path.read_bytes())
        dest.rename(output)
    return pin


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--exclude-metadata", type=Path, action="append", default=[])
    p.add_argument("--include-official-metadata", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    print(prepare(args.labels, args.exclude_metadata, args.output, official=args.include_official_metadata))


if __name__ == "__main__":
    main()
