"""Freeze a stratified prompt roster from two official catalogs and one RL pool."""
import argparse
from collections import Counter, defaultdict
from itertools import pairwise
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile
import tempfile

from scrl.sft.data import prompt_key


WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an of and the its in on to is are with as by it from for then they their at onto into that this these those slightly gently person initially begins finally clearly visible frame very small large".split())
AVOID = re.compile(r"\b(?:blood|gore|suicide|nude|naked|pistol|gun|rifle|gunshot|bomb|grenade|weapon|ammunition|self-harm|stabbing|sexual|shootout|assault|sniper|archery|boxing|wrestling|sword|knife fight|shooting|martial arts|fighting|explosive|decapitat\w*|ignore previous instructions)\b", re.I)
SLOW = re.compile(r"\b(?:slowly|gradually|steadily|continuously|over time|time.lapse|sustain\w*|melt\w*)\b", re.I)
QUICK = re.compile(r"\b(?:quick\w*|instant\w*|suddenly|abrupt\w*|immediate\w*|brief\w*)\b", re.I)
ORDER = re.compile(r"\b(?:then|next|finally|subsequently|afterwards?|before|followed by)\b", re.I)
CAMERA = re.compile(r"\b(?:zoom|pan|tilt|dolly|tracking|timelapse|time.lapse)(?:s|ning|ing)?\b", re.I)
FAMILIES = (
    "toy car", "traffic light", "ice cube", "garage door", "bird feeder", "squirrel",
    "candle", "robotic arm", "watering can", "sunflower", "conveyor belt", "coffee machine",
    "vending machine", "kettle", "balloon", "domino", "light switch", "desk lamp",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_source(spec):
    if "::" in str(spec):
        path, member = str(spec).split("::", 1)
        with tarfile.open(path) as archive:
            info = archive.getmember(member)
            if not info.isfile() or info.size > 32 << 20:
                raise ValueError("Expected a small, regular metadata file in archive")
            return archive.extractfile(info).read()
    return Path(spec).read_bytes()


def tags(text):
    return [label for label, regex in (("gradual", SLOW), ("brief", QUICK), ("ordered", ORDER),
                                        ("camera", CAMERA)) if regex.search(text)]


def candidates(official, rl, cutoffs):
    seen, pools = set(), {"official": [], "rl": []}
    for source, records in (("official", official), ("rl", rl)):
        for index, record in enumerate(records):
            text = record["prompt_en" if source == "official" else "prompt"]
            key = prompt_key(text)
            if not text.strip() or key in seen or AVOID.search(text):
                continue
            seen.add(key)
            length = len(text.split())
            if source == "official":
                dimensions = record["dimension"]
                bucket = dimensions[0]
                src = ["VBench", "VBench-Long"]
            else:
                dimensions = []
                bucket = ("short", "medium", "long")[sum(length > n for n in cutoffs)]
                src = ["rl5000"]
            pools[source].append(dict(prompt=text, prompt_id=key, sources=src, source_index=index,
                                      source_id=record.get("id", index), source_row=record.get("source_row", index),
                                      dimensions=dimensions, stratum=bucket, words=length, cues=tags(text)))
    return pools


def similarities(rows, threshold):
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    docs = []
    freq = Counter()
    for r in rows:
        words = [w for w in WORD.findall(r["prompt"].lower()) if w not in STOP]
        terms = Counter(words + [a+" "+b for a, b in pairwise(words)])
        docs.append(terms)
        freq.update(terms.keys())
    vocab = {t: i for i, t in enumerate(sorted(freq))}
    data, cols, ends = [], [], [0]
    for doc in docs:
        vector = [(vocab[t], (1+math.log(n))*(1+math.log((len(rows)+1)/(freq[t]+1)))) for t,n in doc.items()]
        norm = math.sqrt(sum(v*v for _,v in vector)) or 1.
        cols.extend(i for i,_ in vector)
        data.extend(v/norm for _,v in vector)
        ends.append(len(cols))
    x = csr_matrix((np.asarray(data, dtype=np.float32), cols, ends), shape=(len(rows),len(vocab)))
    sim = (x @ x.T).toarray()
    _, labels = connected_components(csr_matrix(sim >= threshold), directed=False)
    # Group close variants for the downstream group-based train/validation split.
    # Broad recurring RL scenarios also share a group even when rewritten with very different words.
    family = {}
    for i,r in enumerate(rows):
        if r["sources"] == ["rl5000"]:
            match = next((f for f in FAMILIES if f in r["prompt"].lower()), None)
            if match:
                family.setdefault(labels[i], Counter())[match] += 1
    ids = {}
    for i,r in enumerate(rows):
        label = labels[i]
        group = ids.setdefault(label, r["prompt_id"])
        if label in family:
            group = "scenario:" + sorted(family[label], key=lambda f:(-family[label][f],f))[0]
        r["group"] = "source-family-v1:" + group
    return sim


def pick(rows, quotas, seed, threshold, rl=False):
    import numpy as np
    sim = similarities(rows, threshold)
    by = defaultdict(list)
    for i,r in enumerate(rows):
        by[r["stratum"]].append(i)
    for k,n in quotas.items():
        if len(by[k]) < n:
            raise ValueError(f"Insufficient {k} candidates: {len(by[k])} < {n}")
    rng = np.random.default_rng(seed)
    novelty = 1 + .001*rng.random(len(rows))
    used, emitted, count = set(), [], Counter()
    groups = Counter()
    while len(emitted) < sum(quotas.values()):
        for k,n in quotas.items():
            if count[k] == n:
                continue
            # Prefer a range of temporal cues, prompt lengths and scene families within each quota.
            def criterion(i):
                row = rows[i]
                facet = tuple(row["cues"]) if rl else min(row["words"]//4, 4)
                return (novelty[i] - .026*math.log1p(groups[row["group"]])
                        - .055*math.log1p(groups[(k,facet)]))
            i = max((i for i in by[k] if i not in used), key=criterion)
            r = rows[i]
            facet = tuple(r["cues"]) if rl else min(r["words"]//4, 4)
            groups[r["group"]] += 1
            groups[(k,facet)] += 1
            count[k] += 1
            used.add(i)
            emitted.append(r)
            novelty = np.minimum(novelty, 1 - sim[i])
    return emitted


def select(recipe_path, official_spec, long_spec, rl_spec, output):
    recipe_raw = recipe_path.read_bytes()
    cfg = json.loads(recipe_raw)
    inputs = {k:read_source(v) for k,v in (("vbench",official_spec),("vbench_long",long_spec),("rl",rl_spec))}
    hashes = {k:sha(raw) for k,raw in inputs.items()}
    if not hashes["vbench"] == hashes["vbench_long"] == cfg["official_sha256"] or hashes["rl"] != cfg["rl_sha256"]:
        raise ValueError("Unexpected source hashes; VBench and VBench-Long must match this pinned recipe")
    pools = candidates(json.loads(inputs["vbench"]), json.loads(inputs["rl"]), cfg["rl_word_cutoffs"])
    rows = []
    for name, quotas in (("official",cfg["official_by_primary_dimension"]),("rl",cfg["rl_by_length"])):
        rows.extend(pick(pools[name], quotas, cfg["seed"], cfg["near_duplicate_cosine"], rl=name == "rl"))
    rows.sort(key=lambda r:sha(f'{cfg["seed"]}:{r["prompt_id"]}'.encode()))
    raw = "".join(json.dumps(r, ensure_ascii=False, separators=(",",":"))+"\n" for r in rows).encode()
    report = dict(format="sft-prompt-selection-v1", count=len(rows), roster_sha256=sha(raw),
                  source_sha256=hashes, sources=dict(vbench=str(official_spec), vbench_long=str(long_spec), rl=str(rl_spec)),
                  official_training_overlap="intentional", code_sha256=sha(Path(__file__).read_bytes()),
                  recipe_sha256=sha(recipe_raw), recipe=cfg, eligible={k:len(v) for k,v in pools.items()},
                  selected_source=Counter(r["sources"][0] for r in rows), selected_stratum=Counter(r["stratum"] for r in rows),
                  selected_dimensions=Counter(d for r in rows for d in r["dimensions"]),
                  time_cues=Counter(c for r in rows for c in r["cues"]), unique_groups=len({r["group"] for r in rows}),
                  projected_validation_count=sum(int(prompt_key(r["group"])[:8],16)%10==0 for r in rows),
                  scope="Time cues and lexical clusters are selection heuristics, not verified event or duration labels.")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        dest = Path(tmp)/"selection"
        dest.mkdir()
        (dest/"prompts.jsonl").write_bytes(raw)
        (dest/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
        dest.rename(output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("recipe", "vbench", "vbench-long", "rl", "output"):
        p.add_argument("--"+key, required=True, type=Path)
    a = p.parse_args()
    report = select(a.recipe, a.vbench, a.vbench_long, a.rl, a.output)
    print(json.dumps({k: report[k] for k in ("count", "roster_sha256", "selected_source", "unique_groups")}))


if __name__ == "__main__":
    main()
