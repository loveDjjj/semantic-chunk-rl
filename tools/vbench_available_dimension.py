"""Score one dimension with an explicitly identified missing video, outside the strict report."""
import argparse
import json
import os
from pathlib import Path
import sys

from scrl.evaluation import vbench
from scrl.evaluation.io import atomic_json, read
from scrl.evaluation.protocol import required_tasks
from scrl.plan import digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('arm')
    ap.add_argument('dimension', choices=['background_consistency', 'scene', 'human_action'])
    ap.add_argument('missing_id')
    args = ap.parse_args()
    run, arm, dim = args.run, args.arm, args.dimension
    cfg = read(run/'config.json')
    tasks = read(run/'inputs/tasks.json')
    metadata = read(run/'inputs/metadata.json')
    protocol = cfg['scoring_protocol']
    if (arm not in cfg['arms'] or digest(tasks) != protocol['tasks_sha256']
            or digest(metadata) != protocol['metadata_sha256']):
        raise ValueError('Unexpected arm or changed input manifest')
    present = {p.stem for p in (run/'records'/arm).glob('*.json')}
    if {t['id'] for t in tasks} ^ present != {args.missing_id} or args.missing_id in present:
        raise ValueError('Published task set differs from the declared single omission')
    needed = required_tasks(tasks, dim)
    if args.missing_id not in {t['id'] for t in needed}:
        raise ValueError('Use the strict scorer for an unaffected dimension')
    kept = [t for t in needed if t['id'] in present]
    records = vbench.checked_records(run, arm, cfg, kept, dim)
    stage = run/'evaluation'/arm/'available_only'/dim
    stage.mkdir(parents=True, exist_ok=False)
    folder = stage/'evaluation'/arm
    folder.mkdir(parents=True)
    (stage/'inputs').symlink_to(run/'inputs')
    entries = []
    for row in metadata:
        if dim not in row['dimension']:
            continue
        videos = [records[t['id']]['video'] for t in kept if t['prompt'] == row['prompt_en']]
        if not videos:
            raise ValueError('Cannot drop an entire prompt')
        entries.append(dict(row, dimension=[dim], video_list=videos))
    weights = Path(cfg['vbench_weights'])
    os.environ.update(VBENCH_CACHE_DIR=str(weights), VBENCH_BERT_PATH=str(weights/'bert-base-uncased'),
                      TORCH_HOME=str(weights/'torch_hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, cfg['vbench_source'])
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    binding = vbench.receipt_binding(cfg, arm, dim, kept, records)
    result = vbench.evaluate(stage, arm, dim, cfg, kept, records, binding, entries=entries)
    if result['state'] != 'complete' or result['coverage']['scored_videos'] != len(kept):
        raise ValueError('Available-only scoring did not cover every retained video')
    result.update(scope='available_videos_only', missing_ids=[args.missing_id], planned_inputs=len(needed))
    atomic_json(stage/'result.json', result)
    print(json.dumps({'dimension': dim, 'score': result['score'], 'coverage': result['coverage']}))


if __name__ == '__main__':
    main()
