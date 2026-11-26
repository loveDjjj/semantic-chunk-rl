"""Group identical denoising shapes without sharing a task's prompt, RNG or cache."""

from collections import defaultdict


def shape_groups(jobs, size):
    if size not in (1, 2):
        raise ValueError("Only single videos and same-shape pairs are supported")
    if size == 1:
        return [[job] for job in jobs]
    buckets = defaultdict(list)
    for job in jobs:
        shape = tuple(segment['num_chunks'] for segment in job['plan']['segments'])
        buckets[shape].append(job)
    # Start with one short pair so a real batched output is checked early.
    pairs, singles = [], []
    for shape in sorted(buckets, key=lambda s: (sum(s), s)):
        rows = buckets[shape]
        pairs.extend(rows[i:i+2] for i in range(0, len(rows)-1, 2))
        singles.extend([rows[-1:]] if len(rows) % 2 else [])
    return pairs[:1] + pairs[1:][::-1] + singles[::-1]
