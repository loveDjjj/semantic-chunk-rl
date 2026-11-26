"""Audit reward attribution from completed rollout reports, without loading any models."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median, pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from scrl.plan import FinitePlan, Segment
from scrl.rl.objectives import CROSSED_CREDIT, WHOLE_CREDIT, _crossed_advantages, candidate_advantages


def mean(values):
    return math.fsum(values) / len(values)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def unique_winner(means):
    winners = [key for key, value in means.items() if value == max(means.values())]
    return winners[0] if len(winners) == 1 else None


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def read_batch(folder, *, global_sq=False):
    report = json.loads((folder/'report.json').read_bytes())
    check(report['state'] == 'passed', 'report is not a completed update')
    rows = report['rollouts']
    logical = {(r['group_id'], r['candidate_id'], r['seed']): r for r in rows}
    check(len(logical) == len(rows) == report['videos'], 'duplicate or missing logical rollout')
    physical = defaultdict(dict)
    alias_sets = {}
    joined = defaultdict(lambda: defaultdict(dict))
    job_prefix = 'b%d' % int(folder.name)
    with (folder/'jobs.jsonl').open() as stream:
        for line in stream:
            job = json.loads(line)
            parent = job.get('parent_job', job['job_id'])
            aliases = job.get('logical_job_ids', [parent])
            check(parent in aliases and len(set(aliases)) == len(aliases), 'invalid job aliases')
            check(alias_sets.setdefault(parent, set(aliases)) == set(aliases), 'split aliases differ')
            check(len(job['rewards']) == job['physical_videos'], 'wrong physical job size')
            for item in job['rewards']:
                seed, ret = item['seed'], item['returns']
                check(seed not in physical[parent], 'duplicated physical job and seed')
                fields = ('planner','executor','quality_global','semantic_global')
                check(all(type(ret[k]) in (int, float) and math.isfinite(ret[k])
                          for k in (*fields, 'global' if global_sq else 'semantic_local')),
                      'invalid scalar return')
                if global_sq:
                    s, q = ret['semantic_global'], ret['quality_global']
                    check(0 <= s <= 1 and 0 <= q <= 1 and all(math.isclose(ret[k], (s+q)/2,
                          abs_tol=1e-10, rel_tol=1e-10) for k in ('global','planner','executor')),
                          'global S/Q reward differs from recorded return')
                physical[parent][seed] = ret
                execution = None
                for alias in aliases:
                    batch, group, candidate = alias.split('/')
                    check(batch == job_prefix, 'job references another batch')
                    row = logical.pop((group, candidate, seed), None)
                    check(row is not None, 'unknown or duplicate logical alias/seed')
                    check(all(row[k+'_return'] == ret[k] for k in ('planner', 'executor')),
                          'report and job returns disagree')
                    signature = (group, row['prefix_id'], row['target_segment'], canon(row['plan']))
                    check(execution is None or signature == execution, 'alias changed plan or prefix')
                    execution = signature
                    check(row['stamp']['reward'] == report['stamp']['reward'], 'rollout reward stamp differs')
                    joined[group][candidate][seed] = row, ret, item.get('components', {}), parent
    check(not logical, 'unmatched rollout in report')
    physical_count = sum(map(len, physical.values()))
    check(physical_count == report.get('physical_videos', physical_count), 'physical video count differs')
    check(len(joined) == report['prompts'], 'incomplete prompt group count')
    return report, joined, physical


def prompt_stats(group, candidates, report, floor, details, credit='candidate-v1'):
    names = sorted(candidates)
    seed_ids = sorted(candidates[names[0]])
    check(len(names) == report['plans_per_prompt'] >= 2, 'incomplete candidate group')
    check(len(seed_ids) == report['seeds_per_plan'] >= 2, 'incomplete seed group')
    states, plans, scores = set(), {}, {}
    detail = {}
    fields = ('planner', 'quality_global', 'semantic_global')
    means = {k:{} for k in fields}
    for name in names:
        samples = candidates[name]
        check(sorted(samples) == seed_ids, 'candidate seeds do not match')
        rows, returns, components, parents = zip(*(samples[s] for s in seed_ids))
        for row in rows:
            states.add((row['prefix_id'], row['target_segment'], row['original_prompt'], canon(row['stamp'])))
        check(len({canon(r['plan']) for r in rows}) == 1, 'candidate plan varies across seeds')
        plans[name] = rows[0]['plan']['segments']
        scores[name] = [r['planner'] for r in returns]
        for k in fields:
            means[k][name] = mean([r[k] for r in returns])
        if details:
            detail[name] = dict(plan=rows[0]['plan'], seed_returns=[
                dict(seed=s, physical_candidate=parent,
                     **{k:r[k] for k in (*fields, 'executor','semantic_local')},
                     components={k:c[k] for k in ('global_scores','segment_scores','boundary_quality') if k in c})
                for s, r, c, parent in zip(seed_ids, returns, components, parents)])
    check(len(states) == 1, 'mixed prefix, target, prompt or stamp')
    _, target, _, _ = states.pop()
    check(all(0 <= target < len(p) for p in plans.values()), 'invalid target index')
    check(len({canon(p[:target]) for p in plans.values()}) == 1, 'nonidentical semantic prefix')
    totals = means['planner']
    std = pstdev(totals.values())
    crossed = credit == CROSSED_CREDIT
    if crossed:
        check(names == ['c0', 'c1', 'c2', 'c3'], 'invalid crossed cell IDs')
        cells = [FinitePlan(tuple(Segment(**s) for s in plans[k])) for k in names]
        axes = dict(zip(names, _crossed_advantages(cells, target, [totals[k] for k in names], floor)))
        recorded = report['planner_field_advantages'][group]
        check(isinstance(recorded, dict) and set(recorded) == set(names), 'incomplete recorded cells')
        for k, values in axes.items():
            check(isinstance(recorded[k], dict) and recorded[k].keys() == values.keys()
                  and all(type(x) in (int, float) and math.isfinite(x) for x in recorded[k].values()),
                  'invalid recorded crossed advantages')
            if floor == .02:
                check(all(math.isclose(recorded[k][key], value, abs_tol=1e-10, rel_tol=1e-10)
                          for key, value in values.items()), 'recorded crossed advantage differs')
    halves = []
    if len(seed_ids) % 2 == 0:
        mid = len(seed_ids)//2
        halves = [unique_winner({k:mean(v[i:j]) for k,v in scores.items()})
                  for i,j in [(0,mid),(mid,len(seed_ids))]]
    result = dict(group=group, candidates=len(names), unique_plans=len({canon(v) for v in plans.values()}),
                  planner_std=std, quality_std=pstdev(means['quality_global'].values()),
                  semantic_std=pstdev(means['semantic_global'].values()), split_winners=halves)
    if crossed:
        result['zero_by_axis'] = {axis: all(abs(v[axis]) <= 1e-8 for v in axes.values())
                                  for axis in ('caption', 'duration')}
    else:
        actions = [(plans[k][target]['caption'], plans[k][target]['num_chunks']) for k in names]
        def compare_credit(method):
            return dict(zip(names, candidate_advantages([totals[k] for k in names], actions, credit=method, floor=floor)))
        advantages = compare_credit('candidate-v1')
        focal = compare_credit('focal-mean-v1')
        repeated = [(a,b) for a,b in combinations(names, 2) if plans[a][target] == plans[b][target]]
        def conflicts(adv):
            return [(a,b) for a,b in repeated if min(adv[a],adv[b]) < -1e-8 and max(adv[a],adv[b]) > 1e-8]
        opposite = conflicts(advantages)
        result.update(same_target_pairs=len(repeated), opposite_pairs=len(opposite),
                      focal_opposite_pairs=len(conflicts(focal)),
                      legacy_zero=all(abs(v)<=1e-8 for v in advantages.values()),
                      focal_zero=all(abs(v)<=1e-8 for v in focal.values()))
        if details:
            result['opposite_target_pairs'] = opposite
    if details:
        for k, row in detail.items():
            if crossed:
                row['field_advantages'] = axes[k]
            else:
                row.update(advantage=advantages[k], focal_mean_advantage=focal[k])
            row['means'] = {f: v[k] for f, v in means.items()}
        result.update(target_segment=target, candidate_detail=detail)
    return result


def fraction(count, total):
    return dict(count=count, total=total, rate=count/total if total else None)


def whole_diagnostics(config, available, first, last, floor, details):
    from scrl.plan import PolicyStamp
    from scrl.rl.objectives import ScoredRollout, planner_advantages
    fields = {'S':'semantic_global', 'Q':'quality_global', 'R':'global'}
    logical, physical, groups, spreads = [], [], [], []
    protocols = set()
    def summary(items):
        return {name:{'mean':mean([r[k] for r in items]), 'std':pstdev(r[k] for r in items),
                      'min':min(r[k] for r in items), 'max':max(r[k] for r in items)}
                for name,k in fields.items()}
    for idx in range(first, last+1):
        try:
            report, joined, raw = read_batch(available[idx], global_sq=True)
            check(report['planner_credit'] == WHOLE_CREDIT, 'mixed planner credit')
            check(report['plans_per_prompt'] == report['seeds_per_plan'] == 4, 'expected 4 plans x 4 seeds')
            stamp = PolicyStamp(**report['stamp'])
            protocols.add(stamp.reward)
            check(len(protocols) == 1, 'mixed reward protocols')
            for samples in raw.values():
                check(len(samples) == 4, 'incomplete physical seed group')
                physical.extend(samples.values())
                spreads.append(pstdev(r['executor'] for r in samples.values()))
            for key, candidates in sorted(joined.items()):
                names = [f'c{i}' for i in range(4)]
                check(set(candidates) == set(names), 'unexpected whole-plan candidate IDs')
                seeds = sorted(candidates[names[0]])
                rows, plans, candidate_rows = [], set(), {}
                for name in names:
                    items = candidates[name]
                    check(sorted(items) == seeds and len(seeds) == 4, 'incomplete paired seeds')
                    ret = []
                    for seed in seeds:
                        row, scores, _, parent = items[seed]
                        plan = FinitePlan(tuple(Segment(**s) for s in row['plan']['segments']))
                        rows.append(ScoredRollout(**dict(row, plan=plan, stamp=PolicyStamp(**row['stamp']))))
                        logical.append(scores)
                        ret.append(scores)
                        plans.add(canon(row['plan']))
                    candidate_rows[name] = {label:mean([r[k] for r in ret]) for label,k in fields.items()}
                    if details:
                        candidate_rows[name].update(plan=row['plan'], replay_segment=row['target_segment'],
                                                    physical_candidate=parent)
                adv = planner_advantages(rows, stamp, expected_candidates=names, expected_seeds=seeds,
                                         floor=floor, credit=WHOLE_CREDIT)
                info = dict(batch=idx, group=key, unique_plans=len(plans),
                            zero_planner_advantage=all(abs(v) <= 1e-8 for v in adv.values()),
                            planner_std=pstdev(candidate_rows[c]['R'] for c in names))
                if details:
                    info.update(candidates=candidate_rows, advantages=adv, seeds=seeds)
                groups.append(info)
        except (KeyError, ValueError, TypeError, OSError) as error:
            raise ValueError(f'batch {idx}: {error}') from error
    out = dict(schema_version=2, batch_range=[first,last], training_mode=config['training']['mode'],
               reward_protocol=next(iter(protocols)), aggregation='global-sq-v1', normalization_floor=floor,
               logical_videos=len(logical), physical_videos=len(physical),
               logical_scores=summary(logical), physical_scores=summary(physical),
               planner=dict(credit=WHOLE_CREDIT, prompt_groups=len(groups),
                    mean_unique_plans=mean([g['unique_plans'] for g in groups]),
                    zero_advantage_groups=fraction(sum(g['zero_planner_advantage'] for g in groups),len(groups)),
                    std_median=median(g['planner_std'] for g in groups)),
               executor=dict(physical_candidate_groups=len(spreads), std_median=median(spreads),
                             zero_spread=fraction(spreads.count(0),len(spreads))))
    if details:out['groups'] = groups
    return out


def diagnose(run, first=None, last=None, floor=.02, details=False):
    check(math.isfinite(floor) and floor > 0, 'floor must be positive and finite')
    config = json.loads((run/'config.json').read_bytes())
    reward_mode = config['reward'].get('mode', 'quality-first-v1')
    weight = config['reward'].get('quality_weight', .8)
    check(0 <= weight <= 1, 'invalid quality weight')
    available = {int(p.name):p for p in (run/'batches').iterdir()
                 if p.is_dir() and p.name.isdigit() and (p/'report.json').is_file()}
    check(available, 'no completed reports')
    first = min(available) if first is None else first
    last = max(available) if last is None else last
    check(0 <= first <= last, 'invalid inclusive batch range')
    if config['reward'].get('aggregation') == 'global-sq-v1':
        check(config['training'].get('planner_credit') == WHOLE_CREDIT,
              'global S/Q diagnostics expect whole-plan credit')
        return whole_diagnostics(config,available,first,last,floor,details)
    groups, exec_stds, local_ties, logical, actual = [], [], 0, 0, 0
    protocols = set()
    credits = set()
    configured_credit = config['training'].get('planner_credit')
    for batch in range(first,last+1):
        try:
            check(batch in available, 'missing report')
            report, joined, physical = read_batch(available[batch])
            protocols.add(report['stamp']['reward'])
            check(len(protocols) == 1, 'reward protocol changed within selection')
            credit = report.get('planner_credit', configured_credit or 'candidate-v1')
            credits.add(credit)
            check(len(credits) == 1 and configured_credit in (None, credit), 'planner credit changed')
            if credit == CROSSED_CREDIT:
                check(set(report['planner_field_advantages']) == set(joined), 'recorded prompt groups differ')
            for key,candidates in sorted(joined.items()):
                groups.append(dict(batch=batch, **prompt_stats(key,candidates,report,floor,details,credit)))
            for seeds in physical.values():
                check(len(seeds) == report['seeds_per_plan'], 'incomplete physical seed group')
                exec_stds.append(pstdev(r['executor'] for r in seeds.values()))
                local_ties += len({r['semantic_local'] for r in seeds.values()}) == 1
                actual += len(seeds)
            logical += report['videos']
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError(f'batch {batch}: {error}') from error
    n, e = len(groups), len(exec_stds)
    winners = [g['split_winners'] for g in groups]
    unique = [pair for pair in winners if len(pair)==2 and None not in pair]
    out = dict(schema_version=1, batch_range=[first,last], training_mode=config['training']['mode'],
               reward_protocol=next(iter(protocols)), normalization_floor=floor, quality_weight=weight,
               logical_videos=logical, physical_videos=actual,
               planner=dict(prompt_groups=n, mean_unique_plans=mean([g['unique_plans'] for g in groups]),
                   std_median=median(g['planner_std'] for g in groups),
                   std_below_floor=fraction(sum(g['planner_std'] < floor for g in groups),n),
                   semantic_all_tied=fraction(sum(g['semantic_std'] == 0 for g in groups),n),
                   mean_weighted_quality_std=weight*mean([g['quality_std'] for g in groups]),
                   mean_weighted_semantic_std=(1-weight)*mean([g['semantic_std'] for g in groups]),
                   even_seed_groups=sum(bool(p) for p in winners),
                   seed_split_agreement=fraction(sum(a==b for a,b in unique),len(unique))),
               executor=dict(physical_candidate_groups=e, std_median=median(exec_stds),
                   std_below_floor=fraction(sum(v < floor for v in exec_stds),e),
                   zero_spread=fraction(exec_stds.count(0),e),
                   local_semantic_all_tied=fraction(local_ties,e)))
    if details:
        out['groups'] = groups
    planner = out['planner']
    planner['credit'] = next(iter(credits))
    if planner['credit'] == CROSSED_CREDIT:
        planner['zero_by_axis'] = {axis: fraction(sum(g['zero_by_axis'][axis] for g in groups), n)
                                   for axis in ('caption', 'duration')}
        planner['checked_against_training_receipt'] = floor == .02
    else:
        pairs = sum(g['same_target_pairs'] for g in groups)
        planner.update(opposite_same_target_pairs=fraction(sum(g['opposite_pairs'] for g in groups),pairs),
                       opposite_same_target_groups=fraction(sum(g['opposite_pairs'] > 0 for g in groups),n))
        planner['credit_comparison'] = {
            'before':'candidate-v1', 'after':'focal-mean-v1', 'zero_tolerance':1e-8,
            'conflict_groups_before':planner['opposite_same_target_groups'],
            'conflict_groups_after':fraction(sum(g['focal_opposite_pairs']>0 for g in groups),n),
            'zero_groups_before':fraction(sum(g['legacy_zero'] for g in groups),n),
            'zero_groups_after':fraction(sum(g['focal_zero'] for g in groups),n),
            'newly_zero_groups':fraction(sum(g['focal_zero'] and not g['legacy_zero'] for g in groups),n)}
    if reward_mode == 'event-v2':
        # A multiplicative gate has no independent additive quality contribution.
        out['reward_mode'] = reward_mode
        out.pop('quality_weight')
        planner = out['planner']
        planner.pop('mean_weighted_quality_std')
        planner.pop('mean_weighted_semantic_std')
        planner.update(mean_quality_std=mean([g['quality_std'] for g in groups]),
                       mean_task_std=mean([g['semantic_std'] for g in groups]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--first-batch',type=int,help='inclusive directory index, not completed round')
    p.add_argument('--last-batch',type=int,help='inclusive directory index')
    p.add_argument('--floor',type=float,default=.02)
    p.add_argument('--details',action='store_true',help='include candidate plans and per-seed scores')
    a = p.parse_args(argv)
    try:
        result = diagnose(a.run,a.first_batch,a.last_batch,a.floor,a.details)
    except (ValueError, KeyError, TypeError, OSError) as error:
        p.exit(2, f'{p.prog}: {error}\n')
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == '__main__':
    main()
