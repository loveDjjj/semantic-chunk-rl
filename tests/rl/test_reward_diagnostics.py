import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


from scrl.rl import diagnostics as audit


class RewardAuditTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run_path = Path(temp.name)
        self.batch = self.run_path/'batches'/'00164'
        self.batch.mkdir(parents=True)
        (self.run_path/'config.json').write_text(json.dumps({
            'reward':{'quality_weight':.8}, 'training':{'mode':'planner-only'}}))

    def fixture(self, values=((.7,.8,.7,.8),(.4,.4,.4,.4)), alias=False):
        jobs, rollouts = [], []
        candidates = len(values)+int(alias)
        seed_count = len(values[0])
        stamp = {'reward':'unit'}
        for i, scores in enumerate(values):
            aliases = [i,candidates-1] if alias and i==1 else [i]
            name = f'b164/p1/c{i}'
            # Physical parent may return seed shards in any order.
            for seed,score in reversed(list(enumerate(scores))):
                ret = dict(planner=score,executor=score,quality_global=(score-.1)/.8,
                           semantic_global=.5,semantic_local=.25)
                jobs.append(dict(parent_job=name,job_id=f'{name}/s{seed}',
                                 logical_job_ids=[f'b164/p1/c{a}' for a in aliases],
                                 physical_videos=1, rewards=[dict(seed=seed,returns=ret)]))
                for a in aliases:
                    segments = [dict(caption='target',num_chunks=2),dict(caption=f'suffix {i}',num_chunks=1)]
                    rollouts.append(dict(group_id='p1',candidate_id=f'c{a}',seed=seed,plan={'segments':segments},
                                         original_prompt='text',stamp=stamp,prefix_id='prefix',target_segment=0,
                                         planner_return=score,executor_return=score))
        (self.batch/'report.json').write_text(json.dumps(dict(state='passed',stamp=stamp,
            prompts=1, plans_per_prompt=candidates, seeds_per_plan=seed_count, rollouts=rollouts,
            physical_videos=len(jobs),videos=len(rollouts))))
        (self.batch/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))

    def test_global_sq_aliases_no_local_fields_and_independent_targets(self):
        self.fixture(((.8,)*4,(.5,)*4,(.3,)*4),alias=True)
        (self.run_path/'config.json').write_text(json.dumps({'reward':{'aggregation':'global-sq-v1'},
            'training':{'mode':'joint','planner_credit':audit.WHOLE_CREDIT}}))
        path = self.batch/'jobs.jsonl'
        jobs = [json.loads(l) for l in path.read_text().splitlines()]
        for job in jobs:
            ret = job['rewards'][0]['returns']
            del ret['semantic_local']
            ret.update({'global':ret['planner'], 'quality_global':ret['planner'], 'semantic_global':ret['planner']})
        path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
        report_path = self.batch/'report.json'
        report = json.loads(report_path.read_bytes())
        report['planner_credit'] = audit.WHOLE_CREDIT
        for row in report['rollouts']:
            if row['candidate_id'] in ('c1', 'c3'):
                row.update(prefix_id='other-prefix',target_segment=1)
        report_path.write_text(json.dumps(report))
        result = audit.diagnose(self.run_path, details=True)
        self.assertEqual((result['logical_videos'],result['physical_videos']), (16,12))
        self.assertAlmostEqual(result['logical_scores']['S']['mean'],.525)
        self.assertAlmostEqual(result['physical_scores']['R']['mean'],1.6/3)
        self.assertEqual(result['groups'][0]['candidates']['c3']['replay_segment'],1)
        jobs[0]['rewards'][0]['returns']['quality_global'] = 0
        path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
        with self.assertRaisesRegex(ValueError,'S/Q reward'):
            audit.diagnose(self.run_path)

    def test_split_aliases_focal_conflicts_and_details(self):
        self.fixture(alias=True)
        out = audit.diagnose(self.run_path, details=True)
        self.assertEqual((out['logical_videos'],out['physical_videos']), (12,8))
        p, e = out['planner'], out['executor']
        self.assertEqual(p['semantic_all_tied']['count'],1)
        self.assertEqual(p['opposite_same_target_pairs'],audit.fraction(2,3))
        comparison = p['credit_comparison']
        self.assertEqual(comparison['conflict_groups_before'],audit.fraction(1,1))
        self.assertEqual(comparison['conflict_groups_after'],audit.fraction(0,1))
        self.assertEqual(comparison['newly_zero_groups'],audit.fraction(1,1))
        self.assertEqual(p['seed_split_agreement'],audit.fraction(1,1))
        self.assertEqual(p['mean_unique_plans'],2)
        self.assertEqual(e['zero_spread'],audit.fraction(1,2))
        detail = out['groups'][0]['candidate_detail']['c2']
        self.assertLess(detail['advantage'],0)
        self.assertAlmostEqual(detail['focal_mean_advantage'],0)
        self.assertEqual([r['seed'] for r in detail['seed_returns']],[0,1,2,3])
        self.assertEqual(detail['seed_returns'][0]['physical_candidate'],'b164/p1/c1')

    def test_disagreement_ties_odd_k_and_floor(self):
        self.fixture(((.49,.49,.51,.51), (.51,.51,.49,.49)))
        p = audit.diagnose(self.run_path)['planner']
        self.assertEqual(p['seed_split_agreement'],audit.fraction(0,1))
        self.assertEqual(p['std_below_floor'],audit.fraction(1,1))
        self.fixture(((.5,)*4,(.5,)*4))
        p = audit.diagnose(self.run_path)['planner']
        self.assertEqual(p['seed_split_agreement'],audit.fraction(0,0))
        self.assertEqual(p['credit_comparison']['zero_groups_after'],audit.fraction(1,1))
        self.assertEqual(p['credit_comparison']['newly_zero_groups'],audit.fraction(0,1))
        self.fixture(((.5,)*3,(.6,)*3))
        self.assertEqual(audit.diagnose(self.run_path)['planner']['even_seed_groups'],0)

    def test_event_mode_does_not_report_fixed_additive_quality_contribution(self):
        self.fixture()
        (self.run_path/'config.json').write_text(json.dumps({
            'reward': {'mode': 'event-v2'}, 'training': {'mode': 'joint'}}))
        report = audit.diagnose(self.run_path)
        self.assertEqual(report['reward_mode'], 'event-v2')
        self.assertNotIn('quality_weight', report)
        self.assertNotIn('mean_weighted_quality_std', report['planner'])
        self.assertIn('mean_quality_std', report['planner'])

    def test_crossed_receipt_separates_and_verifies_actual_field_credit(self):
        self.fixture(((.75,)*2, (.75,)*2, (.25,)*2, (.25,)*2))
        path = self.batch/'report.json'
        report = json.loads(path.read_bytes())
        report['planner_credit'] = audit.CROSSED_CREDIT
        recorded = report['planner_field_advantages'] = {'p1': {}}
        for row in report['rollouts']:
            name = row['candidate_id']
            c = int(name[1:])
            caption, duration = divmod(c, 2)
            row['plan']['segments'] = [dict(caption=f'event {caption}',num_chunks=1+duration),
                                       dict(caption='other',num_chunks=2-duration)]
            recorded['p1'][name] = dict(caption=1. if caption == 0 else -1., duration=0.)
        path.write_text(json.dumps(report))
        result = audit.diagnose(self.run_path, details=True)
        p = result['planner']
        self.assertEqual(p['credit'], audit.CROSSED_CREDIT)
        self.assertEqual(p['zero_by_axis'], {'caption':audit.fraction(0,1), 'duration':audit.fraction(1,1)})
        self.assertTrue(p['checked_against_training_receipt'])
        self.assertNotIn('credit_comparison', p)
        cell = result['groups'][0]['candidate_detail']['c2']
        self.assertEqual(cell['field_advantages'], {'caption': -1., 'duration': 0.})
        self.assertNotIn('focal_mean_advantage', cell)
        recorded['p1']['c0']['duration'] = .1
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'recorded crossed advantage differs'):
            audit.diagnose(self.run_path)
        self.assertFalse(audit.diagnose(self.run_path, floor=.5)['planner']['checked_against_training_receipt'])

    def test_missing_or_duplicated_seed_fails(self):
        self.fixture()
        path = self.batch/'jobs.jsonl'
        lines = path.read_text().splitlines(True)
        path.write_text(''.join(lines[1:]))
        with self.assertRaisesRegex(ValueError,'unmatched'):
            audit.diagnose(self.run_path)
        path.write_text(''.join(lines + [lines[0]]))
        with self.assertRaisesRegex(ValueError,'duplicated'):
            audit.diagnose(self.run_path)

    def test_broken_range_and_corrupt_report_fail(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError,'batch 163: missing'):
            audit.diagnose(self.run_path,163,164)
        path = self.batch/'report.json'
        report = json.loads(path.read_bytes())
        report['rollouts'][0]['planner_return'] = .01
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'returns disagree'):
            audit.diagnose(self.run_path)
