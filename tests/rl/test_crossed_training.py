from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scrl.plan import FinitePlan, PolicyStamp, Segment
from scrl.rl.objectives import CROSSED_CREDIT, ScoredRollout, joint_advantages
from scrl.rl.train import planner_update_batch
from scrl.runtime.launch import prompt_seed
from scrl.runtime.worker import PlanningWorker


class CrossedTrainingTest(unittest.TestCase):
    stamp = PolicyStamp()
    candidates = ['c0', 'c1', 'c2', 'c3']
    seeds = [prompt_seed(10, 0, 0, 1) + 100 + s for s in range(4)]

    def rows(self, *, captions=('x', 'y'), durations=(2, 4), means=(.3, .1, .7, .5)):
        rows = []
        for c, name in enumerate(self.candidates):
            length = durations[c % 2]
            plan = FinitePlan((Segment(captions[c//2], length), Segment('next', 6-length), Segment('end', 2)))
            for s, seed in enumerate(self.seeds):
                rows.append(ScoredRollout('p0', name, seed, 'prompt', 'prefix', self.stamp, plan, 0,
                                          means[c]+s*.01, s*.01+means[c]))
        return rows

    def advantages(self, rows):
        return joint_advantages(rows[::-1], self.stamp, expected_candidates=self.candidates,
                                expected_seeds=self.seeds, credit=CROSSED_CREDIT)

    def test_separate_fields_and_within_cell_executor(self):
        advantages = self.advantages(self.rows())
        for i, c in enumerate(self.candidates):
            self.assertAlmostEqual(advantages['planner'][c]['caption'], -1 if i < 2 else 1)
            self.assertAlmostEqual(advantages['planner'][c]['duration'], 1 if i % 2 == 0 else -1)
            for s, seed in enumerate(self.seeds):
                self.assertAlmostEqual(advantages['executor'][c][seed], (s-1.5)*.5)

    def test_unavailable_duration_and_pure_interaction(self):
        result = self.advantages(self.rows(durations=(2, 2)))['planner']
        self.assertTrue(all(a['duration'] == 0 for a in result.values()))
        self.assertLess(result['c0']['caption'], 0)
        result = self.advantages(self.rows(means=(.8, .2, .2, .8)))['planner']
        self.assertTrue(all(abs(v) < 1e-8 for a in result.values() for v in a.values()))

    def test_rejects_wrong_grid_seed_or_changed_suffix(self):
        with self.assertRaisesRegex(ValueError, 'repeats'):
            self.advantages(self.rows()[:-1])
        for wrong in [FinitePlan((Segment('changed', 4), Segment('next', 2), Segment('end', 2))),
                      FinitePlan((Segment('y', 4), Segment('next', 2), Segment('other', 2))),
                      FinitePlan((Segment('y', 4), Segment('next', 4), Segment('end', 2)))]:
            rows = [replace(r, plan=wrong) if r.candidate_id == 'c3' else r for r in self.rows()]
            with self.assertRaisesRegex(ValueError, 'Crossed plans|Cartesian'):
                self.advantages(rows)

    def test_training_worker_to_coordinator(self):
        rows = self.rows()
        plans = [r.plan for r in rows[::4]]
        worker = PlanningWorker.__new__(PlanningWorker)
        worker.cfg = {'training': {'planner_credit': CROSSED_CREDIT}, 'sampling': {'max_new_tokens': 50}}
        worker.snapshot, worker.policy = 'snap', Mock(version=0)
        worker.policy.sample.return_value = (FinitePlan((Segment('a', 2),)), None, None)
        worker.policy.sample_crossed.return_value = [SimpleNamespace(plan=p, policy_version=0) for p in plans]
        with TemporaryDirectory() as tmp:
            group = worker('plan', dict(prompt='prompt', seed=10, snapshot='snap', version=0,
                                        random_focal=True, candidates=4, output=str(Path(tmp)/'samples.pt')))
            worker.policy.sample.assert_called_once()
            worker.policy.sample_crossed.assert_called_once()
            samples, scores, targets = planner_update_batch(
                {0: group}, rows[::-1], [0], seed=10, epoch=0, total_prompts=1, k=4, s=4,
                stamp=self.stamp, credit=CROSSED_CREDIT)
        self.assertEqual([s.plan for s in samples], plans)
        self.assertEqual(targets, [0]*4)
        self.assertEqual(scores, list(self.advantages(rows)['planner'].values()))
