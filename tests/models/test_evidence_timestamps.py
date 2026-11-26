"""A terminal segment must use the same timestamp convention as its RGB scope."""

import copy
import unittest

from scrl.rl.reward import UnscorableReward
from scrl.models.scoring.protocol import frame_time_range, validate_video_judgment


class EvidenceTimestampTests(unittest.TestCase):
    def judgment(self, time):
        return dict(alignment=2, dynamics=3, physics=4, entry=None, exit=None,
                    evidence=[[time, 'The subject remains visible.']])

    def test_last_segment_target_fits_inside_video_context(self):
        # Exact original failure: target was [16.3125,20.8125] but context ended 20.75.
        context=frame_time_range(253,333,16)
        target=frame_time_range(261,333,16)
        self.assertEqual(context,[15.8125,20.75])
        self.assertEqual(target,[16.3125,20.75])
        self.assertGreaterEqual(target[0],context[0]);self.assertLessEqual(target[1],context[1])
        validated,adjustments=validate_video_judgment(self.judgment(target[1]),begin=253,end=333,fps=16)
        self.assertEqual(validated['evidence'][0][0],20.75);self.assertEqual(adjustments,[])

    def test_exact_exclusive_end_maps_one_frame_without_changing_scores_or_input(self):
        source=self.judgment(20.8125);original=copy.deepcopy(source)
        value,adjustments=validate_video_judgment(source,begin=253,end=333,fps=16)
        self.assertEqual(source,original)
        self.assertEqual(value['evidence'],[[20.75,original['evidence'][0][1]]])
        for key in ('alignment','dynamics','physics'):
            self.assertEqual(value[key],source[key])
        self.assertEqual(adjustments[0]['original_seconds'],20.8125)
        self.assertEqual(adjustments[0]['frame_seconds'],20.75)

    def test_genuinely_out_of_scope_evidence_is_rejected(self):
        for t in (0,15.75,20.82,20.875,1000,float('nan'),True,'20.875','20s','NaN',''):
            with self.subTest(t=t),self.assertRaises(UnscorableReward):
                validate_video_judgment(self.judgment(t),begin=253,end=333,fps=16)

    def test_quoted_seconds_preserve_scores_and_original(self):
        source = self.judgment('20.75')
        value, changes = validate_video_judgment(source, begin=253, end=333, fps=16)
        self.assertEqual(value, self.judgment(20.75))
        self.assertEqual(source, self.judgment('20.75'))
        self.assertEqual(changes[0]['reason'], 'quoted-seconds')

    def test_one_decimal_video_time_rounding_remains_valid(self):
        value,_=validate_video_judgment(self.judgment(20.8),begin=253,end=333,fps=16)
        self.assertEqual(value['evidence'][0][0],20.8)

    def test_nonterminal_scopes_and_other_frame_rates(self):
        for fps in (16,24,30):
            end=57
            value,changes=validate_video_judgment(self.judgment(end/fps),begin=0,end=end,fps=fps)
            self.assertEqual(value['evidence'][0][0],(end-1)/fps)
            self.assertEqual(len(changes),1)
        for begin,end,fps in ((2,2,16),(-1,5,16),(0,4,0),(0,4,float('nan'))):
            with self.assertRaises(ValueError):frame_time_range(begin,end,fps)


if __name__=='__main__':unittest.main()
