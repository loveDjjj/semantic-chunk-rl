import copy
import json
import tempfile
import unittest
from pathlib import Path

from scrl.rl.identity import IDENTITY_KEYS, training_fingerprint, verify_campaign_resume


class ResumeIdentityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.checkpoint = self.root / 'checkpoints' / 'checkpoint-00070'
        self.checkpoint.mkdir(parents=True)
        self.old = dict.fromkeys(IDENTITY_KEYS, 1)
        self.old['assets'] = {'rewards': {r: {'python': '/old', 'weights': 'frozen'}
                                         for r in ('judge', 'quality')}}
        self.saved = training_fingerprint(self.old)
        (self.root / 'config.json').write_text(json.dumps(self.old))
        self.new = copy.deepcopy(self.old)
        for s in self.new['assets']['rewards'].values():
            s['python'] = '/new'

    def test_interpreters_only(self):
        verify_campaign_resume(self.new, self.saved, self.checkpoint)
        self.assertEqual(self.new['assets']['rewards']['judge']['python'], '/new')

    def test_reject_weight_change(self):
        self.new['assets']['rewards']['judge']['weights'] = 'changed'
        with self.assertRaises(ValueError):
            verify_campaign_resume(self.new, self.saved, self.checkpoint)

    def test_reject_parent_mismatch(self):
        (self.root / 'config.json').write_text(json.dumps(self.new))
        with self.assertRaises(ValueError):
            verify_campaign_resume(self.new, self.saved, self.checkpoint)
