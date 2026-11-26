import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class CodeReleaseTests(unittest.TestCase):
    def test_relocatable_source_and_no_private_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp) / 'release with spaces'
            command = ['sh', str(ROOT/'scripts/stage_code.sh'), str(release)]
            inventory = json.loads((ROOT / 'configs/cluster.example.json').read_bytes())
            inventory['service_cards'] = [[2,2],[2,3],[3,2],[3,3],[4,2],[4,3]]
            inventory.pop('reserved_inventory_node')
            inventory['nodes'][3].update(address='198.51.100.4',password='test-only-private-value')
            private = Path(tmp) / 'mock inventory.json'
            private.write_text(json.dumps(inventory))
            env = dict(os.environ, SCRL_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE='1',
                       SCRL_CLUSTER_INVENTORY=str(private), SCRL_CLUSTER_LAYOUT=str(ROOT/'configs/cluster.example.json'))
            subprocess.run(command, cwd=tmp, env=env, check=True, capture_output=True)
            self.assertFalse((release/'configs/cluster.json').exists())
            self.assertFalse((release/'.git').exists())
            for relative in ('docs/running.md', 'docs/server.md', 'docs/vbench.md', 'docs/results.md',
                             'docs/history/vbench-results-through-20261125.md',
                             'docs/experiments/20261126-vbench-full-4b-base-vs-sft-longlive-v1.md', 'docs/changelog.md',
                             'datasets/README.md', 'scripts/vbench_score_when_ready.sh', 'tools/cache_contention_probe.py'):
                self.assertEqual((release/relative).read_bytes(), (ROOT/relative).read_bytes())
                self.assertIn('  '+relative+'\n', (release/'SHA256SUMS').read_text())
            public = release/'configs/cluster.runtime.json'
            self.assertEqual(list((release/'datasets').iterdir()), [release/'datasets/README.md'])
            exported = json.loads(public.read_bytes())
            self.assertEqual(exported['nodes'][3]['address'], '198.51.100.4')
            self.assertEqual(exported['service_cards'], json.loads((ROOT/'configs/cluster.example.json').read_bytes())['service_cards'])
            self.assertEqual(exported['reserved_inventory_node'], 5)
            for node in exported['nodes']:
                self.assertLessEqual(set(node), {'address', 'index', 'gpus', 'physical_gpus'})
            subprocess.run(['sha256sum', '--check', '--status', 'SHA256SUMS'], cwd=release, check=True)
            subprocess.run(['sh', str(release/'scripts/rl.sh'), '--help'], cwd=tmp,
                           env=env, check=True, capture_output=True)
            self.assertNotEqual(subprocess.run(command, cwd=tmp, env=env, capture_output=True).returncode, 0)
