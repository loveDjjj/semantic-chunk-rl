import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scrl.evaluation.planner_pool import plan_concurrently


class PlannerPoolTests(unittest.TestCase):
    def test_concurrent_disjoint_queues_keep_seed_and_existing_outputs(self):
        worker = """
import json, os, sys, time
from pathlib import Path
p = Path(sys.argv[1]); tasks=json.load(sys.stdin)
(p/('lane-'+str(os.getpid()))).touch()
while len(list(p.glob('lane-*'))) < 2: time.sleep(.01)
for task in tasks:
    with (p/task['id']).open('x') as f: json.dump(task,f)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'existing').write_text('keep')
            tasks=[{'id':str(i),'planner_seed':13+i} for i in range(5)]
            plan_concurrently([sys.executable,'-c',worker,tmp],tasks,2,6)
            self.assertEqual([json.loads((root/t['id']).read_text()) for t in tasks],tasks)
            self.assertEqual((root/'existing').read_text(),'keep')
            self.assertEqual(len(list(root.glob('lane-*'))),2)

    def test_failure_and_timeout_stop_other_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tests=[("sys.exit(4) if t=='0' else time.sleep(30)",4,subprocess.CalledProcessError),
                   ('time.sleep(30)',.6,subprocess.TimeoutExpired)]
            for action, timeout, error in tests:
                worker=("import sys,json,time,os;from pathlib import Path;"
                        "t=json.load(sys.stdin)[0]['id'];p=Path(sys.argv[1]);"
                        "(p/t).write_text(str(os.getpid()));"
                        "time.sleep(.15);"+action)
                with self.assertRaises(error):
                    plan_concurrently([sys.executable,'-c',worker,tmp],[{'id':'0'},{'id':'1'}],2,timeout)
                for f in Path(tmp).iterdir():
                    self.assertFalse(Path('/proc',f.read_text()).exists())
                    f.unlink()

    def test_reject_invalid_pool_and_duplicate_tasks(self):
        with self.assertRaisesRegex(ValueError,'LANES'):
            plan_concurrently([],[],4,1)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            plan_concurrently([],[{'id':'x'},{'id':'x'}],2,1)
