import json
from pathlib import Path
import tempfile
import unittest
from scrl.runtime.launch import JsonlTail


class MonitorTests(unittest.TestCase):
    def test_partial_append_is_read_only_after_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';p.write_text('{"n":1}\n{"n":')
            tail=JsonlTail(p);self.assertEqual(tail.read(),[{'n':1}]);self.assertEqual(tail.read(),[])
            with p.open('a') as f:f.write('2}\n')
            self.assertEqual(tail.read(),[{'n':2}]);self.assertEqual(tail.read(),[])

    def test_rotated_file_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';p.write_text('{"old":true}\n');tail=JsonlTail(p);tail.read()
            p.rename(Path(d)/'old.jsonl');p.write_text('{"new":true}\n')
            self.assertEqual(tail.read(),[{'new':True}])


if __name__=='__main__':unittest.main()
