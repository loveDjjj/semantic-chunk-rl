import errno
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scrl.runtime.transport import EvaluationDispatcher
from scrl.runtime.transport import GIB, effective_state, optional_write, wait_for_space


class StorageGuardTests(unittest.TestCase):
    def test_failure_marker_wins_even_if_unreadable_or_heartbeat_says_running(self):
        self.assertEqual(effective_state({'state':'collecting'},True,True),'failed')
        self.assertEqual(effective_state({'state':'collecting'},False,False),'coordinator_unavailable')
        self.assertEqual(effective_state({'state':'drained'},False,False),'drained')

    def test_only_capacity_failures_can_drop_telemetry(self):
        for n in (errno.ENOSPC,errno.EDQUOT):
            self.assertFalse(optional_write(Mock(side_effect=OSError(n,'full'))))
        with self.assertRaises(PermissionError):
            optional_write(Mock(side_effect=PermissionError(errno.EACCES,'denied')))

    def test_monitor_records_failure_on_independent_volume_when_shared_writes_fail(self):
        from scrl.runtime import launch as monitor
        with TemporaryDirectory() as tmp:
            shared=Path(tmp)/'shared';shared.mkdir()
            durable=Path(tmp)/'durable';durable.mkdir()
            (shared/'FAIL.json').touch()
            (durable/'status.json').write_text('{"state":"collecting","completed_rounds":22}')
            fn=monitor.atomic_json
            def write(path, row):
                if shared in path.parents:raise OSError(errno.ENOSPC,'full')
                return fn(path,row)
            cfg=dict(run=str(shared),durable_run=str(durable),nodes=[],executors=[])
            with patch('scrl.config.load_run',return_value=cfg), patch.object(monitor,'atomic_json',write), \
                    patch('sys.argv',['monitor','--once','--config',str(shared/'config.json')]):
                monitor.monitor_main()
            row=json.loads((durable/'monitor/latest.json').read_bytes())
            self.assertEqual((row['effective_state'],row['reported_state']),('failed','collecting'))
            self.assertEqual(row['status']['completed_rounds'],22)
            self.assertTrue((durable/'monitor/history.jsonl').is_file())
            self.assertFalse((shared/'monitor/latest.json').exists())

    def test_pause_uses_hysteresis_and_honors_drain(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);status=Mock();sleep=Mock()
            self.assertTrue(wait_for_space(p,status,Mock(side_effect=[2*GIB,8*GIB,13*GIB]),sleep))
            self.assertEqual(sleep.call_count,2)
            (p/'DRAIN').touch()
            self.assertFalse(wait_for_space(p,status,lambda:0,sleep))

    def test_failed_dispatch_is_never_counted_as_a_completed_score(self):
        d=EvaluationDispatcher({'token':'test','evaluations':[{'address':'192.0.2.1','port':8123}]})
        r={'plan':{'segments':[{'num_chunks':1}]}}
        with patch.object(d.peers[0],'call',side_effect=EOFError('closed')):
            with self.assertRaisesRegex(RuntimeError,'Scorer 0'):d.dispatch(r)
        self.assertEqual(d('health',{})['failed'],[1]);self.assertEqual(d.completed,[0]);self.assertEqual(d.active,[0])
        with patch.object(d.peers[0],'call',return_value={}):d.dispatch(r)
        self.assertEqual(d.completed,[1]);self.assertEqual(d.failed,[1])
