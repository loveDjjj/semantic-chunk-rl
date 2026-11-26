import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scrl.rl.reward import UnscorableReward
from scrl.runtime.transport import Peer, ScoringPeer, EvaluationDispatcher, serve
from scrl.runtime.worker import ExecutionWorker
from scrl.runtime import worker


class ScoreReviewTests(unittest.TestCase):
    def test_evaluation_roles_do_not_publish_global_failure_on_request_rejection(self):
        for role, symbol in [('evaluation','scrl.runtime.worker.EvaluationWorker'),
                             ('dispatcher','scrl.runtime.transport.EvaluationDispatcher')]:
            with self.subTest(role=role), TemporaryDirectory() as tmp:
                engine = Mock(side_effect=[UnscorableReward('no evidence'),{'ok':True},OSError('disk')])
                def exercise(port, token, handler, **kwargs):
                    with self.assertRaises(UnscorableReward):
                        handler('evaluate', {})
                    failure.assert_not_called()
                    self.assertEqual(handler('evaluate',{}), {'ok':True})
                    with self.assertRaises(OSError):
                        handler('evaluate', {})
                    failure.assert_called_once()
                argv = ['worker','--role',role,'--config','unused','--node','0','--port','0']
                with (patch.object(sys,'argv',argv), patch.object(worker.torch,'set_num_threads'),
                      patch('os.umask'), patch('scrl.eventlog.event'),
                      patch('scrl.config.load_run',return_value={'run':tmp,'token':'test'}),
                      patch(symbol, return_value=engine), patch.object(worker,'serve',side_effect=exercise),
                      patch.object(worker,'record_failure') as failure):
                    worker.main()

    def test_evaluate_rejection_survives_peer_transport(self):
        parent, child = socket.socketpair()
        parent.settimeout(3)
        child.settimeout(3)

        class Once:
            def __init__(self, address, handler):
                self.handler = handler

            def __enter__(self):
                return self

            def __exit__(self, *error):
                pass

            def serve_forever(self, **kwargs):
                self.handler(child, ('localhost', 1), self)

        handler = Mock(side_effect=UnscorableReward('invalid evidence'))
        with (parent, child, ThreadPoolExecutor(1) as pool,
              patch('socketserver.TCPServer', Once), patch('socket.create_connection', return_value=parent)):
            server = pool.submit(serve, 0, 'test', handler)
            with self.assertRaisesRegex(UnscorableReward, 'invalid evidence'):
                Peer('localhost', 0, 'test').call('evaluate', attempt='round-1')
            server.result(timeout=3)

    def test_rejected_video_is_reviewed_once_without_regeneration(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'video.pt'
            path.write_bytes(b'unchanged')
            calls = []
            broker = EvaluationDispatcher({'evaluations': []})
            data = dict(global_scores=dict(quality={}, semantic={}), segment_scores=dict(quality={}, semantic={}),
                        protocol='fixed', boundary_quality=None)
            def judge(req):
                calls.append(req)
                self.assertEqual(Path(req['latent_path']).read_bytes(), b'unchanged')
                if len(calls) == 1:
                    raise UnscorableReward('wrong slot')
                return data
            broker.dispatch = judge
            deliveries = []
            def wire(op, **payload):
                deliveries.append(payload['request_id'])
                if len(deliveries) == 1:
                    with self.assertRaises(UnscorableReward):
                        broker(op, payload)
                    raise EOFError('lost rejection reply')
                return broker(op, payload)
            peer = ScoringPeer('dispatcher', 0, 'test')
            payload = dict(job_id='b1/p2/c0', seed=9, attempt='round-1',
                           final_id='unchanged', latent_path=str(path))
            with patch.object(Peer,'call',side_effect=wire), patch('scrl.runtime.transport.time.sleep'):
                _, score, history = ExecutionWorker._score_file(peer, payload, None, None, lambda b: 7,
                                                               2, Path(tmp)/'receipts')
            self.assertEqual(score, 7)
            self.assertEqual(len(calls), 2)
            self.assertEqual(deliveries[:2], [history[0]['request_id']]*2)
            self.assertEqual(deliveries[2:], [history[1]['request_id']])
            self.assertNotEqual(*[r['request_id'] for r in history])
            self.assertEqual([r['state'] for r in history], ['unscorable','complete'])
            self.assertNotEqual(calls[0]['attempt'], calls[1]['attempt'])
            receipt = json.loads(next((Path(tmp)/'receipts').glob('*.json')).read_text())
            self.assertEqual(receipt['attempts'],history)
            self.assertEqual(receipt['seed'],9)
            self.assertFalse(path.exists())

    def test_no_reward_on_exhaustion_and_no_reassessment_on_io_error(self):
        for error, count in [(UnscorableReward('no sampled witness'),2),(OSError('disk'),1)]:
            with self.subTest(error=error), TemporaryDirectory() as tmp:
                peer = SimpleNamespace(call=Mock(side_effect=error))
                admission = Mock()
                latent = Path(tmp)/'video.pt'
                latent.touch()
                payload = dict(job_id='b1/p2/c0', seed=9, final_id='f', attempt='round-1',latent_path=str(latent))
                with self.assertRaises(type(error)) as raised:
                    ExecutionWorker._score_file(peer, payload, admission, 'v', Mock(), 2, Path(tmp)/'reviews')
                if isinstance(error,UnscorableReward):
                    self.assertIn('b1/p2/c0 seed 9', str(raised.exception))
                self.assertEqual(peer.call.call_count,count)
                receipt = json.loads(next((Path(tmp)/'reviews').glob('*.json')).read_text())
                self.assertEqual(len(receipt['attempts']),count)
                admission.call.assert_called_once_with('release_video',key='v')
