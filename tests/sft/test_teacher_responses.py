"""Offline classification and provenance for a single teacher Responses request."""
import json
import os
import unittest
from unittest.mock import Mock, patch

from scrl.plan import digest
from scrl.models import responses as client


def output(*parts):
    return {"status":"completed","output":[{"type":"message","content":list(parts)}]}


class TeacherResponsesTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(client.requests, "Session")
        self.session = patcher.start().return_value.__enter__.return_value
        self.addCleanup(patcher.stop)
        self.key = "private-test-token"
        env = patch.dict(os.environ, OPENAI_API_KEY=self.key)
        env.start()
        self.addCleanup(env.stop)
        self.reply = self.session.post.return_value
        self.reply.status_code = 200
        self.part = {"type":"output_text","text":json.dumps([{"step_index":0,"prompt":"A quiet street.","num_chunks":2}])}

    def teacher(self, effort="low"):
        return client.request_teacher_plan("Street", "teacher-v1", "https://relay.invalid/v1", effort=effort)

    def test_http_opt_in_is_exact_and_not_a_global_switch(self):
        origin = "http://192.0.2.7:8080/v1"
        with patch.dict(os.environ, clear=True):
            with self.assertRaises(ValueError):
                client.relay_url(origin)
            os.environ["SCRL_APPROVED_HTTP_RELAY"] = origin
            self.assertEqual(client.relay_url(origin), origin)
            for other in (origin + "/other", origin.replace(":8080", ":80"), origin.replace(".7", ".8")):
                with self.assertRaises(ValueError):
                    client.relay_url(other)

    def test_receipt_matches_actual_teacher_request_not_eval(self):
        self.reply.json.return_value = dict(output(self.part), id="resp_123", model="teacher-v1")
        plan, receipt = self.teacher()
        req = self.session.post.call_args.kwargs
        self.assertFalse(req["allow_redirects"])
        self.assertFalse(self.session.trust_env)
        self.assertEqual(plan.total_chunks, 2)
        self.assertEqual(req["json"]["reasoning"], {"effort":"low"})
        self.assertEqual(req["json"]["instructions"], client.TEACHER_FORMAT)
        self.assertEqual(receipt, dict(effort="low", template_sha256=client.teacher_template_sha256("low"),
            request_sha256=digest(req["json"]),response_id="resp_123",response_model="teacher-v1"))
        self.assertEqual(receipt["request_sha256"],client.teacher_request_sha256("Street","teacher-v1","low"))
        for args in (("Street","teacher-v2"),("Square","teacher-v1")):
            self.assertNotEqual(receipt["request_sha256"],client.teacher_request_sha256(*args,"low"))
        with patch.object(client, "TEACHER_FORMAT","new template"):
            self.assertNotEqual(receipt["template_sha256"],client.teacher_template_sha256("low"))
        self.reply.json.return_value.update(id=self.key,model="not safe\nmetadata")
        _, no_meta = self.teacher()
        self.assertEqual(set(no_meta),{"effort","template_sha256","request_sha256"})
        _, higher = self.teacher("medium")
        self.assertEqual(higher["effort"], "medium")
        self.assertEqual(self.session.post.call_args.kwargs["json"]["reasoning"]["effort"], "medium")
        for field in ("template_sha256", "request_sha256"):
            self.assertNotEqual(receipt[field], higher[field])
        self.assertEqual(client.request_plan("Street","teacher-v1","https://relay.invalid/v1"),plan)
        self.assertEqual(self.session.post.call_args.kwargs["json"]["instructions"],client.FORMAT)

    def test_http_and_transport_errors_are_classified_without_sensitive_details(self):
        for status, error in [(429,client.RetryableTeacherError),(500,client.RetryableTeacherError),
                              (503,client.RetryableTeacherError),(307,client.FatalTeacherError),
                              (401,client.FatalTeacherError),(403,client.FatalTeacherError),(400,client.FatalTeacherError)]:
            self.reply.status_code = status
            with self.subTest(status=status), self.assertRaises(error) as failure:
                self.teacher()
            self.assertEqual(str(failure.exception),f"Teacher relay HTTP {status}")
        for exc,error in ((client.requests.Timeout(self.key),client.RetryableTeacherError),
                          (client.requests.exceptions.SSLError(self.key),client.FatalTeacherError)):
            self.session.post.side_effect = exc
            with self.assertRaises(error) as failure:
                self.teacher()
            self.assertNotIn(self.key,str(failure.exception))

    def test_invalid_plans_retry_but_refusals_do_not(self):
        cases = [(output(self.part,self.part),client.InvalidTeacherPlanError),
                 (output({"type":"output_text","text":"not JSON"}),client.InvalidTeacherPlanError),
                 ({"status":"incomplete","output":[]},client.InvalidTeacherPlanError),
                 ({"status":"failed","error":{"code":"server_error"}},client.RetryableTeacherError),
                 ({"status":"incomplete","incomplete_details":{"reason":"content_filter"}},client.FatalTeacherError),
                 (output({"type":"refusal","refusal":self.key}),client.FatalTeacherError)]
        for body, error in cases:
            self.reply.json.return_value = body
            with self.subTest(error=error), self.assertRaises(error) as failure:
                self.teacher()
            self.assertNotIn(self.key,str(failure.exception))
        self.reply.json.side_effect = ValueError(self.key)
        with self.assertRaises(client.InvalidTeacherPlanError) as failure:
            self.teacher()
        self.assertNotIn(self.key,str(failure.exception))
