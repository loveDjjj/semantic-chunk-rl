from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from unittest import TestCase
from unittest.mock import patch

from scrl.plan import digest
from scrl.models.scoring.protocol import IDENTITY, OBJECT_IDENTITY, RULES, install
from scrl.models.scoring.protocol import JUDGE_PROMPT
from scrl.rl.identity import IDENTITY_KEYS, rating_format_transition, training_fingerprint


class ScoringReleaseTests(TestCase):
    def test_rating_prompt_does_not_change_contract_or_task(self):
        from scrl.models.scoring.http_judge import HttpJudge
        with TemporaryDirectory() as tmp:
            cfg=dict(max_frames=16,max_new_tokens=240,pixels_per_frame=65536,contract_cache=tmp)
            judge=HttpJudge(cfg,8000,6)
            install(judge,IDENTITY)
            render=lambda s:judge.processor.apply_chat_template([{'content':[dict(type='text',text=s)]}])
            for original in ['extract a contract','other instruction']:
                self.assertEqual(render(original),original+'<|im_end|>')
            base=JUDGE_PROMPT+'\nTASK: {"scope":"global"}'
            self.assertEqual(render(base),base+'\n'+RULES+'<|im_end|>')
            self.assertEqual(IDENTITY['sha256'],digest(RULES))
            with self.assertRaisesRegex(ValueError,'Unknown'):install(judge,{'id':'unversioned'})

    def test_only_verified_prompt_transition_is_allowed(self):
        old={k:1 for k in IDENTITY_KEYS}
        old.update(training={'mode':'planner-only'},assets={'rewards':{'judge':'fixed'}},asset_manifest_sha256='assets',
                   reward_protocol=dict(version='v1',inference_backend='vllm-http-six-v1'),stamp={})
        new=deepcopy(old);new['reward_protocol']['rating_format']=OBJECT_IDENTITY
        for c in (old,new):
            c['stamp']['reward']='v1:'+digest([c['reward'],c['reward_protocol'],c['assets']['rewards']])
        fp=training_fingerprint(old)
        self.assertNotEqual(fp,training_fingerprint(new))
        self.assertEqual(rating_format_transition(old,new,fp)['old_reward'],old['stamp']['reward'])
        for key,value in [('reward',999),('dataset_sha256','changed'),('stamp',{'reward':'forged'})]:
            with self.assertRaises(ValueError):rating_format_transition(old,dict(new,**{key:value}),fp)
        with self.assertRaises(ValueError):rating_format_transition(old,new,'forged')

    def test_syntax_constraint_never_replaces_grounding_or_contract_instructions(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from scrl.models.scoring.protocol import OBJECT_CONSTRAINT, SYNTAX, SYNTAX_IDENTITY
        from scrl.models.scoring.http_judge import UserText
        send=Mock();judge=SimpleNamespace(processor=UserText(),request_text=send)
        install(judge,SYNTAX_IDENTITY)
        text=judge.processor.apply_chat_template([{'content':[dict(type='text',text=JUDGE_PROMPT+'{}')]}])
        self.assertIn(SYNTAX,text)
        judge.request_text(text);send.assert_called_with(text,constraint=OBJECT_CONSTRAINT)
        exact={'choice':['a','b']}
        judge.request_text(text,constraint=exact);send.assert_called_with(text,constraint=exact)
        judge.request_text('contract prompt');send.assert_called_with('contract prompt')

    def test_independent_requests_overlap_and_return_in_scope_order(self):
        from scrl.models.scoring.http_judge import parallel_class
        from scrl.models.scoring.http_judge import HttpJudge
        both=Barrier(2,timeout=3);calls=[];lock=Lock()
        def fake(_,texts,videos,metadata,**kwargs):
            with lock:calls.append((texts,videos,metadata,kwargs))
            if len(texts)==1 and texts[0] in ('global','local'):both.wait()
            return list(texts)
        with TemporaryDirectory() as tmp, patch.object(HttpJudge,'_generate',fake):
            judge=parallel_class()(dict(max_frames=16,max_new_tokens=240,pixels_per_frame=65536,contract_cache=tmp),8000,6)
            self.addCleanup(judge.close)
            self.assertEqual(judge._generate(['global','local'],[1,2],[3,4],return_invalid=True),['global','local'])
            self.assertCountEqual(calls,[(['global'],[1],[3],{'return_invalid':True}),(['local'],[2],[4],{'return_invalid':True})])
            self.assertEqual(judge._generate(['repair'],[2],[4]),['repair'])
