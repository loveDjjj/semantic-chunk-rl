import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


from scrl.rl import preflight as audit


class FreshJointTests(unittest.TestCase):
    def setUp(self):
        self.spec = audit.read(audit.SPEC)['preflight']
        self.cfg = dict(experiment_protocol=self.spec['protocol'], stamp=dict(planner=0,executor=0),
            training=dict(mode='joint',planner_backend='verl'),executor_denoising='segment-joint',
            resume=None,init_from=None)

    def test_cannot_resume_or_inherit_or_pretend_chunkwise_capture(self):
        for fields in ({'resume':'renamed'}, {'init_from':'some/alias'}, {'stamp':{'planner':205,'executor':0}},
                       {'training':{'mode':'planner-only','planner_backend':'native'}},
                       {'files':{'planner.pt':'hash'}}, {'experiment_protocol':'legacy'},
                       {'executor_denoising':'chunkwise'}, {'eval_chunkwise':True}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                audit.check_origin(dict(self.cfg, **fields),self.spec)
        audit.check_origin(self.cfg,self.spec)

    def test_no_inputs_reports_separate_blockers(self):
        out = audit.audit()
        self.assertFalse(out['launchable'])
        self.assertEqual(len(out['input_blockers']),4)
        self.assertEqual(out['input_status'],'blocked')

    def test_official_overlap_requires_explicit_matching_approval(self):
        official, extra = 'a'*64, 'b'*64
        approval = dict(official_eval_metadata_sha256=official)
        manifest = dict(evaluation_metadata_sha256=[], official_training_metadata_sha256=[official],
                        official_training_prompt_count=600, counts=dict(train=1800, validation=200))
        field = 'approved_official_training_metadata_sha256'
        approved = dict(approval, **{field: [official]})
        audit.check_official_scope(manifest, approved)
        audit.check_official_scope(dict(evaluation_metadata_sha256=[official]), approval)
        for m, a in [
            (manifest, approval), (manifest, dict(approval, **{field: [extra]})),
            (manifest, dict(approval, **{field: [official, extra]})),
            (manifest, dict(approved, official_eval_metadata_sha256=extra)),
            (dict(manifest, official_training_metadata_sha256=[official, extra]), approved),
            (dict(manifest, evaluation_metadata_sha256=[official]), approved),
            (dict(manifest, official_training_prompt_count=True), approved),
            (dict(manifest, official_training_prompt_count=2001), approved),
            (dict(evaluation_metadata_sha256=[]), approval),
            (dict(evaluation_metadata_sha256=[official]), approved),
        ]:
            with self.subTest(m=m, a=a), self.assertRaises(ValueError):
                audit.check_official_scope(m, a)

    def test_input_pins_do_not_qualify_hardware(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(name, data):
                path = root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data if isinstance(data,bytes) else json.dumps(data).encode())
                return path
            digest = lambda data: hashlib.sha256(data).hexdigest()
            sha = 'c'*64
            split_hashes = {}
            for split in ('train','validation'):
                group = digest(split.encode())
                row = dict(prompt_id=group,group_id=group,reviewer='owner',teacher_model='fixture',template_sha256=sha)
                split_hashes[split+'.jsonl'] = audit.file_hash(write('dataset/'+split+'.jsonl',row))
            manifest = write('dataset/manifest.json',dict(format='reviewed-teacher-sft-v1',teachers=['fixture'],
                sha256=split_hashes,counts={'train':1,'validation':1},evaluation_metadata_sha256=[sha]))
            teacher_pin = audit.file_hash(manifest)
            write('sft/comparison.json',dict(dataset_manifest_sha256=teacher_pin,updates=1,validation=1))
            for adapter in ('sft/adapter','executor'):
                write(adapter+'/adapter_config.json',dict(peft_type='LORA'))
                write(adapter+'/adapter_model.safetensors',adapter.encode())
            write('model/config.json',dict(model_type='qwen3_5',text_config={'hidden_size':2560,'num_hidden_layers':32}))
            ex = dict(policy_adapter=str(root/'executor'),upstream=str(root),wan=str(root))
            for key in ('backbone','base_adapter'):
                ex[key] = str(write(key,key.encode()))
                self.spec['executor'][key+'_sha256'] = digest(key.encode())
            protocol = write('spec.json',self.spec)
            cfg = write('candidate.json',dict(self.cfg,assets=dict(planner_model=str(root/'model'),
                planner_adapter=str(root/'sft/adapter'),executor=ex)))
            inputs = dict(teacher_dataset=str(root/'dataset'),teacher_manifest_sha256=teacher_pin,
                official_eval_metadata_sha256=sha,sft_run=str(root/'sft'),
                planner_adapter_sha256=digest(b'sft/adapter'),executor_policy_sha256=digest(b'executor'))
            with patch.object(audit,'SPEC',protocol):
                out = audit.audit(cfg,write('approved.json',inputs))
                self.assertEqual(out['input_blockers'],{})
                self.assertFalse(out['launchable'])
                inputs['teacher_manifest_sha256'] = 'a'*64
                bad = audit.audit(cfg,write('approved.json',inputs))['input_blockers']
                self.assertIn('reviewed_teacher_manifest',bad)
                self.assertIn('teacher_sft_planner',bad)
