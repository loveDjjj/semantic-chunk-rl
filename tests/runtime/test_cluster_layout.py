"""Hardware resizing must not change policy identities or sample normalization."""

import unittest
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import Mock
from scrl.runtime.launch import compact_layout, explicit_layout, validate_nodes
from scrl.runtime.launch import baseline_pid, wait_ready


class ClusterLayoutTests(unittest.TestCase):
    def test_heterogeneous_inventory_validates_without_allocating_unknown_cards(self):
        nodes = [dict(index=0, gpus=2, address="node-0"), dict(index=1, gpus=8, address="node-1")]
        validate_nodes(nodes)
        layout = explicit_layout(nodes, [(0, 1), (1, 7)], actor_node=0)
        self.assertEqual(layout["executor_count"], 8)
        for field, invalid in (("gpus",0),("gpus",True),("index",0),("address","node-0"),("address",None)):
            with self.subTest(field=field,value=invalid), self.assertRaises(ValueError):
                validate_nodes([nodes[0], dict(nodes[1], **{field: invalid})])

    def nodes(self,n):
        return [dict(index=i,address=f'node-{i}',gpus=4) for i in range(n)]

    def test_96_cards_have_unique_ranks_ports_and_balanced_services(self):
        c=compact_layout(self.nodes(24),14,24,32150)
        self.assertEqual(len(c['executors']),82)
        self.assertEqual(len(c['evaluations']),14)
        self.assertEqual(len(c['planners']),24)
        roles=c['executors']+c['evaluations']+c['planners']
        self.assertEqual(len({(r['node'],r['port']) for r in roles}),len(roles))
        self.assertEqual({r['rank'] for r in c['executors']},set(range(82)))
        self.assertEqual(len({(r['node'],r['device']) for r in c['executors']+c['evaluations']}),96)
        self.assertEqual(c['rendezvous'],'tcp://node-0:32152')

    def test_old_layout_still_uses_38_executor_cards(self):
        self.assertEqual(compact_layout(self.nodes(11))['executor_count'],38)

    def test_mainline_uses_all_16_without_colocating_models(self):
        cfg = explicit_layout(self.nodes(4),[(2,2),(2,3),(3,2),(3,3)],actor_node=2,dedicated=True)
        roles = cfg['executors'] + cfg['evaluations'] + cfg['planners']
        self.assertEqual([len(cfg[k]) for k in ('executors','evaluations','planners')],[12,2,1])
        devices = {(r['node'],r['device']) for r in roles}
        self.assertNotIn((cfg['actor_node'],cfg['actor_physical_device']),devices)
        self.assertEqual(len(devices),15)

    def test_32_card_layout_covers_all_cards_with_27_executors(self):
        cfg=compact_layout(self.nodes(8),5,8,33150)
        self.assertEqual(cfg['executor_count'],27)
        self.assertEqual(len(cfg['evaluations']),5)
        self.assertEqual(len(cfg['planners']),8)
        roles=cfg['executors']+cfg['evaluations']+cfg['planners']
        self.assertEqual(len({(r['node'],r['device']) for r in cfg['executors']+cfg['evaluations']}),32)
        self.assertEqual(len({(r['node'],r['port']) for r in roles}),len(roles))

    def test_shared_actor_inventory_only_overlaps_sampling_and_update(self):
        raw=dict(service_cards=[(1,2),(1,3),(2,2),(2,3),(3,2),(3,3)],actor_node=2,actor_samples=True)
        cfg=explicit_layout(self.nodes(4),raw['service_cards'],actor_node=raw['actor_node'],
                            dedicated=True,actor_samples=raw['actor_samples'])
        roles=cfg['executors']+cfg['evaluations']+cfg['planners']
        self.assertEqual([len(cfg[k]) for k in ('executors','evaluations','planners')],[10,4,2])
        self.assertEqual(len({(r['node'],r['device']) for r in roles}),16)
        self.assertIn((cfg['actor_node'],cfg['actor_physical_device']),[(p['node'],p['device']) for p in cfg['planners']])
        self.assertEqual(len({(r['node'],r['port']) for r in roles}),16)

    def test_owned_judge_refuses_unknown_processes_and_exits_on_stop(self):
        self.assertEqual(baseline_pid(''), '')
        previous='237, 3030, /example/reserved-python'
        reservation=dict(commands=['/example/reserved-python'],max_mib=3030)
        with self.assertRaises(RuntimeError):baseline_pid(previous)
        self.assertEqual(baseline_pid(previous,reservation),'237')
        for conflict in [previous+'\n'+previous,previous.replace('3030','5000'),'34, 1, /other/job']:
            with self.assertRaises(RuntimeError):baseline_pid(conflict,reservation)
        for invalid in [dict(commands=['relative'],max_mib=1),dict(commands=['/ok'],max_mib=True),{}]:
            with self.assertRaises(ValueError):baseline_pid('',invalid)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'STOP').touch()
            self.assertFalse(wait_ready({8053:Mock()},Path(tmp)))

    def test_deployment_scripts_take_paths_from_environment(self):
        scripts = Path(__file__).resolve().parents[2]/'scripts'
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'nvidia-smi'
            fake.write_text('#!/usr/bin/env bash\nif [[ "$*" == *memory.free* ]]; then echo 8192; fi\n')
            fake.chmod(0o700)
            env = dict(PATH=tmp+os.pathsep+os.environ['PATH'], HOME=tmp)
            cmd = ['bash',str(scripts/'serve_video_judge.sh'),'0']
            result = subprocess.run(cmd,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('JUDGE_PYTHON',result.stderr)
            configured = dict(env,JUDGE_PYTHON=shutil.which('echo'),JUDGE_MODEL='/example/weights',JUDGE_MIN_FREE_MIB='4096')
            result = subprocess.run(cmd,env=configured,check=True,capture_output=True,text=True)
            self.assertIn('--model /example/weights',result.stdout)
            result = subprocess.run(cmd,env=dict(configured,JUDGE_MIN_FREE_MIB='9000'),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            result = subprocess.run(['bash',str(scripts/'install_research_env.sh'),'planner-4b'],
                                    env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('SCRL_DEPS',result.stderr)

    def test_invalid_nodes_and_excess_planner_copies_are_rejected(self):
        nodes=self.nodes(24);nodes[1]['address']=nodes[0]['address']
        with self.assertRaises(ValueError):compact_layout(nodes)
        nodes=self.nodes(24);nodes[1]['index']=2
        with self.assertRaises(ValueError):compact_layout(nodes)
        with self.assertRaises(ValueError):compact_layout(self.nodes(24),14,29)

if __name__=='__main__':unittest.main()
