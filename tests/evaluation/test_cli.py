import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys

from scrl.evaluation.io import atomic_json, read, settings
from scrl.evaluation.generate import native_plan
from scrl.evaluation.protocol import DIMENSIONS
from scrl.evaluation.vbench import main
from scrl.cli import main as cli


class SummaryTests(unittest.TestCase):
    def test_prepare_public_cli_no_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/'model').mkdir()
            (base/'official/vbench').mkdir(parents=True)
            for name in DIMENSIONS:(base/'official/vbench'/(name+'.py')).touch()
            for name in ('weights.bin','lora.bin'):(base/name).write_bytes(name.encode())
            assets=dict(planner_model='model',planner_adapter=None,executor={
                'upstream':'model','wan':'model','backbone':'weights.bin','base_adapter':'lora.bin'})
            atomic_json(base/'paths.json',dict(asset_root=tmp,assets=assets))
            atomic_json(base/'official.json',[{'prompt_en':'Blue sky','dimension':list(DIMENSIONS)}])
            args=['eval','prepare','--run',str(base/'run'),'--arm','base','--paths',str(base/'paths.json'),
                  '--vbench-weights',str(base/'model'),'--vbench-source',str(base/'official'),
                  '--official-metadata',str(base/'official.json')]
            with contextlib.redirect_stdout(io.StringIO()):cli(args)
            cfg=read(base/'run/config.json')
            self.assertEqual(cfg['arms']['base'],{'checkpoint':None,'version':0,'planner_variant':'pre_sft'})
            tasks=read(base/'run/inputs/tasks.json')
            self.assertEqual(len(tasks),25)
            self.assertEqual(len({t['video_seed'] for t in tasks}),25)
            self.assertNotIn('rewards',cfg['assets'])
            with self.assertRaises(FileExistsError):cli(args)
            with self.assertRaisesRegex(ValueError,'records'):
                cli(['eval','vbench',str(base/'run'),'--arm','base'])
            args[3]=str(base/'native')
            # A native run must neither require a planner nor load a training adapter.
            assets.pop('planner_model')
            assets['executor']['policy_adapter']='does-not-exist'
            atomic_json(base/'paths.json',dict(asset_root=tmp,assets=assets))
            selected={**settings(),'fps':24,'max_new_tokens':101,'planner_seed_base':123,'video_seed_base':31415}
            selected['formula']['QUALITY_WEIGHT']=2
            external=base/'custom-recipe.json'
            atomic_json(external,selected)
            cmd=[sys.executable,'-m','scrl',*args,'--config',str(external),'--native-chunks','7']
            env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src')}
            bad=dict(selected,seed_stride=0)
            atomic_json(external,bad)
            invalid=subprocess.run(cmd,cwd=base,env=env,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(invalid.returncode,0)
            self.assertIn('seed_stride',invalid.stderr)
            self.assertFalse((base/'native').exists())
            atomic_json(external,selected)
            subprocess.run(cmd,cwd=base,env=env,check=True,capture_output=True,text=True,timeout=30)
            cfg=read(base/'native/config.json')
            task=read(base/'native/inputs/tasks.json')[0]
            self.assertEqual((cfg['fps'],cfg['max_new_tokens'],task['video_seed']),(24,101,31415))
            self.assertEqual(task['planner_seed'],123)
            self.assertEqual(cfg['recipe'],selected)
            self.assertEqual(cfg['formula']['QUALITY_WEIGHT'],2)
            self.assertIsNone(cfg['assets']['planner_model'])
            self.assertIsNone(cfg['assets']['executor']['policy_adapter'])
            self.assertEqual(cfg['denoising_mode'],'causal-chunkwise')
            plan=native_plan(task['prompt'],cfg['native_chunks'])
            self.assertEqual([(s.caption,s.num_chunks) for s in plan.segments],[('Blue sky',7)])
            self.assertEqual(plan.total_chunks*12-3,81)

    def test_complete_partial_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            formula=read(Path(__file__).resolve().parents[2]/'configs/vbench.json')['formula']
            atomic_json(run/'config.json',{})
            folder=run/'evaluation/frozen'
            for dim in DIMENSIONS:
                name=dim.replace('_',' ')
                bounds=formula['NORMALIZE_DIC'][name]
                want=.75 if name in formula['QUALITY_LIST'] else .25
                raw=bounds['Min']+want*(bounds['Max']-bounds['Min'])
                atomic_json(folder/(dim+'.json'),{'dimension':dim,'state':'complete','score':raw,'coverage':{'scored_videos':2}})
            output=run/'result.json'
            main([str(run),'--arm','frozen','--summary-only','--output',str(output)])
            score=read(output)['aggregate']
            for k,v in dict(quality=.75,semantic=.25,total=.65).items():self.assertAlmostEqual(score[k],v)
            with self.assertRaises(FileExistsError):
                main([str(run),'--arm','frozen','--summary-only','--output',str(output)])
            atomic_json(folder/'color.json',dict(dimension='color',score=None,state='unscorable',coverage={'scored_videos':0}))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                main([str(run),'--arm','frozen','--summary-only'])
            self.assertIsNone(json.loads(out.getvalue())['aggregate']['total'])
            with self.assertRaises(FileNotFoundError):
                main([str(run),'--arm','frozen'])
