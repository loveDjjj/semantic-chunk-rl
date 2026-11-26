"""Read-only CPU input preflight for the fresh teacher-4B joint experiment; never launches."""
import argparse
import json
from pathlib import Path
import re
import sys

from scrl.plan import digest
from scrl.config import DEFAULT_CONFIGS, file_hash, read
from scrl.sft.data import load_splits

SPEC = DEFAULT_CONFIGS/'rl.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'independent SHA-256 required')
    return value


def check_official_scope(manifest, approvals):
    official = pin(approvals['official_eval_metadata_sha256'])
    excluded = manifest['evaluation_metadata_sha256']
    included = manifest.get('official_training_metadata_sha256', [])
    approved = approvals.get('approved_official_training_metadata_sha256', [])
    for values in (excluded, included, approved):
        require(isinstance(values, list), 'expected a list of official metadata digests')
        for value in values:
            pin(value)
    require(set(excluded).isdisjoint(included), 'metadata cannot be both included and excluded')
    require(set(approved) == set(included), 'official training metadata requires matching independent approval')
    if included:
        n = manifest.get('official_training_prompt_count')
        require(type(n) is int and 0 <= n <= sum(manifest['counts'].values()),
                'missing or invalid official training prompt count')
    require(official in excluded or official in included,
            'official evaluation metadata must be excluded or explicitly approved for training')


def check_origin(cfg, spec):
    require(cfg.get('format') not in ('planner-only-v1','split-policies-v1') and 'files' not in cfg,
            'checkpoint payload is not a fresh campaign config')
    require('resume' in cfg and 'init_from' in cfg and cfg['resume'] is None and cfg['init_from'] is None,
            'resume/init_from prohibited, including renamed or linked 205/250 checkpoints')
    require(all(type(cfg['stamp'].get(k)) is int and cfg['stamp'][k] == 0 for k in ('planner','executor')),
            'initial planner and executor stamps must both be zero')
    require(all(cfg['training'].get(k)==v for k,v in spec['training'].items()), 'requires joint mode and verl actor')
    require(cfg.get('experiment_protocol') == spec['protocol'], 'missing new experiment_protocol identity')
    require(cfg.get('executor_denoising') == spec['executor']['denoising'], 'requires segment-joint denoising')
    require(not cfg.get('eval_chunkwise'), 'chunkwise capture is not implemented for executor training')


def audit(candidate=None, approvals=None, *, recipe_path=None):
    recipe = read(recipe_path or SPEC)
    spec = recipe.get('preflight', recipe)
    evidence, errors = {}, {}
    def probe(name, fn):
        try:
            evidence[name] = fn()
        except KeyError as exc:
            errors[name] = f'missing required field: {exc.args[0]}'
        except (ValueError, TypeError, OSError, AttributeError) as exc:
            errors[name] = str(exc)
    cfg = candidate if isinstance(candidate, dict) else (read(candidate) if candidate else {})
    approvals = approvals if isinstance(approvals, dict) else (read(approvals) if approvals else {})
    def origin():
        require(cfg, 'supply a proposed fresh effective config; no existing run is selected')
        check_origin(cfg,spec)
        return digest(candidate) if isinstance(candidate, dict) else file_hash(candidate)
    def teacher():
        require(approvals.get('teacher_dataset'), 'missing independently approved teacher dataset and manifest pin')
        ds = Path(approvals['teacher_dataset'])
        sha = pin(approvals['teacher_manifest_sha256'])
        manifest = read(ds/'manifest.json')
        require(manifest.get('format') == spec['planner']['teacher_manifest_format'], 'not a reviewed teacher export')
        check_official_scope(manifest, approvals)
        train, val = load_splits(ds,manifest_sha256=sha)
        require(manifest.get('teachers') and {r['group_id'] for r in train}.isdisjoint(r['group_id'] for r in val),
                'missing teacher provenance or overlaps across split groups')
        require(all(r.get('reviewer') and r.get('teacher_model') and r.get('template_sha256')
                    for rows in (train,val) for r in rows), 'incomplete review fields')
        return sha
    def planner():
        require(approvals.get('sft_run'), 'missing approved teacher-SFT completion receipt and adapter pin')
        model = Path(cfg['assets']['planner_model'])
        arch = read(model/'config.json')
        expected = spec['planner']
        require(arch['model_type']==expected['model_type'] and
                arch['text_config']['hidden_size']==expected['text_hidden_size'] and
                arch['text_config']['num_hidden_layers']==expected['text_layers'], 'requires Qwen3.5-4B architecture')
        adapter = Path(cfg['assets']['planner_adapter'])
        require(adapter.resolve() == (Path(approvals['sft_run'])/'adapter').resolve(),
                'planner adapter does not match reviewed SFT output')
        comparison = read(adapter.parent/'comparison.json')
        sha = file_hash(adapter/'adapter_model.safetensors')
        require(sha != expected['forbidden_legacy_sft_sha256'], 'legacy mainline SFT is prohibited')
        require(comparison.get('dataset_manifest_sha256') == pin(approvals['teacher_manifest_sha256']) and
                comparison.get('updates',0)>0 and comparison.get('validation',0)>0,
                'SFT completion receipt does not match approved teacher dataset')
        require(sha == pin(approvals['planner_adapter_sha256']), 'SFT adapter digest mismatch')
        require(read(adapter/'adapter_config.json')['peft_type'] == 'LORA', 'planner LoRA required')
        return sha
    def executor():
        require(cfg.get('assets',{}).get('executor'), 'missing executor asset inventory and trainable policy')
        ex = cfg['assets']['executor']
        require(ex.get('policy_adapter'), 'missing trainable executor.policy_adapter, frozen backbone is insufficient')
        policy = Path(ex['policy_adapter'])
        require(read(policy/'adapter_config.json')['peft_type'] == 'LORA', 'executor LoRA required')
        digest = file_hash(policy/'adapter_model.safetensors')
        require(digest == pin(approvals['executor_policy_sha256']), 'executor policy digest mismatch')
        for k in ('backbone','base_adapter'):
            require(file_hash(ex[k]) == spec['executor'][k+'_sha256'], f'wrong clean LongLive 1.3B {k}')
        require(Path(ex['wan']).is_dir() and Path(ex['upstream']).is_dir(), 'Wan assets or upstream directory missing')
        return digest
    for name,fn in [('fresh_origin',origin),('reviewed_teacher_manifest',teacher),
                    ('teacher_sft_planner',planner),('trainable_executor',executor)]:
        probe(name,fn)
    return dict(protocol=spec['protocol'], spec_sha256=digest(spec), launchable=False,
                input_status='blocked' if errors else 'static_inputs_checked', checked_sha256=evidence,
                input_blockers=errors, outstanding_gates=spec['acceptance_gates'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True,help='联合训练使用的配方及静态核验规则')
    p.add_argument('--candidate',type=Path,help='proposed effective config; never runs or resumes it')
    p.add_argument('--approvals',type=Path,help='independently reviewed teacher/SFT/clean executor input pins')
    args = p.parse_args(argv)
    try:
        result = audit(args.candidate, args.approvals, recipe_path=args.recipe)
    except (ValueError, OSError) as exc:
        p.exit(2,f'{p.prog}: {exc}\n')
    print(json.dumps(result,indent=2))
    return 2 if result['input_blockers'] else 0


if __name__=='__main__':
    sys.exit(main())
