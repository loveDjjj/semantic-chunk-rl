"""Strict restoration identity, including the frozen reward protocol in new stages."""

from copy import deepcopy
import json
from pathlib import Path

from scrl.plan import digest


IDENTITY_KEYS = ('assets', 'plan', 'reward', 'training', 'adaptive_sampling', 'seed', 'dataset_sha256')


def training_fingerprint(cfg):
    values = {key: cfg[key] for key in IDENTITY_KEYS}
    from scrl.rl.objectives import CROSSED_CREDIT, WHOLE_CREDIT
    if isinstance(cfg['training'], dict) and cfg['training'].get('planner_credit') in (CROSSED_CREDIT, WHOLE_CREDIT):
        values.update(sampling=cfg['sampling'], fixed_sampling=cfg['fixed_sampling'])
    if 'reward_protocol' in cfg:
        values['reward_protocol'] = cfg['reward_protocol']
        values['asset_manifest_sha256'] = cfg['asset_manifest_sha256']
    if 'event_cache_backend' in cfg:
        values['event_cache_backend'] = cfg['event_cache_backend']
    return digest(values)


def quote_anchor_transition(parent, cfg):
    """Recognize only the reviewed scorer builds on either side of the quote-parser fix."""
    old = 'd1a70405ac7dfb9a54cc3dc107ea9d2dee5330b8527e6b121ae9d6175189494e'
    new = 'd0752b5baf9681299784a742a2330a0a4f9d812adaf3135996af950aa76916fa'
    normalized = deepcopy(cfg)
    if (parent['reward_protocol'].get('implementation_sha256') != old or
            normalized['reward_protocol'].get('implementation_sha256') != new):
        raise ValueError('Checkpoint training identity changed: unrecognized quote parser transition')
    normalized['reward_protocol']['implementation_sha256'] = old
    if training_fingerprint(normalized) != training_fingerprint(parent):
        raise ValueError('Additional training changes during quote parser migration')
    for run in (parent,cfg):
        p = run['reward_protocol']
        if run['stamp']['reward'] != p['version']+':'+digest([run['reward'],p,run['assets']['rewards']]):
            raise ValueError('Invalid reward stamp')
    return dict(kind='source-quote-anchor-v1', old_reward=parent['stamp']['reward'], new_reward=cfg['stamp']['reward'])


def rating_format_transition(parent, cfg, saved_fingerprint):
    """Allow precisely one opt-in prompt change; do not relax a normal resume."""
    from scrl.models.scoring.protocol import OBJECT_IDENTITY
    protocol=cfg['reward_protocol']
    if (parent['training']['mode']!='planner-only' or 'rating_format' in parent['reward_protocol'] or
            parent['reward_protocol'].get('inference_backend')!='vllm-http-six-v1' or
            protocol.get('rating_format')!=OBJECT_IDENTITY or training_fingerprint(parent)!=saved_fingerprint):
        raise ValueError('Not a supported judge-format transition')
    normalized=dict(cfg,reward_protocol={k:v for k,v in protocol.items() if k!='rating_format'})
    if training_fingerprint(normalized)!=saved_fingerprint:
        raise ValueError('Other training settings changed along with the judge prompt')
    for c in (parent,cfg):
        p=c['reward_protocol'];expected=p['version']+':'+digest([c['reward'],p,c['assets']['rewards']])
        if c['stamp']['reward']!=expected:raise ValueError('Unverified reward identity at transition')
    return dict(kind='rating-format-only',format=protocol['rating_format'],old_reward=parent['stamp']['reward'],
                new_reward=cfg['stamp']['reward'],parent_fingerprint=saved_fingerprint)


def verify_campaign_resume(cfg, saved_fingerprint, checkpoint):
    """Authenticate the original run before allowing a scorer interpreter relocation."""
    if training_fingerprint(cfg) == saved_fingerprint:
        return
    path = Path(checkpoint).resolve()
    if path.parent.name != 'checkpoints':
        raise ValueError('Resume training identities changed: no parent run')
    parent = json.loads((path.parent.parent/'config.json').read_text())
    if training_fingerprint(parent) != saved_fingerprint:
        raise ValueError('Parent configuration does not identify this checkpoint')
    candidate = deepcopy(cfg)
    for role in ('quality', 'judge'):
        old = parent['assets'].get('rewards', {}).get(role, {})
        new = candidate['assets'].get('rewards', {}).get(role, {})
        if isinstance(old.get('python'), str) and isinstance(new.get('python'), str) and new['python']:
            new['python'] = old['python']
    if training_fingerprint(candidate) != saved_fingerprint:
        raise ValueError('Resume training identities changed')


def validate_training_mode(mode):
    if mode not in ("joint", "planner-only"):
        raise ValueError(f"Unsupported training mode {mode!r}; expected 'joint' or 'planner-only'")
    return mode
