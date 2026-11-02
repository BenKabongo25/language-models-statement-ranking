"""Configuration: built-in defaults (mirrored by ``configs/base.yaml``), composition and ``--set`` overrides.

A run configuration is ``DEFAULTS <- base.yaml <- models/<m>.yaml <- datasets/<d>.yaml [<- hardware/..]
[<- variants/..] [<- init/..] <- --set section.key=value``. Configurations saved by earlier versions stay loadable:
settings they did not have take the values they used (:func:`legacy_defaults`), and runs trained with extensions
that no longer exist are reported by :func:`removed_features`.
"""
from __future__ import annotations

import copy

import yaml

INSTRUCTION = ("Retrieve explanation statements matching the target item to the user's preferences based on "
               "previous interactions.")

DEFAULTS = {
    'seed': 40,
    'dataset': {
        'name': None,                             # Beauty | Toys | Sports | Yelp
        'root': '../data/processed',
        'final_subdir': 'statement_ranking/final',
    },
    'model': {
        'name': 'Qwen/Qwen3-Embedding-0.6B',      # any Hugging Face encoder-only or decoder-only backbone
        'pooling': 'eos',                         # eos (last non-padding token) | mean | cls (first token)
        'append_eos': 'auto',                     # auto: append EOS when the tokenizer adds no trailing special token
        'padding_side': 'right',                  # right (exact for causal and bidirectional models) | left
        'query_template': 'Instruct: {instruction}\nQuery: ',
        'document_prefix': '',                    # e.g. "passage: " (E5), "search_document: " (Nomic)
        'instruction': INSTRUCTION,
        'query_max_length': 4096,                 # history + target-item prompt, packed token-exactly
        'document_max_length': 256,
        'use_lora': True,                         # false: full fine-tuning of the trained tower(s)
        'towers': 'independent',                  # independent | shared (one tower) | query_only (frozen statements)
        'lora': {'r': 16, 'alpha': 32, 'dropout': 0.0, 'target_modules': 'auto'},
        'attention': 'sdpa',                      # sdpa | eager | flash_attention_2
        'dtype': 'bfloat16',                      # weights (float32 for full fine-tuning)
        'autocast': None,                         # null | bfloat16: mixed-precision forward (float32 weights)
        'trust_remote_code': False,
        'query_gradient_checkpointing': True,
        'document_gradient_checkpointing': False,
    },
    'history': {
        'max_statements_per_interaction': None,   # null = all statements of a past interaction
        'order': 'recency',                       # recency | chronological | random | random_selected
        'packing': 'stop',                        # stop | skip
        'views': 1,                               # training views (random orders re-drawn per view)
        'seed': None,                             # seed of the random orders (null: run seed)
        'target_position': 'last',                # last | first (target item before the history)
    },
    'training': {
        'epochs': 10,
        'batch_size': 128,                        # GLOBAL physical batch (all GPUs)
        'gradient_accumulation_steps': 2,
        'bucket_size': 1024,                      # length bucketing of the shuffled training examples
        'document_microbatch_size': 256,          # per GPU: statement forward / gradient replay chunk
        'document_eval_batch_size': 512,          # per GPU
        'eval_batch_size': 128,                   # per GPU
        'num_negatives': 32,
        'temperature': 0.1,
        'learning_rate_query': 1.0e-4,
        'learning_rate_document': 1.0e-4,
        'weight_decay': 0.01,
        'warmup_ratio': 0.05,
        'max_grad_norm': 1.0,
        'eval_steps': 50,
        'log_steps': 10,
        'early_stopping_patience': 5,
        'loss': 'infonce',
        'negative_sampling': {'strategy': 'uniform',          # uniform | uniform_userpop
                              'userpop_ratio': 0.25, 'hard_pool_multiplier': 3},
        'init': 'scratch',                        # scratch | checkpoint
        'init_checkpoint': None,                  # <run>/best (or <run>) when init = checkpoint
    },
    'evaluation': {'cutoffs': [5, 10], 'metric_scale': 100.0, 'selection_metric': 'ndcg@5',
                   'test_context': 'fit_plus_val'},
    'output': {'root': '../data/processed/LMStaR'},
    'runtime': {'gpus': 1, 'dist_timeout_minutes': 240},
}


def deep_update(base, update):
    """Recursively merge ``update`` into ``base`` (in place) and return ``base``."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve(*configs):
    """DEFAULTS updated by every given (partial) configuration, left to right."""
    cfg = copy.deepcopy(DEFAULTS)
    for c in configs:
        if c:
            deep_update(cfg, copy.deepcopy(c))
    return cfg


def load_yaml(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load(paths, sets=(), error=None):
    """Compose the YAML files ``paths`` over the defaults, then apply ``--set`` overrides."""
    return apply_sets(resolve(*[load_yaml(p) for p in paths]), sets, error)


def apply_sets(cfg, settings, error=None):
    """Apply ``SECTION.KEY=VALUE`` overrides (values parsed as YAML). Keys must exist: typos are errors."""
    def fail(message):
        if error:
            error(message)
        raise ValueError(message)

    for setting in settings or []:
        if '=' not in setting:
            fail('Expected SECTION.KEY=VALUE')
        key, value = setting.split('=', 1)
        path = key.split('.')
        node = cfg
        for part in path[:-1]:
            if part not in node or not isinstance(node[part], dict):
                fail(f'Unknown section: {key}')
            node = node[part]
        if path[-1] not in node:
            fail(f'Unknown key: {key}')
        node[path[-1]] = yaml.safe_load(value)
    return cfg


def public(cfg):
    """The configuration without the runtime-only entries (keys starting with ``_``)."""
    return {k: v for k, v in cfg.items() if not k.startswith('_')}


def legacy_defaults(raw):
    """Settings missing from a configuration saved by an earlier version take the value that version used
    (left padding). Applied to the raw saved configuration before :func:`resolve`."""
    raw = copy.deepcopy(raw or {})
    model = raw.setdefault('model', {})
    if 'padding_side' not in model:
        model['padding_side'] = 'left'
    return raw


def removed_features(cfg):
    """Extensions of earlier versions enabled in ``cfg`` that LMStaR no longer implements."""
    out = []
    if ((cfg.get('history') or {}).get('item_prior') or {}).get('enabled'):
        out.append('history.item_prior (target-item frequent statements, v1.3)')
    if (cfg.get('rating') or {}).get('enabled'):
        out.append('rating (<rate> supervision, v1.4)')
    if (cfg.get('aspect') or {}).get('enabled'):
        out.append('aspect (LMStaR-Aspect, v2.0-2.1)')
    if (cfg.get('li') or {}).get('enabled') or (cfg.get('scoring') or {}).get('model', 'dense') != 'dense':
        out.append('late interaction (LMStaR-LI, v3.0-3.1)')
    return out
