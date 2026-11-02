"""Encoder towers on any Hugging Face backbone (encoder-only or decoder-only), with LoRA or full fine-tuning.

Pooling: ``eos`` = hidden state of the last non-padding token (the EOS ensured by ``tokenizer.py``), ``mean`` = mean
over the non-padding tokens, ``cls`` = first non-padding token. Embeddings are L2-normalised. LoRA
``target_modules: auto`` selects the attention projections of the architecture. Dropout is disabled in both towers:
the exact gradient cache replays the statement forward, which must be deterministic.

Towers (``model.towers``): ``independent`` (two towers, default), ``shared`` (one tower encodes queries and
statements; statements usually get an instruction, ``model.document_prefix``) or ``query_only`` (only the query
tower is trained; statements are encoded by the frozen backbone).

Checkpoints: ``best/{query,statement}_lora`` (adapters) or ``best/{query,statement}_model`` (complete weights);
``best/shared_{lora,model}`` for a shared tower; ``query_only`` saves the query tower only.
"""
from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.padding import pad_token_ids

LOG = logging.getLogger('lmstar')
POOLINGS = ('eos', 'mean', 'cls')
TOWERS = ('independent', 'shared', 'query_only')
AUTOCAST = {None: None, 'none': None, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
# attention projections per architecture family (the first fully present set wins)
TARGET_SETS = [('q_proj', 'k_proj', 'v_proj', 'o_proj'),       # Llama, Qwen, Gemma, Mistral, OLMo, SmolLM, ...
               ('qkv_proj', 'o_proj'),                         # Phi-3/4, GTE (NewModel)
               ('Wqkv', 'Wo'), ('Wqkv', 'out_proj'),           # ModernBERT, NomicBERT
               ('query', 'key', 'value'),                      # BERT, RoBERTa, XLM-R, MPNet, E5, BGE, BLaIR
               ('q_lin', 'k_lin', 'v_lin', 'out_lin'),         # DistilBERT
               ('query_key_value', 'dense'),                   # GPT-NeoX, Falcon
               ('c_attn', 'c_proj')]                           # GPT-2


# ------------------------------------------------------------------------------------------------- pooling
def last_token_pool(hidden, mask):
    """Hidden state of the last non-padding position (left or right padding)."""
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    idx = positions.masked_fill(mask == 0, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def first_token_pool(hidden, mask):
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    idx = positions.masked_fill(mask == 0, mask.shape[1]).min(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def pool(hidden, mask, pooling):
    if pooling == 'eos':
        return last_token_pool(hidden, mask)
    if pooling == 'cls':
        return first_token_pool(hidden, mask)
    if pooling == 'mean':
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    raise ValueError(f'pooling must be one of {POOLINGS}')


# ------------------------------------------------------------------------------------------- checkpoints
def tower_mode(cfg):
    """``model.towers`` of a configuration (independent | shared | query_only)."""
    mode = (cfg.get('model') or {}).get('towers') or 'independent'
    if mode not in TOWERS:
        raise ValueError(f'model.towers must be one of {TOWERS}')
    return mode


def tower_dir(role, use_lora):
    """Checkpoint sub-folder of a tower: ``query_lora`` / ``statement_lora`` / ``shared_lora`` or ``query_model``
    / ``statement_model`` / ``shared_model``."""
    name = role if role in ('query', 'shared') else 'statement'
    return f"{name}_{'lora' if use_lora else 'model'}"


def find_tower(folder, role):
    """Existing checkpoint sub-folder of a tower in ``<run>/best`` (adapter or complete weights; the shared tower
    serves both roles), or None."""
    for name in (role, 'shared'):
        for use_lora in (True, False):
            d = Path(folder) / tower_dir(name, use_lora)
            if d.is_dir():
                return d
    return None


def checkpoint_complete(folder, mode='independent'):
    """Whether ``<run>/best`` holds the towers of its mode (query_only: the query tower)."""
    return find_tower(folder, 'query') is not None and (mode == 'query_only' or find_tower(folder, 'document')
                                                         is not None)


# ----------------------------------------------------------------------------------------------- helpers
def disable_dropout(model):
    """Set every dropout probability to 0 (modules, float attributes, config). Returns the number of non-zero
    settings found."""
    n = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            n += module.p > 0
            module.p = 0.0
        for key, value in list(vars(module).items()):
            if 'dropout' in key and isinstance(value, float):
                n += value > 0
                setattr(module, key, 0.0)
    config = getattr(model, 'config', None)
    if config is not None:
        for key, value in list(vars(config).items()):
            if 'dropout' in key and isinstance(value, float):
                setattr(config, key, 0.0)
    return int(n)


def resolve_target_modules(model, targets='auto'):
    if targets not in (None, 'auto'):
        return list(targets)
    names = {n.rsplit('.', 1)[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    for candidate in TARGET_SETS:
        if set(candidate) <= names:
            return list(candidate)
    raise ValueError(f'cannot infer LoRA target modules (linear layers: {sorted(names)[:20]}...): '
                     'set model.lora.target_modules')


def _from_pretrained(name, dtype, attention, trust_remote_code):
    import transformers
    from transformers import AutoModel
    major, minor = (int(x) for x in transformers.__version__.split('.')[:2])
    kw = {('dtype' if (major, minor) >= (4, 56) else 'torch_dtype'): getattr(torch, dtype),
          'trust_remote_code': trust_remote_code}
    try:
        model = AutoModel.from_pretrained(name, attn_implementation=attention, **kw)
    except (ValueError, ImportError) as e:            # attention implementation not supported by the architecture
        if attention == 'eager':
            raise
        LOG.warning('%s: attn_implementation=%s unavailable (%s) -> eager', name, attention, str(e).splitlines()[0])
        model = AutoModel.from_pretrained(name, attn_implementation='eager', **kw)
    if getattr(model.config, 'is_encoder_decoder', False):
        raise ValueError(f'{name} is an encoder-decoder model: only encoder-only and decoder-only backbones are '
                         'supported')
    return model


# ------------------------------------------------------------------------------------------------- tower
class Encoder(nn.Module):
    """One tower: backbone (+ LoRA), padding, pooling, L2 normalisation."""

    def __init__(self, name, use_lora=True, lora_cfg=None, gradient_checkpointing=False, attention='sdpa',
                 dtype='bfloat16', pooling='eos', padding_side='right', pad_id=0, trust_remote_code=False, model=None,
                 autocast=None):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f'model.pooling must be one of {POOLINGS}')
        if autocast not in AUTOCAST:
            raise ValueError(f'model.autocast must be one of {sorted(AUTOCAST, key=str)}')
        self.lora_cfg = dict(lora_cfg or {})
        self.use_lora, self.pooling, self.padding_side, self.pad_id = use_lora, pooling, padding_side, int(pad_id)
        self.autocast = AUTOCAST[autocast]
        if model is not None:                          # restored tower (from_adapter / from_full)
            self.model = model
            disable_dropout(self.model)
            return
        self.model = _from_pretrained(name, dtype, attention, trust_remote_code)
        n = disable_dropout(self.model)
        if n:
            LOG.info('%s: %d dropout setting(s) set to 0 (exact gradient cache)', name, n)
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lc = self.lora_cfg
            lc['target_modules'] = resolve_target_modules(self.model, lc.get('target_modules', 'auto'))
            self.model = get_peft_model(self.model, LoraConfig(
                r=int(lc.get('r', 16)), lora_alpha=int(lc.get('alpha', 32)), lora_dropout=float(lc.get('dropout', 0.0)),
                target_modules=lc['target_modules'], bias='none', task_type='FEATURE_EXTRACTION'))
        if gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            except (TypeError, ValueError):
                self.model.gradient_checkpointing_enable()
            if use_lora and hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()
        if use_lora:
            self.model.print_trainable_parameters()
        else:
            LOG.info('%s: full fine-tuning (%s trainable parameters)', name,
                     f'{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

    @classmethod
    def from_adapter(cls, name, adapter_dir, attention='sdpa', dtype='bfloat16', pooling='eos', padding_side='right',
                     pad_id=0, trust_remote_code=False, lora_cfg=None, autocast=None):
        """Frozen tower: base backbone + saved LoRA adapter."""
        from peft import PeftModel
        base = _from_pretrained(name, dtype, attention, trust_remote_code)
        model = PeftModel.from_pretrained(base, Path(adapter_dir), is_trainable=False)
        return cls(name, True, lora_cfg, pooling=pooling, padding_side=padding_side, pad_id=pad_id, model=model,
                   autocast=autocast)

    @classmethod
    def from_full(cls, model_dir, attention='sdpa', dtype='bfloat16', pooling='eos', padding_side='right', pad_id=0,
                  trust_remote_code=False, autocast=None):
        """Frozen fully fine-tuned tower."""
        model = _from_pretrained(str(model_dir), dtype, attention, trust_remote_code)
        model.requires_grad_(False)
        return cls(str(model_dir), False, None, pooling=pooling, padding_side=padding_side, pad_id=pad_id, model=model,
                   autocast=autocast)

    @property
    def hidden_size(self):
        return self.model.get_input_embeddings().weight.shape[1]

    def collate(self, rows, device=None):
        batch = pad_token_ids(rows, self.pad_id, self.padding_side)
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()} if device is not None else batch

    def _forward_kwargs(self):
        """``use_cache=False`` for the backbones that accept it (decoders); encoders such as ModernBERT do not."""
        if not hasattr(self, '_fkw'):
            base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
            params = inspect.signature(base.forward).parameters
            var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
            accepts = 'use_cache' in params or (var_kw and getattr(base.config, 'use_cache', None) is not None)
            self._fkw = {'use_cache': False} if accepts else {}
        return self._fkw

    def forward(self, input_ids, attention_mask):
        with torch.autocast(input_ids.device.type, dtype=self.autocast, enabled=self.autocast is not None):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                             **self._forward_kwargs())
        hidden = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        return F.normalize(pool(hidden, attention_mask, self.pooling).float(), dim=-1)

    # ------------------------------------------------------------------------------------- checkpoint I/O
    def save(self, folder):
        """Adapter (LoRA) or complete weights (full fine-tuning), safetensors."""
        self.model.save_pretrained(folder, safe_serialization=True)

    def load_checkpoint(self, folder):
        """Initialise from a saved tower of the same kind (adapter for LoRA, complete weights otherwise)."""
        return self.load_adapter(folder) if self.use_lora else self.load_full(folder)

    def load_adapter(self, folder):
        """Load the LoRA weights of a saved adapter in place."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        folder = Path(folder)
        saved = json.loads((folder / 'adapter_config.json').read_text())
        want = {'r': int(self.lora_cfg.get('r', 16)), 'lora_alpha': int(self.lora_cfg.get('alpha', 32)),
                'target_modules': sorted(self.lora_cfg['target_modules'])}
        have = {'r': int(saved['r']), 'lora_alpha': int(saved['lora_alpha']),
                'target_modules': sorted(saved['target_modules'])}
        if want != have:
            raise ValueError(f'LoRA config mismatch with checkpoint {folder}: {have} != {want}')
        result = set_peft_model_state_dict(self.model, load_file(str(folder / 'adapter_model.safetensors')))
        bad = [k for k in list(getattr(result, 'unexpected_keys', [])) + list(getattr(result, 'missing_keys', []))
               if 'lora_' in k]
        if bad:
            raise RuntimeError(f'Unexpected/missing LoRA keys when loading {folder}: {bad[:5]}')
        return self

    def load_full(self, folder):
        """Load the complete weights of a fully fine-tuned tower in place."""
        from safetensors.torch import load_file
        folder = Path(folder)
        files = sorted(folder.glob('*.safetensors'))
        if not files:
            raise FileNotFoundError(f'no safetensors weights in {folder}')
        state = {}
        for f in files:
            state.update(load_file(str(f)))
        result = self.model.load_state_dict(state, strict=False)
        tied = set(getattr(self.model, '_tied_weights_keys', None) or [])
        missing = [k for k in result.missing_keys if k not in tied]
        if missing or result.unexpected_keys:
            raise RuntimeError(f'Weights of {folder} do not match {type(self.model).__name__}: '
                               f'missing={missing[:5]} unexpected={result.unexpected_keys[:5]}')
        return self


# ------------------------------------------------------------------------------------------ construction
def max_positions(model):
    """Usable input length of a backbone (None if unknown): ``max_position_embeddings``, minus the padding offset of
    RoBERTa-style position ids."""
    config = getattr(model, 'config', None)
    n = getattr(config, 'max_position_embeddings', None)
    if not n:
        return None
    return int(n) - 2 if getattr(config, 'model_type', '') in ('roberta', 'xlm-roberta', 'camembert') else int(n)


def check_lengths(cfg, encoder):
    limit = max_positions(encoder.model)
    for key in ('query_max_length', 'document_max_length'):
        if limit and int(cfg['model'][key]) > limit:
            raise ValueError(f"model.{key}={cfg['model'][key]} exceeds the {limit} positions of {cfg['model']['name']}")


def build_encoder(cfg, tok, role, use_lora=None, trainable=True):
    """Tower from the ``model`` section; ``role`` = query | document (gradient-checkpointing setting).
    ``use_lora`` defaults to ``model.use_lora``; ``trainable=False`` gives the frozen backbone (zero-shot)."""
    mc = cfg['model']
    use_lora = bool(mc.get('use_lora', True)) if use_lora is None else use_lora
    key = 'query_gradient_checkpointing' if role == 'query' else 'document_gradient_checkpointing'
    checkpointing = mc.get(key, role == 'query')
    enc = Encoder(mc['name'], use_lora and trainable, mc.get('lora'), checkpointing and trainable,
                  mc.get('attention', 'sdpa'), mc.get('dtype', 'bfloat16'), mc.get('pooling', 'eos'),
                  mc.get('padding_side', 'right'), tok.pad_token_id, mc.get('trust_remote_code', False),
                  autocast=mc.get('autocast'))
    if not trainable:
        enc.model.requires_grad_(False)
    return enc


def restore_encoder(cfg, tok, tower):
    """Frozen tower from ``<run>/best/<tower>``: LoRA adapter (``adapter_config.json``) or complete weights."""
    mc = cfg['model']
    tower = Path(tower)
    common = dict(attention=mc.get('attention', 'sdpa'), dtype=mc.get('dtype', 'bfloat16'),
                  pooling=mc.get('pooling', 'eos'), padding_side=mc.get('padding_side', 'right'),
                  pad_id=tok.pad_token_id, trust_remote_code=mc.get('trust_remote_code', False),
                  autocast=mc.get('autocast'))
    if not (tower / 'adapter_config.json').exists() and (tower / 'config.json').exists():
        return Encoder.from_full(tower, **common)
    return Encoder.from_adapter(mc['name'], tower, lora_cfg=mc.get('lora'), **common)
