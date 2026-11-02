"""Tokenizers of any backbone, with a guaranteed final special token for EOS pooling.

Embedding tokenizers end every input with a special token (Qwen3-Embedding: ``<|endoftext|>``, BERT-like: ``[SEP]``).
LLM tokenizers (Qwen3, Llama, Gemma, ...) do not: with ``model.append_eos: auto`` they are wrapped so that every
encoded input ends with an EOS token (``<|endoftext|>`` when the vocabulary has it, else ``eos_token``), after the
tokenizer's own special tokens (e.g. BOS).
"""
from __future__ import annotations

import logging

LOG = logging.getLogger('lmstar')


class EosAppendingTokenizer:
    """Tokenizer proxy appending ``eos_id`` (offset ``(0, 0)``) when ``add_special_tokens=True``; truncation keeps
    room for it. Everything else is delegated to the wrapped tokenizer."""

    def __init__(self, tok, eos_id):
        self._tok, self.appended_eos_id = tok, int(eos_id)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __len__(self):
        return len(self._tok)

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, **kw):
        if not add_special_tokens:
            return self._tok(text, add_special_tokens=False, truncation=truncation, max_length=max_length, **kw)
        limit = max_length - 1 if (truncation and max_length) else None
        enc = self._tok(text, add_special_tokens=True, truncation=limit is not None, max_length=limit, **kw)
        out = dict(enc)
        eos = self.appended_eos_id
        if isinstance(text, (list, tuple)):
            out['input_ids'] = [list(r) + [eos] for r in enc['input_ids']]
            if 'offset_mapping' in enc:
                out['offset_mapping'] = [list(r) + [(0, 0)] for r in enc['offset_mapping']]
            if 'attention_mask' in enc:
                out['attention_mask'] = [list(r) + [1] for r in enc['attention_mask']]
        else:
            out['input_ids'] = list(enc['input_ids']) + [eos]
            if 'offset_mapping' in enc:
                out['offset_mapping'] = list(enc['offset_mapping']) + [(0, 0)]
            if 'attention_mask' in enc:
                out['attention_mask'] = list(enc['attention_mask']) + [1]
        return out


def ends_with_special_token(tok):
    """True if ``add_special_tokens=True`` ends inputs with a special token (Qwen3-Embedding, BERT-like)."""
    with_special, without = (list(tok('a b', add_special_tokens=v)['input_ids']) for v in (True, False))
    special = {int(i) for i in (getattr(tok, 'all_special_ids', None) or [])}
    return (bool(with_special) and len(with_special) > len(without) and int(with_special[-1]) in special
            and (not without or with_special[-1] != without[-1]))


def eos_id_of(tok):
    eos = tok.convert_tokens_to_ids('<|endoftext|>')
    if eos is None or eos == getattr(tok, 'unk_token_id', None) or not isinstance(eos, int):
        eos = tok.eos_token_id if tok.eos_token_id is not None else getattr(tok, 'sep_token_id', None)
    if eos is None:
        raise RuntimeError('tokenizer has neither an EOS nor a SEP token: set model.pooling=mean')
    return int(eos)


def load_tokenizer(name, append_eos='auto', trust_remote_code=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
    if str(append_eos).lower() in ('false', 'none', '0', 'never') or ends_with_special_token(tok):
        return tok
    eos = eos_id_of(tok)
    LOG.info('Tokenizer %s does not end inputs with a special token: appending %r (id %d)',
             name, tok.convert_ids_to_tokens(eos), eos)
    return EosAppendingTokenizer(tok, eos)
