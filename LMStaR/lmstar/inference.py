"""Exhaustive ranking evaluation (optionally multi-GPU) and checkpoint I/O.

Every candidate statement s of the target item is scored by <q_x, d_s>; ties are broken by statement id. Metrics
are identical for any number of GPUs.
"""
from __future__ import annotations

import csv
import heapq
import json
import logging
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import public
from .distributed import DistEnv, all_gather_object, barrier, gather_rows
from .metrics import metrics_at_k
from .model.encoder import find_tower, restore_encoder, tower_dir, tower_mode

LOG = logging.getLogger('lmstar')
PRED_HEADER = ['interaction_id', 'user_id', 'item_id', 'candidate_statement_id', 'score', 'label', 'rank']


@torch.no_grad()
def eval_documents(model, data, examples, token_rows, index, cfg, env):
    """Encode every distinct candidate once (sharded over the ranks, then all-gathered).

    Returns (embeddings [n, d], row_of [len(token_rows)]) with ``row_of[statement index]`` = row or -1.
    """
    needed = set()
    for ex in examples:
        needed.update(index[s] for s in data.candidates(ex.item_id) if s in index)
    needed = sorted(needed)
    if not needed:
        raise RuntimeError('No candidate statements in evaluation split')
    bs = int(cfg['training']['document_eval_batch_size'])
    order = sorted(needed, key=lambda i: len(token_rows[i]))
    chunks = [order[j:j + bs] for j in range(0, len(order), bs)]
    LOG.info('Document evaluation: encoding %d distinct candidates in batches of %d on %d rank(s)',
             len(needed), bs, env.world)
    start = time.time()
    embs, idxs = [], []
    for positions in tqdm(chunks[env.rank::env.world], desc='Encode evaluation statements', leave=False,
                          disable=not env.main):
        embs.append(model(**model.collate([token_rows[i] for i in positions], env.device)).float())
        idxs.extend(positions)
    dims = [d for d in all_gather_object(embs[0].shape[1] if embs else None, env) if d is not None]
    local = torch.cat(embs) if embs else torch.zeros((0, dims[0]), device=env.device)
    all_idx, all_emb = gather_rows(torch.tensor(idxs, dtype=torch.long, device=env.device), local, env)
    row_of = torch.full((len(token_rows),), -1, dtype=torch.long, device=env.device)
    row_of[all_idx.to(env.device)] = torch.arange(len(all_idx), device=env.device)
    LOG.info('Document evaluation encoding complete | %.1fs | %d vectors', time.time() - start, len(all_idx))
    return all_emb.to(env.device), row_of


@torch.no_grad()
def evaluate(query, document, data, examples, contexts, token_rows, index, cfg, env, split, predictions=None):
    """Exhaustive ranking of all the candidates of each target item; same metrics for any number of ranks.
    With ``predictions`` (a CSV path), every ranked candidate is written (score, label, rank)."""
    if not isinstance(env, DistEnv):
        env = DistEnv(0, 1, torch.device(env))
    query.eval()
    document.eval()
    device = env.device
    doc_emb, row_of = eval_documents(document, data, examples, token_rows, index, cfg, env)
    cutoffs = tuple(map(int, cfg['evaluation']['cutoffs']))
    mine = list(range(env.rank, len(examples), env.world))
    if predictions is None:                        # less padding; predictions keep the example order (merge)
        mine.sort(key=lambda p: len(contexts[examples[p].interaction_id]))
    part = Path(f'{predictions}.part{env.rank}') if predictions else None
    fp = open(part, 'w', newline='', encoding='utf-8') if part else None
    writer = csv.writer(fp) if fp else None
    results, missing, empty = [], 0, 0
    lowest, highest, start = float('inf'), -float('inf'), time.time()
    try:
        bs = int(cfg['training']['eval_batch_size'])
        for s in tqdm(range(0, len(mine), bs), desc=f'{split} exhaustive', leave=False, disable=not env.main):
            chunk = mine[s:s + bs]
            q = query(**query.collate([contexts[examples[p].interaction_id] for p in chunk], device))
            for j, p in enumerate(chunk):
                ex = examples[p]
                ids = list(dict.fromkeys(sid for sid in data.candidates(ex.item_id) if sid in index))
                if not ids:
                    empty += 1
                    continue
                positives = set(ex.positives)
                missing += len(positives - set(ids))
                rows = row_of[torch.tensor([index[sid] for sid in ids], device=device)]
                if bool((rows < 0).any()):
                    raise RuntimeError('Candidate embedding missing')
                scores = (doc_emb[rows] @ q[j]).float().cpu().tolist()
                lowest, highest = min(lowest, min(scores)), max(highest, max(scores))
                ranked = sorted(zip(ids, scores), key=lambda r: (-r[1], r[0]))
                metrics = {}
                for k in cutoffs:
                    metrics.update(metrics_at_k([sid for sid, _ in ranked], positives, k))
                results.append((p, metrics))
                if writer:
                    for rank, (sid, score) in enumerate(ranked, 1):
                        writer.writerow([p, ex.interaction_id, ex.user_id, ex.item_id, sid, score,
                                         int(sid in positives), rank])
    finally:
        if fp:
            fp.close()
    gathered = all_gather_object((results, empty, missing, lowest, highest), env)
    results = sorted((r for g in gathered for r in g[0]), key=lambda r: r[0])
    empty, missing = sum(g[1] for g in gathered), sum(g[2] for g in gathered)
    lowest, highest = min(g[3] for g in gathered), max(g[4] for g in gathered)
    if predictions:
        barrier(env)
        if env.main:
            _merge_predictions(predictions, env.world)
        barrier(env)
    rows = [m for _, m in results]
    scale = float(cfg['evaluation'].get('metric_scale', 100.0))
    metrics = {k: round(sum(r[k] for r in rows) / len(rows) * scale, 6) for k in rows[0]} if rows else {}
    LOG.info('%s | evaluated=%d/%d | empty_candidates=%d | missing_positives=%d | ranking_time=%.1fs | '
             'scores=[%.4f, %.4f] | %s', split, len(rows), len(examples), empty, missing, time.time() - start,
             lowest, highest, ' '.join(f'{k}={v:.3f}' for k, v in metrics.items()))
    if len(rows) != len(examples):
        raise RuntimeError(f'{split}: {len(examples) - len(rows)} examples have no item candidates; '
                           'cannot preserve baseline denominator')
    return metrics


def _merge_predictions(path, world):
    parts = [Path(f'{path}.part{r}') for r in range(world)]
    files = [open(p, newline='', encoding='utf-8') for p in parts]
    try:
        with open(path, 'w', newline='', encoding='utf-8') as out:
            w = csv.writer(out)
            w.writerow(PRED_HEADER)
            for row in heapq.merge(*(csv.reader(f) for f in files), key=lambda r: int(r[0])):
                w.writerow(row[1:])
    finally:
        for f in files:
            f.close()
    for p in parts:
        p.unlink()


def save_best(query, document, tok, folder, cfg, epoch, step, metrics):
    """``<run>/best``: the trained tower(s) (both; the shared one; the query one for ``query_only``), the tokenizer
    and ``metadata.json`` (epoch, step, validation metrics, config)."""
    folder.mkdir(parents=True, exist_ok=True)
    mode = tower_mode(cfg)
    query.save(folder / tower_dir('shared' if mode == 'shared' else 'query', query.use_lora))
    if mode == 'independent':
        document.save(folder / tower_dir('document', document.use_lora))
    tok.save_pretrained(folder / 'tokenizer')
    meta = {'epoch': epoch, 'step': step, 'val': metrics, 'model': cfg['model']['name'], 'config': public(cfg)}
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    LOG.info('BEST checkpoint saved | step=%d | %s', step, folder)


def restore_best(folder, cfg, device):
    """Both towers of ``<run>/best`` (LoRA adapters, LMStarDual v1.2+, or complete weights), frozen, in eval mode:
    one object for a shared tower, the frozen backbone as statement tower for ``query_only``.
    ``cfg['_tokenizer']`` must hold the tokenizer (padding id)."""
    from .model import encoder as ENC
    tok, mode = cfg['_tokenizer'], tower_mode(cfg)
    q_dir = find_tower(folder, 'query')
    d_dir = None if mode == 'query_only' else find_tower(folder, 'document')
    if q_dir is None or (mode != 'query_only' and d_dir is None):
        raise FileNotFoundError(f'{folder}: query / statement towers not found')
    query = restore_encoder(cfg, tok, q_dir).to(device).eval()
    if mode == 'query_only':
        document = ENC.build_encoder(cfg, tok, 'document', trainable=False).to(device).eval()
    elif d_dir == q_dir:                               # shared tower
        document = query
    else:
        document = restore_encoder(cfg, tok, d_dir).to(device).eval()
    return query, document
