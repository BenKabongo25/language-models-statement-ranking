"""Exact gradient cache over both towers.

The statement embeddings of a (micro)batch are computed without graph; the loss is back-propagated to the query
tower and to these embeddings; the statement forward is then replayed chunk by chunk with the upstream gradients.
This requires a deterministic statement forward (dropout disabled, see ``model/encoder.py``).
"""
from __future__ import annotations

import numpy as np
import torch

from .objective import multi_positive_infonce


def doc_forward(model, doc_rows, device, chunk_size, grad):
    """Embeddings of ``doc_rows`` in length-sorted chunks (less padding), returned in the original order."""
    if not doc_rows:
        raise ValueError('Empty document batch')
    order = sorted(range(len(doc_rows)), key=lambda i: len(doc_rows[i]))
    outputs = []
    with (torch.enable_grad if grad else torch.no_grad)():
        for j in range(0, len(order), chunk_size):
            outputs.append(model(**model.collate([doc_rows[i] for i in order[j:j + chunk_size]], device)))
    inverse = torch.tensor(np.argsort(order).copy(), device=device)
    return torch.cat(outputs, dim=0).index_select(0, inverse)


def replay_doc_gradients(model, rows, device, chunk_size, gradients):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    for j in range(0, len(order), chunk_size):
        indices = order[j:j + chunk_size]
        model(**model.collate([rows[i] for i in indices], device)).backward(gradients[indices])


def gradcache_step(query, document, query_rows, doc_rows, idx, pos_mask, neg_mask, temperature, scale, device,
                   doc_chunk, replay=True):
    """One (micro)batch: accumulates ``scale * dL/dtheta`` into the ``.grad`` of both towers and returns the
    unscaled loss. Rows only score their own positives and negatives (no in-batch negatives), so splitting a batch
    across ranks and summing the scaled gradients is exactly the full-batch gradient. A shared tower
    (``query is document``) accumulates the gradients of both roles; ``replay=False`` (frozen statement tower)
    skips the statement backward."""
    d = doc_forward(document, doc_rows, device, doc_chunk, grad=False)
    leaf = d.detach().requires_grad_(True)
    q = query(**query.collate(query_rows, device))
    scores = (q.unsqueeze(1) * leaf[idx.to(device)]).sum(dim=-1)
    loss = multi_positive_infonce(scores, pos_mask.to(device), neg_mask.to(device), float(temperature))
    (loss * scale).backward()
    if leaf.grad is None:
        raise RuntimeError('No document embedding gradient')
    if replay:
        replay_doc_gradients(document, doc_rows, device, doc_chunk, leaf.grad.detach())
    return float(loss.detach())
