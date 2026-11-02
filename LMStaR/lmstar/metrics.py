"""Ranking metrics of one ranked candidate list against the set of ground-truth statements."""
from __future__ import annotations

import math


def metrics_at_k(ranked, positives, k):
    """Precision, recall, F1, hit rate and nDCG at ``k`` (fractions in [0, 1])."""
    top = ranked[:k]
    hits = sum(x in positives for x in top)
    precision = hits / k
    recall = hits / len(positives) if positives else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    dcg = sum(1.0 / math.log2(j + 2) for j, x in enumerate(top) if x in positives)
    idcg = sum(1.0 / math.log2(j + 2) for j in range(min(k, len(positives))))
    return {f'precision@{k}': precision, f'recall@{k}': recall, f'f1@{k}': f1, f'hr@{k}': float(hits > 0),
            f'ndcg@{k}': dcg / idcg if idcg else 0.}
