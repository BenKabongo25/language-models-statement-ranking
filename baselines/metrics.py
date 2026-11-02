from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


def metrics_at_k(ranked: list[int] | np.ndarray, positives: set[int], k: int) -> dict[str, float]:
    if not positives:
        return {f"precision@{k}": 0., f"recall@{k}": 0., f"f1@{k}": 0., f"ndcg@{k}": 0., f"hr@{k}": 0.}
    top = list(ranked[:k])
    hits = [1 if x in positives else 0 for x in top]
    h = sum(hits)
    p = h / k
    r = h / len(positives)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(hits))
    ideal_n = min(k, len(positives))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_n))
    return {
        f"precision@{k}": p, f"recall@{k}": r, f"f1@{k}": f,
        f"ndcg@{k}": 0.0 if idcg == 0 else dcg / idcg,
        f"hr@{k}": 1.0 if h > 0 else 0.0,
    }


def evaluate_rows(rows: Iterable[tuple[np.ndarray, set[int]]], ks=(5, 10)) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    n = 0
    for ranked, positives in rows:
        n += 1
        for k in ks:
            for key, value in metrics_at_k(ranked, positives, int(k)).items():
                sums[key] += value
    if n == 0:
        return {name: float("nan") for k in ks for name in [f"precision@{k}", f"recall@{k}", f"f1@{k}", f"ndcg@{k}", f"hr@{k}"]}
    return {k: v / n for k, v in sums.items()}
