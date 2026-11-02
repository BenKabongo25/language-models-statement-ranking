from __future__ import annotations

from pathlib import Path
from typing import Any

from clustering_pipeline.storage import TYPES
from clustering_pipeline.utils import load_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    types = tuple(cfg.get("types", TYPES))
    unknown = sorted(set(types) - set(TYPES))
    if unknown:
        raise ValueError(f"Unsupported statement types: {unknown}")

    ncfg = cfg.get("neighbors", {})
    if int(ncfg.get("k", 128)) < 1:
        raise ValueError("neighbors.k must be >= 1")
    min_cos = float(ncfg.get("min_cosine", 0.0))
    if not -1.0 <= min_cos <= 1.0:
        raise ValueError("neighbors.min_cosine must be in [-1, 1]")

    rcfg = cfg.get("reranker", {})
    threshold = float(rcfg.get("threshold", 0.95))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("reranker.threshold must be in (0, 1]")
    instructions = rcfg.get("instructions") or {}
    for typ in TYPES:
        if not str(instructions.get(typ, "")).strip():
            raise ValueError(f"Missing reranker instruction for type={typ}")

    ccfg = cfg.get("cluster", {})
    if str(ccfg.get("algorithm", "sentinel_frontier_agglomerative")) != "sentinel_frontier_agglomerative":
        raise ValueError("cluster.algorithm must be 'sentinel_frontier_agglomerative'")
    q = int(ccfg.get("sentinels", 4))
    if q < 1:
        raise ValueError("cluster.sentinels must be >= 1")
    if q > 32:
        raise ValueError("cluster.sentinels > 32 is unsupported; use a small boundary set (default 4)")
    if int(ccfg.get("candidate_batch_size", 2048)) < 1:
        raise ValueError("cluster.candidate_batch_size must be >= 1")
    if int(ccfg.get("verification_batch_size", 512)) < 1:
        raise ValueError("cluster.verification_batch_size must be >= 1")
    if int(ccfg.get("cache_flush_pairs", 100000)) < 1:
        raise ValueError("cluster.cache_flush_pairs must be >= 1")
    if int(ccfg.get("checkpoint_every_waves", 10)) < 1:
        raise ValueError("cluster.checkpoint_every_waves must be >= 1")
    for key in ("max_merge_attempts", "max_new_reranker_pairs"):
        value = ccfg.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"cluster.{key} must be null or >= 1")
