from __future__ import annotations

from pathlib import Path
from typing import Any

from clustering_pipeline.utils import quick_file_fingerprint, stable_hash

# Upstream schema stays compatible with v4-v10 so completed
# prepare/embed/neighbors/rerank artifacts can be adopted without recomputation.
SCHEMA_VERSION = 2
CLUSTER_IMPL_VERSION = "sentinel-frontier-agglomerative-v2"
AUDIT_IMPL_VERSION = "pair-audit-with-sentinel-frontier-v2"


def prepare_artifact_id(input_file: Path, cfg: dict[str, Any]) -> str:
    return stable_hash(
        {
            "schema": SCHEMA_VERSION,
            "stage": "prepare",
            "input": quick_file_fingerprint(input_file),
            "prepare": cfg.get("prepare", {}),
            "type_aliases": cfg.get("type_aliases", {}),
        }
    )


def embedding_semantics(cfg: dict[str, Any]) -> dict[str, Any]:
    ecfg = cfg["embedding"]
    return {
        "model": ecfg["model"],
        "revision": ecfg.get("revision"),
        "trust_remote_code": bool(ecfg.get("trust_remote_code", False)),
        "pooling": str(ecfg.get("pooling", "last")),
        "max_length": int(ecfg.get("max_length", 256)),
        "normalize": bool(ecfg.get("normalize", True)),
        "truncate_dim": int(ecfg.get("truncate_dim", -1)),
        "output_dtype": str(ecfg.get("output_dtype", "float16")),
    }


def embedding_artifact_id(prepare_id: str, cfg: dict[str, Any], statement_type: str) -> str:
    return stable_hash(
        {
            "schema": SCHEMA_VERSION,
            "stage": "embed",
            "type": statement_type,
            "source": prepare_id,
            "semantics": embedding_semantics(cfg),
        }
    )


def neighbors_semantics(cfg: dict[str, Any]) -> dict[str, Any]:
    ncfg = cfg["neighbors"]
    return {
        "k": int(ncfg.get("k", 128)),
        "min_cosine": float(ncfg.get("min_cosine", 0.80)),
    }


def neighbors_artifact_id(embed_id: str, cfg: dict[str, Any], statement_type: str) -> str:
    return stable_hash(
        {
            "schema": SCHEMA_VERSION,
            "stage": "neighbors",
            "type": statement_type,
            "source": embed_id,
            "semantics": neighbors_semantics(cfg),
        }
    )


def rerank_semantics(cfg: dict[str, Any], statement_type: str) -> dict[str, Any]:
    rcfg = cfg["reranker"]
    enabled = bool(rcfg.get("enabled", True))
    return {
        "enabled": enabled,
        "model": rcfg.get("model") if enabled else None,
        "revision": rcfg.get("revision") if enabled else None,
        "trust_remote_code": bool(rcfg.get("trust_remote_code", False)) if enabled else False,
        "max_length": int(rcfg.get("max_length", 512)) if enabled else None,
        "instruction": str((rcfg.get("instructions") or {}).get(statement_type, "")) if enabled else None,
    }


def rerank_artifact_id(neighbors_id: str, cfg: dict[str, Any], statement_type: str) -> str:
    # Deliberately excludes threshold and batch size: reranking stores raw scores.
    return stable_hash(
        {
            "schema": SCHEMA_VERSION,
            "stage": "rerank",
            "type": statement_type,
            "source": neighbors_id,
            "semantics": rerank_semantics(cfg, statement_type),
        }
    )


def cluster_semantics(cfg: dict[str, Any], statement_type: str) -> dict[str, Any]:
    ccfg = cfg["cluster"]
    return {
        "implementation": CLUSTER_IMPL_VERSION,
        "algorithm": "sentinel_frontier_agglomerative",
        "threshold": float(cfg["reranker"].get("threshold", 0.95)),
        "sentinels": int(ccfg.get("sentinels", 4)),
        "selection": "embedding_medoid_then_farthest_first",
        "merge_rule": "all_cross_sentinel_reranker_scores_gte_threshold",
        "proposal_order": "descending_initial_reranker_score_with_priority_retries",
        "stopping_rule": "seed_stream_exhausted_and_retry_frontier_empty",
        "all_observed_rejections_are_hard_constraints": True,
        "resume_previous_state": bool(ccfg.get("resume_previous_state", True)),
        "include_members": bool(ccfg.get("include_members", True)),
        "include_cluster_stats": bool(ccfg.get("include_cluster_stats", True)),
    }


def cluster_artifact_id(rerank_id: str, cfg: dict[str, Any], statement_type: str) -> str:
    # candidate_batch_size, verification_batch_size and cache flushing are
    # execution-only and intentionally excluded from the semantic identity.
    return stable_hash(
        {
            "schema": SCHEMA_VERSION,
            "stage": "cluster",
            "type": statement_type,
            "source": rerank_id,
            "semantics": cluster_semantics(cfg, statement_type),
        }
    )


def finalize_artifact_id(cluster_ids: dict[str, str], cfg: dict[str, Any]) -> str:
    return stable_hash(
        {
            "schema": SCHEMA_VERSION,
            "stage": "finalize",
            "sources": cluster_ids,
            "semantics": cfg.get("finalize", {}),
        }
    )
