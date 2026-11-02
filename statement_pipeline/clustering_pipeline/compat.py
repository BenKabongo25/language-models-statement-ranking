from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from clustering_pipeline.storage import (
    TYPES,
    STAGE_EMBED,
    STAGE_NEIGHBORS,
    STAGE_RERANK,
    embeddings_path,
    embedding_ids_path,
    neighbors_dir,
    rerank_dir,
    statements_path,
    interactions_raw_path,
    evidence_raw_path,
)
from clustering_pipeline.utils import load_json, outputs_exist, read_jsonl


def legacy_prepare_usable(output_root: Path, input_file: Path) -> bool:
    manifest = load_json(output_root / "00_prepare" / "manifest.json")
    outputs = [
        *(statements_path(output_root, t) for t in TYPES),
        interactions_raw_path(output_root),
        evidence_raw_path(output_root),
    ]
    if not manifest.get("complete") or not outputs_exist(outputs):
        return False
    old_input = manifest.get("input_file")
    return old_input is None or Path(str(old_input)).resolve() == input_file.resolve()


def legacy_embed_usable(output_root: Path, cfg: dict[str, Any], statement_type: str) -> bool:
    # Only adopt an old embedding when the prepare stage itself was adopted from
    # the same legacy run. If prepare was recomputed, texts may have changed.
    prepare_manifest = load_json(output_root / "00_prepare" / "manifest.json")
    if not prepare_manifest.get("legacy_adopted"):
        return False
    manifest = load_json(output_root / STAGE_EMBED / f"{statement_type}.manifest.json")
    out = embeddings_path(output_root, statement_type)
    ids = embedding_ids_path(output_root, statement_type)
    if not manifest.get("complete") or not outputs_exist([out, ids]):
        return False
    if manifest.get("artifact_id"):
        return False
    if str(manifest.get("model")) != str(cfg["embedding"]["model"]):
        return False
    try:
        arr = np.load(out, mmap_mode="r")
        id_arr = np.load(ids, mmap_mode="r")
        n_statements = sum(1 for _ in read_jsonl(statements_path(output_root, statement_type)))
        return arr.ndim == 2 and len(arr) == len(id_arr) == n_statements
    except Exception:
        return False


def legacy_neighbors_usable(output_root: Path, cfg: dict[str, Any], statement_type: str) -> bool:
    # Legacy neighbors are only safe to adopt when the embedding they were built
    # from was itself adopted, not recomputed with a new model/semantic config.
    embed_manifest = load_json(output_root / STAGE_EMBED / f"{statement_type}.manifest.json")
    if not embed_manifest.get("legacy_adopted"):
        return False
    manifest = load_json(output_root / STAGE_NEIGHBORS / f"{statement_type}.manifest.json")
    if not manifest.get("complete") or manifest.get("artifact_id"):
        return False
    if int(manifest.get("k", -1)) != int(cfg["neighbors"].get("k", 128)):
        return False
    try:
        if not math.isclose(
            float(manifest.get("min_cosine")),
            float(cfg["neighbors"].get("min_cosine", 0.60)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    except Exception:
        return False
    out = neighbors_dir(output_root, statement_type)
    shards = sorted(out.glob("part-*.npz")) if out.exists() else []
    return bool(shards) or int(manifest.get("pairs", 0)) == 0


def legacy_rerank_usable(output_root: Path, cfg: dict[str, Any], statement_type: str) -> bool:
    manifest = load_json(output_root / STAGE_RERANK / f"{statement_type}.manifest.json")
    if not manifest.get("complete") or manifest.get("artifact_id"):
        return False
    enabled = bool(cfg["reranker"].get("enabled", True))
    expected_model = cfg["reranker"].get("model") if enabled else None
    if manifest.get("model") != expected_model:
        return False
    src = sorted(neighbors_dir(output_root, statement_type).glob("part-*.npz"))
    dst = sorted(rerank_dir(output_root, statement_type).glob("part-*.npz"))
    if len(src) != len(dst):
        return False
    for a, b in zip(src, dst):
        try:
            sa, sb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
            if len(sa["a"]) != len(sb["a"]) or "score" not in sb.files:
                return False
            if not np.array_equal(sa["a"], sb["a"]) or not np.array_equal(sa["b"], sb["b"]):
                return False
        except Exception:
            return False
    return True
