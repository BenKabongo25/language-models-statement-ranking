from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from clustering_pipeline.signatures import SCHEMA_VERSION, finalize_artifact_id
from clustering_pipeline.storage import (
    STAGE_FINAL,
    TYPES,
    clusters_path,
    evidence_raw_path,
    interactions_raw_path,
    mapping_path,
)
from clustering_pipeline.utils import (
    artifact_id_from_manifest,
    atomic_write_json,
    load_json,
    manifest_current,
    read_jsonl,
    reset_dir,
    unique_preserve,
)


def _load_mapping_array(path: Path) -> np.ndarray:
    """Load raw_statement_id -> local cluster id as a compact dense array."""
    with np.load(path, allow_pickle=False) as d:
        raw = np.asarray(d["raw_statement_id"], dtype=np.int64)
        cluster = np.asarray(d["cluster_statement_id"], dtype=np.int64)
    if len(raw) == 0:
        return np.empty(0, dtype=np.int64)
    if len(raw) == int(raw.max()) + 1 and np.array_equal(raw, np.arange(len(raw), dtype=np.int64)):
        return cluster
    out = np.full(int(raw.max()) + 1, -1, dtype=np.int64)
    out[raw] = cluster
    return out


def _convert_ids(values: Iterable[int], mapping: np.ndarray, offset: int) -> list[int]:
    """Map raw ids to global canonical ids, preserving first occurrence."""
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        raw = int(value)
        if raw < 0 or raw >= len(mapping):
            continue
        local = int(mapping[raw])
        if local < 0:
            continue
        gid = int(offset + local)
        if gid not in seen:
            seen.add(gid)
            out.append(gid)
    return out


def run_finalize(*, output_root: Path, cfg: dict[str, Any], force: bool = False) -> None:
    out_dir = output_root / STAGE_FINAL
    manifest_path = out_dir / "manifest.json"
    cluster_ids = {
        typ: artifact_id_from_manifest(output_root / "04_clusters" / f"{typ}.manifest.json")
        for typ in TYPES
    }
    artifact_id = finalize_artifact_id(cluster_ids, cfg)
    required = [
        out_dir / "statements.jsonl",
        out_dir / "interactions.jsonl",
        out_dir / "users.jsonl",
        out_dir / "items.jsonl",
        out_dir / "item_candidates.jsonl",
        out_dir / "evidence.jsonl",
        out_dir / "id_offsets.json",
    ]
    if not force and manifest_current(manifest_path, artifact_id, required):
        print("finalize: up to date")
        return

    reset_dir(out_dir)
    mappings = {typ: _load_mapping_array(mapping_path(output_root, typ)) for typ in TYPES}
    cluster_counts: dict[str, int] = {}
    for typ in TYPES:
        meta = load_json(output_root / "04_clusters" / f"{typ}.manifest.json")
        if "clusters" in meta:
            cluster_counts[typ] = int(meta["clusters"])
        else:
            cluster_counts[typ] = 0 if len(mappings[typ]) == 0 else int(mappings[typ].max()) + 1

    offsets: dict[str, int] = {}
    offset = 0
    for typ in TYPES:
        offsets[typ] = offset
        offset += cluster_counts[typ]

    write_type = bool(cfg.get("finalize", {}).get("write_type_dictionaries", True))
    with (out_dir / "statements.jsonl").open("w", encoding="utf-8") as all_out:
        for typ in TYPES:
            type_path = out_dir / f"statements_{typ}.jsonl"
            type_out = type_path.open("w", encoding="utf-8") if write_type else None
            try:
                for _, rec in read_jsonl(clusters_path(output_root, typ)):
                    local = int(rec["cluster_statement_id"])
                    out = dict(rec)
                    out["type_statement_id"] = local
                    out["statement_id"] = offsets[typ] + local
                    out.pop("cluster_statement_id", None)
                    line = json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n"
                    all_out.write(line)
                    if type_out is not None:
                        type_out.write(line)
            finally:
                if type_out is not None:
                    type_out.close()

    # Profiles are intentionally kept in interaction order. Lists are therefore
    # accumulated and deduplicated only at final write time.
    user_profiles: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"user": [], "match": []})
    item_profiles: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"item": [], "match": []})
    item_candidates: dict[str, list[int]] = defaultdict(list)
    interaction_count = 0

    with (out_dir / "interactions.jsonl").open("w", encoding="utf-8") as f:
        iterator = read_jsonl(interactions_raw_path(output_root))
        for _, row in tqdm(iterator, desc="finalize:interactions", unit="interaction"):
            converted = {
                typ: _convert_ids(
                    row.get(f"{typ}_raw_statement_ids", []),
                    mappings[typ],
                    offsets[typ],
                )
                for typ in TYPES
            }
            out = {
                "interaction_id": int(row["interaction_id"]),
                "line_index": int(row["line_index"]),
                "user_id": row.get("user_id"),
                "item_id": row.get("item_id"),
                "timestamp": row.get("timestamp"),
                "rating": row.get("rating"),
                "user_statement_ids": converted["user"],
                "item_statement_ids": converted["item"],
                "match_statement_ids": converted["match"],
            }
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            uid, iid = str(row.get("user_id")), str(row.get("item_id"))
            user_profiles[uid]["user"].extend(converted["user"])
            user_profiles[uid]["match"].extend(converted["match"])
            item_profiles[iid]["item"].extend(converted["item"])
            item_profiles[iid]["match"].extend(converted["match"])
            item_candidates[iid].extend(converted["match"])
            interaction_count += 1

    with (out_dir / "users.jsonl").open("w", encoding="utf-8") as f:
        for uid in tqdm(sorted(user_profiles), desc="finalize:users", unit="user"):
            u = unique_preserve(user_profiles[uid]["user"])
            m = unique_preserve(user_profiles[uid]["match"])
            f.write(
                json.dumps(
                    {
                        "user_id": uid,
                        "user_statement_ids": u,
                        "match_statement_ids": m,
                        "profile_statement_ids": unique_preserve(u + m),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        for iid in tqdm(sorted(item_profiles), desc="finalize:items", unit="item"):
            it = unique_preserve(item_profiles[iid]["item"])
            m = unique_preserve(item_profiles[iid]["match"])
            f.write(
                json.dumps(
                    {
                        "item_id": iid,
                        "item_statement_ids": it,
                        "match_statement_ids": m,
                        "profile_statement_ids": unique_preserve(it + m),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    with (out_dir / "item_candidates.jsonl").open("w", encoding="utf-8") as f:
        for iid in tqdm(sorted(item_candidates), desc="finalize:item-candidates", unit="item"):
            f.write(
                json.dumps(
                    {"item_id": iid, "match_statement_ids": unique_preserve(item_candidates[iid])},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    with (out_dir / "evidence.jsonl").open("w", encoding="utf-8") as f:
        for _, row in tqdm(read_jsonl(evidence_raw_path(output_root)), desc="finalize:evidence", unit="evidence"):
            out = {
                k: row.get(k)
                for k in [
                    "evidence_id",
                    "interaction_id",
                    "line_index",
                    "statement_index",
                    "user_id",
                    "item_id",
                    "source_span",
                    "stance",
                ]
            }
            for typ in TYPES:
                raw = row.get(f"{typ}_raw_statement_id")
                if raw is None:
                    out[f"{typ}_statement_id"] = None
                else:
                    rid = int(raw)
                    local = int(mappings[typ][rid]) if 0 <= rid < len(mappings[typ]) else -1
                    out[f"{typ}_statement_id"] = None if local < 0 else int(offsets[typ] + local)
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")

    atomic_write_json(
        out_dir / "id_offsets.json",
        {"offsets": offsets, "counts": cluster_counts, "total": offset},
    )
    atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "artifact_id": artifact_id,
            "source_artifact_ids": cluster_ids,
            "statements": offset,
            "interactions": interaction_count,
            "users": len(user_profiles),
            "items": len(item_profiles),
            "type_offsets": offsets,
            "note": "users/items are corpus-level canonical evidence aggregates; build split-specific profiles downstream to avoid leakage.",
        },
    )
