from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from statement_ranking.dataset import DatasetPaths, load_canonical_statement_dictionary
from statement_ranking.storage import (
    TYPES,
    final_all_interactions_path,
    final_canonical_statements_path,
    final_id2original_path,
    final_id2statement_path,
    final_interactions_path,
    final_item_candidates_path,
    final_items_path,
    final_original_statements_path,
    final_users_path,
)
from statement_ranking.utils import atomic_write_json, encode_ids, read_jsonl, unique_preserve, write_csv


VIEW_INTERACTION_FIELDS = [
    "interaction_id", "line_index", "user_id", "item_id", "rating", "timestamp", "split",
    "user_sequence_index", "user_sequence_length", "item_sequence_index", "item_sequence_length",
    "global_sequence_index", "statement_ids", "aligned_statement_ids", "original_statement_ids",
]

ALL_INTERACTION_FIELDS = [
    "interaction_id", "line_index", "user_id", "item_id", "rating", "timestamp", "split",
    "user_sequence_index", "user_sequence_length", "item_sequence_index", "item_sequence_length",
    "global_sequence_index",
    "user_statement_ids", "user_aligned_statement_ids", "user_original_statement_ids",
    "item_statement_ids", "item_aligned_statement_ids", "item_original_statement_ids",
    "match_statement_ids", "match_aligned_statement_ids", "match_original_statement_ids",
]

CANONICAL_FIELDS = [
    "statement_id", "type_statement_id", "type", "text",
    "representative_raw_statement_id", "representative_method", "member_count", "occurrence_count",
    "member_raw_statement_ids", "verification_kind", "sentinel_count", "sentinel_raw_statement_ids",
    "embedding_cover_radius", "min_merge_sentinel_score", "merge_verification_pair_count",
]

ORIGINAL_FIELDS = [
    "original_statement_id", "type", "canonical_statement_id", "interaction_id", "line_index",
    "statement_index", "user_id", "item_id", "source_span", "text", "text_source", "stance",
]


def _view_rows(interactions: list[dict[str, Any]], typ: str):
    for row in interactions:
        yield {
            "interaction_id": int(row["interaction_id"]),
            "line_index": int(row["line_index"]),
            "user_id": row["user_id"],
            "item_id": row["item_id"],
            "rating": "" if row.get("rating") is None else row["rating"],
            "timestamp": int(row["timestamp"]),
            "split": row["split"],
            "user_sequence_index": int(row["user_sequence_index"]),
            "user_sequence_length": int(row["user_sequence_length"]),
            "item_sequence_index": int(row["item_sequence_index"]),
            "item_sequence_length": int(row["item_sequence_length"]),
            "global_sequence_index": int(row["global_sequence_index"]),
            "statement_ids": encode_ids(row.get(f"{typ}_statement_ids", [])),
            "aligned_statement_ids": encode_ids(row.get(f"{typ}_aligned_statement_ids", [])),
            "original_statement_ids": encode_ids(row.get(f"{typ}_original_statement_ids", [])),
        }


def _all_rows(interactions: list[dict[str, Any]]):
    for row in interactions:
        out = {
            "interaction_id": int(row["interaction_id"]),
            "line_index": int(row["line_index"]),
            "user_id": row["user_id"],
            "item_id": row["item_id"],
            "rating": "" if row.get("rating") is None else row["rating"],
            "timestamp": int(row["timestamp"]),
            "split": row["split"],
            "user_sequence_index": int(row["user_sequence_index"]),
            "user_sequence_length": int(row["user_sequence_length"]),
            "item_sequence_index": int(row["item_sequence_index"]),
            "item_sequence_length": int(row["item_sequence_length"]),
            "global_sequence_index": int(row["global_sequence_index"]),
        }
        for typ in TYPES:
            out[f"{typ}_statement_ids"] = encode_ids(row.get(f"{typ}_statement_ids", []))
            out[f"{typ}_aligned_statement_ids"] = encode_ids(row.get(f"{typ}_aligned_statement_ids", []))
            out[f"{typ}_original_statement_ids"] = encode_ids(row.get(f"{typ}_original_statement_ids", []))
        yield out


def _referenced(interactions: list[dict[str, Any]], typ: str) -> tuple[set[int], set[int]]:
    canonical: set[int] = set()
    original: set[int] = set()
    for row in interactions:
        canonical.update(int(v) for v in row.get(f"{typ}_statement_ids", []))
        original.update(int(v) for v in row.get(f"{typ}_original_statement_ids", []))
    return canonical, original


def _canonical_rows(dictionary: dict[int, dict[str, Any]], ids: set[int]):
    for sid in sorted(ids):
        record = dictionary[sid]
        yield {
            "statement_id": sid,
            "type_statement_id": record.get("type_statement_id", ""),
            "type": record.get("type", ""),
            "text": record.get("text", ""),
            "representative_raw_statement_id": record.get("representative_raw_statement_id", ""),
            "representative_method": record.get("representative_method", ""),
            "member_count": record.get("member_count", ""),
            "occurrence_count": record.get("occurrence_count", ""),
            # These are retained only as clustering provenance. The ranking module does
            # not require the deleted raw-statement dictionaries that gave them meaning.
            "member_raw_statement_ids": encode_ids(record.get("member_raw_statement_ids", [])),
            "verification_kind": record.get("verification_kind", ""),
            "sentinel_count": record.get("sentinel_count", ""),
            "sentinel_raw_statement_ids": encode_ids(record.get("sentinel_raw_statement_ids", [])),
            "embedding_cover_radius": record.get("embedding_cover_radius", ""),
            "min_merge_sentinel_score": record.get("min_merge_sentinel_score", ""),
            "merge_verification_pair_count": record.get("merge_verification_pair_count", ""),
        }


def _load_needed_original_occurrences(
    paths: DatasetPaths,
    needed_by_type: dict[str, set[int]],
) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {typ: {} for typ in TYPES}
    key_to_targets: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)

    for _, ev in read_jsonl(paths.clustering_final / "evidence.jsonl"):
        evidence_id = int(ev["evidence_id"])
        for typ in TYPES:
            if evidence_id not in needed_by_type[typ]:
                continue
            canonical = ev.get(f"{typ}_statement_id")
            if canonical is None:
                raise ValueError(
                    f"Referenced original evidence_id={evidence_id} has no {typ}_statement_id"
                )
            rec = {
                "original_statement_id": evidence_id,
                "type": typ,
                "canonical_statement_id": int(canonical),
                "interaction_id": int(ev["interaction_id"]),
                "line_index": int(ev["line_index"]),
                "statement_index": int(ev["statement_index"]),
                "user_id": str(ev["user_id"]),
                "item_id": str(ev["item_id"]),
                "source_span": str(ev.get("source_span", "")),
                "text": str(ev.get("source_span", "")),
                "text_source": "source_span",
                "stance": "" if ev.get("stance") is None else str(ev.get("stance")),
            }
            result[typ][evidence_id] = rec
            key_to_targets[(int(ev["line_index"]), int(ev["statement_index"]))].append((typ, evidence_id))

    for typ in TYPES:
        missing = needed_by_type[typ] - set(result[typ])
        if missing:
            raise ValueError(f"Missing final evidence occurrence IDs for {typ}: {sorted(missing)[:10]}")

    # Enrich occurrence text from the retained evidence-pipeline FINAL artifact when
    # available. It is optional: source_span from clustering final is already sufficient
    # to make the exported occurrence dictionary self-contained.
    source = paths.evidence_final / "reviews_with_statements.jsonl"
    if source.exists() and key_to_targets:
        needed_lines = {line for line, _ in key_to_targets}
        for file_index, review in read_jsonl(source):
            line_index = int(review.get("line_index", file_index))
            if line_index not in needed_lines:
                continue
            statements = review.get("statements") or []
            for statement_index, statement in enumerate(statements):
                targets = key_to_targets.get((line_index, statement_index))
                if not targets:
                    continue
                formalizations = statement.get("formalizations") or {}
                for typ, evidence_id in targets:
                    text = str(formalizations.get(typ, "")).strip()
                    if text:
                        result[typ][evidence_id]["text"] = text
                        result[typ][evidence_id]["text_source"] = "evidence_pipeline_final_formalization"
    return result


def _export_entities(paths: DatasetPaths, output_root: Path, interactions: list[dict[str, Any]]) -> None:
    users = {str(row["user_id"]) for row in interactions}
    items = {str(row["item_id"]) for row in interactions}

    user_meta: dict[str, dict[str, Any]] = {}
    if paths.meta_users.exists():
        for _, record in read_jsonl(paths.meta_users):
            uid = str(record.get("user_id", ""))
            if uid in users:
                user_meta[uid] = record
    item_meta: dict[str, dict[str, Any]] = {}
    if paths.meta_items.exists():
        for _, record in read_jsonl(paths.meta_items):
            iid = str(record.get("item_id", ""))
            if iid in items:
                item_meta[iid] = record

    write_csv(
        final_users_path(output_root),
        ({"user_id": uid, "metadata_json": json.dumps(user_meta.get(uid, {}), ensure_ascii=False, separators=(",", ":"))}
         for uid in sorted(users)),
        ["user_id", "metadata_json"],
    )
    write_csv(
        final_items_path(output_root),
        ({"item_id": iid, "metadata_json": json.dumps(item_meta.get(iid, {}), ensure_ascii=False, separators=(",", ":"))}
         for iid in sorted(items)),
        ["item_id", "metadata_json"],
    )


def export_final(
    *,
    paths: DatasetPaths,
    output_root: Path,
    interactions: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    pipeline_stats: dict[str, Any],
) -> dict[str, Any]:
    final_root = output_root / "final"
    final_root.mkdir(parents=True, exist_ok=True)

    write_csv(final_all_interactions_path(output_root), _all_rows(interactions), ALL_INTERACTION_FIELDS)

    needed_original: dict[str, set[int]] = {}
    canonical_ids_by_type: dict[str, set[int]] = {}
    dictionaries: dict[str, dict[int, dict[str, Any]]] = {}
    for typ in TYPES:
        canonical_ids, original_ids = _referenced(interactions, typ)
        canonical_ids_by_type[typ] = canonical_ids
        needed_original[typ] = original_ids
        dictionaries[typ] = load_canonical_statement_dictionary(paths, typ)
        missing = canonical_ids - set(dictionaries[typ])
        if missing:
            raise ValueError(f"Missing canonical {typ} statement IDs: {sorted(missing)[:10]}")

    original_records = _load_needed_original_occurrences(paths, needed_original)

    export_stats: dict[str, Any] = {"views": {}}
    for typ in TYPES:
        write_csv(final_interactions_path(output_root, typ), _view_rows(interactions, typ), VIEW_INTERACTION_FIELDS)
        write_csv(
            final_canonical_statements_path(output_root, typ),
            _canonical_rows(dictionaries[typ], canonical_ids_by_type[typ]),
            CANONICAL_FIELDS,
        )
        write_csv(
            final_original_statements_path(output_root, typ),
            (original_records[typ][sid] for sid in sorted(original_records[typ])),
            ORIGINAL_FIELDS,
        )
        atomic_write_json(
            final_id2statement_path(output_root, typ),
            {str(sid): str(dictionaries[typ][sid].get("text", "")) for sid in sorted(canonical_ids_by_type[typ])},
        )
        atomic_write_json(
            final_id2original_path(output_root, typ),
            {str(sid): str(original_records[typ][sid].get("text", "")) for sid in sorted(original_records[typ])},
        )

        by_item: dict[str, list[int]] = defaultdict(list)
        for row in interactions:
            by_item[str(row["item_id"])].extend(int(v) for v in row.get(f"{typ}_statement_ids", []))
        candidate_rows = [
            {"item_id": item_id, "statement_ids": encode_ids(unique_preserve(ids))}
            for item_id, ids in sorted(by_item.items())
        ]
        write_csv(final_item_candidates_path(output_root, typ), candidate_rows, ["item_id", "statement_ids"])

        export_stats["views"][typ] = {
            "canonical_statements": len(canonical_ids_by_type[typ]),
            "original_occurrences": len(original_records[typ]),
            "items_with_candidates": len(candidate_rows),
        }

    _export_entities(paths, output_root, interactions)

    train_ids = [int(row["interaction_id"]) for row in interactions if row["split"] == "train"]
    test_ids = [int(row["interaction_id"]) for row in interactions if row["split"] == "test"]
    (final_root / "train.index").write_text(" ".join(map(str, train_ids)) + "\n", encoding="utf-8")
    (final_root / "test.index").write_text(" ".join(map(str, test_ids)) + "\n", encoding="utf-8")

    if excluded:
        write_csv(
            final_root / "excluded_cold_start.csv",
            ({
                "interaction_id": row["interaction_id"], "user_id": row["user_id"],
                "item_id": row["item_id"], "timestamp": row["timestamp"],
                "reason": row.get("exclusion_reason", "item_cold_start"),
            } for row in excluded),
            ["interaction_id", "user_id", "item_id", "timestamp", "reason"],
        )

    manifest = {
        "complete": True,
        "dataset": paths.dataset,
        "format": "statement-ranking-csv-v2-final-only",
        "sources": {
            "required": [
                "clustering_pipeline/final/interactions.jsonl",
                "clustering_pipeline/final/evidence.jsonl",
                "clustering_pipeline/final/statements_{user,item,match}.jsonl",
            ],
            "optional_enrichment": [
                "evidence_pipeline/final/reviews_with_statements.jsonl",
                "meta_users.jsonl",
                "meta_items.jsonl",
            ],
            "intermediate_upstream_artifacts_required": False,
        },
        "split": {
            "strategy": "user_last_temporal",
            "train_index": "train.index",
            "test_index": "test.index",
            "cold_start_policy": "drop_test",
        },
        "semantics": {
            "statement_ids": "unique canonical cluster IDs referenced by the interaction",
            "original_statement_ids": (
                "occurrence-level evidence_id values from clustering_pipeline/final/evidence.jsonl; "
                "these play the role of EXTRA senID"
            ),
            "aligned_statement_ids": (
                "canonical statement ID aligned one-to-one with original_statement_ids; "
                "these play the role of EXTRA expID"
            ),
        },
        "stats": {**pipeline_stats, **export_stats},
    }
    atomic_write_json(final_root / "manifest.json", manifest)
    return manifest
