from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from evidence_pipeline.recovery.policy import choose_record, payload_present, should_recover
from evidence_pipeline.storage import (
    latest_records_from_shards,
    load_manifest,
    manifest_path,
    merged_path,
    recovery_dir,
    resolved_path,
    utc_now,
)
from evidence_pipeline.utils import atomic_write_json, read_jsonl, sha256_file, write_jsonl


def _key_field(stage: int) -> str:
    return "line_index" if stage == 1 else "statement_id"


def _sort_key(stage: int, record: dict[str, Any]):
    if stage == 1:
        return int(record["line_index"])
    return int(record["line_index"]), int(record["statement_index"])


def merge_recovery_run(
    *,
    stage: int,
    base_dir: Path,
    recovery_run: str,
) -> dict[str, Any]:
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    base_manifest = load_manifest(base_dir)
    if not base_manifest.get("complete"):
        raise RuntimeError(f"Base run is not complete: {base_dir}")

    primary_records = [record for _, record in read_jsonl(merged_path(base_dir))]
    targets = [record for record in primary_records if should_recover(stage, record)]
    key_field = _key_field(stage)
    target_keys = {record[key_field] for record in targets}

    rec_dir = recovery_dir(base_dir, recovery_run)
    manifest = load_manifest(rec_dir)
    if manifest.get("parent_results_sha256") != sha256_file(merged_path(base_dir)):
        raise RuntimeError("Recovery parent hash no longer matches the base results file.")

    latest = latest_records_from_shards(rec_dir, key_field)
    reruns = [latest[key] for key in target_keys if key in latest]
    reruns.sort(key=lambda record: _sort_key(stage, record))
    write_jsonl(merged_path(rec_dir), reruns)

    missing = target_keys - set(latest)
    recovered_by_key = {record[key_field]: record for record in reruns}
    resolved_records: list[dict[str, Any]] = []
    selected_recovery = 0
    still_targeted = 0

    for primary in primary_records:
        key = primary[key_field]
        if key not in target_keys:
            resolved_records.append(primary)
            continue
        recovered = recovered_by_key.get(key)
        chosen, source = choose_record(stage, primary, recovered)
        chosen = deepcopy(chosen)
        chosen["recovery_resolution"] = {
            "targeted": True,
            "selected": source,
            "recovery_run": recovery_run,
            "primary_status": primary.get("status"),
            "recovery_status": None if recovered is None else recovered.get("status"),
        }
        if source == "recovery":
            selected_recovery += 1
        if should_recover(stage, chosen):
            still_targeted += 1
        resolved_records.append(chosen)

    resolved_records.sort(key=lambda record: _sort_key(stage, record))
    write_jsonl(resolved_path(rec_dir), resolved_records)

    statuses = [record.get("status") for record in reruns]
    resolved_statuses = [record.get("status") for record in resolved_records]
    manifest.update(
        {
            "updated_at": utc_now(),
            "complete": not missing,
            "expected_count": len(target_keys),
            "record_count": len(reruns),
            "ok_count": statuses.count("ok"),
            "partial_count": statuses.count("partial"),
            "error_count": statuses.count("error"),
            "missing_count": len(missing),
            "missing_keys_sample": sorted(map(str, missing))[:20],
            "results_sha256": sha256_file(merged_path(rec_dir)),
            "resolved_results_sha256": sha256_file(resolved_path(rec_dir)),
            "resolved_record_count": len(resolved_records),
            "resolved_ok_count": resolved_statuses.count("ok"),
            "resolved_partial_count": resolved_statuses.count("partial"),
            "resolved_error_count": resolved_statuses.count("error"),
            "selected_recovery_count": selected_recovery,
            "unresolved_target_count": still_targeted,
        }
    )
    atomic_write_json(manifest_path(rec_dir), manifest)
    return manifest
