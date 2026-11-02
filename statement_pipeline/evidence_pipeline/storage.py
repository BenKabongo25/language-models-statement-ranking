from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from evidence_pipeline.dataset import dataset_paths, selected_reviews
from evidence_pipeline.utils import (
    atomic_write_json,
    load_json,
    read_jsonl,
    sanitize_name,
    sha256_file,
    stable_bucket,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


MANIFEST_SCHEMA_VERSION = 2


def _upgrade_legacy_manifest(
    current: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Upgrade manifests written by pre-recovery primary runs.

    The staged pipeline before recovery support did not record ``run_kind`` or
    thinking metadata. Those runs were necessarily primary, non-thinking runs,
    so adding these values is a lossless metadata migration. No other missing
    immutable field is guessed here.
    """
    upgraded = dict(current)
    migrations: list[str] = []

    if "run_kind" not in upgraded and expected.get("run_kind") == "primary":
        upgraded["run_kind"] = "primary"
        migrations.append("add-run-kind-primary")

    inferred_primary = upgraded.get("run_kind") == "primary"
    if inferred_primary and expected.get("thinking_requested") is False:
        if "thinking_requested" not in upgraded:
            upgraded["thinking_requested"] = False
            migrations.append("add-thinking-requested-false")
    if inferred_primary and expected.get("thinking_enabled") is False:
        if "thinking_enabled" not in upgraded:
            upgraded["thinking_enabled"] = False
            migrations.append("add-thinking-enabled-false")

    if upgraded.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        upgraded["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
        migrations.append(f"manifest-schema-v{MANIFEST_SCHEMA_VERSION}")

    if migrations:
        history = list(upgraded.get("manifest_migrations", []))
        history.append({"at": utc_now(), "changes": migrations})
        upgraded["manifest_migrations"] = history
        upgraded["updated_at"] = utc_now()

    return upgraded, migrations


def repair_jsonl_tail(path: str | Path) -> bool:
    """Repair only a malformed *trailing* JSONL record after interruption.

    A Slurm timeout can terminate Python between writing and flushing a record.
    Appending to such a file would concatenate new JSON onto the broken tail and
    make the shard unreadable. We therefore preserve the malformed tail in a
    sidecar backup and truncate only that final unusable record. Any corruption
    before the final non-empty line remains a hard error.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False

    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    nonempty = [i for i, raw in enumerate(lines) if raw.strip()]
    if not nonempty:
        return False
    last_nonempty = nonempty[-1]

    offset = 0
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped:
            try:
                text = stripped.decode("utf-8")
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError("JSONL record is not an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                if index != last_nonempty:
                    raise RuntimeError(
                        f"Malformed JSONL record before the tail in {path} at line {index + 1}; "
                        "refusing automatic repair."
                    ) from exc

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup = path.with_name(f"{path.name}.corrupt-tail-{timestamp}.bak")
                # Preserve the discarded bytes (including any trailing blank data).
                backup.write_bytes(data[offset:])
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                print(
                    f"Repaired interrupted JSONL tail in {path}; discarded bytes saved to {backup}.",
                    flush=True,
                )
                return True
        offset += len(raw)

    # A valid final JSON object without a newline is safe, but normalize it before
    # opening the shard in append mode.
    if data and not data.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
        return True
    return False


def latest_records_from_file(path: str | Path, key_field: str) -> dict[Any, dict[str, Any]]:
    path = Path(path)
    latest: dict[Any, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for _, record in read_jsonl(path):
        if key_field not in record:
            continue
        key = record[key_field]
        previous = latest.get(key)
        if previous is None or str(record.get("finished_at", "")) >= str(previous.get("finished_at", "")):
            latest[key] = record
    return latest


def load_ok_keys_from_file(path: str | Path, key_field: str) -> set[Any]:
    return {
        key
        for key, record in latest_records_from_file(path, key_field).items()
        if record.get("status") == "ok"
    }


_PRIMARY_TERMINAL_STATUSES = {"ok", "partial", "error"}


def load_primary_completed_keys_from_file(path: str | Path, key_field: str) -> set[Any]:
    """Keys already attempted by a primary pass.

    Primary resume is continuation, not recovery: records already emitted as
    ok/partial/error are left untouched and the dedicated recovery pass handles
    partial/error/empty outputs later.
    """
    return {
        key
        for key, record in latest_records_from_file(path, key_field).items()
        if record.get("status") in _PRIMARY_TERMINAL_STATUSES
    }


def model_run_name(model_name: str, revision: str) -> str:
    return sanitize_name(f"{model_name.split('/')[-1]}__{revision}")


def recovery_run_name(model_name: str, revision: str, *, thinking: bool) -> str:
    mode = "thinking" if thinking else "nonthinking"
    return sanitize_name(f"{model_name.split('/')[-1]}__{revision}__{mode}")


def stage1_run_name(
    model_name: str,
    revision: str,
    *,
    limit: int | None = None,
    indices: set[int] | None = None,
) -> str:
    base = model_run_name(model_name, revision)
    if limit is not None and indices is not None:
        raise ValueError("limit and indices are mutually exclusive")
    if limit is not None:
        return sanitize_name(f"{base}__limit{int(limit)}")
    if indices is not None:
        import hashlib

        payload = ",".join(str(value) for value in sorted(indices))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
        return sanitize_name(f"{base}__indices-{digest}")
    return base


def stage1_dir(dataset: str, pipeline_cfg: dict[str, Any], run_name: str) -> Path:
    return dataset_paths(dataset, pipeline_cfg)["pipeline"] / "stage1" / sanitize_name(run_name)


def _stage1_source_component(stage1_recovery_run: str | None) -> str | None:
    if not stage1_recovery_run:
        return None
    return sanitize_name(f"recovered-by={stage1_recovery_run}")


def stage2_dir(
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage1_run: str,
    run_name: str,
    *,
    stage1_recovery_run: str | None = None,
) -> Path:
    path = dataset_paths(dataset, pipeline_cfg)["pipeline"] / "stage2" / sanitize_name(stage1_run)
    source_component = _stage1_source_component(stage1_recovery_run)
    if source_component:
        path = path / source_component
    return path / sanitize_name(run_name)


def stage3_dir(
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage1_run: str,
    run_name: str,
    *,
    stage1_recovery_run: str | None = None,
) -> Path:
    path = dataset_paths(dataset, pipeline_cfg)["pipeline"] / "stage3" / sanitize_name(stage1_run)
    source_component = _stage1_source_component(stage1_recovery_run)
    if source_component:
        path = path / source_component
    return path / sanitize_name(run_name)


def recovery_dir(run_dir: Path, recovery_run: str) -> Path:
    return run_dir / "recovery" / sanitize_name(recovery_run)


def resolved_path(recovery_run_dir: Path) -> Path:
    return recovery_run_dir / "resolved_results.jsonl"


def combined_dir(
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage1_run: str,
    stage2_run: str,
    stage3_run: str,
    *,
    stage1_recovery_run: str | None = None,
    stage2_recovery_run: str | None = None,
    stage3_recovery_run: str | None = None,
) -> Path:
    s1 = sanitize_name(stage1_run)
    if stage1_recovery_run:
        s1 = sanitize_name(f"{s1}__r={stage1_recovery_run}")
    combo = f"s2={stage2_run}"
    if stage2_recovery_run:
        combo += f"__r2={stage2_recovery_run}"
    combo += f"__s3={stage3_run}"
    if stage3_recovery_run:
        combo += f"__r3={stage3_recovery_run}"
    return dataset_paths(dataset, pipeline_cfg)["pipeline"] / "combined" / s1 / sanitize_name(combo)


def shard_path(run_dir: Path, shard_index: int, num_shards: int) -> Path:
    return run_dir / "shards" / f"shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def merged_path(run_dir: Path) -> Path:
    return run_dir / "results.jsonl"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return load_json(path)


def require_complete(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    if not manifest.get("complete"):
        raise RuntimeError(f"Run is not complete: {run_dir}")
    if not merged_path(run_dir).exists():
        raise RuntimeError(f"Run is marked complete but results.jsonl is missing: {run_dir}")
    return manifest


def require_recovery_complete(run_dir: Path, recovery_run: str) -> dict[str, Any]:
    rec_dir = recovery_dir(run_dir, recovery_run)
    manifest = require_complete(rec_dir)
    if not resolved_path(rec_dir).exists():
        raise RuntimeError(f"Recovery is complete but resolved_results.jsonl is missing: {rec_dir}")
    return manifest


def effective_results_path(run_dir: Path, recovery_run: str | None = None) -> Path:
    if recovery_run:
        require_recovery_complete(run_dir, recovery_run)
        return resolved_path(recovery_dir(run_dir, recovery_run))
    require_complete(run_dir)
    return merged_path(run_dir)


def effective_results_sha256(run_dir: Path, recovery_run: str | None = None) -> str:
    return sha256_file(effective_results_path(run_dir, recovery_run))


def ensure_manifest(run_dir: Path, expected: dict[str, Any], force: bool = False) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    path = manifest_path(run_dir)
    if path.exists() and not force:
        current = load_json(path)
        current, migrations = _upgrade_legacy_manifest(current, expected)
        if migrations:
            atomic_write_json(path, current)
            print(
                f"Upgraded legacy manifest in {run_dir}: {', '.join(migrations)}",
                flush=True,
            )

        immutable_keys = [
            "run_kind",
            "stage",
            "dataset",
            "run_name",
            "model",
            "model_family",
            "prompt_revision",
            "prompt_sha256",
            "parent_stage1_run",
            "parent_stage1_recovery_run",
            "parent_stage1_results_sha256",
            "parent_run",
            "parent_results_sha256",
            "recovery_policy_revision",
            "selection",
            "eligible_types",
            "target_policy",
            "thinking_requested",
            "thinking_enabled",
            "num_shards",
        ]
        mismatches = [
            key
            for key in immutable_keys
            if current.get(key) != expected.get(key) and expected.get(key) is not None
        ]
        if mismatches:
            details = {
                key: {"existing": current.get(key), "requested": expected.get(key)}
                for key in mismatches
            }
            raise RuntimeError(
                f"Existing run manifest is incompatible for {run_dir}; mismatched fields: {mismatches}. "
                f"Details: {details}"
            )
        return current

    manifest = dict(expected)
    manifest.update(
        {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "complete": False,
            "record_count": 0,
            "ok_count": 0,
            "partial_count": 0,
            "error_count": 0,
        }
    )
    atomic_write_json(path, manifest)
    return manifest

def latest_records_from_shards(run_dir: Path, key_field: str) -> dict[Any, dict[str, Any]]:
    latest: dict[Any, dict[str, Any]] = {}
    shards_dir = run_dir / "shards"
    if not shards_dir.exists():
        return latest
    for path in sorted(shards_dir.glob("shard-*.jsonl")):
        for _, record in read_jsonl(path):
            if key_field not in record:
                continue
            key = record[key_field]
            previous = latest.get(key)
            if previous is None or str(record.get("finished_at", "")) >= str(previous.get("finished_at", "")):
                latest[key] = record
    return latest


def load_ok_keys(run_dir: Path, key_field: str) -> set[Any]:
    return {
        key
        for key, record in latest_records_from_shards(run_dir, key_field).items()
        if record.get("status") == "ok"
    }


def stage1_expected_indices(
    dataset: str,
    pipeline_cfg: dict[str, Any],
    selection: dict[str, Any],
) -> list[int]:
    paths = dataset_paths(dataset, pipeline_cfg)
    mode = selection.get("mode", "all")
    limit = int(selection["limit"]) if mode == "limit" else None
    indices = set(selection.get("indices", [])) if mode == "indices" else None
    return [index for index, _ in selected_reviews(paths["reviews"], limit=limit, indices=indices)]


def load_stage1_records(
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage1_run: str,
    *,
    stage1_recovery_run: str | None = None,
) -> list[dict[str, Any]]:
    run_dir = stage1_dir(dataset, pipeline_cfg, stage1_run)
    path = effective_results_path(run_dir, stage1_recovery_run)
    return [record for _, record in read_jsonl(path)]


def stage1_candidates(
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage1_run: str,
    eligible_types: set[str],
    *,
    stage1_recovery_run: str | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in load_stage1_records(
        dataset,
        pipeline_cfg,
        stage1_run,
        stage1_recovery_run=stage1_recovery_run,
    ):
        for statement in record.get("statements", []):
            if statement.get("type") in eligible_types:
                candidates.append(
                    {
                        "statement_id": statement["statement_id"],
                        "line_index": record["line_index"],
                        "statement_index": statement["statement_index"],
                        "item_id": record.get("item_id", ""),
                        "user_id": record.get("user_id", ""),
                        "source_span": statement["source_span"],
                        "statement": statement["statement"],
                        "type": statement["type"],
                    }
                )
    return candidates


def merge_run(
    *,
    run_dir: Path,
    expected_keys: Iterable[Any],
    key_field: str,
    sort_key,
) -> dict[str, Any]:
    expected = set(expected_keys)
    latest = latest_records_from_shards(run_dir, key_field)
    selected = [latest[key] for key in expected if key in latest]
    selected.sort(key=sort_key)
    write_jsonl(merged_path(run_dir), selected)

    missing = expected - set(latest)
    statuses = [record.get("status") for record in selected]
    manifest = load_manifest(run_dir)
    manifest.update(
        {
            "updated_at": utc_now(),
            "complete": not missing,
            "expected_count": len(expected),
            "record_count": len(selected),
            "ok_count": statuses.count("ok"),
            "partial_count": statuses.count("partial"),
            "error_count": statuses.count("error"),
            "missing_count": len(missing),
            "missing_keys_sample": sorted(map(str, missing))[:20],
            "results_sha256": sha256_file(merged_path(run_dir)),
        }
    )
    atomic_write_json(manifest_path(run_dir), manifest)
    return manifest


def candidate_belongs(statement_id: str, shard_index: int, num_shards: int) -> bool:
    return stable_bucket(statement_id, num_shards) == shard_index
