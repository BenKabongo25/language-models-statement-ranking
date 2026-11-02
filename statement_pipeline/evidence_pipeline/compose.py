from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from evidence_pipeline.dataset import dataset_paths, load_pipeline_config
from evidence_pipeline.storage import (
    combined_dir,
    effective_results_path,
    load_manifest,
    recovery_dir,
    stage1_dir,
    stage2_dir,
    stage3_dir,
)
from evidence_pipeline.utils import atomic_write_json, read_jsonl, sha256_file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _statement_key(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["line_index"]), int(record["statement_index"])


def _effective_results_hash(run_dir: Path, recovery_run: str | None, path: Path) -> str:
    """Return the hash already recorded by a completed stage when available.

    Re-hashing multi-GB files before composition used to add several full I/O
    passes. The merge/recovery steps already record the authoritative hashes in
    their manifests, so composition reuses those values and falls back to a
    direct hash only for older manifests.
    """
    if recovery_run:
        manifest = load_manifest(recovery_dir(run_dir, recovery_run))
        value = manifest.get("resolved_results_sha256")
    else:
        manifest = load_manifest(run_dir)
        value = manifest.get("results_sha256")
    return str(value) if value else sha256_file(path)


@dataclass
class _SortedStatementCursor:
    """Streaming lookup over a Stage 2/3 results file sorted by statement order.

    Stage 2 and Stage 3 merged/resolved outputs are sorted by
    (line_index, statement_index). Composition therefore needs only one current
    record from each file instead of building a dictionary containing the whole
    file in memory.
    """

    path: Path

    def __post_init__(self) -> None:
        self._iterator: Iterator[tuple[int, dict[str, Any]]] = read_jsonl(self.path)
        self._current: dict[str, Any] | None = None
        self._current_key: tuple[int, int] | None = None
        self._last_key: tuple[int, int] | None = None
        self._advance()

    def _advance(self) -> None:
        try:
            _, record = next(self._iterator)
        except StopIteration:
            self._current = None
            self._current_key = None
            return

        key = _statement_key(record)
        if self._last_key is not None and key < self._last_key:
            raise RuntimeError(
                f"Expected sorted Stage results but {self.path} is out of order: "
                f"{key} follows {self._last_key}. Re-run the stage merge/recovery merge."
            )
        self._last_key = key
        self._current = record
        self._current_key = key

    def get(self, target_key: tuple[int, int], statement_id: str) -> dict[str, Any] | None:
        while self._current is not None and self._current_key is not None and self._current_key < target_key:
            self._advance()

        if self._current is None or self._current_key != target_key:
            return None

        if str(self._current.get("statement_id", "")) != str(statement_id):
            raise RuntimeError(
                f"Statement identity mismatch in {self.path}: expected {statement_id} at {target_key}, "
                f"found {self._current.get('statement_id')}."
            )
        return self._current


@dataclass
class _ReviewCursor:
    path: Path

    def __post_init__(self) -> None:
        self._iterator = read_jsonl(self.path)
        self._current_index: int | None = None
        self._current_record: dict[str, Any] | None = None
        self._advance()

    def _advance(self) -> None:
        try:
            index, record = next(self._iterator)
        except StopIteration:
            self._current_index = None
            self._current_record = None
            return
        self._current_index = int(index)
        self._current_record = record

    def get(self, line_index: int) -> dict[str, Any]:
        while self._current_index is not None and self._current_index < line_index:
            self._advance()
        if self._current_index != line_index or self._current_record is None:
            raise RuntimeError(f"Could not find review line_index={line_index} in {self.path}")
        return self._current_record


class _AtomicHashedJsonlWriter:
    """Stream JSONL to a temporary file, hash while writing, then atomically publish."""

    def __init__(self, path: Path):
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.tmp_path.open("wb")
        self._digest = hashlib.sha256()
        self.count = 0
        self._closed = False

    def write(self, record: dict[str, Any]) -> None:
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self._handle.write(payload)
        self._digest.update(payload)
        self.count += 1

    def close(self) -> str:
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True
        return self._digest.hexdigest()

    def publish(self) -> str:
        digest = self.close()
        self.tmp_path.replace(self.path)
        return digest

    def abort(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True
        # Keep the .tmp file for post-mortem inspection. A later compose run
        # overwrites it from scratch.


def compose(
    *,
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage1_run: str,
    stage2_run: str,
    stage3_run: str,
    stage1_recovery_run: str | None = None,
    stage2_recovery_run: str | None = None,
    stage3_recovery_run: str | None = None,
    strict: bool = False,
) -> Path:
    s1_dir = stage1_dir(dataset, pipeline_cfg, stage1_run)
    s1_path = effective_results_path(s1_dir, stage1_recovery_run)
    s1_hash = _effective_results_hash(s1_dir, stage1_recovery_run, s1_path)

    s2_dir = stage2_dir(
        dataset,
        pipeline_cfg,
        stage1_run,
        stage2_run,
        stage1_recovery_run=stage1_recovery_run,
    )
    s3_dir = stage3_dir(
        dataset,
        pipeline_cfg,
        stage1_run,
        stage3_run,
        stage1_recovery_run=stage1_recovery_run,
    )
    s2_path = effective_results_path(s2_dir, stage2_recovery_run)
    s3_path = effective_results_path(s3_dir, stage3_recovery_run)

    s2_manifest = load_manifest(s2_dir)
    s3_manifest = load_manifest(s3_dir)
    if s2_manifest.get("parent_stage1_results_sha256") != s1_hash:
        raise RuntimeError("Stage 2 was produced from a different effective Stage 1 results file.")
    if s3_manifest.get("parent_stage1_results_sha256") != s1_hash:
        raise RuntimeError("Stage 3 was produced from a different effective Stage 1 results file.")

    s2_hash = _effective_results_hash(s2_dir, stage2_recovery_run, s2_path)
    s3_hash = _effective_results_hash(s3_dir, stage3_recovery_run, s3_path)

    out_dir = combined_dir(
        dataset,
        pipeline_cfg,
        stage1_run,
        stage2_run,
        stage3_run,
        stage1_recovery_run=stage1_recovery_run,
        stage2_recovery_run=stage2_recovery_run,
        stage3_recovery_run=stage3_recovery_run,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.jsonl"
    reviews_path = out_dir / "reviews_with_statements.jsonl"
    results_writer = _AtomicHashedJsonlWriter(results_path)
    reviews_writer = _AtomicHashedJsonlWriter(reviews_path)

    projections = _SortedStatementCursor(s2_path)
    stances = _SortedStatementCursor(s3_path)
    reviews = _ReviewCursor(dataset_paths(dataset, pipeline_cfg)["reviews"])

    s1_manifest = load_manifest(s1_dir if not stage1_recovery_run else recovery_dir(s1_dir, stage1_recovery_run))
    total = s1_manifest.get("resolved_record_count") if stage1_recovery_run else s1_manifest.get("record_count")
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None

    unresolved = 0
    review_count = 0
    ok_count = 0
    partial_count = 0
    error_count = 0

    try:
        with tqdm(total=total_int, desc=f"compose {dataset}", unit="reviews", dynamic_ncols=True) as progress:
            for _, s1_record in read_jsonl(s1_path):
                line_index = int(s1_record["line_index"])
                final_statements: list[dict[str, Any]] = []
                errors: list[str] = list(s1_record.get("errors", []))

                statements = sorted(
                    s1_record.get("statements", []),
                    key=lambda statement: int(statement.get("statement_index", 0)),
                )
                for statement in statements:
                    sid = str(statement["statement_id"])
                    statement_index = int(statement["statement_index"])
                    key = (line_index, statement_index)
                    kind = statement["type"]
                    stance: str | None = None
                    formalizations: dict[str, str]

                    if kind == "item":
                        formalizations = {"item": statement["statement"]}
                    elif kind == "external":
                        formalizations = {"external": statement["statement"]}
                    elif kind == "user":
                        stance_record = stances.get(key, sid)
                        if not stance_record or stance_record.get("status") != "ok" or not stance_record.get("stance"):
                            unresolved += 1
                            errors.append(f"Missing valid Stage 3 stance for {sid}")
                            if strict:
                                continue
                        else:
                            stance = stance_record["stance"]
                        formalizations = {"user": statement["statement"]}
                    elif kind == "interaction":
                        projection_record = projections.get(key, sid)
                        stance_record = stances.get(key, sid)

                        if (
                            not projection_record
                            or projection_record.get("status") != "ok"
                            or not projection_record.get("projection")
                        ):
                            unresolved += 1
                            errors.append(f"Missing valid Stage 2 projection for {sid}")
                            if strict:
                                continue
                            projection = None
                        else:
                            projection = projection_record["projection"]

                        if not stance_record or stance_record.get("status") != "ok" or not stance_record.get("stance"):
                            unresolved += 1
                            errors.append(f"Missing valid Stage 3 stance for {sid}")
                            if strict:
                                continue
                        else:
                            stance = stance_record["stance"]

                        if projection is None:
                            continue
                        formalizations = {
                            "user": projection["user"],
                            "item": projection["item"],
                            "match": statement["statement"],
                        }
                    else:  # pragma: no cover
                        continue

                    final_statements.append(
                        {
                            "source_span": statement["source_span"],
                            "stance": stance,
                            "formalizations": formalizations,
                        }
                    )

                status = "ok" if not errors else ("partial" if final_statements else "error")
                record: dict[str, Any] = {
                    "line_index": line_index,
                    "status": status,
                    "statements": final_statements,
                    "dataset": dataset,
                    "user_id": s1_record.get("user_id", ""),
                    "item_id": s1_record.get("item_id", ""),
                }
                if errors:
                    record["errors"] = errors
                results_writer.write(record)

                review = dict(reviews.get(line_index))
                review["statements"] = final_statements
                review["statement_extraction_status"] = status
                reviews_writer.write(review)

                review_count += 1
                if status == "ok":
                    ok_count += 1
                elif status == "partial":
                    partial_count += 1
                else:
                    error_count += 1
                progress.update(1)
    except Exception:
        results_writer.abort()
        reviews_writer.abort()
        raise

    results_hash = results_writer.publish()
    reviews_hash = reviews_writer.publish()

    manifest = {
        "dataset": dataset,
        "created_at": _now(),
        "compose_impl": "streaming-v1",
        "stage1_run": stage1_run,
        "stage1_recovery_run": stage1_recovery_run,
        "stage2_run": stage2_run,
        "stage2_recovery_run": stage2_recovery_run,
        "stage3_run": stage3_run,
        "stage3_recovery_run": stage3_recovery_run,
        "stage1_results_sha256": s1_hash,
        "stage2_results_sha256": s2_hash,
        "stage3_results_sha256": s3_hash,
        "results_sha256": results_hash,
        "reviews_with_statements_sha256": reviews_hash,
        "review_count": review_count,
        "ok_count": ok_count,
        "partial_count": partial_count,
        "error_count": error_count,
        "unresolved_dependency_count": unresolved,
        "strict": strict,
        "stance_policy": "Stage 3 applies to USER and INTERACTION; ITEM and EXTERNAL have null stance.",
        "recovery_policy": "Each selected recovery overlays the primary stage with the better result per target key.",
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    if strict and unresolved:
        raise RuntimeError(f"Strict composition found {unresolved} unresolved Stage 2/3 dependencies.")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose Stage 1/2/3 outputs into the final normalized statement format.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1-run", required=True)
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--stage2-run", required=True)
    parser.add_argument("--stage2-recovery-run", default=None)
    parser.add_argument("--stage3-run", required=True)
    parser.add_argument("--stage3-recovery-run", default=None)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    pipeline_cfg = load_pipeline_config(args.pipeline_config)
    out_dir = compose(
        dataset=args.dataset,
        pipeline_cfg=pipeline_cfg,
        stage1_run=args.stage1_run,
        stage1_recovery_run=args.stage1_recovery_run,
        stage2_run=args.stage2_run,
        stage2_recovery_run=args.stage2_recovery_run,
        stage3_run=args.stage3_run,
        stage3_recovery_run=args.stage3_recovery_run,
        strict=args.strict,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
