from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from clustering_pipeline.compat import legacy_prepare_usable
from clustering_pipeline.signatures import SCHEMA_VERSION, prepare_artifact_id
from clustering_pipeline.storage import (
    STAGE_PREPARE,
    TYPES,
    evidence_raw_path,
    interactions_raw_path,
    statements_path,
)
from clustering_pipeline.utils import (
    atomic_write_json,
    clean_text,
    manifest_current,
    quick_file_fingerprint,
    reset_dir,
)

PREPARE_IMPL_VERSION = "python-dedup-v2"


def _iter_input(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_index + 1}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_index + 1}")
            yield line_index, value


def _get_or_create(
    statement_type: str,
    text: str,
    text_to_id: dict[str, dict[str, int]],
    texts: dict[str, list[str]],
    occurrence_counts: dict[str, list[int]],
) -> int:
    mapping = text_to_id[statement_type]
    raw_id = mapping.get(text)
    if raw_id is None:
        raw_id = len(texts[statement_type])
        mapping[text] = raw_id
        texts[statement_type].append(text)
        occurrence_counts[statement_type].append(1)
    else:
        occurrence_counts[statement_type][raw_id] += 1
    return raw_id


def run_prepare(*, input_file: Path, output_root: Path, cfg: dict[str, Any], force: bool = False) -> None:
    stage = output_root / STAGE_PREPARE
    artifact_id = prepare_artifact_id(input_file, cfg)
    outputs = [
        *(statements_path(output_root, t) for t in TYPES),
        interactions_raw_path(output_root),
        evidence_raw_path(output_root),
    ]
    manifest_path = stage / "manifest.json"

    if not force and manifest_current(manifest_path, artifact_id, outputs):
        print("prepare: up to date")
        return

    # Adopt complete outputs produced by the previous pipeline instead of
    # re-reading millions of reviews just because the manifest schema changed.
    if not force and legacy_prepare_usable(output_root, input_file):
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old.update(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "implementation": PREPARE_IMPL_VERSION,
                "legacy_adopted": True,
                "input_fingerprint": quick_file_fingerprint(input_file),
            }
        )
        atomic_write_json(manifest_path, old)
        print("prepare: adopted compatible legacy outputs")
        return

    reset_dir(stage)

    p_cfg = cfg.get("prepare", {})
    aliases = {str(k): str(v) for k, v in (cfg.get("type_aliases") or {}).items()}
    only_ok = bool(p_cfg.get("only_status_ok", True))

    text_to_id: dict[str, dict[str, int]] = {t: {} for t in TYPES}
    texts: dict[str, list[str]] = {t: [] for t in TYPES}
    occurrence_counts: dict[str, list[int]] = {t: [] for t in TYPES}

    interaction_count = 0
    evidence_count = 0
    skipped_status = 0

    with interactions_raw_path(output_root).open("w", encoding="utf-8") as interaction_out, evidence_raw_path(
        output_root
    ).open("w", encoding="utf-8") as evidence_out:
        for line_index, review in tqdm(_iter_input(input_file), desc="prepare"):
            status = review.get("statement_extraction_status")
            if only_ok and status not in {None, "ok"}:
                skipped_status += 1
                continue

            by_type: dict[str, list[int]] = {t: [] for t in TYPES}
            for statement_index, statement in enumerate(review.get("statements") or []):
                if not isinstance(statement, dict):
                    continue
                forms = statement.get("formalizations") or {}
                if not isinstance(forms, dict):
                    continue

                evidence_ids: dict[str, int | None] = {t: None for t in TYPES}
                for source_type, value in forms.items():
                    typ = aliases.get(str(source_type), str(source_type))
                    if typ not in TYPES:
                        continue
                    text = clean_text(value)
                    if not text:
                        continue
                    raw_id = _get_or_create(typ, text, text_to_id, texts, occurrence_counts)
                    evidence_ids[typ] = raw_id
                    by_type[typ].append(raw_id)

                if any(v is not None for v in evidence_ids.values()):
                    evidence_out.write(
                        json.dumps(
                            {
                                "evidence_id": evidence_count,
                                "interaction_id": interaction_count,
                                "line_index": line_index,
                                "statement_index": statement_index,
                                "user_id": review.get("user_id"),
                                "item_id": review.get("item_id"),
                                "source_span": clean_text(statement.get("source_span")),
                                "stance": statement.get("stance"),
                                "user_raw_statement_id": evidence_ids["user"],
                                "item_raw_statement_id": evidence_ids["item"],
                                "match_raw_statement_id": evidence_ids["match"],
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    evidence_count += 1

            interaction_out.write(
                json.dumps(
                    {
                        "interaction_id": interaction_count,
                        "line_index": line_index,
                        "user_id": review.get("user_id"),
                        "item_id": review.get("item_id"),
                        "timestamp": review.get("timestamp"),
                        "rating": review.get("rating"),
                        "user_raw_statement_ids": list(dict.fromkeys(by_type["user"])),
                        "item_raw_statement_ids": list(dict.fromkeys(by_type["item"])),
                        "match_raw_statement_ids": list(dict.fromkeys(by_type["match"])),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            interaction_count += 1

    counts: dict[str, int] = {}
    for typ in TYPES:
        with statements_path(output_root, typ).open("w", encoding="utf-8") as f:
            for raw_id, (text, count) in enumerate(zip(texts[typ], occurrence_counts[typ])):
                f.write(
                    json.dumps(
                        {
                            "raw_statement_id": raw_id,
                            "type": typ,
                            "text": text,
                            "occurrence_count": int(count),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        counts[typ] = len(texts[typ])

    atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "artifact_id": artifact_id,
            "implementation": PREPARE_IMPL_VERSION,
            "input_file": str(input_file),
            "input_fingerprint": quick_file_fingerprint(input_file),
            "interactions": interaction_count,
            "evidence_records": evidence_count,
            "unique_statements": counts,
            "skipped_non_ok_reviews": skipped_status,
            "dedup_backend": "python_dict",
        },
    )
