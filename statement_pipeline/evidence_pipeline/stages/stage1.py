from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evidence_pipeline.chunking import chunk_text
from evidence_pipeline.dataset import dataset_paths, has_review_title, load_item_contexts, selected_reviews
from evidence_pipeline.model_runtime import (
    GenerationMode,
    ModelSpec,
    make_llm,
    make_sampling_params,
    resolve_generation_mode,
)
from evidence_pipeline.prompts import load_stage1_prompt, render
from evidence_pipeline.retry import batch_generate_and_validate
from evidence_pipeline.schemas import STAGE1_JSON_SCHEMA, validate_stage1
from evidence_pipeline.storage import ensure_manifest, load_primary_completed_keys_from_file, repair_jsonl_tail, shard_path, stage1_dir
from evidence_pipeline.utils import append_jsonl


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe(statements: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for statement in statements:
        key = (statement["source_span"], statement["statement"], statement["type"])
        if key not in seen:
            seen.add(key)
            output.append(statement)
    return output


def generate_stage1_records(
    *,
    dataset: str,
    pipeline_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    model_spec: ModelSpec,
    records: list[tuple[int, dict[str, Any]]],
    output_path: Path,
    run_name: str,
    shard_index: int,
    num_shards: int,
    mode: GenerationMode,
    run_kind: str,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    batch_size: int | None,
) -> None:
    if not records:
        return

    paths = dataset_paths(dataset, pipeline_cfg)
    title_enabled = has_review_title(dataset, pipeline_cfg)
    template = load_stage1_prompt(title_enabled)
    contexts = load_item_contexts(paths["items"])

    llm = make_llm(
        model_spec,
        mode=mode,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sampling = make_sampling_params(generation_cfg, STAGE1_JSON_SCHEMA, mode=mode)
    batch_n = int(batch_size or generation_cfg.get("batch_size", 48))
    retries = int(generation_cfg.get("retries", 1))
    max_chars = int(generation_cfg.get("max_review_chars_per_prompt", 10000))
    overlap = int(generation_cfg.get("chunk_overlap_chars", 700))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out:
        for start in tqdm(range(0, len(records), batch_n), desc=f"stage1 {run_kind} shard {shard_index}"):
            batch = records[start : start + batch_n]
            requests: list[dict[str, Any]] = []
            for line_index, review_record in batch:
                review_text = str(review_record.get("review", ""))
                chunks = chunk_text(review_text, max_chars=max_chars, overlap=overlap)
                context = contexts.get(str(review_record.get("item_id", "")), {})
                for chunk_index, chunk in enumerate(chunks):
                    prompt = render(
                        template,
                        {
                            "item_title": context.get("name", ""),
                            "review_title": review_record.get("review_title", ""),
                            "review": chunk,
                        },
                    )
                    requests.append(
                        {
                            "line_index": line_index,
                            "chunk_index": chunk_index,
                            "prompt": prompt,
                            "review": review_text,
                            "review_title": str(review_record.get("review_title", "")) if title_enabled else None,
                            "forbidden_brand": context.get("brand", ""),
                        }
                    )

            prompts = [request["prompt"] for request in requests]
            validators = [
                (
                    lambda value, request=request: validate_stage1(
                        value,
                        review=request["review"],
                        review_title=request["review_title"],
                        forbidden_brand=request["forbidden_brand"],
                    )
                )
                for request in requests
            ]
            generated = batch_generate_and_validate(
                llm=llm,
                spec=model_spec,
                sampling_params=sampling,
                prompts=prompts,
                validators=validators,
                retries=retries,
                mode=mode,
            )

            by_line: dict[int, list[tuple[dict[str, Any], Any]]] = defaultdict(list)
            for request, result in zip(requests, generated, strict=True):
                by_line[request["line_index"]].append((request, result))

            for line_index, review_record in batch:
                all_statements: list[dict[str, str]] = []
                errors: list[str] = []
                raw_outputs: list[str] = []
                for request, result in by_line.get(line_index, []):
                    if result.value is not None:
                        all_statements.extend(result.value)
                    if result.errors:
                        errors.extend(f"chunk[{request['chunk_index']}]: {error}" for error in result.errors)
                        raw_outputs.extend(result.raw_answers)

                deduped = _dedupe(all_statements)
                enriched = [
                    {
                        "statement_id": f"{line_index}:{statement_index}",
                        "statement_index": statement_index,
                        **statement,
                    }
                    for statement_index, statement in enumerate(deduped)
                ]
                if errors and enriched:
                    status = "partial"
                elif errors:
                    status = "error"
                else:
                    status = "ok"

                record: dict[str, Any] = {
                    "line_index": line_index,
                    "status": status,
                    "statements": enriched,
                    "dataset": dataset,
                    "user_id": review_record.get("user_id", ""),
                    "item_id": review_record.get("item_id", ""),
                    "run_name": run_name,
                    "run_kind": run_kind,
                    "model": model_spec.name,
                    "model_family": model_spec.family,
                    "prompt_revision": generation_cfg["prompt_revision"],
                    "prompt_variant": template.name,
                    "prompt_sha256": template.sha256,
                    "thinking_requested": mode.thinking_requested,
                    "thinking_enabled": mode.thinking_enabled,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "finished_at": _now(),
                }
                if errors:
                    record["errors"] = errors
                    record["raw_outputs"] = raw_outputs
                append_jsonl(out, record)


def run_stage1(
    *,
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    model_spec: ModelSpec,
    run_name: str,
    shard_index: int,
    num_shards: int,
    limit: int | None,
    indices: set[int] | None,
    force: bool,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    batch_size: int | None,
) -> None:
    paths = dataset_paths(dataset, pipeline_cfg)
    title_enabled = has_review_title(dataset, pipeline_cfg)
    template = load_stage1_prompt(title_enabled)

    if indices is not None:
        selection = {"mode": "indices", "indices": sorted(indices)}
    elif limit is not None:
        selection = {"mode": "limit", "limit": int(limit)}
    else:
        selection = {"mode": "all"}

    run_dir = stage1_dir(dataset, pipeline_cfg, run_name)
    mode = resolve_generation_mode(model_spec, thinking=False)
    ensure_manifest(
        run_dir,
        {
            "run_kind": "primary",
            "stage": 1,
            "dataset": dataset,
            "run_name": run_name,
            "model": model_spec.name,
            "model_family": model_spec.family,
            "prompt_revision": str(stage_cfg["prompt_revision"]),
            "prompt_variant": template.name,
            "prompt_sha256": template.sha256,
            "selection": selection,
            "thinking_requested": False,
            "thinking_enabled": False,
            "num_shards": num_shards,
        },
        force=False,
    )

    output_path = shard_path(run_dir, shard_index, num_shards)
    if force and output_path.exists():
        output_path.unlink()
    if not force:
        repair_jsonl_tail(output_path)
    completed = load_primary_completed_keys_from_file(output_path, "line_index") if not force else set()
    records = [
        (line_index, review)
        for line_index, review in selected_reviews(paths["reviews"], limit=limit, indices=indices)
        if line_index % num_shards == shard_index and line_index not in completed
    ]
    if not records:
        print(f"Stage 1 primary shard {shard_index}/{num_shards}: nothing to do.", flush=True)
        return

    generate_stage1_records(
        dataset=dataset,
        pipeline_cfg=pipeline_cfg,
        generation_cfg=stage_cfg,
        model_spec=model_spec,
        records=records,
        output_path=output_path,
        run_name=run_name,
        shard_index=shard_index,
        num_shards=num_shards,
        mode=mode,
        run_kind="primary",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        batch_size=batch_size,
    )
