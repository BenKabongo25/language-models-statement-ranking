from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evidence_pipeline.dataset import dataset_paths, load_item_contexts
from evidence_pipeline.model_runtime import (
    GenerationMode,
    ModelSpec,
    make_llm,
    make_sampling_params,
    resolve_generation_mode,
)
from evidence_pipeline.prompts import load_stage2_prompt, render
from evidence_pipeline.retry import batch_generate_and_validate
from evidence_pipeline.schemas import STAGE2_JSON_SCHEMA, validate_stage2
from evidence_pipeline.storage import (
    candidate_belongs,
    effective_results_path,
    ensure_manifest,
    load_manifest,
    load_primary_completed_keys_from_file,
    repair_jsonl_tail,
    shard_path,
    stage1_candidates,
    stage1_dir,
    recovery_dir,
    stage2_dir,
)
from evidence_pipeline.utils import append_jsonl, sha256_file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_stage2_records(
    *,
    dataset: str,
    pipeline_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    model_spec: ModelSpec,
    candidates: list[dict[str, Any]],
    output_path: Path,
    run_name: str,
    stage1_run: str,
    stage1_recovery_run: str | None,
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
    if not candidates:
        return

    contexts = load_item_contexts(dataset_paths(dataset, pipeline_cfg)["items"])
    template = load_stage2_prompt()
    llm = make_llm(
        model_spec,
        mode=mode,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sampling = make_sampling_params(generation_cfg, STAGE2_JSON_SCHEMA, mode=mode)
    batch_n = int(batch_size or generation_cfg.get("batch_size", 128))
    retries = int(generation_cfg.get("retries", 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out:
        for start in tqdm(range(0, len(candidates), batch_n), desc=f"stage2 {run_kind} shard {shard_index}"):
            batch = candidates[start : start + batch_n]
            prompts = [
                render(template, {"source_span": candidate["source_span"], "statement": candidate["statement"]})
                for candidate in batch
            ]
            validators = [
                (
                    lambda value, brand=contexts.get(str(candidate.get("item_id", "")), {}).get("brand", ""): validate_stage2(
                        value, forbidden_brand=brand
                    )
                )
                for candidate in batch
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
            for candidate, result in zip(batch, generated, strict=True):
                record: dict[str, Any] = {
                    "statement_id": candidate["statement_id"],
                    "line_index": candidate["line_index"],
                    "statement_index": candidate["statement_index"],
                    "source_span": candidate["source_span"],
                    "interaction": candidate["statement"],
                    "status": "ok" if result.ok else "error",
                    "projection": result.value if result.ok else None,
                    "dataset": dataset,
                    "run_name": run_name,
                    "run_kind": run_kind,
                    "parent_stage1_run": stage1_run,
                    "parent_stage1_recovery_run": stage1_recovery_run,
                    "model": model_spec.name,
                    "model_family": model_spec.family,
                    "prompt_revision": generation_cfg["prompt_revision"],
                    "prompt_sha256": template.sha256,
                    "thinking_requested": mode.thinking_requested,
                    "thinking_enabled": mode.thinking_enabled,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "finished_at": _now(),
                }
                if not result.ok:
                    record["errors"] = result.errors
                    record["raw_outputs"] = result.raw_answers
                append_jsonl(out, record)


def run_stage2(
    *,
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    model_spec: ModelSpec,
    run_name: str,
    stage1_run: str,
    stage1_recovery_run: str | None,
    shard_index: int,
    num_shards: int,
    force: bool,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    batch_size: int | None,
) -> None:
    parent_dir = stage1_dir(dataset, pipeline_cfg, stage1_run)
    parent_path = effective_results_path(parent_dir, stage1_recovery_run)
    parent_manifest = load_manifest(
        parent_dir if not stage1_recovery_run else recovery_dir(parent_dir, stage1_recovery_run)
    )
    parent_hash = sha256_file(parent_path)

    template = load_stage2_prompt()
    run_dir = stage2_dir(
        dataset,
        pipeline_cfg,
        stage1_run,
        run_name,
        stage1_recovery_run=stage1_recovery_run,
    )
    mode = resolve_generation_mode(model_spec, thinking=False)
    ensure_manifest(
        run_dir,
        {
            "run_kind": "primary",
            "stage": 2,
            "dataset": dataset,
            "run_name": run_name,
            "model": model_spec.name,
            "model_family": model_spec.family,
            "prompt_revision": stage_cfg["prompt_revision"],
            "prompt_variant": template.name,
            "prompt_sha256": template.sha256,
            "parent_stage1_run": stage1_run,
            "parent_stage1_recovery_run": stage1_recovery_run,
            "parent_stage1_results_sha256": parent_hash,
            "selection": parent_manifest.get("selection"),
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
    completed = load_primary_completed_keys_from_file(output_path, "statement_id") if not force else set()

    candidates = [
        candidate
        for candidate in stage1_candidates(
            dataset,
            pipeline_cfg,
            stage1_run,
            {"interaction"},
            stage1_recovery_run=stage1_recovery_run,
        )
        if candidate_belongs(candidate["statement_id"], shard_index, num_shards)
        and candidate["statement_id"] not in completed
    ]
    if not candidates:
        print(f"Stage 2 primary shard {shard_index}/{num_shards}: nothing to do.", flush=True)
        return

    generate_stage2_records(
        dataset=dataset,
        pipeline_cfg=pipeline_cfg,
        generation_cfg=stage_cfg,
        model_spec=model_spec,
        candidates=candidates,
        output_path=output_path,
        run_name=run_name,
        stage1_run=stage1_run,
        stage1_recovery_run=stage1_recovery_run,
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
