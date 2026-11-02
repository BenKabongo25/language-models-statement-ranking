from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from evidence_pipeline.dataset import dataset_paths, has_review_title
from evidence_pipeline.model_runtime import ModelSpec, resolve_generation_mode
from evidence_pipeline.prompts import load_stage1_prompt, load_stage2_prompt, load_stage3_prompt
from evidence_pipeline.recovery.policy import recovery_succeeded, should_recover, target_policy_name
from evidence_pipeline.stages.stage1 import generate_stage1_records
from evidence_pipeline.stages.stage2 import generate_stage2_records
from evidence_pipeline.stages.stage3 import generate_stage3_records
from evidence_pipeline.storage import (
    candidate_belongs,
    effective_results_path,
    ensure_manifest,
    latest_records_from_file,
    load_manifest,
    merged_path,
    recovery_dir,
    repair_jsonl_tail,
    shard_path,
    stage1_candidates,
    stage1_dir,
    stage2_dir,
    stage3_dir,
)
from evidence_pipeline.utils import read_jsonl, sha256_file


def recovery_generation_config(stage_cfg: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(stage_cfg)
    recovery = deepcopy(stage_cfg.get("recovery", {}))
    sampling = deepcopy(recovery.pop("sampling", {}))
    cfg.update(recovery)
    if sampling:
        cfg["sampling"] = sampling
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key in {"temperature", "top_p", "top_k", "max_tokens", "seed"}:
                cfg.setdefault("sampling", {})[key] = value
            else:
                cfg[key] = value
    # Recovery always uses the exact same prompt revision as the primary stage.
    cfg["prompt_revision"] = stage_cfg["prompt_revision"]
    return cfg


def _base_run_dir(
    *,
    stage: int,
    dataset: str,
    pipeline_cfg: dict[str, Any],
    run_name: str,
    stage1_run: str | None,
    stage1_recovery_run: str | None,
) -> Path:
    if stage == 1:
        return stage1_dir(dataset, pipeline_cfg, run_name)
    if not stage1_run:
        raise ValueError("stage1_run is required for Stage 2/3 recovery")
    if stage == 2:
        return stage2_dir(
            dataset,
            pipeline_cfg,
            stage1_run,
            run_name,
            stage1_recovery_run=stage1_recovery_run,
        )
    if stage == 3:
        return stage3_dir(
            dataset,
            pipeline_cfg,
            stage1_run,
            run_name,
            stage1_recovery_run=stage1_recovery_run,
        )
    raise ValueError(f"Unsupported stage: {stage}")


def _target_records(stage: int, base_dir: Path) -> list[dict[str, Any]]:
    base_manifest = load_manifest(base_dir)
    if not base_manifest.get("complete"):
        raise RuntimeError(f"Primary Stage {stage} run must be merged and complete before recovery: {base_dir}")
    return [record for _, record in read_jsonl(merged_path(base_dir)) if should_recover(stage, record)]


def run_recovery(
    *,
    stage: int,
    dataset: str,
    pipeline_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    model_spec: ModelSpec,
    run_name: str,
    recovery_run_name: str,
    stage1_run: str | None,
    stage1_recovery_run: str | None,
    shard_index: int,
    num_shards: int,
    force: bool,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    batch_size: int | None,
    overrides: dict[str, Any] | None = None,
) -> None:
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    base_dir = _base_run_dir(
        stage=stage,
        dataset=dataset,
        pipeline_cfg=pipeline_cfg,
        run_name=run_name,
        stage1_run=stage1_run,
        stage1_recovery_run=stage1_recovery_run,
    )
    base_manifest = load_manifest(base_dir)
    if not base_manifest.get("complete"):
        raise RuntimeError(f"Primary run is not complete: {base_dir}")
    base_hash = sha256_file(merged_path(base_dir))
    targets = _target_records(stage, base_dir)

    generation_cfg = recovery_generation_config(stage_cfg, overrides)
    thinking_requested = bool(generation_cfg.get("thinking", True))
    thinking_budget = generation_cfg.get("thinking_token_budget")
    mode = resolve_generation_mode(
        model_spec,
        thinking=thinking_requested,
        thinking_token_budget=None if thinking_budget is None else int(thinking_budget),
    )

    if stage == 1:
        template = load_stage1_prompt(has_review_title(dataset, pipeline_cfg))
    elif stage == 2:
        template = load_stage2_prompt()
    else:
        template = load_stage3_prompt()
    primary_prompt_hash = base_manifest.get("prompt_sha256")
    if primary_prompt_hash and primary_prompt_hash != template.sha256:
        raise RuntimeError(
            "Recovery must reuse the exact primary prompt, but the current prompt file has a different SHA256. "
            "Restore the original prompt or create a new primary run instead."
        )

    rec_dir = recovery_dir(base_dir, recovery_run_name)
    ensure_manifest(
        rec_dir,
        {
            "run_kind": "recovery",
            "stage": stage,
            "dataset": dataset,
            "run_name": recovery_run_name,
            "model": model_spec.name,
            "model_family": model_spec.family,
            "prompt_revision": stage_cfg["prompt_revision"],
            "prompt_variant": template.name,
            "prompt_sha256": template.sha256,
            "parent_run": run_name,
            "parent_results_sha256": base_hash,
            "parent_stage1_run": stage1_run if stage != 1 else None,
            "parent_stage1_recovery_run": stage1_recovery_run if stage != 1 else None,
            "parent_stage1_results_sha256": base_manifest.get("parent_stage1_results_sha256") if stage != 1 else None,
            "selection": base_manifest.get("selection"),
            "eligible_types": base_manifest.get("eligible_types"),
            "recovery_policy_revision": str(stage_cfg.get("recovery", {}).get("revision", "recovery-v1")),
            "target_policy": target_policy_name(),
            "target_count": len(targets),
            "thinking_requested": mode.thinking_requested,
            "thinking_enabled": mode.thinking_enabled,
            "thinking_token_budget": mode.thinking_token_budget,
            "num_shards": num_shards,
        },
        force=False,
    )

    output_path = shard_path(rec_dir, shard_index, num_shards)
    if force and output_path.exists():
        output_path.unlink()
    if not force:
        repair_jsonl_tail(output_path)
    previous = latest_records_from_file(output_path, "line_index" if stage == 1 else "statement_id") if not force else {}

    target_keys = {
        record["line_index"] if stage == 1 else record["statement_id"]
        for record in targets
    }
    completed = {
        key for key, record in previous.items() if key in target_keys and recovery_succeeded(stage, record)
    }

    if stage == 1:
        wanted = {
            int(record["line_index"])
            for record in targets
            if int(record["line_index"]) % num_shards == shard_index and record["line_index"] not in completed
        }
        review_map = {index: record for index, record in read_jsonl(dataset_paths(dataset, pipeline_cfg)["reviews"])}
        records = [(index, review_map[index]) for index in sorted(wanted) if index in review_map]
        if not records:
            print(f"Stage 1 recovery shard {shard_index}/{num_shards}: nothing to do.", flush=True)
            return
        generate_stage1_records(
            dataset=dataset,
            pipeline_cfg=pipeline_cfg,
            generation_cfg=generation_cfg,
            model_spec=model_spec,
            records=records,
            output_path=output_path,
            run_name=recovery_run_name,
            shard_index=shard_index,
            num_shards=num_shards,
            mode=mode,
            run_kind="recovery",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            batch_size=batch_size,
        )
        return

    assert stage1_run is not None
    if stage == 2:
        candidate_map = {
            candidate["statement_id"]: candidate
            for candidate in stage1_candidates(
                dataset,
                pipeline_cfg,
                stage1_run,
                {"interaction"},
                stage1_recovery_run=stage1_recovery_run,
            )
        }
    else:
        eligible = set(stage_cfg.get("eligible_types", ["user", "interaction"]))
        candidate_map = {
            candidate["statement_id"]: candidate
            for candidate in stage1_candidates(
                dataset,
                pipeline_cfg,
                stage1_run,
                eligible,
                stage1_recovery_run=stage1_recovery_run,
            )
        }

    candidates = []
    for record in targets:
        sid = str(record["statement_id"])
        if sid in completed or not candidate_belongs(sid, shard_index, num_shards):
            continue
        candidate = candidate_map.get(sid)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        print(f"Stage {stage} recovery shard {shard_index}/{num_shards}: nothing to do.", flush=True)
        return

    if stage == 2:
        generate_stage2_records(
            dataset=dataset,
            pipeline_cfg=pipeline_cfg,
            generation_cfg=generation_cfg,
            model_spec=model_spec,
            candidates=candidates,
            output_path=output_path,
            run_name=recovery_run_name,
            stage1_run=stage1_run,
            stage1_recovery_run=stage1_recovery_run,
            shard_index=shard_index,
            num_shards=num_shards,
            mode=mode,
            run_kind="recovery",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            batch_size=batch_size,
        )
    else:
        generate_stage3_records(
            dataset=dataset,
            generation_cfg=generation_cfg,
            model_spec=model_spec,
            candidates=candidates,
            output_path=output_path,
            run_name=recovery_run_name,
            stage1_run=stage1_run,
            stage1_recovery_run=stage1_recovery_run,
            shard_index=shard_index,
            num_shards=num_shards,
            mode=mode,
            run_kind="recovery",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            batch_size=batch_size,
        )
