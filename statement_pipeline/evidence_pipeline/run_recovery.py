from __future__ import annotations

import argparse

from evidence_pipeline.dataset import load_pipeline_config
from evidence_pipeline.model_runtime import assert_model_access, load_model_spec, resolve_generation_mode
from evidence_pipeline.recovery.run import run_recovery
from evidence_pipeline.storage import recovery_run_name
from evidence_pipeline.utils import project_root, resolve_path, sanitize_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one worker shard of a targeted stage recovery pass.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--run-name", required=True, help="Primary run to recover.")
    parser.add_argument("--recovery-run-name", default=None)
    parser.add_argument("--stage1-run", default=None, help="Required for Stage 2/3 recovery.")
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--thinking-token-budget", type=int, default=None)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage in {2, 3} and not args.stage1_run:
        parser.error("--stage1-run is required for Stage 2/3 recovery")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard configuration")

    root = project_root()
    pipeline_cfg = load_pipeline_config(resolve_path(root, args.pipeline_config))
    model_spec = load_model_spec(resolve_path(root, args.model_config))
    assert_model_access(model_spec)
    stage_cfg = pipeline_cfg["stages"][f"stage{args.stage}"]
    recovery_cfg = stage_cfg.get("recovery", {})
    requested_thinking = bool(recovery_cfg.get("thinking", True)) if args.thinking is None else args.thinking
    budget = recovery_cfg.get("thinking_token_budget") if args.thinking_token_budget is None else args.thinking_token_budget
    mode = resolve_generation_mode(
        model_spec,
        thinking=requested_thinking,
        thinking_token_budget=None if budget is None else int(budget),
    )
    if args.recovery_run_name:
        resolved_recovery_run = sanitize_name(args.recovery_run_name)
    else:
        resolved_recovery_run = recovery_run_name(
            model_spec.name,
            str(recovery_cfg.get("revision", "recovery-v1")),
            thinking=mode.thinking_enabled,
        )

    overrides = {
        "thinking": requested_thinking,
        "thinking_token_budget": budget,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "retries": args.retries,
    }
    run_recovery(
        stage=args.stage,
        dataset=args.dataset,
        pipeline_cfg=pipeline_cfg,
        stage_cfg=stage_cfg,
        model_spec=model_spec,
        run_name=sanitize_name(args.run_name),
        recovery_run_name=resolved_recovery_run,
        stage1_run=None if args.stage == 1 else sanitize_name(str(args.stage1_run)),
        stage1_recovery_run=None if not args.stage1_recovery_run else sanitize_name(args.stage1_recovery_run),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        force=args.force,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.batch_size,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
