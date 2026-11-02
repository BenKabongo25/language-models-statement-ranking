from __future__ import annotations

import argparse

from evidence_pipeline.dataset import load_indices, load_pipeline_config
from evidence_pipeline.model_runtime import assert_model_access, load_model_spec
from evidence_pipeline.stages.stage1 import run_stage1
from evidence_pipeline.stages.stage2 import run_stage2
from evidence_pipeline.stages.stage3 import run_stage3
from evidence_pipeline.storage import model_run_name, stage1_run_name
from evidence_pipeline.utils import project_root, resolve_path, sanitize_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one worker shard of the staged evidence pipeline.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage1-run", default=None, help="Required by stages 2 and 3.")
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument("--indices-file", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.stage in {2, 3} and not args.stage1_run:
        parser.error("--stage1-run is required for stages 2 and 3")
    if args.stage != 1 and (args.limit is not None or args.indices_file):
        parser.error("Selection belongs to Stage 1. Stages 2 and 3 inherit the Stage 1 selection.")
    if args.stage == 1 and args.stage1_recovery_run:
        parser.error("--stage1-recovery-run applies only to Stage 2/3")

    root = project_root()
    pipeline_path = resolve_path(root, args.pipeline_config)
    model_path = resolve_path(root, args.model_config)
    pipeline_cfg = load_pipeline_config(pipeline_path)
    model_spec = load_model_spec(model_path)
    assert_model_access(model_spec)
    stage_cfg = pipeline_cfg["stages"][f"stage{args.stage}"]
    indices = load_indices(resolve_path(root, args.indices_file)) if args.indices_file else None
    if args.run_name:
        run_name = sanitize_name(args.run_name)
    elif args.stage == 1:
        run_name = stage1_run_name(model_spec.name, stage_cfg["prompt_revision"], limit=args.limit, indices=indices)
    else:
        run_name = model_run_name(model_spec.name, stage_cfg["prompt_revision"])

    common = dict(
        dataset=args.dataset,
        pipeline_cfg=pipeline_cfg,
        stage_cfg=stage_cfg,
        model_spec=model_spec,
        run_name=run_name,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        force=args.force,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.batch_size,
    )
    if args.stage == 1:
        run_stage1(**common, limit=args.limit, indices=indices)
    elif args.stage == 2:
        run_stage2(
            **common,
            stage1_run=str(args.stage1_run),
            stage1_recovery_run=args.stage1_recovery_run,
        )
    else:
        run_stage3(
            **common,
            stage1_run=str(args.stage1_run),
            stage1_recovery_run=args.stage1_recovery_run,
        )


if __name__ == "__main__":
    main()
