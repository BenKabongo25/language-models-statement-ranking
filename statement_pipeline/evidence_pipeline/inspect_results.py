from __future__ import annotations

import argparse
import json

from evidence_pipeline.dataset import load_pipeline_config
from evidence_pipeline.storage import (
    merged_path,
    recovery_dir,
    resolved_path,
    stage1_dir,
    stage2_dir,
    stage3_dir,
)
from evidence_pipeline.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect primary, recovery, or resolved records from one staged run.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage1-run", default=None)
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--recovery-run", default=None)
    parser.add_argument("--resolved", action="store_true", help="With --recovery-run, inspect resolved_results.jsonl.")
    parser.add_argument("--status", default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_pipeline_config(args.pipeline_config)

    if args.stage == 1:
        run_dir = stage1_dir(args.dataset, cfg, args.run_name)
    elif args.stage == 2:
        if not args.stage1_run:
            parser.error("--stage1-run required")
        run_dir = stage2_dir(
            args.dataset,
            cfg,
            args.stage1_run,
            args.run_name,
            stage1_recovery_run=args.stage1_recovery_run,
        )
    else:
        if not args.stage1_run:
            parser.error("--stage1-run required")
        run_dir = stage3_dir(
            args.dataset,
            cfg,
            args.stage1_run,
            args.run_name,
            stage1_recovery_run=args.stage1_recovery_run,
        )

    if args.recovery_run:
        rec_dir = recovery_dir(run_dir, args.recovery_run)
        path = resolved_path(rec_dir) if args.resolved else merged_path(rec_dir)
    else:
        path = merged_path(run_dir)

    shown = 0
    for _, record in read_jsonl(path):
        if args.status and record.get("status") != args.status:
            continue
        print(json.dumps(record, ensure_ascii=False, indent=2))
        shown += 1
        if shown >= args.n:
            break


if __name__ == "__main__":
    main()
