from __future__ import annotations

import argparse
from collections import Counter

from evidence_pipeline.dataset import load_pipeline_config
from evidence_pipeline.recovery.policy import should_recover
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
    parser = argparse.ArgumentParser(description="Summarize a primary, recovery, or resolved extraction run.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage1-run", default=None)
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--recovery-run", default=None)
    parser.add_argument("--resolved", action="store_true")
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

    statuses = Counter()
    types = Counter()
    statements = 0
    recovery_targets = 0
    records = 0
    for _, record in read_jsonl(path):
        records += 1
        statuses[record.get("status", "unknown")] += 1
        recovery_targets += int(should_recover(args.stage, record))
        if args.stage == 1:
            for statement in record.get("statements", []):
                statements += 1
                types[statement.get("type", "unknown")] += 1

    print(f"run={path}")
    print(f"records={records}")
    print("status:", dict(statuses))
    print(f"recovery_targets={recovery_targets}")
    if args.stage == 1:
        print(f"statements={statements}")
        print("types:", dict(types))


if __name__ == "__main__":
    main()
