from __future__ import annotations

import argparse

from evidence_pipeline.dataset import load_pipeline_config
from evidence_pipeline.recovery.merge import merge_recovery_run
from evidence_pipeline.storage import stage1_dir, stage2_dir, stage3_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a targeted recovery pass and create its resolved full results file.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True, help="Primary run being recovered.")
    parser.add_argument("--recovery-run-name", required=True)
    parser.add_argument("--stage1-run", default=None)
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    args = parser.parse_args()

    pipeline_cfg = load_pipeline_config(args.pipeline_config)
    if args.stage == 1:
        base_dir = stage1_dir(args.dataset, pipeline_cfg, args.run_name)
    else:
        if not args.stage1_run:
            parser.error("--stage1-run is required for Stage 2/3 recovery")
        if args.stage == 2:
            base_dir = stage2_dir(
                args.dataset,
                pipeline_cfg,
                args.stage1_run,
                args.run_name,
                stage1_recovery_run=args.stage1_recovery_run,
            )
        else:
            base_dir = stage3_dir(
                args.dataset,
                pipeline_cfg,
                args.stage1_run,
                args.run_name,
                stage1_recovery_run=args.stage1_recovery_run,
            )

    manifest = merge_recovery_run(
        stage=args.stage,
        base_dir=base_dir,
        recovery_run=args.recovery_run_name,
    )
    print(
        f"stage={args.stage} recovery_complete={manifest['complete']} "
        f"targets={manifest['expected_count']} rerun={manifest['record_count']} "
        f"selected_recovery={manifest['selected_recovery_count']} "
        f"unresolved={manifest['unresolved_target_count']}"
    )
    if not manifest["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
