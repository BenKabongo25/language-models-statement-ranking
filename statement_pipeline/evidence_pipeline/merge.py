from __future__ import annotations

import argparse

from evidence_pipeline.dataset import load_pipeline_config
from evidence_pipeline.storage import (
    load_manifest,
    merge_run,
    stage1_candidates,
    stage1_dir,
    stage1_expected_indices,
    stage2_dir,
    stage3_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge primary stage shards and mark coverage complete.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage1-run", default=None)
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    args = parser.parse_args()

    pipeline_cfg = load_pipeline_config(args.pipeline_config)
    if args.stage == 1:
        run_dir = stage1_dir(args.dataset, pipeline_cfg, args.run_name)
        manifest = load_manifest(run_dir)
        expected = stage1_expected_indices(args.dataset, pipeline_cfg, manifest.get("selection", {"mode": "all"}))
        merged = merge_run(
            run_dir=run_dir,
            expected_keys=expected,
            key_field="line_index",
            sort_key=lambda record: int(record["line_index"]),
        )
    elif args.stage == 2:
        if not args.stage1_run:
            parser.error("--stage1-run is required for Stage 2")
        run_dir = stage2_dir(
            args.dataset,
            pipeline_cfg,
            args.stage1_run,
            args.run_name,
            stage1_recovery_run=args.stage1_recovery_run,
        )
        expected = [
            candidate["statement_id"]
            for candidate in stage1_candidates(
                args.dataset,
                pipeline_cfg,
                args.stage1_run,
                {"interaction"},
                stage1_recovery_run=args.stage1_recovery_run,
            )
        ]
        merged = merge_run(
            run_dir=run_dir,
            expected_keys=expected,
            key_field="statement_id",
            sort_key=lambda record: (int(record["line_index"]), int(record["statement_index"])),
        )
    else:
        if not args.stage1_run:
            parser.error("--stage1-run is required for Stage 3")
        run_dir = stage3_dir(
            args.dataset,
            pipeline_cfg,
            args.stage1_run,
            args.run_name,
            stage1_recovery_run=args.stage1_recovery_run,
        )
        manifest = load_manifest(run_dir)
        eligible = set(
            manifest.get(
                "eligible_types",
                pipeline_cfg["stages"]["stage3"].get("eligible_types", ["user", "interaction"]),
            )
        )
        expected = [
            candidate["statement_id"]
            for candidate in stage1_candidates(
                args.dataset,
                pipeline_cfg,
                args.stage1_run,
                eligible,
                stage1_recovery_run=args.stage1_recovery_run,
            )
        ]
        merged = merge_run(
            run_dir=run_dir,
            expected_keys=expected,
            key_field="statement_id",
            sort_key=lambda record: (int(record["line_index"]), int(record["statement_index"])),
        )

    print(
        f"stage={args.stage} complete={merged['complete']} records={merged['record_count']}/{merged['expected_count']} "
        f"ok={merged['ok_count']} partial={merged['partial_count']} error={merged['error_count']}"
    )
    if not merged["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
