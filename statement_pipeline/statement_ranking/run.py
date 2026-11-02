from __future__ import annotations

import argparse
from pathlib import Path

from statement_ranking.config import load_config
from statement_ranking.pipeline import STAGES, build_plan, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build temporal statement-ranking datasets using only upstream FINAL artifacts."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="statement_ranking/configs/pipeline.yaml")
    parser.add_argument(
        "--dataset-root", default=None,
        help="Optional explicit processed dataset root, e.g. /.../processed/Beauty.",
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Optional explicit statement_ranking output root.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stage", choices=["all", *STAGES], default=None)
    mode.add_argument("--from-stage", choices=list(STAGES), default=None)
    parser.add_argument("--to-stage", choices=list(STAGES), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    plan = build_plan(stage=args.stage, from_stage=args.from_stage, to_stage=args.to_stage, force=args.force)
    run_pipeline(
        dataset=args.dataset, cfg=cfg, plan=plan,
        dataset_root=args.dataset_root, output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
