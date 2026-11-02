from __future__ import annotations

import argparse
from pathlib import Path

from clustering_pipeline.config import load_config
from clustering_pipeline.pipeline import STAGES, build_plan, run_pipeline
from clustering_pipeline.storage import TYPES


def infer_output_root(input_file: Path) -> Path:
    if input_file.parent.name == "final" and input_file.parent.parent.name == "evidence_pipeline":
        return input_file.parent.parent.parent / "clustering_pipeline"
    return input_file.parent / "clustering_pipeline"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Type-aware USER/ITEM/MATCH clustering with resumable stage orchestration."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--config", default="clustering_pipeline/configs/pipeline.yaml")
    parser.add_argument("--types", nargs="+", choices=list(TYPES), default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stage",
        choices=["all", *STAGES, "doctor"],
        default=None,
        help="Run through one stage (dependencies are reused/validated). 'all' runs the complete pipeline.",
    )
    mode.add_argument(
        "--from-stage",
        choices=list(STAGES),
        default=None,
        help="Force this stage and every downstream stage, while reusing upstream artifacts.",
    )
    parser.add_argument(
        "--to-stage",
        choices=list(STAGES),
        default=None,
        help="Optional last stage when using --from-stage, or last stage for a normal resumable run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force the selected stage/range. Usually --from-stage is clearer for downstream recomputation.",
    )
    parser.add_argument(
        "--audit-ranks",
        choices=["auto", "available", "recompute"],
        default=None,
    )
    parser.add_argument(
        "--fresh-cluster",
        action="store_true",
        help="Ignore clustering checkpoints/legacy cluster state and rebuild clustering from singleton statements.",
    )
    args = parser.parse_args()

    input_file = Path(args.input).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else infer_output_root(input_file)
    )
    cfg = load_config(args.config)
    if args.device:
        cfg.setdefault("runtime", {})["device"] = args.device
    if args.fresh_cluster:
        cfg.setdefault("cluster", {})["resume_previous_state"] = False
    types = tuple(args.types or cfg.get("types", TYPES))

    if args.stage == "doctor":
        from clustering_pipeline.doctor import run_doctor

        run_doctor(input_file=input_file, output_root=output_root, cfg=cfg)
        return

    # No mode means the normal resumable full pipeline: every completed
    # compatible artifact is reused and only missing/stale stages run.
    plan = build_plan(
        stage=args.stage if args.from_stage is None else None,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        force=args.force,
    )
    run_pipeline(
        input_file=input_file,
        output_root=output_root,
        cfg=cfg,
        types=types,
        plan=plan,
        audit_ranks=args.audit_ranks,
    )


if __name__ == "__main__":
    main()
