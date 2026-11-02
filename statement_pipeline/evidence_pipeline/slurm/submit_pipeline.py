from __future__ import annotations

import argparse
import shlex
import subprocess

from evidence_pipeline.slurm.submit_recovery import build_recovery_submission, submit_recovery
from evidence_pipeline.slurm.submit_stage import build_submission, submit
from evidence_pipeline.utils import load_yaml, project_root, resolve_path


def _job_id(output: str | None) -> str:
    if not output:
        raise RuntimeError("Slurm did not return a job id")
    return output.split(";", 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit Stage 1 primary -> Stage 1 recovery -> Stage 2/3 primary in parallel -> "
            "Stage 2/3 recovery -> final composition."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")

    parser.add_argument("--stage1-model", default="evidence_pipeline/configs/models/qwen38_27b.yaml")
    parser.add_argument("--stage2-model", default="evidence_pipeline/configs/models/qwen35_9b.yaml")
    parser.add_argument("--stage3-model", default="evidence_pipeline/configs/models/qwen35_9b.yaml")
    parser.add_argument("--stage1-recovery-model", default=None)
    parser.add_argument("--stage2-recovery-model", default=None)
    parser.add_argument("--stage3-recovery-model", default=None)

    parser.add_argument("--stage1-profile", default="evidence_pipeline/configs/slurm/h200.yaml")
    parser.add_argument("--stage2-profile", default="evidence_pipeline/configs/slurm/a6000.yaml")
    parser.add_argument("--stage3-profile", default="evidence_pipeline/configs/slurm/a6000.yaml")
    parser.add_argument("--stage1-recovery-profile", default=None)
    parser.add_argument("--stage2-recovery-profile", default=None)
    parser.add_argument("--stage3-recovery-profile", default=None)

    parser.add_argument("--stage1-gpus", type=int, default=1)
    parser.add_argument("--stage2-gpus", type=int, default=1)
    parser.add_argument("--stage3-gpus", type=int, default=1)
    parser.add_argument("--stage1-recovery-gpus", type=int, default=None)
    parser.add_argument("--stage2-recovery-gpus", type=int, default=None)
    parser.add_argument("--stage3-recovery-gpus", type=int, default=None)

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument("--indices-file", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-compose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_root()
    pipeline_cfg = load_yaml(resolve_path(root, args.pipeline_config))
    s1_recovery_model = args.stage1_recovery_model or args.stage1_model
    s2_recovery_model = args.stage2_recovery_model or args.stage2_model
    s3_recovery_model = args.stage3_recovery_model or args.stage3_model
    s1_recovery_profile = args.stage1_recovery_profile or args.stage1_profile
    s2_recovery_profile = args.stage2_recovery_profile or args.stage2_profile
    s3_recovery_profile = args.stage3_recovery_profile or args.stage3_profile

    # Stage 1 primary.
    s1 = build_submission(
        stage=1,
        dataset=args.dataset,
        model_config=args.stage1_model,
        profile=args.stage1_profile,
        pipeline_config=args.pipeline_config,
        gpus=args.stage1_gpus,
        limit=args.limit,
        indices_file=args.indices_file,
        force=args.force,
        parsable=True,
    )
    print(f"Stage 1 primary run: {s1.run_name}")
    if args.dry_run:
        print(shlex.join(s1.command))
        dep_s1 = "afterok:<stage1-primary-jobid>"
    else:
        dep_s1 = f"afterok:{_job_id(submit(s1))}"

    # Stage 1 targeted recovery.
    r1 = build_recovery_submission(
        stage=1,
        dataset=args.dataset,
        run_name=s1.run_name,
        model_config=s1_recovery_model,
        profile=s1_recovery_profile,
        pipeline_config=args.pipeline_config,
        gpus=args.stage1_recovery_gpus or args.stage1_gpus,
        force=args.force,
        dependency=dep_s1,
        parsable=True,
    )
    print(f"Stage 1 recovery run: {r1.recovery_run_name}")
    if args.dry_run:
        print(shlex.join(r1.command))
        dep_r1 = "afterok:<stage1-recovery-jobid>"
    else:
        dep_r1 = f"afterok:{_job_id(submit_recovery(r1))}"

    # Stage 2 and 3 primary consume the resolved Stage 1 recovery output.
    s2 = build_submission(
        stage=2,
        dataset=args.dataset,
        model_config=args.stage2_model,
        profile=args.stage2_profile,
        pipeline_config=args.pipeline_config,
        stage1_run=s1.run_name,
        stage1_recovery_run=r1.recovery_run_name,
        gpus=args.stage2_gpus,
        force=args.force,
        dependency=dep_r1,
        parsable=True,
    )
    s3 = build_submission(
        stage=3,
        dataset=args.dataset,
        model_config=args.stage3_model,
        profile=args.stage3_profile,
        pipeline_config=args.pipeline_config,
        stage1_run=s1.run_name,
        stage1_recovery_run=r1.recovery_run_name,
        gpus=args.stage3_gpus,
        force=args.force,
        dependency=dep_r1,
        parsable=True,
    )
    print(f"Stage 2 primary run: {s2.run_name}")
    print(f"Stage 3 primary run: {s3.run_name}")
    if args.dry_run:
        print(shlex.join(s2.command))
        print(shlex.join(s3.command))
        dep_s2 = "afterok:<stage2-primary-jobid>"
        dep_s3 = "afterok:<stage3-primary-jobid>"
    else:
        dep_s2 = f"afterok:{_job_id(submit(s2))}"
        dep_s3 = f"afterok:{_job_id(submit(s3))}"

    # Targeted recovery for Stage 2 and Stage 3.
    r2 = build_recovery_submission(
        stage=2,
        dataset=args.dataset,
        run_name=s2.run_name,
        model_config=s2_recovery_model,
        profile=s2_recovery_profile,
        pipeline_config=args.pipeline_config,
        stage1_run=s1.run_name,
        stage1_recovery_run=r1.recovery_run_name,
        gpus=args.stage2_recovery_gpus or args.stage2_gpus,
        force=args.force,
        dependency=dep_s2,
        parsable=True,
    )
    r3 = build_recovery_submission(
        stage=3,
        dataset=args.dataset,
        run_name=s3.run_name,
        model_config=s3_recovery_model,
        profile=s3_recovery_profile,
        pipeline_config=args.pipeline_config,
        stage1_run=s1.run_name,
        stage1_recovery_run=r1.recovery_run_name,
        gpus=args.stage3_recovery_gpus or args.stage3_gpus,
        force=args.force,
        dependency=dep_s3,
        parsable=True,
    )
    print(f"Stage 2 recovery run: {r2.recovery_run_name}")
    print(f"Stage 3 recovery run: {r3.recovery_run_name}")

    if args.dry_run:
        print(shlex.join(r2.command))
        print(shlex.join(r3.command))
        dep_recovery = "afterok:<stage2-recovery-jobid>:<stage3-recovery-jobid>"
    else:
        r2_job = _job_id(submit_recovery(r2))
        r3_job = _job_id(submit_recovery(r3))
        dep_recovery = f"afterok:{r2_job}:{r3_job}"

    # Composition is CPU-only and consumes the resolved outputs of all three stages.
    compose_profile = load_yaml(resolve_path(root, args.stage2_profile))
    compose_cfg = pipeline_cfg.get("compose", {}).get("slurm", {})
    log_root = root / "evidence_pipeline" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    export_values = {
        "PROJECT_ROOT": str(root),
        "PIPELINE_CONFIG": str(resolve_path(root, args.pipeline_config)),
        "DATASET": args.dataset,
        "STAGE1_RUN": s1.run_name,
        "STAGE1_RECOVERY_RUN": r1.recovery_run_name,
        "STAGE2_RUN": s2.run_name,
        "STAGE2_RECOVERY_RUN": r2.recovery_run_name,
        "STAGE3_RUN": s3.run_name,
        "STAGE3_RECOVERY_RUN": r3.recovery_run_name,
        "CONDA_ENV": str(compose_profile.get("conda_env", "statementrec")),
        "STRICT_COMPOSE": "1" if args.strict_compose else "0",
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    compose_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=ev-compose-{args.dataset}",
        f"--partition={compose_profile['partition']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={int(compose_cfg.get('cpus', 2))}",
        f"--mem={compose_cfg.get('memory', '8G')}",
        f"--time={compose_cfg.get('time', '12:00:00')}",
        f"--dependency={dep_recovery}",
        f"--output={log_root}/%x-%j.out",
        f"--error={log_root}/%x-%j.err",
        f"--export={export}",
        str(root / "evidence_pipeline" / "slurm" / "compose.sbatch"),
    ]
    if args.dry_run:
        print(shlex.join(compose_cmd))
    else:
        completed = subprocess.run(compose_cmd, check=True, text=True, capture_output=True)
        print(f"Compose job: {completed.stdout.strip()}")


if __name__ == "__main__":
    main()
