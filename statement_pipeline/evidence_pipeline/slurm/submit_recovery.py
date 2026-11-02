from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass

from evidence_pipeline.dataset import load_pipeline_config
from evidence_pipeline.model_runtime import assert_model_access, load_model_spec, resolve_generation_mode
from evidence_pipeline.storage import recovery_run_name
from evidence_pipeline.utils import load_yaml, project_root, resolve_path, sanitize_name


@dataclass(frozen=True)
class RecoverySubmission:
    command: list[str]
    recovery_run_name: str


def build_recovery_submission(
    *,
    stage: int,
    dataset: str,
    run_name: str,
    model_config: str,
    profile: str,
    pipeline_config: str = "evidence_pipeline/configs/pipeline.yaml",
    stage1_run: str | None = None,
    stage1_recovery_run: str | None = None,
    recovery_run_name_override: str | None = None,
    gpus: int | None = None,
    nodelist: str | None = None,
    batch_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    retries: int | None = None,
    thinking: bool | None = None,
    thinking_token_budget: int | None = None,
    force: bool = False,
    no_prefetch: bool = False,
    memory: str | None = None,
    time: str | None = None,
    dependency: str | None = None,
    parsable: bool = False,
) -> RecoverySubmission:
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    if stage in {2, 3} and not stage1_run:
        raise ValueError("stage1_run is required for Stage 2/3 recovery")
    if stage == 1 and stage1_recovery_run:
        raise ValueError("stage1_recovery_run only applies to Stage 2/3 recovery")

    root = project_root()
    pipeline_path = resolve_path(root, pipeline_config)
    model_path = resolve_path(root, model_config)
    profile_path = resolve_path(root, profile)
    pipeline_cfg = load_pipeline_config(pipeline_path)
    model_spec = load_model_spec(model_path)
    assert_model_access(model_spec)
    profile_cfg = load_yaml(profile_path)
    stage_cfg = pipeline_cfg["stages"][f"stage{stage}"]
    recovery_cfg = stage_cfg.get("recovery", {})

    requested_thinking = bool(recovery_cfg.get("thinking", True)) if thinking is None else bool(thinking)
    budget = recovery_cfg.get("thinking_token_budget") if thinking_token_budget is None else thinking_token_budget
    mode = resolve_generation_mode(
        model_spec,
        thinking=requested_thinking,
        thinking_token_budget=None if budget is None else int(budget),
    )
    if recovery_run_name_override:
        resolved_recovery_run = sanitize_name(recovery_run_name_override)
    else:
        resolved_recovery_run = recovery_run_name(
            model_spec.name,
            str(recovery_cfg.get("revision", "recovery-v1")),
            thinking=mode.thinking_enabled,
        )

    total_gpus = int(gpus if gpus is not None else profile_cfg.get("default_gpus", 1))
    max_gpus = int(profile_cfg.get("max_gpus", total_gpus))
    gpus_per_worker = int(profile_cfg.get("gpus_per_worker", 1))
    tp = int(profile_cfg.get("tensor_parallel_size", gpus_per_worker))
    if total_gpus < 1 or total_gpus > max_gpus:
        raise ValueError(f"GPU count must be between 1 and {max_gpus} for profile {profile_cfg.get('profile_name')}")
    if total_gpus % gpus_per_worker != 0:
        raise ValueError("Total GPUs must be divisible by gpus_per_worker")
    if tp != gpus_per_worker:
        raise ValueError("tensor_parallel_size must equal gpus_per_worker")
    workers = total_gpus // gpus_per_worker
    cpus = int(profile_cfg.get("cpus_per_worker", 8))

    log_root = root / "evidence_pipeline" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    mem = memory or f"{int(profile_cfg.get('memory_per_gpu_gb', 52)) * total_gpus}G"

    command = ["sbatch"]
    if parsable:
        command.append("--parsable")
    command.extend(
        [
            f"--job-name=ev-r{stage}-{dataset}",
            f"--partition={profile_cfg['partition']}",
            "--nodes=1",
            f"--ntasks={workers}",
            f"--cpus-per-task={cpus}",
            f"--gres=gpu:{profile_cfg['gpu_type']}:{total_gpus}",
            f"--mem={mem}",
            f"--time={time or profile_cfg.get('time', '48:00:00')}",
            f"--output={log_root}/%x-%j.out",
            f"--error={log_root}/%x-%j.err",
        ]
    )
    selected_node = nodelist if nodelist is not None else profile_cfg.get("nodelist")
    if selected_node:
        command.append(f"--nodelist={selected_node}")
    if profile_cfg.get("account"):
        command.append(f"--account={profile_cfg['account']}")
    if profile_cfg.get("qos"):
        command.append(f"--qos={profile_cfg['qos']}")
    if profile_cfg.get("exclusive"):
        command.append("--exclusive")
    if dependency:
        command.append(f"--dependency={dependency}")

    env_values = {
        "PROJECT_ROOT": str(root),
        "PIPELINE_CONFIG": str(pipeline_path),
        "MODEL_CONFIG": str(model_path),
        "STAGE": str(stage),
        "DATASET": dataset,
        "RUN_NAME": sanitize_name(run_name),
        "RECOVERY_RUN_NAME": resolved_recovery_run,
        "STAGE1_RUN": stage1_run or "",
        "STAGE1_RECOVERY_RUN": stage1_recovery_run or "",
        "CONDA_ENV": str(profile_cfg.get("conda_env", "statementrec")),
        "TOTAL_GPUS": str(total_gpus),
        "NUM_WORKERS": str(workers),
        "GPUS_PER_WORKER": str(gpus_per_worker),
        "TENSOR_PARALLEL_SIZE": str(tp),
        "CPUS_PER_WORKER": str(cpus),
        "GPU_BIND": str(profile_cfg.get("gpu_bind") or ""),
        "BATCH_SIZE": "" if batch_size is None else str(batch_size),
        "GPU_MEMORY_UTILIZATION": "" if gpu_memory_utilization is None else str(gpu_memory_utilization),
        "MAX_MODEL_LEN": "" if max_model_len is None else str(max_model_len),
        "MAX_NUM_SEQS": "" if max_num_seqs is None else str(max_num_seqs),
        "RECOVERY_TEMPERATURE": "" if temperature is None else str(temperature),
        "RECOVERY_TOP_P": "" if top_p is None else str(top_p),
        "RECOVERY_TOP_K": "" if top_k is None else str(top_k),
        "RECOVERY_MAX_TOKENS": "" if max_tokens is None else str(max_tokens),
        "RECOVERY_RETRIES": "" if retries is None else str(retries),
        "RECOVERY_THINKING": "1" if requested_thinking else "0",
        "RECOVERY_THINKING_BUDGET": "" if budget is None else str(budget),
        "FORCE": "1" if force else "0",
        "PREFETCH_MODEL": "0" if no_prefetch else ("1" if profile_cfg.get("prefetch_model", True) else "0"),
        "DISABLE_FLASHINFER_SAMPLER": "1" if profile_cfg.get("disable_flashinfer_sampler", True) else "0",
        "MODEL_NAME": model_spec.name,
        "MODEL_FAMILY": model_spec.family,
        "HF_TOKEN_ENV": model_spec.hf_token_env,
        "REQUIRES_HF_TOKEN": "1" if model_spec.requires_hf_token else "0",
        "LOG_ROOT": str(log_root),
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env_values.items())
    command.extend([f"--export={export}", str(root / "evidence_pipeline" / "slurm" / "run_recovery.sbatch")])
    return RecoverySubmission(command=command, recovery_run_name=resolved_recovery_run)


def submit_recovery(submission: RecoverySubmission, *, dry_run: bool = False) -> str | None:
    print(shlex.join(submission.command))
    if dry_run:
        return None
    completed = subprocess.run(submission.command, check=True, text=True, capture_output=True)
    output = completed.stdout.strip()
    if output:
        print(output)
    return output.split(";", 1)[0] if output else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a targeted recovery pass for one completed primary stage.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True, help="Primary run to recover.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    parser.add_argument("--stage1-run", default=None)
    parser.add_argument("--stage1-recovery-run", default=None)
    parser.add_argument("--recovery-run-name", dest="recovery_run_name_override", default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--nodelist", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--thinking-token-budget", type=int, default=None)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--thinking", dest="thinking", action="store_true")
    thinking_group.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--memory", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--dependency", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    kwargs = vars(args).copy()
    dry_run = kwargs.pop("dry_run")
    submission = build_recovery_submission(**kwargs, parsable=False)
    print(f"recovery_run_name={submission.recovery_run_name}")
    submit_recovery(submission, dry_run=dry_run)


if __name__ == "__main__":
    main()
