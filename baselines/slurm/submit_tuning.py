from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from baselines.registry import DETERMINISTIC, REGISTRY
from baselines.utils import load_yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Submit one hyperparameter-tuning job per learned model x dataset.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--search-config", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42])
    p.add_argument("--profile", default="baselines/configs/slurm/a6000.yaml")
    p.add_argument("--extra-config", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    profile = load_yaml(a.profile)
    root = Path(__file__).resolve().parents[2]
    (root / "baselines" / "logs").mkdir(parents=True, exist_ok=True)
    for dataset in a.datasets:
        for model in a.models:
            if model not in REGISTRY:
                raise ValueError(f"Unknown model: {model}")
            if model in DETERMINISTIC or model == "rand":
                raise ValueError(f"Tuning is not defined for {model}")
            job = f"tune-{model}-{dataset}"[:120]
            cmd = [
                "sbatch", "--job-name", job,
                "--partition", str(profile["partition"]),
                "--nodes", "1", "--ntasks", "1",
                "--cpus-per-task", str(profile.get("cpus", 12)),
                "--gres", f"gpu:{int(profile.get('gpus', 1))}",
                "--mem", str(profile.get("memory", "64G")),
                "--time", str(profile.get("time", "48:00:00")),
            ]
            if profile.get("nodelist"):
                cmd += ["--nodelist", str(profile["nodelist"])]
            exports = {
                "DATASET": dataset,
                "MODEL": model,
                "SEARCH_CONFIG": a.search_config,
                "TUNE_SEEDS": " ".join(map(str, a.seeds)),
                "PROJECT_ROOT": str(root),
                "CONDA_ENV": str(profile.get("conda_env", "statementrec")),
            }
            if a.extra_config:
                exports["EXTRA_CONFIG"] = a.extra_config
            export_str = ",".join(["ALL"] + [f"{k}={v}" for k, v in exports.items()])
            cmd += ["--export", export_str, str(root / "baselines" / "slurm" / "tune_one.sbatch")]
            print(shlex.join(cmd))
            if not a.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
