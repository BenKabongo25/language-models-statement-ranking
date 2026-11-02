from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from baselines.registry import DETERMINISTIC, REGISTRY, TRANSFORMER
from baselines.utils import load_yaml


def _profile_for(model: str, requested: str, root: Path) -> str:
    if requested != "auto":
        return requested
    if model in DETERMINISTIC or model == "rand":
        return str(root / "baselines/configs/slurm/cpu.yaml")
    if model in TRANSFORMER or model == "expgcn":
        return str(root / "baselines/configs/slurm/h200.yaml")
    return str(root / "baselines/configs/slurm/a6000.yaml")


def command(dataset: str, model: str, seed: int, profile_path: str, extra_configs: list[str] | None, dry=False):
    root=Path(__file__).resolve().parents[2]
    profile_path=_profile_for(model, profile_path, root)
    profile=load_yaml(profile_path);logs=root/"baselines"/"logs";logs.mkdir(parents=True,exist_ok=True)
    job=f"bl-{model}-{dataset}-s{seed}"[:120]
    cmd=["sbatch","--job-name",job,"--partition",str(profile["partition"]),"--nodes","1","--ntasks","1","--cpus-per-task",str(profile.get("cpus",12)),"--mem",str(profile.get("memory","64G")),"--time",str(profile.get("time","48:00:00"))]
    gpus=int(profile.get("gpus",1))
    if gpus > 0:
        cmd += ["--gres",f"gpu:{gpus}"]
    if profile.get("nodelist"):cmd += ["--nodelist",str(profile["nodelist"])]
    exports={"ALL":"","DATASET":dataset,"MODEL":model,"SEED":str(seed),"PROJECT_ROOT":str(root),"CONDA_ENV":str(profile.get("conda_env","statementrec"))}
    if extra_configs:exports["EXTRA_CONFIGS"]="|".join(extra_configs)
    export_str=",".join(["ALL"]+[f"{k}={v}" for k,v in exports.items() if k!="ALL"])
    cmd += ["--export",export_str,str(root/"baselines"/"slurm"/"run_one.sbatch")]
    print(shlex.join(cmd))
    if not dry: subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser(description="Submit one-job-per-method×dataset×seed baseline grid.")
    p.add_argument("--datasets",nargs="+",default=["Beauty","Toys","Yelp"])
    p.add_argument("--models",nargs="+",default=list(REGISTRY))
    p.add_argument("--seeds",nargs="+",type=int,default=[40,41,42])
    p.add_argument("--profile",default="auto",help="auto or path to a Slurm profile YAML")
    p.add_argument("--extra-config",action="append",default=[]);p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    for d in a.datasets:
        for m in a.models:
            seeds=[0] if m in DETERMINISTIC else a.seeds
            for s in seeds:command(d,m,s,a.profile,a.extra_config,a.dry_run)

if __name__=="__main__":main()
