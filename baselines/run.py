from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_merged
from .registry import DETERMINISTIC, REGISTRY
from .runner import run_experiment


def main():
    p=argparse.ArgumentParser(description="Run one item-level statement-ranking baseline experiment.")
    p.add_argument("--dataset",required=True);p.add_argument("--model",choices=sorted(REGISTRY),required=True);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--base-config",default="baselines/configs/base.yaml");p.add_argument("--model-config",default=None);p.add_argument("--extra-config",action="append",default=[]);p.add_argument("--set",action="append",default=[])
    a=p.parse_args();model_cfg=a.model_config or f"baselines/configs/models/{a.model}.yaml";cfg=load_merged([a.base_config,model_cfg,*a.extra_config],a.set);cfg["seed"]=a.seed;cfg["dataset"]["name"]=a.dataset
    cfg["dataset"]["root"]=str(cfg["dataset"]["root"]).format(dataset=a.dataset)
    base=Path(str(cfg["output"]["root"]).format(dataset=a.dataset));cfg["output"]["dir"]=str(base/a.model/f"seed_{a.seed}")
    run_experiment(cfg)

if __name__=="__main__":main()
