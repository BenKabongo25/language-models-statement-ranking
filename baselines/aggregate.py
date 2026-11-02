from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .registry import DETERMINISTIC
from .utils import atomic_json, write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--model",required=True);p.add_argument("--root",default="data/processed/{dataset}/baselines");a=p.parse_args();root=Path(a.root.format(dataset=a.dataset))/a.model
    runs=[]
    for path in sorted(root.glob("seed_*/metrics.json")):
        d=json.loads(path.read_text());d["path"]=str(path);runs.append(d)
    if not runs:raise SystemExit(f"No metrics found under {root}")
    metric_names=sorted(k for k,v in runs[0].items() if isinstance(v,(int,float)) and "@" in k)
    agg={"dataset":a.dataset,"model":a.model,"runs":len(runs),"seeds":[r.get("seed") for r in runs],"metrics":{}}
    for m in metric_names:
        vals=np.asarray([float(r[m]) for r in runs],dtype=float);agg["metrics"][m]={"mean":float(np.mean(vals)),"std":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,"values":vals.tolist()}
    atomic_json(root/"aggregate.json",agg)
    rows=[{"metric":m,**agg["metrics"][m]} for m in metric_names];write_csv(root/"aggregate.csv",rows,["metric","mean","std","values"]);print(json.dumps(agg,indent=2))

if __name__=="__main__":main()
