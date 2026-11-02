from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import IndexedInteraction, RankingData
from .metrics import metrics_at_k
from .registry import DETERMINISTIC, REGISTRY
from .utils import atomic_json, atomic_yaml, fingerprint_cfg, seed_everything, set_dotted, write_csv


def _device(cfg: dict[str, Any]) -> str:
    req=str(cfg.get("runtime",{}).get("device","auto"))
    if req != "auto": return req
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _evaluate(model, data: RankingData, interactions: list[IndexedInteraction], ks: list[int], save_full=False):
    totals: dict[str,float]={};rows=[];full=[];n=0
    maxk=max(ks)
    for x in interactions:
        cand=data.item_candidates.get(x.item,np.empty(0,dtype=np.int64))
        pos=set(x.statements) & set(map(int,cand))
        if not len(cand) or not pos: continue
        ranked,scores=model.rank(x.user,x.item,cand);n+=1;per={}
        for k in ks:
            m=metrics_at_k(ranked,pos,k);per.update(m)
            for name,val in m.items(): totals[name]=totals.get(name,0.)+val
        canonical_pos=[int(data.idx_to_sid[s]) for s in sorted(pos)]
        top=[int(data.idx_to_sid[s]) for s in ranked[:maxk]]
        row={
            "interaction_id":x.interaction_id,"user_id":data.idx_to_user[x.user],"item_id":data.idx_to_item[x.item],
            "timestamp":x.timestamp,"n_candidates":len(cand),"n_positives":len(pos),
            "positive_statement_ids":":".join(map(str,canonical_pos)),
            f"top{maxk}_statement_ids":":".join(map(str,top)),
        }
        row.update(per);rows.append(row)
        if save_full:
            full.append({"interaction_id":x.interaction_id,"ranked_statement_ids":[int(data.idx_to_sid[s]) for s in ranked],"scores":[float(v) for v in scores]})
    metrics={k:v/n for k,v in totals.items()} if n else {f"ndcg@{k}":float("nan") for k in ks}
    metrics["evaluated_interactions"]=n
    return metrics,rows,full


def _grid(cfg: dict[str,Any]):
    sel=cfg.get("selection",{})
    if not bool(sel.get("enabled",False)) or not sel.get("grid"):
        yield {}
        return
    keys=list(sel["grid"]); vals=[v if isinstance(v,list) else [v] for v in (sel["grid"][k] for k in keys)]
    for combo in itertools.product(*vals): yield dict(zip(keys,combo))


def _apply(cfg, combo):
    out=copy.deepcopy(cfg)
    for k,v in combo.items():set_dotted(out,k,v)
    return out


def run_experiment(cfg: dict[str,Any]) -> dict[str,Any]:
    model_name=str(cfg["model"]["name"]);seed=int(cfg.get("seed",42));seed_everything(seed,bool(cfg.get("runtime",{}).get("deterministic",False)))
    if bool(cfg.get("selection", {}).get("enabled", False)) and not bool(cfg.get("selection", {}).get("allow_per_seed", False)):
        raise ValueError(
            "selection.enabled is intentionally disabled inside final test runs. "
            "Use `python -m baselines.tune ...` to select hyperparameters on validation, "
            "then pass its selected_overrides.yaml as --extra-config to all test seeds. "
            "Set selection.allow_per_seed=true only for debugging/legacy experiments."
        )
    data=RankingData(cfg["dataset"]["root"],str(cfg["dataset"].get("statement_type","match")));split=data.temporal_split(bool(cfg.get("split",{}).get("drop_cold_start_validation",True)))
    cfg.setdefault("dataset",{})["stats"]=data.split_stats(split)
    out=Path(cfg["output"]["dir"]);out.mkdir(parents=True,exist_ok=True);atomic_yaml(out/"config_resolved.yaml",cfg);atomic_json(out/"split_stats.json",data.split_stats(split))
    dev=_device(cfg);ks=[int(k) for k in cfg.get("evaluation",{}).get("k",[5,10])];model_cls=REGISTRY[model_name];selection_records=[];best_cfg=copy.deepcopy(cfg)
    # Deterministic/statistical methods are fitted once on train+validation as requested.
    if model_name not in DETERMINISTIC and model_name != "rand" and split.val_eval and bool(cfg.get("selection",{}).get("enabled",False)):
        metric_name=str(cfg.get("selection",{}).get("metric","ndcg@10"));best=(-float("inf"),None)
        for trial,combo in enumerate(_grid(cfg)):
            tcfg=_apply(cfg,combo);seed_everything(seed)
            model=model_cls(data,tcfg,seed,dev);started=time.time();info=model.fit(split.train,validation=split.val_eval);vm,_,_=_evaluate(model,data,split.val_eval,ks,False);score=float(vm.get(metric_name,float("-inf")))
            rec={"trial":trial,"overrides":combo,"metric":metric_name,"score":score,"validation_metrics":vm,"seconds":time.time()-started};selection_records.append(rec)
            if score>best[0]:best=(score,tcfg)
            del model
            try:
                import torch
                if torch.cuda.is_available():torch.cuda.empty_cache()
            except Exception:pass
        if best[1] is not None:best_cfg=best[1]
    atomic_json(out/"selection.json",selection_records)
    # Final fit includes validation but never test. Baselines may freely use all interactions in this fit set (non-causal), matching their original collaborative setup.
    fit=split.train+split.val;seed_everything(seed);model=model_cls(data,best_cfg,seed,dev);started=time.time();fit_info=model.fit(fit,validation=None);fit_seconds=time.time()-started
    tm,rows,full=_evaluate(model,data,split.test,ks,bool(cfg.get("evaluation",{}).get("save_full_ranking",False)));tm["fit_seconds"]=fit_seconds;tm["seed"]=seed;tm["model"]=model_name;tm["config_fingerprint"]=fingerprint_cfg(best_cfg)
    atomic_json(out/"metrics.json",tm);atomic_json(out/"fit_info.json",fit_info);atomic_yaml(out/"best_config.yaml",best_cfg)
    metric_fields=[f"{m}@{k}" for k in ks for m in ["precision","recall","f1","ndcg","hr"]]
    write_csv(out/"per_interaction.csv",rows,["interaction_id","user_id","item_id","timestamp","n_candidates","n_positives","positive_statement_ids",f"top{max(ks)}_statement_ids",*metric_fields])
    if full:
        with (out/"full_rankings.jsonl").open("w",encoding="utf-8") as f:
            for r in full:f.write(json.dumps(r,separators=(",",":"))+"\n")
    print(json.dumps(tm,indent=2,sort_keys=True));return tm
