from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_merged
from .data import RankingData
from .registry import DETERMINISTIC, REGISTRY
from .runner import _device, _evaluate
from .utils import atomic_json, atomic_yaml, seed_everything, set_dotted


def _grid(grid: dict[str, Any]):
    if not grid:
        yield {}
        return
    keys = list(grid)
    values = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _apply(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        set_dotted(out, key, value)
    # Disable nested selection inside each trial.  Tuning is coordinated here.
    out.setdefault("selection", {})["enabled"] = False
    return out


def _nested_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        set_dotted(out, key, value)
    return out


def tune(cfg: dict[str, Any], seeds: list[int], output_dir: Path) -> dict[str, Any]:
    model_name = str(cfg["model"]["name"])
    if model_name in DETERMINISTIC or model_name == "rand":
        raise ValueError(f"{model_name} has no learned hyperparameters to tune with this runner")

    data = RankingData(cfg["dataset"]["root"], str(cfg["dataset"].get("statement_type", "match")))
    split = data.temporal_split(bool(cfg.get("split", {}).get("drop_cold_start_validation", True)))
    if not split.val_eval:
        raise RuntimeError("No evaluable validation interactions after cold-item filtering")

    selection = cfg.get("selection", {})
    grid = selection.get("grid", {})
    if not grid:
        raise ValueError("selection.grid is empty; pass a search regime YAML")
    metric_name = str(selection.get("metric", "ndcg@10"))
    ks = [int(k) for k in cfg.get("evaluation", {}).get("k", [5, 10])]
    model_cls = REGISTRY[model_name]
    dev = _device(cfg)

    records: list[dict[str, Any]] = []
    best_key = None
    best_cfg = None
    best_overrides = None

    for trial, overrides in enumerate(_grid(grid)):
        trial_cfg = _apply(cfg, overrides)
        per_seed = []
        started = time.time()
        for seed in seeds:
            seed_everything(seed, bool(cfg.get("runtime", {}).get("deterministic", False)))
            model = model_cls(data, trial_cfg, seed, dev)
            fit_info = model.fit(split.train, validation=split.val_eval)
            metrics, _, _ = _evaluate(model, data, split.val_eval, ks, False)
            per_seed.append({
                "seed": seed,
                "metric": float(metrics.get(metric_name, float("nan"))),
                "metrics": metrics,
                "fit_info_summary": {
                    "eligible_interactions": fit_info.get("eligible_interactions") if isinstance(fit_info, dict) else None,
                },
            })
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        vals = np.asarray([r["metric"] for r in per_seed], dtype=np.float64)
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        record = {
            "trial": trial,
            "overrides": overrides,
            "metric": metric_name,
            "mean": mean,
            "std": std,
            "seeds": per_seed,
            "seconds": time.time() - started,
        }
        records.append(record)
        # Maximize mean validation metric; on exact ties prefer the lower std, then
        # the earlier grid entry for deterministic reproducibility.
        key = (mean, -std, -trial)
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = trial_cfg
            best_overrides = overrides
        print(json.dumps({"trial": trial, "mean": mean, "std": std, "overrides": overrides}, sort_keys=True), flush=True)

    assert best_cfg is not None and best_overrides is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": cfg["dataset"].get("name"),
        "model": model_name,
        "metric": metric_name,
        "seeds": seeds,
        "split_stats": data.split_stats(split),
        "best_overrides": best_overrides,
        "best_mean": best_key[0],
        "best_std": -best_key[1],
        "trials": records,
    }
    atomic_json(output_dir / "tuning_results.json", summary)
    atomic_yaml(output_dir / "selected_overrides.yaml", _nested_overrides(best_overrides))
    atomic_yaml(output_dir / "selected_full_config.yaml", best_cfg)
    print(json.dumps({k: summary[k] for k in ["dataset", "model", "metric", "best_overrides", "best_mean", "best_std"]}, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Tune one learned baseline on the temporal validation interaction(s) and freeze hyperparameters before test runs.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=sorted(REGISTRY), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42])
    p.add_argument("--base-config", default="baselines/configs/base.yaml")
    p.add_argument("--model-config", default=None)
    p.add_argument("--search-config", required=True)
    p.add_argument("--extra-config", action="append", default=[])
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    model_cfg = args.model_config or f"baselines/configs/models/{args.model}.yaml"
    cfg = load_merged([args.base_config, model_cfg, *args.extra_config, args.search_config], args.set)
    cfg["dataset"]["name"] = args.dataset
    cfg["dataset"]["root"] = str(cfg["dataset"]["root"]).format(dataset=args.dataset)
    root = Path(str(cfg["output"]["root"]).format(dataset=args.dataset))
    out = Path(args.output) if args.output else root / args.model / "tuning"
    tune(cfg, list(args.seeds), out)


if __name__ == "__main__":
    main()
