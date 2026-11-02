from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import deep_update, load_yaml, parse_scalar, set_dotted


def load_merged(paths: list[str | Path], overrides: list[str] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for p in paths:
        cfg = deep_update(cfg, load_yaml(p))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item}")
        k, v = item.split("=", 1)
        set_dotted(cfg, k, parse_scalar(v))
    validate(cfg)
    return cfg


def validate(cfg: dict[str, Any]) -> None:
    if "model" not in cfg or not cfg["model"].get("name"):
        raise ValueError("model.name is required")
    if not cfg.get("dataset", {}).get("root"):
        raise ValueError("dataset.root is required")
    ks = [int(k) for k in cfg.get("evaluation", {}).get("k", [5, 10])]
    if not ks or min(ks) < 1:
        raise ValueError("evaluation.k must contain positive integers")
    scope = cfg.get("sampling", {}).get("scope", "item")
    if scope != "item":
        raise ValueError("This benchmark intentionally enforces sampling.scope=item")
    source = cfg.get("sampling", {}).get("candidate_source", "fit_only")
    if source not in {"fit_only", "all"}:
        raise ValueError("sampling.candidate_source must be fit_only or all")
    if int(cfg.get("training", {}).get("num_negatives", 1)) < 1:
        raise ValueError("training.num_negatives must be >=1")
