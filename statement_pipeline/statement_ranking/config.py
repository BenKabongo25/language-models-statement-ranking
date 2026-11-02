from __future__ import annotations

from pathlib import Path
from typing import Any

from statement_ranking.utils import load_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    k = int(cfg.get("k_core", {}).get("k", 5))
    if k < 1:
        raise ValueError("k_core.k must be >= 1")
    split = cfg.get("split", {})
    if split.get("strategy", "user_last") != "user_last":
        raise ValueError("Only split.strategy=user_last is supported")
    if split.get("tie_breaker", "interaction_id") not in {"interaction_id", "line_index"}:
        raise ValueError("split.tie_breaker must be interaction_id or line_index")
    if split.get("cold_start_items", "drop_test") != "drop_test":
        raise ValueError("Only split.cold_start_items=drop_test is supported")
    if cfg.get("selection", {}).get("missing_timestamp", "error") not in {"error", "drop"}:
        raise ValueError("selection.missing_timestamp must be error or drop")
