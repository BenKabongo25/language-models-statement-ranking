from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return _expand_env(value)


def _expand_env(v: Any) -> Any:
    if isinstance(v, str):
        return os.path.expandvars(os.path.expanduser(v))
    if isinstance(v, list):
        return [_expand_env(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand_env(x) for k, x in v.items()}
    return v


def deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_yaml(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(value, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def decode_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    return [int(x) for x in text.split(":") if x != ""]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def stable_hash64(values: np.ndarray, seed: int) -> np.ndarray:
    z = values.astype(np.uint64, copy=False) + np.uint64(seed)
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


def fingerprint_cfg(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
