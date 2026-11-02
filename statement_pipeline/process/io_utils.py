from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Iterator

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return expand_env(data)


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def iter_records(path: str | Path, fmt: str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                if fmt == "json":
                    record = json.loads(line)
                elif fmt == "python_literal":
                    record = ast.literal_eval(line)
                else:
                    raise ValueError(f"Unsupported record format: {fmt}")
            except Exception as exc:
                raise ValueError(f"Failed to parse {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def write_json_line(handle, record: dict[str, Any], ensure_ascii: bool = False) -> None:
    handle.write(json.dumps(record, ensure_ascii=ensure_ascii, separators=(",", ":")) + "\n")


def drop_empty(record: dict[str, Any]) -> dict[str, Any]:
    def keep(value: Any) -> bool:
        return value is not None and value != "" and value != [] and value != {}

    return {key: value for key, value in record.items() if keep(value)}
