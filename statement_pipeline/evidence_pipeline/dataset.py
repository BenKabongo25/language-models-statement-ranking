from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from evidence_pipeline.utils import load_yaml, project_root, read_jsonl, resolve_path


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else project_root() / "evidence_pipeline" / "configs" / "pipeline.yaml"
    if not path.is_absolute():
        path = project_root() / path
    return load_yaml(path)


def dataset_paths(dataset: str, pipeline_cfg: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    paths_cfg = load_yaml(resolve_path(root, pipeline_cfg.get("paths_config", "configs/paths.yaml")))
    datasets_cfg = load_yaml(resolve_path(root, pipeline_cfg.get("datasets_config", "configs/datasets.yaml")))["datasets"]
    if dataset not in datasets_cfg:
        raise KeyError(f"Unknown dataset: {dataset}")
    processed_root = Path(paths_cfg["processed_root"])
    base = processed_root / dataset
    return {
        "base": base,
        "reviews": base / "reviews.jsonl",
        "items": base / "meta_items.jsonl",
        "stats": base / "stats.json",
        "pipeline": base / str(pipeline_cfg.get("results_subdir", "_evidence_pipeline")),
    }


def has_review_title(dataset: str, pipeline_cfg: dict[str, Any]) -> bool:
    root = project_root()
    datasets_cfg = load_yaml(resolve_path(root, pipeline_cfg.get("datasets_config", "configs/datasets.yaml")))["datasets"]
    return bool(datasets_cfg[dataset].get("review_title", False))


def load_item_contexts(items_path: Path) -> dict[str, dict[str, str]]:
    contexts: dict[str, dict[str, str]] = {}
    if not items_path.exists():
        return contexts
    for _, item in read_jsonl(items_path):
        item_id = str(item.get("item_id", ""))
        if item_id:
            contexts[item_id] = {
                "name": str(item.get("name", "")),
                "brand": str(item.get("brand", "")),
            }
    return contexts


def load_indices(path: str | Path) -> set[int]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        values = json.loads(text)
    else:
        values = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    indices = {int(value) for value in values}
    if any(value < 0 for value in indices):
        raise ValueError("Review indices must be non-negative.")
    return indices


def selected_reviews(
    reviews_path: Path,
    *,
    limit: int | None = None,
    indices: set[int] | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if limit is not None and indices is not None:
        raise ValueError("limit and indices are mutually exclusive")
    for line_index, record in read_jsonl(reviews_path):
        if limit is not None and line_index >= limit:
            break
        if indices is not None and line_index not in indices:
            continue
        yield line_index, record
