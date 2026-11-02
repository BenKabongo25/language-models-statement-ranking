from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from statement_ranking.storage import TYPES
from statement_ranking.utils import load_yaml, project_root, read_jsonl, resolve_path


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    base: Path
    clustering_final: Path
    evidence_final: Path
    statement_ranking_root: Path
    reviews: Path
    meta_items: Path
    meta_users: Path


def resolve_dataset_paths(
    dataset: str,
    cfg: dict[str, Any],
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> DatasetPaths:
    if dataset_root is None:
        root = project_root()
        paths_cfg = load_yaml(resolve_path(root, cfg.get("paths_config", "configs/paths.yaml")))
        base = Path(paths_cfg["processed_root"]).expanduser().resolve() / dataset
    else:
        base = Path(dataset_root).expanduser().resolve()

    ranking_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else base / str(cfg.get("results_subdir", "statement_ranking"))
    )
    return DatasetPaths(
        dataset=dataset,
        base=base,
        clustering_final=base / str(cfg.get("clustering_final_subdir", "clustering_pipeline/final")),
        evidence_final=base / str(cfg.get("evidence_final_subdir", "evidence_pipeline/final")),
        statement_ranking_root=ranking_root,
        reviews=base / "reviews.jsonl",
        meta_items=base / "meta_items.jsonl",
        meta_users=base / "meta_users.jsonl",
    )


def required_input_paths(paths: DatasetPaths) -> list[Path]:
    # Deliberately final-only: no 00_prepare, 01_embeddings, 04_clusters, mappings,
    # or any other transient upstream artifact is required.
    required = [
        paths.clustering_final / "interactions.jsonl",
        paths.clustering_final / "evidence.jsonl",
    ]
    required.extend(paths.clustering_final / f"statements_{typ}.jsonl" for typ in TYPES)
    return required


def optional_input_paths(paths: DatasetPaths) -> list[Path]:
    return [
        paths.clustering_final / "id_offsets.json",
        paths.clustering_final / "item_candidates.jsonl",
        paths.clustering_final / "manifest.json",
        paths.evidence_final / "reviews_with_statements.jsonl",
        paths.evidence_final / "manifest.json",
        paths.reviews,
        paths.meta_items,
        paths.meta_users,
    ]


def assert_inputs(paths: DatasetPaths) -> None:
    missing = [path for path in required_input_paths(paths) if not path.exists()]
    if missing:
        details = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required final statement-ranking inputs. The module never reads "
            f"upstream intermediate stages:\n{details}"
        )


def load_final_interactions(paths: DatasetPaths) -> Iterator[dict[str, Any]]:
    for _, record in read_jsonl(paths.clustering_final / "interactions.jsonl"):
        yield record


def load_canonical_statement_dictionary(paths: DatasetPaths, typ: str) -> dict[int, dict[str, Any]]:
    if typ not in TYPES:
        raise ValueError(f"Unsupported statement type: {typ}")
    path = paths.clustering_final / f"statements_{typ}.jsonl"
    out: dict[int, dict[str, Any]] = {}
    for _, record in read_jsonl(path):
        sid = int(record["statement_id"])
        if sid in out:
            raise ValueError(f"Duplicate canonical statement_id={sid} in {path}")
        out[sid] = record
    return out


def load_id_offsets(paths: DatasetPaths) -> dict[str, Any] | None:
    path = paths.clustering_final / "id_offsets.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def iter_evidence_groups(paths: DatasetPaths) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Stream final evidence grouped by interaction_id.

    clustering_pipeline/final/evidence.jsonl is expected to preserve interaction order.
    We validate monotonicity so that preparing the ranking dataset needs no in-memory
    evidence index and no deleted upstream mapping files.
    """
    current_id: int | None = None
    current: list[dict[str, Any]] = []
    last_id = -1
    for _, record in read_jsonl(paths.clustering_final / "evidence.jsonl"):
        iid = int(record["interaction_id"])
        if iid < last_id:
            raise ValueError(
                "clustering_pipeline/final/evidence.jsonl is not ordered by interaction_id; "
                "cannot perform the final-only streaming merge safely"
            )
        last_id = iid
        if current_id is None:
            current_id = iid
        if iid != current_id:
            yield current_id, current
            current_id = iid
            current = []
        current.append(record)
    if current_id is not None:
        yield current_id, current


def iter_evidence_final_reviews(paths: DatasetPaths) -> Iterator[tuple[int, dict[str, Any]]]:
    path = paths.evidence_final / "reviews_with_statements.jsonl"
    if not path.exists():
        return
    yield from read_jsonl(path)
