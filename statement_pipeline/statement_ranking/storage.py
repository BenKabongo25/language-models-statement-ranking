from __future__ import annotations

from pathlib import Path

TYPES = ("user", "item", "match")

STAGE_PREPARE = "00_prepare"
STAGE_KCORE = "01_kcore"
STAGE_SPLIT = "02_split"
STAGE_FINAL = "final"


def prepare_interactions_path(root: Path) -> Path:
    return root / STAGE_PREPARE / "interactions.jsonl"


def kcore_interactions_path(root: Path) -> Path:
    return root / STAGE_KCORE / "interactions.jsonl"


def split_interactions_path(root: Path) -> Path:
    return root / STAGE_SPLIT / "interactions.jsonl"


def split_excluded_path(root: Path) -> Path:
    return root / STAGE_SPLIT / "excluded_cold_start.jsonl"


def final_dir(root: Path) -> Path:
    return root / STAGE_FINAL


def final_interactions_path(root: Path, typ: str) -> Path:
    return final_dir(root) / f"interactions_{typ}.csv"


def final_all_interactions_path(root: Path) -> Path:
    return final_dir(root) / "interactions.csv"


def final_canonical_statements_path(root: Path, typ: str) -> Path:
    return final_dir(root) / f"statements_{typ}.csv"


def final_original_statements_path(root: Path, typ: str) -> Path:
    return final_dir(root) / f"original_statements_{typ}.csv"


def final_id2statement_path(root: Path, typ: str) -> Path:
    return final_dir(root) / f"id2statement_{typ}.json"


def final_id2original_path(root: Path, typ: str) -> Path:
    return final_dir(root) / f"id2original_{typ}.json"


def final_item_candidates_path(root: Path, typ: str) -> Path:
    return final_dir(root) / f"item_candidates_{typ}.csv"


def final_users_path(root: Path) -> Path:
    return final_dir(root) / "users.csv"


def final_items_path(root: Path) -> Path:
    return final_dir(root) / "items.csv"
