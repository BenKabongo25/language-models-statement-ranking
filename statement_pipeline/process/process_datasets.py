from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from process.adapters import ADAPTERS
from process.io_utils import drop_empty, iter_records, load_yaml, write_json_line


def resolve(root: Path, relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else root / path


def process_dataset(
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    data_root: Path,
    processed_root: Path,
    processing_cfg: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    kind = dataset_cfg["kind"]
    if kind not in ADAPTERS:
        raise ValueError(f"Unknown dataset kind '{kind}' for {dataset_name}")
    adapter = ADAPTERS[kind]

    output_dir = processed_root / dataset_name
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_ascii = bool(processing_cfg.get("ensure_ascii", False))
    drop_empty_fields = bool(processing_cfg.get("drop_empty_metadata_fields", True))
    keep_only_referenced = bool(processing_cfg.get("keep_only_referenced_metadata", True))

    review_path = resolve(data_root, dataset_cfg["reviews"])
    item_path = resolve(data_root, dataset_cfg["items"])
    user_path = resolve(data_root, dataset_cfg["users"]) if dataset_cfg.get("users") else None

    referenced_users: set[str] = set()
    referenced_items: set[str] = set()
    derived_users: dict[str, dict[str, Any]] = {}
    review_count = 0
    max_review_chars = 0
    empty_reviews = 0

    reviews_output = output_dir / "reviews.jsonl"
    with reviews_output.open("w", encoding="utf-8") as out:
        for raw in tqdm(iter_records(review_path, adapter["review_format"]), desc=f"{dataset_name}: reviews"):
            review = adapter["review"](raw)
            if not review["user_id"] or not review["item_id"]:
                continue
            referenced_users.add(review["user_id"])
            referenced_items.add(review["item_id"])
            review_count += 1
            max_review_chars = max(max_review_chars, len(review.get("review", "")))
            if not review.get("review"):
                empty_reviews += 1
            write_json_line(out, review, ensure_ascii=ensure_ascii)

            if (
                kind == "amazon2014"
                and dataset_cfg.get("derive_user_metadata", False)
                and processing_cfg.get("write_derived_user_metadata", True)
            ):
                user = adapter["user_from_review"](raw)
                user_id = user.get("user_id")
                if user_id:
                    current = derived_users.get(user_id)
                    if current is None or (not current.get("name") and user.get("name")):
                        derived_users[user_id] = user

    item_count = 0
    written_item_ids: set[str] = set()
    with (output_dir / "meta_items.jsonl").open("w", encoding="utf-8") as out:
        for raw in tqdm(iter_records(item_path, adapter["item_format"]), desc=f"{dataset_name}: items"):
            item = adapter["item"](raw)
            item_id = item.get("item_id", "")
            if not item_id or item_id in written_item_ids:
                continue
            if keep_only_referenced and item_id not in referenced_items:
                continue
            if drop_empty_fields:
                item = drop_empty(item)
            write_json_line(out, item, ensure_ascii=ensure_ascii)
            written_item_ids.add(item_id)
            item_count += 1

    user_count = 0
    meta_users_path = output_dir / "meta_users.jsonl"
    if user_path is not None:
        written_user_ids: set[str] = set()
        with meta_users_path.open("w", encoding="utf-8") as out:
            for raw in tqdm(iter_records(user_path, adapter["user_format"]), desc=f"{dataset_name}: users"):
                user = adapter["user"](raw)
                user_id = user.get("user_id", "")
                if not user_id or user_id in written_user_ids:
                    continue
                if keep_only_referenced and user_id not in referenced_users:
                    continue
                if drop_empty_fields:
                    user = drop_empty(user)
                write_json_line(out, user, ensure_ascii=ensure_ascii)
                written_user_ids.add(user_id)
                user_count += 1
    elif derived_users:
        with meta_users_path.open("w", encoding="utf-8") as out:
            for user_id in sorted(referenced_users):
                user = derived_users.get(user_id, {"user_id": user_id})
                if drop_empty_fields:
                    user = drop_empty(user)
                write_json_line(out, user, ensure_ascii=ensure_ascii)
                user_count += 1

    stats = {
        "dataset": dataset_name,
        "kind": kind,
        "reviews": review_count,
        "users_in_reviews": len(referenced_users),
        "items_in_reviews": len(referenced_items),
        "metadata_items_written": item_count,
        "metadata_users_written": user_count,
        "missing_item_metadata": len(referenced_items - written_item_ids),
        "empty_reviews": empty_reviews,
        "max_review_chars": max_review_chars,
        "has_review_title": bool(dataset_cfg.get("review_title", False)),
    }

    if processing_cfg.get("write_stats", True):
        with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Process and harmonize recommendation datasets.")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--processing-config", default="configs/processing.yaml")
    parser.add_argument("--dataset", action="append", help="Dataset to process. Repeat for multiple datasets.")
    parser.add_argument("--all", action="store_true", help="Process every dataset from datasets.yaml.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths_cfg = load_yaml(args.paths_config)
    datasets_cfg = load_yaml(args.datasets_config)["datasets"]
    processing_cfg = load_yaml(args.processing_config)

    if args.all:
        selected = list(datasets_cfg)
    elif args.dataset:
        selected = args.dataset
    else:
        parser.error("Use --all or at least one --dataset NAME.")

    data_root = Path(paths_cfg["data_root"])
    processed_root = Path(paths_cfg["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for name in selected:
        if name not in datasets_cfg:
            raise KeyError(f"Unknown dataset: {name}")
        stats = process_dataset(
            name,
            datasets_cfg[name],
            data_root,
            processed_root,
            processing_cfg,
            overwrite=args.overwrite,
        )
        all_stats.append(stats)
        print(json.dumps(stats, indent=2))

    with (processed_root / "processing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
