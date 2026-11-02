from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from statement_ranking.dataset import DatasetPaths, load_canonical_statement_dictionary
from statement_ranking.storage import TYPES
from statement_ranking.utils import unique_preserve


def audit_interactions(
    interactions: list[dict[str, Any]],
    kcore_rows: list[dict[str, Any]],
    k: int,
    *,
    paths: DatasetPaths,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    ids = [int(row["interaction_id"]) for row in interactions]
    if len(ids) != len(set(ids)):
        errors.append("Final dataset contains duplicate interaction_id values.")

    if any(not row.get("match_statement_ids") for row in interactions):
        errors.append("Retained dataset contains an interaction without MATCH statements.")

    user_degree = Counter(str(row["user_id"]) for row in kcore_rows)
    item_degree = Counter(str(row["item_id"]) for row in kcore_rows)
    if user_degree and min(user_degree.values()) < k:
        errors.append("Pre-split user k-core invariant is violated.")
    if item_degree and min(item_degree.values()) < k:
        errors.append("Pre-split item k-core invariant is violated.")

    train = [row for row in interactions if row["split"] == "train"]
    test = [row for row in interactions if row["split"] == "test"]
    train_items = {str(row["item_id"]) for row in train}
    cold = [row for row in test if str(row["item_id"]) not in train_items]
    if cold:
        errors.append(f"Found {len(cold)} item-cold-start test interactions after filtering.")

    tests_per_user = Counter(str(row["user_id"]) for row in test)
    if any(count > 1 for count in tests_per_user.values()):
        errors.append("A user has more than one retained test interaction.")

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)
    temporal_violations = 0
    for uid, rows in by_user.items():
        ordered = sorted(rows, key=lambda r: (int(r["timestamp"]), int(r["interaction_id"])))
        tests = [row for row in ordered if row["split"] == "test"]
        if not tests:
            continue
        if len(tests) != 1 or int(tests[0]["interaction_id"]) != int(ordered[-1]["interaction_id"]):
            temporal_violations += 1
    if temporal_violations:
        errors.append(f"Found {temporal_violations} user-level temporal split violations.")

    dictionaries = {typ: load_canonical_statement_dictionary(paths, typ) for typ in TYPES}
    alignment_errors = 0
    missing_canonical = {typ: set() for typ in TYPES}
    for row in interactions:
        for typ in TYPES:
            original = [int(v) for v in row.get(f"{typ}_original_statement_ids", [])]
            aligned = [int(v) for v in row.get(f"{typ}_aligned_statement_ids", [])]
            canonical = [int(v) for v in row.get(f"{typ}_statement_ids", [])]
            if len(original) != len(aligned) or unique_preserve(aligned) != canonical:
                alignment_errors += 1
            missing_canonical[typ].update(sid for sid in canonical if sid not in dictionaries[typ])
    if alignment_errors:
        errors.append(f"Found {alignment_errors} final evidence/canonical alignment mismatches.")
    for typ in TYPES:
        if missing_canonical[typ]:
            errors.append(
                f"Final interactions reference unknown canonical {typ} IDs: "
                f"{sorted(missing_canonical[typ])[:10]}"
            )

    users_without_test = len({str(row["user_id"]) for row in train} - set(tests_per_user))
    if users_without_test:
        warnings.append(
            f"{users_without_test} users have train history but no retained test event because "
            "their held-out item was cold-start."
        )

    return {
        "complete": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "pre_split_min_user_degree": min(user_degree.values(), default=0),
            "pre_split_min_item_degree": min(item_degree.values(), default=0),
            "train_interactions": len(train),
            "test_interactions": len(test),
            "test_users": len(tests_per_user),
            "users_without_test_after_cold_start_filter": users_without_test,
            "cold_start_test_interactions": len(cold),
            "temporal_violations": temporal_violations,
            "alignment_errors": alignment_errors,
            "missing_canonical_ids": {typ: len(missing_canonical[typ]) for typ in TYPES},
        },
    }
