from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def user_last_temporal_split(
    interactions: list[dict[str, Any]],
    *,
    tie_breaker: str = "interaction_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Leave the chronologically last interaction of each user for testing.

    Item cold-start test rows are excluded from both train and test; they are not moved
    back into train, preserving the temporal interpretation of the split.
    """
    if tie_breaker not in {"interaction_id", "line_index"}:
        raise ValueError("tie_breaker must be interaction_id or line_index")

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)

    train: list[dict[str, Any]] = []
    tentative_test: list[dict[str, Any]] = []
    timestamp_ties = 0

    for user_id, rows in by_user.items():
        ordered = sorted(
            rows,
            key=lambda r: (int(r["timestamp"]), int(r.get(tie_breaker, r["interaction_id"]))),
        )
        if len(ordered) >= 2 and int(ordered[-1]["timestamp"]) == int(ordered[-2]["timestamp"]):
            timestamp_ties += 1
        for pos, row in enumerate(ordered):
            copied = dict(row)
            copied["user_sequence_index"] = pos
            copied["user_sequence_length"] = len(ordered)
            if pos == len(ordered) - 1:
                tentative_test.append(copied)
            else:
                train.append(copied)

    train_items = {str(row["item_id"]) for row in train}
    test: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in tentative_test:
        if str(row["item_id"]) not in train_items:
            copied = dict(row)
            copied["exclusion_reason"] = "item_cold_start"
            excluded.append(copied)
        else:
            test.append(row)

    # Recompute user and item chronology over the retained train+test interactions.
    # This matters for users whose held-out future event was excluded as item-cold-start.
    retained = train + test
    user_positions: dict[int, tuple[int, int]] = {}
    retained_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        retained_by_user[str(row["user_id"])].append(row)
    for rows in retained_by_user.values():
        ordered = sorted(rows, key=lambda r: (int(r["timestamp"]), int(r["interaction_id"])))
        for pos, row in enumerate(ordered):
            user_positions[int(row["interaction_id"])] = (pos, len(ordered))

    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        by_item[str(row["item_id"])].append(row)
    item_positions: dict[int, tuple[int, int]] = {}
    for rows in by_item.values():
        ordered = sorted(rows, key=lambda r: (int(r["timestamp"]), int(r["interaction_id"])))
        for pos, row in enumerate(ordered):
            item_positions[int(row["interaction_id"])] = (pos, len(ordered))

    train_ids = {int(row["interaction_id"]) for row in train}
    final_rows: list[dict[str, Any]] = []
    for row in retained:
        copied = dict(row)
        user_pos, user_len = user_positions[int(row["interaction_id"])]
        item_pos, item_len = item_positions[int(row["interaction_id"])]
        copied["user_sequence_index"] = user_pos
        copied["user_sequence_length"] = user_len
        copied["item_sequence_index"] = item_pos
        copied["item_sequence_length"] = item_len
        copied["split"] = "train" if int(row["interaction_id"]) in train_ids else "test"
        final_rows.append(copied)

    final_rows.sort(key=lambda r: (int(r["timestamp"]), int(r["interaction_id"])))
    for global_pos, row in enumerate(final_rows):
        row["global_sequence_index"] = global_pos

    train_count = sum(row["split"] == "train" for row in final_rows)
    test_count = sum(row["split"] == "test" for row in final_rows)
    train_user_counts = Counter(str(row["user_id"]) for row in final_rows if row["split"] == "train")
    test_user_counts = Counter(str(row["user_id"]) for row in final_rows if row["split"] == "test")

    stats = {
        "strategy": "user_last",
        "interactions_input": len(interactions),
        "interactions_retained": len(final_rows),
        "train_interactions": train_count,
        "test_interactions": test_count,
        "excluded_cold_start_test_interactions": len(excluded),
        "users_total": len({str(row["user_id"]) for row in final_rows}),
        "users_with_test": len(test_user_counts),
        "users_train_only_after_cold_start_filter": len(set(train_user_counts) - set(test_user_counts)),
        "items_train": len({str(row["item_id"]) for row in final_rows if row["split"] == "train"}),
        "items_test": len({str(row["item_id"]) for row in final_rows if row["split"] == "test"}),
        "timestamp_ties_at_user_boundary": timestamp_ties,
    }
    return final_rows, excluded, stats
