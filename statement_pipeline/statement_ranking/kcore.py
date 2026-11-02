from __future__ import annotations

from collections import Counter
from typing import Any


def iterative_k_core(
    interactions: list[dict[str, Any]],
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Iteratively retain users and items with at least k interactions.

    Degrees are recomputed after every pruning wave until a fixed point is reached.
    Interactions, rather than unique counterpart entities, define degree, matching the
    usual k-core preprocessing used in recommender-system datasets.
    """
    current = list(interactions)
    rounds: list[dict[str, int]] = []

    while True:
        user_degree = Counter(str(row["user_id"]) for row in current)
        item_degree = Counter(str(row["item_id"]) for row in current)
        keep_users = {u for u, degree in user_degree.items() if degree >= k}
        keep_items = {i for i, degree in item_degree.items() if degree >= k}
        filtered = [
            row
            for row in current
            if str(row["user_id"]) in keep_users and str(row["item_id"]) in keep_items
        ]
        rounds.append(
            {
                "input_interactions": len(current),
                "output_interactions": len(filtered),
                "users": len(keep_users),
                "items": len(keep_items),
            }
        )
        if len(filtered) == len(current):
            current = filtered
            break
        current = filtered
        if not current:
            break

    final_users = Counter(str(row["user_id"]) for row in current)
    final_items = Counter(str(row["item_id"]) for row in current)
    if current:
        assert min(final_users.values()) >= k
        assert min(final_items.values()) >= k

    stats = {
        "k": int(k),
        "rounds": rounds,
        "round_count": len(rounds),
        "interactions": len(current),
        "users": len(final_users),
        "items": len(final_items),
        "min_user_degree": min(final_users.values(), default=0),
        "min_item_degree": min(final_items.values(), default=0),
    }
    return current, stats
