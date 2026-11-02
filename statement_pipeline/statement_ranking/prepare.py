from __future__ import annotations

from typing import Any, Iterator

from statement_ranking.dataset import DatasetPaths, iter_evidence_groups, load_final_interactions
from statement_ranking.storage import TYPES
from statement_ranking.utils import unique_preserve


def _advance_group(iterator, current):
    try:
        return next(iterator)
    except StopIteration:
        return None


def iter_prepared_interactions(
    paths: DatasetPaths,
    cfg: dict[str, Any],
) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    """Merge only clustering_pipeline/final/interactions + final/evidence.

    `original_statement_ids` are occurrence-level final evidence IDs. This is closer to
    EXTRA's senID semantics than the deleted clustering raw_statement_id namespace,
    which had already deduplicated identical normalized texts.
    """
    selection = cfg.get("selection", {})
    require_match = bool(selection.get("require_match_statements", True))
    timestamp_policy = str(selection.get("missing_timestamp", "error"))
    if timestamp_policy not in {"error", "drop"}:
        raise ValueError("selection.missing_timestamp must be error or drop")

    stats: dict[str, Any] = {
        "input_interactions": 0,
        "retained_match_interactions": 0,
        "dropped_without_match_statements": 0,
        "dropped_missing_timestamp": 0,
        "evidence_occurrences": {typ: 0 for typ in TYPES},
    }

    evidence_it = iter(iter_evidence_groups(paths))
    current_group = _advance_group(evidence_it, None)

    def generate() -> Iterator[dict[str, Any]]:
        nonlocal current_group
        previous_iid = -1
        for final in load_final_interactions(paths):
            stats["input_interactions"] += 1
            iid = int(final["interaction_id"])
            if iid < previous_iid:
                raise ValueError(
                    "clustering_pipeline/final/interactions.jsonl is not ordered by interaction_id"
                )
            previous_iid = iid

            while current_group is not None and current_group[0] < iid:
                current_group = _advance_group(evidence_it, current_group)
            evidence = current_group[1] if current_group is not None and current_group[0] == iid else []

            timestamp = final.get("timestamp")
            if timestamp is None:
                if timestamp_policy == "drop":
                    stats["dropped_missing_timestamp"] += 1
                    continue
                raise ValueError(f"Missing timestamp for interaction_id={iid}")

            match_ids = [int(v) for v in final.get("match_statement_ids", [])]
            if require_match and not match_ids:
                stats["dropped_without_match_statements"] += 1
                continue

            row: dict[str, Any] = {
                "interaction_id": iid,
                "line_index": int(final.get("line_index", iid)),
                "user_id": str(final["user_id"]),
                "item_id": str(final["item_id"]),
                "timestamp": int(timestamp),
                "rating": None if final.get("rating") is None else float(final["rating"]),
            }

            ordered_evidence = sorted(
                evidence,
                key=lambda e: (int(e.get("statement_index", 0)), int(e.get("evidence_id", 0))),
            )
            for typ in TYPES:
                canonical = [int(v) for v in final.get(f"{typ}_statement_ids", [])]
                original: list[int] = []
                aligned: list[int] = []
                for ev in ordered_evidence:
                    value = ev.get(f"{typ}_statement_id")
                    if value is None:
                        continue
                    original.append(int(ev["evidence_id"]))
                    aligned.append(int(value))
                stats["evidence_occurrences"][typ] += len(original)

                mapped_unique = unique_preserve(aligned)
                if canonical != mapped_unique:
                    # Finalize preserves first occurrence, so exact order should agree.
                    # A set-only match would hide a broken lineage/order contract.
                    raise ValueError(
                        f"Final evidence alignment mismatch for interaction_id={iid}, type={typ}: "
                        f"interaction={canonical[:10]} evidence={mapped_unique[:10]}"
                    )
                row[f"{typ}_statement_ids"] = canonical
                row[f"{typ}_original_statement_ids"] = original
                row[f"{typ}_aligned_statement_ids"] = aligned

            stats["retained_match_interactions"] += 1
            yield row

    return generate(), stats
