from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import decode_ids


@dataclass(frozen=True)
class Interaction:
    interaction_id: int
    user_id: str
    item_id: str
    timestamp: int
    rating: float | None
    statements: tuple[int, ...]
    split: str


@dataclass(frozen=True)
class IndexedInteraction:
    interaction_id: int
    user: int
    item: int
    timestamp: int
    rating: float | None
    statements: tuple[int, ...]
    split: str


@dataclass
class SplitBundle:
    # All interactions strictly before the two held-out events.
    train: list[IndexedInteraction]
    # The penultimate event for every eligible user.  These rows are kept for the
    # final train+validation refit even when their item is cold relative to train.
    val: list[IndexedInteraction]
    # Validation rows that can actually be ranked using a model fitted on train.
    val_eval: list[IndexedInteraction]
    # Last event, after removal of item-cold-start test cases with respect to
    # train+validation.
    test: list[IndexedInteraction]
    val_skipped_cold_item: int
    test_skipped_cold_item: int


class RankingData:
    """In-memory index over statement_ranking/final.

    All model IDs are compact local indices. Canonical statement IDs are preserved via
    sid_to_idx / idx_to_sid for reproducible output.
    """

    def __init__(self, root: str | Path, statement_type: str = "match"):
        root = Path(root)
        self.root = root
        self.final = root / "final" if (root / "final").exists() else root
        path = self.final / f"interactions_{statement_type}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw: list[Interaction] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                s = tuple(dict.fromkeys(decode_ids(r.get("statement_ids"))))
                if not s:
                    continue
                raw.append(Interaction(
                    interaction_id=int(r["interaction_id"]), user_id=str(r["user_id"]),
                    item_id=str(r["item_id"]), timestamp=int(r["timestamp"]),
                    rating=None if r.get("rating", "") == "" else float(r["rating"]), statements=s,
                    split=str(r.get("split", "train") or "train").strip().lower(),
                ))
        if not raw:
            raise ValueError(f"No interactions in {path}")
        self.raw = raw
        users = sorted({x.user_id for x in raw})
        items = sorted({x.item_id for x in raw})
        # Include all canonical statements from dictionary/candidates, not only positives.
        stmt_dict_path = self.final / f"id2statement_{statement_type}.json"
        if stmt_dict_path.exists():
            id2text = {int(k): str(v) for k, v in json.loads(stmt_dict_path.read_text(encoding="utf-8")).items()}
        else:
            id2text = {}
        candidates_raw = self._load_candidates(statement_type)
        statement_ids = sorted(set(id2text) | {s for x in raw for s in x.statements} | {s for v in candidates_raw.values() for s in v})
        self.user_to_idx = {x: j for j, x in enumerate(users)}
        self.item_to_idx = {x: j for j, x in enumerate(items)}
        self.sid_to_idx = {x: j for j, x in enumerate(statement_ids)}
        self.idx_to_user = users
        self.idx_to_item = items
        self.idx_to_sid = np.asarray(statement_ids, dtype=np.int64)
        self.statement_texts = [id2text.get(s, "") for s in statement_ids]
        self.interactions = [IndexedInteraction(
            interaction_id=x.interaction_id, user=self.user_to_idx[x.user_id], item=self.item_to_idx[x.item_id],
            timestamp=x.timestamp, rating=x.rating,
            statements=tuple(self.sid_to_idx[s] for s in x.statements if s in self.sid_to_idx),
            split=x.split,
        ) for x in raw]
        self.item_candidates: dict[int, np.ndarray] = {}
        for item_id, sids in candidates_raw.items():
            if item_id not in self.item_to_idx:
                continue
            vals = sorted({self.sid_to_idx[s] for s in sids if s in self.sid_to_idx})
            self.item_candidates[self.item_to_idx[item_id]] = np.asarray(vals, dtype=np.int64)
        # Fallback if candidates file is absent/incomplete.
        by_item: dict[int, set[int]] = defaultdict(set)
        for x in self.interactions:
            by_item[x.item].update(x.statements)
        for i, vals in by_item.items():
            if i not in self.item_candidates or len(self.item_candidates[i]) == 0:
                self.item_candidates[i] = np.asarray(sorted(vals), dtype=np.int64)

    def _load_candidates(self, typ: str) -> dict[str, list[int]]:
        path = self.final / f"item_candidates_{typ}.csv"
        if not path.exists():
            return {}
        out: dict[str, list[int]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out[str(r["item_id"])] = decode_ids(r.get("statement_ids"))
        return out

    @property
    def n_users(self) -> int: return len(self.idx_to_user)
    @property
    def n_items(self) -> int: return len(self.idx_to_item)
    @property
    def n_statements(self) -> int: return len(self.idx_to_sid)

    def temporal_split(self, drop_cold_start_val: bool = True) -> SplitBundle:
        """Build validation from the penultimate event while preserving upstream test.

        ``statement_ranking`` already performed the scientific leave-last-out split and
        removed item-cold-start test rows.  Re-splitting all retained rows would silently
        create a different benchmark, especially for users whose true last interaction was
        dropped upstream.  We therefore treat rows labelled ``test`` as authoritative.

        For users that have an upstream test row, the latest upstream-train interaction is
        the penultimate event and is held out for validation.  Users without an upstream
        test row remain entirely in training because their actual future event is no longer
        available in the benchmark.  Final fitting always uses train + validation.
        """
        upstream_train = [x for x in self.interactions if x.split != "test"]
        upstream_test = [x for x in self.interactions if x.split == "test"]

        # Backward-compatible fallback for hand-built/legacy files with no split labels.
        # Production statement_ranking datasets are expected to contain explicit test rows.
        if not upstream_test:
            by_user_all: dict[int, list[IndexedInteraction]] = defaultdict(list)
            for x in self.interactions:
                by_user_all[x.user].append(x)
            upstream_train = []
            upstream_test = []
            for rows in by_user_all.values():
                rows = sorted(rows, key=lambda x: (x.timestamp, x.interaction_id))
                if len(rows) >= 2:
                    upstream_train.extend(rows[:-1])
                    upstream_test.append(rows[-1])
                else:
                    upstream_train.extend(rows)

        test_users = {x.user for x in upstream_test}

        by_user_train: dict[int, list[IndexedInteraction]] = defaultdict(list)
        for x in upstream_train:
            by_user_train[x.user].append(x)

        train: list[IndexedInteraction] = []
        val: list[IndexedInteraction] = []
        for user, rows in by_user_train.items():
            rows = sorted(rows, key=lambda x: (x.timestamp, x.interaction_id))
            if user in test_users and rows:
                train.extend(rows[:-1])
                val.append(rows[-1])
            else:
                train.extend(rows)

        # Validation is evaluated only when the target item was already observed in the
        # core training slice.  The row is nevertheless restored for the final refit.
        train_items = {x.item for x in train}
        if drop_cold_start_val:
            val_eval = [x for x in val if x.item in train_items]
        else:
            val_eval = list(val)
        val_skipped = len(val) - len(val_eval)

        # Upstream construction guarantees non-cold test items with respect to its train.
        # Re-assert the invariant against the final fit set rather than inventing new test
        # examples if a malformed/manual dataset violates it.
        final_fit_items = train_items | {x.item for x in val}
        test = [x for x in upstream_test if x.item in final_fit_items]
        test_skipped = len(upstream_test) - len(test)

        return SplitBundle(
            train=train, val=val, val_eval=val_eval, test=test,
            val_skipped_cold_item=val_skipped,
            test_skipped_cold_item=test_skipped,
        )

    @staticmethod
    def positives_by_pair(interactions: Iterable[IndexedInteraction]) -> dict[tuple[int, int], set[int]]:
        out: dict[tuple[int, int], set[int]] = defaultdict(set)
        for x in interactions:
            out[(x.user, x.item)].update(x.statements)
        return out

    @staticmethod
    def observed_statements_by_item(interactions: Iterable[IndexedInteraction]) -> dict[int, np.ndarray]:
        d: dict[int, set[int]] = defaultdict(set)
        for x in interactions:
            d[x.item].update(x.statements)
        return {i: np.asarray(sorted(s), dtype=np.int64) for i, s in d.items()}

    @staticmethod
    def observed_items_by_user(interactions: Iterable[IndexedInteraction]) -> dict[int, set[int]]:
        d: dict[int, set[int]] = defaultdict(set)
        for x in interactions: d[x.user].add(x.item)
        return d

    def fit_candidate_pool(self, item: int, fit: list[IndexedInteraction], source: str = "fit_only") -> np.ndarray:
        if source == "all":
            return self.item_candidates.get(item, np.empty(0, dtype=np.int64))
        observed = self.observed_statements_by_item(fit)
        return observed.get(item, np.empty(0, dtype=np.int64))

    def split_stats(self, split: SplitBundle) -> dict[str, int]:
        return {
            "users": self.n_users, "items": self.n_items, "statements": self.n_statements,
            "train_interactions": len(split.train),
            "validation_interactions": len(split.val),
            "validation_evaluable_interactions": len(split.val_eval),
            "test_interactions": len(split.test),
            "validation_skipped_cold_item": split.val_skipped_cold_item,
            "test_skipped_cold_item": split.test_skipped_cold_item,
        }
