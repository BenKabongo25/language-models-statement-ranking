from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .data import IndexedInteraction, RankingData


@dataclass
class ItemNegativeSampler:
    data: RankingData
    fit: list[IndexedInteraction]
    candidate_source: str = "fit_only"
    seed: int = 42

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.fit_item_pool = self.data.observed_statements_by_item(self.fit)
        # Exclude every observed positive for the target (user,item) pair.  This is
        # stricter than excluding only the current row and prevents repeated
        # interactions for the same pair from supplying false negatives.
        self.fit_pair_positives = self.data.positives_by_pair(self.fit)

    def pool(self, x: IndexedInteraction) -> list[int]:
        if self.candidate_source == "all":
            base = self.data.item_candidates.get(x.item, np.empty(0, dtype=np.int64))
        else:
            base = self.fit_item_pool.get(x.item, np.empty(0, dtype=np.int64))
        pos = self.fit_pair_positives.get((x.user, x.item), set(x.statements))
        return [int(s) for s in base if int(s) not in pos]

    def sample(self, x: IndexedInteraction, n: int = 1, replace: bool = True) -> list[int]:
        pool = self.pool(x)
        if not pool:
            return []
        if replace:
            return [self.rng.choice(pool) for _ in range(n)]
        if n >= len(pool):
            out = list(pool)
            self.rng.shuffle(out)
            return out
        return self.rng.sample(pool, n)

    def one_pair(self, x: IndexedInteraction) -> tuple[int, int] | None:
        if not x.statements:
            return None
        neg = self.sample(x, 1)
        if not neg:
            return None
        return self.rng.choice(list(x.statements)), neg[0]
