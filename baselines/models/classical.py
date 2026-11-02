from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from baselines.data import IndexedInteraction, RankingData
from baselines.models.base import BaseRanker
from baselines.utils import stable_hash64


class RandomRanker(BaseRanker):
    deterministic = False
    def fit(self, interactions, *, validation=None): return {"fit_interactions": len(interactions)}
    def score(self, user, item, candidates):
        keys = stable_hash64(np.asarray(candidates, dtype=np.int64), self.seed ^ ((user + 1) * 1000003) ^ ((item + 1) * 9176))
        # Convert uint keys to stable pseudo-scores; higher keys = higher rank.
        return keys.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


class _Popularity(BaseRanker):
    deterministic = True
    axis = "global"
    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        self.global_counts = Counter()
        self.user_counts: dict[int, Counter] = defaultdict(Counter)
        self.item_counts: dict[int, Counter] = defaultdict(Counter)
        for x in interactions:
            for s in set(x.statements):
                self.global_counts[s] += 1
                self.user_counts[x.user][s] += 1
                self.item_counts[x.item][s] += 1
        return {"fit_interactions": len(interactions)}
    def score(self, user, item, candidates):
        if self.axis == "global": c = self.global_counts
        elif self.axis == "user": c = self.user_counts.get(user, {})
        else: c = self.item_counts.get(item, {})
        return np.asarray([float(c.get(int(s), 0)) for s in candidates], dtype=np.float64)

class GlobalPop(_Popularity): axis = "global"
class UserPop(_Popularity): axis = "user"
class ItemPop(_Popularity): axis = "item"


class RUCF(BaseRanker):
    """User-based CF from EXTRA/BPER: Jaccard on explanation sets, neighbor evidence aggregation."""
    deterministic = True
    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        user_exp: dict[int, set[int]] = defaultdict(set)
        item_users: dict[int, set[int]] = defaultdict(set)
        item_stmt_users: dict[tuple[int, int], set[int]] = defaultdict(set)
        for x in interactions:
            user_exp[x.user].update(x.statements)
            item_users[x.item].add(x.user)
            for s in x.statements: item_stmt_users[(x.item, s)].add(x.user)
        self.user_exp = user_exp; self.item_users = item_users; self.item_stmt_users = item_stmt_users
        self.neighbor_k = self.cfg.get("model", {}).get("neighbor_k")
        return {"fit_interactions": len(interactions)}

    def _sim(self, a: int, b: int) -> float:
        A, B = self.user_exp.get(a, set()), self.user_exp.get(b, set())
        if not A or not B: return 0.0
        return len(A & B) / len(A | B)

    def score(self, user, item, candidates):
        peers = [v for v in self.item_users.get(item, set()) if v != user]
        sims = [(v, self._sim(user, v)) for v in peers]
        sims = [(v, s) for v, s in sims if s > 0]
        sims.sort(key=lambda z: (-z[1], z[0]))
        if self.neighbor_k is not None: sims = sims[:int(self.neighbor_k)]
        sim = dict(sims)
        out = []
        for e in candidates:
            out.append(sum(sim.get(v, 0.0) for v in self.item_stmt_users.get((item, int(e)), set())))
        return np.asarray(out, dtype=np.float64)


class RICF(BaseRanker):
    """Item-based CF from EXTRA/BPER: Jaccard on explanation sets, user-history aggregation."""
    deterministic = True
    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        item_exp: dict[int, set[int]] = defaultdict(set)
        user_items: dict[int, set[int]] = defaultdict(set)
        item_stmt: dict[int, set[int]] = defaultdict(set)
        for x in interactions:
            item_exp[x.item].update(x.statements); item_stmt[x.item].update(x.statements); user_items[x.user].add(x.item)
        self.item_exp = item_exp; self.user_items = user_items; self.item_stmt = item_stmt
        self.neighbor_k = self.cfg.get("model", {}).get("neighbor_k")
        return {"fit_interactions": len(interactions)}

    def _sim(self, a: int, b: int) -> float:
        A, B = self.item_exp.get(a, set()), self.item_exp.get(b, set())
        if not A or not B: return 0.0
        return len(A & B) / len(A | B)

    def score(self, user, item, candidates):
        hist = [j for j in self.user_items.get(user, set()) if j != item]
        sims = [(j, self._sim(item, j)) for j in hist]
        sims = [(j, s) for j, s in sims if s > 0]
        sims.sort(key=lambda z: (-z[1], z[0]))
        if self.neighbor_k is not None: sims = sims[:int(self.neighbor_k)]
        out = []
        for e in candidates:
            e = int(e)
            out.append(sum(s for j, s in sims if e in self.item_stmt.get(j, set())))
        return np.asarray(out, dtype=np.float64)
