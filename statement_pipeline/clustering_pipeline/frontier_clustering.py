from __future__ import annotations

import csv
import heapq
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm

from clustering_pipeline.pair_store import PairScoreStore, pack_pair_keys
from clustering_pipeline.rejection_index import KnownRejectionIndex
from clustering_pipeline.utils import atomic_write_json, load_json, stable_hash


@dataclass(slots=True, frozen=True)
class SeedProposal:
    """One initial reranker-positive edge used only to propose a cluster merge."""

    a_pos: int
    b_pos: int
    seed_score: float
    seed_cosine: float
    seed_index: int


class SentinelSelector:
    """Embedding medoid followed by farthest-first sentinels.

    Embeddings must be L2-normalized. Statement positions index ``rows`` which
    maps the statement dictionary order to the embedding matrix order.
    """

    def __init__(self, embeddings: np.ndarray, rows: np.ndarray, q: int = 4) -> None:
        self.embeddings = embeddings
        self.rows = np.asarray(rows, dtype=np.int64)
        self.q = max(1, int(q))

    def select(self, members_pos: np.ndarray) -> tuple[np.ndarray, float]:
        members = np.asarray(members_pos, dtype=np.int64)
        if len(members) == 0:
            return np.empty(0, dtype=np.int64), 0.0
        if len(members) == 1:
            return members.copy(), 0.0

        x = np.asarray(self.embeddings[self.rows[members]], dtype=np.float32)
        center_sum = x.sum(axis=0, dtype=np.float32)
        first = int(np.argmax(x @ center_sum))

        chosen = np.zeros(len(members), dtype=bool)
        chosen[first] = True
        selected = [first]
        min_dist = 1.0 - (x @ x[first])
        min_dist[first] = 0.0

        target = min(self.q, len(members))
        while len(selected) < target:
            scores = np.where(chosen, -np.inf, min_dist)
            nxt = int(np.argmax(scores))
            if not np.isfinite(scores[nxt]):
                break
            selected.append(nxt)
            chosen[nxt] = True
            dist = 1.0 - (x @ x[nxt])
            min_dist = np.minimum(min_dist, dist)
            min_dist[chosen] = 0.0

        radius = float(np.max(min_dist)) if len(members) > len(selected) else 0.0
        return members[np.asarray(selected, dtype=np.int64)], radius


class ClusterState:
    """Union-find plus members and sentinel state.

    Roots are deterministic (minimum positional id). Non-singleton member arrays
    are materialized because they are needed both for exact sentinel selection
    and for permanent-veto queries.
    """

    def __init__(self, n: int, q: int) -> None:
        self.n = int(n)
        self.q = max(1, int(q))
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.sentinel = np.full((n, self.q), -1, dtype=np.int64)
        self.sentinel[:, 0] = np.arange(n, dtype=np.int64)
        self.sentinel_count = np.ones(n, dtype=np.uint8)
        self.cover_radius = np.zeros(n, dtype=np.float32)
        self.merge_min_score = np.ones(n, dtype=np.float32)
        self.verification_pairs = np.zeros(n, dtype=np.int64)
        self._members: dict[int, np.ndarray] = {}

    def find(self, x: int) -> int:
        x = int(x)
        while int(self.parent[x]) != x:
            self.parent[x] = self.parent[int(self.parent[x])]
            x = int(self.parent[x])
        return x

    def find_many(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        roots = values.copy()
        if not len(roots):
            return roots
        while True:
            nxt = self.parent[roots]
            changed = nxt != roots
            if not np.any(changed):
                break
            roots[changed] = nxt[changed]
        self.parent[values] = roots
        return roots

    def members(self, root: int) -> np.ndarray:
        r = self.find(root)
        arr = self._members.get(r)
        if arr is None:
            return np.asarray([r], dtype=np.int64)
        return arr

    def sentinels(self, root: int) -> np.ndarray:
        r = self.find(root)
        return self.sentinel[r, : int(self.sentinel_count[r])].copy()

    def roots(self) -> np.ndarray:
        idx = np.arange(self.n, dtype=np.int64)
        roots = self.find_many(idx)
        return np.flatnonzero(roots == idx).astype(np.int64)

    def labels(self) -> np.ndarray:
        return self.find_many(np.arange(self.n, dtype=np.int64))

    @classmethod
    def from_labels(
        cls,
        labels: np.ndarray,
        *,
        q: int,
        selector: SentinelSelector,
        merge_min_score: np.ndarray | None = None,
        verification_pairs: np.ndarray | None = None,
    ) -> "ClusterState":
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        state = cls(n, q)
        if n == 0:
            return state

        # Canonicalize every cluster root to its minimum positional member.
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        starts = np.r_[0, 1 + np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1])]
        ends = np.r_[starts[1:], len(order)]
        for s, e in zip(starts, ends):
            members = np.sort(order[s:e].astype(np.int64, copy=False))
            root = int(members[0])
            state.parent[members] = root
            state.parent[root] = root
            state.size[root] = len(members)
            if len(members) > 1:
                state._members[root] = members
            sent, radius = selector.select(members)
            state.sentinel[root].fill(-1)
            state.sentinel[root, : len(sent)] = sent
            state.sentinel_count[root] = len(sent)
            state.cover_radius[root] = np.float32(radius)
            if merge_min_score is not None:
                old_roots = labels[members]
                vals = np.asarray(merge_min_score, dtype=np.float32)[old_roots]
                state.merge_min_score[root] = np.float32(np.min(vals, initial=1.0))
            if verification_pairs is not None:
                old_roots = np.unique(labels[members])
                state.verification_pairs[root] = int(
                    np.asarray(verification_pairs, dtype=np.int64)[old_roots].sum(dtype=np.int64)
                )
        return state

    def union(
        self,
        root_a: int,
        root_b: int,
        *,
        selector: SentinelSelector,
        merge_score: float,
        probed_pairs: int,
    ) -> int:
        ra, rb = self.find(root_a), self.find(root_b)
        if ra == rb:
            return ra
        root, other = (ra, rb) if ra < rb else (rb, ra)
        ma = self.members(ra)
        mb = self.members(rb)
        merged = np.concatenate((ma, mb)).astype(np.int64, copy=False)
        merged.sort()
        sent, radius = selector.select(merged)

        self.parent[other] = root
        self.parent[root] = root
        self.size[root] = len(merged)
        self._members[root] = merged
        self._members.pop(other, None)
        self.sentinel[root].fill(-1)
        self.sentinel[root, : len(sent)] = sent
        self.sentinel_count[root] = len(sent)
        self.cover_radius[root] = np.float32(radius)
        self.merge_min_score[root] = np.float32(
            min(float(self.merge_min_score[ra]), float(self.merge_min_score[rb]), float(merge_score))
        )
        self.verification_pairs[root] = (
            int(self.verification_pairs[ra]) + int(self.verification_pairs[rb]) + int(probed_pairs)
        )
        return root


class DynamicRejectionIndex:
    """Sparse in-memory vetoes discovered after the static rejection index was built."""

    def __init__(self, n: int) -> None:
        self.adj: dict[int, set[int]] = {}
        self.marks = np.zeros(int(n), dtype=np.uint32)
        self.token = np.uint32(0)

    def add(self, a: int, b: int) -> None:
        a, b = int(a), int(b)
        if a == b:
            return
        self.adj.setdefault(a, set()).add(b)
        self.adj.setdefault(b, set()).add(a)

    def add_many(self, a: np.ndarray, b: np.ndarray) -> None:
        for x, y in zip(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)):
            self.add(int(x), int(y))

    def _next_token(self) -> np.uint32:
        self.token = np.uint32(self.token + np.uint32(1))
        if self.token == 0:
            self.marks.fill(0)
            self.token = np.uint32(1)
        return self.token

    def find_cross(self, members_a: np.ndarray, members_b: np.ndarray) -> tuple[int, int] | None:
        a = np.asarray(members_a, dtype=np.int64)
        b = np.asarray(members_b, dtype=np.int64)
        if not len(a) or not len(b) or not self.adj:
            return None
        if len(a) <= len(b):
            target, scan = a, b
        else:
            target, scan = b, a
        token = self._next_token()
        self.marks[target] = token
        for src in scan:
            nbrs = self.adj.get(int(src))
            if not nbrs:
                continue
            for dst in nbrs:
                if self.marks[int(dst)] == token:
                    return int(src), int(dst)
        return None


def sentinel_cross_pairs(
    sent_a: np.ndarray,
    sent_b: np.ndarray,
    *,
    embeddings: np.ndarray,
    embedding_rows_by_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sa = np.asarray(sent_a, dtype=np.int64)
    sb = np.asarray(sent_b, dtype=np.int64)
    pa = np.repeat(sa, len(sb))
    pb = np.tile(sb, len(sa))
    xa = np.asarray(embeddings[embedding_rows_by_position[pa]], dtype=np.float32)
    xb = np.asarray(embeddings[embedding_rows_by_position[pb]], dtype=np.float32)
    cosine = np.einsum("ij,ij->i", xa, xb, dtype=np.float32)
    return pa, pb, cosine.astype(np.float32, copy=False)


def trace_kind(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        first = path.open("r", encoding="utf-8").readline().strip().split("\t")
    except Exception:
        return None
    if first[:3] == ["round", "root_a", "root_b"]:
        return "legacy_rounds"
    if first and first[0] == "event":
        return "frontier"
    return "unknown"


def _legacy_backup_path(path: Path) -> Path:
    return path.with_name(path.stem + ".legacy-rounds" + path.suffix)


def recover_legacy_trace(
    trace_path: Path,
    *,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | None:
    """Replay accepted v11 round-based merges without any model calls."""
    if trace_kind(trace_path) != "legacy_rounds":
        return None

    parent = np.arange(n, dtype=np.int64)
    min_score = np.ones(n, dtype=np.float32)
    verification_pairs = np.zeros(n, dtype=np.int64)

    def find(x: int) -> int:
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    accepted = rows = 0
    with trace_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows += 1
            try:
                is_accepted = int(row.get("accepted", "0") or 0) == 1
            except Exception:
                is_accepted = False
            if not is_accepted:
                continue
            try:
                a = int(row["root_a"]); b = int(row["root_b"])
            except Exception:
                continue
            if not (0 <= a < n and 0 <= b < n):
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            root, other = (ra, rb) if ra < rb else (rb, ra)
            try:
                score = float(row.get("min_score", "1") or 1.0)
            except Exception:
                score = 1.0
            try:
                probed = int(row.get("sentinel_pairs", "0") or 0)
            except Exception:
                probed = 0
            parent[other] = root
            min_score[root] = np.float32(min(float(min_score[ra]), float(min_score[rb]), score))
            verification_pairs[root] = int(verification_pairs[ra]) + int(verification_pairs[rb]) + probed
            accepted += 1

    labels = np.arange(n, dtype=np.int64)
    for i in range(n):
        labels[i] = find(i)

    # Preserve the legacy trace before the new schema starts writing.
    backup = _legacy_backup_path(trace_path)
    if not backup.exists():
        os.replace(trace_path, backup)
    else:
        trace_path.unlink(missing_ok=True)
    return labels, min_score, verification_pairs, {
        "legacy_trace_rows": rows,
        "legacy_accepted_merges": accepted,
    }


def recover_legacy_mapping(
    mapping_path: Path,
    *,
    raw_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | None:
    """Fallback for a completed old clustering when no partial trace is available."""
    if not mapping_path.exists():
        return None
    try:
        with np.load(mapping_path, allow_pickle=False) as d:
            old_raw = np.asarray(d["raw_statement_id"], dtype=np.int64)
            old_cluster = np.asarray(d["cluster_statement_id"], dtype=np.int64)
    except Exception:
        return None
    if len(old_raw) != len(raw_ids):
        return None
    order = np.argsort(old_raw, kind="stable")
    if not np.array_equal(old_raw[order], raw_ids):
        return None
    cluster = old_cluster[order]
    labels = np.empty(len(raw_ids), dtype=np.int64)
    for cid in np.unique(cluster):
        members = np.flatnonzero(cluster == cid)
        labels[members] = int(members.min())
    return labels, np.ones(len(raw_ids), dtype=np.float32), np.zeros(len(raw_ids), dtype=np.int64), {
        "legacy_mapping_clusters": int(len(np.unique(cluster))),
    }


def _checkpoint_state_id(*, implementation: str, rerank_artifact_id: str, threshold: float, q: int, n: int) -> str:
    return stable_hash({
        "implementation": implementation,
        "rerank_artifact_id": rerank_artifact_id,
        "threshold": float(threshold),
        "sentinels": int(q),
        "statements": int(n),
    })


def save_checkpoint(
    *,
    checkpoint_npz: Path,
    checkpoint_json: Path,
    implementation: str,
    rerank_artifact_id: str,
    threshold: float,
    q: int,
    uf: ClusterState,
    seed_cursor: int,
    retry_heap: list[tuple[float, int, int, int, float]],
    counters: dict[str, int],
    trace_rows: int,
    progress_rows: int,
    store: PairScoreStore,
) -> None:
    store.flush()
    checkpoint_npz.parent.mkdir(parents=True, exist_ok=True)
    labels = uf.labels()
    if retry_heap:
        retry = sorted(retry_heap)
        retry_score = np.asarray([-x[0] for x in retry], dtype=np.float32)
        retry_index = np.asarray([x[1] for x in retry], dtype=np.int64)
        retry_a = np.asarray([x[2] for x in retry], dtype=np.int64)
        retry_b = np.asarray([x[3] for x in retry], dtype=np.int64)
        retry_cos = np.asarray([x[4] for x in retry], dtype=np.float32)
    else:
        retry_score = np.empty(0, dtype=np.float32)
        retry_index = np.empty(0, dtype=np.int64)
        retry_a = np.empty(0, dtype=np.int64)
        retry_b = np.empty(0, dtype=np.int64)
        retry_cos = np.empty(0, dtype=np.float32)

    tmp = checkpoint_npz.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        labels=labels,
        seed_cursor=np.asarray(int(seed_cursor), dtype=np.int64),
        retry_score=retry_score,
        retry_index=retry_index,
        retry_a=retry_a,
        retry_b=retry_b,
        retry_cosine=retry_cos,
        merge_min_score=uf.merge_min_score,
        verification_pairs=uf.verification_pairs,
        trace_rows=np.asarray(int(trace_rows), dtype=np.int64),
        progress_rows=np.asarray(int(progress_rows), dtype=np.int64),
    )
    tmp.replace(checkpoint_npz)
    atomic_write_json(checkpoint_json, {
        "complete": True,
        "state_id": _checkpoint_state_id(
            implementation=implementation,
            rerank_artifact_id=rerank_artifact_id,
            threshold=threshold,
            q=q,
            n=uf.n,
        ),
        "seed_cursor": int(seed_cursor),
        "retry_candidates": int(len(retry_heap)),
        "clusters": int(len(uf.roots())),
        "counters": {k: int(v) for k, v in counters.items()},
        "trace_rows": int(trace_rows),
        "progress_rows": int(progress_rows),
    })


def load_checkpoint(
    *,
    checkpoint_npz: Path,
    checkpoint_json: Path,
    implementation: str,
    rerank_artifact_id: str,
    threshold: float,
    q: int,
    n: int,
    selector: SentinelSelector,
) -> tuple[ClusterState, int, list[tuple[float, int, int, int, float]], dict[str, int], int, int] | None:
    meta = load_json(checkpoint_json)
    expected = _checkpoint_state_id(
        implementation=implementation,
        rerank_artifact_id=rerank_artifact_id,
        threshold=threshold,
        q=q,
        n=n,
    )
    if not (meta.get("complete") and meta.get("state_id") == expected and checkpoint_npz.exists()):
        return None
    try:
        with np.load(checkpoint_npz, allow_pickle=False) as d:
            labels = np.asarray(d["labels"], dtype=np.int64)
            cursor = int(d["seed_cursor"])
            rscore = np.asarray(d["retry_score"], dtype=np.float32)
            rindex = np.asarray(d["retry_index"], dtype=np.int64)
            ra = np.asarray(d["retry_a"], dtype=np.int64)
            rb = np.asarray(d["retry_b"], dtype=np.int64)
            rcos = np.asarray(d["retry_cosine"], dtype=np.float32)
            min_score = np.asarray(d["merge_min_score"], dtype=np.float32)
            vpairs = np.asarray(d["verification_pairs"], dtype=np.int64)
            trace_rows = int(d["trace_rows"]) if "trace_rows" in d.files else 0
            progress_rows = int(d["progress_rows"]) if "progress_rows" in d.files else 0
    except Exception:
        return None
    if len(labels) != n:
        return None
    uf = ClusterState.from_labels(
        labels,
        q=q,
        selector=selector,
        merge_min_score=min_score,
        verification_pairs=vpairs,
    )
    retry: list[tuple[float, int, int, int, float]] = []
    for s, idx, a, b, c in zip(rscore, rindex, ra, rb, rcos):
        retry.append((-float(s), int(idx), int(a), int(b), float(c)))
    heapq.heapify(retry)
    counters = {k: int(v) for k, v in (meta.get("counters") or {}).items()}
    return uf, cursor, retry, counters, trace_rows, progress_rows


def truncate_tsv(path: Path, data_rows: int) -> None:
    if not path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".truncate.tmp")
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        header = src.readline()
        if header:
            dst.write(header)
        for i, line in enumerate(src):
            if i >= data_rows:
                break
            dst.write(line)
    os.replace(tmp, path)


def _proposal_from_heap_or_seed(
    *,
    retry_heap: list[tuple[float, int, int, int, float]],
    seed_a: np.ndarray,
    seed_b: np.ndarray,
    seed_scores: np.ndarray,
    seed_cosines: np.ndarray,
    seed_cursor: int,
) -> tuple[SeedProposal | None, int, bool]:
    """Pop the highest-priority proposal across retries and the monotone seed stream."""
    have_retry = bool(retry_heap)
    have_seed = seed_cursor < len(seed_scores)
    if not have_retry and not have_seed:
        return None, seed_cursor, False

    if have_retry:
        retry_score = -float(retry_heap[0][0])
    else:
        retry_score = -np.inf
    seed_score = float(seed_scores[seed_cursor]) if have_seed else -np.inf

    if have_retry and retry_score >= seed_score:
        neg, idx, a, b, cos = heapq.heappop(retry_heap)
        return SeedProposal(int(a), int(b), float(-neg), float(cos), int(idx)), seed_cursor, True

    i = int(seed_cursor)
    prop = SeedProposal(
        int(seed_a[i]), int(seed_b[i]), float(seed_scores[i]), float(seed_cosines[i]), i
    )
    return prop, i + 1, False


def frontier_sentinel_agglomeration(
    *,
    implementation: str,
    rerank_artifact_id: str,
    raw_ids: np.ndarray,
    seed_a_pos: np.ndarray,
    seed_b_pos: np.ndarray,
    seed_scores: np.ndarray,
    seed_cosines: np.ndarray,
    embeddings: np.ndarray,
    embedding_rows_by_position: np.ndarray,
    threshold: float,
    q: int,
    store: PairScoreStore,
    static_rejections: KnownRejectionIndex,
    score_missing: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    candidate_batch_size: int,
    selector: SentinelSelector,
    uf: ClusterState,
    seed_cursor: int,
    retry_heap: list[tuple[float, int, int, int, float]],
    counters: dict[str, int],
    trace_path: Path | None,
    progress_path: Path | None,
    checkpoint_npz: Path,
    checkpoint_json: Path,
    checkpoint_every_waves: int,
    trace_rows: int = 0,
    progress_rows: int = 0,
    max_merge_attempts: int | None = None,
    max_new_reranker_pairs: int | None = None,
) -> tuple[ClusterState, dict[str, int | float | bool | str]]:
    """Frontier-driven sentinel clustering with exact frontier exhaustion.

    Initial seed edges are consumed once in descending reranker-score order.
    Candidate merges that pass sentinel verification but conflict with another
    accepted merge in the same GPU wave are put back into a priority frontier.
    A rejected merge creates a permanent statement-level veto, so it never needs
    to be reconsidered for any supercluster containing those statements.

    Convergence is exact for this greedy algorithm when both the seed stream and
    retry frontier are exhausted. No global round/recontraction is performed.
    """
    n = len(raw_ids)
    dynamic_rejections = DynamicRejectionIndex(n)
    candidate_batch_size = max(1, int(candidate_batch_size))
    checkpoint_every_waves = max(1, int(checkpoint_every_waves))

    default_counters = {
        "waves": 0,
        "seed_edges_consumed": int(seed_cursor),
        "attempted_merges": 0,
        "accepted_merges": 0,
        "rejected_merges": 0,
        "known_rejection_vetoes": 0,
        "cached_sentinel_rejections": 0,
        "sentinel_rejections": 0,
        "passed_but_retried": 0,
        "new_reranker_pairs": 0,
    }
    for key, value in default_counters.items():
        counters.setdefault(key, value)
    counters.setdefault("current_clusters", int(len(uf.roots())))

    trace_file = trace_writer = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        exists = trace_path.exists() and trace_path.stat().st_size > 0
        trace_file = trace_path.open("a", encoding="utf-8", newline="")
        trace_writer = csv.writer(trace_file, delimiter="\t")
        if not exists:
            trace_writer.writerow([
                "event", "wave", "seed_index", "root_a", "root_b", "size_a", "size_b",
                "seed_score", "seed_cosine", "sentinels_a", "sentinels_b", "sentinel_pairs",
                "known_pairs", "new_pairs", "min_score", "accepted", "reason", "veto_a", "veto_b",
            ])
            trace_file.flush()

    progress_file = progress_writer = None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        exists = progress_path.exists() and progress_path.stat().st_size > 0
        progress_file = progress_path.open("a", encoding="utf-8", newline="")
        progress_writer = csv.writer(progress_file, delimiter="\t")
        if not exists:
            progress_writer.writerow([
                "wave", "seed_cursor", "seed_total", "retry_frontier", "clusters",
                "candidates", "accepted", "rejected", "retried", "new_reranker_pairs",
            ])
            progress_file.flush()

    progress = tqdm(
        total=len(seed_scores),
        initial=min(seed_cursor, len(seed_scores)),
        desc="cluster:frontier:seed-edges",
        unit="edge",
    )

    converged = False
    stop_reason = "frontier_exhausted"
    try:
        while seed_cursor < len(seed_scores) or retry_heap:
            counters["waves"] += 1
            wave = int(counters["waves"])
            wave_candidates: list[SeedProposal] = []
            root_pair_to_idx: dict[tuple[int, int], int] = {}
            consumed_this_wave = 0

            # Pull a bounded number of *unique current cluster pairs* from the
            # combined priority frontier. Overlapping cluster pairs are allowed;
            # all are verified in one GPU batch and conflicts are resolved only
            # after scores are known.
            while len(wave_candidates) < candidate_batch_size:
                prop, new_cursor, from_retry = _proposal_from_heap_or_seed(
                    retry_heap=retry_heap,
                    seed_a=seed_a_pos,
                    seed_b=seed_b_pos,
                    seed_scores=seed_scores,
                    seed_cosines=seed_cosines,
                    seed_cursor=seed_cursor,
                )
                if prop is None:
                    break
                if not from_retry:
                    consumed = new_cursor - seed_cursor
                    seed_cursor = new_cursor
                    consumed_this_wave += consumed
                    counters["seed_edges_consumed"] = int(seed_cursor)
                ra, rb = uf.find(prop.a_pos), uf.find(prop.b_pos)
                if ra == rb:
                    continue
                key = (ra, rb) if ra < rb else (rb, ra)
                old_idx = root_pair_to_idx.get(key)
                if old_idx is None:
                    root_pair_to_idx[key] = len(wave_candidates)
                    wave_candidates.append(prop)
                else:
                    # Same current cluster pair: one sentinel decision is enough.
                    # Keep the stronger seed proposal for deterministic priority.
                    if prop.seed_score > wave_candidates[old_idx].seed_score:
                        wave_candidates[old_idx] = prop

            if consumed_this_wave:
                progress.update(consumed_this_wave)
            if not wave_candidates:
                # The stream may contain only edges internal to already-merged
                # clusters. Continue until both stream and retry heap are empty.
                continue

            candidate_state: list[dict[str, object]] = []
            rejected_wave = 0
            new_scores_wave = 0

            for prop in wave_candidates:
                ra, rb = uf.find(prop.a_pos), uf.find(prop.b_pos)
                if ra == rb:
                    continue
                ma, mb = uf.members(ra), uf.members(rb)
                veto = static_rejections.find_cross_rejection(ma, mb)
                if veto is None:
                    veto = dynamic_rejections.find_cross(ma, mb)
                if veto is not None:
                    counters["attempted_merges"] += 1
                    counters["rejected_merges"] += 1
                    counters["known_rejection_vetoes"] += 1
                    rejected_wave += 1
                    if trace_writer is not None:
                        trace_writer.writerow([
                            "decision", wave, prop.seed_index, ra, rb, len(ma), len(mb),
                            prop.seed_score, prop.seed_cosine, len(uf.sentinels(ra)), len(uf.sentinels(rb)),
                            0, 0, 0, "", 0, "known_rejection_veto", int(veto[0]), int(veto[1]),
                        ])
                        trace_rows += 1
                    continue

                sent_a = uf.sentinels(ra)
                sent_b = uf.sentinels(rb)
                pa, pb, cosine = sentinel_cross_pairs(
                    sent_a, sent_b,
                    embeddings=embeddings,
                    embedding_rows_by_position=embedding_rows_by_position,
                )
                keys = pack_pair_keys(raw_ids[pa], raw_ids[pb])
                scores, _ = store.lookup(keys)
                finite = np.isfinite(scores)
                failed_cached = finite & (scores < threshold)
                if np.any(failed_cached):
                    bad_idx = np.flatnonzero(failed_cached)
                    dynamic_rejections.add_many(pa[bad_idx], pb[bad_idx])
                    counters["attempted_merges"] += 1
                    counters["rejected_merges"] += 1
                    counters["cached_sentinel_rejections"] += 1
                    rejected_wave += 1
                    bad = int(bad_idx[np.argmin(scores[bad_idx])])
                    if trace_writer is not None:
                        trace_writer.writerow([
                            "decision", wave, prop.seed_index, ra, rb, len(ma), len(mb),
                            prop.seed_score, prop.seed_cosine, len(sent_a), len(sent_b), len(pa),
                            int(finite.sum()), 0, float(np.nanmin(scores)), 0,
                            "cached_sentinel_rejection", int(pa[bad]), int(pb[bad]),
                        ])
                        trace_rows += 1
                    continue

                candidate_state.append({
                    "proposal": prop,
                    "root_a": ra,
                    "root_b": rb,
                    "members_a": ma,
                    "members_b": mb,
                    "sent_a": sent_a,
                    "sent_b": sent_b,
                    "pa": pa,
                    "pb": pb,
                    "cosine": cosine,
                    "keys": keys,
                    "scores": scores,
                    "known": int(finite.sum()),
                })

            # Score every *unique* missing sentinel pair across the whole wave.
            all_keys: list[np.ndarray] = []
            all_raw_a: list[np.ndarray] = []
            all_raw_b: list[np.ndarray] = []
            all_cos: list[np.ndarray] = []
            owners: list[np.ndarray] = []
            locals_: list[np.ndarray] = []
            for owner, st in enumerate(candidate_state):
                scores = np.asarray(st["scores"], dtype=np.float32)
                missing = np.flatnonzero(~np.isfinite(scores))
                if not len(missing):
                    continue
                pa = np.asarray(st["pa"], dtype=np.int64)
                pb = np.asarray(st["pb"], dtype=np.int64)
                cosine = np.asarray(st["cosine"], dtype=np.float32)
                keys = np.asarray(st["keys"], dtype=np.uint64)
                all_keys.append(keys[missing])
                all_raw_a.append(raw_ids[pa[missing]])
                all_raw_b.append(raw_ids[pb[missing]])
                all_cos.append(cosine[missing])
                owners.append(np.full(len(missing), owner, dtype=np.int64))
                locals_.append(missing.astype(np.int64, copy=False))

            if all_keys:
                key_cat = np.concatenate(all_keys)
                raw_a_cat = np.concatenate(all_raw_a)
                raw_b_cat = np.concatenate(all_raw_b)
                cos_cat = np.concatenate(all_cos)
                owner_cat = np.concatenate(owners)
                local_cat = np.concatenate(locals_)

                # Stable unique keeps the first occurrence's raw pair/cosine and
                # lets one cross-encoder score feed every candidate that needs it.
                unique_keys, first, inverse = np.unique(key_cat, return_index=True, return_inverse=True)
                unique_scores = score_missing(raw_a_cat[first], raw_b_cat[first], cos_cat[first])
                counters["new_reranker_pairs"] += len(unique_scores)
                new_scores_wave += len(unique_scores)
                expanded = unique_scores[inverse]
                for owner in range(len(candidate_state)):
                    take = np.flatnonzero(owner_cat == owner)
                    if not len(take):
                        continue
                    st = candidate_state[owner]
                    scores = np.asarray(st["scores"], dtype=np.float32)
                    scores[local_cat[take]] = expanded[take]
                    st["scores"] = scores
                    failed = take[expanded[take] < threshold]
                    if len(failed):
                        pa = np.asarray(st["pa"], dtype=np.int64)
                        pb = np.asarray(st["pb"], dtype=np.int64)
                        loc = local_cat[failed]
                        dynamic_rejections.add_many(pa[loc], pb[loc])

            passed: list[tuple[SeedProposal, dict[str, object], float]] = []
            for st in candidate_state:
                prop = st["proposal"]
                assert isinstance(prop, SeedProposal)
                scores = np.asarray(st["scores"], dtype=np.float32)
                if not np.all(np.isfinite(scores)):
                    raise RuntimeError("Missing sentinel scores after frontier verification")
                min_score = float(scores.min(initial=1.0))
                counters["attempted_merges"] += 1
                if min_score < threshold:
                    bad_idx = np.flatnonzero(scores < threshold)
                    pa = np.asarray(st["pa"], dtype=np.int64)
                    pb = np.asarray(st["pb"], dtype=np.int64)
                    dynamic_rejections.add_many(pa[bad_idx], pb[bad_idx])
                    counters["rejected_merges"] += 1
                    counters["sentinel_rejections"] += 1
                    rejected_wave += 1
                    bad = int(np.argmin(scores))
                    if trace_writer is not None:
                        trace_writer.writerow([
                            "decision", wave, prop.seed_index, int(st["root_a"]), int(st["root_b"]),
                            len(st["members_a"]), len(st["members_b"]), prop.seed_score, prop.seed_cosine,
                            len(st["sent_a"]), len(st["sent_b"]), len(scores), int(st["known"]),
                            len(scores) - int(st["known"]), min_score, 0, "sentinel_rejection",
                            int(pa[bad]), int(pb[bad]),
                        ])
                        trace_rows += 1
                else:
                    passed.append((prop, st, min_score))

            # Greedy maximal matching among passed proposals. This is the only
            # place where simultaneous merges interact. Passed proposals that
            # overlap a selected merge are *not lost*: they return to the retry
            # priority frontier and are revalidated against the new sentinels.
            passed.sort(key=lambda x: (-x[0].seed_score, x[0].seed_index))
            used_roots: set[int] = set()
            selected: list[tuple[SeedProposal, dict[str, object], float]] = []
            retried_wave = 0
            for prop, st, min_score in passed:
                ra, rb = uf.find(int(st["root_a"])), uf.find(int(st["root_b"]))
                if ra == rb:
                    continue
                if ra in used_roots or rb in used_roots:
                    heapq.heappush(
                        retry_heap,
                        (-float(prop.seed_score), int(prop.seed_index), int(prop.a_pos), int(prop.b_pos), float(prop.seed_cosine)),
                    )
                    counters["passed_but_retried"] += 1
                    retried_wave += 1
                    if trace_writer is not None:
                        trace_writer.writerow([
                            "retry", wave, prop.seed_index, ra, rb, len(uf.members(ra)), len(uf.members(rb)),
                            prop.seed_score, prop.seed_cosine, len(uf.sentinels(ra)), len(uf.sentinels(rb)),
                            0, 0, 0, min_score, "", "overlap_with_selected_merge", "", "",
                        ])
                        trace_rows += 1
                    continue
                used_roots.add(ra); used_roots.add(rb)
                selected.append((prop, st, min_score))

            accepted_wave = 0
            for prop, st, min_score in selected:
                ra, rb = uf.find(prop.a_pos), uf.find(prop.b_pos)
                if ra == rb:
                    continue
                # No other selected merge touches these roots, so their sentinels
                # are still exactly the sentinels that were verified above.
                new_root = uf.union(
                    ra,
                    rb,
                    selector=selector,
                    merge_score=min_score,
                    probed_pairs=len(np.asarray(st["scores"])),
                )
                counters["accepted_merges"] += 1
                counters["current_clusters"] -= 1
                accepted_wave += 1
                if trace_writer is not None:
                    trace_writer.writerow([
                        "decision", wave, prop.seed_index, ra, rb, len(st["members_a"]), len(st["members_b"]),
                        prop.seed_score, prop.seed_cosine, len(st["sent_a"]), len(st["sent_b"]),
                        len(st["scores"]), int(st["known"]), len(st["scores"]) - int(st["known"]),
                        min_score, 1, f"all_sentinels_passed:new_root={new_root}", "", "",
                    ])
                    trace_rows += 1

            if progress_writer is not None:
                progress_writer.writerow([
                    wave, seed_cursor, len(seed_scores), len(retry_heap), int(counters["current_clusters"]),
                    len(wave_candidates), accepted_wave, rejected_wave, retried_wave, new_scores_wave,
                ])
                progress_rows += 1

            if trace_file is not None:
                trace_file.flush()
            if progress_file is not None:
                progress_file.flush()

            progress.set_postfix(
                clusters=int(counters["current_clusters"]),
                retry=len(retry_heap),
                acc=accepted_wave,
                rej=rejected_wave,
            )

            if max_merge_attempts is not None and counters["attempted_merges"] >= int(max_merge_attempts):
                stop_reason = "max_merge_attempts"
                break
            if max_new_reranker_pairs is not None and counters["new_reranker_pairs"] >= int(max_new_reranker_pairs):
                stop_reason = "max_new_reranker_pairs"
                break

            if wave % checkpoint_every_waves == 0:
                save_checkpoint(
                    checkpoint_npz=checkpoint_npz,
                    checkpoint_json=checkpoint_json,
                    implementation=implementation,
                    rerank_artifact_id=rerank_artifact_id,
                    threshold=threshold,
                    q=q,
                    uf=uf,
                    seed_cursor=seed_cursor,
                    retry_heap=retry_heap,
                    counters=counters,
                    trace_rows=trace_rows,
                    progress_rows=progress_rows,
                    store=store,
                )

        converged = seed_cursor >= len(seed_scores) and not retry_heap
        if not converged and stop_reason == "frontier_exhausted":
            stop_reason = "budget_stop"

        save_checkpoint(
            checkpoint_npz=checkpoint_npz,
            checkpoint_json=checkpoint_json,
            implementation=implementation,
            rerank_artifact_id=rerank_artifact_id,
            threshold=threshold,
            q=q,
            uf=uf,
            seed_cursor=seed_cursor,
            retry_heap=retry_heap,
            counters=counters,
            trace_rows=trace_rows,
            progress_rows=progress_rows,
            store=store,
        )
    finally:
        progress.close()
        if trace_file is not None:
            trace_file.close()
        if progress_file is not None:
            progress_file.close()

    return uf, {
        **{k: int(v) for k, v in counters.items()},
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "seed_cursor": int(seed_cursor),
        "seed_edges_total": int(len(seed_scores)),
        "remaining_retry_frontier": int(len(retry_heap)),
    }


def recover_frontier_trace(
    trace_path: Path,
    *,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | None:
    """Best-effort recovery when a frontier trace exists but no checkpoint does.

    Only accepted merges are replayed. The seed cursor cannot be reconstructed
    exactly from skipped/internal edges, so callers must restart the seed stream
    from zero. This still preserves all accepted clustering work and every lazy
    reranker score stored on disk.
    """
    if trace_kind(trace_path) != "frontier":
        return None
    parent = np.arange(n, dtype=np.int64)
    min_score = np.ones(n, dtype=np.float32)
    verification_pairs = np.zeros(n, dtype=np.int64)

    def find(x: int) -> int:
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    accepted = rows = 0
    with trace_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows += 1
            try:
                is_accepted = int(row.get("accepted", "0") or 0) == 1
            except Exception:
                is_accepted = False
            if not is_accepted:
                continue
            try:
                a = int(row["root_a"]); b = int(row["root_b"])
            except Exception:
                continue
            if not (0 <= a < n and 0 <= b < n):
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            root, other = (ra, rb) if ra < rb else (rb, ra)
            try:
                score = float(row.get("min_score", "1") or 1.0)
            except Exception:
                score = 1.0
            try:
                probed = int(row.get("sentinel_pairs", "0") or 0)
            except Exception:
                probed = 0
            parent[other] = root
            min_score[root] = np.float32(min(float(min_score[ra]), float(min_score[rb]), score))
            verification_pairs[root] = int(verification_pairs[ra]) + int(verification_pairs[rb]) + probed
            accepted += 1

    labels = np.arange(n, dtype=np.int64)
    for i in range(n):
        labels[i] = find(i)
    return labels, min_score, verification_pairs, {
        "frontier_trace_rows_replayed": rows,
        "frontier_accepted_merges_replayed": accepted,
    }
