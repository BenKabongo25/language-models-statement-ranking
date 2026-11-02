from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from clustering_pipeline.pair_store import PairScoreStore, unpack_pair_keys
from clustering_pipeline.utils import atomic_write_json, load_json, reset_dir, stable_hash

_MASK32 = np.uint64(0xFFFFFFFF)


def _raw_to_pos(values: np.ndarray, raw_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    if len(raw_ids) and np.array_equal(raw_ids, np.arange(len(raw_ids), dtype=np.int64)):
        return values.astype(np.int64, copy=False)
    pos = np.searchsorted(raw_ids, values)
    if len(values):
        valid = pos < len(raw_ids)
        clipped = np.minimum(pos, max(0, len(raw_ids) - 1))
        valid &= raw_ids[clipped] == values if len(raw_ids) else False
        if not np.all(valid):
            raise ValueError(f"Unknown raw_statement_id values while building rejection index: {values[~valid][:10].tolist()}")
    return pos.astype(np.int64, copy=False)


def _cache_state(store: PairScoreStore) -> dict[str, int | str]:
    manifest = load_json(store.manifest_path)
    return {
        "rerank_artifact_id": str(store.rerank_artifact_id),
        "base_pairs": int(manifest.get("base_pairs", len(store.base.keys))),
        "lazy_pairs": int(manifest.get("lazy_pairs", 0)),
        "lazy_shards": int(manifest.get("lazy_shards", 0)),
    }


class KnownRejectionIndex:
    """Disk-backed CSR-like index of every *known* pair with score < threshold.

    The index is built from the initial reranker scores plus lazy verification
    scores already cached before the current clustering run. It guarantees that
    an explicitly rejected pair is never silently merged simply because it was
    not selected as a sentinel for the current merge.
    """

    def __init__(
        self,
        *,
        root: Path,
        statement_type: str,
        raw_ids: np.ndarray,
        threshold: float,
        store: PairScoreStore,
        coarse_labels: np.ndarray | None = None,
        coarse_source_id: str | None = None,
        include_lazy: bool = True,
    ) -> None:
        self.raw_ids = np.asarray(raw_ids, dtype=np.int64)
        self.n = len(self.raw_ids)
        if self.n >= (1 << 32):
            raise ValueError("KnownRejectionIndex currently supports fewer than 2^32 statements per type")
        self.threshold = float(threshold)
        self.coarse_labels = None if coarse_labels is None else np.asarray(coarse_labels, dtype=np.int64)
        self.coarse_source_id = None if coarse_source_id is None else str(coarse_source_id)
        self.include_lazy = bool(include_lazy)
        if self.coarse_labels is not None and len(self.coarse_labels) != self.n:
            raise ValueError("coarse_labels length does not match raw_ids")
        self.dir = root / "04_clusters" / f"known_rejections_{statement_type}"
        self.keys_path = self.dir / "directed_keys.npy"
        self.indptr_path = self.dir / "indptr.npy"
        self.manifest_path = self.dir / "manifest.json"
        self._state = _cache_state(store)
        if not self.include_lazy:
            self._state = {
                "rerank_artifact_id": self._state["rerank_artifact_id"],
                "base_pairs": self._state["base_pairs"],
                "lazy_pairs": 0,
                "lazy_shards": 0,
            }
        self.artifact_id = stable_hash({
            "implementation": "known-rejection-index-v1",
            "threshold": self.threshold,
            "n": self.n,
            "raw_id_min": int(self.raw_ids.min(initial=0)),
            "raw_id_max": int(self.raw_ids.max(initial=0)),
            "score_cache": self._state,
            "include_lazy": self.include_lazy,
            "coarse_components": None if self.coarse_labels is None else int(len(np.unique(self.coarse_labels))),
            "coarse_source_id": self.coarse_source_id,
        })
        if not self._current():
            self._build(store)
        self.keys = np.load(self.keys_path, mmap_mode="r")
        self.indptr = np.load(self.indptr_path, mmap_mode="r")
        self.degree = np.diff(self.indptr)
        self._marks = np.zeros(self.n, dtype=np.uint32)
        self._token = np.uint32(0)

    def _current(self) -> bool:
        m = load_json(self.manifest_path)
        return bool(
            m.get("complete")
            and m.get("artifact_id") == self.artifact_id
            and self.keys_path.exists()
            and self.indptr_path.exists()
        )

    def _segments(self, store: PairScoreStore) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        yield store.base.keys, store.base.scores
        if self.include_lazy:
            for seg in store.segments:
                yield seg.keys, seg.scores

    def _rejected_positions(self, keys: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = np.asarray(scores) < self.threshold
        if not np.any(keep):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        raw_a, raw_b = unpack_pair_keys(np.asarray(keys)[keep])
        pa = _raw_to_pos(raw_a, self.raw_ids)
        pb = _raw_to_pos(raw_b, self.raw_ids)
        if self.coarse_labels is not None:
            same = self.coarse_labels[pa] == self.coarse_labels[pb]
            pa = pa[same]; pb = pb[same]
        return pa, pb

    def _count(self, store: PairScoreStore) -> int:
        total = 0
        for keys, scores in self._segments(store):
            pa, _ = self._rejected_positions(keys, scores)
            total += 2 * len(pa)
        return total

    def _build(self, store: PairScoreStore) -> None:
        reset_dir(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        total = self._count(store)
        mm = np.lib.format.open_memmap(self.keys_path, mode="w+", dtype=np.uint64, shape=(total,))
        cursor = 0
        segments = list(self._segments(store))
        for keys, scores in tqdm(segments, desc="cluster:build-known-rejections", unit="segment"):
            pa_i, pb_i = self._rejected_positions(keys, scores)
            if not len(pa_i):
                continue
            pa = pa_i.astype(np.uint64, copy=False)
            pb = pb_i.astype(np.uint64, copy=False)
            n = len(pa)
            mm[cursor : cursor + n] = (pa << np.uint64(32)) | pb
            cursor += n
            mm[cursor : cursor + n] = (pb << np.uint64(32)) | pa
            cursor += n
        if cursor != total:
            raise RuntimeError(f"Known rejection index count mismatch: expected {total}, wrote {cursor}")
        if total:
            # In-place quicksort keeps extra memory modest even for large indexes.
            mm.sort(kind="quicksort")
        mm.flush()
        # For sorted directed keys, source-node ranges are obtained by binary search.
        boundaries = np.arange(self.n + 1, dtype=np.uint64) << np.uint64(32)
        indptr = np.searchsorted(mm, boundaries, side="left").astype(np.int64)
        np.save(self.indptr_path, indptr)
        del mm
        atomic_write_json(self.manifest_path, {
            "complete": True,
            "artifact_id": self.artifact_id,
            "threshold": self.threshold,
            "statements": self.n,
            "directed_rejections": int(total),
            "undirected_rejections": int(total // 2),
            "score_cache": self._state,
            "include_lazy": self.include_lazy,
            "coarse_components": None if self.coarse_labels is None else int(len(np.unique(self.coarse_labels))),
            "coarse_source_id": self.coarse_source_id,
        })

    def _next_token(self) -> np.uint32:
        self._token = np.uint32(self._token + np.uint32(1))
        if self._token == 0:
            self._marks.fill(0)
            self._token = np.uint32(1)
        return self._token

    def find_cross_rejection(self, members_a: np.ndarray, members_b: np.ndarray) -> tuple[int, int] | None:
        """Return one known rejected positional pair crossing A/B, if any."""
        a = np.asarray(members_a, dtype=np.int64)
        b = np.asarray(members_b, dtype=np.int64)
        if len(a) == 0 or len(b) == 0 or len(self.keys) == 0:
            return None

        # Scan the side with the cheaper estimated cost: mark the other cluster,
        # then traverse only known-rejection adjacency lists of the scan side.
        cost_ab = int(len(b) + self.degree[a].sum(dtype=np.int64))
        cost_ba = int(len(a) + self.degree[b].sum(dtype=np.int64))
        if cost_ba < cost_ab:
            scan, target = b, a
        else:
            scan, target = a, b

        token = self._next_token()
        self._marks[target] = token
        for src in scan:
            s = int(src)
            lo = int(self.indptr[s]); hi = int(self.indptr[s + 1])
            if hi <= lo:
                continue
            nbrs = (np.asarray(self.keys[lo:hi]) & _MASK32).astype(np.int64, copy=False)
            hit = np.flatnonzero(self._marks[nbrs] == token)
            if len(hit):
                return s, int(nbrs[int(hit[0])])
        return None
