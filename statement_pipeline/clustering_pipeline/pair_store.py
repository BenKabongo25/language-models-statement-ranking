from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

from clustering_pipeline.storage import rerank_dir, verification_cache_dir
from clustering_pipeline.utils import atomic_write_json, load_json, reset_dir

UNKNOWN_SCORE = np.float32(np.nan)
_LAZY_RE = re.compile(r"^lazy-part-(\d+)\.npz$")


def pack_pair_keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if len(a) == 0:
        return np.empty(0, dtype=np.uint64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    max_id = max(int(lo.max(initial=0)), int(hi.max(initial=0)))
    if max_id >= (1 << 32):
        raise ValueError(
            "Pair-key packing currently supports raw_statement_id < 2^32; "
            f"observed maximum {max_id}."
        )
    return (lo.astype(np.uint64) << np.uint64(32)) | hi.astype(np.uint64)


def unpack_pair_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.uint64)
    a = (keys >> np.uint64(32)).astype(np.int64)
    b = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    return a, b


@dataclass(slots=True)
class SortedSegment:
    keys: np.ndarray
    scores: np.ndarray
    cosines: np.ndarray | None = None

    def lookup(self, query: np.ndarray, out: np.ndarray, cosine_out: np.ndarray | None = None) -> None:
        if len(self.keys) == 0 or len(query) == 0:
            return
        missing = ~np.isfinite(out)
        if not np.any(missing):
            return
        idxs = np.flatnonzero(missing)
        q = query[idxs]
        pos = np.searchsorted(self.keys, q)
        valid = pos < len(self.keys)
        if not np.any(valid):
            return
        vv = np.flatnonzero(valid)
        matched = self.keys[pos[vv]] == q[vv]
        if not np.any(matched):
            return
        take = vv[matched]
        dst = idxs[take]
        src = pos[take]
        out[dst] = self.scores[src]
        if cosine_out is not None and self.cosines is not None:
            cosine_out[dst] = self.cosines[src]


def _validate_lazy_npz(path: Path, *, materialize: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate a lazy-score shard.

    A lazy shard is a disposable memoization artifact, not scientific source data.
    Interrupted older pipeline versions may leave a zero-byte/truncated NPZ.  Such
    a shard must never abort checkpoint recovery: missing scores can be recomputed
    lazily if they are needed again.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        with np.load(path, allow_pickle=False) as d:
            required = {"key", "score", "cosine"}
            if not required.issubset(d.files):
                return None
            keys = np.asarray(d["key"], dtype=np.uint64)
            scores = np.asarray(d["score"], dtype=np.float32)
            cosines = np.asarray(d["cosine"], dtype=np.float32)
        if not (keys.ndim == scores.ndim == cosines.ndim == 1):
            return None
        if not (len(keys) == len(scores) == len(cosines)):
            return None
        if len(keys) > 1 and np.any(keys[1:] < keys[:-1]):
            return None
        if materialize:
            # Detach from the ZipFile before leaving the validation routine.
            keys = np.array(keys, dtype=np.uint64, copy=True)
            scores = np.array(scores, dtype=np.float32, copy=True)
            cosines = np.array(cosines, dtype=np.float32, copy=True)
        return keys, scores, cosines
    except (EOFError, OSError, ValueError, KeyError, zipfile_error_types()):
        return None
    except Exception:
        # np.load can surface backend-specific zip/decompression exceptions.
        return None


def zipfile_error_types():
    # Kept as a helper to avoid importing zipfile on the hot lookup path.
    import zipfile
    return zipfile.BadZipFile


def _quarantine(path: Path, *, reason: str = "corrupt") -> Path:
    qdir = path.parent / "corrupt"
    qdir.mkdir(parents=True, exist_ok=True)
    target = qdir / f"{path.name}.{reason}"
    suffix = 1
    while target.exists():
        target = qdir / f"{path.name}.{reason}.{suffix}"
        suffix += 1
    try:
        os.replace(path, target)
    except OSError:
        shutil.move(str(path), str(target))
    return target


def _next_lazy_part(cache_dir: Path) -> int:
    max_part = -1
    for path in cache_dir.glob("lazy-part-*.npz"):
        m = _LAZY_RE.match(path.name)
        if m:
            max_part = max(max_part, int(m.group(1)))
    return max_part + 1


def _durable_npz_replace(out: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ atomically and fsync file+directory before returning."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}"
    try:
        # File handle avoids numpy silently appending '.npz' to a temporary name.
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        try:
            dfd = os.open(str(out.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


class PairScoreStore:
    """Read-through score cache for initial and lazily verified reranker pairs.

    Initial reranker outputs are indexed once as sorted uint64 pair keys. New
    lazy verification scores are checkpointed into sorted NPZ segments. Lazy
    segments are *cache*: a corrupt/interrupted segment is quarantined and its
    scores are recomputed only if later requested. This lets a valid clustering
    checkpoint survive a partially written historical cache shard.
    """

    def __init__(
        self,
        *,
        output_root: Path,
        statement_type: str,
        rerank_artifact_id: str,
        flush_pairs: int = 100_000,
    ) -> None:
        self.output_root = output_root
        self.statement_type = statement_type
        self.rerank_artifact_id = rerank_artifact_id
        self.flush_pairs = max(1, int(flush_pairs))
        self.cache_dir = verification_cache_dir(output_root, statement_type)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.base_keys_path = self.cache_dir / "base_keys.npy"
        self.base_scores_path = self.cache_dir / "base_scores.npy"
        self.base_cosines_path = self.cache_dir / "base_cosines.npy"
        self.pending: dict[int, tuple[float, float]] = {}
        self.segments: list[SortedSegment] = []
        self.lazy_pairs_written = 0
        self.quarantined_shards: list[str] = []
        self._ensure_compatible_cache()
        self._ensure_base_index()
        self._load_base_or_rebuild()
        self._load_lazy_segments()

    def _ensure_compatible_cache(self) -> None:
        manifest = load_json(self.manifest_path)
        cached_source = manifest.get("rerank_artifact_id")
        has_lazy = self.cache_dir.exists() and any(self.cache_dir.glob("lazy-part-*.npz"))
        if self.cache_dir.exists() and (
            (cached_source is not None and cached_source != self.rerank_artifact_id)
            or (cached_source is None and has_lazy)
        ):
            reset_dir(self.cache_dir)
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _base_index_current(self) -> bool:
        manifest = load_json(self.manifest_path)
        return bool(
            manifest.get("rerank_artifact_id") == self.rerank_artifact_id
            and manifest.get("base_complete")
            and self.base_keys_path.exists()
            and self.base_scores_path.exists()
            and self.base_cosines_path.exists()
        )

    def _invalidate_base_index(self) -> None:
        for p in (self.base_keys_path, self.base_scores_path, self.base_cosines_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        manifest = load_json(self.manifest_path)
        manifest["base_complete"] = False
        atomic_write_json(self.manifest_path, manifest)

    def _ensure_base_index(self) -> None:
        if self._base_index_current():
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shards = sorted(rerank_dir(self.output_root, self.statement_type).glob("part-*.npz"))
        counts: list[int] = []
        total = 0
        for path in tqdm(shards, desc=f"cluster:{self.statement_type}:index-rerank-count", unit="shard"):
            with np.load(path, allow_pickle=False) as d:
                n = len(d["a"])
            counts.append(n)
            total += n

        tmp_keys = self.cache_dir / "base_keys.unsorted.npy"
        tmp_scores = self.cache_dir / "base_scores.unsorted.npy"
        tmp_cos = self.cache_dir / "base_cosines.unsorted.npy"
        keys_mm = np.lib.format.open_memmap(tmp_keys, mode="w+", dtype=np.uint64, shape=(total,))
        score_mm = np.lib.format.open_memmap(tmp_scores, mode="w+", dtype=np.float32, shape=(total,))
        cos_mm = np.lib.format.open_memmap(tmp_cos, mode="w+", dtype=np.float32, shape=(total,))

        cursor = 0
        for path, n in tqdm(list(zip(shards, counts)), desc=f"cluster:{self.statement_type}:index-rerank-fill", unit="shard"):
            if n == 0:
                continue
            with np.load(path, allow_pickle=False) as d:
                a = np.asarray(d["a"], dtype=np.int64)
                b = np.asarray(d["b"], dtype=np.int64)
                score = np.asarray(d["score"], dtype=np.float32)
                cosine = np.asarray(d["cosine"], dtype=np.float32)
            sl = slice(cursor, cursor + n)
            keys_mm[sl] = pack_pair_keys(a, b)
            score_mm[sl] = score
            cos_mm[sl] = cosine
            cursor += n
        keys_mm.flush(); score_mm.flush(); cos_mm.flush()

        if total:
            order = np.argsort(np.asarray(keys_mm), kind="stable")
            sorted_keys = np.asarray(keys_mm)[order]
            sorted_scores = np.asarray(score_mm)[order]
            sorted_cos = np.asarray(cos_mm)[order]
            first = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
            if np.all(first):
                np.save(self.base_keys_path, sorted_keys)
                np.save(self.base_scores_path, sorted_scores)
                np.save(self.base_cosines_path, sorted_cos)
                unique_count = total
            else:
                starts = np.flatnonzero(first)
                ends = np.r_[starts[1:], len(sorted_keys)]
                uk = sorted_keys[starts]
                us = np.empty(len(starts), dtype=np.float32)
                uc = np.empty(len(starts), dtype=np.float32)
                for i, (s, e) in enumerate(zip(starts, ends)):
                    us[i] = float(np.max(sorted_scores[s:e]))
                    uc[i] = float(np.max(sorted_cos[s:e]))
                np.save(self.base_keys_path, uk)
                np.save(self.base_scores_path, us)
                np.save(self.base_cosines_path, uc)
                unique_count = len(uk)
        else:
            np.save(self.base_keys_path, np.empty(0, dtype=np.uint64))
            np.save(self.base_scores_path, np.empty(0, dtype=np.float32))
            np.save(self.base_cosines_path, np.empty(0, dtype=np.float32))
            unique_count = 0

        del keys_mm, score_mm, cos_mm
        for p in (tmp_keys, tmp_scores, tmp_cos):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        manifest = load_json(self.manifest_path)
        manifest.update(
            {
                "rerank_artifact_id": self.rerank_artifact_id,
                "base_complete": True,
                "base_pairs": int(unique_count),
                "base_shards": len(shards),
                "lazy_shards": int(manifest.get("lazy_shards", 0)),
                "lazy_pairs": int(manifest.get("lazy_pairs", 0)),
            }
        )
        atomic_write_json(self.manifest_path, manifest)

    def _load_base_or_rebuild(self) -> None:
        try:
            keys = np.load(self.base_keys_path, mmap_mode="r")
            scores = np.load(self.base_scores_path, mmap_mode="r")
            cosines = np.load(self.base_cosines_path, mmap_mode="r")
            if not (len(keys) == len(scores) == len(cosines)):
                raise ValueError("base pair-score arrays have inconsistent lengths")
            self.base = SortedSegment(keys, scores, cosines)
        except Exception as exc:
            print(
                f"cluster:{self.statement_type}: pair-score base index is unreadable ({exc}); rebuilding from 03_rerank",
                flush=True,
            )
            self._invalidate_base_index()
            self._ensure_base_index()
            self.base = SortedSegment(
                np.load(self.base_keys_path, mmap_mode="r"),
                np.load(self.base_scores_path, mmap_mode="r"),
                np.load(self.base_cosines_path, mmap_mode="r"),
            )

    def _reconcile_lazy_manifest(self, *, valid_pairs: int, valid_shards: int) -> None:
        manifest = load_json(self.manifest_path)
        manifest.update(
            {
                "rerank_artifact_id": self.rerank_artifact_id,
                "base_complete": True,
                "lazy_shards": int(valid_shards),
                "lazy_pairs": int(valid_pairs),
                "quarantined_lazy_shards": int(manifest.get("quarantined_lazy_shards", 0)) + len(self.quarantined_shards),
            }
        )
        if self.quarantined_shards:
            manifest["last_quarantined_lazy_shards"] = self.quarantined_shards[-20:]
        atomic_write_json(self.manifest_path, manifest)

    def _load_lazy_segments(self) -> None:
        self.segments.clear()
        self.quarantined_shards.clear()
        valid_pairs = 0
        paths = sorted(self.cache_dir.glob("lazy-part-*.npz"))
        for path in paths:
            loaded = _validate_lazy_npz(path)
            if loaded is None:
                target = _quarantine(path)
                self.quarantined_shards.append(target.name)
                print(
                    f"cluster:{self.statement_type}: WARNING quarantined unreadable lazy cache shard "
                    f"{path.name} -> corrupt/{target.name}; missing scores will be recomputed on demand",
                    flush=True,
                )
                continue
            keys, scores, cosines = loaded
            self.segments.append(SortedSegment(keys, scores, cosines))
            valid_pairs += len(keys)
        self._reconcile_lazy_manifest(valid_pairs=valid_pairs, valid_shards=len(self.segments))

    def lookup(self, keys: np.ndarray, *, with_cosine: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        keys = np.asarray(keys, dtype=np.uint64)
        scores = np.full(len(keys), np.nan, dtype=np.float32)
        cosines = np.full(len(keys), np.nan, dtype=np.float32) if with_cosine else None
        self.base.lookup(keys, scores, cosines)
        for seg in self.segments:
            seg.lookup(keys, scores, cosines)
            if np.all(np.isfinite(scores)):
                break
        if self.pending:
            missing = np.flatnonzero(~np.isfinite(scores))
            for idx in missing:
                value = self.pending.get(int(keys[idx]))
                if value is not None:
                    scores[idx] = np.float32(value[0])
                    if cosines is not None:
                        cosines[idx] = np.float32(value[1])
        return scores, cosines

    def add(self, keys: np.ndarray, scores: np.ndarray, cosines: np.ndarray) -> None:
        keys = np.asarray(keys, dtype=np.uint64)
        scores = np.asarray(scores, dtype=np.float32)
        cosines = np.asarray(cosines, dtype=np.float32)
        if not (len(keys) == len(scores) == len(cosines)):
            raise ValueError("PairScoreStore.add received arrays with different lengths")
        for k, s, c in zip(keys, scores, cosines):
            self.pending[int(k)] = (float(s), float(c))
        if len(self.pending) >= self.flush_pairs:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        keys = np.fromiter(self.pending.keys(), dtype=np.uint64, count=len(self.pending))
        vals = list(self.pending.values())
        scores = np.fromiter((x[0] for x in vals), dtype=np.float32, count=len(vals))
        cosines = np.fromiter((x[1] for x in vals), dtype=np.float32, count=len(vals))
        order = np.argsort(keys, kind="stable")
        keys = keys[order]; scores = scores[order]; cosines = cosines[order]
        a, b = unpack_pair_keys(keys)
        part = _next_lazy_part(self.cache_dir)
        out = self.cache_dir / f"lazy-part-{part:06d}.npz"
        _durable_npz_replace(out, key=keys, a=a, b=b, score=scores, cosine=cosines)
        self.segments.append(SortedSegment(keys.copy(), scores.copy(), cosines.copy()))
        self.lazy_pairs_written += len(keys)
        self.pending.clear()
        manifest = load_json(self.manifest_path)
        manifest.update(
            {
                "rerank_artifact_id": self.rerank_artifact_id,
                "base_complete": True,
                "lazy_shards": len(self.segments),
                "lazy_pairs": int(sum(len(seg.keys) for seg in self.segments)),
                "next_lazy_part": part + 1,
            }
        )
        atomic_write_json(self.manifest_path, manifest)

    def close(self) -> None:
        self.flush()

    @property
    def cached_lazy_pairs(self) -> int:
        return int(sum(len(seg.keys) for seg in self.segments)) + len(self.pending)


def iter_lazy_pair_shards(output_root: Path, statement_type: str) -> Iterator[Path]:
    """Yield only readable lazy shards.

    Audit is read-only and therefore does not quarantine files itself. The cluster
    stage performs quarantine/reconciliation when PairScoreStore is constructed.
    """
    cache_dir = verification_cache_dir(output_root, statement_type)
    for path in sorted(cache_dir.glob("lazy-part-*.npz")):
        if _validate_lazy_npz(path, materialize=False) is None:
            print(
                f"audit:{statement_type}: WARNING skipping unreadable lazy cache shard {path}",
                flush=True,
            )
            continue
        yield path
