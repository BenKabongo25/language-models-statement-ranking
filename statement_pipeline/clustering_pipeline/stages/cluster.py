from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from clustering_pipeline.pair_store import PairScoreStore, pack_pair_keys, unpack_pair_keys
from clustering_pipeline.rejection_index import KnownRejectionIndex
from clustering_pipeline.runtime import is_oom
from clustering_pipeline.frontier_clustering import (
    ClusterState,
    SentinelSelector,
    frontier_sentinel_agglomeration,
    load_checkpoint,
    recover_frontier_trace,
    recover_legacy_mapping,
    recover_legacy_trace,
    trace_kind,
    truncate_tsv,
)
from clustering_pipeline.signatures import (
    CLUSTER_IMPL_VERSION,
    SCHEMA_VERSION,
    cluster_artifact_id,
    cluster_semantics,
)
from clustering_pipeline.storage import (
    STAGE_CLUSTER,
    cluster_trace_path,
    cluster_progress_path,
    cluster_checkpoint_path,
    cluster_checkpoint_manifest_path,
    clusters_path,
    embedding_ids_path,
    embeddings_path,
    mapping_path,
    rerank_dir,
    statements_path,
)
from clustering_pipeline.utils import (
    artifact_id_from_manifest,
    atomic_write_json,
    manifest_current,
    read_jsonl,
    stable_hash,
)


def _load_statements(path: Path, statement_type: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    raw_ids: list[int] = []
    texts: list[str] = []
    counts: list[int] = []
    for _, rec in tqdm(read_jsonl(path), desc=f"cluster:{statement_type}:load-statements", unit="stmt"):
        raw_ids.append(int(rec["raw_statement_id"]))
        texts.append(str(rec["text"]))
        counts.append(int(rec.get("occurrence_count", 1)))
    rid = np.asarray(raw_ids, dtype=np.int64)
    cnt = np.asarray(counts, dtype=np.int64)
    if len(rid) == 0:
        return rid, texts, cnt
    order = np.argsort(rid, kind="stable")
    if not np.array_equal(order, np.arange(len(rid))):
        rid = rid[order]
        cnt = cnt[order]
        texts = [texts[int(i)] for i in order]
    if len(np.unique(rid)) != len(rid):
        raise ValueError(f"Duplicate raw_statement_id values in {path}")
    return rid, texts, cnt


def _raw_to_pos(values: np.ndarray, raw_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    if len(raw_ids) and np.array_equal(raw_ids, np.arange(len(raw_ids), dtype=np.int64)):
        if values.min() < 0 or values.max() >= len(raw_ids):
            raise ValueError("Pair references raw_statement_id outside statement dictionary")
        return values.astype(np.int64, copy=False)
    pos = np.searchsorted(raw_ids, values)
    valid = pos < len(raw_ids)
    clipped = np.minimum(pos, max(0, len(raw_ids) - 1))
    valid &= raw_ids[clipped] == values if len(raw_ids) else False
    if not np.all(valid):
        raise ValueError(f"Unknown raw_statement_id values: {values[~valid][:10].tolist()}")
    return pos.astype(np.int64, copy=False)


def _embedding_rows_by_statement_position(raw_ids: np.ndarray, emb_ids: np.ndarray) -> np.ndarray:
    emb_ids = np.asarray(emb_ids, dtype=np.int64)
    if len(raw_ids) != len(emb_ids):
        raise ValueError("Embedding ids and statement dictionary have different lengths")
    if np.array_equal(raw_ids, emb_ids):
        return np.arange(len(raw_ids), dtype=np.int64)
    order = np.argsort(emb_ids, kind="stable")
    sorted_ids = emb_ids[order]
    pos = np.searchsorted(sorted_ids, raw_ids)
    if np.any(pos >= len(sorted_ids)) or not np.array_equal(sorted_ids[pos], raw_ids):
        raise ValueError("Embeddings are missing one or more raw_statement_id values")
    return order[pos].astype(np.int64, copy=False)


def _load_seed_edges(
    *,
    output_root: Path,
    statement_type: str,
    raw_ids: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load initial reranker-positive pairs only.

    Lazy scores from older clustering runs are deliberately *not* used as seed
    proposals. They may be reused if the new deterministic sentinel algorithm
    asks for exactly the same pair, but they cannot change which merges are
    proposed. This keeps clustering reproducible independently of cache history.
    """
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    initial_pairs = 0
    for path in tqdm(
        sorted(rerank_dir(output_root, statement_type).glob("part-*.npz")),
        desc=f"cluster:{statement_type}:load-seeds",
        unit="shard",
    ):
        with np.load(path, allow_pickle=False) as d:
            a = np.asarray(d["a"], dtype=np.int64)
            b = np.asarray(d["b"], dtype=np.int64)
            score = np.asarray(d["score"], dtype=np.float32)
            cosine = np.asarray(d["cosine"], dtype=np.float32)
        initial_pairs += len(a)
        keep = score >= threshold
        if np.any(keep):
            a_parts.append(a[keep])
            b_parts.append(b[keep])
            s_parts.append(score[keep])
            c_parts.append(cosine[keep])

    if not a_parts:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy(), initial_pairs

    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    score = np.concatenate(s_parts)
    cosine = np.concatenate(c_parts)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keys = pack_pair_keys(lo, hi)

    # Highest reranker score wins if the same undirected pair appears twice.
    order = np.lexsort((-cosine, -score, keys))
    keys = keys[order]
    lo = lo[order]
    hi = hi[order]
    score = score[order]
    cosine = cosine[order]
    keep = np.r_[True, keys[1:] != keys[:-1]]
    lo = lo[keep]
    hi = hi[keep]
    score = score[keep]
    cosine = cosine[keep]

    # Global descending score order is the deterministic proposal priority.
    order = np.argsort(-score, kind="stable")
    lo = lo[order]
    hi = hi[order]
    score = score[order]
    cosine = cosine[order]
    return _raw_to_pos(lo, raw_ids), _raw_to_pos(hi, raw_ids), score, cosine, initial_pairs


def _coarse_positive_components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast connected components of the positive seed graph.

    Components are used only to prune the static initial-rejection index. Final
    clusters are produced by sentinel verification, never by connected components.
    """
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if len(a) == 0:
        return np.arange(n, dtype=np.int64)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception as exc:  # pragma: no cover - environment error path
        raise RuntimeError("scipy>=1.11 is required for fast coarse component construction") from exc

    data = np.ones(len(a) * 2, dtype=np.uint8)
    row = np.concatenate((a, b)).astype(np.int64, copy=False)
    col = np.concatenate((b, a)).astype(np.int64, copy=False)
    graph = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64, copy=False)


class LazyCrossEncoderScorer:
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        statement_type: str,
        raw_ids: np.ndarray,
        texts: list[str],
        store: PairScoreStore,
        batch_size: int,
    ) -> None:
        self.cfg = cfg
        self.statement_type = statement_type
        self.raw_ids = raw_ids
        self.texts = texts
        self.store = store
        self.model = None
        self.batch_size = max(1, int(batch_size))
        self.instruction = str((cfg["reranker"].get("instructions") or {})[statement_type])
        self.total_scored = 0

    def _ensure_model(self):
        if self.model is None:
            from clustering_pipeline.stages.rerank import Qwen3Reranker

            self.model = Qwen3Reranker(self.cfg["reranker"], self.cfg.get("runtime", {}))
        return self.model

    def _texts_for_raw(self, values: np.ndarray) -> list[str]:
        pos = _raw_to_pos(values, self.raw_ids)
        return [self.texts[int(p)] for p in pos]

    def __call__(self, raw_a: np.ndarray, raw_b: np.ndarray, cosine: np.ndarray) -> np.ndarray:
        raw_a = np.asarray(raw_a, dtype=np.int64)
        raw_b = np.asarray(raw_b, dtype=np.int64)
        cosine = np.asarray(cosine, dtype=np.float32)
        if len(raw_a) == 0:
            return np.empty(0, dtype=np.float32)
        out = np.empty(len(raw_a), dtype=np.float32)
        model = self._ensure_model()
        ta = self._texts_for_raw(raw_a)
        tb = self._texts_for_raw(raw_b)
        pos = 0
        current_bs = self.batch_size
        while pos < len(raw_a):
            bs = min(current_bs, len(raw_a) - pos)
            pairs = list(zip(ta[pos : pos + bs], tb[pos : pos + bs]))
            try:
                scores = model.score(pairs, self.instruction)
            except Exception as exc:
                if bs > 1 and is_oom(exc):
                    current_bs = max(1, bs // 2)
                    self.batch_size = current_bs
                    continue
                raise
            out[pos : pos + bs] = scores
            keys = pack_pair_keys(raw_a[pos : pos + bs], raw_b[pos : pos + bs])
            self.store.add(keys, scores, cosine[pos : pos + bs])
            self.total_scored += bs
            pos += bs
        return out


def _iter_score_segments(store: PairScoreStore) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    yield store.base.keys, store.base.scores
    for seg in store.segments:
        yield seg.keys, seg.scores


def _verify_no_observed_rejection_inside_clusters(
    *,
    store: PairScoreStore,
    raw_ids: np.ndarray,
    raw_to_cluster: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    checked = 0
    violations = 0
    example: tuple[int, int, float] | None = None
    for keys, scores in tqdm(list(_iter_score_segments(store)), desc="cluster:verify-vetoes", unit="segment"):
        scores = np.asarray(scores, dtype=np.float32)
        rejected = scores < threshold
        if not np.any(rejected):
            continue
        rk = np.asarray(keys, dtype=np.uint64)[rejected]
        rs = scores[rejected]
        ra, rb = unpack_pair_keys(rk)
        pa = _raw_to_pos(ra, raw_ids)
        pb = _raw_to_pos(rb, raw_ids)
        same = raw_to_cluster[pa] == raw_to_cluster[pb]
        checked += len(ra)
        if np.any(same):
            violations += int(same.sum())
            if example is None:
                i = int(np.flatnonzero(same)[0])
                example = (int(ra[i]), int(rb[i]), float(rs[i]))
    if violations:
        raise RuntimeError(
            f"Clustering violated {violations} observed reranker rejection(s) below {threshold:.3f}; "
            f"example={example}. This indicates an internal clustering bug."
        )
    return checked, violations



def _archive_file(path: Path, tag: str) -> Path | None:
    if not path.exists():
        return None
    candidate = path.with_name(f"{path.stem}.{tag}{path.suffix}")
    if candidate.exists():
        i = 1
        while True:
            alt = path.with_name(f"{path.stem}.{tag}.{i}{path.suffix}")
            if not alt.exists():
                candidate = alt
                break
            i += 1
    path.replace(candidate)
    return candidate


def _recover_previous_state(
    *,
    output_root: Path,
    statement_type: str,
    raw_ids: np.ndarray,
    selector: SentinelSelector,
    q: int,
    trace: Path,
    progress: Path,
    checkpoint_npz: Path,
    checkpoint_json: Path,
    implementation: str,
    rerank_id: str,
    threshold: float,
    resume_previous_state: bool,
) -> tuple[ClusterState, int, list[tuple[float, int, int, int, float]], dict[str, int], int, int, str, dict[str, Any]]:
    n = len(raw_ids)
    if not resume_previous_state:
        _archive_file(trace, "previous")
        _archive_file(progress, "previous")
        checkpoint_npz.unlink(missing_ok=True)
        checkpoint_json.unlink(missing_ok=True)
        return ClusterState(n, q), 0, [], {}, 0, 0, "fresh", {}

    loaded = load_checkpoint(
        checkpoint_npz=checkpoint_npz,
        checkpoint_json=checkpoint_json,
        implementation=implementation,
        rerank_artifact_id=rerank_id,
        threshold=threshold,
        q=q,
        n=n,
        selector=selector,
    )
    if loaded is not None:
        uf, cursor, retry, counters, trace_rows, progress_rows = loaded
        # Discard any trace/progress rows written after the last atomic checkpoint.
        truncate_tsv(trace, trace_rows)
        truncate_tsv(progress, progress_rows)
        return uf, cursor, retry, counters, trace_rows, progress_rows, "checkpoint", {
            "checkpoint_seed_cursor": int(cursor),
            "checkpoint_retry_frontier": int(len(retry)),
        }

    kind = trace_kind(trace)
    if kind == "legacy_rounds":
        recovered = recover_legacy_trace(trace, n=n)
        if recovered is not None:
            labels, min_score, vpairs, meta = recovered
            uf = ClusterState.from_labels(
                labels,
                q=q,
                selector=selector,
                merge_min_score=min_score,
                verification_pairs=vpairs,
            )
            # Old round logs are provenance only; keep them but never read them again.
            old_rounds = output_root / STAGE_CLUSTER / f"rounds_{statement_type}.tsv"
            _archive_file(old_rounds, "legacy-rounds")
            counters = {
                "accepted_merges": int(meta.get("legacy_accepted_merges", 0)),
                "recovered_accepted_merges": int(meta.get("legacy_accepted_merges", 0)),
            }
            return uf, 0, [], counters, 0, 0, "legacy_round_trace", meta

    if kind == "frontier":
        recovered = recover_frontier_trace(trace, n=n)
        if recovered is not None:
            labels, min_score, vpairs, meta = recovered
            uf = ClusterState.from_labels(
                labels,
                q=q,
                selector=selector,
                merge_min_score=min_score,
                verification_pairs=vpairs,
            )
            _archive_file(trace, "replayed-no-checkpoint")
            _archive_file(progress, "replayed-no-checkpoint")
            counters = {
                "accepted_merges": int(meta.get("frontier_accepted_merges_replayed", 0)),
                "recovered_accepted_merges": int(meta.get("frontier_accepted_merges_replayed", 0)),
            }
            return uf, 0, [], counters, 0, 0, "frontier_trace_replay", meta

    # Last-resort migration for a completed old cluster stage without a usable trace.
    recovered = recover_legacy_mapping(mapping_path(output_root, statement_type), raw_ids=raw_ids)
    if recovered is not None:
        labels, min_score, vpairs, meta = recovered
        uf = ClusterState.from_labels(
            labels,
            q=q,
            selector=selector,
            merge_min_score=min_score,
            verification_pairs=vpairs,
        )
        _archive_file(trace, "unrecognized")
        _archive_file(progress, "previous")
        return uf, 0, [], {"recovered_accepted_merges": int(n - len(uf.roots()))}, 0, 0, "legacy_final_mapping", meta

    _archive_file(trace, "unrecognized")
    _archive_file(progress, "previous")
    return ClusterState(n, q), 0, [], {}, 0, 0, "fresh", {}


def run_cluster(*, output_root: Path, cfg: dict[str, Any], statement_type: str, force: bool = False) -> None:
    ccfg = cfg["cluster"]
    threshold = float(cfg["reranker"].get("threshold", 0.95))
    algorithm = str(ccfg.get("algorithm", "sentinel_frontier_agglomerative"))
    if algorithm != "sentinel_frontier_agglomerative":
        raise ValueError("cluster.algorithm must be sentinel_frontier_agglomerative")
    q = max(1, int(ccfg.get("sentinels", 4)))
    resume_previous_state = bool(ccfg.get("resume_previous_state", True))

    rerank_id = artifact_id_from_manifest(output_root / "03_rerank" / f"{statement_type}.manifest.json")
    artifact_id = cluster_artifact_id(rerank_id, cfg, statement_type)
    manifest_path = output_root / STAGE_CLUSTER / f"{statement_type}.manifest.json"
    out_clusters = clusters_path(output_root, statement_type)
    out_mapping = mapping_path(output_root, statement_type)
    required = [out_clusters, out_mapping]
    if not force and manifest_current(manifest_path, artifact_id, required):
        print(f"cluster:{statement_type}: up to date")
        return

    (output_root / STAGE_CLUSTER).mkdir(parents=True, exist_ok=True)
    raw_ids, texts, counts = _load_statements(statements_path(output_root, statement_type), statement_type)
    n = len(raw_ids)
    if n == 0:
        out_clusters.write_text("", encoding="utf-8")
        np.savez(
            out_mapping,
            raw_statement_id=np.empty(0, dtype=np.int64),
            cluster_statement_id=np.empty(0, dtype=np.int64),
        )
        atomic_write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "artifact_id": artifact_id,
                "source_artifact_id": rerank_id,
                "semantics": cluster_semantics(cfg, statement_type),
                "clusters": 0,
                "statements": 0,
                "converged": True,
                "stop_reason": "empty",
            },
        )
        return

    embeddings = np.load(embeddings_path(output_root, statement_type), mmap_mode="r")
    emb_ids = np.load(embedding_ids_path(output_root, statement_type), mmap_mode="r")
    emb_rows = _embedding_rows_by_statement_position(raw_ids, emb_ids)
    selector = SentinelSelector(embeddings, emb_rows, q=q)

    store = PairScoreStore(
        output_root=output_root,
        statement_type=statement_type,
        rerank_artifact_id=rerank_id,
        flush_pairs=int(ccfg.get("cache_flush_pairs", 100_000)),
    )

    seed_a, seed_b, seed_scores, seed_cosines, initial_pairs = _load_seed_edges(
        output_root=output_root,
        statement_type=statement_type,
        raw_ids=raw_ids,
        threshold=threshold,
    )
    print(
        f"cluster:{statement_type}: statements={n:,} initial_pairs={initial_pairs:,} "
        f"seed_pairs(score>={threshold:.3f})={len(seed_scores):,} q={q}",
        flush=True,
    )

    print(f"cluster:{statement_type}: building coarse positive blocks", flush=True)
    coarse_labels = _coarse_positive_components(n, seed_a, seed_b)
    coarse_source_id = stable_hash({
        "implementation": "positive-blocks-v1",
        "rerank_artifact_id": rerank_id,
        "threshold": threshold,
    })
    rejection_index = KnownRejectionIndex(
        root=output_root,
        statement_type=statement_type,
        raw_ids=raw_ids,
        threshold=threshold,
        store=store,
        coarse_labels=coarse_labels,
        coarse_source_id=coarse_source_id,
        include_lazy=True,
    )

    scorer = LazyCrossEncoderScorer(
        cfg=cfg,
        statement_type=statement_type,
        raw_ids=raw_ids,
        texts=texts,
        store=store,
        batch_size=max(1, int(ccfg.get("verification_batch_size", 512))),
    )

    trace = cluster_trace_path(output_root, statement_type)
    progress_path = cluster_progress_path(output_root, statement_type)
    checkpoint_npz = cluster_checkpoint_path(output_root, statement_type)
    checkpoint_json = cluster_checkpoint_manifest_path(output_root, statement_type)

    uf, seed_cursor, retry_heap, counters, trace_rows, progress_rows, resume_source, resume_meta = _recover_previous_state(
        output_root=output_root,
        statement_type=statement_type,
        raw_ids=raw_ids,
        selector=selector,
        q=q,
        trace=trace,
        progress=progress_path,
        checkpoint_npz=checkpoint_npz,
        checkpoint_json=checkpoint_json,
        implementation=CLUSTER_IMPL_VERSION,
        rerank_id=rerank_id,
        threshold=threshold,
        resume_previous_state=resume_previous_state,
    )
    initial_clusters = len(uf.roots())
    print(
        f"cluster:{statement_type}: initialization={resume_source}; clusters={initial_clusters:,}; "
        f"seed_cursor={seed_cursor:,}/{len(seed_scores):,}; retry_frontier={len(retry_heap):,}",
        flush=True,
    )

    max_attempts = ccfg.get("max_merge_attempts")
    max_new_pairs = ccfg.get("max_new_reranker_pairs")
    merge_stats: dict[str, Any]
    try:
        uf, merge_stats = frontier_sentinel_agglomeration(
            implementation=CLUSTER_IMPL_VERSION,
            rerank_artifact_id=rerank_id,
            raw_ids=raw_ids,
            seed_a_pos=seed_a,
            seed_b_pos=seed_b,
            seed_scores=seed_scores,
            seed_cosines=seed_cosines,
            embeddings=embeddings,
            embedding_rows_by_position=emb_rows,
            threshold=threshold,
            q=q,
            store=store,
            static_rejections=rejection_index,
            score_missing=scorer,
            candidate_batch_size=max(1, int(ccfg.get("candidate_batch_size", 2048))),
            selector=selector,
            uf=uf,
            seed_cursor=seed_cursor,
            retry_heap=retry_heap,
            counters=counters,
            trace_path=trace if bool(ccfg.get("write_merge_trace", True)) else None,
            progress_path=progress_path,
            checkpoint_npz=checkpoint_npz,
            checkpoint_json=checkpoint_json,
            checkpoint_every_waves=max(1, int(ccfg.get("checkpoint_every_waves", 10))),
            trace_rows=trace_rows,
            progress_rows=progress_rows,
            max_merge_attempts=None if max_attempts in (None, "null") else int(max_attempts),
            max_new_reranker_pairs=None if max_new_pairs in (None, "null") else int(max_new_pairs),
        )
    finally:
        store.close()

    if not bool(merge_stats.get("converged")):
        raise RuntimeError(
            f"cluster:{statement_type} stopped before frontier exhaustion: {merge_stats.get('stop_reason')}. "
            f"State was checkpointed and can be resumed by rerunning the same stage."
        )

    roots = uf.roots()
    roots = roots[np.argsort([int(raw_ids[uf.members(int(r))].min()) for r in roots], kind="stable")]
    raw_to_cluster = np.full(n, -1, dtype=np.int64)
    include_members = bool(ccfg.get("include_members", True))
    include_stats = bool(ccfg.get("include_cluster_stats", True))

    tmp_clusters = out_clusters.with_suffix(".tmp.jsonl")
    with tmp_clusters.open("w", encoding="utf-8") as f:
        for cid, root in enumerate(tqdm(roots, desc=f"cluster:{statement_type}:finalize", unit="cluster")):
            root = int(root)
            members_pos = uf.members(root)
            members_pos = members_pos[np.argsort(raw_ids[members_pos], kind="stable")]
            raw_to_cluster[members_pos] = cid
            sent = uf.sentinels(root)
            rep_pos = int(sent[0]) if len(sent) else int(members_pos[0])
            rec: dict[str, Any] = {
                "cluster_statement_id": cid,
                "type": statement_type,
                "text": texts[rep_pos],
                "representative_raw_statement_id": int(raw_ids[rep_pos]),
                "representative_method": "embedding_medoid_sentinel",
                "member_count": int(len(members_pos)),
                "occurrence_count": int(counts[members_pos].sum()),
            }
            if include_members:
                rec["member_raw_statement_ids"] = raw_ids[members_pos].astype(np.int64).tolist()
            if include_stats:
                rec.update(
                    {
                        "verification_kind": "sentinel_frontier_verified",
                        "sentinel_count": int(len(sent)),
                        "sentinel_raw_statement_ids": raw_ids[sent].astype(np.int64).tolist(),
                        "embedding_cover_radius": float(uf.cover_radius[root]),
                        "min_merge_sentinel_score": float(uf.merge_min_score[root]),
                        "merge_verification_pair_count": int(uf.verification_pairs[root]),
                    }
                )
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_clusters.replace(out_clusters)

    if np.any(raw_to_cluster < 0):
        raise RuntimeError("Internal clustering error: some statements were not assigned to a cluster")

    checked_rejections, violations = _verify_no_observed_rejection_inside_clusters(
        store=store,
        raw_ids=raw_ids,
        raw_to_cluster=raw_to_cluster,
        threshold=threshold,
    )

    tmp_mapping = out_mapping.with_suffix(".tmp.npz")
    np.savez(
        tmp_mapping,
        raw_statement_id=raw_ids,
        cluster_statement_id=raw_to_cluster,
        threshold=np.asarray(threshold, dtype=np.float32),
        artifact_id=np.asarray(artifact_id),
    )
    tmp_mapping.replace(out_mapping)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "implementation": CLUSTER_IMPL_VERSION,
        "complete": True,
        "artifact_id": artifact_id,
        "source_artifact_id": rerank_id,
        "semantics": cluster_semantics(cfg, statement_type),
        "threshold": threshold,
        "sentinels": q,
        "statements": n,
        "clusters": int(len(roots)),
        "initialization": resume_source,
        "initial_clusters": int(initial_clusters),
        "resume_metadata": resume_meta,
        "coarse_positive_components": int(len(np.unique(coarse_labels))),
        "initial_reranked_pairs": int(initial_pairs),
        "seed_edges": int(len(seed_scores)),
        **merge_stats,
        "lazy_verification_pairs_scored_this_run": int(scorer.total_scored),
        "lazy_verification_pairs_cached_total": int(store.cached_lazy_pairs),
        "observed_rejections_checked": int(checked_rejections),
        "observed_rejection_violations": int(violations),
        "verification_cache": str(store.cache_dir),
        "known_initial_rejection_index": str(rejection_index.dir),
        "checkpoint": str(checkpoint_npz),
        "quality_policy": (
            "Initial reranker-positive edges are consumed once in descending score order. Current clusters are represented "
            "by q embedding sentinels (medoid + farthest-first). A candidate merge is accepted iff every cross-sentinel "
            "score is at least the reranker threshold. Passed but overlapping candidates return to a priority retry frontier. "
            "Every observed rejection is a permanent veto. Convergence is reached exactly when the seed stream and retry "
            "frontier are both exhausted; no global clustering rounds are used."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"cluster:{statement_type}: done — clusters={len(roots):,}; accepted_merges={merge_stats['accepted_merges']:,}; "
        f"seed_edges_consumed={merge_stats['seed_edges_consumed']:,}/{len(seed_scores):,}; "
        f"new_reranker_pairs={scorer.total_scored:,}; violations={violations}",
        flush=True,
    )
