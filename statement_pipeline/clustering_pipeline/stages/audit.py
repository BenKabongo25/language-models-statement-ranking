from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from clustering_pipeline.pair_store import iter_lazy_pair_shards, pack_pair_keys
from clustering_pipeline.runtime import choose_device
from clustering_pipeline.signatures import AUDIT_IMPL_VERSION
from clustering_pipeline.storage import (
    STAGE_ANALYSIS,
    clusters_path,
    embedding_ids_path,
    embeddings_path,
    mapping_path,
    neighbors_dir,
    pair_analysis_path,
    pair_analysis_sample_path,
    rank_meta_dir,
    rerank_dir,
    statements_path,
    verification_cache_dir,
)
from clustering_pipeline.utils import atomic_write_json, load_json, read_jsonl, stable_hash

UNKNOWN_RANK = -2
NOT_IN_TOPK = -1

COLUMNS = [
    "pair_id", "type", "pair_source",
    "statement_a_raw_id", "statement_a_text", "statement_a_occurrence_count",
    "statement_b_raw_id", "statement_b_text", "statement_b_occurrence_count",
    "cosine_similarity", "rank_b_in_a_topk", "rank_a_in_b_topk",
    "b_in_a_topk", "a_in_b_topk", "mutual_topk",
    "reranker_score", "threshold", "passes_threshold",
    "cluster_id_a", "cluster_id_b", "same_cluster",
    "canonical_statement_id_a", "canonical_statement_id_b",
    "cluster_size_if_same", "cluster_representative_raw_id_if_same",
    "cluster_representative_text_if_same", "a_is_cluster_representative",
    "b_is_cluster_representative", "relation_category",
]


def _file_state(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _dir_state(path: Path, pattern: str) -> dict[str, int] | None:
    if not path.exists():
        return None
    files = list(path.glob(pattern))
    if not files:
        return {"files": 0, "total_size": 0, "max_mtime_ns": 0}
    stats = [p.stat() for p in files]
    return {
        "files": len(files),
        "total_size": int(sum(s.st_size for s in stats)),
        "max_mtime_ns": int(max(s.st_mtime_ns for s in stats)),
    }


def _audit_id(output_root: Path, cfg: dict[str, Any], typ: str, rank_mode: str) -> str:
    return stable_hash({
        "implementation": AUDIT_IMPL_VERSION,
        "type": typ,
        "threshold": float(cfg["reranker"].get("threshold", 0.95)),
        "rank_mode": rank_mode,
        "include_text": bool(cfg.get("pair_analysis", {}).get("include_text", True)),
        "rerank": _dir_state(rerank_dir(output_root, typ), "part-*.npz"),
        "lazy": _dir_state(verification_cache_dir(output_root, typ), "lazy-part-*.npz"),
        "clusters": _file_state(mapping_path(output_root, typ)),
        "cluster_dictionary": _file_state(clusters_path(output_root, typ)),
        "rank_meta": _dir_state(rank_meta_dir(output_root, typ), "part-*.npz"),
        "final_offsets": _file_state(output_root / "final" / "id_offsets.json"),
    })


def _load_statements(output_root: Path, typ: str) -> tuple[list[str], np.ndarray]:
    rows = [rec for _, rec in read_jsonl(statements_path(output_root, typ))]
    if not rows:
        return [], np.empty(0, dtype=np.int64)
    max_id = max(int(r["raw_statement_id"]) for r in rows)
    texts = [""] * (max_id + 1)
    counts = np.zeros(max_id + 1, dtype=np.int64)
    for r in rows:
        rid = int(r["raw_statement_id"])
        texts[rid] = str(r.get("text", ""))
        counts[rid] = int(r.get("occurrence_count", 1))
    return texts, counts


def _load_pairs(output_root: Path, typ: str, include_lazy: bool) -> tuple[np.ndarray, ...]:
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    src_parts: list[np.ndarray] = []
    rank_ab_parts: list[np.ndarray] = []

    for rp in tqdm(sorted(rerank_dir(output_root, typ).glob("part-*.npz")), desc=f"audit:{typ}:initial", unit="shard"):
        with np.load(rp, allow_pickle=False) as d:
            a = np.asarray(d["a"], dtype=np.int64)
            b = np.asarray(d["b"], dtype=np.int64)
            cosine = np.asarray(d["cosine"], dtype=np.float32)
            score = np.asarray(d["score"], dtype=np.float32)
        rank_ab = np.full(len(a), UNKNOWN_RANK, dtype=np.int32)
        npth = neighbors_dir(output_root, typ) / rp.name
        if npth.exists():
            try:
                with np.load(npth, allow_pickle=False) as nd:
                    if "rank_a_to_b" in nd.files and len(nd["rank_a_to_b"]) == len(a):
                        rank_ab = np.asarray(nd["rank_a_to_b"], dtype=np.int32)
            except Exception:
                pass
        a_parts.append(a); b_parts.append(b); c_parts.append(cosine); s_parts.append(score)
        src_parts.append(np.zeros(len(a), dtype=np.uint8)); rank_ab_parts.append(rank_ab)

    if include_lazy:
        for lp in tqdm(list(iter_lazy_pair_shards(output_root, typ)), desc=f"audit:{typ}:lazy", unit="shard"):
            with np.load(lp, allow_pickle=False) as d:
                a = np.asarray(d["a"], dtype=np.int64)
                b = np.asarray(d["b"], dtype=np.int64)
                cosine = np.asarray(d["cosine"], dtype=np.float32)
                score = np.asarray(d["score"], dtype=np.float32)
            a_parts.append(a); b_parts.append(b); c_parts.append(cosine); s_parts.append(score)
            src_parts.append(np.ones(len(a), dtype=np.uint8))
            rank_ab_parts.append(np.full(len(a), UNKNOWN_RANK, dtype=np.int32))

    if not a_parts:
        return tuple(np.empty(0, dtype=d) for d in (np.int64, np.int64, np.float32, np.float32, np.uint8, np.int32))
    a = np.concatenate(a_parts); b = np.concatenate(b_parts); cosine = np.concatenate(c_parts)
    score = np.concatenate(s_parts); source = np.concatenate(src_parts); rank_ab = np.concatenate(rank_ab_parts)
    return a, b, cosine, score, source, rank_ab


def _fill_ranks_from_sidecars(output_root: Path, typ: str, a: np.ndarray, b: np.ndarray,
                              rank_ab: np.ndarray, rank_ba: np.ndarray) -> bool:
    rdir = rank_meta_dir(output_root, typ)
    manifest = load_json(rdir / "manifest.json")
    if not manifest.get("complete"):
        return False
    sidecars = sorted(rdir.glob("part-*.npz"))
    if not sidecars:
        return False
    keys = pack_pair_keys(a, b)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    for p in tqdm(sidecars, desc=f"audit:{typ}:rank-sidecars", unit="shard"):
        with np.load(p, allow_pickle=False) as d:
            q = np.asarray(d["query"], dtype=np.int64)
            nbr = np.asarray(d["neighbor"], dtype=np.int64)
            rank = np.asarray(d["rank"], dtype=np.int32)
        if not len(q):
            continue
        dkeys = pack_pair_keys(q, nbr)
        loc = np.searchsorted(sorted_keys, dkeys)
        valid = loc < len(sorted_keys)
        idx = np.flatnonzero(valid)
        if not len(idx):
            continue
        idx = idx[sorted_keys[loc[idx]] == dkeys[idx]]
        if not len(idx):
            continue
        pair_idx = order[loc[idx]]
        forward = q[idx] == a[pair_idx]
        rank_ab[pair_idx[forward]] = rank[idx][forward]
        rank_ba[pair_idx[~forward]] = rank[idx][~forward]
    rank_ab[rank_ab == UNKNOWN_RANK] = NOT_IN_TOPK
    rank_ba[rank_ba == UNKNOWN_RANK] = NOT_IN_TOPK
    return True


def _recompute_ranks(output_root: Path, cfg: dict[str, Any], typ: str, a: np.ndarray, b: np.ndarray,
                     rank_ab: np.ndarray, rank_ba: np.ndarray) -> str:
    from clustering_pipeline.stages.neighbors import _faiss_index, _torch_search_resilient
    x = np.load(embeddings_path(output_root, typ), mmap_mode="r")
    ids = np.load(embedding_ids_path(output_root, typ), mmap_mode="r")
    n = len(ids)
    if n <= 1:
        rank_ab[:] = NOT_IN_TOPK; rank_ba[:] = NOT_IN_TOPK
        return "empty"
    ncfg = cfg["neighbors"]
    k_requested = int(ncfg.get("k", 128))
    k_search = min(k_requested + 1, n)
    min_cos = float(ncfg.get("min_cosine", 0.0))
    bs = max(1, int(cfg.get("pair_analysis", {}).get("rank_query_batch_size", 4096)))
    keys = pack_pair_keys(a, b)
    order = np.argsort(keys, kind="stable"); sorted_keys = keys[order]

    requested = str(cfg.get("pair_analysis", {}).get("rank_backend", "auto")).lower()
    backend = "torch"
    device = choose_device(str(cfg.get("runtime", {}).get("device", "auto")),
                           fallback_to_cpu=bool(cfg.get("runtime", {}).get("fallback_to_cpu", True))).device
    index = None
    if requested in {"faiss", "auto"} and device != "cuda":
        try:
            index = _faiss_index(x, ncfg); backend = "faiss"
        except Exception:
            backend = "torch"
    if requested == "faiss" and index is None:
        index = _faiss_index(x, ncfg); backend = "faiss"

    rank_ab[:] = UNKNOWN_RANK; rank_ba[:] = UNKNOWN_RANK
    for start in tqdm(range(0, n, bs), desc=f"audit:{typ}:recompute-ranks", unit="batch"):
        stop = min(start + bs, n)
        if backend == "faiss":
            sims, nbrs = index.search(np.asarray(x[start:stop], dtype=np.float32), k_search)
        else:
            sims, nbrs = _torch_search_resilient(
                x, start, stop, k_search, device=device,
                corpus_block=max(1024, int(ncfg.get("torch_corpus_block", 32768))),
                compute_dtype=str(ncfg.get("torch_compute_dtype", "auto")),
            )
        qrows = np.arange(start, stop, dtype=np.int64)[:, None]
        nonself = (nbrs >= 0) & (nbrs != qrows)
        ranks = np.cumsum(nonself, axis=1, dtype=np.int32)
        valid = nonself & (sims >= min_cos) & (ranks <= k_requested)
        rr, cc = np.nonzero(valid)
        if not len(rr):
            continue
        qraw = np.asarray(ids[qrows[rr, 0]], dtype=np.int64)
        nraw = np.asarray(ids[nbrs[rr, cc]], dtype=np.int64)
        dr = ranks[rr, cc]
        dkeys = pack_pair_keys(qraw, nraw)
        loc = np.searchsorted(sorted_keys, dkeys)
        ok = loc < len(sorted_keys)
        ix = np.flatnonzero(ok)
        if not len(ix):
            continue
        ix = ix[sorted_keys[loc[ix]] == dkeys[ix]]
        pair_idx = order[loc[ix]]
        forward = qraw[ix] == a[pair_idx]
        rank_ab[pair_idx[forward]] = dr[ix][forward]
        rank_ba[pair_idx[~forward]] = dr[ix][~forward]
    rank_ab[rank_ab == UNKNOWN_RANK] = NOT_IN_TOPK
    rank_ba[rank_ba == UNKNOWN_RANK] = NOT_IN_TOPK
    return f"recompute-{backend}-{device}"


def _load_cluster_state(output_root: Path, typ: str, n_raw: int):
    cid = np.full(n_raw, -1, dtype=np.int64)
    reps: list[int] = []; sizes: list[int] = []; rep_texts: list[str] = []
    if mapping_path(output_root, typ).exists():
        with np.load(mapping_path(output_root, typ), allow_pickle=False) as d:
            raw = np.asarray(d["raw_statement_id"], dtype=np.int64)
            cluster = np.asarray(d["cluster_statement_id"], dtype=np.int64)
        valid = (raw >= 0) & (raw < n_raw)
        cid[raw[valid]] = cluster[valid]
    if clusters_path(output_root, typ).exists():
        for _, rec in read_jsonl(clusters_path(output_root, typ)):
            c = int(rec["cluster_statement_id"])
            while len(reps) <= c:
                reps.append(-1); sizes.append(0); rep_texts.append("")
            reps[c] = int(rec.get("representative_raw_statement_id", -1))
            sizes[c] = int(rec.get("member_count", 0)); rep_texts[c] = str(rec.get("text", ""))
    offsets = load_json(output_root / "final" / "id_offsets.json").get("offsets")
    offset = int(offsets[typ]) if isinstance(offsets, dict) and typ in offsets else None
    return cid, reps, sizes, rep_texts, offset


def run_pair_analysis(*, output_root: Path, cfg: dict[str, Any], statement_type: str,
                      force: bool = False, rank_mode_override: str | None = None) -> None:
    acfg = cfg.get("pair_analysis", {})
    out_dir = output_root / STAGE_ANALYSIS; out_dir.mkdir(parents=True, exist_ok=True)
    rank_mode = str(rank_mode_override or acfg.get("rank_mode", "available")).lower()
    if rank_mode == "auto":
        rank_mode = "available"
    if rank_mode not in {"available", "recompute"}:
        raise ValueError("pair_analysis.rank_mode must be available or recompute")
    out_path = pair_analysis_path(output_root, statement_type)
    sample_path = pair_analysis_sample_path(output_root, statement_type)
    manifest_path = out_dir / f"{statement_type}.manifest.json"
    artifact_id = _audit_id(output_root, cfg, statement_type, rank_mode)
    old = load_json(manifest_path)
    if not force and old.get("complete") and old.get("artifact_id") == artifact_id and out_path.exists() and sample_path.exists():
        print(f"audit:{statement_type}: up to date")
        return

    texts, counts = _load_statements(output_root, statement_type)
    a, b, cosine, score, source, rank_ab = _load_pairs(
        output_root, statement_type, bool(acfg.get("include_lazy_verification_pairs", True))
    )
    rank_ba = np.full(len(a), UNKNOWN_RANK, dtype=np.int32)
    rank_source = "unavailable"
    if len(a):
        if _fill_ranks_from_sidecars(output_root, statement_type, a, b, rank_ab, rank_ba):
            rank_source = "sidecars"
        elif rank_mode == "recompute":
            rank_source = _recompute_ranks(output_root, cfg, statement_type, a, b, rank_ab, rank_ba)

    n_raw = len(texts)
    cid, reps, sizes, rep_texts, offset = _load_cluster_state(output_root, statement_type, n_raw)
    threshold = float(cfg["reranker"].get("threshold", 0.95))
    include_text = bool(acfg.get("include_text", True))
    sample_per = max(0, int(acfg.get("sample_per_category", 250)))
    samples: dict[str, list[list[Any]]] = defaultdict(list)
    category_counts: dict[str, int] = defaultdict(int)

    tmp = out_path.with_suffix(".tmp.tsv")
    violations = 0
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(COLUMNS)
        for i in tqdm(range(len(a)), desc=f"audit:{statement_type}:write", unit="pair"):
            ai = int(a[i]); bi = int(b[i]); sc = float(score[i])
            ca = int(cid[ai]) if 0 <= ai < len(cid) else -1
            cb = int(cid[bi]) if 0 <= bi < len(cid) else -1
            same = ca >= 0 and ca == cb
            passed = sc >= threshold
            if passed and same:
                category = "passed_same_cluster"
            elif passed and not same:
                category = "passed_separated"
            elif (not passed) and same:
                category = "threshold_violation"
                violations += 1
            else:
                category = "rejected_separated"
            category_counts[category] += 1
            rab = int(rank_ab[i]); rba = int(rank_ba[i])
            size = sizes[ca] if same and 0 <= ca < len(sizes) else ""
            rep = reps[ca] if same and 0 <= ca < len(reps) else ""
            rep_text = rep_texts[ca] if same and 0 <= ca < len(rep_texts) else ""
            row = [
                i, statement_type, "initial_rerank" if int(source[i]) == 0 else "cluster_verification",
                ai, texts[ai] if include_text and ai < len(texts) else "", int(counts[ai]) if ai < len(counts) else "",
                bi, texts[bi] if include_text and bi < len(texts) else "", int(counts[bi]) if bi < len(counts) else "",
                f"{float(cosine[i]):.6f}", "" if rab < 0 else rab, "" if rba < 0 else rba,
                "" if rab == UNKNOWN_RANK else int(rab > 0), "" if rba == UNKNOWN_RANK else int(rba > 0),
                "" if rab == UNKNOWN_RANK or rba == UNKNOWN_RANK else int(rab > 0 and rba > 0),
                f"{sc:.6f}", f"{threshold:.6f}", int(passed), ca, cb, int(same),
                "" if offset is None or ca < 0 else offset + ca,
                "" if offset is None or cb < 0 else offset + cb,
                size, rep, rep_text, int(same and rep == ai), int(same and rep == bi), category,
            ]
            writer.writerow(row)
            if len(samples[category]) < sample_per:
                samples[category].append(row)
    tmp.replace(out_path)

    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t"); writer.writerow(COLUMNS)
        for category in sorted(samples):
            writer.writerows(samples[category])

    atomic_write_json(manifest_path, {
        "complete": True, "artifact_id": artifact_id, "implementation": AUDIT_IMPL_VERSION,
        "type": statement_type, "pairs": len(a), "threshold": threshold,
        "rank_source": rank_source, "relation_categories": dict(category_counts),
        "threshold_violations": violations,
        "guarantee_check": "threshold_violations must be zero: every reranker-observed pair below threshold is a hard separation constraint.",
    })
    print(f"audit:{statement_type}: pairs={len(a):,} violations={violations} rank_source={rank_source}")
