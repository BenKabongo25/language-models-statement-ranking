from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from clustering_pipeline.compat import legacy_neighbors_usable
from clustering_pipeline.runtime import choose_device
from clustering_pipeline.signatures import SCHEMA_VERSION, neighbors_artifact_id, neighbors_semantics
from clustering_pipeline.storage import STAGE_NEIGHBORS, embedding_ids_path, embeddings_path, neighbors_dir, rank_meta_dir
from clustering_pipeline.utils import (
    artifact_id_from_manifest,
    atomic_write_json,
    load_json,
    manifest_current,
    reset_dir,
)


def _save_npz(path: Path, *, compress: bool, **arrays) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _faiss_index(x: np.ndarray, cfg: dict[str, Any]):
    import faiss

    dim = int(x.shape[1])
    kind = str(cfg.get("index", "hnsw")).lower()
    if kind == "hnsw":
        try:
            index = faiss.IndexHNSWFlat(dim, int(cfg.get("hnsw_m", 32)), faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            index = faiss.IndexHNSWFlat(dim, int(cfg.get("hnsw_m", 32)))
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = int(cfg.get("hnsw_ef_construction", 200))
        index.hnsw.efSearch = int(cfg.get("hnsw_ef_search", 256))
    elif kind == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError("neighbors.index must be 'hnsw' or 'flat'")

    add_bs = max(1, int(cfg.get("add_batch_size", 65536)))
    for start in tqdm(range(0, len(x), add_bs), desc="faiss:add", leave=False):
        index.add(np.asarray(x[start : start + add_bs], dtype=np.float32))
    return index


def _torch_search(
    x: np.ndarray,
    start: int,
    stop: int,
    k: int,
    *,
    device: str,
    corpus_block: int,
    compute_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.float16 if device == "cuda" and compute_dtype in {"auto", "float16"} else torch.float32
    q = torch.as_tensor(np.asarray(x[start:stop], dtype=np.float32), device=device, dtype=dtype)
    qn = q.shape[0]
    top_s = torch.full((qn, k), -torch.inf, device=device, dtype=torch.float32)
    top_i = torch.full((qn, k), -1, device=device, dtype=torch.int64)

    for c0 in range(0, len(x), corpus_block):
        c1 = min(c0 + corpus_block, len(x))
        c = torch.as_tensor(np.asarray(x[c0:c1], dtype=np.float32), device=device, dtype=dtype)
        scores = (q @ c.T).float()
        bk = min(k, scores.shape[1])
        block_s, block_i = torch.topk(scores, k=bk, dim=1, largest=True, sorted=False)
        block_i = block_i.to(torch.int64) + c0
        merged_s = torch.cat((top_s, block_s), dim=1)
        merged_i = torch.cat((top_i, block_i), dim=1)
        top_s, take = torch.topk(merged_s, k=k, dim=1, largest=True, sorted=True)
        top_i = torch.gather(merged_i, 1, take)
        del c, scores, block_s, block_i, merged_s, merged_i, take

    return top_s.cpu().numpy(), top_i.cpu().numpy()

def _torch_search_resilient(
    x: np.ndarray,
    start: int,
    stop: int,
    k: int,
    *,
    device: str,
    corpus_block: int,
    compute_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    block = max(1024, int(corpus_block))
    while True:
        try:
            return _torch_search(
                x, start, stop, k,
                device=device, corpus_block=block, compute_dtype=compute_dtype,
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            text = str(exc).lower()
            if "out of memory" not in text:
                raise
            if device == "cuda":
                torch.cuda.empty_cache()
            if block > 1024:
                block = max(1024, block // 2)
                continue
            if stop - start <= 1:
                raise
            mid = start + (stop - start) // 2
            s1, i1 = _torch_search_resilient(
                x, start, mid, k, device=device, corpus_block=block, compute_dtype=compute_dtype
            )
            s2, i2 = _torch_search_resilient(
                x, mid, stop, k, device=device, corpus_block=block, compute_dtype=compute_dtype
            )
            return np.concatenate((s1, s2), axis=0), np.concatenate((i1, i2), axis=0)


def _valid_modern_shard(path: Path, artifact_id: str, q_start: int, q_stop: int) -> bool:
    if not path.exists():
        return False
    try:
        d = np.load(path, allow_pickle=False)
        if "artifact_id" not in d.files or str(d["artifact_id"].item()) != artifact_id:
            return False
        return int(d["q_start"].item()) == q_start and int(d["q_stop"].item()) == q_stop
    except Exception:
        return False


def _valid_rank_meta_shard(path: Path, artifact_id: str, q_start: int, q_stop: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            return bool(
                "artifact_id" in d.files
                and str(d["artifact_id"].item()) == artifact_id
                and int(d["q_start"].item()) == q_start
                and int(d["q_stop"].item()) == q_stop
                and all(k in d.files for k in ("query", "neighbor", "rank", "cosine"))
            )
    except Exception:
        return False


def run_neighbors(*, output_root: Path, cfg: dict[str, Any], statement_type: str, force: bool = False) -> None:
    ncfg = cfg["neighbors"]
    out_dir = neighbors_dir(output_root, statement_type)
    manifest_path = output_root / STAGE_NEIGHBORS / f"{statement_type}.manifest.json"
    embed_manifest = output_root / "01_embeddings" / f"{statement_type}.manifest.json"
    embed_id = artifact_id_from_manifest(embed_manifest)
    artifact_id = neighbors_artifact_id(embed_id, cfg, statement_type)

    if not force and manifest_current(manifest_path, artifact_id, [out_dir]):
        print(f"neighbors:{statement_type}: up to date")
        return

    if not force and legacy_neighbors_usable(output_root, cfg, statement_type):
        old = load_json(manifest_path)
        old.update(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "source_artifact_id": embed_id,
                "semantics": neighbors_semantics(cfg),
                "legacy_adopted": True,
            }
        )
        atomic_write_json(manifest_path, old)
        print(f"neighbors:{statement_type}: adopted compatible legacy outputs")
        return

    # A semantic change (k/min_cosine/upstream embeddings) invalidates candidate pairs.
    # Execution-only changes never reach here for a completed artifact.
    current = load_json(manifest_path)
    if force or (current.get("artifact_id") and current.get("artifact_id") != artifact_id):
        reset_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(embeddings_path(output_root, statement_type), mmap_mode="r")
    ids = np.load(embedding_ids_path(output_root, statement_type), mmap_mode="r")
    n = len(ids)
    if n <= 1:
        atomic_write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "artifact_id": artifact_id,
                "source_artifact_id": embed_id,
                "semantics": neighbors_semantics(cfg),
                "pairs": 0,
                "shards": 0,
            },
        )
        return

    k = min(int(ncfg.get("k", 128)) + 1, n)
    min_cos = float(ncfg.get("min_cosine", 0.60))
    query_bs = max(1, int(ncfg.get("query_batch_size", 4096)))
    logical_span = max(query_bs, int(ncfg.get("output_query_span", 32768)))
    compress = bool(ncfg.get("compress_shards", False))
    backend = str(ncfg.get("backend", "auto")).lower()
    analysis_cfg = cfg.get("pair_analysis", {})
    capture_ranks = bool(analysis_cfg.get("capture_neighbor_ranks_during_neighbors", True))
    rank_dir = rank_meta_dir(output_root, statement_type)
    if capture_ranks:
        rank_dir.mkdir(parents=True, exist_ok=True)
    directed_pairs_total = 0

    index = None
    used_backend = backend
    torch_device = None
    if backend in {"auto", "faiss"}:
        try:
            import faiss  # noqa: F401

            index = _faiss_index(x, ncfg)
            used_backend = "faiss"
        except Exception as exc:
            if backend == "faiss":
                raise
            warnings.warn(
                f"FAISS unavailable/failed ({exc}); falling back to PyTorch blockwise search.",
                RuntimeWarning,
                stacklevel=2,
            )
            used_backend = "torch"
    elif backend == "torch":
        used_backend = "torch"
    else:
        raise ValueError("neighbors.backend must be auto/faiss/torch")

    if used_backend == "torch":
        decision = choose_device(
            str(cfg.get("runtime", {}).get("device", "auto")),
            fallback_to_cpu=bool(cfg.get("runtime", {}).get("fallback_to_cpu", True)),
        )
        torch_device = decision.device
        if torch_device == "cuda":
            print(f"neighbors:{statement_type}: FAISS fallback uses torch CUDA")
        else:
            print(f"neighbors:{statement_type}: FAISS fallback uses torch CPU")

    total_pairs = 0
    shard_count = 0
    ranges = list(range(0, n, logical_span))
    for q0 in tqdm(ranges, desc=f"neighbors:{statement_type}"):
        q1 = min(q0 + logical_span, n)
        shard_path = out_dir / f"part-q{q0:012d}-{q1:012d}.npz"
        rank_path = rank_dir / shard_path.name if capture_ranks else None
        candidate_valid = (not force and _valid_modern_shard(shard_path, artifact_id, q0, q1))
        rank_valid = (
            not capture_ranks
            or (rank_path is not None and not force and _valid_rank_meta_shard(rank_path, artifact_id, q0, q1))
        )
        if candidate_valid and rank_valid:
            with np.load(shard_path, allow_pickle=False) as d:
                total_pairs += len(d["a"])
            if capture_ranks and rank_path is not None:
                with np.load(rank_path, allow_pickle=False) as d:
                    directed_pairs_total += len(d["query"])
            shard_count += 1
            continue

        parts_a: list[np.ndarray] = []
        parts_b: list[np.ndarray] = []
        parts_c: list[np.ndarray] = []
        parts_r: list[np.ndarray] = []
        directed_q: list[np.ndarray] = []
        directed_n: list[np.ndarray] = []
        directed_r: list[np.ndarray] = []
        directed_c: list[np.ndarray] = []
        for start in range(q0, q1, query_bs):
            stop = min(start + query_bs, q1)
            if used_backend == "faiss":
                sims, nbrs = index.search(np.asarray(x[start:stop], dtype=np.float32), k)
            else:
                sims, nbrs = _torch_search_resilient(
                    x,
                    start,
                    stop,
                    k,
                    device=str(torch_device),
                    corpus_block=max(1024, int(ncfg.get("torch_corpus_block", 32768))),
                    compute_dtype=str(ncfg.get("torch_compute_dtype", "auto")),
                )
            qrows = np.arange(start, stop, dtype=np.int64)[:, None]
            nonself = (nbrs >= 0) & (nbrs != qrows)
            rank_matrix = np.cumsum(nonself, axis=1, dtype=np.int32)
            directed_valid = nonself & (sims >= min_cos) & (rank_matrix <= int(ncfg.get("k", 128)))

            if capture_ranks:
                dr, dc = np.nonzero(directed_valid)
                if len(dr):
                    directed_q.append(np.asarray(ids[qrows[dr, 0]], dtype=np.int64))
                    directed_n.append(np.asarray(ids[nbrs[dr, dc]], dtype=np.int64))
                    directed_r.append(np.asarray(rank_matrix[dr, dc], dtype=np.uint16))
                    directed_c.append(np.asarray(sims[dr, dc], dtype=np.float32))

            valid = directed_valid & (nbrs > qrows)
            rows, cols = np.nonzero(valid)
            if len(rows) == 0:
                continue
            parts_a.append(np.asarray(ids[qrows[rows, 0]], dtype=np.int64))
            parts_b.append(np.asarray(ids[nbrs[rows, cols]], dtype=np.int64))
            parts_c.append(np.asarray(sims[rows, cols], dtype=np.float32))
            parts_r.append(np.asarray(rank_matrix[rows, cols], dtype=np.uint16))

        a = np.concatenate(parts_a) if parts_a else np.empty(0, dtype=np.int64)
        b = np.concatenate(parts_b) if parts_b else np.empty(0, dtype=np.int64)
        cosine = np.concatenate(parts_c) if parts_c else np.empty(0, dtype=np.float32)
        rank_a_to_b = np.concatenate(parts_r) if parts_r else np.empty(0, dtype=np.uint16)
        tmp = shard_path.with_suffix(".tmp.npz")
        _save_npz(
            tmp,
            compress=compress,
            a=a,
            b=b,
            cosine=cosine,
            rank_a_to_b=rank_a_to_b,
            artifact_id=np.asarray(artifact_id),
            q_start=np.asarray(q0, dtype=np.int64),
            q_stop=np.asarray(q1, dtype=np.int64),
        )
        tmp.replace(shard_path)

        if capture_ranks and rank_path is not None:
            dq = np.concatenate(directed_q) if directed_q else np.empty(0, dtype=np.int64)
            dn = np.concatenate(directed_n) if directed_n else np.empty(0, dtype=np.int64)
            dr = np.concatenate(directed_r) if directed_r else np.empty(0, dtype=np.uint16)
            dc = np.concatenate(directed_c) if directed_c else np.empty(0, dtype=np.float32)
            rank_tmp = rank_path.with_suffix(".tmp.npz")
            _save_npz(
                rank_tmp,
                compress=compress,
                query=dq,
                neighbor=dn,
                rank=dr,
                cosine=dc,
                artifact_id=np.asarray(artifact_id),
                q_start=np.asarray(q0, dtype=np.int64),
                q_stop=np.asarray(q1, dtype=np.int64),
            )
            rank_tmp.replace(rank_path)
            directed_pairs_total += len(dq)

        total_pairs += len(a)
        shard_count += 1

    if capture_ranks:
        atomic_write_json(
            rank_dir / "manifest.json",
            {
                "complete": True,
                "source_artifact_id": artifact_id,
                "k": int(ncfg.get("k", 128)),
                "min_cosine": min_cos,
                "shards": shard_count,
                "directed_pairs": int(directed_pairs_total),
                "note": "Directed neighbor metadata for pair-level auditing; does not affect clustering semantics.",
            },
        )

    atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "artifact_id": artifact_id,
            "source_artifact_id": embed_id,
            "semantics": neighbors_semantics(cfg),
            "backend": used_backend,
            "torch_device": torch_device,
            "index": ncfg.get("index", "hnsw") if used_backend == "faiss" else "exact-blockwise",
            "statements": n,
            "pairs": total_pairs,
            "shards": shard_count,
            "execution": {
                "query_batch_size": query_bs,
                "output_query_span": logical_span,
                "add_batch_size": int(ncfg.get("add_batch_size", 65536)),
                "compress_shards": compress,
            },
        },
    )
