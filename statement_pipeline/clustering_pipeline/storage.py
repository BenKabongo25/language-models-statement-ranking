from __future__ import annotations

from pathlib import Path

TYPES = ("user", "item", "match")
STAGE_PREPARE = "00_prepare"
STAGE_EMBED = "01_embeddings"
STAGE_NEIGHBORS = "02_neighbors"
STAGE_RERANK = "03_rerank"
STAGE_CLUSTER = "04_clusters"
STAGE_ANALYSIS = "05_pair_analysis"
STAGE_FINAL = "final"


def stage_dir(root: Path, stage: str) -> Path:
    return root / stage


def statements_path(root: Path, typ: str) -> Path:
    return root / STAGE_PREPARE / f"statements_{typ}.jsonl"


def interactions_raw_path(root: Path) -> Path:
    return root / STAGE_PREPARE / "interactions_raw.jsonl"


def evidence_raw_path(root: Path) -> Path:
    return root / STAGE_PREPARE / "evidence_raw.jsonl"


def embeddings_path(root: Path, typ: str) -> Path:
    return root / STAGE_EMBED / f"{typ}.npy"


def embedding_ids_path(root: Path, typ: str) -> Path:
    return root / STAGE_EMBED / f"{typ}_ids.npy"


def neighbors_dir(root: Path, typ: str) -> Path:
    return root / STAGE_NEIGHBORS / typ


def rank_meta_dir(root: Path, typ: str) -> Path:
    return root / STAGE_NEIGHBORS / typ / "rank_meta"


def rerank_dir(root: Path, typ: str) -> Path:
    return root / STAGE_RERANK / typ


def clusters_path(root: Path, typ: str) -> Path:
    return root / STAGE_CLUSTER / f"clusters_{typ}.jsonl"


def mapping_path(root: Path, typ: str) -> Path:
    return root / STAGE_CLUSTER / f"raw_to_cluster_{typ}.npz"


def cluster_trace_path(root: Path, typ: str) -> Path:
    return root / STAGE_CLUSTER / f"merge_trace_{typ}.tsv"




def cluster_progress_path(root: Path, typ: str) -> Path:
    return root / STAGE_CLUSTER / f"frontier_progress_{typ}.tsv"


def cluster_state_dir(root: Path, typ: str) -> Path:
    return root / STAGE_CLUSTER / f"state_{typ}"


def cluster_checkpoint_path(root: Path, typ: str) -> Path:
    return cluster_state_dir(root, typ) / "checkpoint.npz"


def cluster_checkpoint_manifest_path(root: Path, typ: str) -> Path:
    return cluster_state_dir(root, typ) / "checkpoint.json"

def verification_cache_dir(root: Path, typ: str) -> Path:
    return root / STAGE_CLUSTER / f"verification_{typ}"


def pair_analysis_path(root: Path, typ: str) -> Path:
    return root / STAGE_ANALYSIS / f"pairs_{typ}.tsv"


def pair_analysis_sample_path(root: Path, typ: str) -> Path:
    return root / STAGE_ANALYSIS / f"pairs_{typ}.sample.tsv"
