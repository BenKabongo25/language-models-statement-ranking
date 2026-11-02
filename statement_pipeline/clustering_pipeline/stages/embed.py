from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from clustering_pipeline.compat import legacy_embed_usable
from clustering_pipeline.runtime import (
    choose_device,
    is_cuda_failure,
    is_oom,
    model_attention_backend,
    torch_dtype,
)
from clustering_pipeline.signatures import SCHEMA_VERSION, embedding_artifact_id, embedding_semantics
from clustering_pipeline.storage import STAGE_EMBED, embedding_ids_path, embeddings_path, statements_path
from clustering_pipeline.utils import (
    artifact_id_from_manifest,
    atomic_write_json,
    load_json,
    manifest_current,
    read_jsonl,
)


def _last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), seq_lengths
    ]


class QwenEmbeddingModel:
    def __init__(self, cfg: dict[str, Any], runtime_cfg: dict[str, Any]):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.AutoModel = AutoModel
        self.cfg = cfg
        self.fallback = bool(runtime_cfg.get("fallback_to_cpu", True))
        decision = choose_device(str(runtime_cfg.get("device", "auto")), fallback_to_cpu=self.fallback)
        self.device = decision.device
        self.device_reason = decision.reason

        tok_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
            "padding_side": "left",
        }
        if cfg.get("revision"):
            tok_kwargs["revision"] = cfg["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], **tok_kwargs)
        self.model = None
        self._load(self.device)

    def _model_kwargs(self, device: str, attn: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.cfg.get("trust_remote_code", False)),
            "dtype": torch_dtype(str(self.cfg.get("dtype", "auto")), device),
            "attn_implementation": attn,
        }
        if self.cfg.get("revision"):
            kwargs["revision"] = self.cfg["revision"]
        return kwargs

    def _load(self, device: str) -> None:
        requested_attn = model_attention_backend(str(self.cfg.get("attn_implementation", "auto")), device)
        attempts = [requested_attn] + (["eager"] if requested_attn != "eager" else [])
        last_exc: Exception | None = None
        for attn in attempts:
            try:
                model = self.AutoModel.from_pretrained(self.cfg["model"], **self._model_kwargs(device, attn)).eval()
                if device == "cuda":
                    model = model.to("cuda")
                    self.torch.backends.cuda.matmul.allow_tf32 = True
                self.model = model
                self.device = device
                self.attn_backend = attn
                return
            except Exception as exc:
                last_exc = exc
                try:
                    del model
                except Exception:
                    pass
                gc.collect()
        if device == "cuda" and self.fallback:
            warnings.warn(
                f"Embedding model could not use CUDA ({last_exc}); reloading on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.device_reason = f"CUDA model load failed: {last_exc}"
            self._load("cpu")
            return
        assert last_exc is not None
        raise last_exc

    def switch_to_cpu(self) -> None:
        if self.device == "cpu":
            return
        try:
            del self.model
        finally:
            self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
        self._load("cpu")

    @property
    def output_dim(self) -> int:
        dim = int(getattr(self.model.config, "hidden_size"))
        trunc = int(self.cfg.get("truncate_dim", -1))
        return min(dim, trunc) if trunc > 0 else dim

    @property
    def inference_mode(self):
        return self.torch.inference_mode

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch.nn.functional as F

        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.get("max_length", 256)),
            return_tensors="pt",
        )
        if self.device == "cuda":
            tok = {k: v.to("cuda", non_blocking=True) for k, v in tok.items()}
        try:
            with self.torch.inference_mode():
                out = self.model(**tok)
        except Exception as exc:
            if self.device == "cuda" and self.fallback and is_cuda_failure(exc):
                self.switch_to_cpu()
                tok = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=int(self.cfg.get("max_length", 256)),
                    return_tensors="pt",
                )
                with self.torch.inference_mode():
                    out = self.model(**tok)
            else:
                raise
        emb = _last_token_pool(out.last_hidden_state, tok["attention_mask"]).float()
        trunc = int(self.cfg.get("truncate_dim", -1))
        if trunc > 0:
            emb = emb[:, :trunc]
        if bool(self.cfg.get("normalize", True)):
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()


def _load_records(path: Path) -> tuple[np.ndarray, list[str]]:
    ids: list[int] = []
    texts: list[str] = []
    for _, rec in read_jsonl(path):
        ids.append(int(rec["raw_statement_id"]))
        texts.append(str(rec["text"]))
    return np.asarray(ids, dtype=np.int64), texts


def run_embed(*, output_root: Path, cfg: dict[str, Any], statement_type: str, force: bool = False) -> None:
    stage = output_root / STAGE_EMBED
    stage.mkdir(parents=True, exist_ok=True)
    out_path = embeddings_path(output_root, statement_type)
    ids_path = embedding_ids_path(output_root, statement_type)
    manifest_path = stage / f"{statement_type}.manifest.json"
    progress_path = stage / f"{statement_type}.progress.json"

    prepare_id = artifact_id_from_manifest(output_root / "00_prepare" / "manifest.json")
    artifact_id = embedding_artifact_id(prepare_id, cfg, statement_type)
    outputs = [out_path, ids_path]

    if not force and manifest_current(manifest_path, artifact_id, outputs):
        print(f"embed:{statement_type}: up to date")
        return

    if not force and legacy_embed_usable(output_root, cfg, statement_type):
        old = load_json(manifest_path)
        old.update(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "semantics": embedding_semantics(cfg),
                "source_artifact_id": prepare_id,
                "legacy_adopted": True,
            }
        )
        atomic_write_json(manifest_path, old)
        if progress_path.exists():
            progress_path.unlink()
        print(f"embed:{statement_type}: adopted compatible legacy outputs")
        return

    ids, texts = _load_records(statements_path(output_root, statement_type))
    np.save(ids_path, ids)
    if not texts:
        np.save(out_path, np.empty((0, 0), dtype=np.float32))
        atomic_write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "artifact_id": artifact_id,
                "source_artifact_id": prepare_id,
                "semantics": embedding_semantics(cfg),
                "count": 0,
            },
        )
        return

    ecfg = cfg["embedding"]
    model = QwenEmbeddingModel(ecfg, cfg.get("runtime", {}))
    dim = model.output_dim
    out_dtype_name = str(ecfg.get("output_dtype", "float16"))
    out_dtype = np.float16 if out_dtype_name == "float16" else np.float32

    progress = load_json(progress_path)
    resume = (
        not force
        and out_path.exists()
        and progress.get("artifact_id") == artifact_id
        and int(progress.get("count", -1)) == len(texts)
        and int(progress.get("dimension", -1)) == dim
    )
    if resume:
        mmap = open_memmap(out_path, mode="r+")
        start = max(0, min(int(progress.get("next_index", 0)), len(texts)))
        if mmap.shape != (len(texts), dim):
            resume = False
    if not resume:
        mmap = open_memmap(out_path, mode="w+", dtype=out_dtype, shape=(len(texts), dim))
        start = 0
        atomic_write_json(
            progress_path,
            {
                "artifact_id": artifact_id,
                "count": len(texts),
                "dimension": dim,
                "next_index": 0,
            },
        )

    requested_bs = max(1, int(ecfg.get("batch_size", 128)))
    current_bs = requested_bs
    checkpoint_batches = max(1, int(ecfg.get("checkpoint_every_batches", 50)))
    batches_since_checkpoint = 0

    pbar = tqdm(total=len(texts), initial=start, desc=f"embed:{statement_type}")
    while start < len(texts):
        bs = min(current_bs, len(texts) - start)
        try:
            arr = model.encode(texts[start : start + bs])
        except Exception as exc:
            if bs > 1 and is_oom(exc):
                current_bs = max(1, bs // 2)
                warnings.warn(
                    f"Embedding OOM; reducing batch size to {current_bs}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise
        mmap[start : start + bs] = arr.astype(out_dtype, copy=False)
        start += bs
        pbar.update(bs)
        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_batches:
            mmap.flush()
            atomic_write_json(
                progress_path,
                {
                    "artifact_id": artifact_id,
                    "count": len(texts),
                    "dimension": dim,
                    "next_index": start,
                },
            )
            batches_since_checkpoint = 0
    pbar.close()
    mmap.flush()
    del mmap

    atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "artifact_id": artifact_id,
            "source_artifact_id": prepare_id,
            "semantics": embedding_semantics(cfg),
            "count": len(texts),
            "dimension": dim,
            "model": ecfg["model"],
            "device": model.device,
            "device_reason": model.device_reason,
            "attn_backend": model.attn_backend,
            "batch_size_requested": requested_bs,
            "batch_size_final": current_bs,
            "output_dtype": out_dtype_name,
        },
    )
    if progress_path.exists():
        progress_path.unlink()
