from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from clustering_pipeline.compat import legacy_rerank_usable
from clustering_pipeline.runtime import (
    choose_device,
    is_cuda_failure,
    is_oom,
    model_attention_backend,
    torch_dtype,
)
from clustering_pipeline.signatures import SCHEMA_VERSION, rerank_artifact_id, rerank_semantics
from clustering_pipeline.storage import STAGE_RERANK, neighbors_dir, rerank_dir, statements_path
from clustering_pipeline.utils import (
    artifact_id_from_manifest,
    atomic_write_json,
    load_json,
    manifest_current,
    read_jsonl,
    reset_dir,
)


def _save_npz(path: Path, *, compress: bool, **arrays) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


class Qwen3Reranker:
    def __init__(self, cfg: dict[str, Any], runtime_cfg: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.AutoModelForCausalLM = AutoModelForCausalLM
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
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if not isinstance(self.false_id, int) or not isinstance(self.true_id, int):
            raise RuntimeError("Qwen3-Reranker tokenizer did not expose scalar token IDs for 'no'/'yes'.")

        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.model = None
        self._load(self.device)

    def _kwargs(self, device: str, attn: str) -> dict[str, Any]:
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
                model = self.AutoModelForCausalLM.from_pretrained(self.cfg["model"], **self._kwargs(device, attn)).eval()
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
                f"Reranker could not use CUDA ({last_exc}); reloading on CPU.",
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
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass
        self._load("cpu")

    def _prepare(self, pairs: list[tuple[str, str]], instruction: str):
        body = [f"<Instruct>: {instruction}\n<Query>: {a}\n<Document>: {b}" for a, b in pairs]
        max_len = int(self.cfg.get("max_length", 512))
        body_max = max_len - len(self.prefix_tokens) - len(self.suffix_tokens)
        if body_max <= 16:
            raise ValueError(f"reranker.max_length={max_len} is too small for Qwen3 prompt scaffolding")
        enc = self.tokenizer(
            body,
            padding=False,
            truncation="longest_first",
            max_length=body_max,
            return_attention_mask=False,
        )
        enc["input_ids"] = [self.prefix_tokens + ids + self.suffix_tokens for ids in enc["input_ids"]]
        batch = self.tokenizer.pad(enc, padding=True, return_tensors="pt")
        if self.device == "cuda":
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        return batch

    def score(self, pairs: list[tuple[str, str]], instruction: str) -> np.ndarray:
        batch = self._prepare(pairs, instruction)
        try:
            with self.torch.inference_mode():
                logits = self.model(**batch).logits[:, -1, :]
        except Exception as exc:
            if self.device == "cuda" and self.fallback and is_cuda_failure(exc):
                self.switch_to_cpu()
                batch = self._prepare(pairs, instruction)
                with self.torch.inference_mode():
                    logits = self.model(**batch).logits[:, -1, :]
            else:
                raise
        yn = self.torch.stack((logits[:, self.false_id], logits[:, self.true_id]), dim=1).float()
        return self.torch.softmax(yn, dim=1)[:, 1].cpu().numpy()


def _same_pairs(source_path: Path, output_path: Path) -> bool:
    try:
        src = np.load(source_path, allow_pickle=False)
        dst = np.load(output_path, allow_pickle=False)
        if "score" not in dst.files or len(src["a"]) != len(dst["a"]):
            return False
        return np.array_equal(src["a"], dst["a"]) and np.array_equal(src["b"], dst["b"])
    except Exception:
        return False


def _modern_shard_matches(source_path: Path, output_path: Path, artifact_id: str) -> bool:
    if not output_path.exists() or not _same_pairs(source_path, output_path):
        return False
    try:
        d = np.load(output_path, allow_pickle=False)
        return "artifact_id" in d.files and str(d["artifact_id"].item()) == artifact_id
    except Exception:
        return False


def _legacy_shard_reusable(source_path: Path, output_path: Path) -> bool:
    if not output_path.exists() or not _same_pairs(source_path, output_path):
        return False
    try:
        d = np.load(output_path, allow_pickle=False)
        # Previous pipeline wrote `accepted`; this is a useful format marker.
        return "artifact_id" not in d.files and "score" in d.files
    except Exception:
        return False


def run_rerank(*, output_root: Path, cfg: dict[str, Any], statement_type: str, force: bool = False) -> None:
    rcfg = cfg["reranker"]
    src_dir = neighbors_dir(output_root, statement_type)
    out_dir = rerank_dir(output_root, statement_type)
    manifest_path = output_root / STAGE_RERANK / f"{statement_type}.manifest.json"
    neighbors_id = artifact_id_from_manifest(output_root / "02_neighbors" / f"{statement_type}.manifest.json")
    artifact_id = rerank_artifact_id(neighbors_id, cfg, statement_type)

    if not force and manifest_current(manifest_path, artifact_id, [out_dir]):
        print(f"rerank:{statement_type}: up to date")
        return

    if not force and legacy_rerank_usable(output_root, cfg, statement_type):
        old = load_json(manifest_path)
        old.update(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "source_artifact_id": neighbors_id,
                "semantics": rerank_semantics(cfg, statement_type),
                "legacy_adopted": True,
                "note": "Stored raw scores are reused; thresholding is now performed in cluster stage.",
            }
        )
        atomic_write_json(manifest_path, old)
        print(f"rerank:{statement_type}: adopted compatible legacy outputs")
        return

    old_manifest = load_json(manifest_path)
    if force or (old_manifest.get("artifact_id") and old_manifest.get("artifact_id") != artifact_id):
        reset_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(src_dir.glob("part-*.npz"))
    if not shards:
        atomic_write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "artifact_id": artifact_id,
                "source_artifact_id": neighbors_id,
                "semantics": rerank_semantics(cfg, statement_type),
                "pairs": 0,
                "shards": 0,
            },
        )
        return

    enabled = bool(rcfg.get("enabled", True))
    instruction = str((rcfg.get("instructions") or {})[statement_type])
    requested_bs = max(1, int(rcfg.get("batch_size", 32)))
    current_bs = requested_bs
    compress = bool(rcfg.get("compress_shards", False))
    allow_legacy_partial = bool(rcfg.get("reuse_legacy_partial_shards", True))

    # First decide which shards really need scoring. This avoids loading the
    # reranker at all when a previous/partial run already contains every score.
    work: list[tuple[Path, Path]] = []
    total_pairs = 0
    reused_shards = 0
    legacy_shards = 0
    for src_path in shards:
        dst_path = out_dir / src_path.name
        if not force and _modern_shard_matches(src_path, dst_path, artifact_id):
            total_pairs += len(np.load(dst_path, allow_pickle=False)["a"])
            reused_shards += 1
            continue
        if (
            not force
            and allow_legacy_partial
            and not old_manifest.get("artifact_id")
            and _legacy_shard_reusable(src_path, dst_path)
        ):
            total_pairs += len(np.load(dst_path, allow_pickle=False)["a"])
            reused_shards += 1
            legacy_shards += 1
            continue
        work.append((src_path, dst_path))

    model = None
    texts: dict[int, str] = {}
    if work and enabled:
        texts = {
            int(r["raw_statement_id"]): str(r["text"])
            for _, r in read_jsonl(statements_path(output_root, statement_type))
        }
        model = Qwen3Reranker(rcfg, cfg.get("runtime", {}))

    for src_path, dst_path in tqdm(work, desc=f"rerank:{statement_type}:shards"):
        data = np.load(src_path, allow_pickle=False)
        a = np.asarray(data["a"], dtype=np.int64)
        b = np.asarray(data["b"], dtype=np.int64)
        cosine = np.asarray(data["cosine"], dtype=np.float32)
        scores = np.empty(len(a), dtype=np.float32)
        pos = 0
        while pos < len(a):
            bs = min(current_bs, len(a) - pos)
            if model is None:
                scores[pos : pos + bs] = cosine[pos : pos + bs]
                pos += bs
                continue
            pairs = [(texts[int(x)], texts[int(y)]) for x, y in zip(a[pos : pos + bs], b[pos : pos + bs])]
            try:
                scores[pos : pos + bs] = model.score(pairs, instruction)
            except Exception as exc:
                if bs > 1 and is_oom(exc):
                    current_bs = max(1, bs // 2)
                    warnings.warn(
                        f"Reranker OOM; reducing batch size to {current_bs}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                raise
            pos += bs

        tmp = dst_path.with_suffix(".tmp.npz")
        _save_npz(
            tmp,
            compress=compress,
            a=a,
            b=b,
            cosine=cosine,
            score=scores,
            artifact_id=np.asarray(artifact_id),
        )
        tmp.replace(dst_path)
        total_pairs += len(a)

    atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "artifact_id": artifact_id,
            "source_artifact_id": neighbors_id,
            "semantics": rerank_semantics(cfg, statement_type),
            "model": rcfg.get("model") if enabled else None,
            "device": model.device if model else None,
            "device_reason": model.device_reason if model else None,
            "attn_backend": model.attn_backend if model else None,
            "pairs": total_pairs,
            "shards": len(shards),
            "reused_shards": reused_shards,
            "legacy_partial_shards_reused": legacy_shards,
            "execution": {
                "batch_size_requested": requested_bs,
                "batch_size_final": current_bs,
                "compress_shards": compress,
            },
            "note": "Threshold is intentionally not part of reranking; raw scores are thresholded in cluster stage.",
        },
    )
