from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from evidence_pipeline.utils import load_yaml

MODEL_FAMILIES = {"qwen3plus", "gemma3", "generic"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    tensor_parallel_size: int
    trust_remote_code: bool
    enable_prefix_caching: bool
    enforce_eager: bool
    download_dir: str | None
    requires_hf_token: bool
    hf_token_env: str

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        """Backward-compatible non-thinking chat kwargs used by primary passes."""
        if self.family == "qwen3plus":
            return {"enable_thinking": False}
        return {}

    @property
    def supports_thinking(self) -> bool:
        # The current pipeline explicitly supports Qwen >= 3.5 thinking.
        # Gemma 3 is kept as a non-thinking family.
        return self.family == "qwen3plus"


@dataclass(frozen=True)
class GenerationMode:
    thinking_requested: bool = False
    thinking_enabled: bool = False
    thinking_token_budget: int | None = None

    @property
    def label(self) -> str:
        return "thinking" if self.thinking_enabled else "nonthinking"


def _qwen_version(model_name: str) -> tuple[int, int] | None:
    matches = list(re.finditer(r"qwen\s*([0-9]+)(?:[._]([0-9]+))?", model_name.lower()))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group(1)), int(match.group(2) or 0)


def detect_family(model_name: str) -> str:
    normalized = model_name.lower().replace("_", "-")
    if "gemma-3" in normalized or "gemma3" in normalized:
        return "gemma3"
    version = _qwen_version(model_name)
    if version is not None and version >= (3, 5):
        return "qwen3plus"
    return "generic"


def load_model_spec(path: str | Path) -> ModelSpec:
    cfg = load_yaml(path)
    name = str(cfg["name"])
    family = str(cfg.get("family", "auto")).lower()
    if family == "auto":
        family = detect_family(name)
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family {family!r}")
    return ModelSpec(
        name=name,
        family=family,
        dtype=str(cfg.get("dtype", "bfloat16")),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.90)),
        max_model_len=int(cfg.get("max_model_len", 4096)),
        max_num_seqs=int(cfg.get("max_num_seqs", 48)),
        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        enable_prefix_caching=bool(cfg.get("enable_prefix_caching", True)),
        enforce_eager=bool(cfg.get("enforce_eager", False)),
        download_dir=cfg.get("download_dir"),
        requires_hf_token=bool(cfg.get("requires_hf_token", False)),
        hf_token_env=str(cfg.get("hf_token_env", "HF_TOKEN")),
    )


def resolve_generation_mode(
    spec: ModelSpec,
    *,
    thinking: bool = False,
    thinking_token_budget: int | None = None,
) -> GenerationMode:
    if thinking_token_budget is not None and thinking_token_budget < 0:
        raise ValueError("thinking_token_budget must be >= 0")
    enabled = bool(thinking and spec.supports_thinking)
    budget = thinking_token_budget if enabled else None
    if thinking and not enabled:
        print(
            f"Thinking was requested for {spec.name}, but family={spec.family} has no supported "
            "thinking adapter in this pipeline; continuing without thinking.",
            flush=True,
        )
    return GenerationMode(
        thinking_requested=bool(thinking),
        thinking_enabled=enabled,
        thinking_token_budget=budget,
    )


def chat_template_kwargs(spec: ModelSpec, mode: GenerationMode) -> dict[str, Any]:
    if spec.family == "qwen3plus":
        return {"enable_thinking": mode.thinking_enabled}
    return {}


def assert_model_access(spec: ModelSpec) -> None:
    if spec.requires_hf_token and not os.environ.get(spec.hf_token_env):
        raise RuntimeError(
            f"Model {spec.name} requires Hugging Face authentication. "
            f"Export {spec.hf_token_env} before running or submitting the job."
        )


def _make_reasoning_config(spec: ModelSpec, mode: GenerationMode):
    if spec.family != "qwen3plus" or not mode.thinking_enabled or mode.thinking_token_budget is None:
        return None
    try:
        from vllm.config import ReasoningConfig
    except ImportError:
        # The budget is optional. Thinking itself can still be enabled through the chat template.
        return None
    return ReasoningConfig(
        reasoning_parser="qwen3",
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
    )


def make_llm(
    spec: ModelSpec,
    *,
    mode: GenerationMode | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
):
    try:
        import vllm
        from vllm import LLM
    except Exception as exc:  # pragma: no cover - exercised only on GPU hosts
        raise RuntimeError(f"Unable to import vLLM: {exc}") from exc

    mode = mode or GenerationMode()
    kwargs: dict[str, Any] = {
        "model": spec.name,
        "dtype": spec.dtype,
        "tensor_parallel_size": int(tensor_parallel_size or spec.tensor_parallel_size),
        "gpu_memory_utilization": float(
            spec.gpu_memory_utilization if gpu_memory_utilization is None else gpu_memory_utilization
        ),
        "max_model_len": int(max_model_len or spec.max_model_len),
        "max_num_seqs": int(max_num_seqs or spec.max_num_seqs),
        "trust_remote_code": spec.trust_remote_code,
        "enable_prefix_caching": spec.enable_prefix_caching,
        "enforce_eager": spec.enforce_eager,
    }
    reasoning_config = _make_reasoning_config(spec, mode)
    if reasoning_config is not None:
        kwargs["reasoning_config"] = reasoning_config
    if spec.download_dir:
        kwargs["download_dir"] = spec.download_dir

    print(
        f"Initializing vLLM {getattr(vllm, '__version__', 'unknown')} | "
        f"model={spec.name} | family={spec.family} | thinking={mode.thinking_enabled} | "
        f"TP={kwargs['tensor_parallel_size']} | max_model_len={kwargs['max_model_len']}",
        flush=True,
    )
    return LLM(**kwargs)


def make_sampling_params(
    generation_cfg: dict[str, Any],
    json_schema: dict[str, Any],
    *,
    mode: GenerationMode | None = None,
):
    from vllm import SamplingParams

    mode = mode or GenerationMode()
    sampling = generation_cfg.get("sampling", {})
    kwargs: dict[str, Any] = {
        "temperature": float(sampling.get("temperature", 0.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", -1)),
        "max_tokens": int(sampling.get("max_tokens", 512)),
        "seed": int(sampling.get("seed", 42)),
        "skip_special_tokens": not mode.thinking_enabled,
    }
    if mode.thinking_enabled and mode.thinking_token_budget is not None:
        # Supported by recent vLLM Qwen reasoning builds. If a future build removes it,
        # the user can omit the budget while keeping enable_thinking=true.
        kwargs["thinking_token_budget"] = int(mode.thinking_token_budget)
    if sampling.get("structured_outputs", True):
        try:
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("This vLLM build does not expose StructuredOutputsParams.") from exc
        kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
    return SamplingParams(**kwargs)


@dataclass(frozen=True)
class GeneratedText:
    answer: str
    raw_text: str


def _strip_trailing_sentinels(text: str) -> str:
    output = text.strip()
    sentinels = ("<|im_end|>", "<|endoftext|>", "<turn|>", "<eos>", "</s>")
    changed = True
    while changed:
        changed = False
        for sentinel in sentinels:
            if output.endswith(sentinel):
                output = output[: -len(sentinel)].rstrip()
                changed = True
    return output


def extract_final_answer(raw_text: str, spec: ModelSpec, mode: GenerationMode) -> str:
    text = raw_text.strip()
    if spec.family == "qwen3plus" and mode.thinking_enabled:
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        elif text.startswith("<think>"):
            # Unterminated reasoning should fail JSON validation and be retried.
            return _strip_trailing_sentinels(text)
    return _strip_trailing_sentinels(text)


def _completion_raw_text(llm, completion, mode: GenerationMode) -> str:
    raw_text = completion.text
    if mode.thinking_enabled and getattr(completion, "token_ids", None):
        try:
            tokenizer = llm.get_tokenizer()
            raw_text = tokenizer.decode(completion.token_ids, skip_special_tokens=False)
        except Exception:
            pass
    return raw_text


def generate_texts(
    llm,
    prompts: list[str],
    sampling_params,
    spec: ModelSpec,
    *,
    mode: GenerationMode | None = None,
) -> list[GeneratedText]:
    mode = mode or GenerationMode()
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_template_kwargs(spec, mode),
        use_tqdm=False,
    )
    generated: list[GeneratedText] = []
    for output in outputs:
        completion = output.outputs[0]
        raw = _completion_raw_text(llm, completion, mode)
        answer = extract_final_answer(raw, spec, mode)
        generated.append(GeneratedText(answer=answer, raw_text=raw))
    return generated
