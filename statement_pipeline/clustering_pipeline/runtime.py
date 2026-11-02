from __future__ import annotations

import os
import subprocess
import sys
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceDecision:
    device: str
    reason: str


def probe_cuda(timeout: int = 30) -> tuple[bool, str]:
    """Probe real CUDA execution in a child process.

    torch.cuda.is_available() alone is not enough for binary/driver mismatches.
    """
    code = r'''
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
x = torch.ones((32, 32), device="cuda", dtype=torch.float32)
y = x @ x
_ = float(y[0, 0].item())
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0))
'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except Exception as exc:
        return False, f"CUDA probe failed to run: {exc}"
    if proc.returncode == 0:
        return True, (proc.stdout.strip() or "CUDA probe succeeded")
    detail = proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}"
    return False, detail[-2000:]


def choose_device(requested: str = "auto", *, fallback_to_cpu: bool = True) -> DeviceDecision:
    requested = str(requested or "auto").lower().strip()
    if requested == "cpu":
        return DeviceDecision("cpu", "CPU explicitly requested")
    if requested not in {"auto", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    ok, detail = probe_cuda()
    if ok:
        return DeviceDecision("cuda", detail)
    if requested == "cuda" and not fallback_to_cpu:
        raise RuntimeError(f"CUDA was requested but the runtime probe failed:\n{detail}")
    warnings.warn(
        "CUDA is unavailable or binary-incompatible; using CPU. "
        f"Probe detail: {detail}",
        RuntimeWarning,
        stacklevel=2,
    )
    return DeviceDecision("cpu", detail)


def torch_dtype(name: str, device: str):
    import torch

    name = str(name or "auto").lower()
    if device == "cpu":
        return torch.float32
    if name == "auto":
        return torch.float16
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype={name}; expected auto/float16/bfloat16/float32")
    return mapping[name]


def model_attention_backend(requested: str, device: str) -> str:
    """Choose a PyTorch-native attention backend; never require flash-attn."""
    requested = str(requested or "auto").lower()
    if requested == "auto":
        return "sdpa" if device == "cuda" else "eager"
    if requested not in {"sdpa", "eager"}:
        raise ValueError("attn_implementation must be auto/sdpa/eager")
    return requested


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def is_cuda_failure(exc: BaseException) -> bool:
    text = (type(exc).__name__ + " " + str(exc)).lower()
    needles = ("cuda", "cublas", "cudnn", "acceleratorerror", "named symbol", "driver")
    return any(n in text for n in needles)
