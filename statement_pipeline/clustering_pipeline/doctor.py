from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

from clustering_pipeline.runtime import choose_device
from clustering_pipeline.utils import atomic_write_json, ensure_dir


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_doctor(*, input_file: Path, output_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    versions = {
        name: _version(name)
        for name in ["torch", "transformers", "numpy", "scipy", "PyYAML", "tqdm", "faiss-cpu", "faiss-gpu"]
    }
    missing = [name for name in ["torch", "transformers", "numpy", "scipy", "PyYAML", "tqdm"] if versions[name] is None]
    if missing:
        raise RuntimeError("Missing required packages: " + ", ".join(missing))

    try:
        from packaging.version import Version

        if Version(versions["transformers"] or "0") < Version("4.51.0"):
            raise RuntimeError(f"transformers>=4.51.0 is required for Qwen3; found {versions['transformers']}")
    except ImportError:
        pass

    decision = choose_device(
        str(cfg.get("runtime", {}).get("device", "auto")),
        fallback_to_cpu=bool(cfg.get("runtime", {}).get("fallback_to_cpu", True)),
    )
    try:
        importlib.import_module("faiss")
        faiss_ok = True
    except Exception:
        faiss_ok = False

    report = {
        "input": str(input_file),
        "output_root": str(output_root),
        "versions": versions,
        "selected_device": decision.device,
        "device_reason": decision.reason,
        "faiss_available": faiss_ok,
        "neighbor_fallback": f"torch {decision.device} blockwise exact search" if not faiss_ok else None,
    }
    ensure_dir(output_root)
    atomic_write_json(output_root / "doctor.json", report)
    fallback = "" if faiss_ok else f"; fallback=torch/{decision.device}"
    print(f"doctor: device={decision.device}; faiss={'yes' if faiss_ok else 'no'}{fallback}")
    return report
