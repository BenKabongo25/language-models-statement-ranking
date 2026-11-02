#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
python - <<'PY'
import sys, torch
from importlib.metadata import version
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
print("Python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("vLLM:", version("vllm"))
print("transformers:", version("transformers"))
print("regex:", version("regex"))
print("structured outputs: OK")
if not torch.cuda.is_available(): raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory info:", torch.cuda.mem_get_info())
PY
