#!/bin/bash
set -euo pipefail
ENV_NAME="${1:-statementrec}"
CUDA_BACKEND="${2:-cu129}"
PROJECT_ROOT="${3:-$(cd "$(dirname "$0")/../.." && pwd)}"

if ! command -v conda >/dev/null 2>&1; then echo "conda is required" >&2; exit 1; fi
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists; remove it first for a clean rebuild." >&2
  exit 2
fi
conda create -n "${ENV_NAME}" python=3.11 -y
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export UV_LINK_MODE=copy
python -m pip install --upgrade pip uv
PYTHON_BIN="$(which python)"
uv pip install --python "${PYTHON_BIN}" --prerelease=allow --upgrade vllm \
  --torch-backend="${CUDA_BACKEND}" \
  --extra-index-url="https://wheels.vllm.ai/nightly/${CUDA_BACKEND}"
uv pip install --python "${PYTHON_BIN}" --upgrade "regex>=2025.10.22" PyYAML tqdm pytest
uv pip install --python "${PYTHON_BIN}" -r "${PROJECT_ROOT}/clustering_pipeline/requirements.txt"
uv pip check --python "${PYTHON_BIN}"
python - <<'PY'
import torch
from importlib.metadata import version
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
print("vLLM:", version("vllm"))
print("transformers:", version("transformers"))
print("structured outputs: OK")
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY
