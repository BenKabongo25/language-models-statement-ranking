#!/bin/bash
set -euo pipefail
MODEL="${1:-google/gemma-3-27b-it}"
TOKEN_ENV="${2:-HF_TOKEN}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
if [[ -z "${!TOKEN_ENV:-}" ]]; then echo "${TOKEN_ENV} is not set." >&2; exit 2; fi
TOKEN_ENV="${TOKEN_ENV}" MODEL="${MODEL}" python - <<'PY'
import os
from huggingface_hub import hf_hub_download, whoami
token = os.environ[os.environ["TOKEN_ENV"]]
model = os.environ["MODEL"]
print("Authenticated account:", whoami(token=token).get("name", "unknown"))
print("Access OK:", hf_hub_download(repo_id=model, filename="config.json", token=token))
PY
