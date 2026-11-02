#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=baselines/logs/%x_%j.out
#SBATCH --error=baselines/logs/%x_%j.err

set -euo pipefail

: "${DATASET:?DATASET is required}"
: "${MODEL:?MODEL is required}"
: "${SEED:?SEED is required}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV="${CONDA_ENV:-statementrec}"
BASE_CONFIG="${BASE_CONFIG:-baselines/configs/base.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-baselines/configs/models/${MODEL}.yaml}"
EXTRA_CONFIGS="${EXTRA_CONFIGS:-${EXTRA_CONFIG:-}}"

if [[ -f "${HOME}/.bashrc" ]]; then source "${HOME}/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "conda was not found in PATH" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
mkdir -p baselines/logs

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

printf 'dataset=%s model=%s seed=%s host=%s\n' "$DATASET" "$MODEL" "$SEED" "$(hostname)"
python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))
PY

if [[ "${PREFETCH_MODEL:-1}" == "1" ]]; then
  case "$MODEL" in
    bper_plus|se_bper|se_bper_plus)
      python -m baselines.prefetch --config "$BASE_CONFIG" --config "$MODEL_CONFIG" || exit 1
      ;;
  esac
fi

ARGS=(--dataset "$DATASET" --model "$MODEL" --seed "$SEED" --base-config "$BASE_CONFIG" --model-config "$MODEL_CONFIG")
if [[ -n "$EXTRA_CONFIGS" ]]; then
  IFS='|' read -r -a EXTRA_ARRAY <<< "$EXTRA_CONFIGS"
  for cfg in "${EXTRA_ARRAY[@]}"; do
    [[ -n "$cfg" ]] && ARGS+=(--extra-config "$cfg")
  done
fi
python -m baselines.run "${ARGS[@]}"
