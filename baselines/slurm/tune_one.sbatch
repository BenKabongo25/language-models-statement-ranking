#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=baselines/logs/%x_%j.out
#SBATCH --error=baselines/logs/%x_%j.err

set -euo pipefail

: "${DATASET:?DATASET is required}"
: "${MODEL:?MODEL is required}"
: "${SEARCH_CONFIG:?SEARCH_CONFIG is required}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV="${CONDA_ENV:-statementrec}"
BASE_CONFIG="${BASE_CONFIG:-baselines/configs/base.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-baselines/configs/models/${MODEL}.yaml}"
TUNE_SEEDS="${TUNE_SEEDS:-40 41 42}"
EXTRA_CONFIG="${EXTRA_CONFIG:-}"

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

read -r -a SEED_ARGS <<< "${TUNE_SEEDS}"
ARGS=(--dataset "$DATASET" --model "$MODEL" --seeds "${SEED_ARGS[@]}" --base-config "$BASE_CONFIG" --model-config "$MODEL_CONFIG" --search-config "$SEARCH_CONFIG")
if [[ -n "$EXTRA_CONFIG" ]]; then ARGS+=(--extra-config "$EXTRA_CONFIG"); fi
python -m baselines.tune "${ARGS[@]}"
