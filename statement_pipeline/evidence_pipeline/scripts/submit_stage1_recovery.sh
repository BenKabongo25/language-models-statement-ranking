#!/bin/bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DATASET [STAGE1_RUN]" >&2
  exit 2
fi
DATASET="$1"
STAGE1_RUN="${2:-qwen3.8-27b__atomic-v1}"
cd "$(dirname "$0")/../.."
python -m evidence_pipeline.slurm.submit_recovery \
  --stage 1 \
  --dataset "${DATASET}" \
  --run-name "${STAGE1_RUN}" \
  --model-config evidence_pipeline/configs/models/qwen38_27b.yaml \
  --profile evidence_pipeline/configs/slurm/h200.yaml \
  --gpus 1
