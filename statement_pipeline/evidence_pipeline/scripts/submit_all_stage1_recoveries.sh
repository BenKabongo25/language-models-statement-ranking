#!/bin/bash
set -euo pipefail
STAGE1_RUN="${1:-qwen3.8-27b__atomic-v1}"
cd "$(dirname "$0")/../.."
for DATASET in Beauty Toys Yelp; do
  echo "Submitting Stage-1 recovery for ${DATASET} (${STAGE1_RUN})"
  python -m evidence_pipeline.slurm.submit_recovery \
    --stage 1 \
    --dataset "${DATASET}" \
    --run-name "${STAGE1_RUN}" \
    --model-config evidence_pipeline/configs/models/qwen38_27b.yaml \
    --profile evidence_pipeline/configs/slurm/h200.yaml \
    --gpus 1
done
