#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m evidence_pipeline.slurm.submit_pipeline \
  --dataset Beauty \
  --limit "${1:-50}" \
  --stage1-model evidence_pipeline/configs/models/qwen38_27b.yaml \
  --stage1-profile evidence_pipeline/configs/slurm/h200.yaml \
  --stage2-model evidence_pipeline/configs/models/qwen35_9b.yaml \
  --stage2-profile evidence_pipeline/configs/slurm/a6000.yaml \
  --stage3-model evidence_pipeline/configs/models/qwen35_9b.yaml \
  --stage3-profile evidence_pipeline/configs/slurm/a6000.yaml
