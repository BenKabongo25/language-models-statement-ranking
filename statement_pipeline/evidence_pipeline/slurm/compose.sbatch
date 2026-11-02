#!/bin/bash
set -euo pipefail
if [[ -f "${HOME}/.bashrc" ]]; then source "${HOME}/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
cd "${PROJECT_ROOT}"
ARGS=(
  --dataset "${DATASET}"
  --pipeline-config "${PIPELINE_CONFIG}"
  --stage1-run "${STAGE1_RUN}"
  --stage2-run "${STAGE2_RUN}"
  --stage3-run "${STAGE3_RUN}"
)
if [[ -n "${STAGE1_RECOVERY_RUN:-}" ]]; then ARGS+=(--stage1-recovery-run "${STAGE1_RECOVERY_RUN}"); fi
if [[ -n "${STAGE2_RECOVERY_RUN:-}" ]]; then ARGS+=(--stage2-recovery-run "${STAGE2_RECOVERY_RUN}"); fi
if [[ -n "${STAGE3_RECOVERY_RUN:-}" ]]; then ARGS+=(--stage3-recovery-run "${STAGE3_RECOVERY_RUN}"); fi
if [[ "${STRICT_COMPOSE:-0}" == "1" ]]; then ARGS+=(--strict); fi
python -m evidence_pipeline.compose "${ARGS[@]}"
