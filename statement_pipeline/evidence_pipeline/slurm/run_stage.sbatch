#!/bin/bash
set -euo pipefail

if [[ -f "${HOME}/.bashrc" ]]; then source "${HOME}/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "conda was not found in PATH" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
if [[ "${DISABLE_FLASHINFER_SAMPLER:-1}" == "1" ]]; then export VLLM_USE_FLASHINFER_SAMPLER=0; fi

cd "${PROJECT_ROOT}"
JOB_LOG_DIR="${LOG_ROOT}/jobs/${SLURM_JOB_ID}/stage${STAGE}-${DATASET}-primary"
mkdir -p "${JOB_LOG_DIR}"

HF_TOKEN_PRESENT=0
if [[ -n "${!HF_TOKEN_ENV:-}" ]]; then HF_TOKEN_PRESENT=1; fi
if [[ "${REQUIRES_HF_TOKEN:-0}" == "1" && "${HF_TOKEN_PRESENT}" != "1" ]]; then
  echo "Missing required ${HF_TOKEN_ENV}." >&2
  exit 2
fi

echo "Stage: ${STAGE} (primary)"
echo "Dataset: ${DATASET}"
echo "Run: ${RUN_NAME}"
echo "Stage1 parent: ${STAGE1_RUN:-none}"
echo "Stage1 recovery parent: ${STAGE1_RECOVERY_RUN:-none}"
echo "Model: ${MODEL_NAME}"
echo "Model family: ${MODEL_FAMILY}"
echo "Thinking: disabled for primary passes"
echo "Node: $(hostname)"
echo "GPUs: ${TOTAL_GPUS} | workers: ${NUM_WORKERS} | TP/worker: ${TENSOR_PARALLEL_SIZE}"
echo "Limit: ${LIMIT:-none} | indices: ${INDICES_FILE:-none}"
echo "Worker logs: ${JOB_LOG_DIR}"
echo "HF token available: ${HF_TOKEN_PRESENT}"

python - <<'PY'
import torch
from importlib.metadata import version
print("python/torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("vLLM:", version("vllm"))
print("transformers:", version("transformers"))
PY

if [[ "${PREFETCH_MODEL:-1}" == "1" ]]; then
  python -m evidence_pipeline.prefetch_model --model-config "${MODEL_CONFIG}"
fi

SRUN_ARGS=(
  --exact
  --ntasks="${NUM_WORKERS}"
  --cpus-per-task="${CPUS_PER_WORKER}"
  --gpus-per-task="${GPUS_PER_WORKER}"
  --output="${JOB_LOG_DIR}/worker-%t.out"
  --error="${JOB_LOG_DIR}/worker-%t.err"
)
if [[ -n "${GPU_BIND:-}" ]]; then SRUN_ARGS+=(--gpu-bind="${GPU_BIND}"); fi

srun "${SRUN_ARGS[@]}" bash "${PROJECT_ROOT}/evidence_pipeline/slurm/run_worker.sh"

echo "Workers completed; merging primary Stage ${STAGE}."
MERGE_ARGS=(
  --stage "${STAGE}"
  --dataset "${DATASET}"
  --run-name "${RUN_NAME}"
  --pipeline-config "${PIPELINE_CONFIG}"
)
if [[ -n "${STAGE1_RUN:-}" ]]; then MERGE_ARGS+=(--stage1-run "${STAGE1_RUN}"); fi
if [[ -n "${STAGE1_RECOVERY_RUN:-}" ]]; then MERGE_ARGS+=(--stage1-recovery-run "${STAGE1_RECOVERY_RUN}"); fi
python -m evidence_pipeline.merge "${MERGE_ARGS[@]}"
