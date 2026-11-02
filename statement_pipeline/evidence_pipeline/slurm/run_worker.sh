#!/bin/bash
set -euo pipefail
SHARD_INDEX="${SLURM_PROCID:-0}"
ARGS=(
  --stage "${STAGE}"
  --dataset "${DATASET}"
  --pipeline-config "${PIPELINE_CONFIG}"
  --model-config "${MODEL_CONFIG}"
  --run-name "${RUN_NAME}"
  --num-shards "${NUM_WORKERS}"
  --shard-index "${SHARD_INDEX}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if [[ -n "${STAGE1_RUN:-}" ]]; then ARGS+=(--stage1-run "${STAGE1_RUN}"); fi
if [[ -n "${STAGE1_RECOVERY_RUN:-}" ]]; then ARGS+=(--stage1-recovery-run "${STAGE1_RECOVERY_RUN}"); fi
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "${LIMIT}"); fi
if [[ -n "${INDICES_FILE:-}" ]]; then ARGS+=(--indices-file "${INDICES_FILE}"); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then ARGS+=(--batch-size "${BATCH_SIZE}"); fi
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"); fi
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then ARGS+=(--max-model-len "${MAX_MODEL_LEN}"); fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}"); fi
if [[ "${FORCE:-0}" == "1" ]]; then ARGS+=(--force); fi

echo "worker=${SHARD_INDEX}/${NUM_WORKERS} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec python -m evidence_pipeline.run_stage "${ARGS[@]}"
