#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET [clustering_pipeline.run arguments...]" >&2
  echo "Examples:" >&2
  echo "  $0 Beauty" >&2
  echo "  $0 Beauty --from-stage cluster" >&2
  echo "  $0 Yelp --from-stage rerank --to-stage audit" >&2
  exit 2
fi

DATASET="$1"
shift

DATASETS_ROOT="${DATASETS_ROOT:-${PROJECT_ROOT}/../data/processed}"
DATASET_ROOT="${DATASETS_ROOT}/${DATASET}"
INPUT="${DATASET_ROOT}/evidence_pipeline/final/reviews_with_statements.jsonl"
OUTPUT_ROOT="${DATASET_ROOT}/clustering_pipeline"

if [[ ! -f "$INPUT" ]]; then
  echo "Input not found: $INPUT" >&2
  exit 2
fi

exec python -m clustering_pipeline.run \
  --input "$INPUT" \
  --output-root "$OUTPUT_ROOT" \
  --config clustering_pipeline/configs/pipeline.yaml \
  "$@"
