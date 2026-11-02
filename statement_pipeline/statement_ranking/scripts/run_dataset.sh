#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET [statement_ranking.run arguments...]" >&2
  exit 2
fi

DATASET="$1"
shift
DATASETS_ROOT="${DATASETS_ROOT:-${PROJECT_ROOT}/../data/processed}"
DATASET_ROOT="${DATASETS_ROOT}/${DATASET}"

exec python -m statement_ranking.run \
  --dataset "$DATASET" \
  --dataset-root "$DATASET_ROOT" \
  --config statement_ranking/configs/pipeline.yaml \
  "$@"
