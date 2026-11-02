#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(Beauty Toys Yelp)
fi

for dataset in "${DATASETS[@]}"; do
  echo "=== statement_ranking: ${dataset} ==="
  bash statement_ranking/scripts/run_dataset.sh "${dataset}"
done
