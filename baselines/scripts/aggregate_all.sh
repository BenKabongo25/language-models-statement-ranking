#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
DATASETS=(Beauty Toys Yelp)
MODELS=(rand rucf ricf cd pitf bper bper_plus bper_large se_bper se_bper_plus expgcn userpop itempop globalpop cler)
for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    python -m baselines.aggregate --dataset "$d" --model "$m" || true
  done
done
