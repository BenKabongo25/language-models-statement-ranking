#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m process.process_datasets --all --overwrite
