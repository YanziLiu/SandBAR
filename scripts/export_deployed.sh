#!/usr/bin/env bash
set -euo pipefail

INPUT="${INPUT:-weights/sandbar_train_time_checkpoint.pkl}"
OUTPUT="${OUTPUT:-weights/sandbar_checkpoint.pkl}"

cd "$(dirname "$0")/.."
python3 sandbar_code/tools/export_deployed.py \
  --input "${INPUT}" \
  --output "${OUTPUT}"
