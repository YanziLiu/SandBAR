#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 evaluate.py \
  --data-root "${DATA_ROOT:-dataset/Dataset}" \
  --checkpoint "${CKPT:-weights/sandbar_checkpoint.pkl}" \
  --output "${OUTPUT:-results/evaluation}"
