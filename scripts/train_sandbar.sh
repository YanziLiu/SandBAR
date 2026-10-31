#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-dataset/Dataset}"
OUT_ROOT="${OUT_ROOT:-outputs}"

cd "$(dirname "$0")/.."
python3 sandbar_code/main.py \
  --mode train \
  --model_name SandBAR_train_time \
  --data_dir "${DATA_DIR}" \
  --output_root "${OUT_ROOT}" \
  --pre_ffn gated \
  --post_ffn freqenh \
  --model_blocks 2,4,4,6 \
  --model_dim 32 \
  --ffn_expansion_factor 2.66 \
  --num_epoch 650 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --use_checkpoint \
  --seed 42

