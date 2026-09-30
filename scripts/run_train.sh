#!/usr/bin/env bash
set -euo pipefail

REVISION_ROOT="revisions/2026-07-22_adaptive_reconstruction_study"
python -m src.train \
  --seed 42 \
  --dataset cifar10 \
  --model_config '{"split_point":"current","method":"afd","alpha":0.25,"batch_size":32}' \
  --epochs 1 \
  --selection-stage \
  --max-samples 128 \
  --save_path "${REVISION_ROOT}/tmp/compat_checkpoint.pt" \
  --log_path "${REVISION_ROOT}/tmp/compat_checkpoint.json" \
  "$@"
