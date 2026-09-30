#!/usr/bin/env bash
set -euo pipefail

REVISION_ROOT="revisions/2026-07-22_adaptive_reconstruction_study"
CHECKPOINT="${REVISION_ROOT}/tmp/compat_checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  bash scripts/run_train.sh
fi
python -m src.eval \
  --dataset cifar10 \
  --checkpoint_path "${CHECKPOINT}" \
  --partition validation \
  --max-samples 128 \
  --output-path "${REVISION_ROOT}/tmp/compat_evaluation.json" \
  "$@"
