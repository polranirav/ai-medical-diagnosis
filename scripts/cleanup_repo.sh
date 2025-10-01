#!/usr/bin/env bash
set -euo pipefail
# Clean ancillary artifacts not needed for a lean commit/package
# DOES NOT delete models/ by default (comment in if desired)

KEEP_MODELS=${KEEP_MODELS:-1}

echo "[INFO] Removing Python caches & build artifacts" 
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete

echo "[INFO] Removing notebook checkpoints" 
find notebooks -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} + || true

if [ "${KEEP_MODELS}" = "0" ]; then
  echo "[INFO] Removing model artifacts (models/). Set KEEP_MODELS=1 to retain."
  rm -rf models/* || true
fi

echo "[INFO] Optional: remove W&B logs (set CLEAR_WANDB=1)"
if [ "${CLEAR_WANDB:-0}" = "1" ]; then
  rm -rf wandb/* || true
fi

echo "[INFO] Done."
