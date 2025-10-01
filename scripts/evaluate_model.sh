#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-models/exp_resnet18_tv/best.pt}
CSV=${2:-notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv}
OUT=${3:-results/evaluation/manual_eval}
python -m src.evaluation.evaluate \
  --checkpoint "$CKPT" \
  --csv "$CSV" \
  --output-dir "$OUT"
