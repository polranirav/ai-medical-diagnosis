#!/usr/bin/env bash
set -euo pipefail
CSV=${1:-notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv}
python -m src.training.run_experiment \
  --csv "$CSV" \
  --backend torchvision \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-3 \
  --pretrained \
  --balanced \
  --output-dir models/exp_resnet18_tv
