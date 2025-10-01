# Training Pipeline Overview

Components added:

1. Dataset & Dataloaders (`src/data/dataset.py`)
   - Handles metadata loading, split filtering, path & class column resolution.
   - Provides `build_dataloaders` for train/val(/test) loaders with optional class balancing.
   - Integrates transform factory from `preprocess.py` supporting torchvision or albumentations.

2. Trainer Abstraction (`src/training/trainer.py`)
   - Wraps training / evaluation with mixed precision, early stopping, checkpointing.
   - Saves best & (optionally) last checkpoints plus JSON history.

3. Unified Experiment Script (`src/training/run_experiment.py`)
   - CLI flags for backend, augmentation, normalization, pretrained backbone, balancing, AMP.
   - Example shell script: `scripts/train_example.sh`.

4. Inference Utility (`src/training/inference.py`)
   - Loads a checkpoint and predicts class probabilities for a single image.

5. Legacy Script (`src/training/train_resnet.py`)
   - Kept for reference; superseded by `run_experiment.py` + Trainer.

## Quick Start

```
bash scripts/train_example.sh
```
Or manually:
```
python -m src.training.run_experiment \
  --csv notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv \
  --backend torchvision \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-3 \
  --pretrained \
  --balanced \
  --output-dir models/exp_resnet18_tv
```

## Outputs
- Best checkpoint: `<output_dir>/best.pt`
- Last checkpoint: `<output_dir>/last.pt`
- Training history: `<output_dir>/history.json`

## Extending
- Swap model: create new builder under `src/models/` and import in `run_experiment.py`.
- Add metrics: extend `src/training/metrics.py` and merge into `evaluate`.
- Experiment tracking (optional): integrate Weights & Biases, MLflow, or simple CSV logger in Trainer.

## Visualization
Use the transform visualization notebook to inspect augmentations before training.

