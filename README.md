# AI Medical Diagnosis System

Automated disease detection from medical images using deep learning.

> NOTE: UI Screenshots (Normal vs Pneumonia) — add images to `docs/screenshots/ui_normal.png` and `docs/screenshots/ui_pneumonia.png`.
>
> | Normal Case | Pneumonia Case |
> |-------------|----------------|
> | ![Normal UI](docs/screenshots/ui_normal.png) | ![Pneumonia UI](docs/screenshots/ui_pneumonia.png) |

## Project Structure
- **data/**: Raw and processed datasets
- **src/**: Source code and core modules
- **notebooks/**: Exploratory and preprocessing analysis
- **scripts/**: Shell scripts for pipelines
- **tests/**: Unit testing modules
- **docs/**: Documentation & diagrams

## Quick Start
1. Clone repo  
2. Install dependencies (`pip install -r requirements.txt`)
3. Download dataset (`python src/data/collect_data.py`)  
4. Preprocess (`python src/data/preprocess.py`)  
5. Explore (`jupyter notebook notebooks/exploratory_data_analysis.ipynb`)  
6. Run tests (`pytest tests/`)

## Data Sources
- [Kaggle Chest X-Ray](https://www.kaggle.com/datasets)
- Local hospital dataset (anonymized)

## Core Features
- Modular data pipeline (metadata validation, augmentation, balanced sampling)
- Hydra-configurable experiments (model, data, training blocks)
- Multi-architecture support: ResNet18, DenseNet121, EfficientNet-B0
- Mixed precision (automatic disable on CPU/MPS without CUDA)
- Early stopping & checkpointing (best + last)
- Evaluation: confusion matrix, ROC, per-class metrics, misclassification grids
- Explainability: Grad-CAM batch generation + montage
- Experiment tracking (optional Weights & Biases)

## Recent Experiment (Hydra)
Checkpoint: `models/exp_hydra/best.pt`
Evaluation (validation/test aggregate):

| Metric | Value |
|--------|-------|
| Accuracy | 0.832 |
| Precision (weighted) | 0.789 |
| Recall (weighted) | 0.997 |
| F1 (weighted) | 0.881 |
| ROC AUC | 0.962 |
| Average Precision | (see extended eval) |
| Brier Score | (see extended eval) |

Detailed classification report:
```
NORMAL precision=0.99 recall=0.56 f1=0.71 support=234
PNEUMONIA precision=0.79 recall=1.00 f1=0.88 support=390
Overall accuracy=0.83
```
Confusion Matrix (rows=true, cols=pred):
```
          Pred NORMAL  Pred PNEUMONIA
TRUE NORMAL      130          104
TRUE PNEUMONIA      1          389
```
Key Insight: Model is highly sensitive to Pneumonia (recall ~1.00) but sacrifices specificity (NORMAL recall 0.56). Potential class imbalance or decision threshold bias.

## Grad-CAM Examples
Generated overlays highlight activation focus on opacities in pneumonia cases.
Run:
```
bash scripts/generate_gradcam_examples.sh models/exp_hydra/best.pt \
  notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv results/gradcam_exp_hydra 6 resnet18
```
Montage saved to: `results/gradcam_exp_hydra/_grid_overlay.png`

## Iterating Experiments
Hydra override examples:
```
python -m src.training.run_experiment_hydra model=densenet121 training.epochs=25 \
  training.lr=5e-4 data.batch_size=48 data.augment=false
python -m src.training.run_experiment_hydra model=efficientnet_b0 model.dropout=0.4 \
  training.scheduler=cosine training.patience=7
python -m src.training.run_experiment_hydra model=resnet18 data.balanced=false \
  training.monitor=val_auc training.monitor_mode=max
```

## Extended Evaluation
Run threshold tuning + calibration + PR & reliability plots:
```
python -m src.evaluation.evaluate_extended \
  --checkpoint models/exp_hydra/best.pt \
  --csv notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv \
  --output-dir results/evaluation_ext --calibrate --export-onnx
```
Artifacts:
- threshold_sweep.csv
- precision_recall_curve.png
- reliability_diagram.png (+ calibrated)
- calibration.json (temperature, Brier improvement)
- model.onnx (if exported)

## API Inference
Start FastAPI server:
```
uvicorn src.api.app:app --reload
```
Probability endpoint:
```
POST /predict_proba (multipart-form file)
```
Set checkpoint path:
```
MODEL_CKPT=models/exp_hydra/best.pt uvicorn src.api.app:app --reload
```

## Focal Loss / Class Weights
Enable via Hydra override:
```
python -m src.training.run_experiment_hydra model=resnet18 model.use_focal=true
python -m src.training.run_experiment_hydra model=resnet18 model.class_weights="1.0,1.5"
```

## ONNX Export (Standalone)
Already included in extended evaluation (use --export-onnx) or manual:
```
python -m src.evaluation.evaluate_extended --checkpoint models/exp_hydra/best.pt --csv notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv --export-onnx
```

## Hyperparameter Sweep (Manual Loop Example)
```
for m in resnet18 densenet121 efficientnet_b0; do \
  for lr in 0.0005 0.001 0.002; do \
    for dr in 0.2 0.3 0.4; do \
      python -m src.training.run_experiment_hydra model=$m training.lr=$lr model.dropout=$dr wandb.enable=true wandb.run_name=${m}_lr${lr}_dr${dr}; \
    done; \
  done; \
done
```

## Recommended Next Steps
1. Improve specificity: try threshold tuning (optimize Youden's J) & class-weighted loss.
2. Calibration: add temperature scaling / reliability diagrams.
3. Ensembling: average logits from different seeds or architectures.
4. Deploy: export ONNX / TorchScript + minimal FastAPI inference service.
5. Add model card summarizing intended use, limitations, and ethical considerations.

## Hydra Experiments (Usage)
```
python -m src.training.run_experiment_hydra \
  training.epochs=20 data.batch_size=64 model.freeze_backbone=true wandb.enable=true
```
Configs live in `configs/`:
- `main.yaml` composes model/data/training
- Override any value via command line (`key=value`) or swap sub-configs (`model=resnet18`).

Outputs saved under configured `output_dir`. Use `wandb.enable=true` for tracking.