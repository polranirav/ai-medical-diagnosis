# Pneumonia Detection Model Card

## Model Details
- **Architectures**: ResNet18 (baseline), DenseNet121, EfficientNet-B0 variants
- **Domain**: Chest X-Ray (frontal)
- **Task**: Binary classification (NORMAL vs PNEUMONIA)
- **Framework**: PyTorch + Hydra configuration
- **Loss Functions**: CrossEntropy (default), optional Focal Loss
- **Explainability**: Grad-CAM over final convolutional layer

## Intended Use
Assist radiology workflow triage by prioritizing likely pneumonia cases. Not a diagnostic replacement.

## Out of Scope
- Multi-class lung pathologies
- Lateral X-rays
- Non-human data

## Training Data
Derived from Kaggle Chest X-Ray Pneumonia dataset. Potential class imbalance (more pneumonia cases).

## Evaluation Metrics (Recent Run)
| Metric | Value |
| ------ | ----- |
| Accuracy | 0.832 |
| Weighted F1 | 0.881 |
| Pneumonia Recall | ≈1.00 |
| Normal Recall | 0.56 |
| ROC AUC | 0.962 |

## Calibration
Temperature scaling optional; reliability diagrams generated via extended evaluation script.

## Ethical Considerations
- Risk of over-calling pneumonia → possible unnecessary follow-up.
- Under-representation of certain age groups or comorbidities may bias predictions.

## Limitations
- High false positive rate for NORMAL class.
- Dataset quality (possible label noise).
- Not validated on external institutions.

## Recommendations
- Add external validation set.
- Perform threshold tuning for operational deployment.
- Monitor drift post-deployment.

## Versioning & Reproducibility
Reproduce with: `python -m src.training.run_experiment_hydra model=resnet18 seed=42`.

## Contact
Maintainer: (add name / email)
