# Portfolio Summary: AI Chest X-Ray Pneumonia Detection

## Elevator Pitch
End-to-end medical imaging ML pipeline: curated metadata, robust augmentation, multi-architecture training (ResNet / DenseNet / EfficientNet), calibrated evaluation, explainability (Grad-CAM), FastAPI inference, and Streamlit + Gradio frontends—production-style structure suitable for clinical prototyping (research only).

## Highlights
- 15+ modular Python modules (data, training, evaluation, explainability, API)
- Hydra-driven configuration for reproducibility
- Extended evaluation: threshold sweep, calibration (temperature scaling), PR & reliability diagrams
- Automated Grad-CAM visualization batch + montage
- Focal loss / class-weight options for imbalance mitigation
- ONNX export for deployment portability
- Fully interactive UI (Streamlit) + lightweight demo (Gradio)

## Key Performance
| Metric | Value |
| ------ | ----- |
| Accuracy | 0.832 |
| Weighted F1 | 0.881 |
| Pneumonia Recall | ~1.00 |
| Normal Recall | 0.56 |
| ROC AUC | 0.962 |

Optimized for sensitivity (triage use-case). Next step: improve specificity via threshold tuning + rebalancing.

## Tech Stack
Python, PyTorch, Hydra, FastAPI, Streamlit, Gradio, scikit-learn, Matplotlib, ONNX.

## Architecture Flow
1. Metadata ingestion & validation
2. Augmentation & balanced sampling
3. Train (early stopping + checkpointing + W&B optional)
4. Evaluate (metrics, curves, misclass grids)
5. Explain (Grad-CAM)
6. Calibrate & export (temperature scaling, ONNX)
7. Serve (FastAPI + Streamlit)

## Ethical & Responsible AI
Model card included (bias, limitations, intended use). High false positives addressed via calibration & threshold experimentation.

## Repository Guide
See `README.md` for execution steps. Use `MODEL_CKPT` env var to swap models quickly.

## Future Enhancements
- Multi-label pathology extension
- Active learning loop
- Federated fine-tuning across institutions
- Continuous drift monitoring

---
Prepared for portfolio presentation. Screenshots: add two UI images (Normal case, Pneumonia case) to `docs/screenshots/` and embed in README.
