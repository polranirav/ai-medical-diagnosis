"""Model evaluation utilities (test set performance, confusion matrix, ROC, reports).

CLI Example:
python -m src.evaluation.evaluate \
  --checkpoint models/exp_resnet18_tv/best.pt \
  --csv notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv \
  --output-dir results/evaluation/exp_resnet18_tv

If test split not present, will evaluate on validation split.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Sequence
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from src.models.resnet_model import build_resnet18
from src.data.dataset import build_dataloaders

class ModelEvaluator:
    def __init__(self, model, loader, device, class_names: Sequence[str] = ('NORMAL','PNEUMONIA')):
        self.model = model
        self.loader = loader
        self.device = device
        self.class_names = list(class_names)

    def evaluate(self, save_dir: str | Path):
        save_path = Path(save_dir); save_path.mkdir(parents=True, exist_ok=True)
        y_true, y_pred, y_probs = self._collect()
        metrics = self._compute_metrics(y_true, y_pred, y_probs)
        self._confusion_matrix(y_true, y_pred, save_path)
        self._roc_curve(y_true, y_probs, save_path)
        self._classification_report(y_true, y_pred, save_path)
        self._save_metrics(metrics, save_path)
        return metrics

    def _collect(self):
        self.model.eval()
        y_true, y_pred, y_prob_pos = [], [], []
        with torch.no_grad():
            for batch in self.loader:
                if len(batch) == 3:  # (x,y,path)
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(1)
                y_true.append(labels.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
                # binary probability of class 1 (assumes index 1 is PNEUMONIA)
                if probs.shape[1] > 1:
                    y_prob_pos.append(probs[:,1].cpu().numpy())
                else:  # degenerate case
                    y_prob_pos.append(torch.zeros_like(probs[:,0]).cpu().numpy())
        return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob_pos))

    def _compute_metrics(self, y_true, y_pred, y_probs):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_true, y_probs) if len(np.unique(y_true)) > 1 else (np.array([0,1]), np.array([0,1]), None)
        roc_auc = auc(fpr, tpr) if len(np.unique(y_true)) > 1 else float('nan')
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
        }

    def _confusion_matrix(self, y_true, y_pred, save_path: Path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path / 'confusion_matrix.png', dpi=300)
        plt.close()
        pd.DataFrame(cm, index=self.class_names, columns=self.class_names).to_csv(save_path / 'confusion_matrix.csv')

    def _roc_curve(self, y_true, y_probs, save_path: Path):
        if len(np.unique(y_true)) < 2:
            return
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}', color='darkorange')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.tight_layout()
        plt.savefig(save_path / 'roc_curve.png', dpi=300)
        plt.close()
        pd.DataFrame({'fpr':fpr,'tpr':tpr}).to_csv(save_path / 'roc_curve_points.csv', index=False)

    def _classification_report(self, y_true, y_pred, save_path: Path):
        rep_dict = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        pd.DataFrame(rep_dict).transpose().to_csv(save_path / 'classification_report.csv')
        with (save_path / 'classification_report.txt').open('w') as f:
            f.write(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))

    def _save_metrics(self, metrics: dict, save_path: Path):
        with (save_path / 'metrics_summary.json').open('w') as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame([metrics]).to_csv(save_path / 'metrics_summary.csv', index=False)
        print('=== EVALUATION METRICS ===')
        for k,v in metrics.items():
            print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='results/evaluation/run1')
    p.add_argument('--backend', type=str, default='torchvision')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--no-augment', action='store_true')
    return p.parse_args()


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    return ckpt


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders = build_dataloaders(
        args.csv,
        batch_size=args.batch_size,
        backend=args.backend,
        img_size=args.img_size,
        augment=not args.no_augment,
        normalize=not args.no_normalize,
    )
    eval_loader = loaders['test'] or loaders['val']
    model = build_resnet18(pretrained=False, freeze_backbone=False)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    evaluator = ModelEvaluator(model, eval_loader, device)
    evaluator.evaluate(args.output_dir)

if __name__ == '__main__':
    main()
