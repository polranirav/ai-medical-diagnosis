"""Metrics utilities: AUC, F1, confusion matrix, per-class precision/recall."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        preds = logits.argmax(1).detach().cpu().numpy()
        y_true = targets.detach().cpu().numpy()
        auc = roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float('nan')
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        prec, rec, f1c, support = precision_recall_fscore_support(y_true, preds, zero_division=0)
        return {
            'auc': auc,
            'f1': f1,
            'confusion_matrix': cm,
            'precision_per_class': prec,
            'recall_per_class': rec,
            'f1_per_class': f1c,
            'support_per_class': support
        }

__all__ = ["classification_metrics"]
