"""Checkpointing and early stopping utilities."""
from __future__ import annotations
import torch
from pathlib import Path
from dataclasses import dataclass


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def load_checkpoint(model, optimizer, path: str | Path):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt['epoch'], ckpt.get('metrics', {})

@dataclass
class EarlyStopping:
    patience: int = 5
    mode: str = 'max'  # 'min' or 'max'
    delta: float = 1e-4
    best_value: float | None = None
    counter: int = 0
    stopped: bool = False

    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        improvement = (value - self.best_value) if self.mode == 'max' else (self.best_value - value)
        if improvement > self.delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1-pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

__all__ = ["save_checkpoint","load_checkpoint","EarlyStopping","FocalLoss"]
