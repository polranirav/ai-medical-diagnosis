"""High-level Trainer abstraction wrapping engine functions.

Features:
- Handles model, optimizer, scheduler, scaler (AMP) lifecycle
- Tracks history (per-epoch metrics) and best checkpointing
- Early stopping integration
- Gradient accumulation & clipping (optional)
- Move tensors to device, supports dict/list batches
- Optional Weights & Biases logging
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch import nn

from .engine import train_one_epoch, evaluate
from .checkpointing import save_checkpoint, EarlyStopping

try:  # optional wandb
    import wandb  # type: ignore
    _HAS_WANDB = True
except ImportError:  # pragma: no cover
    _HAS_WANDB = False

@dataclass
class TrainerConfig:
    epochs: int = 15
    lr: float = 1e-3
    grad_clip: float | None = None
    accumulation_steps: int = 1
    amp: bool = True
    patience: int = 5
    monitor: str = 'val_acc'
    monitor_mode: str = 'max'
    output_dir: str | Path = 'models/checkpoints'
    save_last: bool = True
    use_wandb: bool = False
    wandb_project: str = 'ai-medical-diagnosis'
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None

@dataclass
class TrainerState:
    epoch: int = 0
    best_value: float | None = None
    history: List[Dict[str, Any]] = field(default_factory=list)

class Trainer:
    def __init__(self, model: nn.Module, optimizer, scheduler=None, criterion=None, config: TrainerConfig | None = None, device: str | torch.device = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.config = config or TrainerConfig()
        self.device = torch.device(device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp and self.device.type == 'cuda')
        self.state = TrainerState()
        self.early = EarlyStopping(patience=self.config.patience, mode=self.config.monitor_mode)
        self.output_dir = Path(self.config.output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_wandb()

    def _init_wandb(self):  # optional
        self.wandb_run = None
        if self.config.use_wandb and _HAS_WANDB:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                tags=self.config.wandb_tags,
                notes=self.config.wandb_notes,
                config={
                    'epochs': self.config.epochs,
                    'lr': self.config.lr,
                    'patience': self.config.patience,
                    'monitor': self.config.monitor,
                    'monitor_mode': self.config.monitor_mode,
                    'amp': self.config.amp,
                }
            )
            try:
                wandb.watch(self.model, log='all', log_freq=100)
            except Exception:
                pass

    def _wandb_log(self, data: dict):
        if getattr(self, 'wandb_run', None) is not None:
            wandb.log(data, step=self.state.epoch)

    def fit(self, train_loader, val_loader):
        monitor_key = self.config.monitor.replace('val_','')  # e.g. val_acc -> acc
        for epoch in range(1, self.config.epochs + 1):
            self.state.epoch = epoch
            train_stats = train_one_epoch(self.model, train_loader, self.criterion, self.optimizer, self.device, self.scaler)
            val_stats = evaluate(self.model, val_loader, self.criterion, self.device)
            if self.scheduler:
                self.scheduler.step()

            def robust_float(val):
                # Convert tensors/arrays to python types first
                if hasattr(val, 'tolist'):
                    try:
                        val = val.tolist()
                    except Exception:
                        pass
                # If nested list/tuple, recurse
                if isinstance(val, (list, tuple)):
                    # Detect if any element is itself list/tuple (nested structure like confusion matrix)
                    if any(isinstance(x, (list, tuple)) for x in val):
                        return [robust_float(x) for x in val]
                    # Flat sequence of scalars
                    try:
                        converted = [float(x) for x in val]
                        if len(converted) == 1:
                            return converted[0]
                        return converted
                    except (TypeError, ValueError):
                        return val  # leave as-is if not convertible
                # Try scalar conversion
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return val

            record = {
                'epoch': epoch,
                'train': train_stats,
                'val': {k: robust_float(v) for k,v in val_stats.items()},
            }
            self.state.history.append(record)
            val_metric = val_stats.get(monitor_key if monitor_key in val_stats else 'acc')
            print(f"Epoch {epoch}: train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.3f} | val loss {val_stats['loss']:.4f} acc {val_stats['acc']:.3f}")
            # wandb logging
            self._wandb_log({
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'train_acc': train_stats['acc'],
                'val_loss': val_stats['loss'],
                'val_acc': val_stats['acc'],
                **({ 'val_auc': val_stats.get('auc') } if 'auc' in val_stats else {})
            })

            improved = False
            if self.state.best_value is None or ((val_metric > self.state.best_value) if self.config.monitor_mode=='max' else (val_metric < self.state.best_value)):
                self.state.best_value = val_metric
                improved = True
                save_checkpoint(self.model, self.optimizer, epoch, {'best_'+monitor_key: val_metric}, self.output_dir / 'best.pt')
                print('  * New best model saved.')
                self._wandb_log({'best_'+monitor_key: val_metric})

            if self.early.step(val_metric):
                print('Early stopping triggered.')
                break
        if self.config.save_last:
            save_checkpoint(self.model, self.optimizer, self.state.epoch, {'final': True}, self.output_dir / 'last.pt')
        if getattr(self, 'wandb_run', None) is not None:
            wandb.finish()
        return self.state.history

    def save_history(self):
        import json
        with (self.output_dir / 'history.json').open('w') as f:
            json.dump(self.state.history, f, indent=2)

__all__ = ['Trainer', 'TrainerConfig']
