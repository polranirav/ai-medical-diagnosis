"""Training engine with mixed precision & scheduler support."""
from __future__ import annotations
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from .metrics import classification_metrics


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device, scaler: GradScaler | None = None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {'loss': total_loss / max(total,1), 'acc': correct / max(total,1)}

@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_targets = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())
    if all_logits:
        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)
        extra = classification_metrics(logits_cat, targets_cat)
    else:
        extra = {}
    result = {'loss': total_loss / max(total,1), 'acc': correct / max(total,1)}
    result.update(extra)
    return result

__all__ = ["train_one_epoch","evaluate"]
