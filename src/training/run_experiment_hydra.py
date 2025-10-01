"""Hydra-driven experiment launcher.

Usage:
python -m src.training.run_experiment_hydra  \
  training.epochs=20 data.batch_size=64 model.freeze_backbone=true wandb.enable=true

Overrides examples:
python -m src.training.run_experiment_hydra model=resnet18 training.lr=5e-4 data.backend=alb data.augment=false
"""
from __future__ import annotations
import os, random
from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as tvm

from src.data.dataset import build_dataloaders
from src.models.resnet_model import build_resnet18
from src.training.trainer import Trainer, TrainerConfig
from src.evaluation.evaluate import ModelEvaluator
from src.training.checkpointing import FocalLoss


# Determine absolute configs dir (project root / configs)
_CONFIG_DIR = Path(__file__).resolve().parents[2] / 'configs'
if not _CONFIG_DIR.exists():
    raise SystemExit(f"[hydra setup] Expected configs directory at {_CONFIG_DIR}. Create it or adjust path.")


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_architecture(cfg):
    name = cfg.model.name.lower()
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    freeze = getattr(cfg.model, 'freeze_backbone', False)
    dropout = getattr(cfg.model, 'dropout', 0.0)
    if name == 'resnet18':
        model = build_resnet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze, dropout=dropout)
    elif name == 'densenet121':
        weights = tvm.DenseNet121_Weights.DEFAULT if pretrained else None
        model = tvm.densenet121(weights=weights)
        if freeze:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, num_classes)
        )
    elif name == 'efficientnet_b0':
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        if freeze:
            for n,p in model.named_parameters():
                if not n.startswith('classifier.'):
                    p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f"Unsupported model: {name}")
    return model

def build_criterion(cfg):
    import torch
    # Class weights
    weights = None
    cw = getattr(cfg.model, 'class_weights', None)
    if cw not in (None, 'null'):
        try:
            if isinstance(cw, (list, tuple)):
                import torch
                weights = torch.tensor([float(x) for x in cw], dtype=torch.float)
            elif isinstance(cw, str):
                parts = [float(x) for x in cw.split(',')]
                weights = torch.tensor(parts, dtype=torch.float)
        except Exception:
            print('[warn] Could not parse class_weights; using none.')
    if getattr(cfg.model, 'use_focal', False):
        return FocalLoss(alpha=0.25, gamma=2.0)
    else:
        return torch.nn.CrossEntropyLoss(weight=weights)


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name='main')
def main(cfg: DictConfig):
    print('--- CONFIG ROOT ---:', _CONFIG_DIR)
    print('--- CONFIG ---')
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loaders = build_dataloaders(
        cfg.data.metadata_csv,
        batch_size=cfg.data.batch_size,
        backend=cfg.data.backend,
        img_size=cfg.data.img_size,
        augment=cfg.data.augment,
        normalize=cfg.data.normalize,
        balanced=cfg.data.balanced,
        drop_corrupted=cfg.data.drop_corrupted,
    )

    model = build_architecture(cfg).to(device)
    criterion = build_criterion(cfg)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.training.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs) if cfg.training.scheduler == 'cosine' else None

    tcfg = TrainerConfig(
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        patience=cfg.training.patience,
        amp=cfg.training.amp,
        output_dir=cfg.output_dir,
        monitor=cfg.training.monitor,
        monitor_mode=cfg.training.monitor_mode,
        save_last=cfg.training.save_last,
        use_wandb=cfg.wandb.enable,
        wandb_project=cfg.wandb.project,
        wandb_run_name=cfg.wandb.run_name,
        wandb_tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        wandb_notes=cfg.wandb.notes,
    )
    trainer = Trainer(model, optimizer, scheduler, criterion=criterion, config=tcfg, device=device)
    trainer.fit(loaders['train'], loaders['val'])
    trainer.save_history()

    eval_loader = loaders['test'] or loaders['val']
    evaluator = ModelEvaluator(model, eval_loader, device)
    evaluator.evaluate(Path(cfg.results_dir) / Path(cfg.output_dir).name)

    print('Done.')

if __name__ == '__main__':
    main()
