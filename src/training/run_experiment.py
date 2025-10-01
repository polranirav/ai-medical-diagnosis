"""Unified experiment script using dataset + trainer abstractions.

Example:
python -m src.training.run_experiment \
  --csv notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv \
  --backend torchvision --epochs 20 --balanced --pretrained
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import build_dataloaders
from src.models.resnet_model import build_resnet18
from .trainer import Trainer, TrainerConfig
from src.evaluation.evaluate import ModelEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--backend', type=str, default='torchvision', choices=['torchvision','alb','albumentations'])
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--balanced', action='store_true')
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--output-dir', type=str, default='models/exp1')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--eval-output', type=str, default='results/evaluation')
    # new wandb flags
    p.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    p.add_argument('--wandb-project', type=str, default='ai-medical-diagnosis')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-tags', type=str, default=None, help='Comma-separated tags')
    p.add_argument('--wandb-notes', type=str, default=None)
    return p.parse_args()


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
        balanced=args.balanced,
    )

    model = build_resnet18(pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    cfg = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        amp=args.amp,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=[t.strip() for t in args.wandb_tags.split(',')] if args.wandb_tags else None,
        wandb_notes=args.wandb_notes,
    )
    trainer = Trainer(model, optimizer, scheduler, config=cfg, device=device)
    history = trainer.fit(loaders['train'], loaders['val'])
    trainer.save_history()
    print('Training complete. Best value:', trainer.state.best_value)

    # ---- Post-training evaluation ----
    eval_loader = loaders['test'] or loaders['val']
    exp_name = Path(args.output_dir).name
    eval_dir = Path(args.eval_output) / exp_name
    print('\n=== Evaluating on', ('test' if loaders['test'] else 'validation'), 'set ===')
    evaluator = ModelEvaluator(model, eval_loader, device)
    evaluator.evaluate(eval_dir)
    print('Evaluation artifacts saved to:', eval_dir)

if __name__ == '__main__':
    main()
