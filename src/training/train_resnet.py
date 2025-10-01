"""End-to-end training script (ResNet18) with metrics, early stopping, checkpointing, amp, scheduler."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.resnet_model import build_resnet18
from training.engine import train_one_epoch, evaluate
from training.checkpointing import save_checkpoint, EarlyStopping

# Simple dataset/dataloader inline to avoid extra dependency; can be replaced by separate module
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from collections import Counter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, split, transform=None, drop_corrupted=True):
        df = pd.read_csv(csv_file)
        df = df[df['split']==split]
        if drop_corrupted and 'corrupted' in df.columns:
            df = df[~df['corrupted']]
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {'NORMAL':0, 'PNEUMONIA':1}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row['path'] if 'path' in row else row.get('image_path'))
        with Image.open(img_path) as im: im = im.convert('RGB')
        if self.transform: im = self.transform(im)
        return im, self.label_map[row['class'] if 'class' in row else row['label']]

train_tf = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(5),
    T.ColorJitter(0.05,0.05,0.05,0.02),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def make_loader(csv_file, split, batch_size, balanced=False, num_workers=4):
    ds = ChestXrayDataset(csv_file, split, transform=train_tf if split=='train' else val_tf)
    if balanced and split=='train':
        counts = Counter(ds.df['class'] if 'class' in ds.df.columns else ds.df['label'])
        total = sum(counts.values())
        class_weights = {c: total/(len(counts)*counts[c]) for c in counts}
        labels = (ds.df['class'] if 'class' in ds.df.columns else ds.df['label']).map(class_weights).astype(float).values
        sampler = WeightedRandomSampler(labels, num_samples=len(labels), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True, help='Path to metadata CSV (clean or raw).')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--balanced', action='store_true')
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--output-dir', type=str, default='models/checkpoints')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--pretrained', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet18(pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    model.to(device)

    train_loader = make_loader(args.csv, 'train', args.batch_size, balanced=args.balanced)
    val_loader = make_loader(args.csv, 'val', args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device=='cuda' else None

    early = EarlyStopping(patience=args.patience, mode='max')
    best_val_acc = 0.0

    history = []
    for epoch in range(1, args.epochs+1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_stats = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_stats = {
            'epoch': epoch,
            'train': train_stats,
            'val': {k: (float(v.tolist()) if hasattr(v,'tolist') else v) for k,v in val_stats.items()}
        }
        history.append(epoch_stats)
        print(f"Epoch {epoch}: train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.3f} | val loss {val_stats['loss']:.4f} acc {val_stats['acc']:.3f} auc {val_stats.get('auc',float('nan')):.3f}")

        # Checkpoint
        if val_stats['acc'] > best_val_acc:
            best_val_acc = val_stats['acc']
            save_checkpoint(model, optimizer, epoch, {'val_acc': best_val_acc}, output_dir / 'best.pt')
            print('  * New best model saved.')

        # Early stopping
        if early.step(val_stats['acc']):
            print('Early stopping triggered.')
            break

    # Save training history
    with (output_dir / 'history.json').open('w') as f:
        json.dump(history, f, indent=2)
    print('Training complete. Best val acc:', best_val_acc)

if __name__ == '__main__':
    main()
