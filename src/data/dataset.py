"""Chest X-Ray Pneumonia dataset utilities.

Features:
- Metadata loading & column standardization
- Train/val/test subset selection via 'split' column
- Optional corruption filtering
- Transform backend integration (torchvision or albumentations)
- Weighted sampler helper for class imbalance
- Debug mode returning image path alongside tensor
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

from src.data import preprocess as prep

# Public constants
IMAGENET_MEAN = prep.IMAGENET_MEAN
IMAGENET_STD = prep.IMAGENET_STD

class ChestXrayDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        split: str,
        transform=None,
        drop_corrupted: bool = True,
        return_path: bool = False,
    ):
        df = prep.load_clean_metadata(csv_file, filter_corrupted=drop_corrupted)
        df = prep.standardize_metadata_columns(df)
        if 'split' not in df.columns:
            raise ValueError("Metadata must contain 'split' column.")
        df = df[df['split'] == split].copy()
        if len(df) == 0:
            raise ValueError(f"No rows found for split={split} in {csv_file}")
        # Resolve path column
        if 'path' in df.columns:
            path_col = 'path'
        elif 'image_path' in df.columns:
            path_col = 'image_path'
        else:
            raise ValueError("Metadata must contain a 'path' or 'image_path' column.")
        self.path_col = path_col
        # Resolve class column
        if 'class' in df.columns:
            class_col = 'class'
        elif 'label' in df.columns:
            class_col = 'label'
        else:
            raise ValueError("Metadata must contain 'class' or 'label' column.")
        self.class_col = class_col
        # Build label map (stable ordering)
        classes = sorted(df[class_col].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row[self.path_col])
        try:
            with Image.open(img_path) as im:
                im = im.convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to open {img_path}: {e}") from e
        if self.transform:
            im = self.transform(im)
        label = self.class_to_idx[row[self.class_col]]
        if self.return_path:
            return im, label, str(img_path)
        return im, label


def build_transforms(backend: str = 'torchvision', img_size: int = 224, augment: bool = True, normalize: bool = True):
    train_tf, val_tf = prep.get_transforms(kind=backend, img_size=img_size, augment=augment, normalize=normalize)
    return train_tf, val_tf


def make_dataloader(
    csv_file: str | Path,
    split: str,
    batch_size: int,
    transform,
    num_workers: int = 4,
    balanced: bool = False,
    drop_corrupted: bool = True,
    return_path: bool = False,
):
    ds = ChestXrayDataset(csv_file, split, transform=transform, drop_corrupted=drop_corrupted, return_path=return_path)
    if balanced and split == 'train':
        col = ds.class_col
        counts = Counter(ds.df[col])
        total = sum(counts.values())
        class_weights = {c: total / (len(counts) * counts[c]) for c in counts}
        weights = ds.df[col].map(class_weights).astype(float).values
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return ds, DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        return ds, DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers, pin_memory=True)


def build_dataloaders(
    csv_file: str | Path,
    batch_size: int = 32,
    backend: str = 'torchvision',
    img_size: int = 224,
    augment: bool = True,
    normalize: bool = True,
    num_workers: int = 4,
    balanced: bool = False,
    drop_corrupted: bool = True,
) -> Dict[str, Any]:
    train_tf, val_tf = build_transforms(backend, img_size, augment, normalize)
    train_ds, train_loader = make_dataloader(csv_file, 'train', batch_size, train_tf, num_workers, balanced, drop_corrupted)
    val_ds, val_loader = make_dataloader(csv_file, 'val', batch_size, val_tf, num_workers, False, drop_corrupted)
    test_ds, test_loader = None, None
    try:
        test_ds, test_loader = make_dataloader(csv_file, 'test', batch_size, val_tf, num_workers, False, drop_corrupted)
    except ValueError:
        pass  # optional
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'test_dataset': test_ds,
        'class_to_idx': train_ds.class_to_idx,
        'idx_to_class': train_ds.idx_to_class,
    }

__all__ = [
    'ChestXrayDataset',
    'build_transforms',
    'make_dataloader',
    'build_dataloaders',
    'IMAGENET_MEAN',
    'IMAGENET_STD'
]
