"""Preprocessing & augmentation utilities.
- Load clean metadata & summarize
- Torchvision + optional Albumentations transforms
- Unnormalize helper for visualization
- Legacy tensor dump pipeline retained.
"""
import argparse
from pathlib import Path
from loguru import logger
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T
from typing import Tuple, Dict, Any
import os

from src.config import RAW_DIR, PROCESSED_DIR, IMG_SIZE
from src.utils.logger import init_logger

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_IMG_SIZE = 224


try:  # Optional albumentations backend
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except ImportError:  # pragma: no cover
    _HAS_ALB = False


def _resolve_metadata_path(csv_path: str | Path) -> Path:
    p = Path(csv_path)
    if p.is_file():
        return p
    # Always search common locations if direct path not found
    name = p.name
    search_dirs = [
        Path('.'),
        Path('notebooks/data/metadata'),
        Path('notebooks/data'),
        Path('notebooks'),
        Path('data/metadata'),
        Path('data'),
    ]
    candidates = []
    for d in search_dirs:
        cand = d / name
        candidates.append(cand)
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Metadata CSV not found: {csv_path}. Tried: {[str(c) for c in candidates]}"
    )


def load_clean_metadata(csv_path: str | Path, filter_corrupted: bool = True) -> pd.DataFrame:
    csv_path = _resolve_metadata_path(csv_path)
    df = pd.read_csv(csv_path)
    if filter_corrupted and 'corrupted' in df.columns:
        df = df[~df['corrupted']]
    return df.reset_index(drop=True)


def get_torchvision_train_transform(img_size: int = DEFAULT_IMG_SIZE, augment: bool = True, normalize: bool = True) -> T.Compose:
    ops: list[Any] = [T.Resize((img_size, img_size))]
    if augment:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.05,0.05,0.05,0.02)]
    ops += [T.ToTensor()]
    if normalize:
        ops += [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def get_torchvision_val_transform(img_size: int = DEFAULT_IMG_SIZE, normalize: bool = True) -> T.Compose:
    ops: list[Any] = [T.Resize((img_size, img_size)), T.ToTensor()]
    if normalize:
        ops += [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def get_albumentations_train_transform(img_size: int = DEFAULT_IMG_SIZE, normalize: bool = True, light: bool = False):  # pragma: no cover
    if not _HAS_ALB:
        raise ImportError('albumentations not installed. pip install albumentations[imgaug]')
    aug = [A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5)]
    if not light:
        aug += [A.RandomBrightnessContrast(p=0.3, brightness_limit=0.08, contrast_limit=0.08),
                A.CoarseDropout(max_holes=1, max_height=img_size//6, max_width=img_size//6, p=0.15),
                A.GaussianBlur(blur_limit=3, p=0.2)]
    if normalize:
        aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    aug += [ToTensorV2()]
    return A.Compose(aug)


def get_albumentations_val_transform(img_size: int = DEFAULT_IMG_SIZE, normalize: bool = True):  # pragma: no cover
    if not _HAS_ALB:
        raise ImportError('albumentations not installed. pip install albumentations[imgaug]')
    aug = [A.Resize(img_size, img_size)]
    if normalize:
        aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    aug += [ToTensorV2()]
    return A.Compose(aug)


def save_metadata(df: pd.DataFrame, out_csv: str | Path):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def build_basic_transforms():  # legacy
    return T.Compose([
        T.Resize(DEFAULT_IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def process_image(img_path: Path, out_dir: Path, transforms):
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = transforms(img)
        out_file = out_dir / (img_path.stem + '.npy')
        np.save(out_file, tensor.numpy())
        return True
    except Exception as e:
        logger.error(f"Failed processing {img_path}: {e}")
        return False


def run(limit: int | None = None):
    transforms = build_basic_transforms()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    images = list(RAW_DIR.glob('*.jpg')) + list(RAW_DIR.glob('*.png'))
    if limit:
        images = images[:limit]
    success = 0
    for p in images:
        if process_image(p, PROCESSED_DIR, transforms):
            success += 1
    logger.info(f"Processed {success}/{len(images)} images")


def main():
    init_logger()
    parser = argparse.ArgumentParser(description="Preprocess images")
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images')
    args = parser.parse_args()
    run(args.limit)


if __name__ == '__main__':
    main()

# ---- Extended Metadata Utilities ----

def standardize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'label' in df.columns and 'class' not in df.columns:
        df = df.rename(columns={'label':'class'})
    return df

def summarize_class_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    if {'split','class'}.issubset(df.columns):
        for split, g in df.groupby('split'):
            out[split] = g['class'].value_counts().to_dict()
    return out

def print_split_summary(df: pd.DataFrame):
    summary = summarize_class_counts(df)
    if not summary:
        print('No split/class summary available.')
        return
    print('Class counts per split:')
    for split, counts in summary.items():
        total = sum(counts.values())
        parts = ', '.join(f"{k}:{v}" for k,v in counts.items())
        print(f"  {split} ({total}): {parts}")

# ---- Transform Factory ----

def get_transforms(kind: str = 'torchvision', img_size: int = DEFAULT_IMG_SIZE, augment: bool = True, normalize: bool = True):
    if kind == 'torchvision':
        return (get_torchvision_train_transform(img_size, augment, normalize),
                get_torchvision_val_transform(img_size, normalize))
    if kind in {'alb','albumentations'}:
        return (get_albumentations_train_transform(img_size, normalize),
                get_albumentations_val_transform(img_size, normalize))
    raise ValueError(f'Unknown transform kind: {kind}')

# ---- Visualization Helper ----

def unnormalize_tensor(t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    import torch
    if not isinstance(t, torch.Tensor):
        return t
    t = t.clone()
    for c, m, s in zip(t, mean, std):
        c.mul_(s).add_(m)
    return t.clamp(0,1)

# ---- CLI Enhancements ----

def cli_preview(csv_path: str | Path, backend: str = 'torchvision'):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f'[warn] Metadata not found: {csv_path}'); return
    df = load_clean_metadata(csv_path)
    df = standardize_metadata_columns(df)
    print(f'Loaded rows: {len(df)}')
    print_split_summary(df)
    train_tf, val_tf = get_transforms(backend, DEFAULT_IMG_SIZE, augment=True, normalize=True)
    print(f'Train transform ({backend}):', train_tf)
    print(f'Val transform ({backend}):', val_tf)
