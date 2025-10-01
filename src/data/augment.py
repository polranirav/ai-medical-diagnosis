"""Augmentation utilities using Albumentations."""
from __future__ import annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMG_SIZE


def get_train_augmentations():
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_valid_augmentations():
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
