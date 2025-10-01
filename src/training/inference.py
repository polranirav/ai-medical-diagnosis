"""Inference utilities for trained checkpoints.

Usage:
python -m src.training.inference --checkpoint models/exp1/best.pt --image path/to/image.jpg
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T

from src.models.resnet_model import build_resnet18
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

val_tf = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--image', type=str, required=True)
    return p.parse_args()


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    return ckpt


def predict(model, image_path: str):
    model.eval()
    with Image.open(image_path) as im:
        im = im.convert('RGB')
    x = val_tf(im).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_resnet18(pretrained=False, freeze_backbone=False)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    probs = predict(model, args.image)
    classes = ['NORMAL','PNEUMONIA']
    for i, p in enumerate(probs):
        print(f"{classes[i]}: {p.item():.4f}")

if __name__ == '__main__':
    main()
