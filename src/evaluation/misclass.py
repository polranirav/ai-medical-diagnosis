"""Misclassification analysis utilities.

Generates:
- CSV with all predictions, marking FP/FN
- Grids of False Positives & False Negatives (top-N by model confidence)

Usage:
python -m src.evaluation.misclass \
  --checkpoint models/exp_resnet18_tv/best.pt \
  --csv notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv \
  --split val \
  --out results/misclass/exp_resnet18_tv --topn 12
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

from src.models.resnet_model import build_resnet18
from src.data.preprocess import IMAGENET_MEAN, IMAGENET_STD, unnormalize_tensor

VAL_TF = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

CLASS_ORDER = ['NORMAL','PNEUMONIA']
LABEL_MAP = {c:i for i,c in enumerate(CLASS_ORDER)}
INV_LABEL = {v:k for k,v in LABEL_MAP.items()}

def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    return ckpt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--split', default='val', choices=['train','val','test'])
    p.add_argument('--out', default='results/misclass/run1')
    p.add_argument('--topn', type=int, default=12)
    p.add_argument('--img-size', type=int, default=224)
    return p.parse_args()

@torch.no_grad()
def predict_df(model, df, img_size=224):
    model.eval()
    records = []
    tf = T.Compose([
        T.Resize((img_size,img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    for _, row in df.iterrows():
        path = Path(row['path'] if 'path' in row else row['image_path'])
        try:
            with Image.open(path) as im: im = im.convert('RGB')
        except Exception:
            continue
        x = tf(im).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        records.append({
            'image_path': str(path),
            'true_class': row['class'] if 'class' in row else row.get('label'),
            'true_idx': LABEL_MAP[row['class'] if 'class' in row else row.get('label')],
            'pred_idx': pred_idx,
            'pred_class': INV_LABEL[pred_idx],
            'prob_normal': float(probs[0]),
            'prob_pneumonia': float(probs[1]),
            'confidence': float(probs[pred_idx])
        })
    return pd.DataFrame(records)

def make_grid(df, kind: str, out_dir: Path, topn: int, img_size=224):
    if df.empty:
        return None
    sel = df.head(topn)
    n = len(sel)
    cols = min(4, n)
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat: ax.axis('off')
    for (idx, row), ax in zip(sel.iterrows(), axes.flat):
        path = Path(row['image_path'])
        try:
            with Image.open(path) as im: im = im.convert('RGB')
            im = im.resize((img_size,img_size))
            ax.imshow(im)
            ax.set_title(f"{row['true_class']}→{row['pred_class']}\nconf={row['confidence']:.2f}", fontsize=8)
        except Exception:
            ax.set_title('Load err')
    fig.suptitle(f'{kind} (top {len(sel)})', fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f'{kind.lower().replace(" ", "_")}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if 'corrupted' in df.columns:
        df = df[~df['corrupted']]
    df = df[df['split']==args.split].copy()
    if df.empty:
        raise SystemExit(f'No rows for split {args.split}')

    model = build_resnet18(pretrained=False, freeze_backbone=False)
    load_checkpoint(model, args.checkpoint)

    pred_df = predict_df(model, df, img_size=args.img_size)
    pred_df['correct'] = pred_df['true_idx'] == pred_df['pred_idx']
    pred_df['error_type'] = np.where(pred_df['correct'], 'CORRECT', 'FP')
    # FP/FN classification (assuming class 1 is pneumonia)
    pred_df.loc[(pred_df.true_idx==1) & (pred_df.pred_idx==0), 'error_type'] = 'FN'
    pred_df.loc[(pred_df.true_idx==0) & (pred_df.pred_idx==1), 'error_type'] = 'FP'

    pred_df.to_csv(out_dir / 'predictions.csv', index=False)

    # Sort FP (pred pneumonia but actually normal) by prob_pneumonia descending
    fp_df = pred_df[pred_df.error_type=='FP'].sort_values('prob_pneumonia', ascending=False)
    fn_df = pred_df[pred_df.error_type=='FN'].sort_values('prob_pneumonia', ascending=True)  # low pneumonia prob

    fp_grid = make_grid(fp_df, 'False Positives', out_dir, args.topn, img_size=args.img_size)
    fn_grid = make_grid(fn_df, 'False Negatives', out_dir, args.topn, img_size=args.img_size)

    summary = {
        'total': len(pred_df),
        'accuracy': float((pred_df.correct).mean()),
        'fp_count': int((pred_df.error_type=='FP').sum()),
        'fn_count': int((pred_df.error_type=='FN').sum()),
        'fp_grid': str(fp_grid) if fp_grid else None,
        'fn_grid': str(fn_grid) if fn_grid else None,
    }
    import json
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    print('Summary:', summary)
    print('Artifacts in', out_dir)

if __name__ == '__main__':
    main()
