"""Dataset & metadata validation utility.

Usage:
python -m src.data.validate --csv notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv --sample-check 50

Outputs (stdout):
- Column presence & basic stats
- Split counts & class distribution
- Missing / unreadable files summary
- Optional image dimension sampling (first N or random sample)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
from PIL import Image
import random

REQUIRED_COLS = {"split","class","path"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Path to metadata CSV')
    p.add_argument('--sample-check', type=int, default=40, help='How many images to open for dimension sanity (0 to skip)')
    p.add_argument('--random', action='store_true', help='Random sample instead of first N for dimension sanity')
    return p.parse_args()


def check_columns(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f'Missing required columns: {missing}')


def summarize_splits(df: pd.DataFrame):
    print('\n=== Split Counts ===')
    for split, g in df.groupby('split'):
        print(f"  {split:<6} : {len(g):5d}")
    print('\n=== Class Distribution Per Split ===')
    for split, g in df.groupby('split'):
        counts = g['class'].value_counts().to_dict()
        total = sum(counts.values())
        parts = ' '.join(f"{k}:{v}({v/total:.1%})" for k,v in counts.items())
        print(f"  {split:<6} : {parts}")


def check_paths(df: pd.DataFrame):
    print('\n=== File Existence Check ===')
    missing = []
    for p in df['path']:
        if not Path(p).is_file():
            missing.append(p)
    if missing:
        print(f"Missing files: {len(missing)} (showing first 5)")
        for m in missing[:5]:
            print('  -', m)
    else:
        print('All referenced files exist.')
    return missing


def sample_dimensions(df: pd.DataFrame, n: int, random_sample: bool):
    if n <= 0:
        return
    print(f"\n=== Dimension Sampling (n={n}) ===")
    rows = df
    if random_sample:
        rows = df.sample(min(n, len(df)), random_state=42)
    else:
        rows = df.head(n)
    dims = []
    corrupt = 0
    for _, r in rows.iterrows():
        path = Path(r['path'])
        try:
            with Image.open(path) as im:
                w,h = im.size
            dims.append((w,h))
        except Exception:
            corrupt += 1
    if dims:
        widths = [d[0] for d in dims]
        heights = [d[1] for d in dims]
        avg_w = sum(widths)/len(widths)
        avg_h = sum(heights)/len(heights)
        print(f"Avg W x H: {avg_w:.1f} x {avg_h:.1f}")
        print(f"Min W x H: {min(widths)} x {min(heights)} | Max W x H: {max(widths)} x {max(heights)}")
    print('Corrupt (open failures) in sample:', corrupt)


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f'CSV not found: {csv_path}')
    df = pd.read_csv(csv_path)
    print('Loaded rows:', len(df))
    check_columns(df)
    summarize_splits(df)
    missing = check_paths(df)
    sample_dimensions(df[df['split']=='train'], args.sample_check, args.random)
    print('\n=== Summary ===')
    print('Rows:', len(df), '| Missing files:', len(missing))
    print('Done.')

if __name__ == '__main__':
    main()
