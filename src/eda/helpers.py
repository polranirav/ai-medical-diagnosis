"""EDA helper functions for chest X-ray pneumonia dataset.
Split from notebook to allow reuse & testing.
"""
from __future__ import annotations
import os, math, hashlib, random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, it, **kwargs): self.it = it
        def __iter__(self):
            for x in self.it: yield x
        def update(self, *_): pass
        def close(self): pass

# Configuration defaults (can be overridden after import)
EXPECTED_SPLITS: List[str] = ['train','val','test']
EXPECTED_CLASSES: List[str] = ['NORMAL','PNEUMONIA']
FILE_EXTENSIONS = {'.jpeg','.jpg','.png'}
IMBALANCE_THRESHOLD = 0.4
ASPECT_ZSCORE_THRESHOLD = 3.0
FILESIZE_ZSCORE_THRESHOLD = 3.0
HIST_SAMPLE_SIZE = 200
RANDOM_SEED = 42
ENABLE_TQDM = True

sns.set_style('whitegrid')
plt.rcParams.setdefault('figure.dpi', 110)

EXPECTED_META_COLUMNS = ['split','class','path','filename','size_bytes','width','height','aspect_ratio','corrupted']

def _resolve_dataset_root(root: Path) -> Path:
    if all((root / s).exists() for s in EXPECTED_SPLITS):
        return root
    nested = root / root.name
    if nested.exists() and all((nested / s).exists() for s in EXPECTED_SPLITS):
        return nested
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and all((child / s).exists() for s in EXPECTED_SPLITS):
            return child
    return root


def scan_dataset(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Traverse dataset directory building metadata & collecting structural issues."""
    original_root = root
    root = _resolve_dataset_root(root)
    recs=[]; issues=[]
    for split in EXPECTED_SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            issues.append({'type':'missing_split','split':split,'path':str(split_dir)})
            continue
        for cls in EXPECTED_CLASSES:
            class_dir = split_dir / cls
            if not class_dir.exists():
                issues.append({'type':'missing_class_dir','split':split,'class':cls,'path':str(class_dir)})
                continue
            try:
                names = [f for f in os.listdir(class_dir) if (class_dir / f).is_file()]
            except FileNotFoundError:
                issues.append({'type':'unreadable_dir','split':split,'class':cls,'path':str(class_dir)})
                continue
            iterator = tqdm(names, desc=f'Scanning {split}/{cls}', leave=False) if ENABLE_TQDM else names
            for nm in iterator:
                p = class_dir / nm
                if p.suffix.lower() not in FILE_EXTENSIONS: continue
                try: size_bytes = p.stat().st_size
                except FileNotFoundError:
                    issues.append({'type':'missing_file','split':split,'class':cls,'file':nm,'path':str(p)})
                    continue
                w=h=None; corrupted=False
                try:
                    with Image.open(p) as im: im.verify()
                    with Image.open(p) as im2: w,h = im2.size
                except (UnidentifiedImageError, OSError) as e:
                    corrupted=True
                    issues.append({'type':'corrupt_image','split':split,'class':cls,'file':nm,'error':str(e)})
                recs.append({'split':split,'class':cls,'path':str(p),'filename':nm,'size_bytes':size_bytes,'width':w,'height':h,'aspect_ratio': (w/h) if (w and h and h!=0) else None,'corrupted':corrupted})
    df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=EXPECTED_META_COLUMNS)
    if 'corrupted' not in df.columns: df['corrupted']=False
    return df, pd.DataFrame(issues)


def add_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['corrupted','width','height','aspect_ratio','size_bytes']:
        if col not in df.columns: df[col] = np.nan if col!='corrupted' else False
    df['aspect_ratio_outlier']=False; df['size_bytes_outlier']=False
    if df.empty: return df
    valid = df[(~df['corrupted']) & df['width'].notna() & df['height'].notna() & df['aspect_ratio'].notna()]
    if not valid.empty:
        ar = valid['aspect_ratio']; ar_std = ar.std(ddof=0) or 0
        if ar_std: df.loc[valid.index,'aspect_ratio_outlier'] = ((ar-ar.mean()).abs()/ar_std) > ASPECT_ZSCORE_THRESHOLD
        sz = valid['size_bytes'].dropna(); sz_std = sz.std(ddof=0) or 0
        if sz_std: df.loc[sz.index,'size_bytes_outlier'] = ((sz - sz.mean()).abs()/sz_std) > FILESIZE_ZSCORE_THRESHOLD
    return df


def compute_class_distributions(meta: pd.DataFrame):
    if meta.empty: return pd.DataFrame(columns=['split','class','count']), pd.DataFrame(columns=['class','count'])
    ps = meta.groupby(['split','class']).size().reset_index(name='count')
    ov = meta.groupby('class').size().reset_index(name='count')
    return ps, ov


def class_imbalance_warnings(overall: pd.DataFrame):
    warns=[]
    if overall.empty: return warns
    counts = overall.set_index('class')['count']
    if counts.empty: return warns
    ratio = counts.min()/counts.max() if counts.max()>0 else 1
    if ratio < IMBALANCE_THRESHOLD:
        warns.append(f'Min/Max ratio {ratio:.3f} < {IMBALANCE_THRESHOLD}')
    return warns


def plot_class_distributions(ps: pd.DataFrame, ov: pd.DataFrame):
    if not ps.empty:
        plt.figure(figsize=(8,4)); sns.barplot(data=ps,x='split',y='count',hue='class'); plt.title('Class Distribution per Split'); plt.tight_layout(); plt.show()
    if not ov.empty:
        plt.figure(figsize=(4,4)); sns.barplot(data=ov,x='class',y='count'); plt.title('Overall Class Counts'); plt.tight_layout(); plt.show()


def plot_geometry(meta: pd.DataFrame):
    if meta.empty: print('No data for geometry plots.'); return
    valid = meta.dropna(subset=['width','height','aspect_ratio'])
    if valid.empty: print('No valid dimension data.'); return
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    sns.histplot(valid['aspect_ratio'], bins=40, ax=ax[0], color='teal'); ax[0].set_title('Aspect Ratio Histogram')
    sns.boxplot(x=valid['aspect_ratio'], ax=ax[1], color='lightblue'); ax[1].set_title('Aspect Ratio Boxplot')
    plt.tight_layout(); plt.show()
    plt.figure(figsize=(5,5)); plt.scatter(valid['width'], valid['height'], s=10, alpha=0.3); plt.xlabel('Width'); plt.ylabel('Height'); plt.title('Width vs Height'); plt.tight_layout(); plt.show()
    if valid['size_bytes'].notna().any():
        plt.figure(figsize=(6,4)); sns.histplot(valid['size_bytes'], bins=50, color='slateblue'); plt.title('File Size Distribution'); plt.tight_layout(); plt.show()


def pixel_intensity_hist(meta: pd.DataFrame):
    if meta.empty: print('No images for intensity hist.'); return
    valid = meta[~meta['corrupted'] & meta['width'].notna()]
    if valid.empty: print('No valid images for intensity hist.'); return
    sample = valid.sample(min(HIST_SAMPLE_SIZE, len(valid)), random_state=RANDOM_SEED)
    pixels=[]
    for p in sample['path']:
        try:
            with Image.open(p) as im: arr = np.array(im.convert('L')); pixels.append(arr.flatten())
        except Exception: continue
    if not pixels: print('No pixels collected.'); return
    vals = np.concatenate(pixels); mean_v, med_v = vals.mean(), np.median(vals)
    plt.figure(figsize=(6,4)); plt.hist(vals, bins=50, color='steelblue', edgecolor='black'); plt.xlim(0,255)
    plt.title(f'Pixel Intensity Sample\nMean={mean_v:.1f} Median={med_v:.1f}'); plt.xlabel('Intensity'); plt.ylabel('Freq'); plt.tight_layout(); plt.show()


def show_random_grid(meta: pd.DataFrame, cls: str, split: str, n: int=6, cols: int=3):
    subset = meta[(meta['split']==split)&(meta['class']==cls)&(~meta['corrupted'])]
    if subset.empty: print(f'No images for {cls} {split}'); return
    n=min(n,len(subset)); rows=math.ceil(n/cols); chosen=subset.sample(n, random_state=RANDOM_SEED)
    plt.figure(figsize=(cols*3, rows*3))
    for i, (_, r) in enumerate(chosen.iterrows()):
        try:
            with Image.open(r['path']) as im:
                plt.subplot(rows, cols, i+1); plt.imshow(im, cmap='gray'); plt.axis('off'); plt.title(cls)
        except Exception: continue
    plt.suptitle(f'{cls} samples ({split})'); plt.tight_layout(); plt.show()


def consolidated_report(meta: pd.DataFrame, issues: pd.DataFrame):
    ps, ov = compute_class_distributions(meta)
    warns = class_imbalance_warnings(ov)
    return {
        'total_images': int(len(meta)),
        'corrupted_images': int(meta['corrupted'].sum()) if 'corrupted' in meta.columns else 0,
        'missing_splits': int((issues['type']=='missing_split').sum()) if not issues.empty else 0,
        'missing_class_dirs': int((issues['type']=='missing_class_dir').sum()) if not issues.empty else 0,
        'class_counts_overall': ov.set_index('class')['count'].to_dict() if not ov.empty else {},
        'aspect_outliers': int(meta.get('aspect_ratio_outlier', pd.Series(dtype=bool)).sum()) if not meta.empty else 0,
        'size_outliers': int(meta.get('size_bytes_outlier', pd.Series(dtype=bool)).sum()) if not meta.empty else 0,
        'warnings': warns
    }


def md5_file(path: str | Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def compute_duplicates(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty: return pd.DataFrame(columns=['md5','count'])
    if 'md5' not in meta.columns:
        # compute md5 for each file
        hashes=[]
        for p in meta['path']:
            try:
                hashes.append(md5_file(p))
            except Exception:
                hashes.append(None)
        meta = meta.copy(); meta['md5'] = hashes
    dup_counts = meta.groupby('md5').size().reset_index(name='count')
    dup_counts = dup_counts[dup_counts['count']>1].sort_values('count', ascending=False)
    return dup_counts

__all__ = [
    'scan_dataset','add_outlier_flags','compute_class_distributions','class_imbalance_warnings',
    'plot_class_distributions','plot_geometry','pixel_intensity_hist','show_random_grid',
    'consolidated_report','compute_duplicates','md5_file'
]
