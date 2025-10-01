#!/usr/bin/env bash
set -euo pipefail
# Resolve Python interpreter robustly
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    echo "Error: No python interpreter found (tried 'python', 'python3', '.venv/bin/python')." >&2
    exit 127
  fi
fi

CKPT=${1:-models/exp_resnet18_tv/best.pt}
CSV=${2:-notebooks/data/metadata/kaggle_pneumonia_metadata_clean.csv}
OUT=${3:-results/gradcam_examples}
NUM=${4:-5}
MODEL=${5:-resnet18} # resnet18|densenet121|efficientnet_b0
DEVICE=${DEVICE:-cpu}
ALPHA=${ALPHA:-0.5}
IMG_SIZE=${IMG_SIZE:-224}
MONTAGE=${MONTAGE:-1}
UPLOAD_WANDB=${UPLOAD_WANDB:-0}

export CKPT CSV OUT NUM MODEL DEVICE ALPHA IMG_SIZE MONTAGE UPLOAD_WANDB

$PYTHON_BIN - <<'PY'
import os, pandas as pd, torch, math
from pathlib import Path
from PIL import Image
from src.explainability.gradcam import generate_and_save
from src.training.inference import load_checkpoint

ckpt_path = Path(os.environ['CKPT'])
csv_path = Path(os.environ['CSV'])
out_dir = Path(os.environ['OUT']); out_dir.mkdir(parents=True, exist_ok=True)
model_name = os.environ['MODEL'].lower()
device = os.environ['DEVICE']
alpha = float(os.environ['ALPHA'])
img_size = int(os.environ['IMG_SIZE'])
print(f"[INFO] Using device={device} model={model_name} ckpt={ckpt_path}")

import torchvision.models as tvm
num_classes = 2
if model_name == 'resnet18':
    from src.models.resnet_model import build_resnet18
    model = build_resnet18(num_classes=num_classes, pretrained=False, freeze_backbone=False, dropout=0.0)
elif model_name == 'densenet121':
    model = tvm.densenet121(weights=None)
    in_features = model.classifier.in_features
    import torch.nn as nn
    model.classifier = nn.Linear(in_features, num_classes)
elif model_name == 'efficientnet_b0':
    model = tvm.efficientnet_b0(weights=None)
    import torch.nn as nn
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
else:
    raise SystemExit(f"Unsupported model: {model_name}")

try:
    load_checkpoint(model, ckpt_path)
except Exception:
    ck = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ck.get('model_state', ck))
model.to(device).eval()

df = pd.read_csv(csv_path)
if 'split' in df.columns:
    subset = df[df['split']=='test']
    if subset.empty:
        subset = df[df['split']=='val']
else:
    subset = df
subset = subset.sample(min(len(subset), int(os.environ['NUM'])))
col = 'path' if 'path' in subset.columns else 'image_path'

results = []
for img_path in subset[col]:
    try:
        r = generate_and_save(model, img_path, device=device, out_dir=out_dir, img_size=img_size, alpha=alpha)
        results.append(r)
        print('Generated:', r['overlay'])
    except Exception as e:
        print('Failed on', img_path, ':', e)

print(f"[INFO] Generated {len(results)} Grad-CAM overlays -> {out_dir}")

if int(os.environ['MONTAGE']) and results:
    overlays = [r['overlay'] for r in results]
    imgs = [Image.open(p) for p in overlays]
    cols = min(3, len(imgs))
    rows = math.ceil(len(imgs)/cols)
    w,h = imgs[0].size
    canvas = Image.new('RGB', (cols*w, rows*h), (0,0,0))
    for idx,im in enumerate(imgs):
        r,c = divmod(idx, cols)
        canvas.paste(im,(c*w, r*h))
    grid_path = out_dir / '_grid_overlay.png'
    canvas.save(grid_path)
    print('[INFO] Montage saved:', grid_path)

if int(os.environ['UPLOAD_WANDB']):
    try:
        import wandb
        wandb.init(project='ai-medical-diagnosis', name='gradcam_'+ckpt_path.parent.name)
        tbl = wandb.Table(columns=['original','heatmap','overlay','class_idx'])
        for r in results:
            tbl.add_data(str(r['original']), str(r['heatmap']), str(r['overlay']), r['class_idx'])
        wandb.log({'gradcam_examples': tbl})
        if (out_dir / '_grid_overlay.png').exists():
            from wandb import Image as WBImage
            wandb.log({'gradcam_grid': WBImage(str(out_dir / '_grid_overlay.png'))})
        wandb.finish()
        print('[INFO] Logged Grad-CAM artifacts to W&B.')
    except Exception as e:
        print('[WARN] W&B logging skipped:', e)
PY
