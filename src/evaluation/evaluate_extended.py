"""Extended evaluation utilities: threshold tuning, calibration (temperature scaling),
precision-recall curve, reliability (calibration) plot generation, and ONNX export helper.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

from src.models.resnet_model import build_resnet18
from src.data.dataset import build_dataloaders
from .evaluate import ModelEvaluator  # reuse if needed


def collect_logits_targets(model, loader, device):
    model.eval()
    logits_list = []
    targets_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            logits_list.append(out.cpu())
            targets_list.append(y.cpu())
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)
    return logits, targets


def temperature_scale(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50):
    # Optimize temperature T to minimize NLL
    import torch.nn.functional as F
    T = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / T, targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return T.detach()


def apply_temperature(logits: torch.Tensor, T: torch.Tensor):
    return logits / T


def threshold_sweep(probs_pos: np.ndarray, y_true: np.ndarray, steps: int = 101):
    thresholds = np.linspace(0,1,steps)
    rows = []
    for t in thresholds:
        preds = (probs_pos >= t).astype(int)
        tp = ((preds==1)&(y_true==1)).sum()
        tn = ((preds==0)&(y_true==0)).sum()
        fp = ((preds==1)&(y_true==0)).sum()
        fn = ((preds==0)&(y_true==1)).sum()
        precision = tp / max(tp+fp,1)
        recall = tp / max(tp+fn,1)
        specificity = tn / max(tn+fp,1)
        f1 = (2*precision*recall)/max(precision+recall,1e-9)
        rows.append({'threshold':t,'precision':precision,'recall':recall,'specificity':specificity,'f1':f1})
    return pd.DataFrame(rows)


def plot_pr_curve(y_true, probs_pos, out_dir: Path):
    precision, recall, _ = precision_recall_curve(y_true, probs_pos)
    ap = average_precision_score(y_true, probs_pos)
    plt.figure()
    plt.plot(recall, precision, label=f'PR (AP={ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    path = out_dir / 'precision_recall_curve.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path, ap


def plot_reliability(y_true, probs_pos, out_dir: Path, bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, probs_pos, n_bins=bins, strategy='uniform')
    plt.figure()
    plt.plot(mean_pred, frac_pos, 'o-', label='Empirical')
    plt.plot([0,1],[0,1],'--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Reliability Diagram')
    plt.legend()
    path = out_dir / 'reliability_diagram.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def export_onnx(model, out_path: Path, img_size: int = 224):
    model.eval()
    dummy = torch.randn(1,3,img_size,img_size)
    torch.onnx.export(model, dummy, out_path, input_names=['input'], output_names=['logits'], opset_version=17)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Extended evaluation: threshold tuning, calibration, PR & reliability, ONNX export.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output-dir', default='results/evaluation_ext')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders = build_dataloaders(args.csv, batch_size=32, backend='torchvision', img_size=args.img_size, augment=False, normalize=True)
    eval_loader = loaders['test'] or loaders['val']

    model = build_resnet18(pretrained=False, freeze_backbone=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    logits, targets = collect_logits_targets(model, eval_loader, device)
    probs = torch.softmax(logits, dim=1)[:,1].numpy()
    y_true = targets.numpy()

    # Threshold sweep
    thr_df = threshold_sweep(probs, y_true)
    thr_df.to_csv(out_dir / 'threshold_sweep.csv', index=False)

    # PR curve
    pr_path, ap = plot_pr_curve(y_true, probs, out_dir)

    # Reliability diagram & Brier score
    rel_path = plot_reliability(y_true, probs, out_dir)
    brier = brier_score_loss(y_true, probs)

    calib_T = None
    if args.calibrate:
        calib_T = temperature_scale(logits, targets)
        logits_cal = apply_temperature(logits, calib_T)
        probs_cal = torch.softmax(logits_cal, dim=1)[:,1].numpy()
        rel_path_cal = plot_reliability(y_true, probs_cal, out_dir)
        with (out_dir / 'calibration.json').open('w') as f:
            json.dump({'temperature': float(calib_T.item()), 'brier_before': brier, 'brier_after': brier_score_loss(y_true, probs_cal)}, f, indent=2)

    # Save summary
    summary = {
        'average_precision': float(ap),
        'brier_score': float(brier),
        'calibrated': calib_T is not None,
        'temperature' : float(calib_T.item()) if calib_T is not None else None,
        'n_samples': int(len(y_true)),
    }
    with (out_dir / 'extended_metrics.json').open('w') as f:
        json.dump(summary, f, indent=2)

    # Optional ONNX export
    if args.export_onnx:
        onnx_path = export_onnx(model, out_dir / 'model.onnx', img_size=args.img_size)
        print('Exported ONNX to', onnx_path)

    print('Extended evaluation complete.')

if __name__ == '__main__':
    main()
