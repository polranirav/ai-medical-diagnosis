"""Grad-CAM utilities for model explainability.

Supports:
- ResNet-style architectures (default target: layer4[-1])
- DenseNet / EfficientNet automatic last conv detection
- Generating heatmap for top predicted or specified class
- Overlay & save functionality

Example usage:
from src.explainability.gradcam import GradCAM, generate_and_save
heatmap_path = generate_and_save(model, image_path, device, out_dir='results/gradcam')
"""
from __future__ import annotations
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer=None):
        self.model = model.eval()
        if target_layer is None:
            target_layer = self._infer_target_layer(model)
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _infer_target_layer(self, model):
        # ResNet style
        if hasattr(model, 'layer4'):
            try:
                return model.layer4[-1]
            except Exception:
                pass
        # torchvision wrappers (sometimes model.model.layer4)
        if hasattr(model, 'model') and hasattr(model.model, 'layer4'):
            try:
                return model.model.layer4[-1]
            except Exception:
                pass
        # DenseNet / EfficientNet / generic features attribute
        if hasattr(model, 'features'):
            # Find the last convolutional layer inside features
            for m in reversed(list(model.features.modules())):
                if isinstance(m, torch.nn.Conv2d):
                    return m
        # Fallback: search entire model
        for m in reversed(list(model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
        raise ValueError('Could not automatically infer target_layer; please provide explicitly.')

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()
        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.handles.append(self.target_layer.register_forward_hook(fwd_hook))
        # Using backward hook (deprecated but sufficient); for newer PyTorch could migrate to hooks on tensors
        self.handles.append(self.target_layer.register_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[:, class_idx]
        score.backward(retain_graph=True)
        grads = self.gradients  # [B,C,H,W]
        acts = self.activations # [B,C,H,W]
        if grads is None or acts is None:
            raise RuntimeError('Gradients or activations not captured. Ensure forward/backward hooks were registered.')
        weights = grads.mean(dim=[2,3], keepdim=True)  # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        cam = cam[0,0]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), class_idx


def default_val_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size,img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def load_image(path: str | Path, transform=None, device='cpu'):
    path = Path(path)
    img = Image.open(path).convert('RGB')
    orig = img.copy()
    if transform:
        tensor = transform(img).unsqueeze(0).to(device)
    else:
        tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    return tensor, orig


def overlay_heatmap(image_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, cmap='jet'):
    import matplotlib.cm as cm
    heatmap_resized = Image.fromarray(np.uint8(heatmap * 255)).resize(image_pil.size, resample=Image.BILINEAR)
    heatmap_np = np.array(heatmap_resized)
    colormap = cm.get_cmap(cmap)
    colored = colormap(heatmap_np/255.0)[:,:,:3]  # drop alpha
    img_np = np.array(image_pil).astype(float)/255.0
    overlay = (1-alpha)*img_np + alpha*colored
    overlay = np.clip(overlay,0,1)
    return (img_np, heatmap_np/255.0, overlay)


def generate_and_save(model, image_path, device='cpu', out_dir='results/gradcam', img_size=224, class_idx=None, alpha=0.5):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    transform = default_val_transform(img_size)
    tensor, orig = load_image(image_path, transform, device)
    gradcam = GradCAM(model)
    heatmap, used_class = gradcam(tensor, class_idx=class_idx)
    gradcam.remove_hooks()
    img_np, heat_np, overlay = overlay_heatmap(orig, heatmap, alpha=alpha)
    stem = Path(image_path).stem
    base = out_dir / f'{stem}_class{used_class}'
    plt.imsave(base.with_suffix('.orig.png'), img_np)
    plt.imsave(base.with_suffix('.heat.png'), heat_np, cmap='jet')
    plt.imsave(base.with_suffix('.overlay.png'), overlay)
    return {
        'original': base.with_suffix('.orig.png'),
        'heatmap': base.with_suffix('.heat.png'),
        'overlay': base.with_suffix('.overlay.png'),
        'class_idx': used_class
    }

__all__ = ['GradCAM','generate_and_save','default_val_transform']
