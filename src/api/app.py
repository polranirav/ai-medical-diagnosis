"""FastAPI application for model inference with optional temperature calibration.
Set environment variables:
  MODEL_CKPT=models/exp_hydra/best.pt  (path to checkpoint)
  CALIBRATION_JSON=results/evaluation_ext/calibration.json (optional; contains temperature)
  PRED_THRESHOLD=0.5 (optional classification threshold for pneumonia probability)
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io
import os
import json
import torch
import torchvision.transforms as T
from typing import Optional
import hashlib

from src.training.checkpointing import FocalLoss
from src.models.resnet_model import build_resnet18
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

app = FastAPI(title="AI Medical Diagnosis API", version="0.2.0", openapi_tags=[
    {"name": "health", "description": "Service health & metadata"},
    {"name": "inference", "description": "Prediction and probability endpoints"}
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Pydantic Response Models -----------------
class HealthResponse(BaseModel):
    status: str = Field(example="ok")
    version: str

class PredictResponse(BaseModel):
    filename: str
    bytes: int
    is_image: bool
    prediction: str

class PredictProbaResponse(BaseModel):
    classes: list[str]
    probabilities: list[float]
    temperature: float
    threshold: float
    top_class: str
    top_prob: float
    calibrated: bool

class ModelInfoResponse(BaseModel):
    architecture: str
    checkpoint_path: str
    checkpoint_sha256: str | None
    temperature: float
    calibrated: bool
    threshold: float

# -----------------------------------------------------------
# Global variables and model loading
# -----------------------------------------------------------
_MODEL = None
_TEMPERATURE: Optional[torch.Tensor] = None
_THR: float | None = None
_MODEL_ARCH = 'resnet18'

VAL_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


def load_model():
    global _MODEL
    if _MODEL is None:
        ckpt_path = os.environ.get('MODEL_CKPT', 'models/exp_hydra/best.pt')
        model = build_resnet18(pretrained=False, freeze_backbone=False)
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state'])
        except Exception as e:
            print('[warn] Failed loading model checkpoint:', e)
        model.eval()
        _MODEL = model
    return _MODEL


def load_temperature():
    global _TEMPERATURE
    if _TEMPERATURE is None:
        calib_path = os.environ.get('CALIBRATION_JSON')
        if calib_path and os.path.exists(calib_path):
            try:
                with open(calib_path) as f:
                    data = json.load(f)
                Tval = data.get('temperature')
                if Tval and isinstance(Tval, (int, float)):
                    _TEMPERATURE = torch.tensor(float(Tval))
                    print(f"[info] Loaded calibration temperature T={_TEMPERATURE.item():.4f}")
            except Exception as e:
                print('[warn] Could not load calibration json:', e)
        if _TEMPERATURE is None:
            _TEMPERATURE = torch.tensor(1.0)
    return _TEMPERATURE


def _hash_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


@app.get('/health', response_model=HealthResponse, tags=["health"])
async def health():
    return {"status": "ok", "version": app.version}


@app.post('/predict', response_model=PredictResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    # Placeholder: echo filename & size
    content = await file.read()
    try:
        Image.open(io.BytesIO(content)).verify()
        valid = True
    except Exception:
        valid = False
    return {
        "filename": file.filename,
        "bytes": len(content),
        "is_image": valid,
        "prediction": "use /predict_proba for probabilities"
    }


@app.post('/predict_proba', response_model=PredictProbaResponse, tags=["inference"])
async def predict_proba(file: UploadFile = File(...)):
    content = await file.read()
    try:
        im = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    x = VAL_TF(im).unsqueeze(0)
    model = load_model()
    Tcal = load_temperature()
    with torch.no_grad():
        logits = model(x)
        logits = logits / Tcal  # temperature adjustment
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    threshold_env = os.environ.get('PRED_THRESHOLD')
    global _THR
    if _THR is None:
        try:
            _THR = float(threshold_env) if threshold_env is not None else 0.5
        except ValueError:
            _THR = 0.5
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    top_prob = probs[pred_idx]
    pneumonia_prob = probs[1]
    predicted_class = "PNEUMONIA" if pneumonia_prob >= _THR else "NORMAL"
    return {"classes": ["NORMAL","PNEUMONIA"],
            "probabilities": probs,
            "temperature": float(Tcal.item()),
            "threshold": float(_THR),
            "top_class": predicted_class,
            "top_prob": pneumonia_prob if predicted_class=="PNEUMONIA" else probs[0],
            "calibrated": bool(Tcal.item() != 1.0)}


@app.get('/model_info', response_model=ModelInfoResponse, tags=["health"])
async def model_info():
    model = load_model()
    Tcal = load_temperature()
    ckpt_path = os.environ.get('MODEL_CKPT', 'models/exp_hydra/best.pt')
    h = _hash_file(ckpt_path)
    thr = float(os.environ.get('PRED_THRESHOLD', 0.5))
    return {
        'architecture': _MODEL_ARCH,
        'checkpoint_path': ckpt_path,
        'checkpoint_sha256': h,
        'temperature': float(Tcal.item()),
        'calibrated': bool(Tcal.item() != 1.0),
        'threshold': thr
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
