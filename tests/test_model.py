import torch
from src.models.create_model import build_model


def test_model_forward():
    model = build_model()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 2)
