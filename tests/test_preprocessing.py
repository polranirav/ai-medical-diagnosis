from pathlib import Path
import numpy as np

from src.config import PROCESSED_DIR


def test_processed_outputs_shape(tmp_path, monkeypatch):
    # Placeholder test: ensure processed directory exists
    assert PROCESSED_DIR.exists()
