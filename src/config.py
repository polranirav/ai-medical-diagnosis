from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
EXTERNAL_DIR = DATA_DIR / 'external'
MODELS_DIR = ROOT / 'models'
LOG_DIR = ROOT / 'logs'

# Training defaults
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 10
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# API
API_HOST = '0.0.0.0'
API_PORT = 8000

# Create directories if not exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
