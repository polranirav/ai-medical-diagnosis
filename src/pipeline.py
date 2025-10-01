"""End-to-end pipeline orchestration."""
from __future__ import annotations
from loguru import logger

from src.data import preprocess
from src.config import RANDOM_SEED
import torch
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(RANDOM_SEED)
    logger.info("Running preprocessing step...")
    preprocess.run(limit=10)
    logger.info("Pipeline completed.")


if __name__ == '__main__':
    main()
