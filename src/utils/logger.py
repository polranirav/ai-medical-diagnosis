from pathlib import Path
from loguru import logger

from src.config import LOG_DIR


def init_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / 'app.log'
    logger.remove()
    logger.add(log_file, rotation='1 MB', retention=3, level='INFO')
    logger.add(lambda msg: print(msg, end=''))
    logger.info('Logger initialized')
    return logger
