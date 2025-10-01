from pathlib import Path
from typing import List


def list_images(directory: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png'}
    return [p for p in directory.iterdir() if p.suffix.lower() in exts]
