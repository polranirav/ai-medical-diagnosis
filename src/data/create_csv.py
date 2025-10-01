"""Utility to generate a simple CSV (image_path,label,split) from the
Kaggle Chest X-Ray Pneumonia dataset directory layout.

Prefers using existing richer metadata if available (metadata clean CSV), but
can fall back to a fresh scan here.
"""
from __future__ import annotations
import os, csv
from pathlib import Path
from typing import List
from PIL import Image, UnidentifiedImageError

EXPECTED_SPLITS: List[str] = ["train", "test", "val"]
EXPECTED_CLASSES: List[str] = ["NORMAL", "PNEUMONIA"]


def create_image_label_csv(data_dir: str | Path, output_csv: str | Path) -> None:
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)
    rows = []
    corrupt_files = []

    for split in EXPECTED_SPLITS:
        split_dir = data_dir / split
        for label in EXPECTED_CLASSES:
            label_dir = split_dir / label
            if not label_dir.exists():
                print(f"Warning: Directory missing: {label_dir}")
                continue
            for filename in os.listdir(label_dir):
                file_path = label_dir / filename
                if not file_path.is_file():
                    continue
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    rows.append([str(file_path), label, split])
                except (UnidentifiedImageError, OSError) as e:
                    print(f"Corrupt image detected: {file_path} | error: {e}")
                    corrupt_files.append(str(file_path))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path','label','split'])
        writer.writerows(rows)

    print(f"CSV written: {output_csv} (rows={len(rows)})")
    if corrupt_files:
        print(f"Corrupt images ({len(corrupt_files)}): first 5 shown")
        for c in corrupt_files[:5]:
            print('  ', c)


def main():  # pragma: no cover (script entrypoint)
    data_directory = Path('data/external/kaggle_pneumonia/chest_xray')
    output_csv_file = Path('data/processed/kaggle_pneumonia_images.csv')
    create_image_label_csv(data_directory, output_csv_file)

if __name__ == '__main__':
    main()
