"""Dataset collection & preparation utilities.

Supported datasets:
- NIH Chest X-Ray (manual / semi-automated via provided file IDs)
- RSNA Pneumonia (Kaggle competition)
- Kaggle Chest X-Ray Pneumonia (binary classification)

Examples:
python -m src.data.collect_data --dataset kaggle --extract
python -m src.data.collect_data --dataset rsna --extract --skip-existing
python -m src.data.collect_data --dataset nih --limit 2  # limit only affects how many NIH zips to attempt
python -m src.data.collect_data --dataset all --extract

Kaggle credentials required (env): KAGGLE_USERNAME / KAGGLE_KEY
NIH file IDs must be supplied or edited in-place (see NIH_ZIP_FILE_IDS placeholder).
"""
import argparse
import os
import subprocess
from pathlib import Path
import sys
import tarfile
import zipfile
import requests
from loguru import logger
import time
from typing import Iterable

from src.config import RAW_DIR, EXTERNAL_DIR


def download_sample(url: str, out_path: Path):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        logger.info(f"Downloaded sample file to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download sample: {e}")


# Dataset directory paths
NIH_DIR = EXTERNAL_DIR / 'nih_chest_xray'
RSNA_DIR = EXTERNAL_DIR / 'rsna_pneumonia'
KAGGLE_DIR = EXTERNAL_DIR / 'kaggle_pneumonia'


def ensure_dirs():
    for d in [RAW_DIR, EXTERNAL_DIR, NIH_DIR, RSNA_DIR, KAGGLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# --- NIH CONFIG -----------------------------------------------------------------
# Placeholder mapping of NIH zip file names to Box file IDs. Fill in real IDs before use.
# Order matters (first N used when --limit provided).
NIH_ZIP_FILE_IDS: list[tuple[str, str]] = [
    # ("images_part_01.zip", "<BOX_FILE_ID_01>"),
    # ("images_part_02.zip", "<BOX_FILE_ID_02>"),
    # ... up to part 12 ...
]
NIH_METADATA_FILES: list[tuple[str, str]] = [
    # ("Data_Entry_2017.csv", "<BOX_FILE_ID_METADATA>")
]

BOX_BASE = "https://nihcc.app.box.com/index.php?rm=download&file_id={file_id}"


def _download_with_progress(url: str, dest: Path, skip_existing: bool, desc: str = ""):
    if skip_existing and dest.exists():
        logger.info(f"Skip existing: {dest.name}")
        return True
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('Content-Length', 0))
            downloaded = 0
            start = time.time()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        sys.stdout.write(f"\r{desc} {pct:5.1f}% ({downloaded/1e6:,.1f}MB/{total/1e6:,.1f}MB)")
                        sys.stdout.flush()
            dur = time.time() - start
            sys.stdout.write("\n")
            logger.info(f"Downloaded {dest.name} in {dur:.1f}s")
        return True
    except Exception as e:
        logger.error(f"Download failed for {dest.name}: {e}")
        return False


# NIH dataset (placeholder: official site requires manual download / AWS links)
NIH_README_URL = "https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468"


def prepare_nih(limit: int | None = None, extract: bool = False, skip_existing: bool = False):
    ensure_dirs()
    if not NIH_ZIP_FILE_IDS:
        logger.warning("NIH_ZIP_FILE_IDS is empty. Populate with actual (filename, file_id) pairs.")
        (NIH_DIR / 'README_MANUAL_DOWNLOAD.txt').write_text(
            "Populate NIH_ZIP_FILE_IDS in collect_data.py with tuples: (filename, box_file_id)\n"
            "Then re-run: python -m src.data.collect_data --dataset nih --extract\n"
        )
        return
    targets = NIH_ZIP_FILE_IDS
    if limit:
        targets = targets[:limit]
    logger.info(f"Attempting {len(targets)} NIH archive downloads")
    for fname, fid in targets:
        url = BOX_BASE.format(file_id=fid)
        dest = NIH_DIR / fname
        _download_with_progress(url, dest, skip_existing, desc=f"NIH {fname}")
    # Metadata
    for fname, fid in NIH_METADATA_FILES:
        url = BOX_BASE.format(file_id=fid)
        dest = NIH_DIR / fname
        _download_with_progress(url, dest, skip_existing, desc=f"NIH {fname}")
    if extract:
        _auto_extract_archives(NIH_DIR)


# RSNA dataset (competition) – requires Kaggle competition acceptance.
RSNA_KAGGLE_REF = "rsna-pneumonia-detection-challenge"


def prepare_rsna(limit: int | None = None, extract: bool = False, skip_existing: bool = False):
    logger.warning("RSNA competition dataset requires Kaggle acceptance.")
    if not _kaggle_available():
        logger.error("Kaggle CLI not available. Skipping RSNA download.")
        return
    try:
        cmd = ["kaggle", "competitions", "download", "-c", RSNA_KAGGLE_REF, "-p", str(RSNA_DIR)]
        subprocess.run(cmd, check=True)
        if extract:
            _auto_extract_archives(RSNA_DIR)
        logger.info("RSNA dataset processed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"RSNA download failed: {e}")
    if limit:
        logger.info("Limit not implemented for RSNA.")


# Kaggle pneumonia dataset (simpler)
KAGGLE_DATASET_ID = "paultimothymooney/chest-xray-pneumonia"


def prepare_kaggle_pneumonia(limit: int | None = None, extract: bool = False, skip_existing: bool = False):
    if not _kaggle_available():
        logger.error("Kaggle CLI not available. Skipping Kaggle pneumonia dataset.")
        return
    try:
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_ID, "-p", str(KAGGLE_DIR)]
        subprocess.run(cmd, check=True)
        if extract:
            _auto_extract_archives(KAGGLE_DIR)
        if limit:
            logger.info("Limit parameter not used (dataset fully downloaded).")
        logger.info("Kaggle pneumonia dataset ready.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle pneumonia download failed: {e}")


def _kaggle_available() -> bool:
    return shutil_which('kaggle') is not None and os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY')


def shutil_which(cmd: str):  # local small helper to avoid importing shutil fully
    from shutil import which
    return which(cmd)


def _auto_extract_archives(folder: Path):
    for arch in folder.glob('*'):
        if arch.is_dir():
            continue
        if arch.suffix.lower() == '.zip':
            try:
                with zipfile.ZipFile(arch, 'r') as z:
                    z.extractall(folder)
                logger.info(f"Extracted {arch.name}")
            except Exception as e:
                logger.error(f"Failed extracting {arch}: {e}")
        elif arch.name.endswith('.tar.gz') or arch.suffix.lower() in {'.tgz'}:
            try:
                with tarfile.open(arch, 'r:*') as t:
                    t.extractall(folder)
                logger.info(f"Extracted {arch.name}")
            except Exception as e:
                logger.error(f"Failed extracting {arch}: {e}")


# Update dispatcher to pass new args
DATASET_FUNCS = {
    'nih': prepare_nih,
    'rsna': prepare_rsna,
    'kaggle': prepare_kaggle_pneumonia,
}


def main():
    parser = argparse.ArgumentParser(description="Collect medical imaging datasets")
    parser.add_argument('--dataset', choices=['nih', 'rsna', 'kaggle', 'all'], help='Dataset to download/prepare')
    parser.add_argument('--limit', type=int, default=None, help='Limit (only applies to NIH archives)')
    parser.add_argument('--sample', action='store_true', help='Download a tiny sample file (test)')
    parser.add_argument('--extract', action='store_true', help='Extract downloaded archives')
    parser.add_argument('--skip-existing', action='store_true', help='Do not re-download files that already exist')
    args = parser.parse_args()

    ensure_dirs()

    if args.sample:
        sample_url = "https://raw.githubusercontent.com/pytorch/hub/master/README.md"
        download_sample(sample_url, RAW_DIR / "sample.txt")

    if args.dataset is None:
        logger.info("No dataset specified. Use --dataset (nih|rsna|kaggle|all)")
        return

    targets = DATASET_FUNCS.keys() if args.dataset == 'all' else [args.dataset]
    for ds in targets:
        logger.info(f"Preparing dataset: {ds}")
        DATASET_FUNCS[ds](
            limit=args.limit,
            extract=args.extract,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
