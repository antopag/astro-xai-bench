"""Download PLAsTiCC-2018 raw data for reproducible benchmarking.

Supports two sources:
  1. Kaggle CLI (requires ~/.kaggle/kaggle.json)
  2. Zenodo mirror (https://zenodo.org/record/2539456) — no auth needed

Only the *training* set files are required for astro-xai-bench:
  - training_set.csv          (object_id, mjd, passband, flux, flux_err, detected)
  - training_set_metadata.csv (object_id, hostgal_specz, hostgal_photoz, target, …)

Usage:
    python scripts/fetch_plasticc.py                 # auto-detect source
    python scripts/fetch_plasticc.py --source kaggle
    python scripts/fetch_plasticc.py --source zenodo
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "plasticc"

ZENODO_BASE = "https://zenodo.org/record/2539456/files"
TRAINING_FILES = [
    "training_set.csv",
    "training_set_metadata.csv",
    "test_set_metadata.csv",  # photo-z reference for Avocado augmentation (§3.6 Boone 2019)
]


def _files_present() -> bool:
    """Check whether the two required training files already exist."""
    return all((RAW_DIR / f).is_file() for f in TRAINING_FILES)


def download_kaggle() -> None:
    """Download full PLAsTiCC-2018 competition data via Kaggle CLI."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install with `pip install kaggle` "
            "and configure ~/.kaggle/kaggle.json."
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading PLAsTiCC from Kaggle…")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "PLAsTiCC-2018", "-p", str(RAW_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr or result.stdout}")

    for zf in RAW_DIR.glob("*.zip"):
        logger.info(f"Extracting {zf.name}")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(RAW_DIR)
        zf.unlink()
    logger.info("Kaggle download complete.")


def download_zenodo() -> None:
    """Download only the training files from the Zenodo mirror."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests is required for Zenodo download: pip install requests")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in TRAINING_FILES:
        dest = RAW_DIR / fname
        if dest.is_file():
            logger.info(f"  {fname} already exists, skipping.")
            continue
        url = f"{ZENODO_BASE}/{fname}"
        logger.info(f"Downloading {url} …")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        logger.info(f"  → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    logger.info("Zenodo download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PLAsTiCC-2018 raw data.")
    parser.add_argument(
        "--source",
        choices=["kaggle", "zenodo", "auto"],
        default="auto",
        help="Download source (default: auto — tries Kaggle first, falls back to Zenodo).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    args = parser.parse_args()

    if _files_present() and not args.force:
        logger.info(f"Training files already present in {RAW_DIR}. Use --force to re-download.")
        return

    if args.source == "kaggle":
        download_kaggle()
    elif args.source == "zenodo":
        download_zenodo()
    else:  # auto
        if shutil.which("kaggle"):
            try:
                download_kaggle()
                return
            except RuntimeError:
                logger.warning("Kaggle download failed, falling back to Zenodo.")
        download_zenodo()

    if not _files_present():
        raise RuntimeError(f"Download completed but required files not found in {RAW_DIR}.")
    logger.info("Done. Raw data ready.")


if __name__ == "__main__":
    main()
