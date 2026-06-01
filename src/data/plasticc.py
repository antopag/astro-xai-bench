"""PLAsTiCC dataset loading and preprocessing."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# PLAsTiCC class mapping: target → human-readable name
CLASS_MAP: dict[int, str] = {
    6: "µ-Lens-Single",
    15: "TDE",
    16: "EBE",
    42: "SNIax",
    52: "SNIa-peculiar",
    53: "SNII-NL",  # non-linear
    62: "SNIbc",
    64: "KN",
    65: "M-dwarf",
    67: "SNIa",
    88: "AGN",
    90: "SNIa-91bg",
    92: "RRL",
    95: "SLSN-I",
}

# Passband mapping in PLAsTiCC
PASSBAND_NAMES: dict[int, str] = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "Y"}
N_PASSBANDS: int = 6


def download_plasticc(output_dir: str | Path) -> None:
    """Download PLAsTiCC dataset from Kaggle.

    Args:
        output_dir: Directory to save raw data files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading PLAsTiCC to {output_dir}")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "PLAsTiCC-2018", "-p", str(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Kaggle stdout: {result.stdout}")
        logger.error(f"Kaggle stderr: {result.stderr}")
        raise RuntimeError(f"Kaggle download failed: {result.stderr or result.stdout}")

    # Extract zip files
    import zipfile

    for zf in output_dir.glob("*.zip"):
        logger.info(f"Extracting {zf.name}")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(output_dir)
        zf.unlink()

    logger.info("Download complete.")


def _build_light_curves(
    meta: pd.DataFrame,
    lc_df: pd.DataFrame,
    max_len: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-length multi-band light curve arrays from raw DataFrames.

    Args:
        meta: Metadata DataFrame with 'object_id' and 'target' columns.
        lc_df: Light curve DataFrame with columns: object_id, mjd, passband, flux, flux_err, detected.
        max_len: Maximum number of timesteps per passband.

    Returns:
        Tuple of (light_curves, labels, object_ids):
            - light_curves: shape (n_objects, n_passbands, max_len, 2) — flux and flux_err
            - labels: shape (n_objects,) — integer class indices (0-based)
            - object_ids: shape (n_objects,)
    """
    # Create 0-based label mapping
    unique_targets = sorted(meta["target"].unique())
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}

    object_ids = meta["object_id"].values
    n_objects = len(object_ids)

    light_curves = np.zeros((n_objects, N_PASSBANDS, max_len, 2), dtype=np.float32)
    labels = np.array([target_to_idx[t] for t in meta["target"].values], dtype=np.int64)

    # Group light curve data by object
    grouped = lc_df.groupby("object_id")

    for i, oid in enumerate(object_ids):
        if oid not in grouped.groups:
            continue
        obj_lc = grouped.get_group(oid)

        for pb in range(N_PASSBANDS):
            band_data = obj_lc[obj_lc["passband"] == pb].sort_values("mjd")
            n_pts = min(len(band_data), max_len)
            if n_pts == 0:
                continue
            light_curves[i, pb, :n_pts, 0] = band_data["flux"].values[:n_pts]
            light_curves[i, pb, :n_pts, 1] = band_data["flux_err"].values[:n_pts]

        if (i + 1) % 5000 == 0:
            logger.info(f"  Processed {i + 1}/{n_objects} objects")

    return light_curves, labels, object_ids


def _normalize_light_curves(light_curves: np.ndarray) -> np.ndarray:
    """Per-object, per-band normalization to zero mean and unit variance.

    Args:
        light_curves: shape (n_objects, n_passbands, max_len, 2).

    Returns:
        Normalized array (same shape). Flux_err is scaled by same factor as flux.
    """
    normed = light_curves.copy()
    for i in range(normed.shape[0]):
        for pb in range(normed.shape[1]):
            flux = normed[i, pb, :, 0]
            mask = flux != 0.0  # non-padded entries
            if mask.sum() < 2:
                continue
            mu = flux[mask].mean()
            std = flux[mask].std()
            if std < 1e-8:
                std = 1.0
            normed[i, pb, mask, 0] = (flux[mask] - mu) / std
            normed[i, pb, mask, 1] = normed[i, pb, mask, 1] / std  # scale errors consistently
    return normed


def preprocess_plasticc(
    raw_dir: str | Path,
    output_dir: str | Path,
    max_len: int = 256,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> None:
    """Preprocess PLAsTiCC: normalize, handle missing data, stratified split.

    Saves .npz files for train/val/test with keys: light_curves, labels, object_ids.
    Also saves label_map.npy (target_to_idx mapping).

    Args:
        raw_dir: Path to raw CSV files.
        output_dir: Path to save processed data.
        max_len: Max timesteps per passband.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining after test).
        seed: Random seed.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training metadata and light curves (we use the training set which has labels)
    logger.info("Loading PLAsTiCC training metadata...")
    meta = pd.read_csv(raw_dir / "training_set_metadata.csv")
    logger.info(f"  {len(meta)} objects, {meta['target'].nunique()} classes")

    logger.info("Loading PLAsTiCC training light curves...")
    lc_df = pd.read_csv(raw_dir / "training_set.csv")
    logger.info(f"  {len(lc_df)} observations")

    # Build fixed-length arrays
    logger.info("Building light curve arrays...")
    light_curves, labels, object_ids = _build_light_curves(meta, lc_df, max_len=max_len)

    # Normalize
    logger.info("Normalizing light curves...")
    light_curves = _normalize_light_curves(light_curves)

    # Stratified split: first train+val vs test, then train vs val
    logger.info("Splitting train/val/test...")
    idx_trainval, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size / (1.0 - test_size),
        stratify=labels[idx_trainval],
        random_state=seed,
    )

    # Save splits
    for split_name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        out_path = output_dir / f"{split_name}.npz"
        np.savez_compressed(
            out_path,
            light_curves=light_curves[idx],
            labels=labels[idx],
            object_ids=object_ids[idx],
        )
        logger.info(f"  {split_name}: {len(idx)} objects → {out_path}")

    # Save label mapping
    unique_targets = sorted(meta["target"].unique())
    target_to_idx = {int(t): i for i, t in enumerate(unique_targets)}
    np.save(output_dir / "label_map.npy", target_to_idx)

    # Save class distribution info
    for split_name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        unique, counts = np.unique(labels[idx], return_counts=True)
        logger.info(f"  {split_name} class distribution: {dict(zip(unique, counts))}")

    logger.info("Preprocessing complete.")


try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except (ImportError, OSError):
    _HAS_TORCH = False
    _TorchDataset = object


class PLAsTiCCDataset(_TorchDataset):
    """PyTorch Dataset for PLAsTiCC light curves.

    Args:
        data_dir: Path to processed PLAsTiCC data (containing train.npz, val.npz, test.npz).
        split: One of 'train', 'val', 'test'.
        representation: 'timeseries' for raw LC, 'gaf' for Gramian Angular Field images.
        transform: Optional torchvision-style transform.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        representation: str = "timeseries",
        transform: Any | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        data = np.load(data_dir / f"{split}.npz")
        self.light_curves = data["light_curves"]  # (N, n_bands, max_len, 2)
        self.labels = data["labels"]  # (N,)
        self.representation = representation
        self.transform = transform

        if representation == "gaf":
            from src.data.augmentation import light_curve_to_gaf

            self._gaf_fn = light_curve_to_gaf

        logger.info(f"PLAsTiCCDataset [{split}]: {len(self)} samples, repr={representation}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        lc = self.light_curves[idx]  # (n_bands, max_len, 2)
        label = int(self.labels[idx])

        if self.representation == "timeseries":
            # Return flux values across all bands: (n_bands, max_len)
            x = torch.from_numpy(lc[:, :, 0]).float()
        elif self.representation == "gaf":
            # Stack GAF images per band: (n_bands, img_size, img_size)
            gaf_images = []
            for pb in range(lc.shape[0]):
                flux = lc[pb, :, 0]
                # Remove zero-padded tail
                nonzero = np.nonzero(flux)[0]
                if len(nonzero) > 2:
                    flux_trimmed = flux[: nonzero[-1] + 1]
                else:
                    flux_trimmed = flux[:10]  # fallback for very sparse LCs
                gaf = self._gaf_fn(flux_trimmed, image_size=224, method="gasf")
                gaf_images.append(gaf)
            x = torch.from_numpy(np.stack(gaf_images, axis=0)).float()
        else:
            raise ValueError(f"Unknown representation: {self.representation}")

        if self.transform is not None:
            x = self.transform(x)

        return x, label
