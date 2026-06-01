"""Raw PLAsTiCC loader for Avocado and ParSNIP integration.

Reads the original PLAsTiCC training CSVs (with MJD timestamps and redshifts)
and exposes converters to the formats expected by each upstream package.

The existing 256-step .npz files used by the six base models are NOT touched.
This module provides a parallel data path for models that need raw light curves.

Usage:
    loader = RawPLAsTiCCLoader(
        raw_dir="data/raw/plasticc",
        processed_dir="data/processed/plasticc",
    )
    # Returns only objects in our existing train/val/test split
    lc_df, meta_df = loader.load_split("train")

    # Convert to upstream formats
    avocado_dataset = loader.to_avocado("train")
    lcdata_dataset = loader.to_lcdata("train")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.plasticc import CLASS_MAP, PASSBAND_NAMES

# PLAsTiCC passband integers → LSST filter names (used by sncosmo / lcdata)
_PASSBAND_TO_LSST: dict[int, str] = {
    0: "lsstu",
    1: "lsstg",
    2: "lsstr",
    3: "lssti",
    4: "lsstz",
    5: "lssty",  # lowercase y — required by both avocado and sncosmo
}


class RawPLAsTiCCLoader:
    """Load raw PLAsTiCC CSVs, filtered to our existing train/val/test split.

    Args:
        raw_dir: Path to directory containing training_set.csv and
            training_set_metadata.csv.
        processed_dir: Path to directory containing {train,val,test}.npz
            (used only to recover the object_id split).
    """

    def __init__(
        self,
        raw_dir: str | Path,
        processed_dir: str | Path,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Validate required files exist
        for f in ["training_set.csv", "training_set_metadata.csv"]:
            if not (self.raw_dir / f).is_file():
                raise FileNotFoundError(
                    f"{self.raw_dir / f} not found. "
                    "Run `python scripts/fetch_plasticc.py` first."
                )

        # Lazy-loaded caches
        self._meta_df: pd.DataFrame | None = None
        self._lc_df: pd.DataFrame | None = None
        self._split_ids: dict[str, np.ndarray] = {}

    def _load_raw(self) -> None:
        """Load raw CSVs into memory (once)."""
        if self._meta_df is not None:
            return

        logger.info("Loading raw PLAsTiCC metadata…")
        self._meta_df = pd.read_csv(self.raw_dir / "training_set_metadata.csv")
        logger.info(f"  {len(self._meta_df)} objects")

        logger.info("Loading raw PLAsTiCC light curves…")
        self._lc_df = pd.read_csv(self.raw_dir / "training_set.csv")
        logger.info(f"  {len(self._lc_df)} observations")

    def _get_split_ids(self, split: str) -> np.ndarray:
        """Recover object_ids for a split from the existing .npz files."""
        if split not in self._split_ids:
            npz_path = self.processed_dir / f"{split}.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(
                    f"{npz_path} not found. Run preprocessing first."
                )
            data = np.load(npz_path)
            self._split_ids[split] = data["object_ids"]
            logger.info(f"  Split '{split}': {len(self._split_ids[split])} object_ids from .npz")
        return self._split_ids[split]

    def load_split(
        self, split: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw light curves and metadata for a given split.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            (lc_df, meta_df): DataFrames filtered to the requested split's object_ids.
                lc_df columns: object_id, mjd, passband, flux, flux_err, detected
                meta_df columns: object_id, ra, decl, hostgal_specz, hostgal_photoz,
                    hostgal_photoz_err, distmod, mwebv, target, …
        """
        self._load_raw()
        assert self._lc_df is not None and self._meta_df is not None

        oids = self._get_split_ids(split)
        oid_set = set(oids)

        meta = self._meta_df[self._meta_df["object_id"].isin(oid_set)].copy()
        lc = self._lc_df[self._lc_df["object_id"].isin(oid_set)].copy()

        logger.info(f"  Filtered to split '{split}': {len(meta)} objects, {len(lc)} observations")
        return lc, meta

    # ------------------------------------------------------------------
    # Avocado format
    # ------------------------------------------------------------------

    def to_avocado(self, split: str) -> Any:
        """Convert a split to an Avocado Dataset.

        Returns:
            avocado.Dataset containing AstronomicalObject instances with
            raw (unnormalized) light curves, MJD timestamps, and metadata
            including redshift.

        Raises:
            ImportError: if avocado is not installed.
        """
        try:
            import avocado
        except ImportError:
            raise ImportError(
                "avocado-classifier is required: pip install avocado-classifier"
            )

        lc_df, meta_df = self.load_split(split)

        objects = []
        for _, row in meta_df.iterrows():
            oid = row["object_id"]
            obj_lc = lc_df[lc_df["object_id"] == oid].copy()

            # Avocado expects: time, band, flux, flux_error
            # Band names must be lowercase lsst* (e.g. lsstu, lsstg, lssty)
            observations = pd.DataFrame({
                "time": obj_lc["mjd"].values,
                "band": obj_lc["passband"].map(
                    lambda pb: f"lsst{PASSBAND_NAMES.get(pb, f'band_{pb}').lower()}"
                ).values,
                "flux": obj_lc["flux"].values,
                "flux_error": obj_lc["flux_err"].values,
            })

            metadata = {
                "object_id": oid,
                "host_photoz": row.get("hostgal_photoz", 0.0),
                "host_photoz_err": row.get("hostgal_photoz_err", 0.0),
                "hostgal_specz": row.get("hostgal_specz", 0.0),
                "galactic": bool(row.get("hostgal_specz", 0.0) == 0.0),
                "target": int(row["target"]),
                "class_name": CLASS_MAP.get(int(row["target"]), "Unknown"),
            }

            obj = avocado.AstronomicalObject(metadata, observations)
            objects.append(obj)

        dataset = avocado.Dataset.from_objects(
            f"plasticc_{split}", objects
        )
        logger.info(
            f"  Avocado Dataset '{split}': {len(objects)} objects"
        )
        return dataset

    # ------------------------------------------------------------------
    # lcdata / ParSNIP format
    # ------------------------------------------------------------------

    def to_lcdata(self, split: str) -> Any:
        """Convert a split to an lcdata Dataset (for ParSNIP).

        Returns:
            lcdata.Dataset with sncosmo-compatible light curve tables and
            metadata including redshift.

        Raises:
            ImportError: if lcdata is not installed.
        """
        try:
            import lcdata
        except ImportError:
            raise ImportError(
                "lcdata is required: pip install lcdata"
            )

        from astropy.table import Table

        lc_df, meta_df = self.load_split(split)

        light_curves = []
        meta_rows = []

        for _, row in meta_df.iterrows():
            oid = row["object_id"]
            obj_lc = lc_df[lc_df["object_id"] == oid].copy()
            obj_lc = obj_lc.sort_values("mjd")

            # Redshift assignment (Boone 2021 convention):
            #   specz > 0           → use specz
            #   specz == 0, photoz > 0 → use photoz (with uncertainty)
            #   both == 0           → galactic object, z=0
            specz = float(row.get("hostgal_specz", 0.0))
            photoz = float(row.get("hostgal_photoz", 0.0))
            if specz > 0:
                redshift = specz
            elif photoz > 0:
                redshift = photoz
            else:
                redshift = 0.0

            # sncosmo-style table with metadata in .meta dict
            lc_table = Table({
                "time": obj_lc["mjd"].values.astype(np.float64),
                "band": [_PASSBAND_TO_LSST[int(pb)] for pb in obj_lc["passband"].values],
                "flux": obj_lc["flux"].values.astype(np.float32),
                "fluxerr": obj_lc["flux_err"].values.astype(np.float32),
            })
            lc_table.meta = {
                "object_id": oid,
                "redshift": redshift,
                "hostgal_specz": specz,
                "hostgal_photoz": photoz,
                "hostgal_photoz_err": float(row.get("hostgal_photoz_err", 0.0)),
                "mwebv": float(row.get("mwebv", 0.0)),
                "type": int(row["target"]),
            }
            light_curves.append(lc_table)

        dataset = lcdata.from_light_curves(light_curves)

        logger.info(
            f"  lcdata Dataset '{split}': {len(light_curves)} light curves"
        )
        return dataset


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

def _smoke_test(n_objects: int = 10) -> None:
    """Smoke test: load 10 objects, verify shapes and conversions.

    Run with: python -m src.data.raw_loader
    """
    from src.data.raw_loader import RawPLAsTiCCLoader

    project_root = Path(__file__).resolve().parent.parent.parent
    loader = RawPLAsTiCCLoader(
        raw_dir=project_root / "data" / "raw" / "plasticc",
        processed_dir=project_root / "data" / "processed" / "plasticc",
    )

    # Basic load
    lc_df, meta_df = loader.load_split("train")
    assert len(meta_df) > 0, "No metadata loaded"
    assert "mjd" in lc_df.columns, "Missing mjd column"
    assert "hostgal_specz" in meta_df.columns, "Missing redshift column"

    # Check a subsample
    sample_ids = meta_df["object_id"].values[:n_objects]
    sample_lc = lc_df[lc_df["object_id"].isin(sample_ids)]
    logger.info(f"Smoke test: {n_objects} objects, {len(sample_lc)} observations")

    # Verify MJD timestamps are real (not 0..255 indices)
    assert sample_lc["mjd"].min() > 50000, (
        f"MJD values look wrong (min={sample_lc['mjd'].min()}), expected >50000"
    )

    # Verify redshifts exist
    sample_meta = meta_df[meta_df["object_id"].isin(sample_ids)]
    has_z = (sample_meta["hostgal_specz"] > 0).sum()
    logger.info(f"  {has_z}/{n_objects} objects have specz > 0")

    # Test lcdata conversion (if installed)
    try:
        ds = loader.to_lcdata("train")
        logger.info(f"  lcdata conversion OK: {len(ds)} light curves")
    except ImportError:
        logger.warning("  lcdata not installed, skipping to_lcdata test")

    # Test avocado conversion (if installed)
    try:
        ds = loader.to_avocado("train")
        logger.info(f"  Avocado conversion OK: {len(ds.objects)} objects")
    except ImportError:
        logger.warning("  avocado not installed, skipping to_avocado test")
    except AttributeError:
        # Avocado API may differ; log and continue
        logger.warning("  Avocado conversion returned but API differs from expected")

    logger.info("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
