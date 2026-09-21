"""Preprocess ZTF BTS light curves into the benchmark's array format.

Input : data/raw/ztf/bts_catalogue.csv, data/raw/ztf/lc/<ZTFID>.json (ALeRCE)
Output: data/processed/ztf/{train,val,test}.npz with keys
        light_curves (N, 2, 256, 2)  normalised flux / flux_err, band order (g, r)
        times        (N, 2, 256)     MJD of each epoch (0 = padding)
        labels       (N,)            0-based class index (see label_map.npy)
        object_ids   (N,)            ZTFID strings
        redshift     (N,)            BTS spectroscopic redshift
        bts_extra    (N, 6)          src.xai.masks_ztf.BTS_EXTRA, from *raw* flux
        absmag       (N,)            absolute r peak magnitude (flat LCDM)
        label_map.npy                {class name: index}

Conventions mirror src/data/plasticc.py (max_len 256, zero padding, per-object
per-band z-normalisation, stratified 70/10/20 split, seed 42).

Usage:  python scripts/prepare_ztf.py [--min-det 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.plasticc import _normalize_light_curves  # noqa: E402
from src.xai.masks_ztf import absolute_peak_mag, extract_bts_extra  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw" / "ztf"
OUT = PROJECT_ROOT / "data" / "processed" / "ztf"
ZP = 27.5
FID_TO_BAND = {1: 0, 2: 1}  # g, r

#: BTS type -> benchmark class.  Types not listed are dropped.
TAXONOMY = {
    "SN Ia": ["SN Ia", "SN Ia-91T", "SN Ia-91bg", "SN Ia-pec", "SN Iax", "SN Ia-CSM", "SN Ia-SC"],
    "SN II": ["SN II", "SN IIP", "SN IIn", "SN II-pec"],
    "SN Ibc": ["SN Ib", "SN Ic", "SN Ic-BL", "SN IIb", "SN Ibn", "SN Ib/c", "SN Icn", "SN Ib-pec"],
    "SLSN": ["SLSN-I", "SLSN-II"],
    "TDE": ["TDE", "TDE-He", "TDE-H-He"],
}
TYPE_TO_CLASS = {t: c for c, ts in TAXONOMY.items() for t in ts}
CLASSES = list(TAXONOMY)


def load_object(oid: str, max_len: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (lc (2, max_len, 2) raw flux/err, times (2, max_len)) or None."""
    p = RAW / "lc" / f"{oid}.json"
    if not p.exists():
        return None
    dets = json.loads(p.read_text())
    lc = np.zeros((2, max_len, 2), dtype=np.float32)
    tt = np.zeros((2, max_len), dtype=np.float64)
    for fid, b in FID_TO_BAND.items():
        rows = [d for d in dets if d.get("fid") == fid and d.get("isdiffpos") in (1, "1", "t", True)
                and d.get("magpsf") is not None and d.get("sigmapsf") is not None]
        rows.sort(key=lambda d: d["mjd"])
        rows = rows[:max_len]
        if not rows:
            continue
        m = np.array([d["magpsf"] for d in rows], dtype=np.float64)
        s = np.array([d["sigmapsf"] for d in rows], dtype=np.float64)
        f = 10 ** (-0.4 * (m - ZP))
        lc[b, :len(rows), 0] = f
        lc[b, :len(rows), 1] = f * s * np.log(10) / 2.5
        tt[b, :len(rows)] = [d["mjd"] for d in rows]
    return lc, tt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--min-det", type=int, default=5, help="min positive detections in r")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(RAW / "bts_catalogue.csv")
    cat["cls"] = cat["type"].map(TYPE_TO_CLASS)
    n_drop_type = cat["cls"].isna().sum()
    cat = cat.dropna(subset=["cls"]).reset_index(drop=True)
    logger.info(f"{len(cat)} objects after taxonomy ({n_drop_type} dropped for type)")
    cat["redshift"] = pd.to_numeric(cat["redshift"], errors="coerce")
    n_drop_z = cat["redshift"].isna().sum()
    cat = cat.dropna(subset=["redshift"]).reset_index(drop=True)
    logger.info(f"{len(cat)} objects with numeric redshift ({n_drop_z} dropped)")

    lcs, tts, keep = [], [], []
    n_missing = n_short = 0
    for i, oid in enumerate(cat["ZTFID"].astype(str)):
        r = load_object(oid, args.max_len)
        if r is None:
            n_missing += 1
            continue
        lc, tt = r
        if (lc[1, :, 0] > 0).sum() < args.min_det:
            n_short += 1
            continue
        lcs.append(lc); tts.append(tt); keep.append(i)
    logger.info(f"kept {len(keep)}: {n_missing} without light curve, {n_short} with < {args.min_det} r detections")
    cat = cat.iloc[keep].reset_index(drop=True)
    raw_lc = np.stack(lcs); times = np.stack(tts)

    # Features that need physical flux: computed before normalisation.
    bts_extra = extract_bts_extra(raw_lc, times, zeropoint=ZP)
    absmag = absolute_peak_mag(bts_extra[:, 4], cat["redshift"].values.astype(float))

    light_curves = _normalize_light_curves(raw_lc)
    labels = np.array([CLASSES.index(c) for c in cat["cls"]], dtype=np.int64)
    object_ids = cat["ZTFID"].astype(str).values
    redshift = cat["redshift"].values.astype(np.float32)

    idx_trainval, idx_test = train_test_split(np.arange(len(labels)), test_size=0.2,
                                              stratify=labels, random_state=args.seed)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.1 / 0.8,
                                          stratify=labels[idx_trainval], random_state=args.seed)
    for name, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        np.savez_compressed(OUT / f"{name}.npz", light_curves=light_curves[idx], times=times[idx],
                            labels=labels[idx], object_ids=object_ids[idx], redshift=redshift[idx],
                            bts_extra=bts_extra[idx], absmag=absmag[idx])
        u, c = np.unique(labels[idx], return_counts=True)
        logger.info(f"{name}: {len(idx)}  " + ", ".join(f"{CLASSES[a]}={b}" for a, b in zip(u, c)))
    np.save(OUT / "label_map.npy", {c: i for i, c in enumerate(CLASSES)})
    logger.info("done")


if __name__ == "__main__":
    main()
