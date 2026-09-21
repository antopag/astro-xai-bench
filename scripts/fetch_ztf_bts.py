"""Download ZTF BTS light curves from the public ALeRCE API.

Reads ``data/raw/ztf/bts_catalogue.csv`` (BTS explorer export) and stores one
JSON file per object in ``data/raw/ztf/lc/<ZTFID>.json`` with the raw ALeRCE
detections.  Restartable: objects with an existing file are skipped.

Usage:  python scripts/fetch_ztf_bts.py [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw" / "ztf"
LC_DIR = RAW / "lc"
API = "https://api.alerce.online/ztf/v1/objects/{oid}/detections"


def fetch_one(oid: str, session: requests.Session, retries: int = 8) -> tuple[str, int]:
    out = LC_DIR / f"{oid}.json"
    if out.exists():
        return oid, -1
    for attempt in range(retries):
        try:
            time.sleep(0.3)
            r = session.get(API.format(oid=oid), timeout=60)
            if r.status_code == 404:
                out.write_text("[]")
                return oid, 0
            r.raise_for_status()
            dets = r.json()
            out.write_text(json.dumps(dets))
            return oid, len(dets)
        except Exception as exc:  # noqa: BLE001
            wait = 5 * 2 ** attempt
            logger.warning(f"{oid}: {exc} (retry in {wait}s)")
            time.sleep(wait)
    return oid, -2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    LC_DIR.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(RAW / "bts_catalogue.csv")
    oids = cat["ZTFID"].astype(str).unique().tolist()
    todo = [o for o in oids if not (LC_DIR / f"{o}.json").exists()]
    logger.info(f"{len(oids)} objects in catalogue, {len(todo)} to fetch")

    session = requests.Session()
    session.headers["User-Agent"] = "astro-xai-bench/paper2 (antonio.pagliaro@inaf.it)"
    n_ok = n_empty = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, o, session) for o in todo]
        for i, f in enumerate(as_completed(futs), 1):
            _, n = f.result()
            if n == -2:
                n_fail += 1
            elif n == 0:
                n_empty += 1
            else:
                n_ok += 1
            if i % 200 == 0:
                logger.info(f"{i}/{len(todo)}  ok={n_ok} empty={n_empty} fail={n_fail}  "
                            f"{(time.time() - t0) / i:.2f}s/obj")
    logger.info(f"done: ok={n_ok} empty={n_empty} fail={n_fail}")


if __name__ == "__main__":
    main()


