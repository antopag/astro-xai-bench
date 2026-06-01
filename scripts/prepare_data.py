"""Python script for downloading and preprocessing PLAsTiCC (works on Windows too)."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess PLAsTiCC")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--raw-dir", type=str, default=str(project_root / "data" / "raw" / "plasticc"))
    parser.add_argument("--processed-dir", type=str, default=str(project_root / "data" / "processed" / "plasticc"))
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    if not args.skip_download:
        from src.data.plasticc import download_plasticc
        download_plasticc(raw_dir)

    from src.data.plasticc import preprocess_plasticc
    preprocess_plasticc(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        max_len=args.max_len,
        seed=args.seed,
    )

    # Quick validation
    import numpy as np
    for split in ["train", "val", "test"]:
        data = np.load(processed_dir / f"{split}.npz")
        logger.info(f"{split}: light_curves={data['light_curves'].shape}, labels={data['labels'].shape}")


if __name__ == "__main__":
    main()
