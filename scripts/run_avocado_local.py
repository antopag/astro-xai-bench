"""Run the Avocado (Boone 2019) benchmark pipeline locally.

Usage:
    # Single seed (for verification):
    python scripts/run_avocado_local.py --seed 42

    # All 5 seeds:
    python scripts/run_avocado_local.py

    # Custom n_jobs for feature extraction parallelism:
    python scripts/run_avocado_local.py --seed 42 --n-jobs 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 456, 789, 1024]


def main() -> None:
    parser = argparse.ArgumentParser(description="Avocado benchmark pipeline.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed (default: all 5 seeds).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for feature extraction (default: all cores).",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip XAI metric computation (classification only).",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip sanity checks.",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS

    from src.models.avocado.train import train_avocado_single_seed
    from src.models.avocado.xai import compute_avocado_xai
    from src.models.avocado.sanity import data_randomization_test

    results_dir = PROJECT_ROOT / "scripts" / "results"
    all_metrics = {}

    for seed in seeds:
        logger.info(f"\n{'='*60}\nAvocado seed {seed}\n{'='*60}")

        # 1. Train
        metrics = train_avocado_single_seed(seed, n_jobs=args.n_jobs)
        logger.info(
            f"Classification: acc={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_macro']:.4f}, auc={metrics.get('auc_ovr', 'N/A')}"
        )

        # 2. XAI
        if not args.skip_xai:
            xai_metrics = compute_avocado_xai(seed)
            metrics.update(xai_metrics)

        # 3. Sanity (seed 42 only)
        if not args.skip_sanity and seed == 42:
            rho = data_randomization_test(seed=42)
            metrics["data_randomization_rho"] = rho

        all_metrics[seed] = metrics

        # Save per-seed results
        seed_dir = results_dir / f"seed_{seed}"
        existing = {}
        results_path = seed_dir / "results.json"
        if results_path.is_file():
            with open(results_path) as f:
                existing = json.load(f)
        existing["Avocado"] = metrics
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"Saved Avocado results to {results_path}")

    # Summary
    if len(seeds) > 1:
        import numpy as np

        logger.info(f"\n{'='*60}\nAvocado summary ({len(seeds)} seeds)\n{'='*60}")
        for key in ["accuracy", "f1_macro", "auc_ovr", "faithfulness_mean",
                     "complexity_mean", "plausibility_mean"]:
            vals = [all_metrics[s].get(key) for s in seeds
                    if isinstance(all_metrics[s].get(key), (int, float))]
            if vals:
                logger.info(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")


if __name__ == "__main__":
    main()
