"""Run the ParSNIP (Boone 2021) benchmark pipeline locally.

Usage:
    # Single seed:
    python scripts/run_parsnip_local.py --seed 42

    # All 5 seeds:
    python scripts/run_parsnip_local.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 456, 789, 1024]


def main() -> None:
    parser = argparse.ArgumentParser(description="ParSNIP benchmark pipeline.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed (default: all 5).")
    parser.add_argument("--skip-xai", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device (default: auto).")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS

    from src.models.parsnip.train import train_parsnip_single_seed
    from src.models.parsnip.xai import compute_parsnip_xai
    from src.models.parsnip.sanity import data_randomization_test

    results_dir = PROJECT_ROOT / "scripts" / "results"
    all_metrics = {}

    for seed in seeds:
        logger.info(f"\n{'='*60}\nParSNIP seed {seed}\n{'='*60}")

        # 1. Train
        metrics = train_parsnip_single_seed(seed, device=args.device)
        logger.info(
            f"Classification: acc={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_macro']:.4f}"
        )

        # 2. XAI
        if not args.skip_xai:
            xai_metrics = compute_parsnip_xai(seed)
            metrics.update(xai_metrics)

        # 3. Sanity — data randomization only (seed 42)
        # No model randomization for tree-based classifiers (Table 6 convention)
        if not args.skip_sanity and seed == 42:
            rho = data_randomization_test(seed=42)
            metrics["data_randomization_rho"] = rho

        all_metrics[seed] = metrics

        # Save per-seed
        seed_dir = results_dir / f"seed_{seed}"
        existing = {}
        results_path = seed_dir / "results.json"
        if results_path.is_file():
            with open(results_path) as f:
                existing = json.load(f)
        existing["ParSNIP"] = metrics
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"Saved ParSNIP results to {results_path}")

    # Summary
    if len(seeds) > 1:
        import numpy as np
        logger.info(f"\n{'='*60}\nParSNIP summary ({len(seeds)} seeds)\n{'='*60}")
        for key in ["accuracy", "f1_macro", "auc_ovr", "faithfulness_mean",
                     "complexity_mean", "plausibility_mean"]:
            vals = [all_metrics[s].get(key) for s in seeds
                    if isinstance(all_metrics[s].get(key), (int, float))]
            if vals:
                logger.info(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")


if __name__ == "__main__":
    main()
