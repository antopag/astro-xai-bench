"""CLI entry point for running the full benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run astro-xai-bench benchmark")
    parser.add_argument("--config", type=str,
                        default=str(project_root / "configs" / "default.yaml"),
                        help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    logger.info(f"Loading config from {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.evaluation.benchmark import run_benchmark

    results = run_benchmark(config_path)
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
