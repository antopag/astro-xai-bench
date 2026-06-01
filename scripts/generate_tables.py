"""Generate LaTeX tables from benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_benchmark_table(results_path: str | Path, output_path: str | Path) -> None:
    """Convert JSON results to LaTeX table for the paper."""
    raise NotImplementedError


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default=str(project_root / "paper" / "tables" / "benchmark.tex"))
    args = parser.parse_args()
    generate_benchmark_table(args.results, args.output)
