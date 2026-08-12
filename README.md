# astro-xai-bench

Reproducible benchmark and open-source framework for astronomical transient classification with deep learning and Explainable AI (xAI).

## Overview

This project provides:
- **Multiple model architectures** for transient classification: Random Forest, XGBoost, LSTM, CNN (1D/2D), Vision Transformer (ViT)
- **Explainability methods**: Grad-CAM, SHAP, Attention Rollout with quantitative faithfulness/plausibility metrics
- **Benchmark suite**: standardized evaluation, instantiated on the PLAsTiCC training set
- **Publication-ready outputs**: tables, figures, and LaTeX integration

The five evaluation axes are dataset-agnostic, but this release instantiates them on
PLAsTiCC only. Porting the benchmark to another survey requires re-deriving the tabular
feature basis, the expert mask, the cross-representation projection and the class
taxonomy; see Sect. 7.2 of the paper. `src/data/ztf.py` is a loader stub for future work
and ELAsTiCC is not implemented.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Download data (requires Kaggle API key)
bash scripts/download_data.sh

# Run full benchmark
python scripts/run_benchmark.py --config configs/default.yaml
```

## Datasets

The benchmark runs on [PLAsTiCC](https://www.kaggle.com/c/PLAsTiCC-2018) (simulated LSST
transients), using the 7,848-object unblinded training set. It is public, as are the two
surveys that are the natural targets for a future extension:
[ZTF BTS](https://sites.astro.caltech.edu/ztf/bts/) and
[ELAsTiCC](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/). Neither is used
in this release.

## Project Structure

```
configs/          # YAML experiment configurations
src/data/         # data loading and preprocessing
src/models/       # model architectures
src/xai/          # explainability methods and metrics
src/training/     # training loop and callbacks
src/evaluation/   # evaluation metrics and benchmark orchestration
notebooks/        # exploratory analysis and paper figures
scripts/          # CLI entry points
tests/            # unit and smoke tests
```

The LaTeX manuscript is maintained separately and is not part of this repository.

## Reproducing the paper

`scripts/recompute_plausibility.py` is the reproducibility entry point for the
plausibility results (Table 5 and Table A.2): it recomputes them from the cached
attributions and first verifies that it reproduces the published numbers.

```bash
python scripts/recompute_plausibility.py 42   # single-seed smoke test
python scripts/recompute_plausibility.py      # all five seeds
```

## Citation

If you use this benchmark, please cite the paper and the archived software release:

> Pagliaro, A. 2026, *A Quantitative Benchmark for Explainable AI in Astronomical
> Transient Classification*, Astronomy & Astrophysics

Machine-readable metadata is in [CITATION.cff](CITATION.cff). The versioned archive is on
Zenodo (DOI to be added on release).

## License

MIT — see [LICENSE](LICENSE).
