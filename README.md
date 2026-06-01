# astro-xai-bench

Reproducible benchmark and open-source framework for astronomical transient classification with deep learning and Explainable AI (xAI).

## Overview

This project provides:
- **Multiple model architectures** for transient classification: Random Forest, XGBoost, LSTM, CNN (1D/2D), Vision Transformer (ViT)
- **Explainability methods**: Grad-CAM, SHAP, Attention Rollout with quantitative faithfulness/plausibility metrics
- **Benchmark suite**: standardized evaluation on PLAsTiCC, ZTF BTS, and ELAsTiCC datasets
- **Publication-ready outputs**: tables, figures, and LaTeX integration

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

All datasets are public:
- [PLAsTiCC](https://www.kaggle.com/c/PLAsTiCC-2018) – simulated LSST transients
- [ZTF BTS](https://sites.astro.caltech.edu/ztf/bts/) – real spectroscopically classified transients
- [ELAsTiCC](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/) – extended LSST simulations

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

## License

MIT
