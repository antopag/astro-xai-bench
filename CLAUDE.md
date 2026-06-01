# astro-xai-bench – Claude Code Instructions

## Project
Benchmark + framework for astronomical transient classification with deep learning and Explainable AI (xAI).
Target journal: Astronomy & Computing or MNRAS.

## Tech Stack
- Python 3.11+
- PyTorch (models, training)
- timm (pretrained ViT)
- scikit-learn, xgboost (baselines)
- shap, captum (xAI)
- loguru (logging)
- matplotlib (publication figures)
- pyyaml (config)
- uv or pip for packaging

## Coding Standards
- **Typing**: mandatory on all function signatures
- **Docstrings**: Google-style on all public functions/classes
- **Config-driven**: all experiment hyperparameters in `configs/*.yaml`, never hardcoded
- **Reproducibility**: seed = 42, set everywhere (torch, numpy, random, CUDA)
- **Hardware target**: single consumer GPU (RTX 3060–4070, 8–12 GB VRAM)
- **Logging**: use `loguru.logger`, no print statements in library code

## Architecture Rules
- All models implement a common interface: `predict()`, `predict_proba()`, `get_embeddings()`
- Data loaders are PyTorch `Dataset`/`DataLoader`
- Experiments are launched via `scripts/run_benchmark.py` reading a YAML config
- Results saved as JSON + CSV for programmatic table/figure generation

## File Organization
- `src/` – installable package (import as `astro_xai_bench`)
- `configs/` – YAML experiment definitions
- `notebooks/` – exploration and paper figures only, no core logic
- `scripts/` – CLI entry points
- `tests/` – pytest
- `paper/` – LaTeX source (maintained locally; excluded from the repository via .gitignore)

## Data
- Only public datasets: PLAsTiCC, ZTF BTS, ELAsTiCC
- Raw data in `data/raw/`, processed in `data/processed/`
- Never commit data files to git (they go in .gitignore)
