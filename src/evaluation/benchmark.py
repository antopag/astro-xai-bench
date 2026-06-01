"""Benchmark orchestrator: run all models, collect results, generate tables."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.data.plasticc import CLASS_MAP, PLAsTiCCDataset
from src.evaluation.metrics import compute_metrics, confusion_matrix_plot
from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper, extract_features
from src.training.trainer import Trainer


def _set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_benchmark(config_path: str | Path) -> dict[str, Any]:
    """Run the full benchmark pipeline from a YAML config.

    Args:
        config_path: Path to experiment YAML config.

    Returns:
        Results dict with per-model metrics.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    _set_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device} | Seed: {seed}")

    data_cfg = config["data"]
    model_cfgs = config["models"]
    output_cfg = config.get("output", {})

    processed_dir = Path(data_cfg["processed_dir"])
    results_dir = Path(output_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(output_cfg.get("figures_dir", "paper/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 0)

    # Load label map
    label_map = np.load(processed_dir / "label_map.npy", allow_pickle=True).item()
    idx_to_target = {v: k for k, v in label_map.items()}
    class_names = [CLASS_MAP.get(idx_to_target[i], str(i)) for i in range(len(label_map))]
    n_classes = len(label_map)

    all_results: dict[str, Any] = {}

    # ─── TABULAR MODELS ───
    if model_cfgs.get("random_forest", {}).get("enabled"):
        all_results["random_forest"] = _run_tabular(
            "random_forest", RFClassifier, model_cfgs["random_forest"],
            processed_dir, class_names, figures_dir, seed,
        )

    if model_cfgs.get("xgboost", {}).get("enabled"):
        all_results["xgboost"] = _run_tabular(
            "xgboost", XGBClassifierWrapper, model_cfgs["xgboost"],
            processed_dir, class_names, figures_dir, seed,
        )

    # ─── PYTORCH MODELS ───
    ts_models = {
        "lstm": ("src.models.lstm_baseline", "LSTMClassifier"),
        "cnn1d": ("src.models.cnn_baseline", "CNN1D"),
    }
    for name, (module_path, class_name) in ts_models.items():
        if model_cfgs.get(name, {}).get("enabled"):
            all_results[name] = _run_torch_model(
                name, module_path, class_name, model_cfgs[name], config,
                processed_dir, "timeseries", batch_size, num_workers,
                device, class_names, n_classes, figures_dir,
            )

    gaf_models = {
        "cnn2d": ("src.models.cnn_baseline", "CNN2D"),
        "vit": ("src.models.vit", "ViTClassifier"),
    }
    for name, (module_path, class_name) in gaf_models.items():
        if model_cfgs.get(name, {}).get("enabled"):
            all_results[name] = _run_torch_model(
                name, module_path, class_name, model_cfgs[name], config,
                processed_dir, "gaf", batch_size, num_workers,
                device, class_names, n_classes, figures_dir,
            )

    # ─── SAVE RESULTS ───
    results_path = results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Print summary table
    _print_summary(all_results)

    return all_results


def _run_tabular(
    name: str,
    model_cls: type,
    model_cfg: dict,
    processed_dir: Path,
    class_names: list[str],
    figures_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Train and evaluate a tabular model."""
    logger.info(f"{'='*60}")
    logger.info(f"Running: {name}")
    logger.info(f"{'='*60}")

    # Load data and extract features
    train_data = np.load(processed_dir / "train.npz")
    val_data = np.load(processed_dir / "val.npz")
    test_data = np.load(processed_dir / "test.npz")

    logger.info("Extracting features...")
    x_train = extract_features(train_data["light_curves"])
    x_val = extract_features(val_data["light_curves"])
    x_test = extract_features(test_data["light_curves"])
    y_train = train_data["labels"]
    y_test = test_data["labels"]

    # Combine train + val for tabular (no early stopping needed)
    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.concatenate([y_train, val_data["labels"]])

    cfg = {k: v for k, v in model_cfg.items() if k != "enabled"}
    cfg["random_state"] = seed
    model = model_cls(**cfg)

    t0 = time.time()
    model.fit(x_trainval, y_trainval)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    inference_time = time.time() - t0

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = inference_time

    confusion_matrix_plot(y_test, y_pred, class_names, save_path=str(figures_dir / f"cm_{name}.png"))

    return metrics


def _run_torch_model(
    name: str,
    module_path: str,
    class_name: str,
    model_cfg: dict,
    full_config: dict,
    processed_dir: Path,
    representation: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    class_names: list[str],
    n_classes: int,
    figures_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate a PyTorch model."""
    logger.info(f"{'='*60}")
    logger.info(f"Running: {name} (repr={representation})")
    logger.info(f"{'='*60}")

    import importlib
    mod = importlib.import_module(module_path)
    model_cls = getattr(mod, class_name)

    # Build data loaders
    train_ds = PLAsTiCCDataset(processed_dir, "train", representation)
    val_ds = PLAsTiCCDataset(processed_dir, "val", representation)
    test_ds = PLAsTiCCDataset(processed_dir, "test", representation)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    # Instantiate model
    n_bands = 6
    model_kwargs: dict[str, Any] = {"n_classes": n_classes, "n_bands": n_bands}

    # Pass model-specific config
    for k in ["hidden_size", "n_layers", "bidirectional", "base_filters", "dropout",
              "model_name", "pretrained"]:
        if k in model_cfg:
            model_kwargs[k] = model_cfg[k]

    model = model_cls(**model_kwargs)

    # Train
    trainer_config = dict(full_config)
    trainer_config["_model_config"] = model_cfg
    trainer = Trainer(model, trainer_config, device=device)

    t0 = time.time()
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0

    # Evaluate
    t0 = time.time()
    metrics = trainer.evaluate(test_loader)
    inference_time = time.time() - t0

    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = inference_time

    # Confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model.predict(bx)
            all_preds.append(preds)
            all_labels.append(by.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    confusion_matrix_plot(y_true, y_pred, class_names, save_path=str(figures_dir / f"cm_{name}.png"))

    # Save checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    trainer.save_checkpoint(ckpt_dir / f"{name}.pt")

    return metrics


def _print_summary(results: dict[str, Any]) -> None:
    """Print a summary table of all model results."""
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*80}")
    header = f"{'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'AUC':>10} {'Train(s)':>10}"
    logger.info(header)
    logger.info("-" * 80)
    for name, m in results.items():
        logger.info(
            f"{name:<20} {m.get('accuracy', 0):.4f}     "
            f"{m.get('f1_macro', 0):.4f}     "
            f"{m.get('f1_weighted', 0):.4f}       "
            f"{m.get('auc_ovr', float('nan')):.4f}     "
            f"{m.get('train_time_s', 0):.1f}"
        )
    logger.info(f"{'='*80}")
