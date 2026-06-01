"""ParSNIP (Boone 2021) training pipeline for astro-xai-bench.

Pipeline:
  1. Load raw PLAsTiCC light curves → lcdata.Dataset (with redshifts)
  2. Train ParSNIP VAE from scratch per seed (~1.5-2h on RTX 4090)
  3. Extract latent predictions (s1, s2, s3, color, luminosity, etc.)
  4. Train downstream LightGBM classifier on latent features (14-class)
  5. Save VAE checkpoint + GBT model + predictions

The VAE is class-agnostic (self-supervised reconstruction loss).
Only the downstream classifier uses labels.

Redshift convention (Boone 2021):
  - hostgal_specz > 0           → use specz
  - hostgal_specz == 0, photoz > 0 → use photoz
  - both == 0                   → galactic object, z=0
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from src.data.raw_loader import RawPLAsTiCCLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SEEDS = [42, 123, 456, 789, 1024]

# ParSNIP default LSST bands
PARSNIP_BANDS = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _prepare_lcdata_dataset(
    loader: RawPLAsTiCCLoader,
    split: str,
    cache_dir: Path,
) -> Any:
    """Prepare an lcdata.Dataset for ParSNIP, with caching.

    Args:
        loader: RawPLAsTiCCLoader instance.
        split: One of 'train', 'val', 'test'.
        cache_dir: Directory for cached HDF5 files.

    Returns:
        lcdata.Dataset ready for ParSNIP.
    """
    import lcdata

    cache_path = cache_dir / f"lcdata_{split}.h5"

    if cache_path.is_file():
        logger.info(f"Loading cached lcdata dataset from {cache_path}")
        dataset = lcdata.read_hdf5(str(cache_path))
        logger.info(f"  Loaded: {len(dataset)} light curves")
        return dataset

    dataset = loader.to_lcdata(split)

    # Cache to HDF5
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.write_hdf5(str(cache_path))
    logger.info(f"Cached lcdata dataset to {cache_path}")

    return dataset


def train_parsnip_single_seed(
    seed: int,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    max_epochs: int = 1000,
    device: str | None = None,
) -> dict[str, Any]:
    """Train ParSNIP VAE + downstream classifier for a single seed.

    Args:
        seed: Random seed.
        raw_dir: Path to raw PLAsTiCC CSVs.
        processed_dir: Path to processed .npz files (for object_id recovery).
        results_dir: Path to results directory.
        max_epochs: Maximum VAE training epochs (default 1000, early-stops via LR schedule).
        device: PyTorch device ('cuda', 'cpu', or None for auto-detect).

    Returns:
        Dict with classification metrics.
    """
    import parsnip
    from src.evaluation.metrics import compute_metrics

    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "plasticc"
    if processed_dir is None:
        processed_dir = PROJECT_ROOT / "data" / "processed" / "plasticc"
    if results_dir is None:
        results_dir = PROJECT_ROOT / "scripts" / "results"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_dir = results_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = seed_dir / "parsnip_cache"
    checkpoint_dir = seed_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)

    # 1. Prepare data
    loader = RawPLAsTiCCLoader(raw_dir=raw_dir, processed_dir=processed_dir)
    train_dataset = _prepare_lcdata_dataset(loader, "train", cache_dir)
    test_dataset = _prepare_lcdata_dataset(loader, "test", cache_dir)

    # 2. Train VAE
    vae_path = checkpoint_dir / f"parsnip_vae_seed{seed}.pt"

    if vae_path.is_file():
        logger.info(f"Loading existing VAE checkpoint from {vae_path}")
        model = parsnip.load_model(str(vae_path))
    else:
        logger.info(f"Training ParSNIP VAE (seed={seed}, device={device})…")

        # Parse training dataset for ParSNIP format
        train_parsed = parsnip.parse_dataset(train_dataset, kind="plasticc")

        model = parsnip.ParsnipModel(
            str(vae_path),
            bands=PARSNIP_BANDS,
            device=device,
        )

        model.fit(
            train_parsed,
            max_epochs=max_epochs,
            augment=True,
        )
        logger.info(f"VAE training complete. Saved to {vae_path}")

    # 3. Extract latent predictions
    logger.info("Extracting latent predictions…")
    train_parsed = parsnip.parse_dataset(train_dataset, kind="plasticc")
    test_parsed = parsnip.parse_dataset(test_dataset, kind="plasticc")

    train_predictions = model.predict_dataset(train_parsed)
    test_predictions = model.predict_dataset(test_parsed)

    logger.info(f"  Train predictions: {len(train_predictions)} objects")
    logger.info(f"  Test predictions: {len(test_predictions)} objects")

    # 4. Train downstream LightGBM classifier (14-class)
    # Extract labels from metadata
    train_labels = np.array(train_dataset.meta["type"])
    test_labels = np.array(test_dataset.meta["type"])

    # Map to 0-based indices
    unique_targets = sorted(set(train_labels) | set(test_labels))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    y_train = np.array([target_to_idx[t] for t in train_labels])
    y_test = np.array([target_to_idx[t] for t in test_labels])

    logger.info(f"Training downstream classifier (14-class, seed={seed})…")
    classifier = parsnip.Classifier()
    classifier.train(
        train_predictions,
        labels=y_train,
        num_folds=10,
        reweight=True,
    )

    # 5. Classify test set
    test_probs = classifier.classify(test_predictions)
    # test_probs is a DataFrame with class columns
    y_proba = test_probs.values if hasattr(test_probs, "values") else np.array(test_probs)
    y_pred = np.argmax(y_proba, axis=1)

    # 6. Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info(
        f"  Accuracy: {metrics['accuracy']:.4f}, "
        f"F1 macro: {metrics['f1_macro']:.4f}, "
        f"AUC: {metrics.get('auc_ovr', 'N/A')}"
    )

    # 7. Save predictions and classifier
    np.savez(
        seed_dir / "parsnip_predictions.npz",
        y_pred=y_pred,
        y_proba=y_proba,
        y_test=y_test,
    )

    # Save the classifier features for XAI and sanity checks
    classifier_features = classifier.keys
    train_features = np.column_stack([
        train_predictions[k] for k in classifier_features
    ])
    test_features = np.column_stack([
        test_predictions[k] for k in classifier_features
    ])
    np.savez(
        seed_dir / "parsnip_classifier_features.npz",
        features=test_features,
        feature_names=np.array(classifier_features),
    )
    np.savez(
        seed_dir / "parsnip_train_features.npz",
        features=train_features,
        labels=y_train,
        feature_names=np.array(classifier_features),
    )

    # Save classifier (pickle the LightGBM models)
    import pickle
    with open(checkpoint_dir / f"parsnip_classifier_seed{seed}.pkl", "wb") as f:
        pickle.dump(classifier, f)

    return metrics


def train_all_seeds(
    seeds: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Train ParSNIP across all seeds."""
    if seeds is None:
        seeds = SEEDS

    all_results = {}
    for seed in seeds:
        logger.info(f"\n{'='*60}\nParSNIP seed {seed}\n{'='*60}")
        metrics = train_parsnip_single_seed(seed, **kwargs)
        all_results[seed] = metrics

    # Aggregate
    if len(seeds) > 1:
        metric_keys = list(all_results[seeds[0]].keys())
        logger.info(f"\nParSNIP aggregated ({len(seeds)} seeds):")
        for k in ["accuracy", "f1_macro", "auc_ovr"]:
            vals = [all_results[s].get(k) for s in seeds
                    if isinstance(all_results[s].get(k), (int, float))]
            if vals:
                logger.info(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")

    return all_results


if __name__ == "__main__":
    train_all_seeds()
