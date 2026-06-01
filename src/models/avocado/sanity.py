"""Sanity checks for Avocado model (data randomization only).

Per Section 4.3 of the paper: tree-based models do not have a meaningful
model-randomisation analogue, so only data randomisation is performed.

Data randomization: retrain LightGBM on label-permuted training features,
compute TreeSHAP, and measure Spearman correlation with the
original-label attributions. Low correlation = attributions
depend on the supervision signal (pass).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from loguru import logger
from scipy.stats import spearmanr

from src.xai.shap_utils import normalize_shap_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def data_randomization_test(
    seed: int = 42,
    results_dir: Path | None = None,
    n_samples: int = 50,
    forced_epochs: int = 500,
) -> float:
    """Data randomization sanity check for Avocado.

    Protocol (matching RF/XGBoost in §4.3):
      1. Load original trained LightGBM booster.
      2. Load training-set features and labels.
      3. Permute training labels.
      4. Retrain LightGBM on (train_features, permuted_labels) for 500 rounds
         without early stopping.
      5. Compute TreeSHAP on test samples for both original and permuted models.
      6. Report Spearman rank correlation. Low ρ = pass.

    Args:
        seed: Random seed.
        results_dir: Path to results directory.
        n_samples: Number of test samples for attribution comparison.
        forced_epochs: Number of boosting rounds for random-label model.

    Returns:
        Spearman rho between original and random-label attributions.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "scripts" / "results"

    seed_dir = results_dir / f"seed_{seed}"

    # Load original model
    model_path = seed_dir / "checkpoints" / "avocado_lgb.txt"
    original_booster = lgb.Booster(model_file=str(model_path))

    # Load test features (for SHAP evaluation only)
    pred_data = np.load(seed_dir / "avocado_predictions.npz")
    test_features = pred_data["test_features"]

    # Load training features and labels (for retraining with permuted labels)
    cache_dir = seed_dir / "avocado_cache"

    feature_files = sorted(cache_dir.glob(f"features_train_aug_seed{seed}*.pkl"))
    if not feature_files:
        raise FileNotFoundError(
            f"No training feature cache found in {cache_dir}. Run training first."
        )
    train_features_df = pd.read_pickle(feature_files[0])
    logger.info(f"Loaded training features from {feature_files[0].name}: {train_features_df.shape}")

    label_files = sorted(cache_dir.glob(f"labels_train_aug_seed{seed}*.npy"))
    if not label_files:
        raise FileNotFoundError(
            f"No training labels cache found in {cache_dir}. Run training first."
        )
    train_labels = np.load(label_files[0])
    unique_targets = sorted(set(train_labels))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    y_train = np.array([target_to_idx[t] for t in train_labels])
    n_classes = len(unique_targets)

    # Permute training labels
    rng = np.random.default_rng(seed)
    y_train_random = rng.permutation(y_train)

    # Retrain on permuted labels — no balanced weights (protocol consistency)
    logger.info("Training random-label LightGBM for data randomization test…")

    dtrain = lgb.Dataset(
        train_features_df.values,
        label=y_train_random,
        feature_name=list(train_features_df.columns),
        free_raw_data=False,
    )

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbose": -1,
        "n_jobs": -1,
    }

    random_booster = lgb.train(params, dtrain, num_boost_round=forced_epochs)

    # Compute TreeSHAP for both models on same test samples
    sample_idx = rng.choice(
        len(test_features), size=min(n_samples, len(test_features)), replace=False
    )
    x_eval = test_features[sample_idx]

    explainer_orig = shap.TreeExplainer(original_booster)
    explainer_rand = shap.TreeExplainer(random_booster)

    attr_orig = normalize_shap_values(explainer_orig.shap_values(x_eval))
    attr_rand = normalize_shap_values(explainer_rand.shap_values(x_eval))

    # Spearman rank correlation
    rho, _ = spearmanr(attr_orig.flatten(), attr_rand.flatten())

    logger.info(f"Avocado data randomization: Spearman rho = {rho:.3f}")

    # Save result — both signed and absolute (mirrors Table A.3)
    result = {
        "model": "Avocado",
        "data_randomization_rho_signed": float(rho),
        "data_randomization_rho_abs": float(abs(rho)),
    }
    with open(seed_dir / "avocado_sanity.json", "w") as f:
        json.dump(result, f, indent=2)

    return float(rho)


if __name__ == "__main__":
    rho = data_randomization_test(seed=42)
    print(f"Data randomization rho: {rho:.4f}")
