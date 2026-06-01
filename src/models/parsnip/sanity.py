"""Sanity checks for ParSNIP model (data randomization only).

Following the convention applied to other tree-based classifiers in this
benchmark (RF, XGBoost), we do not report a model-randomisation result for
ParSNIP. TreeSHAP attributions on a gradient-boosted classifier have no
meaningful model-randomisation analogue — the "---" convention in Table 6
applies here as well.

Data randomization: Retrain the downstream GBT on label-permuted *training*
features (VAE stays fixed), compute TreeSHAP, compare with original-label
attributions. Single seed 42.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap
from loguru import logger
from scipy.stats import spearmanr

from src.xai.shap_utils import normalize_shap_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def data_randomization_test(
    seed: int = 42,
    results_dir: Path | None = None,
    n_samples: int = 50,
) -> float:
    """Data randomization: retrain GBT on label-permuted training features.

    Protocol (matching RF/XGBoost in §4.3):
      1. Load the original GBT trained on real labels.
      2. Load training-set ParSNIP latent features and training labels.
      3. Permute training labels.
      4. Retrain GBT on (train_features, permuted_labels) for 500 rounds
         without early stopping.
      5. Compute TreeSHAP on test samples for both original and permuted models.
      6. Report Spearman rank correlation. Low ρ = pass.

    The VAE encoder is fixed throughout — only the supervised classifier changes.

    Args:
        seed: Random seed.
        results_dir: Path to results directory.
        n_samples: Number of test samples for attribution comparison.

    Returns:
        Spearman ρ between original and random-label attributions.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "scripts" / "results"

    seed_dir = results_dir / f"seed_{seed}"

    # Load original classifier (10-fold ensemble)
    clf_path = seed_dir / "checkpoints" / f"parsnip_classifier_seed{seed}.pkl"
    with open(clf_path, "rb") as f:
        classifier = pickle.load(f)

    # Load test features for SHAP evaluation
    test_data = np.load(seed_dir / "parsnip_classifier_features.npz")
    test_features = test_data["features"]

    # Load training features and labels for retraining
    train_data = np.load(seed_dir / "parsnip_train_features.npz")
    train_features = train_data["features"]
    y_train = train_data["labels"]

    n_classes = len(np.unique(y_train))

    # Select test samples for SHAP
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(
        len(test_features), size=min(n_samples, len(test_features)), replace=False
    )
    x_eval = test_features[sample_idx]

    # Original attributions — average SHAP across all 10 folds for consistency
    # with how train.py produces ensemble predictions
    logger.info("Computing original TreeSHAP (10-fold average)…")
    shap_orig_folds = []
    for clf in classifier.classifiers:
        explainer = shap.TreeExplainer(clf)
        shap_orig_folds.append(normalize_shap_values(explainer.shap_values(x_eval)))
    attr_orig = np.mean(shap_orig_folds, axis=0)  # (n_samples, n_features)

    # Retrain on permuted training labels
    logger.info("Retraining GBT on permuted training labels…")
    y_permuted = rng.permutation(y_train)

    dtrain = lgb.Dataset(
        train_features,
        label=y_permuted,
        free_raw_data=False,
    )
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
    }
    # 500 rounds without early stopping — forced training consistent with
    # the CNN1D data-randomisation protocol (§4.3)
    random_booster = lgb.train(params, dtrain, num_boost_round=500)

    # Randomized attributions
    explainer_rand = shap.TreeExplainer(random_booster)
    attr_rand = normalize_shap_values(explainer_rand.shap_values(x_eval))

    # Spearman rank correlation
    rho, _ = spearmanr(attr_orig.flatten(), attr_rand.flatten())

    logger.info(f"ParSNIP data randomization: Spearman ρ = {rho:.3f}")

    # Save result — model_randomization_rho is null (tree-based, no analogue)
    # Both signed and absolute rho saved (mirrors Table A.3)
    result = {
        "model": "ParSNIP",
        "model_randomization_rho": None,
        "data_randomization_rho_signed": float(rho),
        "data_randomization_rho_abs": float(abs(rho)),
    }
    with open(seed_dir / "parsnip_sanity.json", "w") as f:
        json.dump(result, f, indent=2)

    return float(rho)


if __name__ == "__main__":
    rho = data_randomization_test(seed=42)
    print(f"Data randomization ρ: {rho:.4f}")
