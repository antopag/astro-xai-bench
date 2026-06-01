"""XAI evaluation for ParSNIP model (TreeSHAP on downstream GBT).

Computes faithfulness, complexity, and plausibility metrics using
TreeSHAP attributions on the 11-feature LightGBM classifier.

Plausibility requires projecting ParSNIP's 11-d latent-classifier
features onto our 60-d tabular expert mask. Two approaches:

Phase 3a (mandatory): Heuristic per-dimension mapping using Boone 2021's
  documented latent semantics.

Phase 3b (TODO): Jacobian-based propagation — ∂(decoded LC)/∂(latent)
  at each test sample, propagated through our time-series → 60-d adapter.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import shap
from loguru import logger

from src.xai.metrics import (
    explanation_complexity,
    faithfulness_correlation,
    plausibility_score,
)
from src.xai.shap_utils import normalize_shap_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------
# ParSNIP latent → our 60-d tabular feature mapping (Phase 3a)
# ---------------------------------------------------------------
# ParSNIP classifier features (11):
#   color, color_error, s1, s1_error, s2, s2_error, s3, s3_error,
#   luminosity, luminosity_error, reference_time_error
#
# Mapping rationale (from Boone 2021 latent semantics):
#   s1 → dominant shape mode (amplitude-like)
#   s2 → secondary mode (color evolution)
#   s3 → tertiary mode (timescale)
#   color → E(B-V) dust reddening
#   luminosity → absolute brightness

LATENT_TO_60D: dict[str, list[str]] = {
    # ParSNIP feature → list of 60-d feature names to distribute attribution to
    "s1": ["u_amplitude", "g_amplitude", "r_amplitude",
           "i_amplitude", "z_amplitude", "Y_amplitude"],
    "s2": ["g_mean", "r_mean", "i_mean"],
    "s3": ["g_slope", "r_slope", "i_slope"],
    "color": ["g_median", "r_median", "i_median", "z_median"],
    "luminosity": ["i_mean"],
    # Error features → unmapped (no uncertainty analogue in 60-d space)
    # reference_time_error → unmapped (phase uncertainty)
}


class ParsnipClassifierWrapper:
    """Wrapper to give ParSNIP's classifier ensemble a predict_proba interface.

    Uses the stored list of LightGBM classifiers from parsnip.Classifier.
    """

    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier
        self._classifiers = classifier.classifiers  # list of LightGBM models

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities by averaging fold classifiers."""
        probs = np.zeros((x.shape[0], self._classifiers[0].n_classes_))
        for clf in self._classifiers:
            raw = clf.predict_proba(x, raw_score=True,
                                    num_iteration=clf.best_iteration_)
            exp_scores = np.exp(raw)
            probs += exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs /= len(self._classifiers)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


def compute_parsnip_xai(
    seed: int,
    results_dir: Path | None = None,
    n_samples: int = 50,
) -> dict[str, float]:
    """Compute XAI metrics for a trained ParSNIP model.

    Args:
        seed: Random seed.
        results_dir: Path to results directory.
        n_samples: Number of test samples for XAI evaluation (default 50,
            matching the tabular/timeseries convention in run_final_table.py).

    Returns:
        Dict with faithfulness_mean/std, complexity_mean/std, plausibility_mean/std.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "scripts" / "results"

    seed_dir = results_dir / f"seed_{seed}"

    # Load classifier
    clf_path = seed_dir / "checkpoints" / f"parsnip_classifier_seed{seed}.pkl"
    with open(clf_path, "rb") as f:
        classifier = pickle.load(f)
    wrapper = ParsnipClassifierWrapper(classifier)

    # Load test features
    data = np.load(seed_dir / "parsnip_classifier_features.npz")
    test_features = data["features"]
    feature_names = list(data["feature_names"])

    # Select samples
    rng = np.random.default_rng(seed)
    n_test = len(test_features)
    sample_idx = rng.choice(n_test, size=min(n_samples, n_test), replace=False)

    # TreeSHAP — average across all 10 folds for consistency with
    # how train.py produces ensemble predictions
    logger.info(f"Computing TreeSHAP (seed={seed}, 10-fold average)…")
    shap_folds = []
    for clf in classifier.classifiers:
        explainer = shap.TreeExplainer(clf)
        shap_folds.append(normalize_shap_values(explainer.shap_values(test_features[sample_idx])))
    mean_shap = np.mean(shap_folds, axis=0)  # (n_samples, n_features)

    # 1. Faithfulness
    faithfulness_vals = []
    for i, idx in enumerate(sample_idx):
        x = test_features[idx]
        attr = mean_shap[i]
        f = faithfulness_correlation(wrapper, x, attr, seed=seed)
        faithfulness_vals.append(f)

    # 2. Complexity
    complexity_vals = [explanation_complexity(mean_shap[i]) for i in range(len(sample_idx))]

    # 3. Plausibility — project latent attributions to 60-d
    plausibility_vals = _compute_plausibility_parsnip(mean_shap, feature_names)

    metrics = {
        "faithfulness_mean": float(np.mean(faithfulness_vals)),
        "faithfulness_std": float(np.std(faithfulness_vals)),
        "complexity_mean": float(np.mean(complexity_vals)),
        "complexity_std": float(np.std(complexity_vals)),
        "plausibility_mean": float(np.mean(plausibility_vals)),
        "plausibility_std": float(np.std(plausibility_vals)),
    }

    logger.info(
        f"  ParSNIP XAI (seed={seed}): "
        f"faith={metrics['faithfulness_mean']:.3f}, "
        f"compl={metrics['complexity_mean']:.3f}, "
        f"plaus={metrics['plausibility_mean']:.3f}"
    )
    return metrics


def _compute_plausibility_parsnip(
    attributions: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """Project ParSNIP latent attributions onto 60-d expert mask.

    Each mapped latent dimension distributes its attribution equally
    across all target 60-d slots. E.g., s1 maps to 6 amplitude features,
    so each gets attribution(s1) / 6.

    Unmapped features (errors, reference_time_error) contribute zero.
    """
    from src.xai.metrics import build_expert_mask_tabular

    # Our 60-d feature names
    all_bands = ["u", "g", "r", "i", "z", "Y"]
    stat_names = ["mean", "std", "amplitude", "median", "n_obs",
                  "skewness", "kurtosis", "mean_snr", "slope", "frac_above_mean"]
    our_feature_names = [f"{b}_{s}" for b in all_bands for s in stat_names]
    our_name_to_idx = {name: i for i, name in enumerate(our_feature_names)}

    expert_mask = build_expert_mask_tabular(our_feature_names)

    plaus_vals = []
    for sample_idx in range(attributions.shape[0]):
        attr = attributions[sample_idx]
        projected = np.zeros(60)

        for j, feat_name in enumerate(feature_names):
            if feat_name in LATENT_TO_60D:
                target_features = LATENT_TO_60D[feat_name]
                # Distribute attribution equally
                per_target = attr[j] / len(target_features)
                for tf in target_features:
                    if tf in our_name_to_idx:
                        projected[our_name_to_idx[tf]] += per_target

        plaus = plausibility_score(projected, expert_mask)
        plaus_vals.append(plaus)

    n_mapped = sum(1 for f in feature_names if f in LATENT_TO_60D)
    logger.info(
        f"  Plausibility projection: {n_mapped}/{len(feature_names)} "
        f"ParSNIP features mapped to 60-d space"
    )

    return plaus_vals
