"""XAI evaluation for Avocado model (TreeSHAP on LightGBM).

Computes faithfulness, complexity, and plausibility metrics using
TreeSHAP attributions on the trained LightGBM classifier.

Plausibility requires projecting Avocado's feature space onto our
60-d tabular expert mask. See FEATURE_MAPPING for the mapping.
"""

from __future__ import annotations

import json
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
# Avocado feature → our 60-d tabular feature mapping
# ---------------------------------------------------------------
# Our 60-d features: 10 stats × 6 bands (u,g,r,i,z,Y)
# Stats: mean, std, amplitude, median, n_obs, skewness, kurtosis, mean_snr, slope, frac_above_mean
#
# Avocado features (~20-30 selected features from PlasticcFeaturizer.select_features):
#   host_photoz, host_photoz_error, length_scale, max_mag, pos_flux_ratio,
#   max_flux_ratio_red, max_flux_ratio_blue, min_flux_ratio_red, min_flux_ratio_blue,
#   max_dt_*, frac_positive_*, time_fwd_max_*, time_bwd_max_*, ...
#
# The mapping is partial by design — Avocado's GP-derived features do not
# have a 1:1 correspondence with our hand-crafted per-band statistics.

FEATURE_MAPPING: dict[str, str] = {
    # Avocado feature → our 60-d feature name
    # Rationale documented in REVISION_PARSNIP_AVOCADO.md §4.
    #
    # Flux-ratio features → amplitude proxies
    "pos_flux_ratio": "i_amplitude",         # max_flux / (max-min) ≈ amplitude in dominant band
    "max_flux_ratio_red": "Y_amplitude",     # red fraction of peak flux
    "max_flux_ratio_blue": "g_amplitude",    # blue fraction of peak flux
    "min_flux_ratio_red": "z_amplitude",     # red fraction of min flux
    "min_flux_ratio_blue": "u_amplitude",    # blue fraction of min flux
    # Max magnitude → mean flux proxy (inverse relationship)
    "max_mag": "i_mean",
    # Length scale → slope proxy (longer GP timescale ≈ shallower slope)
    "length_scale": "r_slope",
    # Temporal widths → slope proxies
    "positive_width": "g_slope",             # width above zero ∝ 1/slope
    "time_fwd_max_0.5": "i_slope",           # time to fade to 50% ∝ decline rate
    "time_bwd_max_0.5": "r_std",             # rise time ∝ variability
    # S/N features → mean_snr proxies
    "total_s2n": "r_mean_snr",
    "frac_s2n_5": "i_mean_snr",
    # Percentile differences → skewness/std proxies
    "percentile_diff_10_50": "r_skewness",   # asymmetry of flux distribution
    "percentile_diff_90_50": "i_skewness",
    "percentile_diff_30_50": "g_std",        # spread proxy
    "percentile_diff_70_50": "i_std",
    # Color ratios → cross-band mean proxies
    "time_fwd_max_0.5_ratio_red": "z_mean",
    "time_fwd_max_0.5_ratio_blue": "g_mean",
    "time_bwd_max_0.5_ratio_red": "z_median",
    "time_bwd_max_0.5_ratio_blue": "g_median",
    # Cross-band timing offset → mean proxy
    "max_dt": "r_mean",
    # NOTE: 21/41 Avocado features mapped to 20/60 of our features.
    # Unmapped: 11 astrophysically meaningful (no 60-d analogue) + 9 cadence artefacts.
    # Full rationale in REVISION_AVOCADO_MAPPING.md.
}


class AvocadoModelWrapper:
    """Wrapper to give LightGBM booster a predict_proba interface.

    Compatible with our faithfulness_correlation metric which expects
    model.predict_proba(x) → (N, n_classes).
    """

    def __init__(self, booster: lgb.Booster) -> None:
        self.booster = booster

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            x: Input features, shape (N, n_features).

        Returns:
            Probability array, shape (N, n_classes).
        """
        raw = self.booster.predict(x)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        return raw

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(x), axis=1)


def compute_avocado_xai(
    seed: int,
    results_dir: Path | None = None,
    n_samples: int = 50,
) -> dict[str, float]:
    """Compute XAI metrics for a trained Avocado model.

    Args:
        seed: Random seed (determines which checkpoint to load).
        results_dir: Path to results directory.
        n_samples: Number of test samples for XAI evaluation.

    Returns:
        Dict with faithfulness_mean/std, complexity_mean/std, plausibility_mean/std.
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "scripts" / "results"

    seed_dir = results_dir / f"seed_{seed}"

    # Load model
    model_path = seed_dir / "checkpoints" / "avocado_lgb.txt"
    booster = lgb.Booster(model_file=str(model_path))
    wrapper = AvocadoModelWrapper(booster)

    # Load test data and feature names
    pred_data = np.load(seed_dir / "avocado_predictions.npz")
    test_features = pred_data["test_features"]
    feature_names = list(pred_data["feature_names"])

    # Select n_samples for evaluation
    rng = np.random.default_rng(seed)
    n_test = len(test_features)
    sample_idx = rng.choice(n_test, size=min(n_samples, n_test), replace=False)

    # TreeSHAP
    logger.info(f"Computing TreeSHAP (seed={seed})…")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(test_features[sample_idx])

    mean_shap = normalize_shap_values(shap_values)

    # 1. Faithfulness
    faithfulness_vals = []
    for i, idx in enumerate(sample_idx):
        x = test_features[idx]
        attr = mean_shap[i]
        f = faithfulness_correlation(wrapper, x, attr, seed=seed)
        faithfulness_vals.append(f)

    # 2. Complexity
    complexity_vals = [explanation_complexity(mean_shap[i]) for i in range(len(sample_idx))]

    # 3. Plausibility — project Avocado features onto 60-d expert mask
    plausibility_vals = _compute_plausibility_avocado(mean_shap, feature_names)

    metrics = {
        "faithfulness_mean": float(np.mean(faithfulness_vals)),
        "faithfulness_std": float(np.std(faithfulness_vals)),
        "complexity_mean": float(np.mean(complexity_vals)),
        "complexity_std": float(np.std(complexity_vals)),
        "plausibility_mean": float(np.mean(plausibility_vals)),
        "plausibility_std": float(np.std(plausibility_vals)),
    }

    logger.info(
        f"  Avocado XAI (seed={seed}): "
        f"faith={metrics['faithfulness_mean']:.3f}, "
        f"compl={metrics['complexity_mean']:.3f}, "
        f"plaus={metrics['plausibility_mean']:.3f}"
    )
    return metrics


def _compute_plausibility_avocado(
    attributions: np.ndarray,
    feature_names: list[str],
) -> list[float]:
    """Compute plausibility by projecting Avocado attributions onto 60-d space.

    Features that have a mapping in FEATURE_MAPPING contribute their
    attribution to the corresponding slot in the 60-d vector.
    Unmapped features are zero-padded.

    Args:
        attributions: shape (n_samples, n_avocado_features).
        feature_names: Avocado feature names.

    Returns:
        List of plausibility scores per sample.
    """
    from src.xai.metrics import build_expert_mask_tabular

    # Our 60-d feature names
    all_bands = ["u", "g", "r", "i", "z", "Y"]
    stat_names = ["mean", "std", "amplitude", "median", "n_obs",
                  "skewness", "kurtosis", "mean_snr", "slope", "frac_above_mean"]
    our_feature_names = [f"{b}_{s}" for b in all_bands for s in stat_names]

    expert_mask = build_expert_mask_tabular(our_feature_names)

    # Build projection matrix: (n_avocado_features,) → (60,)
    our_name_to_idx = {name: i for i, name in enumerate(our_feature_names)}

    plaus_vals = []
    for sample_idx in range(attributions.shape[0]):
        attr = attributions[sample_idx]

        # Project to 60-d
        projected = np.zeros(60)
        for j, avocado_feat in enumerate(feature_names):
            if avocado_feat in FEATURE_MAPPING:
                our_feat = FEATURE_MAPPING[avocado_feat]
                if our_feat in our_name_to_idx:
                    projected[our_name_to_idx[our_feat]] += attr[j]
            # Unmapped features → zero (no contribution to plausibility)

        plaus = plausibility_score(projected, expert_mask)
        plaus_vals.append(plaus)

    n_mapped = sum(1 for f in feature_names if f in FEATURE_MAPPING)
    logger.info(
        f"  Plausibility projection: {n_mapped}/{len(feature_names)} "
        f"Avocado features mapped to 60-d space"
    )

    return plaus_vals
