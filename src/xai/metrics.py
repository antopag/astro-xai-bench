"""Quantitative xAI evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


def faithfulness_correlation(
    model: Any,
    x: np.ndarray,
    attribution: np.ndarray,
    n_perturbations: int = 100,
    seed: int = 42,
) -> float:
    """Perturbation-based faithfulness metric.

    Progressively masks features in order of attribution magnitude and measures
    the correlation between cumulative attribution removed and prediction change.

    Args:
        model: Model with predict_proba method.
        x: Single input sample, shape (n_features,).
        attribution: Attribution values, same shape as x.
        n_perturbations: Number of perturbation steps.
        seed: Random seed.

    Returns:
        Pearson correlation coefficient (higher = more faithful).
    """
    rng = np.random.default_rng(seed)
    n_features = len(x)
    n_steps = min(n_perturbations, n_features)

    # Get baseline prediction
    x_batch = x.reshape(1, -1)
    base_prob = model.predict_proba(x_batch)[0]
    pred_class = base_prob.argmax()
    base_score = base_prob[pred_class]

    # Sort features by attribution magnitude (descending)
    order = np.argsort(np.abs(attribution))[::-1]

    # Progressively mask features
    attr_cumsum = []
    pred_changes = []
    x_perturbed = x.copy()

    step_size = max(1, n_features // n_steps)
    for step in range(0, n_features, step_size):
        end = min(step + step_size, n_features)
        features_to_mask = order[step:end]

        # Replace with random baseline (from training distribution)
        x_perturbed[features_to_mask] = rng.normal(0, 1, size=len(features_to_mask))

        new_prob = model.predict_proba(x_perturbed.reshape(1, -1))[0]
        pred_change = base_score - new_prob[pred_class]
        cumulative_attr = np.abs(attribution[order[:end]]).sum()

        attr_cumsum.append(cumulative_attr)
        pred_changes.append(pred_change)

    # Pearson correlation
    if len(attr_cumsum) < 3:
        return 0.0

    attr_arr = np.array(attr_cumsum)
    pred_arr = np.array(pred_changes)

    if attr_arr.std() < 1e-10 or pred_arr.std() < 1e-10:
        return 0.0

    corr = np.corrcoef(attr_arr, pred_arr)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def explanation_complexity(attribution: np.ndarray, threshold: float = 0.05) -> float:
    """Measure explanation complexity via entropy of normalized attributions.

    Lower complexity = fewer dominant features = more interpretable.

    Args:
        attribution: Attribution map (any shape, will be flattened).
        threshold: Minimum attribution value to consider.

    Returns:
        Complexity score (entropy-based, lower = simpler).
    """
    flat = np.abs(attribution).flatten()

    # Use relative threshold: fraction of max attribution
    abs_threshold = threshold * flat.max() if flat.max() > 0 else threshold
    flat = flat[flat > abs_threshold]
    if len(flat) == 0:
        return 0.0

    # Normalize to probability distribution
    total = flat.sum()
    if total < 1e-10:
        return 0.0
    p = flat / total

    # Shannon entropy
    entropy = -np.sum(p * np.log(p + 1e-10))

    # Normalize by max possible entropy
    max_entropy = np.log(len(p))
    if max_entropy < 1e-10:
        return 0.0

    return float(entropy / max_entropy)


def complexity_band_aggregated(
    attribution: np.ndarray,
    representation: str,
    n_bands: int = 6,
) -> float:
    """Band-aggregated explanation complexity.

    The original ``explanation_complexity`` is computed on the raw attribution
    space, which is 60 dimensions for the tabular models but $\\sim$300{,}000
    dimensions for the GAF-image models. Comparing entropies across spaces
    of such different cardinalities biases the metric towards diffuse
    pixel-level attributions.

    This function aggregates the absolute attribution into a fixed,
    physically meaningful basis of size ``n_bands`` (one entry per
    photometric passband) and returns the normalised Shannon entropy of
    that 6-vector. All seven model architectures are then comparable on the
    same 6-dimensional scale.

    Args:
        attribution: Attribution array. Allowed shapes:
            * tabular: ``(60,)`` (10 features per band, in band-major order
              matching ``get_feature_names``);
            * timeseries: ``(6, 256)``;
            * image: ``(6, 224, 224)``.
        representation: One of ``"tabular"``, ``"timeseries"``, ``"image"``.
        n_bands: Number of photometric bands (default 6 for PLAsTiCC).

    Returns:
        Normalised entropy in $[0, 1]$ over the band dimension. Lower
        values mean the explanation concentrates on a few bands.
    """
    a = np.abs(np.asarray(attribution, dtype=np.float64))
    rep = representation.lower()

    if rep == "tabular":
        if a.size != n_bands * 10:
            raise ValueError(
                f"Tabular attribution must have {n_bands * 10} entries, got {a.size}."
            )
        per_band = a.reshape(n_bands, 10).sum(axis=1)  # (n_bands,)
    elif rep == "timeseries":
        if a.ndim != 2 or a.shape[0] != n_bands:
            raise ValueError(
                f"Timeseries attribution must have shape ({n_bands}, T), got {a.shape}."
            )
        per_band = a.sum(axis=1)
    elif rep == "image":
        if a.ndim != 3 or a.shape[0] != n_bands:
            raise ValueError(
                f"Image attribution must have shape ({n_bands}, H, W), got {a.shape}."
            )
        per_band = a.sum(axis=(1, 2))
    else:
        raise ValueError(f"Unknown representation: {representation!r}")

    total = per_band.sum()
    if total < 1e-12:
        return 0.0
    p = per_band / total
    H = float(-(p * np.log(p + 1e-12)).sum())
    Hmax = float(np.log(n_bands))
    return H / Hmax if Hmax > 0 else 0.0


def plausibility_score(
    attribution: np.ndarray,
    expert_mask: np.ndarray,
) -> float:
    """Plausibility: overlap between attribution and astrophysical prior.

    Uses IoU (Intersection over Union) between thresholded attribution
    and expert-defined important regions.

    Args:
        attribution: Model attribution map (any shape).
        expert_mask: Binary mask from domain expert (same shape).

    Returns:
        IoU-like plausibility score in [0, 1].
    """
    # Threshold attribution at median of positive values
    attr_abs = np.abs(attribution)
    positive = attr_abs[attr_abs > 0]
    if len(positive) == 0:
        return 0.0
    thresh = np.median(positive)
    attr_binary = (attr_abs >= thresh).astype(float)

    expert_binary = (expert_mask > 0).astype(float)

    intersection = (attr_binary * expert_binary).sum()
    union = np.clip(attr_binary + expert_binary, 0, 1).sum()

    if union < 1e-10:
        return 0.0

    return float(intersection / union)


def build_expert_mask_tabular(feature_names: list[str]) -> np.ndarray:
    """Build astrophysical expert mask for PLAsTiCC tabular features.

    Encodes domain knowledge about which features are physically meaningful
    for transient classification. Based on Kessler+2019 (PLAsTiCC),
    Muthukrishna+2019 (RAPID), Boone 2019 (Avocado).

    Expert-selected features (21/60):
        - amplitude (all bands): primary brightness discriminant
        - slope (g, r, i): evolutionary timescale
        - std (all bands): total variability proxy
        - skewness (g, r, i): rise/decline asymmetry
        - mean_snr (g, r, i): per-band detectability

    Excluded:
        - mean, median: raw statistical moments, not domain knowledge in
          their own right (removed in revision pass 2 per referee request)
        - n_obs: survey strategy artefact, no physical content
        - frac_above_mean: redundant with skewness/amplitude
        - kurtosis: too sensitive to outliers, rarely used in practice

    Note: the flux-only 60-d representation does not expose timestamp-aware
    features (e.g. time-above-half-peak) or band-difference features
    (e.g. r-i colour at peak); see paper Sect. 4.2.3.

    Args:
        feature_names: List of 60 feature names from get_feature_names().

    Returns:
        Binary mask array of shape (60,), 1 for expert-selected features.
    """
    expert_features = set()

    all_bands = ["u", "g", "r", "i", "z", "Y"]
    gri_bands = ["g", "r", "i"]

    # amplitude — all bands
    for b in all_bands:
        expert_features.add(f"{b}_amplitude")

    # slope — g, r, i
    for b in gri_bands:
        expert_features.add(f"{b}_slope")

    # std — all bands
    for b in all_bands:
        expert_features.add(f"{b}_std")

    # skewness — g, r, i
    for b in gri_bands:
        expert_features.add(f"{b}_skewness")

    # mean_snr — g, r, i
    for b in gri_bands:
        expert_features.add(f"{b}_mean_snr")

    # NOTE: mean and median (formerly g,r,i,z) were removed in revision pass 2:
    # raw statistical moments are not domain knowledge in their own right.
    # Mask cardinality is therefore 21 (was 29).

    mask = np.array([1.0 if fn in expert_features else 0.0 for fn in feature_names])
    logger.info(f"Expert mask: {int(mask.sum())}/{len(mask)} features selected")
    return mask


def compute_plausibility_tabular(
    shap_values: np.ndarray | list,
    expert_mask: np.ndarray,
    n_classes: int | None = None,
) -> dict[str, float]:
    """Compute plausibility score for tabular SHAP attributions.

    Averages plausibility across all test samples.

    Args:
        shap_values: SHAP values (list, 2D, or 3D array).
        expert_mask: Binary mask from build_expert_mask_tabular().
        n_classes: Number of classes (for 3D arrays).

    Returns:
        Dict with plausibility_mean and plausibility_std.
    """
    # Get mean |SHAP| per feature across samples (and classes if multi-class)
    if isinstance(shap_values, list):
        # list of (n_samples, n_features), one per class
        mean_attr = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # (n_samples, n_features)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_features = len(expert_mask)
        if shap_values.shape[2] == n_features:
            # (n_samples, n_classes, n_features)
            mean_attr = np.abs(shap_values).mean(axis=1)
        else:
            # (n_samples, n_features, n_classes)
            mean_attr = np.abs(shap_values).mean(axis=2)
    else:
        mean_attr = np.abs(shap_values)  # (n_samples, n_features)

    # Compute plausibility per sample
    scores = []
    for i in range(len(mean_attr)):
        score = plausibility_score(mean_attr[i], expert_mask)
        scores.append(score)

    result = {
        "plausibility_mean": float(np.mean(scores)),
        "plausibility_std": float(np.std(scores)),
    }
    logger.info(
        f"Plausibility: {result['plausibility_mean']:.4f}±{result['plausibility_std']:.4f}"
    )
    return result


def compute_xai_metrics_tabular(
    model: Any,
    x_test: np.ndarray,
    shap_values: np.ndarray | list,
    n_samples: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Compute xAI metrics for tabular models.

    Args:
        model: Trained model with predict_proba.
        x_test: Test feature matrix.
        shap_values: SHAP values (list for multi-class).
        n_samples: Number of test samples to evaluate.
        seed: Random seed.

    Returns:
        Dict with mean faithfulness and complexity scores.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_test), size=min(n_samples, len(x_test)), replace=False)

    faithfulness_scores = []
    complexity_scores = []

    for idx in indices:
        # Get SHAP attribution for predicted class
        pred_class = model.predict(x_test[idx:idx+1])[0]
        if isinstance(shap_values, list):
            attr = shap_values[pred_class][idx]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            n_features = x_test.shape[1]
            if shap_values.shape[2] == n_features:
                # (n_samples, n_classes, n_features)
                attr = shap_values[idx, pred_class, :]
            else:
                # (n_samples, n_features, n_classes)
                attr = shap_values[idx, :, pred_class]
        else:
            attr = shap_values[idx]

        faith = faithfulness_correlation(model, x_test[idx], attr, seed=seed)
        compl = explanation_complexity(attr)

        faithfulness_scores.append(faith)
        complexity_scores.append(compl)

    metrics = {
        "faithfulness_mean": float(np.mean(faithfulness_scores)),
        "faithfulness_std": float(np.std(faithfulness_scores)),
        "complexity_mean": float(np.mean(complexity_scores)),
        "complexity_std": float(np.std(complexity_scores)),
    }

    logger.info(
        f"xAI Metrics: faithfulness={metrics['faithfulness_mean']:.4f}±{metrics['faithfulness_std']:.4f} | "
        f"complexity={metrics['complexity_mean']:.4f}±{metrics['complexity_std']:.4f}"
    )

    return metrics
