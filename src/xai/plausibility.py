"""Cross-representation plausibility for tabular and DL models.

The "plausibility" of an explanation is the alignment between model
attributions and an astrophysically motivated expert feature mask
(see ``build_expert_mask_tabular``). For tabular models the mask is
applied directly to the 60-dimensional SHAP attribution.

For deep learning models, attributions live in a different space
(timestep or pixel) and must be *projected* into the same 60-dimensional
tabular feature space before the mask can be applied. This module
implements that projection via simple per-band aggregation statistics
that mirror, one-by-one, the 10 features extracted in
``src.models.tabular_rf.extract_features``:

    [mean, std, amplitude, median, n_obs, skewness, kurtosis,
     mean_snr, slope, frac_above_mean]

For the time-series representation (LSTM, CNN1D, attribution shape
``(B, T)``) the projection uses 10 statistics computed on the
attribution itself plus its correlation/slope against the original flux
(used as a slope proxy).

For the GAF-image representation (CNN2D, ViT, attribution shape
``(B, H, W)``) the projection uses 5 statistics per band including the
diagonality of the heatmap, which encodes local temporal evolution in
GAF images and therefore acts as a "slope" proxy.

This is, to our knowledge, the first attempt to compare attribution
plausibility across heterogeneous representations of photometric light
curves on a common astrophysical feature basis.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew


# ----------------------------------------------------------------------------
# Common building blocks
# ----------------------------------------------------------------------------

# Order of the 10 statistics produced per band — must mirror
# src.models.tabular_rf.extract_features so that the resulting 60-vector
# aligns positionally with the expert mask.
FEATURE_ORDER: tuple[str, ...] = (
    "mean", "std", "amplitude", "median", "n_obs",
    "skewness", "kurtosis", "mean_snr", "slope", "frac_above_mean",
)


def _iou(attr: np.ndarray, expert_mask: np.ndarray) -> float:
    """IoU between |attribution| (thresholded at the median of positives) and the expert mask.

    Mirrors ``src.xai.metrics.plausibility_score`` so the value is comparable
    across model families.
    """
    attr_abs = np.abs(attr)
    positive = attr_abs[attr_abs > 0]
    if len(positive) == 0:
        return 0.0
    thresh = float(np.median(positive))
    attr_bin = (attr_abs >= thresh).astype(np.float32)
    mask_bin = (expert_mask > 0).astype(np.float32)
    inter = (attr_bin * mask_bin).sum()
    union = np.clip(attr_bin + mask_bin, 0.0, 1.0).sum()
    if union < 1e-10:
        return 0.0
    return float(inter / union)


# ----------------------------------------------------------------------------
# Tabular (refactor of src.xai.metrics.compute_plausibility_tabular)
# ----------------------------------------------------------------------------

def compute_tabular_plausibility(
    shap_values: np.ndarray | list,
    expert_mask: np.ndarray,
) -> dict[str, float]:
    """Plausibility for tabular models with SHAP attributions.

    Args:
        shap_values: SHAP values (list of per-class arrays, 2D, or 3D ndarray
            shaped ``(N, C, F)`` or ``(N, F, C)``).
        expert_mask: Binary mask of shape ``(F,)`` from
            ``build_expert_mask_tabular``.

    Returns:
        ``{plausibility_mean, plausibility_std}`` averaged over samples.
    """
    if isinstance(shap_values, list):
        mean_attr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_features = len(expert_mask)
        if shap_values.shape[2] == n_features:
            mean_attr = np.abs(shap_values).mean(axis=1)
        else:
            mean_attr = np.abs(shap_values).mean(axis=2)
    else:
        mean_attr = np.abs(shap_values)

    scores = [_iou(mean_attr[i], expert_mask) for i in range(len(mean_attr))]
    return {
        "plausibility_mean": float(np.mean(scores)),
        "plausibility_std": float(np.std(scores)),
    }


# ----------------------------------------------------------------------------
# Time series projection (LSTM, CNN1D)
# ----------------------------------------------------------------------------

def _project_timeseries_band(
    attr_band: np.ndarray,
    flux_band: np.ndarray | None,
) -> np.ndarray:
    """Project a single-band attribution ``(T,)`` into 10 tabular meta-attributions.

    Each output value is a non-negative scalar that approximates how much the
    corresponding tabular feature is "supported" by the band-level attribution.
    The construction is intentionally simple so the meta-attribution lives on
    the same scale as a positive importance.
    """
    a = np.asarray(attr_band, dtype=np.float64)
    a_abs = np.abs(a)

    # Pre-compute reusable statistics
    a_mean = float(a_abs.mean())
    a_std = float(a_abs.std())
    a_max = float(a_abs.max()) if a_abs.size else 0.0
    a_min = float(a_abs.min()) if a_abs.size else 0.0
    a_med = float(np.median(a_abs))
    nz_frac = float((a_abs > 1e-12).mean())  # "n_obs" proxy: fraction of active timesteps

    # Higher-order shape stats of the *signed* attribution profile
    if a.std() > 1e-8:
        a_skew = float(_skew(a))
        a_kurt = float(_kurtosis(a))
    else:
        a_skew = 0.0
        a_kurt = 0.0

    # Slope proxy: correlation between attribution magnitude and the original
    # flux profile. Captures whether IG follows the temporal evolution of the
    # band signal (which is exactly what the tabular `slope` feature encodes).
    if flux_band is not None and len(flux_band) == len(a) and a_abs.std() > 1e-8 and np.std(flux_band) > 1e-8:
        slope_proxy = float(abs(np.corrcoef(a_abs, flux_band)[0, 1]))
    else:
        # Fallback: linear-fit slope of the attribution profile vs time index
        if a_abs.size > 1 and a_abs.std() > 1e-8:
            t = np.arange(a_abs.size, dtype=np.float64)
            slope_proxy = float(abs(np.polyfit(t, a_abs, 1)[0]))
        else:
            slope_proxy = 0.0

    above_mean_frac = float((a_abs > a_mean).mean()) if a_mean > 0 else 0.0

    # Map to FEATURE_ORDER positions:
    return np.array([
        a_mean,                     # mean
        a_std,                      # std
        a_max - a_min,              # amplitude
        a_med,                      # median
        nz_frac,                    # n_obs
        abs(a_skew),                # skewness  (we use |.| since the mask is sign-agnostic)
        abs(a_kurt),                # kurtosis
        a_max,                      # mean_snr   (peak importance proxy)
        slope_proxy,                # slope
        above_mean_frac,            # frac_above_mean
    ], dtype=np.float64)


def project_timeseries_attribution(
    attribution: np.ndarray,
    flux: np.ndarray | None = None,
) -> np.ndarray:
    """Project an LSTM/CNN1D IG attribution to a 60-vector in tabular feature space.

    Args:
        attribution: Per-sample IG attribution, shape ``(B, T)`` (e.g. ``(6, 256)``).
        flux: Optional original light curve of the same shape, used as a slope
            proxy via correlation with attribution magnitude.

    Returns:
        Meta-attribution of shape ``(B * 10,)`` aligned with
        ``get_feature_names``.
    """
    n_bands = attribution.shape[0]
    out = np.zeros(n_bands * 10, dtype=np.float64)
    for b in range(n_bands):
        flux_b = flux[b] if flux is not None else None
        out[b * 10:(b + 1) * 10] = _project_timeseries_band(attribution[b], flux_b)
    return out


def compute_timeseries_plausibility(
    attributions: np.ndarray,
    expert_mask: np.ndarray,
    flux: np.ndarray | None = None,
) -> dict[str, float]:
    """Plausibility for LSTM/CNN1D using band-aggregated meta-attributions.

    Args:
        attributions: IG attributions, shape ``(N, B, T)``.
        expert_mask: 60-d binary expert mask.
        flux: Optional ``(N, B, T)`` original light curves for slope proxy.

    Returns:
        ``{plausibility_mean, plausibility_std}`` averaged over samples.
    """
    scores = []
    for i in range(len(attributions)):
        flux_i = flux[i] if flux is not None else None
        meta = project_timeseries_attribution(attributions[i], flux_i)
        scores.append(_iou(meta, expert_mask))
    return {
        "plausibility_mean": float(np.mean(scores)),
        "plausibility_std": float(np.std(scores)),
    }


# ----------------------------------------------------------------------------
# Image projection (CNN2D, ViT)
# ----------------------------------------------------------------------------

def _diagonality(heatmap: np.ndarray) -> float:
    """Mean attribution along the main diagonal vs off-diagonal mean.

    GAF images encode local temporal evolution along the main diagonal —
    high diagonal energy in the attribution is therefore a "slope" proxy
    (rapid flux evolution being attended to).
    """
    h, w = heatmap.shape
    n = min(h, w)
    if n == 0:
        return 0.0
    diag = np.array([heatmap[i, i] for i in range(n)], dtype=np.float64)
    off_mean = (heatmap.sum() - diag.sum()) / max(h * w - n, 1)
    return float(diag.mean() - off_mean)


def _entropy_of(values: np.ndarray) -> float:
    """Normalised Shannon entropy of a non-negative vector (returns 0 if degenerate)."""
    v = values.flatten().astype(np.float64)
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    p = v / v.sum()
    H = float(-(p * np.log(p + 1e-12)).sum())
    Hmax = float(np.log(p.size))
    return H / Hmax if Hmax > 0 else 0.0


def _project_image_band(heatmap: np.ndarray) -> np.ndarray:
    """Project a single-band GAF heatmap ``(H, W)`` to 10 tabular meta-attributions.

    The construction parallels ``_project_timeseries_band`` but extracts
    geometry-aware statistics from the 2D map. The five "core" stats
    (mean, max, diagonality, concentration via entropy, frac above mean)
    are then mapped to the 10 tabular slots via the lookup table below;
    redundant slots get the closest physically-meaningful proxy.
    """
    h_abs = np.abs(heatmap.astype(np.float64))
    h_mean = float(h_abs.mean())
    h_std = float(h_abs.std())
    h_max = float(h_abs.max()) if h_abs.size else 0.0
    h_min = float(h_abs.min()) if h_abs.size else 0.0
    h_med = float(np.median(h_abs))
    nz_frac = float((h_abs > 1e-12).mean())

    # 1 - entropy: high concentration => few "important" pixels (skew/kurt proxy)
    entropy = _entropy_of(h_abs)
    concentration = 1.0 - entropy

    diag_proxy = max(_diagonality(h_abs), 0.0)
    above_mean_frac = float((h_abs > h_mean).mean()) if h_mean > 0 else 0.0

    # Lookup-table mapping (mirrors FEATURE_ORDER):
    return np.array([
        h_mean,                    # mean             — average activation
        h_std,                     # std              — spread of activations
        h_max - h_min,             # amplitude        — peak vs floor
        h_med,                     # median           — robust centre
        nz_frac,                   # n_obs            — fraction of active pixels
        concentration,             # skewness  proxy  — peakedness via 1-entropy
        concentration,             # kurtosis  proxy  — same source: GAF heatmaps
                                   #                    do not separate moments
                                   #                    cleanly, so we duplicate.
        h_max,                     # mean_snr  proxy  — peak importance
        diag_proxy,                # slope     proxy  — diagonal energy
        above_mean_frac,           # frac_above_mean  — direct analogue
    ], dtype=np.float64)


def project_image_attribution(attribution: np.ndarray) -> np.ndarray:
    """Project a CNN2D/ViT IG attribution ``(B, H, W)`` to a 60-vector."""
    n_bands = attribution.shape[0]
    out = np.zeros(n_bands * 10, dtype=np.float64)
    for b in range(n_bands):
        out[b * 10:(b + 1) * 10] = _project_image_band(attribution[b])
    return out


def compute_image_plausibility(
    attributions: np.ndarray,
    expert_mask: np.ndarray,
) -> dict[str, float]:
    """Plausibility for CNN2D/ViT using GAF-heatmap meta-attributions.

    Args:
        attributions: IG attributions, shape ``(N, B, H, W)``.
        expert_mask: 60-d binary expert mask.
    """
    scores = []
    for i in range(len(attributions)):
        meta = project_image_attribution(attributions[i])
        scores.append(_iou(meta, expert_mask))
    return {
        "plausibility_mean": float(np.mean(scores)),
        "plausibility_std": float(np.std(scores)),
    }


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------

def compute_plausibility(
    attributions,
    expert_mask: np.ndarray,
    model_type: str,
    *,
    flux: np.ndarray | None = None,
) -> dict[str, float]:
    """Dispatch to the right plausibility implementation.

    Args:
        attributions: SHAP values (tabular) or stacked IG attributions
            (timeseries / image), shape ``(N, ...)``.
        expert_mask: 60-d binary expert mask.
        model_type: One of ``"tabular"``, ``"timeseries"``, ``"image"``.
        flux: Optional ``(N, B, T)`` original light curves used as slope
            proxy by the timeseries projection.

    Returns:
        ``{plausibility_mean, plausibility_std}``.
    """
    mt = model_type.lower()
    if mt == "tabular":
        result = compute_tabular_plausibility(attributions, expert_mask)
    elif mt == "timeseries":
        result = compute_timeseries_plausibility(attributions, expert_mask, flux=flux)
    elif mt == "image":
        result = compute_image_plausibility(attributions, expert_mask)
    else:
        raise ValueError(
            f"Unknown model_type={model_type!r}. Use 'tabular', 'timeseries', or 'image'."
        )
    logger.info(
        f"Plausibility ({mt}): "
        f"{result['plausibility_mean']:.4f}±{result['plausibility_std']:.4f}"
    )
    return result
