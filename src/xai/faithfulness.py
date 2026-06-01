"""Insertion / deletion AUC faithfulness metrics.

These metrics, originally introduced in the RISE paper
\\citep{petsiuk2018}, measure how well an attribution map identifies the
features that drive a model's prediction by gradually masking
(``deletion``) or revealing (``insertion``) the input in order of
attribution magnitude. They avoid the off-manifold concerns associated
with Gaussian-noise perturbations because the masking baseline is just a
zero tensor.

Three operating modes are supported:

* **per_feature** — for tabular models (60 atomic units = 60 features);
* **per_band** — for time-series DL models (6 atomic units = 6 photometric
  passbands; deleting one ``band'' zeros all 256 of its timesteps);
* **per_patch** — for GAF-image DL models (atomic units are non-overlapping
  ``patch_size``\\(\\times\\)``patch_size`` patches per band, matching the
  ViT patch grid). The number of curve points is downsampled to keep the
  evaluation tractable on a single consumer GPU.

Each call returns the deletion and insertion curves and their respective
areas under the curve. ``deletion_auc`` lower is better, ``insertion_auc``
higher is better. Both are normalised so that they fall in $[0, 1]$.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from loguru import logger


# ----------------------------------------------------------------------------
# Tabular (per-feature, 60 atomic units)
# ----------------------------------------------------------------------------

def insertion_deletion_tabular(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attribution: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Per-feature insertion/deletion AUC for a single tabular sample.

    Args:
        predict_proba_fn: Callable mapping ``(B, F)`` to ``(B, n_classes)``.
        x: Single tabular sample of shape ``(F,)``.
        attribution: Magnitudes of shape ``(F,)``.
    """
    n = len(x)
    order = np.argsort(np.abs(attribution))[::-1]

    pred_class = int(predict_proba_fn(x.reshape(1, -1))[0].argmax())

    # Deletion: start full, zero in attribution order
    del_batch = np.tile(x.reshape(1, -1), (n + 1, 1)).astype(np.float32)
    for k in range(1, n + 1):
        del_batch[k, order[:k]] = 0.0

    # Insertion: start empty, fill in attribution order
    ins_batch = np.zeros((n + 1, n), dtype=np.float32)
    for k in range(1, n + 1):
        ins_batch[k, order[:k]] = x[order[:k]]

    del_probs = predict_proba_fn(del_batch)[:, pred_class]
    ins_probs = predict_proba_fn(ins_batch)[:, pred_class]
    fractions = np.linspace(0.0, 1.0, n + 1)

    return {
        "fractions": fractions,
        "del_probs": del_probs,
        "ins_probs": ins_probs,
        "del_auc": float(np.trapz(del_probs, fractions)),
        "ins_auc": float(np.trapz(ins_probs, fractions)),
    }


# ----------------------------------------------------------------------------
# Time series (per-band, 6 atomic units)
# ----------------------------------------------------------------------------

def insertion_deletion_timeseries(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attribution: np.ndarray,
    n_bands: int = 6,
) -> dict[str, np.ndarray | float]:
    """Per-band insertion/deletion AUC for a single time-series sample.

    Args:
        predict_proba_fn: Callable mapping ``(B, n_bands, T)`` to ``(B, n_classes)``.
        x: Sample of shape ``(n_bands, T)``.
        attribution: Per-pixel IG attribution of shape ``(n_bands, T)``.
            The aggregation is done by summing $|.|$ along the time axis.
    """
    if x.shape[0] != n_bands:
        raise ValueError(f"Expected {n_bands} bands, got shape {x.shape}")

    band_attr = np.abs(attribution).sum(axis=1)  # (n_bands,)
    order = np.argsort(band_attr)[::-1]

    pred_class = int(predict_proba_fn(x[None])[0].argmax())

    del_batch = np.tile(x[None], (n_bands + 1, 1, 1)).astype(np.float32)
    for k in range(1, n_bands + 1):
        del_batch[k, order[:k], :] = 0.0

    ins_batch = np.zeros((n_bands + 1,) + x.shape, dtype=np.float32)
    for k in range(1, n_bands + 1):
        for b in order[:k]:
            ins_batch[k, b, :] = x[b, :]

    del_probs = predict_proba_fn(del_batch)[:, pred_class]
    ins_probs = predict_proba_fn(ins_batch)[:, pred_class]
    fractions = np.linspace(0.0, 1.0, n_bands + 1)

    return {
        "fractions": fractions,
        "del_probs": del_probs,
        "ins_probs": ins_probs,
        "del_auc": float(np.trapz(del_probs, fractions)),
        "ins_auc": float(np.trapz(ins_probs, fractions)),
    }


# ----------------------------------------------------------------------------
# GAF images (per-patch, ~196 atomic units per band; downsampled curve)
# ----------------------------------------------------------------------------

def insertion_deletion_image(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attribution: np.ndarray,
    patch_size: int = 16,
    max_curve_points: int = 50,
) -> dict[str, np.ndarray | float]:
    """Per-patch insertion/deletion AUC for a single GAF-image sample.

    Args:
        predict_proba_fn: Callable mapping ``(B, n_bands, H, W)`` to
            ``(B, n_classes)``.
        x: Sample of shape ``(n_bands, H, W)``. ``H`` and ``W`` must be
            divisible by ``patch_size``.
        attribution: Per-pixel IG attribution of shape ``(n_bands, H, W)``.
            Aggregated by summing $|.|$ inside each ``patch_size``\\(\\times\\)
            ``patch_size`` patch.
        patch_size: Edge length of the non-overlapping patch grid (default
            16, matching ``vit_small_patch16_224``).
        max_curve_points: Maximum number of points on the deletion/insertion
            curve. With $\\sim$1{,}176 atomic units this caps the number of
            forward passes per sample, making the metric tractable on a
            single consumer GPU.
    """
    n_bands, H, W = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image size {H}x{W} not divisible by patch_size={patch_size}")
    p = patch_size
    n_pH = H // p
    n_pW = W // p
    n_per_band = n_pH * n_pW
    n_total = n_bands * n_per_band

    # Aggregate per (band, patch) via reshape + sum
    abs_attr = np.abs(attribution)
    attr_patch = abs_attr.reshape(n_bands, n_pH, p, n_pW, p).sum(axis=(2, 4))
    flat_attr = attr_patch.flatten()
    order = np.argsort(flat_attr)[::-1]

    # Choose curve cuts
    n_groups = min(max_curve_points, n_total)
    step = max(1, n_total // n_groups)
    cuts = list(range(0, n_total + 1, step))
    if cuts[-1] != n_total:
        cuts.append(n_total)
    fractions = np.array(cuts, dtype=np.float64) / n_total

    pred_class = int(predict_proba_fn(x[None])[0].argmax())

    def _zero_patches(template: np.ndarray, atomic_indices: np.ndarray) -> np.ndarray:
        out = template.copy()
        for idx in atomic_indices:
            b = idx // n_per_band
            patch_idx = idx % n_per_band
            i = (patch_idx // n_pW) * p
            j = (patch_idx % n_pW) * p
            out[b, i:i + p, j:j + p] = 0.0
        return out

    def _insert_patches(template: np.ndarray, atomic_indices: np.ndarray,
                         source: np.ndarray) -> np.ndarray:
        out = template.copy()
        for idx in atomic_indices:
            b = idx // n_per_band
            patch_idx = idx % n_per_band
            i = (patch_idx // n_pW) * p
            j = (patch_idx % n_pW) * p
            out[b, i:i + p, j:j + p] = source[b, i:i + p, j:j + p]
        return out

    del_batch = np.stack([_zero_patches(x, order[:cut]) for cut in cuts]).astype(np.float32)
    ins_batch = np.stack(
        [_insert_patches(np.zeros_like(x), order[:cut], x) for cut in cuts]
    ).astype(np.float32)

    del_probs = predict_proba_fn(del_batch)[:, pred_class]
    ins_probs = predict_proba_fn(ins_batch)[:, pred_class]

    return {
        "fractions": fractions,
        "del_probs": del_probs,
        "ins_probs": ins_probs,
        "del_auc": float(np.trapz(del_probs, fractions)),
        "ins_auc": float(np.trapz(ins_probs, fractions)),
    }


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate_insertion_deletion(
    per_sample_results: list[dict[str, np.ndarray | float]],
) -> dict[str, float]:
    """Cross-sample mean ± std for deletion / insertion AUC."""
    if not per_sample_results:
        return {
            "deletion_auc_mean": float("nan"),
            "deletion_auc_std": float("nan"),
            "insertion_auc_mean": float("nan"),
            "insertion_auc_std": float("nan"),
        }
    del_aucs = np.array([r["del_auc"] for r in per_sample_results])
    ins_aucs = np.array([r["ins_auc"] for r in per_sample_results])
    out = {
        "deletion_auc_mean": float(del_aucs.mean()),
        "deletion_auc_std": float(del_aucs.std()),
        "insertion_auc_mean": float(ins_aucs.mean()),
        "insertion_auc_std": float(ins_aucs.std()),
    }
    logger.info(
        f"Insertion/Deletion: del_auc={out['deletion_auc_mean']:.4f}±{out['deletion_auc_std']:.4f} | "
        f"ins_auc={out['insertion_auc_mean']:.4f}±{out['insertion_auc_std']:.4f}"
    )
    return out
