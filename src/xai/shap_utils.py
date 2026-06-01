"""Shared SHAP value normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_shap_values(shap_values: list | np.ndarray) -> np.ndarray:
    """Reduce SHAP values to mean |SHAP| per sample per feature.

    Handles all formats returned by shap.TreeExplainer.shap_values():
      - list of (n_samples, n_features) arrays, one per class
      - 3D array (n_samples, n_features, n_classes)
      - 2D array (n_samples, n_features) — binary or pre-reduced

    Args:
        shap_values: Raw output from explainer.shap_values().

    Returns:
        Array of shape (n_samples, n_features) with mean |SHAP| across classes.
    """
    if isinstance(shap_values, list):
        return np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return np.abs(shap_values).mean(axis=2)
    else:
        return np.abs(shap_values)
