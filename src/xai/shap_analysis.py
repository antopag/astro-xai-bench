"""SHAP-based explainability for all model types."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
from loguru import logger


def compute_shap_values(
    model: Any,
    x: np.ndarray,
    background: np.ndarray | None = None,
    method: str = "auto",
    n_background: int = 100,
) -> np.ndarray:
    """Compute SHAP values for a model.

    Args:
        model: Trained model with predict_proba method.
        x: Input samples, shape (n_samples, n_features).
        background: Background dataset for KernelSHAP. If None, uses kmeans summary.
        method: 'tree' for tree models, 'kernel' for model-agnostic, 'auto' to detect.
        n_background: Number of background samples for KernelSHAP.

    Returns:
        SHAP values — for tree models: array of shape (n_classes, n_samples, n_features),
        for kernel: list of arrays.
    """
    if method == "auto":
        model_type = type(model).__name__
        if hasattr(model, "model") and hasattr(model.model, "estimators_"):
            method = "tree"
        elif hasattr(model, "model") and hasattr(model.model, "get_booster"):
            method = "tree"
        else:
            method = "kernel"

    logger.info(f"Computing SHAP values with method={method}...")

    if method == "tree":
        # Use the underlying sklearn/xgboost model
        inner_model = model.model if hasattr(model, "model") else model

        # XGBoost multi-class: use built-in predict(pred_contribs=True)
        # to avoid SHAP/XGBoost base_score parsing bug
        if hasattr(inner_model, "get_booster"):
            import xgboost as xgb

            booster = inner_model.get_booster()
            dmat = xgb.DMatrix(x)
            # Returns (n_samples, n_features+1, n_classes) — last feature col is base value
            raw = booster.predict(dmat, pred_contribs=True)
            logger.debug(f"XGBoost pred_contribs raw shape: {raw.shape}, x shape: {x.shape}")
            if raw.ndim == 3:
                # XGBoost returns (n_samples, n_classes, n_features+1)
                shap_values = raw[:, :, :-1]  # drop bias column per class
            else:
                shap_values = raw[:, :-1]
        else:
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(x)
    elif method == "kernel":
        if background is None:
            background = shap.kmeans(x, min(n_background, len(x)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(x, nsamples=200)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info("SHAP computation complete.")
    return shap_values


def shap_summary_plot(
    shap_values: np.ndarray | list,
    x: np.ndarray,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    max_display: int = 20,
    save_path: str | None = None,
) -> None:
    """Generate SHAP summary (beeswarm) plot.

    Args:
        shap_values: SHAP values from compute_shap_values.
        x: Feature matrix used for SHAP computation.
        feature_names: Feature names.
        class_names: Class names for multi-class.
        max_display: Max features to show.
        save_path: If provided, save figure.
    """
    fig = plt.figure(figsize=(12, 8))

    # Convert 3D array to list of 2D arrays (one per class)
    # Shape may be (n_samples, n_classes, n_features) or (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_samples, dim1, dim2 = shap_values.shape
        n_features = x.shape[1]
        if dim2 == n_features:
            # (n_samples, n_classes, n_features)
            sv_list = [shap_values[:, c, :] for c in range(dim1)]
        else:
            # (n_samples, n_features, n_classes)
            sv_list = [shap_values[:, :, c] for c in range(dim2)]
    elif isinstance(shap_values, list):
        sv_list = shap_values
    else:
        sv_list = None

    if sv_list is not None:
        mean_abs_shap = np.mean([np.abs(sv) for sv in sv_list], axis=0)
        shap.summary_plot(
            mean_abs_shap,
            x,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
    else:
        shap.summary_plot(
            shap_values,
            x,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"SHAP summary plot saved to {save_path}")
    plt.show()


def shap_bar_plot(
    shap_values: np.ndarray | list,
    feature_names: list[str] | None = None,
    top_k: int = 20,
    save_path: str | None = None,
) -> None:
    """Bar plot of mean |SHAP| per feature.

    Args:
        shap_values: SHAP values (list for multi-class, or 2D array).
        feature_names: Feature names.
        top_k: Number of top features to show.
        save_path: Save path.
    """
    if isinstance(shap_values, list):
        # List of arrays: one (n_samples, n_features) per class
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Average absolute SHAP over all non-feature axes
        # Works for both (n_samples, n_classes, n_features) and (n_samples, n_features, n_classes)
        # Determine which axis is features by checking against feature_names length
        if feature_names is not None:
            n_feat = len(feature_names)
            if shap_values.shape[2] == n_feat:
                mean_abs = np.abs(shap_values).mean(axis=(0, 1))
            else:
                mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # Assume last axis is classes (most common)
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    n_features = len(mean_abs)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    # Sort and take top_k
    idx = np.argsort(mean_abs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), mean_abs[idx][::-1], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_k} Features by SHAP Importance")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"SHAP bar plot saved to {save_path}")
    plt.show()


def get_feature_names(n_bands: int = 6) -> list[str]:
    """Generate feature names matching extract_features output.

    Returns:
        List of 60 feature names (10 per band).
    """
    band_names = ["u", "g", "r", "i", "z", "Y"][:n_bands]
    stat_names = ["mean", "std", "amplitude", "median", "n_obs",
                  "skewness", "kurtosis", "mean_snr", "slope", "frac_above_mean"]
    names = []
    for band in band_names:
        for stat in stat_names:
            names.append(f"{band}_{stat}")
    return names
