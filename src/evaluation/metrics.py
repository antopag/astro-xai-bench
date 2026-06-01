"""Classification metrics for transient evaluation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute accuracy, F1 macro, per-class F1, and AUC.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Class probabilities (for AUC). Optional.

    Returns:
        Dict with metric names and values.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1 in enumerate(per_class_f1):
        metrics[f"f1_class_{i}"] = f1

    # AUC (one-vs-rest)
    if y_proba is not None:
        try:
            metrics["auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            logger.warning("Could not compute AUC (likely missing classes in predictions)")
            metrics["auc_ovr"] = float("nan")

    logger.info(
        f"Metrics: acc={metrics['accuracy']:.4f} | "
        f"F1_macro={metrics['f1_macro']:.4f} | "
        f"F1_weighted={metrics['f1_weighted']:.4f}"
        + (f" | AUC={metrics.get('auc_ovr', float('nan')):.4f}" if y_proba is not None else "")
    )

    return metrics


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
    normalize: bool = True,
) -> None:
    """Plot and optionally save a publication-ready confusion matrix.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        class_names: List of class names.
        save_path: If provided, save figure to this path.
        normalize: If True, normalize per row (recall-based).
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.show()
