"""Ensemble model via stacking."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from src.models import BaseModel


class StackingEnsemble(BaseModel):
    """Stacking ensemble combining probability outputs from multiple base models.

    Args:
        base_models: List of trained BaseModel instances.
        meta_learner: Type of meta-learner ('logistic' or 'rf').
        random_state: Random seed.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        meta_learner: str = "logistic",
        random_state: int = 42,
    ) -> None:
        self.base_models = base_models

        if meta_learner == "logistic":
            self.meta = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.meta = RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )

    def _get_meta_features(self, x_per_model: list[Any]) -> np.ndarray:
        """Stack predict_proba from all base models.

        Args:
            x_per_model: List of inputs, one per base model (format may differ).

        Returns:
            Meta-feature matrix of shape (n_samples, n_models * n_classes).
        """
        probas = []
        for model, x in zip(self.base_models, x_per_model):
            probas.append(model.predict_proba(x))
        return np.hstack(probas)

    def fit(self, x_per_model: list[Any], y: np.ndarray) -> None:
        """Fit meta-learner on stacked base model probabilities.

        WARNING: leak-prone. Calling this with predictions from base models
        that were themselves trained (or early-stopped) on ``x_per_model``
        causes target leakage into the meta-learner. Prefer ``fit_meta``
        with out-of-fold predictions; see ``compute_oof_meta_features`` in
        ``scripts/run_round2.py``.

        Args:
            x_per_model: List of inputs, one per base model.
            y: True labels.
        """
        meta_features = self._get_meta_features(x_per_model)
        logger.info(f"Fitting meta-learner on {meta_features.shape[1]} meta-features...")
        self.meta.fit(meta_features, y)
        logger.info("Ensemble training complete.")

    def fit_meta(self, meta_features: np.ndarray, y: np.ndarray) -> None:
        """Fit meta-learner directly on pre-computed meta-features.

        Use this with out-of-fold predictions to avoid the target leakage
        introduced by ``fit``: at OOF time the predictions for each sample
        come from a base model that did not see that sample during training,
        so the meta-learner is trained on a clean signal.

        Args:
            meta_features: ``(N, n_models * n_classes)`` array of OOF probas.
            y: ``(N,)`` true labels aligned with ``meta_features``.
        """
        logger.info(
            f"Fitting meta-learner on {meta_features.shape[1]} OOF meta-features "
            f"({meta_features.shape[0]} samples)..."
        )
        self.meta.fit(meta_features, y)
        logger.info("Ensemble (OOF) training complete.")

    def predict(self, x_per_model: list[Any]) -> np.ndarray:
        meta_features = self._get_meta_features(x_per_model)
        return self.meta.predict(meta_features)

    def predict_proba(self, x_per_model: list[Any]) -> np.ndarray:
        meta_features = self._get_meta_features(x_per_model)
        return self.meta.predict_proba(meta_features)

    def get_embeddings(self, x_per_model: list[Any]) -> np.ndarray:
        """Concatenated base model embeddings."""
        embeds = []
        for model, x in zip(self.base_models, x_per_model):
            embeds.append(model.get_embeddings(x))
        return np.hstack(embeds)
