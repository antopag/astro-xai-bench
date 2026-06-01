"""Tabular baselines: Random Forest and XGBoost on extracted features."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from xgboost import XGBClassifier

from src.models import BaseModel


def extract_features(light_curves: np.ndarray) -> np.ndarray:
    """Extract hand-crafted features from multi-band light curves.

    Args:
        light_curves: shape (N, n_bands, max_len, 2) — flux and flux_err.

    Returns:
        Feature matrix of shape (N, n_features).
    """
    n_samples, n_bands = light_curves.shape[0], light_curves.shape[1]
    features_list = []

    for i in range(n_samples):
        feats = []
        for pb in range(n_bands):
            flux = light_curves[i, pb, :, 0]
            flux_err = light_curves[i, pb, :, 1]
            mask = flux != 0.0
            n_obs = mask.sum()

            if n_obs < 2:
                # Pad with zeros if too few observations
                feats.extend([0.0] * 10)
                continue

            f = flux[mask]
            fe = flux_err[mask]

            # Basic statistics
            feats.append(f.mean())
            feats.append(f.std())
            feats.append(f.max() - f.min())       # amplitude
            feats.append(np.median(f))
            feats.append(float(n_obs))              # number of observations

            # Skewness and kurtosis
            if f.std() > 1e-8:
                feats.append(float((((f - f.mean()) / f.std()) ** 3).mean()))  # skew
                feats.append(float((((f - f.mean()) / f.std()) ** 4).mean()))  # kurtosis
            else:
                feats.extend([0.0, 0.0])

            # Mean SNR
            snr = np.abs(f) / (np.abs(fe) + 1e-10)
            feats.append(snr.mean())

            # Linear slope (simple)
            t = np.arange(n_obs, dtype=np.float64)
            if n_obs > 1:
                slope = np.polyfit(t, f.astype(np.float64), 1)[0]
            else:
                slope = 0.0
            feats.append(float(slope))

            # Fraction above mean
            feats.append((f > f.mean()).mean())

        features_list.append(feats)

    features = np.array(features_list, dtype=np.float32)
    # Replace any NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"Extracted features: shape={features.shape}")
    return features


class RFClassifier(BaseModel):
    """Sklearn Random Forest on hand-crafted light curve features."""

    def __init__(self, n_estimators: int = 500, random_state: int = 42, **kwargs: Any) -> None:
        self.model = SklearnRF(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit on feature matrix x and labels y."""
        logger.info(f"Training Random Forest (n_estimators={self.model.n_estimators})...")
        self.model.fit(x, y)
        logger.info(f"Training complete. Train accuracy: {self.model.score(x, y):.4f}")

    def predict(self, x: Any) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: Any) -> np.ndarray:
        return self.model.predict_proba(x)

    def get_embeddings(self, x: Any) -> np.ndarray:
        """Return leaf node indices as embeddings."""
        return self.model.apply(x).astype(np.float32)


class XGBClassifierWrapper(BaseModel):
    """XGBoost on hand-crafted light curve features."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        logger.info(f"Training XGBoost (n_estimators={self.model.n_estimators})...")
        self.model.fit(x, y)
        logger.info("Training complete.")

    def predict(self, x: Any) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: Any) -> np.ndarray:
        return self.model.predict_proba(x)

    def get_embeddings(self, x: Any) -> np.ndarray:
        """Return leaf indices as embeddings."""
        return self.model.apply(x).astype(np.float32)
