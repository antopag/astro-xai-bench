"""1D and 2D CNN baselines for transient classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import BaseTorchModel


class CNN1D(BaseTorchModel):
    """1D CNN operating on raw multi-band light curve time series.

    Input shape: (batch, n_bands, seq_len).

    Args:
        n_bands: Number of input channels (passbands).
        n_classes: Number of output classes.
        base_filters: Number of filters in the first conv layer (doubles each block).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_bands: int = 6,
        n_classes: int = 14,
        base_filters: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        f = base_filters

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_bands, f, kernel_size=7, padding=3),
            nn.BatchNorm1d(f),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(f, f * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(f * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(f * 2, f * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(f * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self._embed_dim = f * 4
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self._embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, n_bands, seq_len) -> logits (batch, n_classes)."""
        h = self.features(x).squeeze(-1)  # (batch, embed_dim)
        h = self.dropout(h)
        return self.fc(h)

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: Any) -> np.ndarray:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        logits = self.forward(x.to(next(self.parameters()).device))
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x: Any) -> np.ndarray:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        logits = self.forward(x.to(next(self.parameters()).device))
        return F.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def get_embeddings(self, x: Any) -> np.ndarray:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self._forward_embed(x.to(next(self.parameters()).device)).cpu().numpy()


class CNN2D(BaseTorchModel):
    """2D CNN operating on GAF image representations of light curves.

    Input shape: (batch, n_bands, H, W) — multi-channel GAF images.

    Args:
        n_bands: Number of input channels (passbands as GAF images).
        n_classes: Number of output classes.
        base_filters: Number of filters in the first conv layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_bands: int = 6,
        n_classes: int = 14,
        base_filters: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        f = base_filters

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(n_bands, f, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(f),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(f, f * 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(f * 4, f * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self._embed_dim = f * 8
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self._embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, n_bands, H, W) -> logits (batch, n_classes)."""
        h = self.features(x).flatten(1)  # (batch, embed_dim)
        h = self.dropout(h)
        return self.fc(h)

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    @torch.no_grad()
    def predict(self, x: Any) -> np.ndarray:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        logits = self.forward(x.to(next(self.parameters()).device))
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x: Any) -> np.ndarray:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        logits = self.forward(x.to(next(self.parameters()).device))
        return F.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def get_embeddings(self, x: Any) -> np.ndarray:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self._forward_embed(x.to(next(self.parameters()).device)).cpu().numpy()
