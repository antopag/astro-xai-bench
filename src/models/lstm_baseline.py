"""LSTM/GRU baseline for light curve sequence classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import BaseTorchModel


class LSTMClassifier(BaseTorchModel):
    """Bidirectional LSTM on raw multi-band light curve sequences.

    Input shape: (batch, n_bands, seq_len) — flux values across passbands.
    The bands are treated as input features at each timestep.

    Args:
        n_bands: Number of passbands (6 for PLAsTiCC).
        hidden_size: LSTM hidden dimension.
        n_classes: Number of output classes.
        n_layers: Number of LSTM layers.
        dropout: Dropout rate between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        n_bands: int = 6,
        hidden_size: int = 128,
        n_classes: int = 14,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_bands,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        embed_dim = hidden_size * self.n_directions
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, n_classes)
        self._embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_bands, seq_len) — transposed to (batch, seq_len, n_bands) for LSTM.

        Returns:
            Logits of shape (batch, n_classes).
        """
        # (batch, n_bands, seq_len) -> (batch, seq_len, n_bands)
        x = x.transpose(1, 2)

        # Compute mask: timestep is valid if any band is non-zero
        mask = (x.abs().sum(dim=-1) > 0)  # (batch, seq_len)
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch,)

        output, (h_n, _) = self.lstm(x)  # output: (batch, seq_len, hidden*dirs)

        # Use masked mean pooling over valid timesteps
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (output * mask_expanded).sum(dim=1) / lengths.unsqueeze(-1).float()

        pooled = self.dropout(pooled)
        return self.fc(pooled)

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings (before classification head)."""
        x = x.transpose(1, 2)
        mask = (x.abs().sum(dim=-1) > 0)
        lengths = mask.sum(dim=1).clamp(min=1)
        output, _ = self.lstm(x)
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (output * mask_expanded).sum(dim=1) / lengths.unsqueeze(-1).float()
        return pooled

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
