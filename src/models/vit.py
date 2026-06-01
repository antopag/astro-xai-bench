"""Vision Transformer (ViT) via timm for GAF image classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import BaseTorchModel


class ViTClassifier(BaseTorchModel):
    """ViT fine-tuned on GAF images of light curves.

    Uses timm pretrained models. The first conv layer is adapted for
    n_bands input channels (instead of 3 RGB).

    Args:
        n_bands: Number of input channels (passbands).
        n_classes: Number of output classes.
        model_name: timm model name.
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout before classification head.
    """

    def __init__(
        self,
        n_bands: int = 6,
        n_classes: int = 14,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=n_bands,
            num_classes=0,  # remove classification head
            drop_rate=dropout,
        )

        self._embed_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self._embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, n_bands, 224, 224) -> logits (batch, n_classes)."""
        h = self.backbone(x)  # (batch, embed_dim)
        h = self.dropout(h)
        return self.fc(h)

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

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
