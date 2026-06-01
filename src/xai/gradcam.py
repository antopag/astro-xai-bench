"""Grad-CAM for CNN and ViT models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class GradCAM:
    """Grad-CAM extractor for any model with convolutional layers.

    Args:
        model: Trained model.
        target_layer: Conv/attention layer to hook into.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap.

        Args:
            input_tensor: Input tensor, shape (1, C, ...).
            target_class: Class index. If None, uses predicted class.

        Returns:
            Heatmap, same spatial dims as target layer output, values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Pool gradients over spatial dims
        if self.gradients is None or self.activations is None:
            logger.warning("No gradients/activations captured")
            return np.zeros((1, 1), dtype=np.float32)

        # For 2D: (1, C, H, W) -> weights (1, C, 1, 1)
        if self.gradients.ndim == 4:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
        # For 1D: (1, C, L) -> weights (1, C, 1)
        elif self.gradients.ndim == 3:
            weights = self.gradients.mean(dim=2, keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
        else:
            cam = np.zeros((1,), dtype=np.float32)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.astype(np.float32)


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: int | None = None,
) -> np.ndarray:
    """Convenience function for single-shot Grad-CAM.

    Args:
        model: Trained model.
        input_tensor: Input image tensor, shape (1, C, H, W) or (1, C, L).
        target_layer: Layer to extract activations from.
        target_class: Class index. If None, uses predicted class.

    Returns:
        Heatmap, values in [0, 1].
    """
    gc = GradCAM(model, target_layer)
    return gc(input_tensor, target_class)


def gradcam_overlay_1d(
    heatmap: np.ndarray,
    light_curve: np.ndarray,
    ax: object | None = None,
    title: str = "",
) -> None:
    """Overlay Grad-CAM heatmap on a 1D light curve plot.

    Args:
        heatmap: 1D heatmap array.
        light_curve: 1D flux array (same or similar length).
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    # Resize heatmap to match light curve length
    if len(heatmap) != len(light_curve):
        heatmap = zoom(heatmap, len(light_curve) / len(heatmap))
    heatmap = np.clip(heatmap, 0.0, 1.0)

    ax.plot(light_curve, color="black", linewidth=0.8)
    ax.fill_between(
        range(len(light_curve)),
        light_curve.min(),
        light_curve.max(),
        alpha=heatmap * 0.5,
        color="red",
    )
    ax.set_title(title)


def gradcam_overlay_2d(
    heatmap: np.ndarray,
    image: np.ndarray,
    ax: object | None = None,
    title: str = "",
) -> None:
    """Overlay Grad-CAM heatmap on a 2D GAF image.

    Args:
        heatmap: 2D heatmap (H', W').
        image: 2D GAF image (H, W).
        ax: Matplotlib axes.
        title: Plot title.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Resize heatmap to image size
    if heatmap.shape != image.shape:
        zoom_factors = (image.shape[0] / heatmap.shape[0], image.shape[1] / heatmap.shape[1])
        heatmap = zoom(heatmap, zoom_factors)

    ax.imshow(image, cmap="viridis", aspect="auto")
    ax.imshow(heatmap, cmap="jet", alpha=0.4, aspect="auto")
    ax.set_title(title)
    ax.axis("off")
