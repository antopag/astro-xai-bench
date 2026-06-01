"""Integrated Gradients attribution via Captum.

Implements Sundararajan et al. (2017),
    IG_i(x) = (x_i - x'_i) * \int_0^1 dF(x' + alpha * (x - x')) / dx_i  d alpha
approximated by Captum's Riemann sum (n_steps interpolation points).

This module replaces vanilla |df/dx| saliency for the deep-learning models in
astro-xai-bench. IG satisfies the completeness and sensitivity axioms and is
the recommended baseline xAI method for differentiable models.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

try:
    from captum.attr import IntegratedGradients

    _CAPTUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CAPTUM_AVAILABLE = False
    logger.warning("captum not available — install with `pip install captum`")


def compute_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
    internal_batch_size: int | None = 16,
) -> np.ndarray:
    """Compute Integrated Gradients for a single sample.

    Args:
        model: Trained PyTorch model in eval mode (will be set internally).
        input_tensor: Input sample of shape (1, ...).
        target_class: Class index to attribute. If None, uses argmax of model(x).
        baseline: Reference input. If None, uses a zero tensor of the same shape
            (standard choice for normalised time series and images).
        n_steps: Number of Riemann sum steps for the path integral (default 50).
        internal_batch_size: Mini-batch size used by Captum to evaluate the
            interpolated inputs. Lower values reduce VRAM at the cost of speed.

    Returns:
        Attribution array with shape matching ``input_tensor`` with the batch
        dimension squeezed away (e.g. (6, 256) for time series, (6, 224, 224)
        for GAF images).
    """
    if not _CAPTUM_AVAILABLE:
        raise ImportError(
            "captum is required for Integrated Gradients. Install with `pip install captum`."
        )

    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.to(device)

    if target_class is None:
        with torch.no_grad():
            target_class = int(model(input_tensor).argmax(dim=-1).item())

    ig = IntegratedGradients(model)

    # cuDNN RNN backward is only supported in training mode. Disabling cuDNN
    # for the LSTM lets IG run multiple backward passes while keeping the model
    # in eval mode (so dropout/batchnorm stay deterministic).
    has_rnn = any(isinstance(m, torch.nn.RNNBase) for m in model.modules())
    cudnn_ctx = torch.backends.cudnn.flags(enabled=False) if has_rnn else _NullCtx()

    with cudnn_ctx:
        attributions = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )

    return attributions.detach().cpu().numpy().squeeze(0)


class _NullCtx:
    """No-op context manager (used when cuDNN toggling is unnecessary)."""

    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None
