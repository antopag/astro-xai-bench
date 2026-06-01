"""Attention rollout visualization for Vision Transformers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.ndimage import zoom


def _get_attention_maps(model: nn.Module, input_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Extract attention maps from all transformer blocks via hooks.

    Args:
        model: ViT model (timm-based).
        input_tensor: (1, C, H, W).

    Returns:
        List of attention weight tensors, each shape (n_heads, n_tokens, n_tokens).
    """
    attention_maps: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
        # timm ViT attention modules output (attn_output, attn_weights) when configured,
        # but by default only attn_output. We hook the softmax directly.
        pass

    # For timm ViT, attention weights are computed inside Attention.forward.
    # We monkey-patch the forward to capture them.
    hooks = []
    original_forwards = {}

    for name, module in model.named_modules():
        if hasattr(module, "attn_drop") and isinstance(module, nn.Module):
            # This is likely a timm Attention block
            original_forward = module.forward
            original_forwards[name] = original_forward

            def make_hook(mod, orig_fwd):
                def hooked_forward(x, **kwargs):
                    B, N, C = x.shape
                    qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    attn = (q @ k.transpose(-2, -1)) * mod.scale
                    attn = attn.softmax(dim=-1)
                    attention_maps.append(attn.detach().cpu())
                    attn = mod.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = mod.proj(x)
                    x = mod.proj_drop(x)
                    return x
                return hooked_forward

            module.forward = make_hook(module, original_forward)

    with torch.no_grad():
        model(input_tensor)

    # Restore original forwards
    for name, module in model.named_modules():
        if name in original_forwards:
            module.forward = original_forwards[name]

    return attention_maps


def attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """Compute attention rollout map for a ViT model.

    Args:
        model: ViT model (timm-based).
        input_tensor: Input image, shape (1, C, H, W).
        head_fusion: How to combine attention heads ('mean', 'max', 'min').
        discard_ratio: Fraction of lowest attention values to discard.

    Returns:
        Attention map of shape (H, W), values in [0, 1].
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    attention_maps = _get_attention_maps(model, input_tensor)

    if len(attention_maps) == 0:
        logger.warning("No attention maps captured — model may not be a standard timm ViT")
        h = w = int(np.sqrt(input_tensor.shape[-1] * input_tensor.shape[-2]) // 16)
        return np.zeros((h, w), dtype=np.float32)

    result = None
    for attn in attention_maps:
        # attn shape: (1, n_heads, n_tokens, n_tokens)
        attn = attn.squeeze(0)  # (n_heads, n_tokens, n_tokens)

        # Fuse heads
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=0)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=0).values
        elif head_fusion == "min":
            attn_fused = attn.min(dim=0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Discard low-attention values
        flat = attn_fused.flatten()
        threshold = flat.quantile(discard_ratio)
        attn_fused = torch.where(attn_fused > threshold, attn_fused, torch.zeros_like(attn_fused))

        # Add identity (residual connections)
        I = torch.eye(attn_fused.shape[0])
        attn_fused = (attn_fused + I) / 2.0

        # Normalize rows
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn_fused
        else:
            result = attn_fused @ result

    # Take CLS token attention to patch tokens
    mask = result[0, 1:]  # exclude CLS token itself
    n_patches = len(mask)
    h = w = int(np.sqrt(n_patches))

    if h * w != n_patches:
        logger.warning(f"Non-square patch grid: {n_patches} patches")
        h = w = int(np.ceil(np.sqrt(n_patches)))
        mask = torch.nn.functional.pad(mask, (0, h * w - n_patches))

    mask = mask.reshape(h, w).numpy()

    # Normalize to [0, 1]
    if mask.max() > 0:
        mask = mask / mask.max()

    return mask.astype(np.float32)


def plot_attention_rollout(
    attn_map: np.ndarray,
    image: np.ndarray | None = None,
    image_size: int = 224,
    ax: object | None = None,
    title: str = "Attention Rollout",
    save_path: str | None = None,
) -> None:
    """Plot attention rollout map, optionally overlaid on a GAF image.

    Args:
        attn_map: Attention map from attention_rollout(), shape (h, w).
        image: Optional GAF image to overlay on, shape (H, W).
        image_size: Target display size.
        ax: Matplotlib axes.
        title: Plot title.
        save_path: Save path.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Upscale attention map to image size
    if attn_map.shape[0] != image_size:
        scale = image_size / attn_map.shape[0]
        attn_map = zoom(attn_map, scale)

    if image is not None:
        ax.imshow(image, cmap="viridis", aspect="auto")
        ax.imshow(attn_map, cmap="jet", alpha=0.4, aspect="auto")
    else:
        ax.imshow(attn_map, cmap="jet", aspect="auto")

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Attention rollout saved to {save_path}")
