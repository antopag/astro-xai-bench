"""Data augmentation strategies for light curves and GAF image generation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def add_gaussian_noise(lc: np.ndarray, sigma: float = 0.05, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to a light curve.

    Args:
        lc: Light curve array of shape (n_timesteps,).
        sigma: Standard deviation of noise (relative to LC std).
        seed: Random seed.

    Returns:
        Noisy light curve (same shape).
    """
    rng = np.random.default_rng(seed)
    noise_scale = sigma * np.std(lc) if np.std(lc) > 1e-8 else sigma
    return lc + rng.normal(0, noise_scale, size=lc.shape).astype(lc.dtype)


def time_warp(lc: np.ndarray, sigma: float = 0.2, seed: int = 42) -> np.ndarray:
    """Apply random time warping to a light curve via cubic interpolation.

    Args:
        lc: Light curve array of shape (n_timesteps,).
        sigma: Strength of warping.
        seed: Random seed.

    Returns:
        Time-warped light curve (same shape).
    """
    rng = np.random.default_rng(seed)
    n = len(lc)
    if n < 4:
        return lc.copy()

    orig_steps = np.arange(n, dtype=np.float64)
    # Random cumulative warp
    warp = rng.normal(1.0, sigma, size=n)
    warp = np.clip(warp, 0.5, 1.5)
    warped_steps = np.cumsum(warp)
    warped_steps = warped_steps / warped_steps[-1] * (n - 1)  # rescale to original range

    f = interp1d(orig_steps, lc, kind="cubic", fill_value="extrapolate")
    return f(warped_steps).astype(lc.dtype)


def _rescale_to_minus1_plus1(x: np.ndarray) -> np.ndarray:
    """Rescale array to [-1, 1] range for GAF computation."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        return np.zeros_like(x)
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


def light_curve_to_gaf(
    lc: np.ndarray,
    image_size: int = 224,
    method: str = "gasf",
) -> np.ndarray:
    """Convert a 1D light curve to a Gramian Angular Field image.

    Implements PAA (Piecewise Aggregate Approximation) for downsampling,
    then computes the Gramian Angular Summation/Difference Field.

    Args:
        lc: Light curve array of shape (n_timesteps,).
        image_size: Output image size (square).
        method: 'gasf' (Gramian Angular Summation Field) or
                'gadf' (Gramian Angular Difference Field).

    Returns:
        GAF image of shape (image_size, image_size), values in [-1, 1].
    """
    # Remove NaN/inf
    lc = np.nan_to_num(lc, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(lc)
    if n == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    # PAA: reduce to image_size points
    if n >= image_size:
        # Piecewise aggregate approximation
        indices = np.array_split(np.arange(n), image_size)
        paa = np.array([lc[idx].mean() for idx in indices])
    else:
        # Upsample via linear interpolation
        x_orig = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, image_size)
        paa = np.interp(x_new, x_orig, lc)

    # Rescale to [-1, 1] and compute angular representation
    scaled = _rescale_to_minus1_plus1(paa)
    # Clip to valid arccos range
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)  # angular representation

    if method == "gasf":
        # Gram matrix: cos(phi_i + phi_j)
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif method == "gadf":
        # Gram matrix: sin(phi_i - phi_j)
        gaf = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError(f"method must be 'gasf' or 'gadf', got '{method}'")

    return gaf.astype(np.float32)


def light_curve_to_multichannel_gaf(
    lc_multiband: np.ndarray,
    image_size: int = 224,
    method: str = "gasf",
) -> np.ndarray:
    """Convert multi-band light curve to multi-channel GAF image.

    Args:
        lc_multiband: Array of shape (n_bands, n_timesteps) or (n_bands, n_timesteps, 2)
                      where channel 0 is flux and channel 1 is flux_err.
        image_size: Output image size.
        method: 'gasf' or 'gadf'.

    Returns:
        GAF image of shape (n_bands, image_size, image_size).
    """
    if lc_multiband.ndim == 3:
        lc_multiband = lc_multiband[:, :, 0]  # take flux only

    n_bands = lc_multiband.shape[0]
    gaf_stack = np.zeros((n_bands, image_size, image_size), dtype=np.float32)

    for b in range(n_bands):
        flux = lc_multiband[b]
        # Trim zero-padding
        nonzero = np.nonzero(flux)[0]
        if len(nonzero) > 2:
            flux = flux[: nonzero[-1] + 1]
        elif len(nonzero) == 0:
            continue
        gaf_stack[b] = light_curve_to_gaf(flux, image_size=image_size, method=method)

    return gaf_stack
