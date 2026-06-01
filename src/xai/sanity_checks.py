"""Saliency sanity checks for the astro-xai-bench benchmark.

Implements the two tests of \\citet{adebayo2018}:

* **Model randomization test** — cascading top-down randomization of the
  network parameters. The Spearman correlation between the attributions
  of the trained model and those of the (progressively) randomized model
  should drop to zero if the attributions actually depend on the learned
  weights.

* **Data randomization test** — retrain the model on a label-permuted
  copy of the training set. The attributions of the random-label model
  should differ from those of the trained model. We report the Spearman
  correlation between the two; values near zero indicate sensitivity to
  the supervision signal.

Tree-based models (Random Forest, XGBoost) only support the data
randomization test, since SHAP attributions on a tree do not have a
well-defined ``model randomization'' analogue.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.stats import spearmanr

from src.xai.integrated_gradients import compute_integrated_gradients


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _topdown_param_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return ``(name, module)`` for every leaf module that owns parameters,
    in **top-down** order (from the output of the network towards the input).

    For sequential / single-branch models the registration order in
    ``named_modules`` is bottom-up, so we reverse it.
    """
    leaves: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        own_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if own_params:
            leaves.append((name, module))
    return list(reversed(leaves))


def _randomize_module(module: nn.Module, std: float = 0.1) -> None:
    """In-place re-initialise every parameter of ``module`` (no recursion)."""
    for p in module.parameters(recurse=False):
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=std)
        else:
            nn.init.zeros_(p)


def _flatten_attrs(attrs_list: list[np.ndarray]) -> np.ndarray:
    """Flatten and concatenate a list of per-sample attribution arrays."""
    return np.concatenate([np.abs(a).flatten() for a in attrs_list])


def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, robust to constant inputs and NaNs."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    rho, _ = spearmanr(a, b)
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


# ----------------------------------------------------------------------------
# Model randomization (DL only)
# ----------------------------------------------------------------------------

def model_randomization_test(
    model: nn.Module,
    x_test: np.ndarray,
    sample_indices: np.ndarray,
    cascading: bool = True,
    n_steps_to_keep: int = 5,
    ig_n_steps: int = 30,
    baseline: torch.Tensor | None = None,
) -> dict[str, float]:
    """Cascading model-randomization sanity check.

    Computes IG attributions for the trained model on a fixed set of
    samples, then progressively randomizes the network top-down and
    recomputes IG. Returns the Spearman correlation between the trained
    attributions and the fully-randomized attributions (final step) plus
    the per-step trace.

    Args:
        model: Trained PyTorch model (in eval mode internally).
        x_test: Test data of shape ``(N, *input_shape)``.
        sample_indices: Indices of the samples to evaluate.
        cascading: If True, randomize layer-by-layer (top-down) and record
            the correlation after each step. If False, randomize the whole
            network at once and record a single value.
        n_steps_to_keep: Maximum number of cascading steps to retain in the
            trace (the actual layer count is downsampled to this number to
            keep runtime bounded). Ignored if ``cascading`` is False.
        ig_n_steps: Number of Riemann steps for IG (kept low for speed).

    Returns:
        ``{
            "spearman_full": float,  # correlation after full randomization
            "trace": list[(step_name, spearman)],
        }``
    """
    device = next(model.parameters()).device
    model.eval()

    # 1. Trained-model attributions
    trained_attrs: list[np.ndarray] = []
    for idx in sample_indices:
        x = torch.from_numpy(x_test[idx:idx + 1]).float().to(device)
        attr = compute_integrated_gradients(model, x, n_steps=ig_n_steps, baseline=baseline)
        trained_attrs.append(attr)
    trained_flat = _flatten_attrs(trained_attrs)

    # 2. Cascading randomization
    rand_model = copy.deepcopy(model).to(device)
    layers_topdown = _topdown_param_layers(rand_model)

    if cascading:
        if n_steps_to_keep > 0 and len(layers_topdown) > n_steps_to_keep:
            stride = max(1, len(layers_topdown) // n_steps_to_keep)
            checkpoints = list(range(0, len(layers_topdown), stride))[:n_steps_to_keep]
            if (len(layers_topdown) - 1) not in checkpoints:
                checkpoints.append(len(layers_topdown) - 1)
        else:
            checkpoints = list(range(len(layers_topdown)))
    else:
        checkpoints = [len(layers_topdown) - 1]

    trace: list[tuple[str, float]] = []
    for step_idx, (name, layer) in enumerate(layers_topdown):
        _randomize_module(layer)

        if step_idx in checkpoints:
            rand_attrs: list[np.ndarray] = []
            for idx in sample_indices:
                x = torch.from_numpy(x_test[idx:idx + 1]).float().to(device)
                try:
                    attr = compute_integrated_gradients(rand_model, x, n_steps=ig_n_steps, baseline=baseline)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"IG failed on randomized model at step {step_idx} ({name}): {exc}"
                    )
                    rand_attrs.append(np.zeros_like(trained_attrs[len(rand_attrs)]))
                    continue
                rand_attrs.append(attr)
            rho = _spearman_safe(trained_flat, _flatten_attrs(rand_attrs))
            trace.append((name, rho))
            logger.debug(f"  cascading step {step_idx} [{name}]: rho={rho:.4f}")

    spearman_full = trace[-1][1] if trace else float("nan")
    logger.info(
        f"Model randomization Spearman (final, full): {spearman_full:.4f} "
        f"({len(trace)} cascading steps logged)"
    )
    return {"spearman_full": float(spearman_full), "trace": trace}


# ----------------------------------------------------------------------------
# Data randomization (all models)
# ----------------------------------------------------------------------------

def shuffle_labels(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Permute the labels in place-safe fashion (returns a new array)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(y).astype(y.dtype)


def data_randomization_spearman_dl(
    model_trained: nn.Module,
    model_random_label: nn.Module,
    x_test: np.ndarray,
    sample_indices: np.ndarray,
    ig_n_steps: int = 30,
    baseline: torch.Tensor | None = None,
) -> float:
    """Spearman correlation between trained and random-label DL attributions.

    Both models are assumed to be already trained (the random-label one
    on permuted labels). The IG attributions are computed for the same
    sample indices in both cases and compared via Spearman.
    """
    device = next(model_trained.parameters()).device

    trained_attrs: list[np.ndarray] = []
    random_attrs: list[np.ndarray] = []
    for idx in sample_indices:
        x = torch.from_numpy(x_test[idx:idx + 1]).float().to(device)
        trained_attrs.append(compute_integrated_gradients(model_trained, x, n_steps=ig_n_steps, baseline=baseline))
        random_attrs.append(compute_integrated_gradients(model_random_label, x, n_steps=ig_n_steps, baseline=baseline))

    rho = _spearman_safe(_flatten_attrs(trained_attrs), _flatten_attrs(random_attrs))
    logger.info(f"Data randomization Spearman (DL): {rho:.4f}")
    return rho


def data_randomization_spearman_tabular(
    shap_trained: np.ndarray | list,
    shap_random_label: np.ndarray | list,
) -> float:
    """Spearman correlation between trained and random-label SHAP attributions.

    The SHAP arrays may be (N, F), list of (N, F), or (N, C, F) / (N, F, C).
    We reduce each to a per-sample feature importance vector by averaging
    ``|.|`` over classes, then flatten and correlate.
    """
    def _reduce(sv: np.ndarray | list) -> np.ndarray:
        if isinstance(sv, list):
            return np.mean([np.abs(s) for s in sv], axis=0)
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            return np.abs(sv).mean(axis=1)
        return np.abs(sv)

    a = _reduce(shap_trained).flatten()
    b = _reduce(shap_random_label).flatten()
    rho = _spearman_safe(a, b)
    logger.info(f"Data randomization Spearman (tabular): {rho:.4f}")
    return rho
