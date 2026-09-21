"""Generic PyTorch training loop."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all RNGs for reproducibility across torch/numpy/random/CUDA.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable cudnn deterministic mode and disable
            benchmark autotuning so that repeated runs with the same seed
            produce identical outputs (at the cost of some throughput).

    Notes:
        Multi-seed benchmarking should call this at the start of each run.
        ``PYTHONHASHSEED`` is also set so subprocess-friendly libraries are
        seeded consistently.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed} (deterministic={deterministic})")


class Trainer:
    """Config-driven trainer for PyTorch models.

    Args:
        model: PyTorch model.
        config: Experiment configuration dict (from YAML).
        device: torch device. Auto-detects if None.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device | None = None,
        seed: int | None = None,
        restore_best: bool = False,
    ) -> None:
        if seed is not None:
            set_global_seed(seed)
        self.seed = seed
        # If True, fit() reloads the weights of the epoch with the lowest
        # validation loss before returning (off by default: the PLAsTiCC
        # models of Paper I were trained without it).
        self.restore_best = restore_best
        self._best_state: dict | None = None
        self._best_val = float("inf")
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        train_cfg = config.get("training", {})
        model_cfg = config.get("_model_config", {})

        lr = model_cfg.get("lr", 1e-3)
        weight_decay = model_cfg.get("weight_decay", 0.0)
        self.epochs = model_cfg.get("epochs", 100)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Loss with class weighting support
        self.criterion = nn.CrossEntropyLoss()

        # Scheduler
        scheduler_type = train_cfg.get("scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        else:
            self.scheduler = None

        # Gradient clipping
        self.grad_clip = train_cfg.get("gradient_clip", 0.0)

        # Early stopping
        patience = train_cfg.get("early_stopping_patience", 10)
        self.early_stopping = EarlyStopping(patience=patience)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, list[float]]:
        """Train the model and return history dict.

        Returns:
            Dict with keys 'train_loss', 'val_loss', 'val_acc' per epoch.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Train
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                train_losses.append(loss.item())

            if self.scheduler is not None:
                self.scheduler.step()

            # Validate
            val_loss, val_acc = self._evaluate_epoch(val_loader)

            train_loss = np.mean(train_losses)
            elapsed = time.time() - t0
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if epoch % 5 == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | lr={lr:.2e} | {elapsed:.1f}s"
                )

            if self.restore_best and val_loss < self._best_val:
                self._best_val = val_loss
                self._best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if self.restore_best and self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info(f"Restored best weights (val_loss={self._best_val:.4f})")
        return history

    @torch.no_grad()
    def _evaluate_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Compute validation loss and accuracy."""
        self.model.eval()
        losses, correct, total = [], 0, 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            losses.append(loss.item())
            correct += (logits.argmax(dim=-1) == batch_y).sum().item()
            total += len(batch_y)
        return np.mean(losses), correct / total if total > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate on test set, return metrics dict."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.device)
            logits = self.model(batch_x)
            probs = torch.softmax(logits, dim=-1)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(batch_y.numpy())
            all_probs.append(probs.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_proba = np.concatenate(all_probs)

        from src.evaluation.metrics import compute_metrics
        return compute_metrics(y_true, y_pred, y_proba)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")
