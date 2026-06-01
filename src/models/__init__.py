"""Model architectures for transient classification."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Common interface for all models in the benchmark."""

    @abstractmethod
    def predict(self, x: Any) -> np.ndarray:
        """Return predicted class labels.

        Args:
            x: Input data (format depends on model type).

        Returns:
            Array of predicted class indices, shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probability estimates.

        Args:
            x: Input data.

        Returns:
            Array of shape (n_samples, n_classes).
        """

    @abstractmethod
    def get_embeddings(self, x: Any) -> np.ndarray:
        """Return learned feature embeddings.

        Args:
            x: Input data.

        Returns:
            Array of shape (n_samples, embedding_dim).
        """


try:
    import torch

    class BaseTorchModel(BaseModel, torch.nn.Module):
        """Base class for PyTorch-based models."""

        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

except (ImportError, OSError):
    # torch not available or DLL load failure — DL models won't work
    # but tabular models (Avocado, ParSNIP GBT) are fine
    BaseTorchModel = None
