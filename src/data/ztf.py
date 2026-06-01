"""ZTF Bright Transient Survey dataset loading.

ZTF BTS is a smaller, real-world dataset with spectroscopic classifications.
It uses 2 bands (g, r) compared to PLAsTiCC's 6.
Implementation deferred to Phase 1b — PLAsTiCC is the primary benchmark dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


class ZTFDataset(Dataset):
    """PyTorch Dataset for ZTF BTS light curves.

    Args:
        data_dir: Path to processed ZTF data.
        split: One of 'train', 'val', 'test'.
        representation: 'timeseries' or 'gaf'.
        transform: Optional transform.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        representation: str = "timeseries",
        transform: Any | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        data = np.load(data_dir / f"{split}.npz")
        self.light_curves = data["light_curves"]
        self.labels = data["labels"]
        self.representation = representation
        self.transform = transform

        if representation == "gaf":
            from src.data.augmentation import light_curve_to_gaf
            self._gaf_fn = light_curve_to_gaf

        logger.info(f"ZTFDataset [{split}]: {len(self)} samples, repr={representation}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        lc = self.light_curves[idx]
        label = int(self.labels[idx])

        if self.representation == "timeseries":
            x = torch.from_numpy(lc[:, :, 0]).float()
        elif self.representation == "gaf":
            from src.data.augmentation import light_curve_to_multichannel_gaf
            gaf = light_curve_to_multichannel_gaf(lc, image_size=224, method="gasf")
            x = torch.from_numpy(gaf).float()
        else:
            raise ValueError(f"Unknown representation: {self.representation}")

        if self.transform is not None:
            x = self.transform(x)

        return x, label
