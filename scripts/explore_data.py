"""Data exploration script for Spyder — PLAsTiCC dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.augmentation import light_curve_to_gaf
from src.data.plasticc import CLASS_MAP, PASSBAND_NAMES

# %% Load data
# Anchor to project root (parent.parent), NOT to scripts/ — Spyder's
# `%runfile --wdir` would otherwise land us in scripts/data/...
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "plasticc"
train = np.load(DATA_DIR / "train.npz")
label_map = np.load(DATA_DIR / "label_map.npy", allow_pickle=True).item()
idx_to_class = {v: k for k, v in label_map.items()}

lc = train["light_curves"]  # (N, 6, 256, 2)
labels = train["labels"]

print(f"Light curves shape: {lc.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Classes: {sorted(label_map.keys())}")

# %% Class distribution
unique, counts = np.unique(labels, return_counts=True)
class_names = [CLASS_MAP.get(idx_to_class[u], str(u)) for u in unique]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(class_names, counts, color="steelblue")
ax.set_xlabel("Transient class")
ax.set_ylabel("Count")
ax.set_title("PLAsTiCC Training Set – Class Distribution")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% Example light curves (one per class, all 6 bands)
COLORS = ["purple", "green", "red", "orange", "brown", "black"]

fig, axes = plt.subplots(2, 7, figsize=(20, 7))
for ax_idx, cls_idx in enumerate(unique):
    ax = axes.flat[ax_idx]
    sample_idx = np.where(labels == cls_idx)[0][0]
    for pb in range(6):
        flux = lc[sample_idx, pb, :, 0]
        nonzero = np.nonzero(flux)[0]
        if len(nonzero) > 0:
            ax.plot(flux[: nonzero[-1] + 1], color=COLORS[pb], label=PASSBAND_NAMES[pb], alpha=0.7, linewidth=0.8)
    ax.set_title(CLASS_MAP.get(idx_to_class[cls_idx], str(cls_idx)), fontsize=9)
    if ax_idx == 0:
        ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
plt.suptitle("Example Light Curves (all 6 bands)", fontsize=13)
plt.tight_layout()
plt.show()

# %% GAF image examples (first sample, all 6 bands)
sample_idx = 0
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
for pb in range(6):
    flux = lc[sample_idx, pb, :, 0]
    nonzero = np.nonzero(flux)[0]
    flux_trim = flux[: nonzero[-1] + 1] if len(nonzero) > 2 else flux[:10]
    gaf = light_curve_to_gaf(flux_trim, image_size=64, method="gasf")
    axes[pb].imshow(gaf, cmap="viridis", aspect="auto")
    axes[pb].set_title(f"Band {PASSBAND_NAMES[pb]}", fontsize=10)
    axes[pb].axis("off")
cls_name = CLASS_MAP.get(idx_to_class[labels[sample_idx]], "?")
plt.suptitle(f"GASF images – class {cls_name}", fontsize=13)
plt.tight_layout()
plt.show()

# %% Light curve length statistics
print("\n--- Light curve length per band (non-zero timesteps) ---")
for pb in range(6):
    lengths = np.array([(lc[i, pb, :, 0] != 0).sum() for i in range(len(lc))])
    print(f"  Band {PASSBAND_NAMES[pb]}: mean={lengths.mean():.1f}, median={np.median(lengths):.0f}, "
          f"min={lengths.min()}, max={lengths.max()}")

# %% DataLoader test
from torch.utils.data import DataLoader
from src.data.plasticc import PLAsTiCCDataset

ds = PLAsTiCCDataset(DATA_DIR, split="train", representation="timeseries")
x, y = ds[0]
print(f"\nTimeseries sample: x.shape={x.shape}, y={y}")

dl = DataLoader(ds, batch_size=16, shuffle=True)
batch_x, batch_y = next(iter(dl))
print(f"Batch: x.shape={batch_x.shape}, y.shape={batch_y.shape}")
