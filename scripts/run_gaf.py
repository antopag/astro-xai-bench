"""Train and evaluate GAF-based models (CNN2D, ViT) — Spyder script.

Converts light curves to Gramian Angular Field images, then trains
CNN2D and ViT classifiers with Grad-CAM and Attention Rollout xAI.

Usage: %runfile run_gaf.py --wdir
"""

from __future__ import annotations

# %% Setup paths (resolve relative to this file, not the cwd)
# Spyder's `%runfile ... --wdir` sets the working directory to the script's
# folder, so naively using "data/processed/plasticc" would land outputs under
# scripts/data/.... Anchor every path to the project root instead.
import random
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# %% Config
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
SEED = 42
BATCH_SIZE = 32
DEVICE = "cuda"
GAF_SIZE = 224
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
CKPT_DIR = PROJECT_ROOT / "scripts" / "results" / "checkpoints"

# CNN2D config
CNN2D_EPOCHS = 100
CNN2D_LR = 1e-3

# ViT config
VIT_EPOCHS = 50
VIT_LR = 1e-4
VIT_WEIGHT_DECAY = 0.01

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.data.plasticc import CLASS_MAP
from src.data.augmentation import light_curve_to_multichannel_gaf
from src.evaluation.metrics import compute_metrics
from src.training.trainer import Trainer

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
fig_dir = FIG_DIR
fig_dir.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Using device: {device} | project root: {PROJECT_ROOT}")

# %% Load data
data_dir = DATA_DIR
label_map = np.load(data_dir / "label_map.npy", allow_pickle=True).item()
idx_to_target = {v: k for k, v in label_map.items()}
class_names = [CLASS_MAP.get(idx_to_target[i], str(i)) for i in range(len(label_map))]

train_data = np.load(data_dir / "train.npz")
val_data = np.load(data_dir / "val.npz")
test_data = np.load(data_dir / "test.npz")

y_train = train_data["labels"].astype(np.int64)
y_val = val_data["labels"].astype(np.int64)
y_test = test_data["labels"].astype(np.int64)

logger.info(f"Samples — Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# %% Convert light curves to GAF images
# This is the slow step — cache result to avoid recomputation
gaf_cache = data_dir / f"gaf_{GAF_SIZE}.npz"

if gaf_cache.exists():
    logger.info(f"Loading cached GAF images from {gaf_cache}")
    gaf_data = np.load(gaf_cache)
    gaf_train = gaf_data["train"]
    gaf_val = gaf_data["val"]
    gaf_test = gaf_data["test"]
else:
    logger.info(f"Computing GAF images (size={GAF_SIZE})... this may take a few minutes")

    def batch_to_gaf(light_curves: np.ndarray, image_size: int) -> np.ndarray:
        """Convert batch of light curves to GAF images."""
        n = len(light_curves)
        result = np.zeros((n, 6, image_size, image_size), dtype=np.float32)
        for i in range(n):
            if i % 500 == 0:
                logger.info(f"  GAF conversion: {i}/{n}")
            result[i] = light_curve_to_multichannel_gaf(light_curves[i], image_size=image_size)
        return result

    gaf_train = batch_to_gaf(train_data["light_curves"], GAF_SIZE)
    gaf_val = batch_to_gaf(val_data["light_curves"], GAF_SIZE)
    gaf_test = batch_to_gaf(test_data["light_curves"], GAF_SIZE)

    logger.info(f"Caching GAF images to {gaf_cache}")
    np.savez_compressed(gaf_cache, train=gaf_train, val=gaf_val, test=gaf_test)

logger.info(f"GAF shapes — Train: {gaf_train.shape}, Val: {gaf_val.shape}, Test: {gaf_test.shape}")

# %% DataLoaders
train_ds = TensorDataset(torch.from_numpy(gaf_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(gaf_val), torch.from_numpy(y_val))
test_ds = TensorDataset(torch.from_numpy(gaf_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# %% CNN2D
# ============================================================
logger.info("=" * 60)
logger.info("Training CNN2D on GAF images...")
logger.info("=" * 60)

from src.models.cnn_baseline import CNN2D

cnn2d = CNN2D(n_bands=6, n_classes=len(class_names), base_filters=32, dropout=0.3)

config_cnn2d = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": CNN2D_LR, "epochs": CNN2D_EPOCHS},
}

trainer_cnn2d = Trainer(cnn2d, config_cnn2d, device=device)
history_cnn2d = trainer_cnn2d.fit(train_loader, val_loader)

# %% CNN2D evaluation
logger.info("Evaluating CNN2D...")
cnn2d.eval()
y_pred_cnn2d, y_proba_cnn2d = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        logits = cnn2d(bx)
        y_pred_cnn2d.append(logits.argmax(dim=-1).cpu().numpy())
        y_proba_cnn2d.append(torch.softmax(logits, dim=-1).cpu().numpy())

y_pred_cnn2d = np.concatenate(y_pred_cnn2d)
y_proba_cnn2d = np.concatenate(y_proba_cnn2d)

metrics_cnn2d = compute_metrics(y_test, y_pred_cnn2d, y_proba_cnn2d)
print("\nCNN2D Test Metrics:")
for k in ["accuracy", "f1_macro", "f1_weighted", "auc_ovr"]:
    print(f"  {k}: {metrics_cnn2d.get(k, float('nan')):.4f}")

trainer_cnn2d.save_checkpoint(CKPT_DIR / "cnn2d.pt")

# %% Grad-CAM for CNN2D
logger.info("Computing Grad-CAM for CNN2D...")
import matplotlib.pyplot as plt
from src.xai.gradcam import GradCAM, gradcam_overlay_2d

# Target: last conv block before adaptive pool
# CNN2D.features[12] = Conv2d(128, 256, k=3)
target_layer_2d = cnn2d.features[12]
gradcam_2d = GradCAM(cnn2d, target_layer_2d)

rng = np.random.default_rng(SEED)
shown_classes = set()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()
plot_idx = 0

for i in rng.permutation(len(y_test)):
    cls = int(y_test[i])
    if cls in shown_classes:
        continue
    shown_classes.add(cls)

    sample = torch.from_numpy(gaf_test[i:i+1]).float().to(device)
    heatmap = gradcam_2d(sample, target_class=cls)

    # Show r-band GAF (index 2) with overlay
    gaf_r = gaf_test[i, 2]  # (224, 224)
    gradcam_overlay_2d(heatmap, gaf_r, ax=axes_flat[plot_idx],
                       title=f"{class_names[cls]}")
    plot_idx += 1
    if plot_idx >= 6:
        break

plt.suptitle("CNN2D Grad-CAM — r-band GAF images", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "gradcam_cnn2d.pdf", dpi=300, bbox_inches="tight")
logger.info(f"Grad-CAM plot saved to {fig_dir / 'gradcam_cnn2d.pdf'}")
plt.show()

# ============================================================
# %% ViT
# ============================================================
logger.info("=" * 60)
logger.info("Training ViT on GAF images...")
logger.info("=" * 60)

from src.models.vit import ViTClassifier

vit = ViTClassifier(
    n_bands=6, n_classes=len(class_names),
    model_name="vit_small_patch16_224", pretrained=True, dropout=0.1,
)

config_vit = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": VIT_LR, "epochs": VIT_EPOCHS, "weight_decay": VIT_WEIGHT_DECAY},
}

trainer_vit = Trainer(vit, config_vit, device=device)
history_vit = trainer_vit.fit(train_loader, val_loader)

# %% ViT evaluation
logger.info("Evaluating ViT...")
vit.eval()
y_pred_vit, y_proba_vit = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        logits = vit(bx)
        y_pred_vit.append(logits.argmax(dim=-1).cpu().numpy())
        y_proba_vit.append(torch.softmax(logits, dim=-1).cpu().numpy())

y_pred_vit = np.concatenate(y_pred_vit)
y_proba_vit = np.concatenate(y_proba_vit)

metrics_vit = compute_metrics(y_test, y_pred_vit, y_proba_vit)
print("\nViT Test Metrics:")
for k in ["accuracy", "f1_macro", "f1_weighted", "auc_ovr"]:
    print(f"  {k}: {metrics_vit.get(k, float('nan')):.4f}")

trainer_vit.save_checkpoint(CKPT_DIR / "vit.pt")

# %% Attention Rollout for ViT
logger.info("Computing Attention Rollout for ViT...")
from src.xai.attention_viz import attention_rollout, plot_attention_rollout

shown_classes = set()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()
plot_idx = 0

for i in rng.permutation(len(y_test)):
    cls = int(y_test[i])
    if cls in shown_classes:
        continue
    shown_classes.add(cls)

    sample = torch.from_numpy(gaf_test[i:i+1]).float().to(device)
    attn_map = attention_rollout(vit, sample)

    gaf_r = gaf_test[i, 2]
    plot_attention_rollout(attn_map, image=gaf_r, ax=axes_flat[plot_idx],
                           title=f"{class_names[cls]}")
    plot_idx += 1
    if plot_idx >= 6:
        break

plt.suptitle("ViT Attention Rollout — r-band GAF images", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "attention_rollout_vit.pdf", dpi=300, bbox_inches="tight")
logger.info(f"Attention rollout saved to {fig_dir / 'attention_rollout_vit.pdf'}")
plt.show()

# ============================================================
# %% Summary comparison
# ============================================================
print("\n" + "=" * 60)
print("GAF MODELS — METRICS COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'CNN2D':>12} {'ViT':>12}")
print("-" * 50)
for k in ["accuracy", "f1_macro", "f1_weighted", "auc_ovr"]:
    v_cnn = metrics_cnn2d.get(k, float("nan"))
    v_vit = metrics_vit.get(k, float("nan"))
    print(f"{k:<20} {v_cnn:>12.4f} {v_vit:>12.4f}")
