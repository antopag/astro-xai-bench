"""Train and evaluate deep learning models (LSTM, CNN1D) — Spyder script.

Usage: %runfile run_dl.py --wdir
"""

from __future__ import annotations

# %% Setup paths (resolve relative to this file, not the cwd)
import random
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# %% Config
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda"
N_SAMPLES_EXPLAIN = 50
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
CKPT_DIR = PROJECT_ROOT / "scripts" / "results" / "checkpoints"

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.data.plasticc import CLASS_MAP
from src.evaluation.metrics import compute_metrics, confusion_matrix_plot
from src.training.trainer import Trainer

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device} | project root: {PROJECT_ROOT}")

# %% Load data
data_dir = DATA_DIR
fig_dir = FIG_DIR
fig_dir.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
label_map = np.load(data_dir / "label_map.npy", allow_pickle=True).item()
idx_to_target = {v: k for k, v in label_map.items()}
class_names = [CLASS_MAP.get(idx_to_target[i], str(i)) for i in range(len(label_map))]

train_data = np.load(data_dir / "train.npz")
val_data = np.load(data_dir / "val.npz")
test_data = np.load(data_dir / "test.npz")

# Light curves: (N, 6, 256, 2) -> take flux only -> (N, 6, 256)
lc_train = train_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_val = val_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_test = test_data["light_curves"][:, :, :, 0].astype(np.float32)

y_train = train_data["labels"].astype(np.int64)
y_val = val_data["labels"].astype(np.int64)
y_test = test_data["labels"].astype(np.int64)

logger.info(f"Train: {lc_train.shape}, Val: {lc_val.shape}, Test: {lc_test.shape}")
logger.info(f"Classes: {len(class_names)} — {class_names}")

# %% DataLoaders
train_ds = TensorDataset(torch.from_numpy(lc_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(lc_val), torch.from_numpy(y_val))
test_ds = TensorDataset(torch.from_numpy(lc_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# %% LSTM
# ============================================================
logger.info("=" * 60)
logger.info("Training LSTM...")
logger.info("=" * 60)

from src.models.lstm_baseline import LSTMClassifier

lstm = LSTMClassifier(
    n_bands=6, hidden_size=128, n_classes=len(class_names),
    n_layers=2, dropout=0.3, bidirectional=True,
)

config_lstm = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": LR, "epochs": EPOCHS},
}

trainer_lstm = Trainer(lstm, config_lstm, device=device)
history_lstm = trainer_lstm.fit(train_loader, val_loader)

# %% LSTM evaluation
logger.info("Evaluating LSTM...")
lstm.eval()
y_pred_lstm = []
y_proba_lstm = []
for bx, by in test_loader:
    bx = bx.to(device)
    logits = lstm(bx)
    y_pred_lstm.append(logits.argmax(dim=-1).cpu().numpy())
    y_proba_lstm.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())

y_pred_lstm = np.concatenate(y_pred_lstm)
y_proba_lstm = np.concatenate(y_proba_lstm)

metrics_lstm = compute_metrics(y_test, y_pred_lstm, y_proba_lstm)
print("\nLSTM Test Metrics:")
print(f"  accuracy: {metrics_lstm['accuracy']:.4f}")
print(f"  f1_macro: {metrics_lstm['f1_macro']:.4f}")
print(f"  f1_weighted: {metrics_lstm['f1_weighted']:.4f}")
if "auc_ovr" in metrics_lstm:
    print(f"  auc_ovr: {metrics_lstm['auc_ovr']:.4f}")

# Save checkpoint
trainer_lstm.save_checkpoint(CKPT_DIR / "lstm.pt")

# ============================================================
# %% CNN1D
# ============================================================
logger.info("=" * 60)
logger.info("Training CNN1D...")
logger.info("=" * 60)

from src.models.cnn_baseline import CNN1D

cnn1d = CNN1D(n_bands=6, n_classes=len(class_names), base_filters=64, dropout=0.3)

config_cnn = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": LR, "epochs": EPOCHS},
}

trainer_cnn = Trainer(cnn1d, config_cnn, device=device)
history_cnn = trainer_cnn.fit(train_loader, val_loader)

# %% CNN1D evaluation
logger.info("Evaluating CNN1D...")
cnn1d.eval()
y_pred_cnn = []
y_proba_cnn = []
for bx, by in test_loader:
    bx = bx.to(device)
    logits = cnn1d(bx)
    y_pred_cnn.append(logits.argmax(dim=-1).cpu().numpy())
    y_proba_cnn.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())

y_pred_cnn = np.concatenate(y_pred_cnn)
y_proba_cnn = np.concatenate(y_proba_cnn)

metrics_cnn = compute_metrics(y_test, y_pred_cnn, y_proba_cnn)
print("\nCNN1D Test Metrics:")
print(f"  accuracy: {metrics_cnn['accuracy']:.4f}")
print(f"  f1_macro: {metrics_cnn['f1_macro']:.4f}")
print(f"  f1_weighted: {metrics_cnn['f1_weighted']:.4f}")
if "auc_ovr" in metrics_cnn:
    print(f"  auc_ovr: {metrics_cnn['auc_ovr']:.4f}")

trainer_cnn.save_checkpoint(CKPT_DIR / "cnn1d.pt")

# ============================================================
# %% Grad-CAM on CNN1D
# ============================================================
logger.info("Computing Grad-CAM for CNN1D...")
import matplotlib.pyplot as plt
from src.xai.gradcam import GradCAM, gradcam_overlay_1d

# Target layer: last conv block (before adaptive pool)
# CNN1D.features[8] = Conv1d(128, 256, k=3)
target_layer = cnn1d.features[8]

gradcam = GradCAM(cnn1d, target_layer)

# Show Grad-CAM for a few test samples (one per class, up to 6)
rng = np.random.default_rng(SEED)
shown_classes = set()
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
axes = axes.flatten()
plot_idx = 0

for i in rng.permutation(len(y_test)):
    cls = int(y_test[i])
    if cls in shown_classes:
        continue
    shown_classes.add(cls)

    sample = torch.from_numpy(lc_test[i:i+1]).float().to(device)
    heatmap = gradcam(sample, target_class=cls)

    # Plot the r-band (index 2) with Grad-CAM overlay
    flux_r = lc_test[i, 2, :]
    nonzero = np.nonzero(flux_r)[0]
    if len(nonzero) > 0:
        flux_r = flux_r[:nonzero[-1] + 1]

    gradcam_overlay_1d(heatmap, flux_r, ax=axes[plot_idx],
                       title=f"{class_names[cls]} (r-band)")
    plot_idx += 1
    if plot_idx >= 6:
        break

plt.suptitle("CNN1D Grad-CAM — r-band light curves", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "gradcam_cnn1d.pdf", dpi=300, bbox_inches="tight")
logger.info(f"Grad-CAM plot saved to {fig_dir / 'gradcam_cnn1d.pdf'}")
plt.show()

# ============================================================
# %% Summary comparison
# ============================================================
print("\n" + "=" * 60)
print("DEEP LEARNING METRICS COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'LSTM':>12} {'CNN1D':>12}")
print("-" * 50)
for k in ["accuracy", "f1_macro", "f1_weighted", "auc_ovr"]:
    v_lstm = metrics_lstm.get(k, float("nan"))
    v_cnn = metrics_cnn.get(k, float("nan"))
    print(f"{k:<20} {v_lstm:>12.4f} {v_cnn:>12.4f}")
