"""Revision experiments requested by reviewer.

Four lightweight analyses:
  1. Plausibility threshold sensitivity (sweep percentile in IoU)
  2. Signed vs absolute Spearman for sanity checks
  3. Tree random-label training accuracy (memorisation check)
  4. Gini sparsity as alternative complexity metric

Usage in Spyder: %runfile run_revision_experiments.py --wdir
"""

from __future__ import annotations

# %% Setup
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
SEED = 42
N_SAMPLES_SANITY = 20
DEVICE = "cuda"

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# %% Load data
train_data = np.load(DATA_DIR / "train.npz")
val_data = np.load(DATA_DIR / "val.npz")
test_data = np.load(DATA_DIR / "test.npz")

y_train = train_data["labels"]
y_val = val_data["labels"]
y_test = test_data["labels"]
n_classes = int(y_train.max()) + 1

lc_train = train_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_val = val_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_test = test_data["light_curves"][:, :, :, 0].astype(np.float32)

from src.data.plasticc import CLASS_MAP
from src.models.tabular_rf import extract_features

label_map = np.load(DATA_DIR / "label_map.npy", allow_pickle=True).item()

x_train_tab = extract_features(np.vstack([train_data["light_curves"], val_data["light_curves"]]))
y_train_tab = np.concatenate([y_train, y_val])
x_test_tab = extract_features(test_data["light_curves"])

# ============================================================
# %% EXPERIMENT 1: Plausibility threshold sensitivity
# ============================================================
logger.info("=" * 60)
logger.info("EXPERIMENT 1: Plausibility threshold sensitivity")
logger.info("=" * 60)

from src.xai.metrics import build_expert_mask_tabular
from src.xai.shap_analysis import get_feature_names

feature_names = get_feature_names()
expert_mask = build_expert_mask_tabular(feature_names)


def iou_with_threshold(attr: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    """IoU with a configurable percentile threshold instead of fixed median."""
    attr_abs = np.abs(attr)
    positive = attr_abs[attr_abs > 0]
    if len(positive) == 0:
        return 0.0
    thresh = float(np.percentile(positive, percentile))
    attr_bin = (attr_abs >= thresh).astype(float)
    mask_bin = (mask > 0).astype(float)
    inter = (attr_bin * mask_bin).sum()
    union = np.clip(attr_bin + mask_bin, 0, 1).sum()
    return float(inter / union) if union > 1e-10 else 0.0


# Load SHAP values for RF and XGBoost (seed 42)
from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper
from src.xai.shap_analysis import compute_shap_values

rf = RFClassifier(n_estimators=500, random_state=SEED)
rf.fit(x_train_tab, y_train_tab)
shap_rf = compute_shap_values(rf, x_test_tab, method="tree")

xgb = XGBClassifierWrapper(n_estimators=500, random_state=SEED)
xgb.fit(x_train_tab, y_train_tab)
shap_xgb = compute_shap_values(xgb, x_test_tab, method="tree")


def reduce_shap(sv, n_features):
    """Reduce multi-class SHAP to per-sample (N, F) mean |SHAP|."""
    if isinstance(sv, list):
        return np.mean([np.abs(s) for s in sv], axis=0)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        if sv.shape[2] == n_features:
            return np.abs(sv).mean(axis=1)
        return np.abs(sv).mean(axis=2)
    return np.abs(sv)


shap_rf_red = reduce_shap(shap_rf, 60)
shap_xgb_red = reduce_shap(shap_xgb, 60)

# Also load DL attributions via IG for timeseries + image models
from src.xai.integrated_gradients import compute_integrated_gradients
from src.xai.plausibility import (
    project_timeseries_attribution,
    project_image_attribution,
)

rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(y_test), size=50, replace=False)

PERCENTILES = [25, 50, 75, 90]
plaus_sensitivity = {}

# Tabular models
for model_name, shap_red in [("RF", shap_rf_red), ("XGBoost", shap_xgb_red)]:
    row = {}
    for pct in PERCENTILES:
        scores = [iou_with_threshold(shap_red[i], expert_mask, pct) for i in sample_idx]
        row[f"p{pct}"] = float(np.mean(scores))
    plaus_sensitivity[model_name] = row
    logger.info(f"  {model_name}: {row}")

# DL models — load checkpoints, compute IG, project, sweep thresholds
ts_baseline_np = lc_train.mean(axis=0).astype(np.float32)
ts_baseline = torch.from_numpy(ts_baseline_np[None]).to(device)


def load_dl_model(factory, ckpt_path):
    model = factory().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


seed_dir = RESULTS_DIR / f"seed_{SEED}" / "checkpoints"

# LSTM
from src.models.lstm_baseline import LSTMClassifier

lstm = load_dl_model(
    lambda: LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                           n_layers=2, dropout=0.3, bidirectional=True),
    seed_dir / "lstm.pt",
)
lstm_proj = []
for idx in sample_idx:
    x = torch.from_numpy(lc_test[idx:idx + 1]).float().to(device)
    attr = compute_integrated_gradients(lstm, x, n_steps=30, baseline=ts_baseline)
    lstm_proj.append(project_timeseries_attribution(attr, lc_test[idx]))
lstm_proj = np.array(lstm_proj)

row = {}
for pct in PERCENTILES:
    scores = [iou_with_threshold(lstm_proj[i], expert_mask, pct) for i in range(len(lstm_proj))]
    row[f"p{pct}"] = float(np.mean(scores))
plaus_sensitivity["LSTM"] = row
logger.info(f"  LSTM: {row}")
del lstm; torch.cuda.empty_cache()

# CNN1D
from src.models.cnn_baseline import CNN1D

cnn1d = load_dl_model(
    lambda: CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3),
    seed_dir / "cnn1d.pt",
)
cnn1d_proj = []
for idx in sample_idx:
    x = torch.from_numpy(lc_test[idx:idx + 1]).float().to(device)
    attr = compute_integrated_gradients(cnn1d, x, n_steps=30, baseline=ts_baseline)
    cnn1d_proj.append(project_timeseries_attribution(attr, lc_test[idx]))
cnn1d_proj = np.array(cnn1d_proj)

row = {}
for pct in PERCENTILES:
    scores = [iou_with_threshold(cnn1d_proj[i], expert_mask, pct) for i in range(len(cnn1d_proj))]
    row[f"p{pct}"] = float(np.mean(scores))
plaus_sensitivity["CNN1D"] = row
logger.info(f"  CNN1D: {row}")
del cnn1d; torch.cuda.empty_cache()

# CNN2D
from src.models.cnn_baseline import CNN2D as CNN2DClassifier

GAF_SIZE = 224
gaf_cache = DATA_DIR / f"gaf_{GAF_SIZE}.npz"
if gaf_cache.exists():
    logger.info(f"Loading cached GAF from {gaf_cache}")
    gaf_test = np.load(gaf_cache)["test"].astype(np.float32)
else:
    from src.data.augmentation import light_curve_to_multichannel_gaf
    logger.info("Computing GAF images (no cache found)...")
    gaf_test = np.zeros((len(y_test), 6, GAF_SIZE, GAF_SIZE), dtype=np.float32)
    for i in range(len(y_test)):
        gaf_test[i] = light_curve_to_multichannel_gaf(test_data["light_curves"][i], image_size=GAF_SIZE)

cnn2d = load_dl_model(
    lambda: CNN2DClassifier(n_bands=6, n_classes=n_classes),
    seed_dir / "cnn2d.pt",
)
gaf_baseline = torch.from_numpy(gaf_test.mean(axis=0, keepdims=True).astype(np.float32)).to(device)
cnn2d_proj = []
for idx in sample_idx:
    x = torch.from_numpy(gaf_test[idx:idx + 1].astype(np.float32)).to(device)
    attr = compute_integrated_gradients(cnn2d, x, n_steps=30, baseline=gaf_baseline)
    cnn2d_proj.append(project_image_attribution(attr))
cnn2d_proj = np.array(cnn2d_proj)

row = {}
for pct in PERCENTILES:
    scores = [iou_with_threshold(cnn2d_proj[i], expert_mask, pct) for i in range(len(cnn2d_proj))]
    row[f"p{pct}"] = float(np.mean(scores))
plaus_sensitivity["CNN2D"] = row
logger.info(f"  CNN2D: {row}")
del cnn2d; torch.cuda.empty_cache()

# ViT
from src.models.vit import ViTClassifier

vit = load_dl_model(
    lambda: ViTClassifier(n_bands=6, n_classes=n_classes),
    seed_dir / "vit.pt",
)
vit_proj = []
for idx in sample_idx:
    x = torch.from_numpy(gaf_test[idx:idx + 1].astype(np.float32)).to(device)
    attr = compute_integrated_gradients(vit, x, n_steps=30, baseline=gaf_baseline)
    vit_proj.append(project_image_attribution(attr))
vit_proj = np.array(vit_proj)

row = {}
for pct in PERCENTILES:
    scores = [iou_with_threshold(vit_proj[i], expert_mask, pct) for i in range(len(vit_proj))]
    row[f"p{pct}"] = float(np.mean(scores))
plaus_sensitivity["ViT"] = row
logger.info(f"  ViT: {row}")
del vit; torch.cuda.empty_cache()

print("\n=== Plausibility Threshold Sensitivity ===")
print(f"{'Model':<12}", end="")
for pct in PERCENTILES:
    print(f"  {'p'+str(pct):>8}", end="")
print()
for m in ["RF", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT"]:
    print(f"{m:<12}", end="")
    for pct in PERCENTILES:
        print(f"  {plaus_sensitivity[m][f'p{pct}']:>8.3f}", end="")
    print()

# ============================================================
# %% EXPERIMENT 2: Signed vs absolute Spearman in sanity checks
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("EXPERIMENT 2: Signed vs absolute Spearman")
logger.info("=" * 60)

from scipy.stats import spearmanr
from src.xai.sanity_checks import shuffle_labels
from src.training.trainer import Trainer, set_global_seed
from torch.utils.data import DataLoader, TensorDataset

set_global_seed(SEED)
rng2 = np.random.default_rng(SEED + 1)
sanity_idx = rng2.choice(len(y_test), size=N_SAMPLES_SANITY, replace=False)

ts_baseline_t = torch.from_numpy(ts_baseline_np[None]).to(device)

# We need: trained model IG attrs vs fully-randomized model IG attrs
# For both signed and absolute flattened vectors
import copy
from src.xai.sanity_checks import _topdown_param_layers, _randomize_module


def compute_ig_batch(model, x_data, indices, baseline=None):
    """Compute IG for a batch of indices, return list of numpy arrays."""
    attrs = []
    for idx in indices:
        x = torch.from_numpy(x_data[idx:idx + 1]).float().to(device)
        attr = compute_integrated_gradients(model, x, n_steps=20, baseline=baseline)
        attrs.append(attr)
    return attrs


def signed_abs_spearman(attrs_a, attrs_b):
    """Return (rho_signed, rho_absolute) from two lists of attribution arrays."""
    signed_a = np.concatenate([a.flatten() for a in attrs_a])
    signed_b = np.concatenate([a.flatten() for a in attrs_b])
    abs_a = np.abs(signed_a)
    abs_b = np.abs(signed_b)

    def safe_spearman(x, y):
        if x.std() < 1e-12 or y.std() < 1e-12:
            return 0.0
        rho, _ = spearmanr(x, y)
        return 0.0 if (rho is None or np.isnan(rho)) else float(rho)

    return safe_spearman(signed_a, signed_b), safe_spearman(abs_a, abs_b)


signed_results = {}

# --- Model randomization: DL models ---
for name, factory, ckpt_name in [
    ("LSTM", lambda: LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                                     n_layers=2, dropout=0.3, bidirectional=True), "lstm.pt"),
    ("CNN1D", lambda: CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3), "cnn1d.pt"),
    ("CNN2D", lambda: CNN2DClassifier(n_bands=6, n_classes=n_classes), "cnn2d.pt"),
    ("ViT", lambda: ViTClassifier(n_bands=6, n_classes=n_classes), "vit.pt"),
]:
    ckpt_path = seed_dir / ckpt_name
    if not ckpt_path.exists():
        logger.warning(f"  {name}: checkpoint missing, skipping")
        continue

    model = load_dl_model(factory, ckpt_path)

    # Choose input data and baseline based on model type
    if name in ("LSTM", "CNN1D"):
        x_data = lc_test
        bl = ts_baseline_t
    else:
        x_data = gaf_test.astype(np.float32)
        bl = gaf_baseline

    # Trained attrs
    trained_attrs = compute_ig_batch(model, x_data, sanity_idx, baseline=bl)

    # Fully randomized model
    rand_model = copy.deepcopy(model).to(device)
    for _, layer in _topdown_param_layers(rand_model):
        _randomize_module(layer)
    rand_attrs = compute_ig_batch(rand_model, x_data, sanity_idx, baseline=bl)

    rho_signed, rho_abs = signed_abs_spearman(trained_attrs, rand_attrs)
    signed_results[name] = {
        "model_rand_signed": rho_signed,
        "model_rand_abs": rho_abs,
    }
    logger.info(f"  {name} model-rand: signed={rho_signed:.4f}, abs={rho_abs:.4f}")
    del model, rand_model; torch.cuda.empty_cache()

# --- Data randomization: retrain on shuffled labels (LSTM only, as representative) ---
logger.info("--- Data randomization: LSTM (signed vs abs) ---")
y_train_shuf = shuffle_labels(y_train, seed=SEED)
y_val_shuf = shuffle_labels(y_val, seed=SEED + 1)
gen = torch.Generator(); gen.manual_seed(SEED)
ts_train_ds = TensorDataset(torch.from_numpy(lc_train), torch.from_numpy(y_train_shuf.astype(np.int64)))
ts_val_ds = TensorDataset(torch.from_numpy(lc_val), torch.from_numpy(y_val_shuf.astype(np.int64)))
ts_train_loader = DataLoader(ts_train_ds, batch_size=64, shuffle=True, generator=gen)
ts_val_loader = DataLoader(ts_val_ds, batch_size=64, shuffle=False)

model_t = load_dl_model(
    lambda: LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                           n_layers=2, dropout=0.3, bidirectional=True),
    seed_dir / "lstm.pt",
)
model_r = LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                          n_layers=2, dropout=0.3, bidirectional=True)
config_ts = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": 1e-3, "epochs": 100},
}
trainer = Trainer(model_r, config_ts, device=device, seed=SEED)
trainer.fit(ts_train_loader, ts_val_loader)

trained_attrs = compute_ig_batch(model_t, lc_test, sanity_idx, baseline=ts_baseline_t)
random_attrs = compute_ig_batch(model_r, lc_test, sanity_idx, baseline=ts_baseline_t)
rho_signed, rho_abs = signed_abs_spearman(trained_attrs, random_attrs)
signed_results["LSTM"]["data_rand_signed"] = rho_signed
signed_results["LSTM"]["data_rand_abs"] = rho_abs
logger.info(f"  LSTM data-rand: signed={rho_signed:.4f}, abs={rho_abs:.4f}")
del model_t, model_r; torch.cuda.empty_cache()

print("\n=== Signed vs Absolute Spearman ===")
print(f"{'Model':<10} {'ModelRand(sgn)':>14} {'ModelRand(abs)':>14} {'DataRand(sgn)':>14} {'DataRand(abs)':>14}")
for m in ["LSTM", "CNN1D", "CNN2D", "ViT"]:
    if m not in signed_results:
        continue
    r = signed_results[m]
    mr_s = f"{r.get('model_rand_signed', float('nan')):.4f}"
    mr_a = f"{r.get('model_rand_abs', float('nan')):.4f}"
    dr_s = f"{r.get('data_rand_signed', float('nan')):.4f}" if 'data_rand_signed' in r else "---"
    dr_a = f"{r.get('data_rand_abs', float('nan')):.4f}" if 'data_rand_abs' in r else "---"
    print(f"{m:<10} {mr_s:>14} {mr_a:>14} {dr_s:>14} {dr_a:>14}")


# ============================================================
# %% EXPERIMENT 3: Tree random-label training accuracy
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("EXPERIMENT 3: Tree random-label training accuracy")
logger.info("=" * 60)

from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper
from sklearn.metrics import accuracy_score

y_train_tab_shuf = shuffle_labels(y_train_tab, seed=SEED)

# RF on shuffled labels
rf_rand = RFClassifier(n_estimators=500, random_state=SEED)
rf_rand.fit(x_train_tab, y_train_tab_shuf)
rf_rand_train_acc = accuracy_score(y_train_tab_shuf, rf_rand.predict(x_train_tab))
rf_rand_test_acc = accuracy_score(y_test, rf_rand.predict(x_test_tab))

# XGB on shuffled labels
xgb_rand = XGBClassifierWrapper(n_estimators=500, random_state=SEED)
xgb_rand.fit(x_train_tab, y_train_tab_shuf)
xgb_rand_train_acc = accuracy_score(y_train_tab_shuf, xgb_rand.predict(x_train_tab))
xgb_rand_test_acc = accuracy_score(y_test, xgb_rand.predict(x_test_tab))

print("\n=== Random-Label Training Accuracy (Memorisation Check) ===")
print(f"{'Model':<15} {'Train Acc (shuf)':>16} {'Test Acc (shuf)':>16} {'Chance':>10}")
chance = 1.0 / n_classes
print(f"{'RF':<15} {rf_rand_train_acc:>16.4f} {rf_rand_test_acc:>16.4f} {chance:>10.4f}")
print(f"{'XGBoost':<15} {xgb_rand_train_acc:>16.4f} {xgb_rand_test_acc:>16.4f} {chance:>10.4f}")

tree_rand_results = {
    "RF_train_acc_shuffled": rf_rand_train_acc,
    "RF_test_acc_shuffled": rf_rand_test_acc,
    "XGB_train_acc_shuffled": xgb_rand_train_acc,
    "XGB_test_acc_shuffled": xgb_rand_test_acc,
    "chance_level": chance,
}

# ============================================================
# %% EXPERIMENT 4: Gini sparsity as alternative complexity
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("EXPERIMENT 4: Gini sparsity index")
logger.info("=" * 60)


def gini_index(attribution: np.ndarray) -> float:
    """Gini index of |attribution|. 0 = perfectly uniform, 1 = maximally sparse."""
    a = np.abs(attribution).flatten()
    if a.sum() < 1e-12:
        return 0.0
    a = np.sort(a)
    n = len(a)
    index = np.arange(1, n + 1)
    return float((2.0 * (index * a).sum() / (n * a.sum())) - (n + 1.0) / n)


# Tabular models: compute Gini on SHAP
gini_results = {}
for name, shap_red in [("RF", shap_rf_red), ("XGBoost", shap_xgb_red)]:
    scores = [gini_index(shap_red[i]) for i in sample_idx]
    gini_results[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

# DL models: Gini on projected attributions (already computed above)
for name, proj in [("LSTM", lstm_proj), ("CNN1D", cnn1d_proj),
                   ("CNN2D", cnn2d_proj), ("ViT", vit_proj)]:
    scores = [gini_index(proj[i]) for i in range(len(proj))]
    gini_results[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

print("\n=== Gini Sparsity Index (alternative to entropy complexity) ===")
print(f"{'Model':<12} {'Gini':>12} {'Entropy':>12}")
# Load entropy values from round2 for comparison
r2 = json.loads((RESULTS_DIR / f"seed_{SEED}" / "round2.json").read_text())
r1 = json.loads((RESULTS_DIR / f"seed_{SEED}" / "results.json").read_text())
for m in ["RF", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT"]:
    g = gini_results.get(m, {})
    # Try to get entropy from results
    entropy_val = r1.get(m, {}).get("complexity_mean", float("nan"))
    print(f"{m:<12} {g.get('mean', float('nan')):>12.4f} {entropy_val:>12.4f}")


# ============================================================
# %% Save all revision results to JSON
# ============================================================
revision_output = {
    "plausibility_threshold_sensitivity": plaus_sensitivity,
    "signed_vs_absolute_spearman": signed_results,
    "tree_random_label_accuracy": tree_rand_results,
    "gini_sparsity": gini_results,
}

out_path = RESULTS_DIR / "revision_experiments.json"
with open(out_path, "w") as f:
    json.dump(revision_output, f, indent=2, default=float)
logger.info(f"Saved revision results to {out_path}")

print("\n=== All revision experiments complete ===")
