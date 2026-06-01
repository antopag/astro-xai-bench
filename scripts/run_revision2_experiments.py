"""Revision 2 experiments requested by reviewer.

Three analyses:
  1. Random/oracle baselines for faithfulness (deletion/insertion AUC)
  2. IG baseline ablation for LSTM/CNN1D (zero, mean, noise; steps 20/50/100)
  3. Grad-CAM vs IG faithfulness for CNN2D

Usage in Spyder: %runfile run_revision2_experiments.py --wdir
"""

from __future__ import annotations

# %% Setup
import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"
SEED = 42
N_SAMPLES = 30
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
lc_test = test_data["light_curves"][:, :, :, 0].astype(np.float32)

from src.models.tabular_rf import extract_features, RFClassifier, XGBClassifierWrapper
from src.xai.shap_analysis import compute_shap_values

x_train_tab = extract_features(np.vstack([train_data["light_curves"], val_data["light_curves"]]))
y_train_tab = np.concatenate([y_train, y_val])
x_test_tab = extract_features(test_data["light_curves"])

rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(y_test), size=N_SAMPLES, replace=False)

# Load GAF
GAF_SIZE = 224
gaf_cache = DATA_DIR / f"gaf_{GAF_SIZE}.npz"
gaf_test = np.load(gaf_cache)["test"].astype(np.float32)

# Helpers
seed_dir = RESULTS_DIR / f"seed_{SEED}" / "checkpoints"


def load_dl_model(factory, ckpt_path):
    model = factory().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ============================================================
# %% EXPERIMENT 1: Random and oracle baselines for faithfulness
# ============================================================
logger.info("=" * 60)
logger.info("EXPERIMENT 1: Random/oracle baselines for deletion/insertion AUC")
logger.info("=" * 60)

from src.xai.faithfulness import (
    insertion_deletion_tabular,
    insertion_deletion_timeseries,
    insertion_deletion_image,
    aggregate_insertion_deletion,
)
from src.xai.integrated_gradients import compute_integrated_gradients

# --- Tabular models (RF, XGBoost) ---
rf = RFClassifier(n_estimators=500, random_state=SEED)
rf.fit(x_train_tab, y_train_tab)
shap_rf = compute_shap_values(rf, x_test_tab, method="tree")

xgb = XGBClassifierWrapper(n_estimators=500, random_state=SEED)
xgb.fit(x_train_tab, y_train_tab)
shap_xgb = compute_shap_values(xgb, x_test_tab, method="tree")


def reduce_shap_to_1d(sv, idx, n_features=60):
    """Get per-feature attribution for sample idx."""
    if isinstance(sv, list):
        pred = rf.predict(x_test_tab[idx:idx + 1])[0]
        return sv[pred][idx]
    if sv.ndim == 3:
        pred = rf.predict(x_test_tab[idx:idx + 1])[0]
        if sv.shape[2] == n_features:
            return sv[idx, pred, :]
        return sv[idx, :, pred]
    return sv[idx]


baseline_results = {}

for model_name, model_obj, shap_vals in [("RF", rf, shap_rf), ("XGBoost", xgb, shap_xgb)]:
    logger.info(f"--- {model_name}: random/oracle baselines ---")
    normal_res, random_res, oracle_res = [], [], []

    for idx in sample_idx:
        attr = reduce_shap_to_1d(shap_vals, idx)

        # Normal (attribution-ordered)
        normal_res.append(insertion_deletion_tabular(model_obj.predict_proba, x_test_tab[idx], attr))

        # Random ordering
        rand_attr = rng.permutation(np.abs(attr))
        random_res.append(insertion_deletion_tabular(model_obj.predict_proba, x_test_tab[idx], rand_attr))

        # Oracle ordering: use actual prediction change per feature
        base_prob = model_obj.predict_proba(x_test_tab[idx:idx + 1])[0]
        pred_class = base_prob.argmax()
        oracle_importance = np.zeros(len(x_test_tab[idx]))
        for f in range(len(x_test_tab[idx])):
            x_masked = x_test_tab[idx].copy()
            x_masked[f] = 0.0
            new_prob = model_obj.predict_proba(x_masked.reshape(1, -1))[0]
            oracle_importance[f] = base_prob[pred_class] - new_prob[pred_class]
        oracle_res.append(insertion_deletion_tabular(model_obj.predict_proba, x_test_tab[idx], oracle_importance))

    baseline_results[model_name] = {
        "normal": aggregate_insertion_deletion(normal_res),
        "random": aggregate_insertion_deletion(random_res),
        "oracle": aggregate_insertion_deletion(oracle_res),
    }
    for variant in ["normal", "random", "oracle"]:
        r = baseline_results[model_name][variant]
        logger.info(f"  {variant}: del={r['deletion_auc_mean']:.4f} ins={r['insertion_auc_mean']:.4f}")

# --- DL models: timeseries (LSTM, CNN1D) ---
from src.models.lstm_baseline import LSTMClassifier
from src.models.cnn_baseline import CNN1D

ts_baseline_np = lc_train.mean(axis=0).astype(np.float32)
ts_baseline_t = torch.from_numpy(ts_baseline_np[None]).to(device)

for model_name, factory, ckpt_name in [
    ("LSTM", lambda: LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                                     n_layers=2, dropout=0.3, bidirectional=True), "lstm.pt"),
    ("CNN1D", lambda: CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3), "cnn1d.pt"),
]:
    model = load_dl_model(factory, seed_dir / ckpt_name)

    def predict_proba_ts(batch):
        with torch.no_grad():
            t = torch.from_numpy(batch).float().to(device)
            return torch.softmax(model(t), dim=-1).cpu().numpy()

    logger.info(f"--- {model_name}: random/oracle baselines ---")
    normal_res, random_res, oracle_res = [], [], []

    for idx in sample_idx:
        x = torch.from_numpy(lc_test[idx:idx + 1]).float().to(device)
        attr = compute_integrated_gradients(model, x, n_steps=30, baseline=ts_baseline_t)

        # Normal
        normal_res.append(insertion_deletion_timeseries(predict_proba_ts, lc_test[idx], attr))

        # Random band ordering
        rand_attr = attr.copy()
        band_importance = np.abs(attr).sum(axis=1)
        rand_attr_flat = rng.permutation(band_importance)
        random_res.append(insertion_deletion_timeseries(predict_proba_ts, lc_test[idx],
                                                        np.ones_like(attr) * rand_attr_flat[:, None]))

        # Oracle: zero each band, measure drop
        base_prob = predict_proba_ts(lc_test[idx:idx + 1])[0]
        pred_class = base_prob.argmax()
        oracle_imp = np.zeros(6)
        for b in range(6):
            x_masked = lc_test[idx].copy()
            x_masked[b, :] = 0.0
            new_prob = predict_proba_ts(x_masked[None])[0]
            oracle_imp[b] = base_prob[pred_class] - new_prob[pred_class]
        oracle_attr = np.ones_like(attr) * oracle_imp[:, None]
        oracle_res.append(insertion_deletion_timeseries(predict_proba_ts, lc_test[idx], oracle_attr))

    baseline_results[model_name] = {
        "normal": aggregate_insertion_deletion(normal_res),
        "random": aggregate_insertion_deletion(random_res),
        "oracle": aggregate_insertion_deletion(oracle_res),
    }
    for variant in ["normal", "random", "oracle"]:
        r = baseline_results[model_name][variant]
        logger.info(f"  {variant}: del={r['deletion_auc_mean']:.4f} ins={r['insertion_auc_mean']:.4f}")
    del model; torch.cuda.empty_cache()

# --- DL models: image (CNN2D, ViT) ---
from src.models.cnn_baseline import CNN2D as CNN2DClassifier
from src.models.vit import ViTClassifier

gaf_baseline_t = torch.from_numpy(gaf_test.mean(axis=0, keepdims=True)).to(device)

for model_name, factory, ckpt_name in [
    ("CNN2D", lambda: CNN2DClassifier(n_bands=6, n_classes=n_classes), "cnn2d.pt"),
    ("ViT", lambda: ViTClassifier(n_bands=6, n_classes=n_classes), "vit.pt"),
]:
    model = load_dl_model(factory, seed_dir / ckpt_name)

    def predict_proba_img(batch):
        with torch.no_grad():
            t = torch.from_numpy(batch).float().to(device)
            return torch.softmax(model(t), dim=-1).cpu().numpy()

    logger.info(f"--- {model_name}: random/oracle baselines ---")
    normal_res, random_res = [], []

    for idx in sample_idx[:15]:  # fewer samples for images (slower)
        x = torch.from_numpy(gaf_test[idx:idx + 1]).float().to(device)
        attr = compute_integrated_gradients(model, x, n_steps=30, baseline=gaf_baseline_t)

        normal_res.append(insertion_deletion_image(predict_proba_img, gaf_test[idx], attr))
        rand_attr = rng.permutation(np.abs(attr).flatten()).reshape(attr.shape)
        random_res.append(insertion_deletion_image(predict_proba_img, gaf_test[idx], rand_attr))

    baseline_results[model_name] = {
        "normal": aggregate_insertion_deletion(normal_res),
        "random": aggregate_insertion_deletion(random_res),
    }
    for variant in ["normal", "random"]:
        r = baseline_results[model_name][variant]
        logger.info(f"  {variant}: del={r['deletion_auc_mean']:.4f} ins={r['insertion_auc_mean']:.4f}")
    del model; torch.cuda.empty_cache()

print("\n=== Random/Oracle Baselines for Deletion/Insertion AUC ===")
print(f"{'Model':<10} {'Del(attr)':>10} {'Del(rand)':>10} {'Del(orac)':>10} {'Ins(attr)':>10} {'Ins(rand)':>10} {'Ins(orac)':>10}")
for m in ["RF", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT"]:
    r = baseline_results[m]
    dn = f"{r['normal']['deletion_auc_mean']:.3f}"
    dr = f"{r['random']['deletion_auc_mean']:.3f}"
    do = f"{r.get('oracle', {}).get('deletion_auc_mean', float('nan')):.3f}"
    in_ = f"{r['normal']['insertion_auc_mean']:.3f}"
    ir = f"{r['random']['insertion_auc_mean']:.3f}"
    io = f"{r.get('oracle', {}).get('insertion_auc_mean', float('nan')):.3f}"
    print(f"{m:<10} {dn:>10} {dr:>10} {do:>10} {in_:>10} {ir:>10} {io:>10}")


# ============================================================
# %% EXPERIMENT 2: IG baseline ablation for LSTM and CNN1D
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("EXPERIMENT 2: IG baseline ablation (LSTM, CNN1D)")
logger.info("=" * 60)

from src.xai.metrics import faithfulness_correlation, explanation_complexity

baselines_config = {
    "zero": None,  # None = zero baseline in compute_integrated_gradients
    "mean": ts_baseline_t,
    "noise": "noise",  # sentinel, generated per sample
}
steps_config = [20, 50, 100]

ig_ablation_results = {}

for model_name, factory, ckpt_name in [
    ("LSTM", lambda: LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                                     n_layers=2, dropout=0.3, bidirectional=True), "lstm.pt"),
    ("CNN1D", lambda: CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3), "cnn1d.pt"),
]:
    model = load_dl_model(factory, seed_dir / ckpt_name)
    ig_ablation_results[model_name] = {}

    for bl_name, bl_val in baselines_config.items():
        for n_steps in steps_config:
            key = f"{bl_name}_s{n_steps}"
            logger.info(f"  {model_name}: baseline={bl_name}, steps={n_steps}")

            attrs_flat = []
            t0 = time.time()
            for idx in sample_idx[:15]:  # 15 samples for speed
                x = torch.from_numpy(lc_test[idx:idx + 1]).float().to(device)
                if bl_val == "noise":
                    noise_bl = torch.randn_like(x) * 0.1
                    attr = compute_integrated_gradients(model, x, n_steps=n_steps, baseline=noise_bl)
                else:
                    attr = compute_integrated_gradients(model, x, n_steps=n_steps, baseline=bl_val)
                attrs_flat.append(attr.flatten())
            elapsed = time.time() - t0

            # Mean attribution magnitude and complexity
            all_attrs = np.array(attrs_flat)
            mean_mag = float(np.abs(all_attrs).mean())
            mean_compl = float(np.mean([explanation_complexity(a) for a in all_attrs]))

            ig_ablation_results[model_name][key] = {
                "mean_magnitude": mean_mag,
                "complexity": mean_compl,
                "time_s": round(elapsed, 1),
            }

    del model; torch.cuda.empty_cache()

print("\n=== IG Baseline Ablation ===")
for m in ["LSTM", "CNN1D"]:
    print(f"\n{m}:")
    print(f"  {'Config':<16} {'|attr| mean':>12} {'Complexity':>12} {'Time (s)':>10}")
    for key, vals in ig_ablation_results[m].items():
        print(f"  {key:<16} {vals['mean_magnitude']:>12.6f} {vals['complexity']:>12.4f} {vals['time_s']:>10.1f}")


# ============================================================
# %% EXPERIMENT 3: Grad-CAM vs IG faithfulness for CNN2D
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("EXPERIMENT 3: Grad-CAM vs IG faithfulness for CNN2D")
logger.info("=" * 60)

from src.xai.gradcam import GradCAM

cnn2d = load_dl_model(lambda: CNN2DClassifier(n_bands=6, n_classes=n_classes), seed_dir / "cnn2d.pt")

# Find target layer (last conv)
target_layer = None
for name, module in cnn2d.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        target_layer = module
        target_name = name
logger.info(f"Grad-CAM target layer: {target_name}")

gradcam = GradCAM(cnn2d, target_layer)


def predict_proba_cnn2d(batch):
    with torch.no_grad():
        t = torch.from_numpy(batch).float().to(device)
        return torch.softmax(cnn2d(t), dim=-1).cpu().numpy()


ig_results_cnn2d = []
gc_results_cnn2d = []

for idx in sample_idx[:15]:
    x_np = gaf_test[idx]
    x_t = torch.from_numpy(x_np[None]).float().to(device)

    # IG attribution
    ig_attr = compute_integrated_gradients(cnn2d, x_t, n_steps=30, baseline=gaf_baseline_t)
    ig_res = insertion_deletion_image(predict_proba_cnn2d, x_np, ig_attr)
    ig_results_cnn2d.append(ig_res)

    # Grad-CAM attribution: get per-band heatmap
    pred_class = int(predict_proba_cnn2d(x_np[None])[0].argmax())
    gc_heatmap_raw = gradcam(x_t, target_class=pred_class)  # (H', W') from last conv
    # Upsample to input size and replicate across bands
    from scipy.ndimage import zoom
    scale_h = x_np.shape[1] / gc_heatmap_raw.shape[0]
    scale_w = x_np.shape[2] / gc_heatmap_raw.shape[1]
    gc_heatmap_full = zoom(gc_heatmap_raw, (scale_h, scale_w), order=1)
    gc_heatmap_full = np.clip(gc_heatmap_full, 0, None)
    # Stack across bands (Grad-CAM is band-agnostic from the conv perspective)
    gc_attr = np.stack([gc_heatmap_full] * 6, axis=0)  # (6, H, W)

    gc_res = insertion_deletion_image(predict_proba_cnn2d, x_np, gc_attr)
    gc_results_cnn2d.append(gc_res)

ig_agg = aggregate_insertion_deletion(ig_results_cnn2d)
gc_agg = aggregate_insertion_deletion(gc_results_cnn2d)

gradcam_vs_ig = {
    "IG": {"del_auc": ig_agg["deletion_auc_mean"], "ins_auc": ig_agg["insertion_auc_mean"]},
    "GradCAM": {"del_auc": gc_agg["deletion_auc_mean"], "ins_auc": gc_agg["insertion_auc_mean"]},
}

print("\n=== Grad-CAM vs IG Faithfulness (CNN2D) ===")
print(f"{'Method':<10} {'Del AUC':>10} {'Ins AUC':>10}")
print(f"{'IG':<10} {ig_agg['deletion_auc_mean']:>10.4f} {ig_agg['insertion_auc_mean']:>10.4f}")
print(f"{'Grad-CAM':<10} {gc_agg['deletion_auc_mean']:>10.4f} {gc_agg['insertion_auc_mean']:>10.4f}")

del cnn2d; torch.cuda.empty_cache()


# ============================================================
# %% EXPERIMENT 4: Per-sample attribution times
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("EXPERIMENT 4: Attribution compute times")
logger.info("=" * 60)

timing_results = {}

# TreeSHAP (RF)
t0 = time.time()
for _ in range(5):
    compute_shap_values(rf, x_test_tab[:100], method="tree")
timing_results["RF_TreeSHAP"] = round((time.time() - t0) / 5 / 100 * 1000, 1)  # ms per sample

# TreeSHAP (XGB)
t0 = time.time()
for _ in range(5):
    compute_shap_values(xgb, x_test_tab[:100], method="tree")
timing_results["XGB_TreeSHAP"] = round((time.time() - t0) / 5 / 100 * 1000, 1)

# IG for DL models
for model_name, factory, ckpt_name, x_data, bl in [
    ("LSTM_IG", lambda: LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                                        n_layers=2, dropout=0.3, bidirectional=True),
     "lstm.pt", lc_test, ts_baseline_t),
    ("CNN1D_IG", lambda: CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3),
     "cnn1d.pt", lc_test, ts_baseline_t),
    ("CNN2D_IG", lambda: CNN2DClassifier(n_bands=6, n_classes=n_classes),
     "cnn2d.pt", gaf_test, gaf_baseline_t),
    ("ViT_IG", lambda: ViTClassifier(n_bands=6, n_classes=n_classes),
     "vit.pt", gaf_test, gaf_baseline_t),
]:
    model = load_dl_model(factory, seed_dir / ckpt_name)
    t0 = time.time()
    n_timed = 10
    for i in range(n_timed):
        x = torch.from_numpy(x_data[i:i + 1]).float().to(device)
        compute_integrated_gradients(model, x, n_steps=50, baseline=bl)
    timing_results[model_name] = round((time.time() - t0) / n_timed * 1000, 1)
    del model; torch.cuda.empty_cache()

print("\n=== Per-Sample Attribution Times (ms) ===")
for k, v in timing_results.items():
    print(f"  {k:<15} {v:>8.1f} ms")


# ============================================================
# %% Save all results
# ============================================================
revision2_output = {
    "random_oracle_baselines": {k: {vk: {kk: vv for kk, vv in vval.items()
                                          if not isinstance(vv, np.ndarray)}
                                     for vk, vval in v.items()}
                                 for k, v in baseline_results.items()},
    "ig_baseline_ablation": ig_ablation_results,
    "gradcam_vs_ig_cnn2d": gradcam_vs_ig,
    "attribution_times_ms": timing_results,
}

out_path = RESULTS_DIR / "revision2_experiments.json"
with open(out_path, "w") as f:
    json.dump(revision2_output, f, indent=2, default=float)
logger.info(f"Saved to {out_path}")

print("\n=== All revision 2 experiments complete ===")
