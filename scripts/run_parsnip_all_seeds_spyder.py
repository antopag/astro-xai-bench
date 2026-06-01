# -*- coding: utf-8 -*-
"""
ParSNIP — remaining 4 seeds (123, 456, 789, 1024). Run in Spyder (F5).

Seed 42 already completed. Each seed: ~2-3h (VAE training ~2h, rest ~10 min).
Total: ~8-12h. All phases cachable — safe to interrupt and restart.

NOTE on plausibility std:
  The "plausibility_std" stored per seed is the per-SAMPLE std on the 50 test
  samples evaluated within that seed. The CROSS-SEED std is computed in the
  aggregation section at the bottom, from the 5 per-seed plausibility_mean values.
"""

# %% Imports and setup
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

SEEDS_REMAINING = [123, 456, 789, 1024]
ALL_SEEDS = [42] + SEEDS_REMAINING

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "plasticc"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

PARSNIP_BANDS = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]
PARSNIP_SETTINGS = {
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
}
CLASSIFIER_KEYS = [
    'color', 'color_error', 's1', 's1_error', 's2', 's2_error',
    's3', 's3_error', 'luminosity', 'luminosity_error', 'reference_time_error',
]
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 14,
    "metric": "multi_logloss",
    "min_child_weight": 50,
    "verbose": -1,
    "n_jobs": -1,
}

if __name__ == "__main__":
    pass  # Spyder guard — all code below runs at module level in Spyder

# %% Load shared data (once)
import parsnip
import lcdata
import lightgbm as lgb
import shap
from scipy.stats import spearmanr
from sklearn.utils.class_weight import compute_sample_weight

from src.data.raw_loader import RawPLAsTiCCLoader
from src.evaluation.metrics import compute_metrics
from src.xai.shap_utils import normalize_shap_values
from src.xai.metrics import (
    faithfulness_correlation, explanation_complexity,
    plausibility_score, build_expert_mask_tabular,
)
from src.models.parsnip.xai import _compute_plausibility_parsnip
from src.models.avocado.xai import AvocadoModelWrapper

loader = RawPLAsTiCCLoader(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

# Prepare lcdata datasets (shared cache across seeds)
SHARED_CACHE = RESULTS_DIR / "seed_42" / "parsnip_cache"
train_lcdata_path = SHARED_CACHE / "lcdata_train.h5"
test_lcdata_path = SHARED_CACHE / "lcdata_test.h5"

train_dataset = lcdata.read_hdf5(str(train_lcdata_path))
test_dataset = lcdata.read_hdf5(str(test_lcdata_path))
print(f"Loaded lcdata: {len(train_dataset)} train, {len(test_dataset)} test")

# Labels (shared)
train_labels_raw = np.array(train_dataset.meta["type"])
test_labels_raw = np.array(test_dataset.meta["type"])
unique_targets = sorted(set(train_labels_raw) | set(test_labels_raw))
target_to_idx = {t: i for i, t in enumerate(unique_targets)}
y_train = np.array([target_to_idx[t] for t in train_labels_raw])
y_test = np.array([target_to_idx[t] for t in test_labels_raw])
n_classes = len(unique_targets)
print(f"Labels: {n_classes} classes, {len(y_train)} train, {len(y_test)} test")

# %% Seed loop
for seed_idx, SEED in enumerate(SEEDS_REMAINING):
    seed_t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"# SEED {SEED} ({seed_idx+1}/{len(SEEDS_REMAINING)})")
    print(f"{'#'*60}")

    seed_dir = RESULTS_DIR / f"seed_{SEED}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = seed_dir / "parsnip_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    results_path = seed_dir / "results.json"
    if results_path.is_file():
        with open(results_path) as f:
            existing = json.load(f)
        if "ParSNIP" in existing and existing["ParSNIP"].get("faithfulness_mean", 0) != 0:
            print(f"  Seed {SEED} already complete, skipping.")
            continue

    # ---- Set seed ----
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()

    LGB_PARAMS["seed"] = SEED

    # ---- FASE 1: VAE training ----
    print(f"\n  FASE 1: VAE training")

    vae_path = ckpt_dir / f"parsnip_vae_seed{SEED}.pt"

    if vae_path.is_file():
        print(f"    Loading existing VAE: {vae_path.name}")
        model = parsnip.load_model(str(vae_path))
        model.threads = 1

        # Collapse check
        train_parsed_check = parsnip.parse_dataset(train_dataset, kind="plasticc", reject_invalid=False)
        test_pred_check = model.predict_dataset(train_parsed_check[:3])
        s1_std = np.std([test_pred_check["s1"][i] for i in range(3)])
        if s1_std < 1e-6:
            print(f"    ⚠ VAE collapsed (s1 std={s1_std:.2e}). Deleting and retraining...")
            os.remove(str(vae_path))
            del model
        else:
            print(f"    VAE OK (s1 std={s1_std:.4f})")

    if not vae_path.is_file():
        train_parsed = parsnip.parse_dataset(train_dataset, kind="plasticc", reject_invalid=False)
        print(f"    Training VAE (device={DEVICE}, lr={PARSNIP_SETTINGS['learning_rate']})...")
        t0 = time.time()

        model = parsnip.ParsnipModel(
            str(vae_path),
            bands=PARSNIP_BANDS,
            device=DEVICE,
            threads=1,
            settings=PARSNIP_SETTINGS,
        )
        model.fit(train_parsed, max_epochs=200, augment=True)
        elapsed = time.time() - t0
        print(f"    VAE training: {elapsed/60:.1f} min")

    # ---- FASE 2: Latent predictions ----
    print(f"\n  FASE 2: Latent predictions")

    train_parsed = parsnip.parse_dataset(train_dataset, kind="plasticc", reject_invalid=False)
    test_parsed = parsnip.parse_dataset(test_dataset, kind="plasticc", reject_invalid=False)

    t0 = time.time()
    train_predictions = model.predict_dataset(train_parsed)
    if "original_object_id" not in train_predictions.colnames:
        train_predictions["original_object_id"] = train_predictions["object_id"]
    print(f"    Train: {len(train_predictions)} in {time.time()-t0:.1f}s")

    t0 = time.time()
    test_predictions = model.predict_dataset(test_parsed)
    if "original_object_id" not in test_predictions.colnames:
        test_predictions["original_object_id"] = test_predictions["object_id"]
    print(f"    Test: {len(test_predictions)} in {time.time()-t0:.1f}s")

    # ---- FASE 3: LightGBM classifier ----
    print(f"\n  FASE 3: LightGBM classifier")

    # Extract and impute features
    train_features = np.column_stack([np.array(train_predictions[k], dtype=float) for k in CLASSIFIER_KEYS])
    test_features = np.column_stack([np.array(test_predictions[k], dtype=float) for k in CLASSIFIER_KEYS])

    for col_idx in range(train_features.shape[1]):
        for feats in [train_features, test_features]:
            nan_mask = np.isnan(feats[:, col_idx])
            if nan_mask.any():
                feats[nan_mask, col_idx] = np.nanmedian(feats[:, col_idx])

    n_nan = np.isnan(train_features).sum() + np.isnan(test_features).sum()
    print(f"    Features: {train_features.shape[1]}, NaN after impute: {n_nan}")

    sample_weights = compute_sample_weight("balanced", y_train)
    dtrain = lgb.Dataset(
        train_features, label=y_train, weight=sample_weights,
        feature_name=CLASSIFIER_KEYS, free_raw_data=False,
    )

    t0 = time.time()
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500)
    print(f"    LightGBM: {time.time()-t0:.1f}s, n_trees={booster.num_trees()}")

    y_proba = booster.predict(test_features)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"    Acc: {metrics['accuracy']:.4f}, F1m: {metrics['f1_macro']:.4f}, AUC: {metrics.get('auc_ovr', 'N/A')}")

    # Save
    booster.save_model(str(ckpt_dir / "parsnip_lgb.txt"))
    np.savez(seed_dir / "parsnip_predictions.npz",
             y_pred=y_pred, y_proba=y_proba, y_test=y_test)
    np.savez(seed_dir / "parsnip_classifier_features.npz",
             features=test_features, feature_names=np.array(CLASSIFIER_KEYS))
    np.savez(seed_dir / "parsnip_train_features.npz",
             features=train_features, labels=y_train, feature_names=np.array(CLASSIFIER_KEYS))

    # ---- FASE 4: XAI ----
    print(f"\n  FASE 4: XAI metrics")

    wrapper = AvocadoModelWrapper(booster)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_features), size=min(50, len(test_features)), replace=False)

    explainer = shap.TreeExplainer(booster)
    mean_shap = normalize_shap_values(explainer.shap_values(test_features[sample_idx]))

    faith_vals = [faithfulness_correlation(wrapper, test_features[i], mean_shap[j], seed=SEED)
                  for j, i in enumerate(sample_idx)]
    compl_vals = [explanation_complexity(mean_shap[j]) for j in range(len(sample_idx))]
    plaus_vals = _compute_plausibility_parsnip(mean_shap, CLASSIFIER_KEYS)

    xai = {
        "faithfulness_mean": float(np.mean(faith_vals)),
        "faithfulness_std": float(np.std(faith_vals)),
        "complexity_mean": float(np.mean(compl_vals)),
        "complexity_std": float(np.std(compl_vals)),
        "plausibility_mean": float(np.mean(plaus_vals)),
        "plausibility_std": float(np.std(plaus_vals)),  # per-sample std, NOT cross-seed
    }
    metrics.update(xai)

    print(f"    Faith: {xai['faithfulness_mean']:.3f}±{xai['faithfulness_std']:.3f}")
    print(f"    Compl: {xai['complexity_mean']:.3f}±{xai['complexity_std']:.3f}")
    print(f"    Plaus: {xai['plausibility_mean']:.3f}±{xai['plausibility_std']:.3f}")

    # ---- Save ----
    existing = {}
    if results_path.is_file():
        with open(results_path) as f:
            existing = json.load(f)
    existing["ParSNIP"] = metrics
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    # Timing and GPU memory
    seed_elapsed = time.time() - seed_t0
    gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    print(f"\n    Seed {SEED} complete: {seed_elapsed/60:.1f} min, GPU peak: {gpu_peak:.2f} GB")

    del model, booster, train_features, test_features, train_predictions, test_predictions
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# %% Aggregation across all 5 seeds
print(f"\n{'='*60}")
print("AGGREGATION (5 seeds)")
print(f"{'='*60}")

agg_metrics = {}
per_seed_vals = {}

for seed in ALL_SEEDS:
    rp = RESULTS_DIR / f"seed_{seed}" / "results.json"
    with open(rp) as f:
        d = json.load(f)
    ps = d.get("ParSNIP", {})
    for k, v in ps.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            per_seed_vals.setdefault(k, []).append(v)

print("\nPer-seed values (check for outliers):")
for k in ["accuracy", "f1_macro", "auc_ovr", "faithfulness_mean",
          "complexity_mean", "plausibility_mean"]:
    vals = per_seed_vals.get(k, [])
    if vals:
        vstr = ", ".join(f"{v:.4f}" for v in vals)
        print(f"  {k:25s} [{vstr}]")
        agg_metrics[k] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1)}

# IMPORTANT: Table 5 reports CROSS-SEED std (std of the 5 per-seed means),
# not the per-sample std within a single seed. The per-sample std is stored
# in the per-seed JSON as e.g. plausibility_std for documentation only.
print("\nCross-seed summary (mean ± std of 5 seed-level means, for Table 5):")
for k in ["accuracy", "f1_macro", "auc_ovr", "faithfulness_mean",
          "complexity_mean", "plausibility_mean"]:
    if k in agg_metrics:
        m = agg_metrics[k]
        print(f"  {k:25s} {m['mean']:.4f} ± {m['std']:.4f}")

# Sanity (seed 42 only)
rp42 = RESULTS_DIR / "seed_42" / "results.json"
with open(rp42) as f:
    d42 = json.load(f)
ps42 = d42.get("ParSNIP", {})
rho_signed = ps42.get("data_randomization_rho_signed", ps42.get("data_randomization_rho"))
print(f"\n  Data rand ρ (seed 42): signed={rho_signed}, abs={abs(rho_signed) if rho_signed else 'N/A'}")

# %% Save CSV + REVISION_RESULTS.md

# Existing base models from published paper (seed-42 values from Table 4)
base_models = {
    "RF":    {"accuracy": 0.646, "f1_macro": 0.430, "auc_ovr": 0.907,
              "acc_std": 0.004, "f1_std": 0.006, "auc_std": 0.001},
    "XGB":   {"accuracy": 0.668, "f1_macro": 0.529, "auc_ovr": 0.909,
              "acc_std": 0.000, "f1_std": 0.000, "auc_std": 0.000},
    "LSTM":  {"accuracy": 0.605, "f1_macro": 0.376, "auc_ovr": 0.886,
              "acc_std": 0.007, "f1_std": 0.009, "auc_std": 0.003},
    "CNN1D": {"accuracy": 0.529, "f1_macro": 0.289, "auc_ovr": 0.834,
              "acc_std": 0.036, "f1_std": 0.040, "auc_std": 0.029},
    "CNN2D": {"accuracy": 0.560, "f1_macro": 0.435, "auc_ovr": 0.880,
              "acc_std": 0.009, "f1_std": 0.006, "auc_std": 0.004},
    "ViT":   {"accuracy": 0.559, "f1_macro": 0.447, "auc_ovr": 0.866,
              "acc_std": 0.012, "f1_std": 0.014, "auc_std": 0.006},
    "Ens.":  {"accuracy": 0.709, "f1_macro": 0.568, "auc_ovr": 0.922,
              "acc_std": 0.006, "f1_std": 0.015, "auc_std": 0.003},
}

# Load Avocado aggregated
avocado_vals = {}
for seed in ALL_SEEDS:
    rp = RESULTS_DIR / f"seed_{seed}" / "results.json"
    with open(rp) as f:
        d = json.load(f)
    av = d.get("Avocado", {})
    for k, v in av.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            avocado_vals.setdefault(k, []).append(v)

# Build markdown table
lines = []
lines.append("# Revised Benchmark Results (with Avocado + ParSNIP)")
lines.append("")
lines.append("| Model | Acc. | F1$_m$ | AUC | Faith. | Compl. | Plaus. | Data ρ |")
lines.append("|-------|------|--------|-----|--------|--------|--------|--------|")

for name, bm in base_models.items():
    lines.append(
        f"| {name} | "
        f"${bm['accuracy']:.3f}\\pm{bm['acc_std']:.3f}$ | "
        f"${bm['f1_macro']:.3f}\\pm{bm['f1_std']:.3f}$ | "
        f"${bm['auc_ovr']:.3f}\\pm{bm['auc_std']:.3f}$ | "
        f"--- | --- | --- | --- |"
    )

# Avocado row
for model_name, vals_dict, sanity_key in [
    ("Avocado", avocado_vals, "data_randomization_rho_signed"),
    ("ParSNIP", per_seed_vals, "data_randomization_rho_signed"),
]:
    row = f"| **{model_name}** | "
    for k in ["accuracy", "f1_macro", "auc_ovr", "faithfulness_mean",
              "complexity_mean"]:
        vs = vals_dict.get(k, [])
        if vs:
            row += f"${np.mean(vs):.3f}\\pm{np.std(vs, ddof=1):.3f}$ | "
        else:
            row += "--- | "
    # Plausibility
    pv = vals_dict.get("plausibility_mean", [])
    if pv:
        if model_name == "Avocado":
            row += f"$0.379^*$ | "  # structural constant
        else:
            row += f"${np.mean(pv):.3f}\\pm{np.std(pv, ddof=1):.3f}$ | "
    else:
        row += "--- | "
    # Sanity
    san = vals_dict.get(sanity_key, [])
    if san:
        row += f"${san[0]:.3f}$ |"
    else:
        row += "--- |"
    lines.append(row)

lines.append("")
lines.append("$^*$ Structural constant (11/29); see §6.3.")
lines.append("")
lines.append("**Note on std columns**: All ± values are **cross-seed std** (std of 5 per-seed")
lines.append("means), matching the convention in Table 5 of the published paper. The per-sample")
lines.append("std within a single seed (stored as e.g. `plausibility_std` in each seed's")
lines.append("`results.json`) is a different, larger quantity and is NOT reported in the table.")
lines.append("")
# Add per-sample std for documentation
lines.append("### Per-sample std (for reference, not in Table 5)")
lines.append("")
lines.append("| Model | Faith. (per-sample) | Compl. (per-sample) | Plaus. (per-sample) |")
lines.append("|-------|--------------------|--------------------|---------------------|")
for model_name, vals_dict in [("Avocado", avocado_vals), ("ParSNIP", per_seed_vals)]:
    row = f"| {model_name} | "
    for k in ["faithfulness_std", "complexity_std", "plausibility_std"]:
        vs = vals_dict.get(k, [])
        if vs:
            row += f"{np.mean(vs):.4f} | "
        else:
            row += "--- | "
    lines.append(row)
lines.append("")
lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")

md_path = PROJECT_ROOT / "REVISION_RESULTS.md"
with open(md_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {md_path}")

# CSV
import csv
csv_path = PROJECT_ROOT / "paper" / "tables" / "avocado_parsnip_table.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std",
                "auc_mean", "auc_std", "faithfulness_mean", "faithfulness_std",
                "complexity_mean", "complexity_std", "plausibility_mean", "plausibility_std",
                "data_rand_rho_signed"])
    for model_name, vals_dict in [("Avocado", avocado_vals), ("ParSNIP", per_seed_vals)]:
        row = [model_name]
        for k in ["accuracy", "f1_macro", "auc_ovr", "faithfulness_mean",
                   "complexity_mean", "plausibility_mean"]:
            vs = vals_dict.get(k, [])
            row.extend([f"{np.mean(vs):.4f}", f"{np.std(vs, ddof=1):.4f}"] if vs else ["", ""])
        san = vals_dict.get("data_randomization_rho_signed", [])
        row.append(f"{san[0]:.4f}" if san else "")
        w.writerow(row)
print(f"Saved: {csv_path}")

print(f"\n{'='*60}")
print("ALL PARSNIP SEEDS COMPLETE")
print(f"{'='*60}")
