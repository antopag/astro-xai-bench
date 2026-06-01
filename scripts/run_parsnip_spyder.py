# -*- coding: utf-8 -*-
"""
ParSNIP seed-42 pipeline — run in Spyder.

VAE training (~1.5-2h on RTX 4090) + LightGBM classifier + XAI + sanity.
Requires: torch, astro-parsnip, lcdata, sncosmo, extinction.
"""

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

SEED = 42
SEED_DIR = PROJECT_ROOT / "scripts" / "results" / "seed_42"
CACHE_DIR = SEED_DIR / "parsnip_cache"
CHECKPOINT_DIR = SEED_DIR / "checkpoints"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "plasticc"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

# =====================================================================
# FASE 1: Prepare lcdata datasets
# =====================================================================
print("\n" + "=" * 60)
print("FASE 1: Prepare lcdata datasets")
print("=" * 60)

import parsnip
import lcdata
from src.data.raw_loader import RawPLAsTiCCLoader

loader = RawPLAsTiCCLoader(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

train_lcdata_path = CACHE_DIR / "lcdata_train.h5"
test_lcdata_path = CACHE_DIR / "lcdata_test.h5"

if train_lcdata_path.is_file():
    print(f"  Train cache hit: {train_lcdata_path.name}")
    train_dataset = lcdata.read_hdf5(str(train_lcdata_path))
else:
    print("  Converting train split to lcdata...")
    train_dataset = loader.to_lcdata("train")
    train_dataset.write_hdf5(str(train_lcdata_path))
    print(f"  Saved: {train_lcdata_path.name}")

if test_lcdata_path.is_file():
    print(f"  Test cache hit: {test_lcdata_path.name}")
    test_dataset = lcdata.read_hdf5(str(test_lcdata_path))
else:
    print("  Converting test split to lcdata...")
    test_dataset = loader.to_lcdata("test")
    test_dataset.write_hdf5(str(test_lcdata_path))
    print(f"  Saved: {test_lcdata_path.name}")

print(f"  Train: {len(train_dataset)} LCs, Test: {len(test_dataset)} LCs")
print(f"  Train meta columns: {list(train_dataset.meta.colnames)}")
print(f"  Redshift NaN count: {np.isnan(train_dataset.meta['redshift']).sum()}")

# =====================================================================
# FASE 2: Train ParSNIP VAE
# =====================================================================
print("\n" + "=" * 60)
print("FASE 2: VAE training")
print("=" * 60)

from src.models.parsnip.train import _set_seed

_set_seed(SEED)

vae_path = CHECKPOINT_DIR / f"parsnip_vae_seed{SEED}.pt"

# VAE training settings — lower LR to prevent posterior collapse
# (default 1e-3 caused loss spike at epoch ~38 and collapse to constant output)
PARSNIP_SETTINGS = {
    "learning_rate": 1e-4,     # 10× lower than default
    "min_learning_rate": 1e-6, # 10× lower than default
}

if vae_path.is_file():
    # Validate checkpoint: load and check if collapsed
    print(f"  Checking existing VAE: {vae_path.name}")
    model = parsnip.load_model(str(vae_path))
    model.threads = 1

    # Quick collapse test: predict 3 objects, check variance
    train_parsed_check = parsnip.parse_dataset(train_dataset, kind="plasticc", reject_invalid=False)
    test_pred = model.predict_dataset(train_parsed_check[:3])
    s1_std = np.std([test_pred["s1"][i] for i in range(3)])
    if s1_std < 1e-6:
        print(f"  ⚠ VAE collapsed (s1 std={s1_std:.2e}). Deleting and retraining...")
        import os
        os.remove(str(vae_path))
        del model
    else:
        print(f"  VAE OK (s1 std={s1_std:.4f})")

if not vae_path.is_file():
    print(f"  Parsing dataset for ParSNIP...")
    train_parsed = parsnip.parse_dataset(train_dataset, kind="plasticc", reject_invalid=False)
    print(f"  Parsed: {len(train_parsed)} LCs")

    print(f"  Training VAE (device={DEVICE}, lr={PARSNIP_SETTINGS['learning_rate']})...")
    print(f"  This will take ~2-4h on RTX 4090.")
    t0 = time.time()

    model = parsnip.ParsnipModel(
        str(vae_path),
        bands=["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"],
        device=DEVICE,
        threads=1,
        settings=PARSNIP_SETTINGS,
    )
    model.fit(train_parsed, max_epochs=1000, augment=True)

    elapsed = time.time() - t0
    print(f"  VAE training complete: {elapsed/60:.1f} min")

# =====================================================================
# FASE 3: Extract latent predictions
# =====================================================================
print("\n" + "=" * 60)
print("FASE 3: Latent predictions")
print("=" * 60)

print("  Parsing datasets...")
# reject_invalid=False: keep all 14 PLAsTiCC classes (including galactic)
# Our meta uses integer target IDs, not string class names
train_parsed = parsnip.parse_dataset(train_dataset, kind="plasticc", reject_invalid=False)
test_parsed = parsnip.parse_dataset(test_dataset, kind="plasticc", reject_invalid=False)

print("  Predicting train set...")
t0 = time.time()
train_predictions = model.predict_dataset(train_parsed)
# Add original_object_id (required by ParSNIP's Classifier for fold splitting)
if "original_object_id" not in train_predictions.colnames:
    train_predictions["original_object_id"] = train_predictions["object_id"]
print(f"  Train: {len(train_predictions)} predictions in {time.time()-t0:.1f}s")

print("  Predicting test set...")
t0 = time.time()
test_predictions = model.predict_dataset(test_parsed)
if "original_object_id" not in test_predictions.colnames:
    test_predictions["original_object_id"] = test_predictions["object_id"]
print(f"  Test: {len(test_predictions)} predictions in {time.time()-t0:.1f}s")

# =====================================================================
# FASE 4: Downstream LightGBM classifier (14-class)
# =====================================================================
print("\n" + "=" * 60)
print("FASE 4: LightGBM classifier")
print("=" * 60)

from src.evaluation.metrics import compute_metrics

# Labels
train_labels = np.array(train_dataset.meta["type"])
test_labels = np.array(test_dataset.meta["type"])
unique_targets = sorted(set(train_labels) | set(test_labels))
target_to_idx = {t: i for i, t in enumerate(unique_targets)}
y_train = np.array([target_to_idx[t] for t in train_labels])
y_test = np.array([target_to_idx[t] for t in test_labels])
n_classes = len(unique_targets)

print(f"  Train: {len(y_train)}, Test: {len(y_test)}, Classes: {n_classes}")

# Extract features from predictions table
CLASSIFIER_KEYS = [
    'color', 'color_error', 's1', 's1_error', 's2', 's2_error',
    's3', 's3_error', 'luminosity', 'luminosity_error', 'reference_time_error',
]

train_features = np.column_stack([np.array(train_predictions[k], dtype=float) for k in CLASSIFIER_KEYS])
test_features = np.column_stack([np.array(test_predictions[k], dtype=float) for k in CLASSIFIER_KEYS])

# Impute NaN (luminosity for z=0 galactic objects) with column median
for col_idx in range(train_features.shape[1]):
    nan_mask = np.isnan(train_features[:, col_idx])
    if nan_mask.any():
        median_val = np.nanmedian(train_features[:, col_idx])
        train_features[nan_mask, col_idx] = median_val
        print(f"  Imputed {nan_mask.sum()} NaN in train {CLASSIFIER_KEYS[col_idx]} → median={median_val:.4f}")
for col_idx in range(test_features.shape[1]):
    nan_mask = np.isnan(test_features[:, col_idx])
    if nan_mask.any():
        median_val = np.nanmedian(test_features[:, col_idx])
        test_features[nan_mask, col_idx] = median_val
        print(f"  Imputed {nan_mask.sum()} NaN in test {CLASSIFIER_KEYS[col_idx]} → median={median_val:.4f}")

print(f"  Features: {train_features.shape[1]}, NaN remaining: {np.isnan(train_features).sum() + np.isnan(test_features).sum()}")

# Train LightGBM directly (bypass ParSNIP's Classifier which has LightGBM 4.6 compat issues)
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight("balanced", y_train)

dtrain = lgb.Dataset(
    train_features, label=y_train, weight=sample_weights,
    feature_name=CLASSIFIER_KEYS, free_raw_data=False,
)

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "min_child_weight": 50,  # ParSNIP default is 1000 but too high for 5493 samples
    "seed": SEED,
    "verbose": -1,
    "n_jobs": -1,
}

print("  Training LightGBM (500 rounds)...")
t0 = time.time()
booster = lgb.train(lgb_params, dtrain, num_boost_round=500)
print(f"  Training: {time.time()-t0:.1f}s, n_trees={booster.num_trees()}")

y_proba = booster.predict(test_features)
y_pred = np.argmax(y_proba, axis=1)

metrics = compute_metrics(y_test, y_pred, y_proba)

print(f"\n  *** RISULTATI PARSNIP (seed {SEED}) ***")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  F1 macro:  {metrics['f1_macro']:.4f}")
print(f"  AUC:       {metrics.get('auc_ovr', 'N/A')}")

# Save
booster.save_model(str(CHECKPOINT_DIR / "parsnip_lgb.txt"))
np.savez(
    SEED_DIR / "parsnip_predictions.npz",
    y_pred=y_pred, y_proba=y_proba, y_test=y_test,
)
np.savez(
    SEED_DIR / "parsnip_classifier_features.npz",
    features=test_features,
    feature_names=np.array(CLASSIFIER_KEYS),
)
np.savez(
    SEED_DIR / "parsnip_train_features.npz",
    features=train_features,
    labels=y_train,
    feature_names=np.array(CLASSIFIER_KEYS),
)

# =====================================================================
# FASE 5: XAI metrics
# =====================================================================
print("\n" + "=" * 60)
print("FASE 5: XAI metrics")
print("=" * 60)

# Compute XAI directly (bypass ParSNIP's Classifier wrapper)
import shap
from src.xai.shap_utils import normalize_shap_values
from src.xai.metrics import faithfulness_correlation, explanation_complexity, plausibility_score, build_expert_mask_tabular
from src.models.avocado.xai import AvocadoModelWrapper  # reuse booster wrapper

wrapper = AvocadoModelWrapper(booster)

rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(test_features), size=min(50, len(test_features)), replace=False)

print("  Computing TreeSHAP...")
explainer = shap.TreeExplainer(booster)
mean_shap = normalize_shap_values(explainer.shap_values(test_features[sample_idx]))
print(f"  SHAP shape: {mean_shap.shape}, sum |SHAP|: {np.abs(mean_shap).sum():.4f}")

# Faithfulness
faith_vals = [faithfulness_correlation(wrapper, test_features[i], mean_shap[j], seed=SEED)
              for j, i in enumerate(sample_idx)]
# Complexity
compl_vals = [explanation_complexity(mean_shap[j]) for j in range(len(sample_idx))]
# Plausibility (project to 60-d)
from src.models.parsnip.xai import _compute_plausibility_parsnip
plaus_vals = _compute_plausibility_parsnip(mean_shap, CLASSIFIER_KEYS)

xai = {
    "faithfulness_mean": float(np.mean(faith_vals)),
    "faithfulness_std": float(np.std(faith_vals)),
    "complexity_mean": float(np.mean(compl_vals)),
    "complexity_std": float(np.std(compl_vals)),
    "plausibility_mean": float(np.mean(plaus_vals)),
    "plausibility_std": float(np.std(plaus_vals)),
}
metrics.update(xai)

print(f"  Faithfulness: {xai['faithfulness_mean']:.4f} +/- {xai['faithfulness_std']:.4f}")
print(f"  Complexity:   {xai['complexity_mean']:.4f} +/- {xai['complexity_std']:.4f}")
print(f"  Plausibility: {xai['plausibility_mean']:.4f} +/- {xai['plausibility_std']:.4f}")

# =====================================================================
# FASE 6: Sanity check (data randomization only)
# =====================================================================
print("\n" + "=" * 60)
print("FASE 6: Sanity check")
print("=" * 60)

# Data randomization: retrain on permuted training labels
from scipy.stats import spearmanr

rng_san = np.random.default_rng(SEED)
y_train_perm = rng_san.permutation(y_train)

dtrain_perm = lgb.Dataset(train_features, label=y_train_perm, free_raw_data=False)
booster_perm = lgb.train(lgb_params, dtrain_perm, num_boost_round=500)

san_idx = rng_san.choice(len(test_features), size=min(50, len(test_features)), replace=False)
x_san = test_features[san_idx]

attr_orig = normalize_shap_values(shap.TreeExplainer(booster).shap_values(x_san))
attr_rand = normalize_shap_values(shap.TreeExplainer(booster_perm).shap_values(x_san))

rho, _ = spearmanr(attr_orig.flatten(), attr_rand.flatten())
metrics["data_randomization_rho"] = float(rho) if not np.isnan(rho) else None
print(f"  Data randomization rho: {rho:.4f}")

# =====================================================================
# SALVATAGGIO
# =====================================================================
print("\n" + "=" * 60)
print("SALVATAGGIO")
print("=" * 60)

results_path = SEED_DIR / "results.json"
existing = {}
if results_path.is_file():
    with open(results_path) as f:
        existing = json.load(f)
existing["ParSNIP"] = metrics
with open(results_path, "w") as f:
    json.dump(existing, f, indent=2)

print(f"  Salvato in {results_path}")
print(f"\n{'='*60}")
print("PARSNIP SEED 42 COMPLETATO")
print(f"{'='*60}")
