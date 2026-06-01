# -*- coding: utf-8 -*-
"""
Avocado seed-42 pipeline — run in Spyder.

GP augmentation 10x, feature extraction seriale, LightGBM, XAI, sanity.
Tempo stimato: ~2h totali. RAM picco: ~2-3 GB.
"""

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

SEED = 42
SEED_DIR = PROJECT_ROOT / "scripts" / "results" / "seed_42"
CACHE_DIR = SEED_DIR / "avocado_cache"
CHECKPOINT_DIR = SEED_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AUGMENTATION_FACTOR = 10

# =====================================================================
# FASE 1: Augmentation (10×)
# =====================================================================
print("=" * 60)
print("FASE 1: GP Augmentation (10×)")
print("=" * 60)

from src.models.avocado.train import (
    _build_avocado_objects, _set_seed, _make_augmentor, _augment_dataset,
    _extract_features_single, _extract_features,
)
from src.data.raw_loader import RawPLAsTiCCLoader

_set_seed(SEED)
raw_dir = PROJECT_ROOT / "data" / "raw" / "plasticc"
processed_dir = PROJECT_ROOT / "data" / "processed" / "plasticc"

aug_pickle = CACHE_DIR / f"augmented_objects_seed{SEED}_{AUGMENTATION_FACTOR}x.pkl"

if aug_pickle.is_file():
    print(f"  Cache hit: {aug_pickle.name} ({aug_pickle.stat().st_size/1e6:.0f} MB)")
else:
    loader = RawPLAsTiCCLoader(raw_dir=raw_dir, processed_dir=processed_dir)
    train_lc, train_meta = loader.load_split("train")

    print("  Building Avocado objects...")
    train_objects = _build_avocado_objects(train_lc, train_meta)
    print(f"  {len(train_objects)} train objects")

    augmented = _augment_dataset(
        train_objects, raw_dir, AUGMENTATION_FACTOR, SEED, CACHE_DIR,
    )
    del augmented, train_objects
    gc.collect()

print(f"  Augmentation completata.")

# =====================================================================
# FASE 2: Feature extraction
# =====================================================================
print("\n" + "=" * 60)
print("FASE 2: Feature extraction")
print("=" * 60)

FEATURES_CACHE = CACHE_DIR / f"features_train_aug_seed{SEED}_{AUGMENTATION_FACTOR}x.pkl"
LABELS_CACHE = CACHE_DIR / f"labels_train_aug_seed{SEED}_{AUGMENTATION_FACTOR}x.npy"

if FEATURES_CACHE.is_file() and LABELS_CACHE.is_file():
    print(f"  Cache hit: {FEATURES_CACHE.name}")
    train_features = pd.read_pickle(FEATURES_CACHE)
    train_labels = np.load(LABELS_CACHE)
    print(f"  Shape: {train_features.shape}, labels: {len(train_labels)}")
else:
    from avocado.plasticc import PlasticcFeaturizer

    print(f"  Caricamento {aug_pickle.name}...")
    with open(aug_pickle, "rb") as f:
        all_objects = pickle.load(f)
    print(f"  {len(all_objects)} oggetti caricati")

    # Labels
    train_labels = np.array([obj.metadata["target"] for obj in all_objects])
    np.save(LABELS_CACHE, train_labels)

    # Feature extraction seriale con progress
    n = len(all_objects)
    raw_dicts = []
    obj_ids = []
    t0 = time.time()

    for i, obj in enumerate(all_objects):
        d = _extract_features_single(obj)
        raw_dicts.append(d)
        obj_ids.append(obj.metadata["object_id"])

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(
                f"  [{i+1}/{n}] {rate:.1f} obj/s, "
                f"ETA {eta/60:.0f} min"
            )

    del all_objects
    gc.collect()

    total = time.time() - t0
    print(f"  Estrazione completata: {n} oggetti in {total/60:.1f} min")

    # Assemble
    keys = raw_dicts[-1].keys()
    raw_features = pd.DataFrame(
        [list(d.values()) for d in raw_dicts],
        index=obj_ids, columns=keys,
    )
    raw_features.index.name = "object_id"
    del raw_dicts, obj_ids
    gc.collect()

    featurizer = PlasticcFeaturizer()
    train_features = featurizer.select_features(raw_features)
    del raw_features
    gc.collect()

    train_features.to_pickle(FEATURES_CACHE)
    print(f"  Features salvate: {train_features.shape}")

# =====================================================================
# FASE 2b: Test features
# =====================================================================
print("\n" + "=" * 60)
print("FASE 2b: Test feature extraction")
print("=" * 60)

TEST_FEATURES_CACHE = CACHE_DIR / "features_test.pkl"

if TEST_FEATURES_CACHE.is_file():
    print(f"  Cache hit: {TEST_FEATURES_CACHE.name}")
    test_features = pd.read_pickle(TEST_FEATURES_CACHE)
else:
    from src.data.raw_loader import RawPLAsTiCCLoader
    from src.models.avocado.train import _build_avocado_objects, _extract_features

    loader = RawPLAsTiCCLoader(raw_dir=raw_dir, processed_dir=processed_dir)
    test_lc, test_meta = loader.load_split("test")
    test_objects = _build_avocado_objects(test_lc, test_meta)
    print(f"  {len(test_objects)} test objects")

    test_features = _extract_features(test_objects, CACHE_DIR, "test", n_jobs=1)
    del test_objects
    gc.collect()

print(f"  Test features: {test_features.shape}")

# Align columns
common_cols = train_features.columns.intersection(test_features.columns)
train_features = train_features[common_cols]
test_features = test_features[common_cols]
print(f"  Aligned: {len(common_cols)} features")

# =====================================================================
# FASE 3: LightGBM
# =====================================================================
print("\n" + "=" * 60)
print("FASE 3: LightGBM training")
print("=" * 60)

import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
from src.evaluation.metrics import compute_metrics

# Labels
loader = RawPLAsTiCCLoader(raw_dir=raw_dir, processed_dir=processed_dir)
_, test_meta = loader.load_split("test")
test_labels = test_meta["target"].values

unique_targets = sorted(set(train_labels) | set(test_labels))
target_to_idx = {t: i for i, t in enumerate(unique_targets)}
y_train = np.array([target_to_idx[t] for t in train_labels])
y_test = np.array([target_to_idx[t] for t in test_labels])
n_classes = len(unique_targets)

print(f"  Train: {len(y_train)}, Test: {len(y_test)}, Classi: {n_classes}")

sample_weights = compute_sample_weight("balanced", y_train)
dtrain = lgb.Dataset(
    train_features.values, label=y_train, weight=sample_weights,
    feature_name=list(train_features.columns), free_raw_data=False,
)

params = {
    "objective": "multiclass", "num_class": n_classes,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 63, "min_child_weight": 100,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "seed": SEED, "verbose": -1, "n_jobs": -1,
}

t0 = time.time()
booster = lgb.train(params, dtrain, num_boost_round=500)
print(f"  Training: {time.time()-t0:.1f}s")

y_proba = booster.predict(test_features.values)
y_pred = np.argmax(y_proba, axis=1)
metrics = compute_metrics(y_test, y_pred, y_proba)

print(f"\n  *** RISULTATI AVOCADO (seed {SEED}) ***")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  F1 macro:  {metrics['f1_macro']:.4f}")
print(f"  AUC:       {metrics.get('auc_ovr', 'N/A')}")

# Save model + predictions
booster.save_model(str(CHECKPOINT_DIR / "avocado_lgb.txt"))
np.savez(
    SEED_DIR / "avocado_predictions.npz",
    y_pred=y_pred, y_proba=y_proba, y_test=y_test,
    feature_names=np.array(list(train_features.columns)),
    test_features=test_features.values,
)
with open(SEED_DIR / "avocado_feature_names.json", "w") as f:
    json.dump(list(train_features.columns), f)

# =====================================================================
# FASE 4: XAI
# =====================================================================
print("\n" + "=" * 60)
print("FASE 4: XAI metrics")
print("=" * 60)

from src.models.avocado.xai import compute_avocado_xai
xai = compute_avocado_xai(SEED)
metrics.update(xai)

print(f"  Faithfulness: {xai['faithfulness_mean']:.4f} +/- {xai['faithfulness_std']:.4f}")
print(f"  Complexity:   {xai['complexity_mean']:.4f} +/- {xai['complexity_std']:.4f}")
print(f"  Plausibility: {xai['plausibility_mean']:.4f} +/- {xai['plausibility_std']:.4f}")

# =====================================================================
# FASE 5: Sanity check
# =====================================================================
print("\n" + "=" * 60)
print("FASE 5: Sanity check (data randomization)")
print("=" * 60)

from src.models.avocado.sanity import data_randomization_test
rho = data_randomization_test(seed=SEED)
metrics["data_randomization_rho"] = rho
print(f"  Data randomization rho: {rho:.4f}")

# =====================================================================
# SALVATAGGIO FINALE
# =====================================================================
print("\n" + "=" * 60)
print("SALVATAGGIO")
print("=" * 60)

results_path = SEED_DIR / "results.json"
existing = {}
if results_path.is_file():
    with open(results_path) as f:
        existing = json.load(f)
existing["Avocado"] = metrics
with open(results_path, "w") as f:
    json.dump(existing, f, indent=2)
print(f"  Salvato in {results_path}")

print(f"\n{'='*60}")
print("AVOCADO SEED 42 COMPLETATO")
print(f"{'='*60}")
