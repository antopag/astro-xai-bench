# -*- coding: utf-8 -*-
"""
Avocado — remaining 4 seeds (123, 456, 789, 1024). Run in Spyder.

Seed 42 already completed. Each seed: ~30 min augmentation + ~1.5h features.
Total: ~8h. All phases cachable — safe to interrupt and resume.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.models.avocado.train import (
    _build_avocado_objects, _set_seed, _make_augmentor, _augment_dataset,
    _extract_features_single, _extract_features,
)
from src.data.raw_loader import RawPLAsTiCCLoader
from avocado.plasticc import PlasticcFeaturizer
from src.evaluation.metrics import compute_metrics

SEEDS = [123, 456, 789, 1024]
AUGMENTATION_FACTOR = 10
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "plasticc"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"

# Load raw data once (shared across seeds)
loader = RawPLAsTiCCLoader(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
train_lc, train_meta = loader.load_split("train")
test_lc, test_meta = loader.load_split("test")

test_labels_raw = test_meta["target"].values

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'#'*60}")
    print(f"# SEED {SEED} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'#'*60}")

    seed_dir = RESULTS_DIR / f"seed_{SEED}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = seed_dir / "avocado_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    aug_pickle = cache_dir / f"augmented_objects_seed{SEED}_{AUGMENTATION_FACTOR}x.pkl"
    features_cache = cache_dir / f"features_train_aug_seed{SEED}_{AUGMENTATION_FACTOR}x.pkl"
    labels_cache = cache_dir / f"labels_train_aug_seed{SEED}_{AUGMENTATION_FACTOR}x.npy"
    test_features_cache = cache_dir / "features_test.pkl"

    # ---- FASE 1: Augmentation ----
    print(f"\n  FASE 1: Augmentation (10x)")
    if aug_pickle.is_file():
        print(f"    Cache hit ({aug_pickle.stat().st_size/1e6:.0f} MB)")
    else:
        _set_seed(SEED)
        train_objects = _build_avocado_objects(train_lc, train_meta)
        augmented = _augment_dataset(
            train_objects, RAW_DIR, AUGMENTATION_FACTOR, SEED, cache_dir,
        )
        del augmented, train_objects
        gc.collect()
    print(f"    Augmentation OK")

    # ---- FASE 2: Feature extraction ----
    print(f"\n  FASE 2: Feature extraction")
    if features_cache.is_file() and labels_cache.is_file():
        print(f"    Cache hit")
        train_features = pd.read_pickle(features_cache)
        train_labels = np.load(labels_cache)
    else:
        print(f"    Caricamento {aug_pickle.name}...")
        with open(aug_pickle, "rb") as f:
            all_objects = pickle.load(f)
        print(f"    {len(all_objects)} oggetti")

        train_labels = np.array([obj.metadata["target"] for obj in all_objects])
        np.save(labels_cache, train_labels)

        n = len(all_objects)
        raw_dicts = []
        obj_ids = []
        t0 = time.time()
        for i, obj in enumerate(all_objects):
            d = _extract_features_single(obj)
            raw_dicts.append(d)
            obj_ids.append(obj.metadata["object_id"])
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                print(f"    [{i+1}/{n}] {rate:.1f} obj/s, ETA {eta/60:.0f} min")

        del all_objects
        gc.collect()
        print(f"    Estrazione: {n} oggetti in {(time.time()-t0)/60:.1f} min")

        keys = raw_dicts[-1].keys()
        raw_features = pd.DataFrame(
            [list(d.values()) for d in raw_dicts], index=obj_ids, columns=keys,
        )
        raw_features.index.name = "object_id"
        del raw_dicts, obj_ids
        gc.collect()

        featurizer = PlasticcFeaturizer()
        train_features = featurizer.select_features(raw_features)
        del raw_features
        gc.collect()
        train_features.to_pickle(features_cache)

    print(f"    Train features: {train_features.shape}")

    # ---- FASE 2b: Test features ----
    if test_features_cache.is_file():
        test_features = pd.read_pickle(test_features_cache)
    else:
        _set_seed(SEED)
        test_objects = _build_avocado_objects(test_lc, test_meta)
        test_features = _extract_features(test_objects, cache_dir, "test", n_jobs=1)
        del test_objects
        gc.collect()

    common_cols = train_features.columns.intersection(test_features.columns)
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]

    # ---- FASE 3: LightGBM ----
    print(f"\n  FASE 3: LightGBM")
    import lightgbm as lgb
    from sklearn.utils.class_weight import compute_sample_weight

    unique_targets = sorted(set(train_labels) | set(test_labels_raw))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    y_train = np.array([target_to_idx[t] for t in train_labels])
    y_test = np.array([target_to_idx[t] for t in test_labels_raw])
    n_classes = len(unique_targets)

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
    booster = lgb.train(params, dtrain, num_boost_round=500)
    y_proba = booster.predict(test_features.values)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"    Acc: {metrics['accuracy']:.4f}, F1m: {metrics['f1_macro']:.4f}, AUC: {metrics.get('auc_ovr', 'N/A')}")

    booster.save_model(str(ckpt_dir / "avocado_lgb.txt"))
    np.savez(
        seed_dir / "avocado_predictions.npz",
        y_pred=y_pred, y_proba=y_proba, y_test=y_test,
        feature_names=np.array(list(train_features.columns)),
        test_features=test_features.values,
    )
    with open(seed_dir / "avocado_feature_names.json", "w") as f:
        json.dump(list(train_features.columns), f)

    # ---- FASE 4: XAI ----
    print(f"\n  FASE 4: XAI")
    from src.models.avocado.xai import compute_avocado_xai
    xai = compute_avocado_xai(SEED)
    metrics.update(xai)
    print(f"    Faith: {xai['faithfulness_mean']:.3f}, Compl: {xai['complexity_mean']:.3f}, Plaus: {xai['plausibility_mean']:.3f}")

    # ---- SAVE ----
    results_path = seed_dir / "results.json"
    existing = {}
    if results_path.is_file():
        with open(results_path) as f:
            existing = json.load(f)
    existing["Avocado"] = metrics
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"    Seed {SEED} salvato.")

    del train_features, train_labels, booster, dtrain
    gc.collect()

# ---- SUMMARY ----
print(f"\n{'='*60}")
print("SUMMARY (5 seeds including seed 42)")
print(f"{'='*60}")

all_seeds = [42] + SEEDS
for k in ["accuracy", "f1_macro", "auc_ovr", "faithfulness_mean", "complexity_mean", "plausibility_mean"]:
    vals = []
    for s in all_seeds:
        rp = RESULTS_DIR / f"seed_{s}" / "results.json"
        with open(rp) as f:
            d = json.load(f)
        v = d.get("Avocado", {}).get(k)
        if isinstance(v, (int, float)):
            vals.append(v)
    if vals:
        print(f"  {k}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}")
