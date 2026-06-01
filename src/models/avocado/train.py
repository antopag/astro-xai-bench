"""Avocado (Boone 2019) training pipeline for astro-xai-bench.

Reproduces the published PLAsTiCC-winning pipeline:
  1. Load raw PLAsTiCC light curves (with MJD + redshift)
  2. GP augmentation (100× factor per Boone 2019 defaults)
  3. Feature extraction via PlasticcFeaturizer
  4. LightGBM classification (14-class)

Predictions are saved in the same JSON format as the other base models.

NOTE: Our Avocado will likely underperform Boone's published numbers because
he trained on the full 3.5M PLAsTiCC test set as unlabeled augmentation source,
while we restrict to the 7,848 unblinded training set. This is documented in
the paper discussion as a fair caveat.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.plasticc import CLASS_MAP, PASSBAND_NAMES
from src.data.raw_loader import RawPLAsTiCCLoader

# Avocado imports (will fail early if not installed)
import avocado
from avocado.plasticc import PlasticcAugmentor, PlasticcFeaturizer
from avocado.classifier import LightGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SEEDS = [42, 123, 456, 789, 1024]
# DECISION: Augmentation factor reduced from Boone 2019's 100× to 10×.
# Reason: The 100× pipeline generates ~550K augmented objects whose pickle
# exceeds available RAM (5.1 GB pickle vs ~8 GB free on the workstation).
# With 10×, training data (~47K objects) still substantially exceeds the
# unaugmented set (5,493) and includes the full redshift augmentation per
# Boone's protocol. Classification performance may represent a lower bound
# relative to the original 100× configuration.
# See REVISION_AVOCADO_DECISIONS.md §1 for the paper-text insert.
AUGMENTATION_FACTOR = 10


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _build_avocado_objects(
    lc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> list[avocado.AstronomicalObject]:
    """Convert raw PLAsTiCC DataFrames to Avocado AstronomicalObject list.

    Args:
        lc_df: Light curve DataFrame (object_id, mjd, passband, flux, flux_err, detected).
        meta_df: Metadata DataFrame (object_id, hostgal_specz, hostgal_photoz, target, …).

    Returns:
        List of AstronomicalObject instances.
    """
    objects = []
    grouped_lc = lc_df.groupby("object_id")

    for _, row in meta_df.iterrows():
        oid = row["object_id"]
        if oid not in grouped_lc.groups:
            continue

        obj_lc = grouped_lc.get_group(oid).sort_values("mjd")

        # Avocado expects columns: time, band, flux, flux_error
        observations = pd.DataFrame({
            "time": obj_lc["mjd"].values,
            "band": [f"lsst{PASSBAND_NAMES[int(pb)].lower()}" for pb in obj_lc["passband"].values],
            "flux": obj_lc["flux"].values,
            "flux_error": obj_lc["flux_err"].values,
        })

        metadata = {
            "object_id": oid,
            "host_specz": float(row.get("hostgal_specz", 0.0)),
            "host_photoz": float(row.get("hostgal_photoz", 0.0)),
            "host_photoz_error": float(row.get("hostgal_photoz_err", 0.0)),
            "ra": float(row.get("ra", 0.0)),
            "decl": float(row.get("decl", 0.0)),
            "mwebv": float(row.get("mwebv", 0.0)),
            "ddf": bool(row.get("ddf", False)),
            "redshift": float(row.get("hostgal_specz", 0.0)),
            "galactic": bool(float(row.get("hostgal_specz", 0.0)) == 0.0),
            "target": int(row["target"]),
            "class": int(row["target"]),  # Avocado uses 'class' internally
        }

        obj = avocado.AstronomicalObject(metadata, observations)
        objects.append(obj)

    return objects


def _make_augmentor(raw_dir: Path) -> PlasticcAugmentor:
    """Create a PlasticcAugmentor with photo-z reference from PLAsTiCC test metadata.

    Avocado's augmentation (Boone 2019, §3.6) samples redshifts from the
    photo-z distribution of the full PLAsTiCC test-set metadata. The default
    PlasticcAugmentor expects this data as an HDF5 file; we load from the
    original CSV instead.

    NOTE: Only the test-set *metadata* (photo-z column) is used — as the
    empirical redshift distribution for augmentation sampling. Test-set
    light curves and labels are never accessed.
    """
    from astropy.cosmology import FlatLambdaCDM

    test_meta_path = raw_dir / "test_set_metadata.csv"
    if not test_meta_path.is_file():
        raise FileNotFoundError(
            f"{test_meta_path} not found. "
            "Run `python scripts/fetch_plasticc.py` to download it."
        )

    logger.info("Loading PLAsTiCC test-set metadata for photo-z reference…")
    test_meta = pd.read_csv(test_meta_path)
    mask = test_meta["hostgal_specz"] > 0
    ref_data = np.column_stack([
        test_meta.loc[mask, "hostgal_specz"].values,
        test_meta.loc[mask, "hostgal_photoz"].values,
        test_meta.loc[mask, "hostgal_photoz_err"].values,
    ])

    # Create augmentor, bypassing __init__ to avoid HDF5 file load,
    # but manually setting all required attributes
    augmentor = object.__new__(PlasticcAugmentor)
    augmentor.cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    augmentor._photoz_reference = ref_data

    logger.info(
        f"Photo-z reference: {len(ref_data)} objects from test_set_metadata.csv "
        f"(specz range {ref_data[:, 0].min():.3f}–{ref_data[:, 0].max():.3f})"
    )
    return augmentor


def _augment_dataset(
    objects: list[avocado.AstronomicalObject],
    raw_dir: Path,
    augmentation_factor: int,
    seed: int,
    cache_dir: Path,
) -> list[avocado.AstronomicalObject]:
    """GP-augment the training objects.

    Augmented objects are cached to disk to avoid re-computation.
    """
    cache_path = cache_dir / f"augmented_objects_seed{seed}_{augmentation_factor}x.pkl"

    if cache_path.is_file():
        logger.info(f"Loading cached augmented objects from {cache_path}")
        with open(cache_path, "rb") as f:
            augmented = pickle.load(f)
        logger.info(f"  Loaded {len(augmented)} augmented objects")
        return augmented

    _set_seed(seed)
    augmentor = _make_augmentor(raw_dir)

    augmented = list(objects)  # start with originals
    n_originals = len(objects)
    n_target = n_originals * augmentation_factor

    logger.info(
        f"Augmenting {n_originals} objects × {augmentation_factor} "
        f"= {n_target} target augmented objects…"
    )

    n_generated = 0
    n_failed = 0
    for i, ref_obj in enumerate(objects):
        for _ in range(augmentation_factor - 1):  # -1 because original is included
            aug_obj = augmentor.augment_object(ref_obj, force_success=False)
            if aug_obj is not None:
                augmented.append(aug_obj)
                n_generated += 1
            else:
                n_failed += 1

        if (i + 1) % 500 == 0:
            logger.info(
                f"  [{i + 1}/{n_originals}] generated={n_generated}, failed={n_failed}"
            )

    logger.info(
        f"Augmentation complete: {len(augmented)} total objects "
        f"({n_generated} generated, {n_failed} failed)"
    )

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(augmented, f)
    logger.info(f"Cached augmented objects to {cache_path}")

    return augmented


def _extract_features_single(obj: avocado.AstronomicalObject) -> dict:
    """Extract raw features for a single object (picklable for joblib)."""
    featurizer = PlasticcFeaturizer()
    return featurizer.extract_raw_features(obj)


def _extract_features(
    objects: list[avocado.AstronomicalObject],
    cache_dir: Path,
    tag: str,
    n_jobs: int = -1,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Extract Avocado features from a list of objects.

    Uses joblib for parallel GP fitting, processing in chunks to
    limit peak memory usage (~5K objects ≈ 60 MB raw features).

    Args:
        objects: List of AstronomicalObject instances.
        cache_dir: Directory for caching extracted features.
        tag: Cache key suffix.
        n_jobs: Number of parallel workers (-1 = all cores).
        chunk_size: Number of objects per processing chunk.

    Returns:
        DataFrame with shape (n_objects, n_features). Index is object_id.
    """
    cache_path = cache_dir / f"features_{tag}.pkl"

    if cache_path.is_file():
        logger.info(f"Loading cached features from {cache_path}")
        features = pd.read_pickle(cache_path)
        logger.info(f"  Loaded features: {features.shape}")
        return features

    from joblib import Parallel, delayed

    n_objects = len(objects)
    n_chunks = (n_objects + chunk_size - 1) // chunk_size
    logger.info(
        f"Extracting raw features for {n_objects} objects "
        f"(n_jobs={n_jobs}, {n_chunks} chunks of {chunk_size})…"
    )

    all_raw_dicts = []
    all_object_ids = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_objects)
        chunk = objects[start:end]

        logger.info(f"  Chunk {chunk_idx + 1}/{n_chunks}: objects {start}–{end - 1}")

        chunk_dicts = Parallel(n_jobs=n_jobs, verbose=5, batch_size="auto")(
            delayed(_extract_features_single)(obj) for obj in chunk
        )

        all_raw_dicts.extend(chunk_dicts)
        all_object_ids.extend(obj.metadata["object_id"] for obj in chunk)

    # Assemble into DataFrame
    keys = all_raw_dicts[-1].keys()
    raw_features = pd.DataFrame(
        [list(d.values()) for d in all_raw_dicts],
        index=all_object_ids,
        columns=keys,
    )
    raw_features.index.name = "object_id"
    logger.info(f"  Raw features shape: {raw_features.shape}")

    # Select classifier-ready features
    featurizer = PlasticcFeaturizer()
    features = featurizer.select_features(raw_features)
    logger.info(f"  Selected features shape: {features.shape}")

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_pickle(cache_path)
    logger.info(f"Cached features to {cache_path}")

    return features


def _extract_features_from_pickle(
    pickle_path: Path,
    cache_dir: Path,
    tag: str,
    n_jobs: int = -1,
    chunk_size: int = 5000,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract features from a cached pickle of augmented objects.

    Streams the pickle in chunks to avoid loading all objects into memory
    at once (the augmented-object pickle can exceed 5 GB).

    Args:
        pickle_path: Path to the augmented objects pickle.
        cache_dir: Directory for caching extracted features.
        tag: Cache key suffix.
        n_jobs: Number of parallel workers.
        chunk_size: Objects per processing chunk.

    Returns:
        (features_df, labels_array): Selected features and target labels.
    """
    features_cache = cache_dir / f"features_{tag}.pkl"
    labels_cache = cache_dir / f"labels_{tag}.npy"

    if features_cache.is_file() and labels_cache.is_file():
        logger.info(f"Loading cached features from {features_cache}")
        features = pd.read_pickle(features_cache)
        labels = np.load(labels_cache)
        logger.info(f"  Loaded: {features.shape}, {len(labels)} labels")
        return features, labels

    from joblib import Parallel, delayed

    import gc

    logger.info(f"Loading augmented objects from {pickle_path}…")
    with open(pickle_path, "rb") as f:
        all_objects = pickle.load(f)

    n_objects = len(all_objects)
    n_chunks = (n_objects + chunk_size - 1) // chunk_size
    logger.info(
        f"Extracting features for {n_objects} objects "
        f"({n_chunks} chunks of {chunk_size}, n_jobs={n_jobs})…"
    )

    # Extract labels first (cheap, just metadata access)
    labels = np.array([obj.metadata["target"] for obj in all_objects])

    all_raw_dicts = []
    all_object_ids = []

    # Process from the END of the list so we can pop and free memory
    # as we go, rather than holding all objects until the end
    for chunk_idx in range(n_chunks):
        # Take a chunk from the end (pop is O(1) from end)
        chunk = []
        ids = []
        for _ in range(min(chunk_size, len(all_objects))):
            obj = all_objects.pop()
            ids.append(obj.metadata["object_id"])
            chunk.append(obj)

        logger.info(
            f"  Chunk {chunk_idx + 1}/{n_chunks}: {len(chunk)} objects "
            f"(remaining: {len(all_objects)})"
        )

        chunk_dicts = Parallel(n_jobs=n_jobs, verbose=5, batch_size="auto")(
            delayed(_extract_features_single)(obj) for obj in chunk
        )

        all_raw_dicts.extend(chunk_dicts)
        all_object_ids.extend(ids)

        del chunk, chunk_dicts, ids
        gc.collect()

    del all_objects
    gc.collect()

    # Assemble into DataFrame
    keys = all_raw_dicts[-1].keys()
    raw_features = pd.DataFrame(
        [list(d.values()) for d in all_raw_dicts],
        index=all_object_ids,
        columns=keys,
    )
    raw_features.index.name = "object_id"
    del all_raw_dicts, all_object_ids

    logger.info(f"  Raw features shape: {raw_features.shape}")

    featurizer = PlasticcFeaturizer()
    features = featurizer.select_features(raw_features)
    del raw_features
    logger.info(f"  Selected features shape: {features.shape}")

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_pickle(features_cache)
    np.save(labels_cache, labels)
    logger.info(f"Cached features to {features_cache}")

    return features, labels


def train_avocado_single_seed(
    seed: int,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    augmentation_factor: int = AUGMENTATION_FACTOR,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Train Avocado for a single seed.

    Returns:
        Dict with classification metrics in the same schema as other models.
    """
    from src.evaluation.metrics import compute_metrics

    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "plasticc"
    if processed_dir is None:
        processed_dir = PROJECT_ROOT / "data" / "processed" / "plasticc"
    if results_dir is None:
        results_dir = PROJECT_ROOT / "scripts" / "results"

    seed_dir = results_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = seed_dir / "avocado_cache"

    _set_seed(seed)

    # 1. Load raw data
    loader = RawPLAsTiCCLoader(raw_dir=raw_dir, processed_dir=processed_dir)
    train_lc, train_meta = loader.load_split("train")
    test_lc, test_meta = loader.load_split("test")

    # 2. Build Avocado objects
    logger.info("Building Avocado objects…")
    train_objects = _build_avocado_objects(train_lc, train_meta)
    test_objects = _build_avocado_objects(test_lc, test_meta)
    logger.info(f"  Train: {len(train_objects)}, Test: {len(test_objects)}")

    # 3. GP augmentation on training set
    aug_pickle_path = cache_dir / f"augmented_objects_seed{seed}_{augmentation_factor}x.pkl"
    if not aug_pickle_path.is_file():
        augmented_train = _augment_dataset(
            train_objects, raw_dir, augmentation_factor, seed, cache_dir
        )
        del augmented_train  # cached to disk, free memory

    # 4. Feature extraction — stream augmented objects from pickle in chunks
    #    to avoid loading 5+ GB into memory simultaneously
    train_features, train_labels = _extract_features_from_pickle(
        aug_pickle_path, cache_dir, f"train_aug_seed{seed}_{augmentation_factor}x", n_jobs=n_jobs,
    )

    test_features = _extract_features(test_objects, cache_dir, "test", n_jobs=n_jobs)

    # Align feature columns (test may lack some augmentation-derived features)
    common_cols = train_features.columns.intersection(test_features.columns)
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]

    # 5. Prepare labels
    test_labels = np.array([obj.metadata["target"] for obj in test_objects])

    # Map to 0-based indices (same as our existing pipeline)
    unique_targets = sorted(set(train_labels) | set(test_labels))
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    y_test = np.array([target_to_idx[t] for t in test_labels])

    # 6. Train LightGBM directly (bypassing Avocado's built-in Classifier
    #    which has tight coupling to Dataset internals)
    import lightgbm as lgb
    from sklearn.utils.class_weight import compute_sample_weight

    y_train = np.array([target_to_idx[t] for t in train_labels])
    n_classes = len(unique_targets)

    sample_weights = compute_sample_weight("balanced", y_train)

    dtrain = lgb.Dataset(
        train_features.values,
        label=y_train,
        weight=sample_weights,
        feature_name=list(train_features.columns),
        free_raw_data=False,
    )

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbose": -1,
        "n_jobs": -1,
    }

    logger.info(f"Training LightGBM (seed={seed}, n_classes={n_classes})…")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        # No validation set for final training — matches Boone's approach
    )

    # 7. Predict
    y_proba = booster.predict(test_features.values)  # (N, n_classes)
    y_pred = np.argmax(y_proba, axis=1)

    # 8. Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1 macro: {metrics['f1_macro']:.4f}")

    # 9. Save model + predictions
    booster.save_model(str(seed_dir / "checkpoints" / "avocado_lgb.txt"))
    np.savez(
        seed_dir / "avocado_predictions.npz",
        y_pred=y_pred,
        y_proba=y_proba,
        y_test=y_test,
        feature_names=np.array(list(train_features.columns)),
        test_features=test_features.values,
    )

    # Save feature names for XAI
    with open(seed_dir / "avocado_feature_names.json", "w") as f:
        json.dump(list(train_features.columns), f)

    return metrics


def train_all_seeds(
    seeds: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Train Avocado across all seeds and return aggregated results."""
    if seeds is None:
        seeds = SEEDS

    all_results = {}
    for seed in seeds:
        logger.info(f"\n{'='*60}\nAvocado seed {seed}\n{'='*60}")
        metrics = train_avocado_single_seed(seed, **kwargs)
        all_results[seed] = metrics

    # Aggregate
    metric_keys = list(all_results[seeds[0]].keys())
    aggregated = {}
    for key in metric_keys:
        vals = [all_results[s][key] for s in seeds if isinstance(all_results[s].get(key), (int, float))]
        if vals:
            aggregated[key] = {
                "mean_seed": float(np.mean(vals)),
                "std_seed": float(np.std(vals, ddof=1)),
                "values": vals,
            }

    logger.info(f"\nAvocado aggregated (5 seeds):")
    for k in ["accuracy", "f1_macro", "auc_ovr"]:
        if k in aggregated:
            logger.info(f"  {k}: {aggregated[k]['mean_seed']:.4f} ± {aggregated[k]['std_seed']:.4f}")

    return {"per_seed": all_results, "aggregated": aggregated}


if __name__ == "__main__":
    train_all_seeds()
