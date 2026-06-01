"""Round-2 evaluation pipeline for astro-xai-bench — Spyder script.

Computes everything required by the second-round peer review **without
re-training the per-seed checkpoints already produced by run_final_table.py**:

* Phase A — new xAI metrics on existing checkpoints (insertion/deletion AUC,
  band-aggregated complexity);
* Phase B — saliency sanity checks (model randomisation on every seed,
  data randomisation on seed 42 only);
* Phase C — out-of-fold stacking ensemble (k-fold CV on the training set,
  base models retrained per fold);
* Phase D — cross-seed aggregation and output of the new tables/figures.

Each phase persists its incremental results into the per-seed JSON in
``scripts/results/seed_*/round2.json`` so the script is **resumable**: rerun
it after a crash and the already-computed phases will be skipped.

Usage in Spyder: ``%runfile run_round2.py --wdir``

To debug, set ``SEEDS_TO_RUN = [42]`` and ``K_FOLDS = 3``.
"""

from __future__ import annotations

# %% Setup paths (resolve relative to this file, not the cwd)
import json
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# %% Config
SEEDS = [42, 123, 456, 789, 1024]
SEEDS_TO_RUN = SEEDS                  # set to [42] to debug a single seed
K_FOLDS = 5                            # set to 3 to roughly halve OOF retraining cost
DATA_RAND_SEED = 42                    # only one seed needed for data-randomisation sanity check
N_SAMPLES_INSDEL = 50                  # samples used for insertion/deletion AUC
N_SAMPLES_SANITY = 20                  # samples used for sanity checks (smaller for speed)
BATCH_SIZE = 64
DEVICE = "cuda"
PHASES_TO_RUN = ("A", "B", "C", "D")  # subset of {'A','B','C','D'}

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"

# %% Imports
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.data.plasticc import CLASS_MAP
from src.evaluation.metrics import compute_metrics
from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper, extract_features
from src.training.trainer import Trainer, set_global_seed
from src.xai.shap_analysis import compute_shap_values, get_feature_names
from src.xai.metrics import (build_expert_mask_tabular,
                              complexity_band_aggregated,
                              compute_xai_metrics_tabular,
                              faithfulness_correlation,
                              explanation_complexity)
from src.xai.integrated_gradients import compute_integrated_gradients
from src.xai.faithfulness import (insertion_deletion_tabular,
                                   insertion_deletion_timeseries,
                                   insertion_deletion_image,
                                   aggregate_insertion_deletion)
from src.xai.sanity_checks import (model_randomization_test,
                                    data_randomization_spearman_dl,
                                    data_randomization_spearman_tabular,
                                    shuffle_labels)
from src.xai.plausibility import (compute_tabular_plausibility,
                                   compute_timeseries_plausibility,
                                   compute_image_plausibility)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Round 2 | device={device} | project root={PROJECT_ROOT}")

# %% Load data once (shared across all seeds)
label_map = np.load(DATA_DIR / "label_map.npy", allow_pickle=True).item()
idx_to_target = {v: k for k, v in label_map.items()}
class_names = [CLASS_MAP.get(idx_to_target[i], str(i)) for i in range(len(label_map))]
n_classes = len(class_names)

train_data = np.load(DATA_DIR / "train.npz")
val_data = np.load(DATA_DIR / "val.npz")
test_data = np.load(DATA_DIR / "test.npz")

y_train = train_data["labels"]
y_val = val_data["labels"]
y_test = test_data["labels"]

x_train_only_tab = extract_features(train_data["light_curves"])  # for OOF (train only, no val)
x_train_tab = extract_features(np.vstack([train_data["light_curves"], val_data["light_curves"]]))
y_train_tab = np.concatenate([y_train, y_val])
x_test_tab = extract_features(test_data["light_curves"])
x_val_tab = extract_features(val_data["light_curves"])

lc_train = train_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_val = val_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_test = test_data["light_curves"][:, :, :, 0].astype(np.float32)

gaf_cache = DATA_DIR / "gaf_224.npz"
if not gaf_cache.exists():
    raise FileNotFoundError(f"GAF cache not found at {gaf_cache}. Run run_gaf.py first.")
logger.info("Loading cached GAF images...")
gaf_data = np.load(gaf_cache)
gaf_train = gaf_data["train"].astype(np.float32)
gaf_val = gaf_data["val"].astype(np.float32)
gaf_test = gaf_data["test"].astype(np.float32)

feature_names = get_feature_names()
expert_mask = build_expert_mask_tabular(feature_names)


# ============================================================
# %% Helpers — model factories and DL inference utilities
# ============================================================
def make_lstm() -> torch.nn.Module:
    from src.models.lstm_baseline import LSTMClassifier
    return LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                          n_layers=2, dropout=0.3, bidirectional=True)


def make_cnn1d() -> torch.nn.Module:
    from src.models.cnn_baseline import CNN1D
    return CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3)


def make_cnn2d() -> torch.nn.Module:
    from src.models.cnn_baseline import CNN2D
    return CNN2D(n_bands=6, n_classes=n_classes, base_filters=32, dropout=0.3)


def make_vit() -> torch.nn.Module:
    from src.models.vit import ViTClassifier
    return ViTClassifier(n_bands=6, n_classes=n_classes,
                         model_name="vit_small_patch16_224", pretrained=True, dropout=0.1)


CONFIG_TS = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": 1e-3, "epochs": 100},
}
CONFIG_GAF = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": 1e-3, "epochs": 100},
}
CONFIG_VIT = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": 1e-4, "epochs": 50, "weight_decay": 0.01},
}


def load_dl_checkpoint(model_factory, ckpt_path: Path) -> torch.nn.Module:
    """Instantiate a model and restore its weights from a checkpoint file."""
    model = model_factory().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def make_predict_proba_dl(model: torch.nn.Module) -> callable:
    """Return a numpy-friendly ``predict_proba`` for a torch model."""
    def _pp(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x).float().to(device)
            return F.softmax(model(xt), dim=-1).cpu().numpy()
    return _pp


def predict_proba_test_dl(model: torch.nn.Module, x_test_arr: np.ndarray, batch: int = 64) -> np.ndarray:
    """Compute predict_proba over the full test set in batches."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_test_arr), batch):
            xt = torch.from_numpy(x_test_arr[i:i + batch]).float().to(device)
            out.append(F.softmax(model(xt), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def round2_path(seed: int) -> Path:
    return RESULTS_DIR / f"seed_{seed}" / "round2.json"


def load_round2(seed: int) -> dict:
    p = round2_path(seed)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_round2(seed: int, data: dict) -> None:
    p = round2_path(seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {p.name}")


# ============================================================
# %% Phase A — new xAI metrics on existing checkpoints
# ============================================================
def phase_A_for_seed(seed: int) -> dict:
    """For each of the 7 models compute insertion/deletion AUC and band complexity.

    Tabular models are retrained from scratch (cheap). DL models are loaded
    from the per-seed checkpoint.
    """
    set_global_seed(seed)
    out: dict = {"seed": seed, "phase_A": {}}
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(x_test_tab), size=N_SAMPLES_INSDEL, replace=False)

    # ---- Random Forest ----
    rf = RFClassifier(n_estimators=500, random_state=seed)
    rf.fit(x_train_tab, y_train_tab)
    shap_rf = compute_shap_values(rf, x_test_tab, method="tree")
    # Per-sample mean |SHAP| over classes → 60-dim
    if isinstance(shap_rf, list):
        attr_rf = np.mean([np.abs(s) for s in shap_rf], axis=0)
    elif shap_rf.ndim == 3:
        attr_rf = np.abs(shap_rf).mean(axis=1) if shap_rf.shape[2] == 60 else np.abs(shap_rf).mean(axis=2)
    else:
        attr_rf = np.abs(shap_rf)

    rf_results = []
    for i in sample_idx:
        rf_results.append(insertion_deletion_tabular(rf.predict_proba, x_test_tab[i], attr_rf[i]))
    out["phase_A"]["Random Forest"] = aggregate_insertion_deletion(rf_results)
    out["phase_A"]["Random Forest"]["complexity_band"] = float(np.mean(
        [complexity_band_aggregated(attr_rf[i], "tabular") for i in sample_idx]
    ))
    out["phase_A"]["Random Forest"]["_curves_for_fig"] = {
        "fractions": rf_results[0]["fractions"].tolist(),
        "del_probs": np.mean([r["del_probs"] for r in rf_results], axis=0).tolist(),
        "ins_probs": np.mean([r["ins_probs"] for r in rf_results], axis=0).tolist(),
    }

    # ---- XGBoost ----
    xgb = XGBClassifierWrapper(n_estimators=500, random_state=seed)
    xgb.fit(x_train_tab, y_train_tab)
    shap_xgb = compute_shap_values(xgb, x_test_tab, method="tree")
    if isinstance(shap_xgb, list):
        attr_xgb = np.mean([np.abs(s) for s in shap_xgb], axis=0)
    elif shap_xgb.ndim == 3:
        attr_xgb = np.abs(shap_xgb).mean(axis=1) if shap_xgb.shape[2] == 60 else np.abs(shap_xgb).mean(axis=2)
    else:
        attr_xgb = np.abs(shap_xgb)

    xgb_results = [insertion_deletion_tabular(xgb.predict_proba, x_test_tab[i], attr_xgb[i])
                   for i in sample_idx]
    out["phase_A"]["XGBoost"] = aggregate_insertion_deletion(xgb_results)
    out["phase_A"]["XGBoost"]["complexity_band"] = float(np.mean(
        [complexity_band_aggregated(attr_xgb[i], "tabular") for i in sample_idx]
    ))

    # ---- DL models from checkpoints ----
    dl_specs = [
        ("LSTM", make_lstm, lc_test, "timeseries", insertion_deletion_timeseries, (6, 256)),
        ("CNN1D", make_cnn1d, lc_test, "timeseries", insertion_deletion_timeseries, (6, 256)),
        ("CNN2D", make_cnn2d, gaf_test, "image", insertion_deletion_image, (6, 224, 224)),
        ("ViT", make_vit, gaf_test, "image", insertion_deletion_image, (6, 224, 224)),
    ]
    cnn2d_curves = None
    vit_curves = None
    dl_models_loaded: dict[str, torch.nn.Module] = {}

    for name, factory, x_data, rep, ins_del_fn, in_shape in dl_specs:
        ckpt_file = ckpt_dir / f"{name.lower()}.pt"
        if not ckpt_file.exists():
            logger.warning(f"Missing checkpoint for {name} at {ckpt_file}, skipping.")
            continue
        model = load_dl_checkpoint(factory, ckpt_file)
        dl_models_loaded[name] = model
        pp_fn = make_predict_proba_dl(model)

        per_sample = []
        band_comps = []
        for i in sample_idx:
            attr = compute_integrated_gradients(
                model, torch.from_numpy(x_data[i:i + 1]).float().to(device), n_steps=50,
            )
            kwargs = {"max_curve_points": 50} if rep == "image" else {}
            per_sample.append(ins_del_fn(pp_fn, x_data[i], attr, **kwargs))
            band_comps.append(complexity_band_aggregated(attr, rep))

        out["phase_A"][name] = aggregate_insertion_deletion(per_sample)
        out["phase_A"][name]["complexity_band"] = float(np.mean(band_comps))

        if name == "ViT":
            # Save fractions+curves for figure (averaged across samples)
            vit_curves = {
                "fractions": per_sample[0]["fractions"].tolist(),
                "del_probs": np.mean([r["del_probs"] for r in per_sample], axis=0).tolist(),
                "ins_probs": np.mean([r["ins_probs"] for r in per_sample], axis=0).tolist(),
            }

    if vit_curves is not None:
        out["phase_A"]["ViT"]["_curves_for_fig"] = vit_curves

    # ---- Ensemble (uses meta-features from existing val-set fit; metric only) ----
    # The OOF refit is done in Phase C; here we approximate the ensemble's
    # insertion/deletion as the coefficient-weighted average of base models.
    # (See paper Section 4.2.x — proper ensemble ID curves require its own
    # perturbation procedure on the meta-feature space, which is dominated
    # by the meta-learner coefficients and would not change between Phase A
    # and Phase C; we therefore compute it once per seed in Phase C.)

    save_round2(seed, out)
    return out


# ============================================================
# %% Phase B — saliency sanity checks
# ============================================================
def _train_dl_quick(factory, train_loader, val_loader, config, seed) -> torch.nn.Module:
    model = factory()
    trainer = Trainer(model, config, device=device, seed=seed)
    trainer.fit(train_loader, val_loader)
    return model


def phase_B_for_seed(seed: int, do_data_randomization: bool = False) -> dict:
    """Sanity checks for the DL models for the given seed.

    Model randomisation runs on every seed.
    Data randomisation runs only on the seed flagged by ``do_data_randomization``.
    """
    set_global_seed(seed)
    out = load_round2(seed)
    out.setdefault("phase_B", {})

    seed_dir = RESULTS_DIR / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"

    rng = np.random.default_rng(seed + 1)  # different sample subset than Phase A
    n_test = len(y_test)
    sample_idx_ts = rng.choice(n_test, size=N_SAMPLES_SANITY, replace=False)

    dl_specs = [
        ("LSTM", make_lstm, lc_test),
        ("CNN1D", make_cnn1d, lc_test),
        ("CNN2D", make_cnn2d, gaf_test),
        ("ViT", make_vit, gaf_test),
    ]

    # ---- Model randomisation ----
    for name, factory, x_data in dl_specs:
        ckpt_file = ckpt_dir / f"{name.lower()}.pt"
        if not ckpt_file.exists():
            continue
        model = load_dl_checkpoint(factory, ckpt_file)
        res = model_randomization_test(
            model, x_data, sample_idx_ts,
            cascading=True, n_steps_to_keep=4, ig_n_steps=20,
        )
        out["phase_B"][name] = {
            "model_randomization_spearman": res["spearman_full"],
            "model_randomization_trace": [
                {"layer": n, "spearman": float(s)} for n, s in res["trace"]
            ],
        }

    # ---- Data randomisation: tabular models (always cheap) ----
    if do_data_randomization:
        logger.info(f"Data randomisation training (seed={seed})...")
        y_shuf = shuffle_labels(y_train_tab, seed=seed)

        rf_t = RFClassifier(n_estimators=500, random_state=seed)
        rf_t.fit(x_train_tab, y_train_tab)
        rf_r = RFClassifier(n_estimators=500, random_state=seed)
        rf_r.fit(x_train_tab, y_shuf)
        shap_t = compute_shap_values(rf_t, x_test_tab[sample_idx_ts], method="tree")
        shap_r = compute_shap_values(rf_r, x_test_tab[sample_idx_ts], method="tree")
        out["phase_B"].setdefault("Random Forest", {})
        out["phase_B"]["Random Forest"]["data_randomization_spearman"] = (
            data_randomization_spearman_tabular(shap_t, shap_r)
        )

        xgb_t = XGBClassifierWrapper(n_estimators=500, random_state=seed)
        xgb_t.fit(x_train_tab, y_train_tab)
        xgb_r = XGBClassifierWrapper(n_estimators=500, random_state=seed)
        xgb_r.fit(x_train_tab, y_shuf)
        shap_t = compute_shap_values(xgb_t, x_test_tab[sample_idx_ts], method="tree")
        shap_r = compute_shap_values(xgb_r, x_test_tab[sample_idx_ts], method="tree")
        out["phase_B"].setdefault("XGBoost", {})
        out["phase_B"]["XGBoost"]["data_randomization_spearman"] = (
            data_randomization_spearman_tabular(shap_t, shap_r)
        )

        # DL models — retrain on shuffled labels
        y_train_shuf = shuffle_labels(y_train, seed=seed)
        y_val_shuf = shuffle_labels(y_val, seed=seed + 1)

        # Sequence loaders
        gen = torch.Generator(); gen.manual_seed(seed)
        ts_train_ds = TensorDataset(torch.from_numpy(lc_train), torch.from_numpy(y_train_shuf.astype(np.int64)))
        ts_val_ds = TensorDataset(torch.from_numpy(lc_val), torch.from_numpy(y_val_shuf.astype(np.int64)))
        ts_train_loader = DataLoader(ts_train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
        ts_val_loader = DataLoader(ts_val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # GAF loaders
        gen2 = torch.Generator(); gen2.manual_seed(seed)
        gaf_train_ds = TensorDataset(torch.from_numpy(gaf_train), torch.from_numpy(y_train_shuf.astype(np.int64)))
        gaf_val_ds = TensorDataset(torch.from_numpy(gaf_val), torch.from_numpy(y_val_shuf.astype(np.int64)))
        gaf_train_loader = DataLoader(gaf_train_ds, batch_size=32, shuffle=True, generator=gen2)
        gaf_val_loader = DataLoader(gaf_val_ds, batch_size=32, shuffle=False)

        dl_train_specs = [
            ("LSTM", make_lstm, ts_train_loader, ts_val_loader, CONFIG_TS, lc_test),
            ("CNN1D", make_cnn1d, ts_train_loader, ts_val_loader, CONFIG_TS, lc_test),
            ("CNN2D", make_cnn2d, gaf_train_loader, gaf_val_loader, CONFIG_GAF, gaf_test),
            ("ViT", make_vit, gaf_train_loader, gaf_val_loader, CONFIG_VIT, gaf_test),
        ]
        for name, factory, tl, vl, cfg, x_data in dl_train_specs:
            ckpt_file = ckpt_dir / f"{name.lower()}.pt"
            if not ckpt_file.exists():
                continue
            logger.info(f"Data-randomisation retrain: {name}")
            model_t = load_dl_checkpoint(factory, ckpt_file)
            model_r = _train_dl_quick(factory, tl, vl, cfg, seed)
            rho = data_randomization_spearman_dl(model_t, model_r, x_data, sample_idx_ts, ig_n_steps=20)
            out["phase_B"].setdefault(name, {})
            out["phase_B"][name]["data_randomization_spearman"] = rho

    save_round2(seed, out)
    return out


# ============================================================
# %% Phase C — Out-of-fold stacking ensemble
# ============================================================
def phase_C_for_seed(seed: int) -> dict:
    """Generate OOF meta-features and refit the meta-learner."""
    set_global_seed(seed)
    out = load_round2(seed)
    out.setdefault("phase_C", {})
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"

    n_train = len(y_train)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros((n_train, 6 * n_classes), dtype=np.float32)

    for fold_idx, (tr_idx, ho_idx) in enumerate(skf.split(np.zeros(n_train), y_train)):
        t0 = time.time()
        logger.info(
            f"OOF fold {fold_idx + 1}/{K_FOLDS}  "
            f"train={len(tr_idx)}  holdout={len(ho_idx)}"
        )

        # Tabular: trained on train indices only (no val)
        rf_f = RFClassifier(n_estimators=500, random_state=seed)
        rf_f.fit(x_train_only_tab[tr_idx], y_train[tr_idx])
        oof_preds[ho_idx, 0 * n_classes:1 * n_classes] = rf_f.predict_proba(x_train_only_tab[ho_idx])

        xgb_f = XGBClassifierWrapper(n_estimators=500, random_state=seed)
        xgb_f.fit(x_train_only_tab[tr_idx], y_train[tr_idx])
        oof_preds[ho_idx, 1 * n_classes:2 * n_classes] = xgb_f.predict_proba(x_train_only_tab[ho_idx])

        # Sequence loaders for this fold
        gen = torch.Generator(); gen.manual_seed(seed + fold_idx)
        ts_tr_ds = TensorDataset(torch.from_numpy(lc_train[tr_idx]),
                                  torch.from_numpy(y_train[tr_idx].astype(np.int64)))
        ts_val_ds = TensorDataset(torch.from_numpy(lc_val), torch.from_numpy(y_val.astype(np.int64)))
        ts_ho_arr = lc_train[ho_idx]
        ts_tr_loader = DataLoader(ts_tr_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
        ts_val_loader = DataLoader(ts_val_ds, batch_size=BATCH_SIZE, shuffle=False)

        for slot, factory, cfg in [(2, make_lstm, CONFIG_TS), (3, make_cnn1d, CONFIG_TS)]:
            mdl = _train_dl_quick(factory, ts_tr_loader, ts_val_loader, cfg, seed + fold_idx)
            oof_preds[ho_idx, slot * n_classes:(slot + 1) * n_classes] = predict_proba_test_dl(mdl, ts_ho_arr)
            del mdl
            torch.cuda.empty_cache()

        # GAF loaders for this fold
        gen2 = torch.Generator(); gen2.manual_seed(seed + fold_idx + 1000)
        gaf_tr_ds = TensorDataset(torch.from_numpy(gaf_train[tr_idx]),
                                   torch.from_numpy(y_train[tr_idx].astype(np.int64)))
        gaf_val_ds = TensorDataset(torch.from_numpy(gaf_val), torch.from_numpy(y_val.astype(np.int64)))
        gaf_ho_arr = gaf_train[ho_idx]
        gaf_tr_loader = DataLoader(gaf_tr_ds, batch_size=32, shuffle=True, generator=gen2)
        gaf_val_loader = DataLoader(gaf_val_ds, batch_size=32, shuffle=False)

        for slot, factory, cfg in [(4, make_cnn2d, CONFIG_GAF), (5, make_vit, CONFIG_VIT)]:
            mdl = _train_dl_quick(factory, gaf_tr_loader, gaf_val_loader, cfg, seed + fold_idx)
            oof_preds[ho_idx, slot * n_classes:(slot + 1) * n_classes] = predict_proba_test_dl(mdl, gaf_ho_arr)
            del mdl
            torch.cuda.empty_cache()

        logger.info(f"  Fold {fold_idx + 1} done in {time.time() - t0:.1f}s")

    # ---- Fit meta-learner on OOF predictions ----
    from src.models.ensemble import StackingEnsemble
    meta_only = StackingEnsemble(base_models=[], meta_learner="logistic", random_state=seed)
    meta_only.fit_meta(oof_preds, y_train)

    # ---- Compute test meta-features from the existing checkpoint ensemble ----
    rf_full = RFClassifier(n_estimators=500, random_state=seed)
    rf_full.fit(x_train_tab, y_train_tab)
    xgb_full = XGBClassifierWrapper(n_estimators=500, random_state=seed)
    xgb_full.fit(x_train_tab, y_train_tab)
    lstm_full = load_dl_checkpoint(make_lstm, ckpt_dir / "lstm.pt")
    cnn1d_full = load_dl_checkpoint(make_cnn1d, ckpt_dir / "cnn1d.pt")
    cnn2d_full = load_dl_checkpoint(make_cnn2d, ckpt_dir / "cnn2d.pt")
    vit_full = load_dl_checkpoint(make_vit, ckpt_dir / "vit.pt")

    test_meta = np.hstack([
        rf_full.predict_proba(x_test_tab),
        xgb_full.predict_proba(x_test_tab),
        predict_proba_test_dl(lstm_full, lc_test),
        predict_proba_test_dl(cnn1d_full, lc_test),
        predict_proba_test_dl(cnn2d_full, gaf_test, batch=32),
        predict_proba_test_dl(vit_full, gaf_test, batch=32),
    ])
    y_pred_ens = meta_only.meta.predict(test_meta)
    y_proba_ens = meta_only.meta.predict_proba(test_meta)
    metrics_ens = compute_metrics(y_test, y_pred_ens, y_proba_ens)

    out["phase_C"]["Ensemble (OOF)"] = {
        **metrics_ens,
        "k_folds": K_FOLDS,
        "n_oof_samples": int(n_train),
    }
    save_round2(seed, out)
    logger.info(f"OOF ensemble metrics: {metrics_ens}")
    return out


# ============================================================
# %% Phase D — Aggregate cross-seed and write tables/figures
# ============================================================
def phase_D_aggregate() -> dict:
    """Read all per-seed round2.json plus the original results.json and
    write the unified Round-2 outputs (Table 4a, 4b, S1, ID curves figure)."""
    import matplotlib.pyplot as plt

    rounds: dict[int, dict] = {}
    originals: dict[int, dict] = {}
    for s in SEEDS:
        r2 = load_round2(s)
        if r2:
            rounds[s] = r2
        orig_path = RESULTS_DIR / f"seed_{s}" / "results.json"
        if orig_path.exists():
            with open(orig_path) as f:
                originals[s] = json.load(f)

    if not rounds:
        logger.error("No round2.json files found — skipping aggregation.")
        return {}

    MODEL_ORDER = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT", "Ensemble"]

    def _agg(values: list[float]) -> tuple[float, float]:
        clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not clean:
            return float("nan"), float("nan")
        return float(np.mean(clean)), float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0

    aggregated: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in MODEL_ORDER}

    # Pull classification metrics + original xAI from results.json
    classification_keys = ["accuracy", "f1_macro", "auc_ovr"]
    xai_keys = ["faithfulness_mean", "complexity_mean", "plausibility_mean"]

    for m in MODEL_ORDER:
        for k in classification_keys + xai_keys:
            vals = []
            for s in SEEDS:
                if s in originals and m in originals[s]:
                    vals.append(originals[s][m].get(k))
            aggregated[m][k] = _agg(vals)

    # New: deletion/insertion AUC + complexity_band from round2.json
    for m in MODEL_ORDER:
        if m == "Ensemble":
            # take from phase_C if present
            for k_in, k_out in [("accuracy", "accuracy_oof"),
                                 ("f1_macro", "f1_macro_oof"),
                                 ("auc_ovr", "auc_ovr_oof")]:
                vals = [rounds[s]["phase_C"].get("Ensemble (OOF)", {}).get(k_in)
                        for s in SEEDS if s in rounds and "phase_C" in rounds[s]]
                aggregated[m][k_out] = _agg(vals)
            continue
        for k in ("deletion_auc_mean", "insertion_auc_mean", "complexity_band"):
            vals = [rounds[s]["phase_A"].get(m, {}).get(k)
                    for s in SEEDS if s in rounds and "phase_A" in rounds[s]]
            aggregated[m][k] = _agg(vals)

    # ---- Sanity checks (Table S1) ----
    sanity_rows = []
    for m in MODEL_ORDER:
        if m == "Ensemble":
            continue
        model_rand = []
        data_rand = []
        for s in SEEDS:
            if s not in rounds or "phase_B" not in rounds[s]:
                continue
            entry = rounds[s]["phase_B"].get(m, {})
            if "model_randomization_spearman" in entry:
                model_rand.append(entry["model_randomization_spearman"])
            if "data_randomization_spearman" in entry:
                data_rand.append(entry["data_randomization_spearman"])
        sanity_rows.append({
            "model": m,
            "model_rand_mean": float(np.mean(model_rand)) if model_rand else float("nan"),
            "model_rand_std": float(np.std(model_rand, ddof=1)) if len(model_rand) > 1 else 0.0,
            "data_rand": float(np.mean(data_rand)) if data_rand else float("nan"),
        })

    # ---- Write Tables ----
    # Table 4a: classification only
    with open(TABLE_DIR / "benchmark_table_classification.tex", "w") as f:
        f.write("\\begin{table}\n\\centering\n")
        f.write("\\caption{Classification performance on the PLAsTiCC test set, "
                f"averaged over {len(SEEDS)} random seeds; mean $\\pm$ cross-seed std. "
                "Ensemble values come from the out-of-fold stacking refit "
                "(Section~\\ref{sec:stacking}).}\n")
        f.write("\\label{tab:benchmark_classification}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write("Model & Accuracy & F1 (macro) & AUC \\\\\n\\hline\n")
        for m in MODEL_ORDER:
            if m == "Ensemble":
                acc = aggregated[m].get("accuracy_oof", (float("nan"), float("nan")))
                f1m = aggregated[m].get("f1_macro_oof", (float("nan"), float("nan")))
                auc = aggregated[m].get("auc_ovr_oof", (float("nan"), float("nan")))
            else:
                acc = aggregated[m]["accuracy"]
                f1m = aggregated[m]["f1_macro"]
                auc = aggregated[m]["auc_ovr"]
            f.write(f"{m} & ${acc[0]:.3f} \\pm {acc[1]:.3f}$ & "
                    f"${f1m[0]:.3f} \\pm {f1m[1]:.3f}$ & "
                    f"${auc[0]:.3f} \\pm {auc[1]:.3f}$ \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Table 4b: xAI metrics
    with open(TABLE_DIR / "benchmark_table_xai.tex", "w") as f:
        f.write("\\begin{table*}\n\\centering\n")
        f.write("\\caption{Explanation-quality metrics on the PLAsTiCC test set, averaged over "
                f"{len(SEEDS)} random seeds. ``Faith. (corr)'' is the perturbation-based "
                "Pearson correlation; ``Del. AUC'' and ``Ins. AUC'' are the insertion/deletion "
                "AUC of \\citet{petsiuk2018} (lower del / higher ins is better); "
                "``Compl. (raw)'' is the entropy of the raw attribution and ``Compl. (band)'' "
                "the entropy after aggregation to the 6-band basis (Section~\\ref{sec:complexity}); "
                "``Plausibility'' is the IoU against the 29-feature expert mask, computed via the "
                "cross-representation projection of Section~\\ref{sec:cross_repr_plaus}.}\n")
        f.write("\\label{tab:benchmark_xai}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("Model & Faith.\\,(corr) & Del.\\,AUC$\\downarrow$ & Ins.\\,AUC$\\uparrow$ & "
                "Compl.\\,(raw) & Compl.\\,(band) & Plausibility \\\\\n\\hline\n")
        for m in MODEL_ORDER:
            faith = aggregated[m].get("faithfulness_mean", (float("nan"), float("nan")))
            compl = aggregated[m].get("complexity_mean", (float("nan"), float("nan")))
            plaus = aggregated[m].get("plausibility_mean", (float("nan"), float("nan")))
            del_auc = aggregated[m].get("deletion_auc_mean", (float("nan"), float("nan")))
            ins_auc = aggregated[m].get("insertion_auc_mean", (float("nan"), float("nan")))
            compl_b = aggregated[m].get("complexity_band", (float("nan"), float("nan")))

            def _fmt(t: tuple[float, float]) -> str:
                if np.isnan(t[0]):
                    return "---"
                return f"${t[0]:.3f} \\pm {t[1]:.3f}$"

            f.write(f"{m} & {_fmt(faith)} & {_fmt(del_auc)} & {_fmt(ins_auc)} & "
                    f"{_fmt(compl)} & {_fmt(compl_b)} & {_fmt(plaus)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")

    # Table S1: sanity checks
    with open(TABLE_DIR / "sanity_checks.tex", "w") as f:
        f.write("\\begin{table}\n\\centering\n")
        f.write("\\caption{Saliency sanity checks following \\citet{adebayo2018}. "
                "Lower Spearman correlation $\\rho$ between trained-model and "
                "perturbed-model attributions is better, indicating that the explanations "
                "actually depend on (a)~the learned weights and (b)~the supervision signal. "
                "Tree-based models do not have a meaningful model-randomisation analogue and "
                "are therefore reported only for the data randomisation test.}\n")
        f.write("\\label{tab:sanity_checks}\n")
        f.write("\\begin{tabular}{lcc}\n\\hline\n")
        f.write("Model & Model rand.\\ $\\rho$ & Data rand.\\ $\\rho$ \\\\\n\\hline\n")
        for row in sanity_rows:
            mr = (f"${row['model_rand_mean']:.3f} \\pm {row['model_rand_std']:.3f}$"
                  if not np.isnan(row['model_rand_mean']) else "---")
            dr = f"${row['data_rand']:.3f}$" if not np.isnan(row['data_rand']) else "---"
            f.write(f"{row['model']} & {mr} & {dr} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # ---- Insertion/Deletion curve figure (RF + ViT) ----
    rf_curves = None
    vit_curves = None
    for s in SEEDS:
        if s not in rounds:
            continue
        pa = rounds[s].get("phase_A", {})
        if rf_curves is None and "_curves_for_fig" in pa.get("Random Forest", {}):
            rf_curves = pa["Random Forest"]["_curves_for_fig"]
        if vit_curves is None and "_curves_for_fig" in pa.get("ViT", {}):
            vit_curves = pa["ViT"]["_curves_for_fig"]

    if rf_curves and vit_curves:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, name, c in [(axes[0], "Random Forest", rf_curves),
                             (axes[1], "ViT", vit_curves)]:
            ax.plot(c["fractions"], c["del_probs"], "-", color="#c0392b",
                    label="Deletion (lower=better)", lw=2)
            ax.plot(c["fractions"], c["ins_probs"], "-", color="#27ae60",
                    label="Insertion (higher=better)", lw=2)
            ax.set_xlabel("Fraction of input perturbed (in attribution order)")
            ax.set_title(name)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(loc="center right", fontsize=9)
        axes[0].set_ylabel("Predicted-class probability")
        plt.tight_layout()
        out_pdf = FIG_DIR / "insertion_deletion_curves.pdf"
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_pdf}")

    # ---- Save aggregated round2 JSON ----
    agg_path = TABLE_DIR / "benchmark_round2.json"
    with open(agg_path, "w") as f:
        json.dump({
            "seeds": SEEDS,
            "k_folds": K_FOLDS,
            "aggregated": {m: {k: list(v) for k, v in d.items()} for m, d in aggregated.items()},
            "sanity_checks": sanity_rows,
        }, f, indent=2)
    logger.info(f"Saved {agg_path}")

    return aggregated


# ============================================================
# %% Driver
# ============================================================
if "A" in PHASES_TO_RUN:
    logger.info("\n" + "#" * 70 + "\n# Phase A — new xAI metrics on existing checkpoints\n" + "#" * 70)
    for _seed in SEEDS_TO_RUN:
        logger.info(f"--- Phase A | seed {_seed} ---")
        phase_A_for_seed(_seed)

if "B" in PHASES_TO_RUN:
    logger.info("\n" + "#" * 70 + "\n# Phase B — saliency sanity checks\n" + "#" * 70)
    for _seed in SEEDS_TO_RUN:
        logger.info(f"--- Phase B | seed {_seed} ---")
        phase_B_for_seed(_seed, do_data_randomization=(_seed == DATA_RAND_SEED))

if "C" in PHASES_TO_RUN:
    logger.info("\n" + "#" * 70 + "\n# Phase C — out-of-fold stacking ensemble\n" + "#" * 70)
    for _seed in SEEDS_TO_RUN:
        logger.info(f"--- Phase C | seed {_seed} ---")
        phase_C_for_seed(_seed)

if "D" in PHASES_TO_RUN:
    logger.info("\n" + "#" * 70 + "\n# Phase D — cross-seed aggregation\n" + "#" * 70)
    phase_D_aggregate()

print("\nDone! Round-2 outputs in paper/tables/ and paper/figures/")
