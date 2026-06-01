"""Generate final multi-seed comparison table + confusion matrices — Spyder script.

Trains all 7 models for every seed in ``SEEDS``, computes classification + xAI
metrics (faithfulness, complexity, cross-representation plausibility), and
aggregates the results into a single ``mean ± std`` benchmark table where the
``±`` denotes the **cross-seed** standard deviation.

Per-sample standard deviations (already computed inside ``compute_xai_metrics_*``)
are persisted in the per-seed JSON files but kept separate from the cross-seed
``±`` shown in the LaTeX table.

Usage in Spyder: ``%runfile run_final_table.py --wdir``

To debug a single seed, set ``SEEDS = [42]`` at the top.
"""

from __future__ import annotations

# %% Setup paths (resolve relative to this file, not the cwd)
# Spyder's `%runfile ... --wdir` sets the working directory to the script's
# folder, so naively using "paper/figures" would land outputs under
# scripts/paper/figures. We anchor every path to the project root instead.
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# %% Config
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
SEEDS = [42, 123, 456, 789, 1024]   # multi-seed benchmark
BATCH_SIZE = 64
DEVICE = "cuda"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"
N_SAMPLES_EXPLAIN = 50

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.data.plasticc import CLASS_MAP
from src.evaluation.metrics import compute_metrics, confusion_matrix_plot
from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper, extract_features
from src.training.trainer import Trainer, set_global_seed
from src.xai.shap_analysis import compute_shap_values, get_feature_names
from src.xai.metrics import (compute_xai_metrics_tabular, faithfulness_correlation,
                              explanation_complexity, build_expert_mask_tabular)
from src.xai.integrated_gradients import compute_integrated_gradients
from src.xai.plausibility import (compute_tabular_plausibility,
                                   compute_timeseries_plausibility,
                                   compute_image_plausibility)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
fig_dir = FIG_DIR
table_dir = TABLE_DIR
results_dir = RESULTS_DIR
fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Using device: {device} | project root: {PROJECT_ROOT}")

# %% Load data (shared across all seeds)
data_dir = DATA_DIR
label_map = np.load(data_dir / "label_map.npy", allow_pickle=True).item()
idx_to_target = {v: k for k, v in label_map.items()}
class_names = [CLASS_MAP.get(idx_to_target[i], str(i)) for i in range(len(label_map))]
n_classes = len(class_names)

train_data = np.load(data_dir / "train.npz")
val_data = np.load(data_dir / "val.npz")
test_data = np.load(data_dir / "test.npz")

y_train = train_data["labels"]
y_val = val_data["labels"]
y_test = test_data["labels"]

# Tabular features
x_train_tab = extract_features(np.vstack([train_data["light_curves"], val_data["light_curves"]]))
y_train_tab = np.concatenate([y_train, y_val])
x_test_tab = extract_features(test_data["light_curves"])
x_val_tab = extract_features(val_data["light_curves"])

# Timeseries: (N, 6, 256)
lc_train = train_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_val = val_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_test = test_data["light_curves"][:, :, :, 0].astype(np.float32)

# GAF cache (built by run_gaf.py)
gaf_cache = data_dir / "gaf_224.npz"
if not gaf_cache.exists():
    raise FileNotFoundError(f"GAF cache not found at {gaf_cache}. Run run_gaf.py first.")
logger.info("Loading cached GAF images...")
gaf_data = np.load(gaf_cache)
gaf_train, gaf_val, gaf_test = gaf_data["train"], gaf_data["val"], gaf_data["test"]

# Expert mask for plausibility
feature_names = get_feature_names()
expert_mask = build_expert_mask_tabular(feature_names)


# ============================================================
# %% Helper: DL wrapper + IG-based xAI metrics
# ============================================================
class DLModelWrapper:
    """Wraps a PyTorch model for faithfulness_correlation compatibility.

    Flattens input so faithfulness can perturb individual features.
    """

    def __init__(self, model: torch.nn.Module, input_shape: tuple, device: torch.device) -> None:
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_t = torch.from_numpy(x.reshape(-1, *self.input_shape)).float().to(self.device)
        with torch.no_grad():
            logits = self.model(x_t)
            return F.softmax(logits, dim=-1).cpu().numpy()


def compute_xai_metrics_dl(
    model: torch.nn.Module,
    x_test: np.ndarray,
    input_shape: tuple,
    device: torch.device,
    n_samples: int = 50,
    seed: int = 42,
    n_steps: int = 50,
    return_attributions: bool = False,
):
    """Faithfulness + complexity for a DL model using Integrated Gradients."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_test), size=min(n_samples, len(x_test)), replace=False)

    wrapper = DLModelWrapper(model, input_shape, device)
    faith_scores: list[float] = []
    compl_scores: list[float] = []
    attr_list: list[np.ndarray] = []

    for idx in indices:
        sample = torch.from_numpy(x_test[idx:idx + 1]).float().to(device)
        attr = compute_integrated_gradients(model, sample, target_class=None, n_steps=n_steps)
        attr_list.append(attr)

        attr_flat = np.abs(attr).flatten()
        x_flat = x_test[idx].flatten()

        faith_scores.append(faithfulness_correlation(
            wrapper, x_flat, attr_flat, n_perturbations=50, seed=seed,
        ))
        compl_scores.append(explanation_complexity(attr_flat))

    metrics = {
        "faithfulness_mean": float(np.mean(faith_scores)),
        "faithfulness_std": float(np.std(faith_scores)),
        "complexity_mean": float(np.mean(compl_scores)),
        "complexity_std": float(np.std(compl_scores)),
    }
    logger.info(
        f"IG xAI: faithfulness={metrics['faithfulness_mean']:.4f}±{metrics['faithfulness_std']:.4f} | "
        f"complexity={metrics['complexity_mean']:.4f}±{metrics['complexity_std']:.4f}"
    )
    if return_attributions:
        return metrics, np.stack(attr_list, axis=0), indices
    return metrics


def train_and_eval_dl(model, name, train_loader, val_loader, test_loader,
                      config, x_test_raw, input_shape, representation,
                      seed: int, save_cm: bool, ckpt_dir: Path):
    """Train a DL model, evaluate classification + xAI (incl. plausibility)."""
    logger.info("=" * 60)
    logger.info(f"Training {name} (seed={seed})")
    logger.info("=" * 60)

    trainer = Trainer(model, config, device=device, seed=seed)
    trainer.fit(train_loader, val_loader)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_dir / f"{name.lower().replace(' ', '_')}.pt")

    # Classification
    model.eval()
    y_pred, y_proba = [], []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            y_pred.append(logits.argmax(dim=-1).cpu().numpy())
            y_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
    y_pred = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    if save_cm:
        confusion_matrix_plot(
            y_test, y_pred, class_names,
            save_path=fig_dir / f"cm_{name.lower().replace(' ', '_')}.pdf",
        )

    # xAI — IG faithfulness/complexity, returning structured attributions
    logger.info(f"Computing IG xAI metrics for {name}...")
    xai, ig_attr, ig_idx = compute_xai_metrics_dl(
        model, x_test_raw, input_shape, device,
        n_samples=N_SAMPLES_EXPLAIN, seed=seed, return_attributions=True,
    )

    # Cross-representation plausibility on the same sampled subset
    logger.info(f"Computing plausibility ({representation}) for {name}...")
    if representation == "timeseries":
        plaus = compute_timeseries_plausibility(
            ig_attr, expert_mask, flux=x_test_raw[ig_idx],
        )
    elif representation == "image":
        plaus = compute_image_plausibility(ig_attr, expert_mask)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return {**metrics, **xai, **plaus}


# ============================================================
# %% Single-seed runner
# ============================================================
def run_one_seed(seed: int, save_figures: bool = False) -> dict[str, dict[str, float]]:
    """Train + evaluate all 7 models for a given seed.

    Args:
        seed: Random seed (controls torch/np/random/CUDA + sklearn random_state).
        save_figures: If True, dump confusion matrices to ``paper/figures/``
            for this seed (typically only the first seed in the multi-seed loop).

    Returns:
        ``{model_name: {metric_name: value, ...}}`` with classification and
        xAI metrics for all 7 architectures.
    """
    set_global_seed(seed)
    seed_dir = results_dir / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    seed_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}

    # ---- Random Forest ----
    logger.info("=" * 60)
    logger.info(f"Random Forest (seed={seed})")
    logger.info("=" * 60)
    rf = RFClassifier(n_estimators=500, random_state=seed)
    rf.fit(x_train_tab, y_train_tab)
    y_pred_rf = rf.predict(x_test_tab)
    y_proba_rf = rf.predict_proba(x_test_tab)
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_proba_rf)
    if save_figures:
        confusion_matrix_plot(y_test, y_pred_rf, class_names, save_path=fig_dir / "cm_rf.pdf")
    shap_rf = compute_shap_values(rf, x_test_tab, method="tree")
    xai_rf = compute_xai_metrics_tabular(rf, x_test_tab, shap_rf, n_samples=N_SAMPLES_EXPLAIN, seed=seed)
    plaus_rf = compute_tabular_plausibility(shap_rf, expert_mask)
    results["Random Forest"] = {**metrics_rf, **xai_rf, **plaus_rf}

    # ---- XGBoost ----
    logger.info("=" * 60)
    logger.info(f"XGBoost (seed={seed})")
    logger.info("=" * 60)
    xgb = XGBClassifierWrapper(n_estimators=500, random_state=seed)
    xgb.fit(x_train_tab, y_train_tab)
    y_pred_xgb = xgb.predict(x_test_tab)
    y_proba_xgb = xgb.predict_proba(x_test_tab)
    metrics_xgb = compute_metrics(y_test, y_pred_xgb, y_proba_xgb)
    if save_figures:
        confusion_matrix_plot(y_test, y_pred_xgb, class_names, save_path=fig_dir / "cm_xgb.pdf")
    shap_xgb = compute_shap_values(xgb, x_test_tab, method="tree")
    xai_xgb = compute_xai_metrics_tabular(xgb, x_test_tab, shap_xgb, n_samples=N_SAMPLES_EXPLAIN, seed=seed)
    plaus_xgb = compute_tabular_plausibility(shap_xgb, expert_mask)
    results["XGBoost"] = {**metrics_xgb, **xai_xgb, **plaus_xgb}

    # ---- Sequence-model loaders (rebuilt per seed for shuffle determinism) ----
    g_ts = torch.Generator()
    g_ts.manual_seed(seed)
    train_ds = TensorDataset(torch.from_numpy(lc_train), torch.from_numpy(y_train.astype(np.int64)))
    val_ds = TensorDataset(torch.from_numpy(lc_val), torch.from_numpy(y_val.astype(np.int64)))
    test_ds = TensorDataset(torch.from_numpy(lc_test), torch.from_numpy(y_test.astype(np.int64)))
    ts_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_ts)
    ts_val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    ts_test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- LSTM ----
    from src.models.lstm_baseline import LSTMClassifier
    lstm = LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                          n_layers=2, dropout=0.3, bidirectional=True)
    config_ts = {
        "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
        "_model_config": {"lr": 1e-3, "epochs": 100},
    }
    results["LSTM"] = train_and_eval_dl(
        lstm, "LSTM", ts_train_loader, ts_val_loader, ts_test_loader,
        config_ts, lc_test, (6, 256), representation="timeseries",
        seed=seed, save_cm=save_figures, ckpt_dir=ckpt_dir,
    )

    # ---- CNN1D ----
    from src.models.cnn_baseline import CNN1D
    cnn1d = CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3)
    results["CNN1D"] = train_and_eval_dl(
        cnn1d, "CNN1D", ts_train_loader, ts_val_loader, ts_test_loader,
        config_ts, lc_test, (6, 256), representation="timeseries",
        seed=seed, save_cm=save_figures, ckpt_dir=ckpt_dir,
    )

    # ---- GAF loaders ----
    g_gaf = torch.Generator()
    g_gaf.manual_seed(seed)
    gaf_train_ds = TensorDataset(torch.from_numpy(gaf_train), torch.from_numpy(y_train.astype(np.int64)))
    gaf_val_ds = TensorDataset(torch.from_numpy(gaf_val), torch.from_numpy(y_val.astype(np.int64)))
    gaf_test_ds = TensorDataset(torch.from_numpy(gaf_test), torch.from_numpy(y_test.astype(np.int64)))
    gaf_train_loader = DataLoader(gaf_train_ds, batch_size=32, shuffle=True, generator=g_gaf)
    gaf_val_loader = DataLoader(gaf_val_ds, batch_size=32, shuffle=False)
    gaf_test_loader = DataLoader(gaf_test_ds, batch_size=32, shuffle=False)

    # ---- CNN2D ----
    from src.models.cnn_baseline import CNN2D
    cnn2d = CNN2D(n_bands=6, n_classes=n_classes, base_filters=32, dropout=0.3)
    config_gaf = {
        "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
        "_model_config": {"lr": 1e-3, "epochs": 100},
    }
    results["CNN2D"] = train_and_eval_dl(
        cnn2d, "CNN2D", gaf_train_loader, gaf_val_loader, gaf_test_loader,
        config_gaf, gaf_test, (6, 224, 224), representation="image",
        seed=seed, save_cm=save_figures, ckpt_dir=ckpt_dir,
    )

    # ---- ViT ----
    from src.models.vit import ViTClassifier
    vit = ViTClassifier(n_bands=6, n_classes=n_classes,
                        model_name="vit_small_patch16_224", pretrained=True, dropout=0.1)
    config_vit = {
        "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
        "_model_config": {"lr": 1e-4, "epochs": 50, "weight_decay": 0.01},
    }
    results["ViT"] = train_and_eval_dl(
        vit, "ViT", gaf_train_loader, gaf_val_loader, gaf_test_loader,
        config_vit, gaf_test, (6, 224, 224), representation="image",
        seed=seed, save_cm=save_figures, ckpt_dir=ckpt_dir,
    )

    # ---- Stacking Ensemble ----
    logger.info("=" * 60)
    logger.info(f"Stacking Ensemble (seed={seed})")
    logger.info("=" * 60)
    from src.models.ensemble import StackingEnsemble

    val_inputs = [x_val_tab, x_val_tab, lc_val, lc_val, gaf_val, gaf_val]
    test_inputs = [x_test_tab, x_test_tab, lc_test, lc_test, gaf_test, gaf_test]
    base_models = [rf, xgb, lstm, cnn1d, cnn2d, vit]
    ensemble = StackingEnsemble(base_models, meta_learner="logistic", random_state=seed)
    ensemble.fit(val_inputs, y_val)

    y_pred_ens = ensemble.predict(test_inputs)
    y_proba_ens = ensemble.predict_proba(test_inputs)
    metrics_ens = compute_metrics(y_test, y_pred_ens, y_proba_ens)
    if save_figures:
        confusion_matrix_plot(y_test, y_pred_ens, class_names, save_path=fig_dir / "cm_ensemble.pdf")

    meta_test = ensemble._get_meta_features(test_inputs)

    class EnsembleMetaWrapper:
        def __init__(self, meta_model):
            self.meta = meta_model
        def predict_proba(self, x):
            return self.meta.predict_proba(x.reshape(1, -1) if x.ndim == 1 else x)
        def predict(self, x):
            return self.meta.predict(x.reshape(1, -1) if x.ndim == 1 else x)

    ens_wrapper = EnsembleMetaWrapper(ensemble.meta)
    rng_ens = np.random.default_rng(seed)
    ens_indices = rng_ens.choice(len(meta_test), size=min(N_SAMPLES_EXPLAIN, len(meta_test)), replace=False)

    faith_scores_ens, compl_scores_ens = [], []
    for idx in ens_indices:
        meta_sample = meta_test[idx]
        pred_class = ens_wrapper.predict(meta_sample)[0]
        attr = np.abs(ensemble.meta.coef_[pred_class])
        faith_scores_ens.append(faithfulness_correlation(
            ens_wrapper, meta_sample, attr, n_perturbations=50, seed=seed,
        ))
        compl_scores_ens.append(explanation_complexity(attr))

    xai_ens = {
        "faithfulness_mean": float(np.mean(faith_scores_ens)),
        "faithfulness_std": float(np.std(faith_scores_ens)),
        "complexity_mean": float(np.mean(compl_scores_ens)),
        "complexity_std": float(np.std(compl_scores_ens)),
    }
    logger.info(
        f"Ensemble xAI: faithfulness={xai_ens['faithfulness_mean']:.4f}±{xai_ens['faithfulness_std']:.4f} | "
        f"complexity={xai_ens['complexity_mean']:.4f}±{xai_ens['complexity_std']:.4f}"
    )

    # Coef-weighted average of base-model plausibilities
    base_names_ord = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT"]
    coef_mag = np.abs(ensemble.meta.coef_).sum(axis=0)
    weights = np.array([coef_mag[i*n_classes:(i+1)*n_classes].sum() for i in range(6)])
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(6) / 6
    plaus_means = np.array([results[n]["plausibility_mean"] for n in base_names_ord])
    plaus_stds = np.array([results[n]["plausibility_std"] for n in base_names_ord])
    plaus_ens = {
        "plausibility_mean": float((weights * plaus_means).sum()),
        "plausibility_std": float(np.sqrt((weights**2 * plaus_stds**2).sum())),
    }
    logger.info(
        f"Ensemble plausibility (coef-weighted): "
        f"{plaus_ens['plausibility_mean']:.4f}±{plaus_ens['plausibility_std']:.4f}"
    )
    results["Ensemble"] = {**metrics_ens, **xai_ens, **plaus_ens}

    # Persist per-seed results
    with open(seed_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Per-seed results saved to {seed_dir / 'results.json'}")

    return results


# ============================================================
# %% Multi-seed loop
# ============================================================
all_runs: list[dict[str, dict[str, float]]] = []
for i, _seed in enumerate(SEEDS):
    logger.info("\n" + "#" * 70)
    logger.info(f"# SEED {_seed}  ({i + 1}/{len(SEEDS)})")
    logger.info("#" * 70)
    res = run_one_seed(_seed, save_figures=(i == 0))
    all_runs.append(res)


# ============================================================
# %% Aggregate cross-seed mean ± std
# ============================================================
KEY_METRICS = ["accuracy", "f1_macro", "f1_weighted", "auc_ovr",
               "faithfulness_mean", "complexity_mean", "plausibility_mean"]
MODEL_ORDER = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT", "Ensemble"]

# aggregated[model][metric] -> {"mean_seed", "std_seed", "values"}
aggregated: dict[str, dict[str, dict[str, float | list[float]]]] = {}
for model_name in MODEL_ORDER:
    aggregated[model_name] = {}
    for k in KEY_METRICS:
        vals = [run[model_name].get(k, float("nan")) for run in all_runs]
        vals_clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if len(vals_clean) == 0:
            aggregated[model_name][k] = {"mean_seed": float("nan"),
                                         "std_seed": float("nan"),
                                         "values": vals}
            continue
        aggregated[model_name][k] = {
            "mean_seed": float(np.mean(vals_clean)),
            "std_seed": float(np.std(vals_clean, ddof=1)) if len(vals_clean) > 1 else 0.0,
            "values": vals,
        }

# Also persist the cross-sample stds (the "_std" siblings already in each run)
SAMPLE_STD_KEYS = ["faithfulness_std", "complexity_std", "plausibility_std"]
for model_name in MODEL_ORDER:
    for k in SAMPLE_STD_KEYS:
        vals = [run[model_name].get(k, float("nan")) for run in all_runs]
        vals_clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        aggregated[model_name][k] = {
            "mean_seed": float(np.mean(vals_clean)) if vals_clean else float("nan"),
            "std_seed": float(np.std(vals_clean, ddof=1)) if len(vals_clean) > 1 else 0.0,
            "values": vals,
        }

# ============================================================
# %% Print + save aggregated table
# ============================================================
print("\n" + "=" * 130)
print(f"FULL BENCHMARK — {len(SEEDS)}-seed mean ± std (cross-seed)")
print(f"Seeds: {SEEDS}")
print("=" * 130)
header = f"{'Model':<16}" + "".join(f"{k:>18}" for k in KEY_METRICS)
print(header)
print("-" * 142)
for model_name in MODEL_ORDER:
    row = f"{model_name:<16}"
    for k in KEY_METRICS:
        m = aggregated[model_name][k]["mean_seed"]
        s = aggregated[model_name][k]["std_seed"]
        if np.isnan(m):
            row += f"{'N/A':>18}"
        else:
            row += f"{m:>9.3f}±{s:<7.3f}"
    print(row)

# Save aggregated JSON
agg_path = table_dir / "benchmark_results.json"
with open(agg_path, "w") as f:
    json.dump({
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "aggregated": aggregated,
        "per_seed": all_runs,
    }, f, indent=2)
logger.info(f"Aggregated results saved to {agg_path}")

# %% Save LaTeX table (cross-seed mean ± std)
latex_path = table_dir / "benchmark_table.tex"
with open(latex_path, "w") as f:
    f.write("\\begin{table*}\n")
    f.write("\\centering\n")
    f.write(
        "\\caption{Classification and xAI metrics for all models on the PLAsTiCC test set, "
        f"averaged over {len(SEEDS)} random seeds (\\(\\{{{', '.join(map(str, SEEDS))}\\}}\\)). "
        "Values are reported as cross-seed mean $\\pm$ standard deviation. Per-sample standard "
        "deviations for the xAI metrics are persisted in the JSON results file but kept separate "
        "from the cross-seed $\\pm$ shown here. Plausibility is computed via IoU against the "
        "29-feature astrophysical expert mask; for sequence/image models the IG attributions are "
        "first projected to the tabular feature space (Section~\\ref{sec:cross_repr_plaus}), and "
        "for the ensemble it is reported as the meta-coefficient-weighted average of the base-"
        "model plausibilities.}\n"
    )
    f.write("\\label{tab:benchmark}\n")
    f.write("\\begin{tabular}{lcccccc}\n")
    f.write("\\hline\n")
    f.write(
        "Model & Accuracy & F1 (macro) & AUC & Faithfulness & Complexity & Plausibility \\\\\n"
    )
    f.write("\\hline\n")

    def _fmt(model_name: str, key: str) -> str:
        m = aggregated[model_name][key]["mean_seed"]
        s = aggregated[model_name][key]["std_seed"]
        if np.isnan(m):
            return "---"
        return f"${m:.3f} \\pm {s:.3f}$"

    for model_name in MODEL_ORDER:
        f.write(
            f"{model_name} & "
            f"{_fmt(model_name, 'accuracy')} & "
            f"{_fmt(model_name, 'f1_macro')} & "
            f"{_fmt(model_name, 'auc_ovr')} & "
            f"{_fmt(model_name, 'faithfulness_mean')} & "
            f"{_fmt(model_name, 'complexity_mean')} & "
            f"{_fmt(model_name, 'plausibility_mean')} \\\\\n"
        )
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table*}\n")

logger.info(f"LaTeX table saved to {latex_path}")
print("\nDone! Per-seed JSON in scripts/results/seed_*/, aggregated in paper/tables/")
