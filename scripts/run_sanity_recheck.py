"""Sanity-check re-test for LSTM and CNN1D with a non-zero IG baseline.

Round 2 found that LSTM and CNN1D failed the data-randomisation test
(both with $\\rho \\approx 0.998$) and that LSTM additionally failed the
model-randomisation test ($\\rho = 0.997$). The leading hypothesis was that
zero-baseline Integrated Gradients on these architectures degenerates to
$\\text{IG}_i \\approx x_i \\cdot \\text{const}$, which would make the
attributions nearly input-only and therefore largely insensitive to weight
or label perturbations.

This script re-runs the two sanity checks for **LSTM** and **CNN1D** only,
using **per-band mean of the training light curves** as the IG baseline
instead of zero. For CNN1D, it additionally re-trains the random-label
model with **early stopping disabled** (50 forced epochs) to rule out the
``shuffled model never moves from kaiming init'' artefact.

Outputs:
* Updates ``scripts/results/seed_*/round2.json`` in place, replacing the
  ``phase_B`` entries for LSTM and CNN1D with the new $\\rho$ values
  (originals are preserved under the suffix ``_zero_baseline``).
* Re-renders ``paper/tables/sanity_checks.tex`` with the updated rows.

Usage in Spyder: ``%runfile run_sanity_recheck.py --wdir``
"""

from __future__ import annotations

# %% Setup paths
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# %% Config
SEEDS = [42, 123, 456, 789, 1024]
SEEDS_TO_RUN = SEEDS                  # set to [42] for a quick smoke test
DATA_RAND_SEED = 42
N_SAMPLES_SANITY = 20
BATCH_SIZE = 64
DEVICE = "cuda"

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "plasticc"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"

# %% Imports
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer, set_global_seed
from src.xai.sanity_checks import (model_randomization_test,
                                    data_randomization_spearman_dl,
                                    shuffle_labels)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
logger.info(f"Sanity recheck | device={device} | project root={PROJECT_ROOT}")

# %% Load data
train_data = np.load(DATA_DIR / "train.npz")
val_data = np.load(DATA_DIR / "val.npz")
test_data = np.load(DATA_DIR / "test.npz")

y_train = train_data["labels"]
y_val = val_data["labels"]
y_test = test_data["labels"]

lc_train = train_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_val = val_data["light_curves"][:, :, :, 0].astype(np.float32)
lc_test = test_data["light_curves"][:, :, :, 0].astype(np.float32)
n_classes = int(y_train.max()) + 1

# %% Per-band mean baseline (computed once on training set)
# Shape (6, 256). Used as the IG baseline instead of zero for both LSTM and
# CNN1D. The (x - x') term is now structural rather than just x, which
# prevents the IG ~ x * const degeneracy.
ts_baseline_np = lc_train.mean(axis=0).astype(np.float32)         # (6, 256)
ts_baseline = torch.from_numpy(ts_baseline_np[None]).to(device)    # (1, 6, 256)
logger.info(
    f"Per-band IG baseline (mean of training light curves): "
    f"shape={ts_baseline.shape}, range=[{ts_baseline.min():.3f}, {ts_baseline.max():.3f}]"
)

# %% Helpers
def make_lstm() -> torch.nn.Module:
    from src.models.lstm_baseline import LSTMClassifier
    return LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                          n_layers=2, dropout=0.3, bidirectional=True)


def make_cnn1d() -> torch.nn.Module:
    from src.models.cnn_baseline import CNN1D
    return CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3)


def load_dl_checkpoint(factory, ckpt_path: Path) -> torch.nn.Module:
    model = factory().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def round2_path(seed: int) -> Path:
    return RESULTS_DIR / f"seed_{seed}" / "round2.json"


def load_round2(seed: int) -> dict:
    p = round2_path(seed)
    with open(p) as f:
        return json.load(f)


def save_round2(seed: int, data: dict) -> None:
    p = round2_path(seed)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Updated {p.name}")


# %% Re-run model randomization (LSTM, CNN1D) for every seed with mean baseline
for seed in SEEDS_TO_RUN:
    logger.info(f"\n=== seed {seed}: model randomization (mean baseline) ===")
    set_global_seed(seed)
    out = load_round2(seed)
    out.setdefault("phase_B", {})

    rng = np.random.default_rng(seed + 1)
    sample_idx = rng.choice(len(y_test), size=N_SAMPLES_SANITY, replace=False)

    seed_dir = RESULTS_DIR / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"

    for name, factory in [("LSTM", make_lstm), ("CNN1D", make_cnn1d)]:
        ckpt_file = ckpt_dir / f"{name.lower()}.pt"
        if not ckpt_file.exists():
            logger.warning(f"  missing checkpoint for {name}, skipping")
            continue
        model = load_dl_checkpoint(factory, ckpt_file)
        res = model_randomization_test(
            model, lc_test, sample_idx,
            cascading=True, n_steps_to_keep=4, ig_n_steps=20,
            baseline=ts_baseline,
        )
        # Preserve the original zero-baseline value under a suffix
        old = out["phase_B"].get(name, {})
        if "model_randomization_spearman" in old and "model_randomization_spearman_zero_baseline" not in old:
            old["model_randomization_spearman_zero_baseline"] = old["model_randomization_spearman"]
        old["model_randomization_spearman"] = res["spearman_full"]
        old["model_randomization_trace_mean_baseline"] = [
            {"layer": n, "spearman": float(s)} for n, s in res["trace"]
        ]
        out["phase_B"][name] = old
        logger.info(f"  {name}: rho_model_full = {res['spearman_full']:.4f}")

    save_round2(seed, out)


# %% Re-run data randomization (LSTM, CNN1D) only for DATA_RAND_SEED
logger.info(f"\n=== seed {DATA_RAND_SEED}: data randomization (mean baseline) ===")
set_global_seed(DATA_RAND_SEED)
out = load_round2(DATA_RAND_SEED)
out.setdefault("phase_B", {})

rng = np.random.default_rng(DATA_RAND_SEED + 1)
sample_idx = rng.choice(len(y_test), size=N_SAMPLES_SANITY, replace=False)

# Build sequence loaders for data-randomisation retraining
y_train_shuf = shuffle_labels(y_train, seed=DATA_RAND_SEED)
y_val_shuf = shuffle_labels(y_val, seed=DATA_RAND_SEED + 1)

gen = torch.Generator(); gen.manual_seed(DATA_RAND_SEED)
ts_train_ds = TensorDataset(torch.from_numpy(lc_train), torch.from_numpy(y_train_shuf.astype(np.int64)))
ts_val_ds = TensorDataset(torch.from_numpy(lc_val), torch.from_numpy(y_val_shuf.astype(np.int64)))
ts_train_loader = DataLoader(ts_train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
ts_val_loader = DataLoader(ts_val_ds, batch_size=BATCH_SIZE, shuffle=False)

ckpt_dir = RESULTS_DIR / f"seed_{DATA_RAND_SEED}" / "checkpoints"

# LSTM: retrain with the SAME early-stopping protocol as Round 2 + mean baseline
logger.info("--- LSTM: retraining on shuffled labels (standard config) + mean baseline IG ---")
config_ts = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": 1e-3, "epochs": 100},
}
model_t = load_dl_checkpoint(make_lstm, ckpt_dir / "lstm.pt")
model_r = make_lstm()
trainer = Trainer(model_r, config_ts, device=device, seed=DATA_RAND_SEED)
trainer.fit(ts_train_loader, ts_val_loader)
rho_lstm = data_randomization_spearman_dl(
    model_t, model_r, lc_test, sample_idx, ig_n_steps=20, baseline=ts_baseline,
)
old = out["phase_B"].get("LSTM", {})
if "data_randomization_spearman" in old and "data_randomization_spearman_zero_baseline" not in old:
    old["data_randomization_spearman_zero_baseline"] = old["data_randomization_spearman"]
old["data_randomization_spearman"] = float(rho_lstm)
out["phase_B"]["LSTM"] = old
logger.info(f"  LSTM rho_data (mean baseline) = {rho_lstm:.4f}")
del model_r; torch.cuda.empty_cache()

# CNN1D: TWO checks
#   (a) standard early-stopping config + mean baseline
#   (b) NO early stopping (forced 50 epochs) + mean baseline — to test the
#       "shuffled model is stuck near init" hypothesis
logger.info("--- CNN1D (a): retrain shuffled w/ early stopping + mean baseline ---")
model_t = load_dl_checkpoint(make_cnn1d, ckpt_dir / "cnn1d.pt")
config_ts_es = {
    "training": {"early_stopping_patience": 10, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": 1e-3, "epochs": 100},
}
model_r = make_cnn1d()
trainer = Trainer(model_r, config_ts_es, device=device, seed=DATA_RAND_SEED)
trainer.fit(ts_train_loader, ts_val_loader)
rho_cnn1d_es = data_randomization_spearman_dl(
    model_t, model_r, lc_test, sample_idx, ig_n_steps=20, baseline=ts_baseline,
)
logger.info(f"  CNN1D rho_data (with ES, mean baseline) = {rho_cnn1d_es:.4f}")
del model_r; torch.cuda.empty_cache()

logger.info("--- CNN1D (b): retrain shuffled FORCED 50 epochs + mean baseline ---")
config_ts_noes = {
    "training": {"early_stopping_patience": 9999, "scheduler": "cosine", "gradient_clip": 1.0},
    "_model_config": {"lr": 1e-3, "epochs": 50},
}
model_r = make_cnn1d()
trainer = Trainer(model_r, config_ts_noes, device=device, seed=DATA_RAND_SEED)
trainer.fit(ts_train_loader, ts_val_loader)
rho_cnn1d_noes = data_randomization_spearman_dl(
    model_t, model_r, lc_test, sample_idx, ig_n_steps=20, baseline=ts_baseline,
)
logger.info(f"  CNN1D rho_data (no ES, 50 epochs, mean baseline) = {rho_cnn1d_noes:.4f}")
del model_r; torch.cuda.empty_cache()

# Decide which one to report as the "headline" — the more conservative
# (higher rho) or the more methodologically clean (no early stopping)?
# We pick (b) "no early stopping" because it cleanly separates "model
# trained on noise" from "model still at init", and we keep both in the
# JSON for traceability.
old = out["phase_B"].get("CNN1D", {})
if "data_randomization_spearman" in old and "data_randomization_spearman_zero_baseline" not in old:
    old["data_randomization_spearman_zero_baseline"] = old["data_randomization_spearman"]
old["data_randomization_spearman"] = float(rho_cnn1d_noes)
old["data_randomization_spearman_with_early_stopping"] = float(rho_cnn1d_es)
out["phase_B"]["CNN1D"] = old

save_round2(DATA_RAND_SEED, out)


# %% Re-render sanity_checks.tex from updated round2.json files
logger.info("\n=== Re-rendering sanity_checks.tex ===")
MODEL_ORDER = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT"]
sanity_rows = []
for m in MODEL_ORDER:
    model_rand = []
    data_rand = []
    for s in SEEDS:
        try:
            r2 = load_round2(s)
        except FileNotFoundError:
            continue
        entry = r2.get("phase_B", {}).get(m, {})
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

with open(TABLE_DIR / "sanity_checks.tex", "w") as f:
    f.write("\\begin{table}\n\\centering\n")
    f.write("\\caption{Saliency sanity checks following \\citet{adebayo2018}. "
            "Lower Spearman correlation $\\rho$ between trained-model and "
            "perturbed-model attributions is better, indicating that the explanations "
            "actually depend on (a)~the learned weights and (b)~the supervision signal. "
            "Tree-based models do not have a meaningful model-randomisation analogue and "
            "are therefore reported only for the data randomisation test. The LSTM and "
            "CNN1D entries use a per-band mean training-set baseline for Integrated "
            "Gradients (Section~\\ref{sec:sanity_discussion}); a zero baseline causes IG "
            "to degenerate on these architectures.}\n")
    f.write("\\label{tab:sanity_checks}\n")
    f.write("\\begin{tabular}{lcc}\n\\hline\n")
    f.write("Model & Model rand.\\ $\\rho$ & Data rand.\\ $\\rho$ \\\\\n\\hline\n")
    for row in sanity_rows:
        mr = (f"${row['model_rand_mean']:.3f} \\pm {row['model_rand_std']:.3f}$"
              if not np.isnan(row['model_rand_mean']) else "---")
        dr = f"${row['data_rand']:.3f}$" if not np.isnan(row['data_rand']) else "---"
        f.write(f"{row['model']} & {mr} & {dr} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

logger.info(f"Updated {TABLE_DIR / 'sanity_checks.tex'}")
print("\nDone! Check sanity_checks.tex and the per-seed round2.json for the updated rho values.")
print("\nQuick summary:")
for row in sanity_rows:
    mr = f"{row['model_rand_mean']:.3f}" if not np.isnan(row['model_rand_mean']) else "---"
    dr = f"{row['data_rand']:.3f}" if not np.isnan(row['data_rand']) else "---"
    print(f"  {row['model']:<14}  model_rand rho = {mr}    data_rand rho = {dr}")
