"""Paper 2, Sect. 6: train the ZTF BTS models and cache per-sample attributions.

Per seed (42, 123, 456, 789, 1024):
  XGBoost-min : 20 per-band features (basis "minimal"),        TreeSHAP
  XGBoost-bts : 26 = 20 + BTS cross-band features,             TreeSHAP
  XGBoost-z   : 27 = 26 + absolute peak magnitude (z-aware),   TreeSHAP
  CNN1D, LSTM : (2, 256) normalised light curve, IG projected to the 20 per-band
                slots (the 6/7 cross-band slots have no counterpart: zero)

XGBoost uses balanced class weights; CNN1D uses square-root-balanced weights,
weight decay 1e-3, dropout 0.5 and best-val-loss weight restore (chosen on seed 42 among 16 training
configurations, see _ztf_cnn_weights.py; the balanced weights of the tabular
models drive the CNN to the minority classes and below majority accuracy).
BTS is 74 % SN Ia.  Attributions are
computed on the full test set.  Cache: scripts/results/ztf/seed_<s>/meta.pkl
with {"meta": {model: (N, D)}, "basis": {model: [names]}, "metrics": {...}}.

Usage (Spyder: %runfile scripts/run_ztf.py --wdir):  python scripts/run_ztf.py [seeds...]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.cnn_baseline import CNN1D  # noqa: E402
from src.models.tabular_rf import XGBClassifierWrapper, extract_features  # noqa: E402
from src.training.trainer import Trainer, set_global_seed  # noqa: E402
from src.xai.integrated_gradients import compute_integrated_gradients  # noqa: E402
from src.xai.masks_ztf import get_feature_names_ztf  # noqa: E402
from src.xai.plausibility import project_timeseries_attribution  # noqa: E402

DATA = PROJECT_ROOT / "data" / "processed" / "ztf"
RES = PROJECT_ROOT / "scripts" / "results" / "ztf"
REDO_CNN = "--redo-cnn" in sys.argv
DEEP = ["CNN1D", "LSTM"]
SEEDS = [int(s) for s in sys.argv[1:] if s.isdigit()] or [42, 123, 456, 789, 1024]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(name: str) -> dict[str, np.ndarray]:
    d = np.load(DATA / f"{name}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def tab_bases(split: dict) -> dict[str, np.ndarray]:
    per_band = extract_features(split["light_curves"])                       # (N, 20)
    bts = np.hstack([per_band, split["bts_extra"]])                          # (N, 26)
    z = np.hstack([bts, split["absmag"][:, None]])                           # (N, 27)
    return {"minimal": per_band, "bts": bts, "z": z}


def metrics(y, yp) -> dict[str, float]:
    return {"acc": float(accuracy_score(y, yp)), "bal_acc": float(balanced_accuracy_score(y, yp)),
            "f1_macro": float(f1_score(y, yp, average="macro"))}


def run_seed(seed: int, tr: dict, va: dict, te: dict) -> None:
    out_dir = RES / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = (out_dir / "meta.pkl").exists()
    if cached and not REDO_CNN:
        blob = pickle.load(open(out_dir / "meta.pkl", "rb"))
        if all(k in blob["meta"] for k in DEEP):
            logger.info(f"seed {seed}: cached, skip")
            return
    set_global_seed(seed)
    t0 = time.time()
    meta, basis, mets = {}, {}, {}
    if cached:
        blob = pickle.load(open(out_dir / "meta.pkl", "rb"))
        meta, basis, mets = blob["meta"], blob["basis"], blob["metrics"]
        logger.info(f"seed {seed}: XGBoost from cache, computing missing deep models")

    y_tr = np.concatenate([tr["labels"], va["labels"]])
    y_te = te["labels"]
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    sw = cw[y_tr]

    # ---- XGBoost on the three bases
    xb_tr, xb_va, xb_te = tab_bases(tr), tab_bases(va), tab_bases(te)
    for key, name, ra in [] if cached else (("minimal", "XGBoost-min", False), ("bts", "XGBoost-bts", False), ("z", "XGBoost-z", True)):
        x_tr = np.vstack([xb_tr[key], xb_va[key]])
        m = XGBClassifierWrapper(n_estimators=500, max_depth=8, learning_rate=0.1, random_state=seed)
        m.model.fit(x_tr, y_tr, sample_weight=sw)
        mets[name] = metrics(y_te, m.predict(xb_te[key]))
        # TreeSHAP via XGBoost's native implementation (identical algorithm to
        # shap.TreeExplainer for gradient-boosted trees; avoids importing shap).
        import xgboost as xgb_lib
        contrib = m.model.get_booster().predict(xgb_lib.DMatrix(xb_te[key]), pred_contribs=True)
        contrib = contrib.reshape(len(xb_te[key]), -1, xb_te[key].shape[1] + 1)[:, :, :-1]  # drop bias
        meta[name] = np.abs(contrib).mean(axis=1)                                          # (N, D)
        basis[name] = get_feature_names_ztf("bts" if key != "minimal" else "minimal", redshift_aware=ra)
        assert meta[name].shape[1] == len(basis[name]), (meta[name].shape, len(basis[name]))
        logger.info(f"seed {seed} {name}: {mets[name]}  meta {meta[name].shape}  ({time.time()-t0:.0f}s)")

    # ---- deep models (CNN1D, LSTM)
    def ts(split):
        return torch.from_numpy(split["light_curves"][:, :, :, 0]).float()
    x_va_ts, x_te_ts = ts(va), ts(te)
    n_classes = int(y_tr.max()) + 1
    raw = te["light_curves"][:, :, :, 0]
    for kind in DEEP:
        if kind in meta and not REDO_CNN:
            continue
        if kind == "CNN1D":
            model = CNN1D(n_bands=2, n_classes=n_classes, base_filters=64, dropout=0.5)
        else:
            from src.models.lstm_baseline import LSTMClassifier
            model = LSTMClassifier(n_bands=2, hidden_size=128, n_classes=n_classes,
                                   n_layers=2, dropout=0.3, bidirectional=True)
        cfg = {"training": {"early_stopping_patience": 15, "scheduler": "cosine", "gradient_clip": 1.0},
               "_model_config": {"epochs": 80, "lr": 1e-3, "weight_decay": 1e-3}}
        trainer = Trainer(model, cfg, device=device, seed=seed, restore_best=True)
        trainer.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(np.sqrt(cw), dtype=torch.float32, device=device))
        tl = DataLoader(TensorDataset(ts(tr), torch.from_numpy(tr["labels"])), batch_size=32, shuffle=True)
        vl = DataLoader(TensorDataset(x_va_ts, torch.from_numpy(va["labels"])), batch_size=32)
        trainer.fit(tl, vl)
        model.eval()
        with torch.no_grad():
            yp = torch.cat([model(xb.to(device)).argmax(1).cpu() for (xb,) in DataLoader(TensorDataset(x_te_ts), batch_size=256)]).numpy()
        mets[kind] = metrics(y_te, yp)
        trainer.save_checkpoint(out_dir / f"{kind.lower()}.pt")
        projected = []
        for i in range(len(raw)):
            attr = compute_integrated_gradients(model, x_te_ts[i:i + 1], target_class=None, n_steps=50)
            projected.append(project_timeseries_attribution(attr, raw[i]))
        meta[kind] = np.stack(projected)                                       # (N, 20)
        basis[kind] = get_feature_names_ztf("minimal")
        logger.info(f"seed {seed} {kind}: {mets[kind]}  meta {meta[kind].shape}  ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    with open(out_dir / "meta.pkl", "wb") as fh:
        pickle.dump({"meta": meta, "basis": basis, "metrics": mets, "labels_test": y_te}, fh)
    json.dump(mets, open(out_dir / "metrics.json", "w"), indent=1)


def main() -> None:
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    logger.info(f"train {len(tr['labels'])}  val {len(va['labels'])}  test {len(te['labels'])}  device {device}")
    for s in SEEDS:
        run_seed(s, tr, va, te)


if __name__ == "__main__":
    main()
