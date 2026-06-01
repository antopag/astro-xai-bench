"""Recompute cross-representation plausibility from cached model artifacts.

Recomputes the plausibility metric from cached model checkpoints and cached
SHAP/attribution artifacts *without* re-running model training. Used in
revision pass 2 to apply the trimmed 21-feature expert mask retroactively to
the cached attribution outputs, bypassing the full (multi-hour) retraining loop.

For each seed it (re)derives every model's per-sample 60-dimensional
meta-attribution once -- TreeSHAP for the tree-based and physics-informed
LightGBM models (Random Forest, XGBoost, Avocado, ParSNIP), Integrated
Gradients for the deep-learning models (LSTM, CNN1D, CNN2D, ViT) loaded from
their checkpoints, and the coefficient-weighted average for the stacking
ensemble -- caches it (``scripts/results/seed_*/_pass2_meta.pkl``), and scores
the IoU plausibility against BOTH the old 29-feature mask (to verify exact
reproduction of the published ``results.json`` values) and the new 21-feature
mask (the reported numbers), at the p25/p50/p75/p90 thresholds. Because the
per-sample binarisation is mask-independent, the cached meta-attributions can
be re-scored against any mask instantly.

The plausibility values in Table 5 and Table A.2 of the revised manuscript are
produced by this script; it is the reproducibility entry point for those
numbers.

Usage (conda env ``astro-xai``):
    python scripts/recompute_plausibility.py 42     # single-seed smoke test
    python scripts/recompute_plausibility.py        # full 5-seed run

Output: ``scripts/results/_pass2_plausibility.json`` (per-seed + cross-seed,
both masks, all quantiles) plus a printed old-mask reproduction check.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import shap
import torch

from src.models.tabular_rf import RFClassifier, XGBClassifierWrapper, extract_features
from src.xai.shap_analysis import compute_shap_values, get_feature_names
from src.xai.shap_utils import normalize_shap_values
from src.xai.integrated_gradients import compute_integrated_gradients
from src.xai.plausibility import (project_timeseries_attribution,
                                  project_image_attribution)
from src.models.avocado.xai import FEATURE_MAPPING as AVO_MAP
from src.models.parsnip.xai import LATENT_TO_60D as PAR_MAP
from src.training.trainer import set_global_seed

DATA = ROOT / "data" / "processed" / "plasticc"
RES = ROOT / "scripts" / "results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 50
QUANTILES = {"p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}
SEEDS = [int(a) for a in sys.argv[1:]] or [42, 123, 456, 789, 1024]

try:  # Windows console/file default cp1252 cannot encode the report glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

print(f"device={device}  seeds={SEEDS}")

# ---------------------------------------------------------------- masks
feat_names = get_feature_names()
assert len(feat_names) == 60, f"expected 60 feature names, got {len(feat_names)}"
fidx = {n: i for i, n in enumerate(feat_names)}


def build_mask(include_mean_median: bool) -> np.ndarray:
    feats = set()
    for b in ["u", "g", "r", "i", "z", "Y"]:
        feats.add(f"{b}_amplitude")
        feats.add(f"{b}_std")
    for b in ["g", "r", "i"]:
        feats.add(f"{b}_slope")
        feats.add(f"{b}_skewness")
        feats.add(f"{b}_mean_snr")
    if include_mean_median:
        for b in ["g", "r", "i", "z"]:
            feats.add(f"{b}_mean")
            feats.add(f"{b}_median")
    return np.array([1.0 if n in feats else 0.0 for n in feat_names])


MASK29 = build_mask(True)
MASK21 = build_mask(False)
assert MASK29.sum() == 29, MASK29.sum()
assert MASK21.sum() == 21, MASK21.sum()
print(f"masks OK: old={int(MASK29.sum())}, new={int(MASK21.sum())}")


def iou_at(attr: np.ndarray, mask: np.ndarray, q: float) -> float:
    a = np.abs(attr)
    pos = a[a > 0]
    if len(pos) == 0:
        return 0.0
    t = float(np.quantile(pos, q))
    ab = (a >= t).astype(float)
    mb = (mask > 0).astype(float)
    inter = (ab * mb).sum()
    union = np.clip(ab + mb, 0.0, 1.0).sum()
    return float(inter / union) if union > 1e-10 else 0.0


def score_block(meta: np.ndarray) -> dict:
    """For a (Nsamp, 60) meta-attribution matrix, return plausibility for both
    masks at all quantiles: {mask: {quantile: {mean, std}}}."""
    out = {}
    for mname, mask in (("m29", MASK29), ("m21", MASK21)):
        out[mname] = {}
        for qname, q in QUANTILES.items():
            scores = [iou_at(meta[i], mask, q) for i in range(len(meta))]
            out[mname][qname] = {"mean": float(np.mean(scores)),
                                 "std": float(np.std(scores))}
    return out


def tab_mean_attr(shap_values, n_features: int = 60) -> np.ndarray:
    """Replicate compute_tabular_plausibility's mean-|SHAP|-over-classes."""
    if isinstance(shap_values, list):
        return np.mean([np.abs(sv) for sv in shap_values], axis=0)
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        if sv.shape[2] == n_features:
            return np.abs(sv).mean(axis=1)
        return np.abs(sv).mean(axis=2)
    return np.abs(sv)


def project_lgb(booster_path: Path, features: np.ndarray, feat_list, mapping,
                idx: np.ndarray, distribute: bool) -> np.ndarray:
    """SHAP -> 60-d projection for a LightGBM head (Avocado / ParSNIP)."""
    booster = lgb.Booster(model_file=str(booster_path))
    ms = normalize_shap_values(shap.TreeExplainer(booster).shap_values(features[idx]))
    meta = np.zeros((len(idx), 60))
    for s in range(len(idx)):
        attr = ms[s]
        for j, fn in enumerate(feat_list):
            if fn in mapping:
                tgt = mapping[fn]
                if distribute:  # ParSNIP: list of targets, equal split
                    for tf in tgt:
                        if tf in fidx:
                            meta[s, fidx[tf]] += attr[j] / len(tgt)
                else:  # Avocado: single target
                    if tgt in fidx:
                        meta[s, fidx[tgt]] += attr[j]
    return meta


# ---------------------------------------------------------------- data (shared)
print("loading data...")
train = np.load(DATA / "train.npz")
val = np.load(DATA / "val.npz")
test = np.load(DATA / "test.npz")
y_train, y_val, y_test = train["labels"], val["labels"], test["labels"]
n_classes = len(np.unique(y_train))

x_train_tab = extract_features(np.vstack([train["light_curves"], val["light_curves"]]))
y_train_tab = np.concatenate([y_train, y_val])
x_val_tab = extract_features(val["light_curves"])
x_test_tab = extract_features(test["light_curves"])

lc_val = val["light_curves"][:, :, :, 0].astype(np.float32)
lc_test = test["light_curves"][:, :, :, 0].astype(np.float32)

gaf = np.load(DATA / "gaf_224.npz")
gaf_val, gaf_test = gaf["val"], gaf["test"]
print(f"data loaded: test tab {x_test_tab.shape}, lc {lc_test.shape}, gaf {gaf_test.shape}")


def load_dl(kind: str, ckpt: Path):
    if kind == "lstm":
        from src.models.lstm_baseline import LSTMClassifier
        m = LSTMClassifier(n_bands=6, hidden_size=128, n_classes=n_classes,
                           n_layers=2, dropout=0.3, bidirectional=True)
    elif kind == "cnn1d":
        from src.models.cnn_baseline import CNN1D
        m = CNN1D(n_bands=6, n_classes=n_classes, base_filters=64, dropout=0.3)
    elif kind == "cnn2d":
        from src.models.cnn_baseline import CNN2D
        m = CNN2D(n_bands=6, n_classes=n_classes, base_filters=32, dropout=0.3)
    elif kind == "vit":
        from src.models.vit import ViTClassifier
        m = ViTClassifier(n_bands=6, n_classes=n_classes,
                          model_name="vit_small_patch16_224", pretrained=False, dropout=0.1)
    sd = torch.load(ckpt, map_location=device)["model_state_dict"]
    m.load_state_dict(sd)
    return m.to(device).eval()


def dl_meta(model, x_raw: np.ndarray, idx: np.ndarray, representation: str) -> np.ndarray:
    metas = []
    for i in idx:
        t = torch.from_numpy(x_raw[i:i + 1]).float().to(device)
        attr = compute_integrated_gradients(model, t, target_class=None, n_steps=50)
        if representation == "timeseries":
            metas.append(project_timeseries_attribution(attr, x_raw[i]))
        else:
            metas.append(project_image_attribution(attr))
    return np.stack(metas, axis=0)


BASE6 = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT"]


def build_meta(seed: int, seed_dir: Path, ck: Path):
    """Return (meta_store, ensemble_weights). Cached to seed_dir/_pass2_meta.pkl
    so the expensive SHAP/IG is computed at most once per seed."""
    cache = seed_dir / "_pass2_meta.pkl"
    if cache.is_file():
        with open(cache, "rb") as f:
            d = pickle.load(f)
        print(f"  [cache] loaded meta for seed {seed}")
        return d["meta"], d.get("w")

    t0 = time.time()
    set_global_seed(seed)
    meta_store = {}

    # RF / XGB : all test samples
    rf = RFClassifier(n_estimators=500, random_state=seed)
    rf.fit(x_train_tab, y_train_tab)
    meta_store["Random Forest"] = tab_mean_attr(compute_shap_values(rf, x_test_tab, method="tree"))
    print(f"  RF meta {meta_store['Random Forest'].shape}  ({time.time()-t0:.0f}s)")
    xgb = XGBClassifierWrapper(n_estimators=500, random_state=seed)
    xgb.fit(x_train_tab, y_train_tab)
    meta_store["XGBoost"] = tab_mean_attr(compute_shap_values(xgb, x_test_tab, method="tree"))
    print(f"  XGB meta {meta_store['XGBoost'].shape}  ({time.time()-t0:.0f}s)")

    # DL : 50-sample subset (same rng as run_final_table)
    idx_dl = np.random.default_rng(seed).choice(len(lc_test), size=min(N, len(lc_test)), replace=False)
    for kind, name, x_raw, rep in [
        ("lstm", "LSTM", lc_test, "timeseries"),
        ("cnn1d", "CNN1D", lc_test, "timeseries"),
        ("cnn2d", "CNN2D", gaf_test, "image"),
        ("vit", "ViT", gaf_test, "image"),
    ]:
        ckpt = ck / f"{kind}.pt"
        if not ckpt.is_file():
            print(f"  !! {name}: checkpoint missing {ckpt}")
            continue
        model = load_dl(kind, ckpt)
        meta_store[name] = dl_meta(model, x_raw, idx_dl, rep)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {name} meta {meta_store[name].shape}  ({time.time()-t0:.0f}s)")

    # Avocado / ParSNIP (cached artifacts, no VAE/augmentation re-run)
    try:
        ap = np.load(seed_dir / "avocado_predictions.npz")
        xa, fa = ap["test_features"], list(ap["feature_names"])
        ia = np.random.default_rng(seed).choice(len(xa), size=min(N, len(xa)), replace=False)
        meta_store["Avocado"] = project_lgb(ck / "avocado_lgb.txt", xa, fa, AVO_MAP, ia, distribute=False)
        print(f"  Avocado meta {meta_store['Avocado'].shape}")
    except Exception as e:
        print(f"  !! Avocado failed: {e!r}")
    try:
        pp = np.load(seed_dir / "parsnip_classifier_features.npz")
        xp, fp = pp["features"], list(pp["feature_names"])
        ip = np.random.default_rng(seed).choice(len(xp), size=min(N, len(xp)), replace=False)
        meta_store["ParSNIP"] = project_lgb(ck / "parsnip_lgb.txt", xp, fp, PAR_MAP, ip, distribute=True)
        print(f"  ParSNIP meta {meta_store['ParSNIP'].shape}")
    except Exception as e:
        print(f"  !! ParSNIP failed: {e!r}")

    # Ensemble coef weights (mask-independent; refit meta-learner on val)
    w = None
    try:
        from src.models.ensemble import StackingEnsemble
        models = {k: load_dl(k, ck / f"{k}.pt") for k in ("lstm", "cnn1d", "cnn2d", "vit")}
        ens = StackingEnsemble([rf, xgb, models["lstm"], models["cnn1d"], models["cnn2d"], models["vit"]],
                               meta_learner="logistic", random_state=seed)
        ens.fit([x_val_tab, x_val_tab, lc_val, lc_val, gaf_val, gaf_val], y_val)
        coef_mag = np.abs(ens.meta.coef_).sum(axis=0)
        w = np.array([coef_mag[i * n_classes:(i + 1) * n_classes].sum() for i in range(6)])
        w = w / w.sum() if w.sum() > 0 else np.ones(6) / 6
        for m in models.values():
            del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  Ensemble weights {np.round(w, 3)}")
    except Exception as e:
        print(f"  !! Ensemble weights failed: {e!r}")

    with open(cache, "wb") as f:
        pickle.dump({"meta": meta_store, "w": (w.tolist() if w is not None else None)}, f)
    print(f"  meta built + cached in {(time.time()-t0)/60:.1f} min")
    return meta_store, (w.tolist() if w is not None else None)


def score_all(meta_store: dict, w) -> dict:
    """Score every model (both masks, all quantiles); add coef-weighted Ensemble."""
    res = {m: score_block(meta) for m, meta in meta_store.items()}
    if w is not None and all(b in res for b in BASE6):
        w = np.asarray(w)
        res["Ensemble"] = {}
        for mname in ("m29", "m21"):
            res["Ensemble"][mname] = {}
            for qname in QUANTILES:
                means = np.array([res[b][mname][qname]["mean"] for b in BASE6])
                stds = np.array([res[b][mname][qname]["std"] for b in BASE6])
                res["Ensemble"][mname][qname] = {
                    "mean": float((w * means).sum()),
                    "std": float(np.sqrt((w**2 * stds**2).sum()))}
    return res


# ---------------------------------------------------------------- main loop
all_results = {}
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    seed_dir = RES / f"seed_{seed}"
    ck = seed_dir / "checkpoints"
    meta_store, w = build_meta(seed, seed_dir, ck)
    all_results[str(seed)] = score_all(meta_store, w)

# ---------------------------------------------------------------- aggregate + verify
out = {"per_seed": all_results, "cross_seed": {}, "verification": {}}
models = sorted({m for s in all_results.values() for m in s})
for m in models:
    out["cross_seed"][m] = {}
    for mname in ("m29", "m21"):
        out["cross_seed"][m][mname] = {}
        for qname in QUANTILES:
            vals = [all_results[s][m][mname][qname]["mean"]
                    for s in all_results if m in all_results[s]]
            out["cross_seed"][m][mname][qname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "per_seed": [round(v, 4) for v in vals]}

# verification vs cached results.json (old 29-mask, p50, cross-seed mean)
print(f"\n{'='*70}\nVERIFICATION: recomputed 29-mask p50 vs cached results.json\n{'='*70}")
print(f"{'Model':16s} {'recomputed':>11s} {'cached':>9s} {'delta':>9s}")
cached = {}
for s in all_results:
    rj = RES / f"seed_{s}" / "results.json"
    if rj.is_file():
        cached[s] = json.load(open(rj))
for m in models:
    rec = out["cross_seed"][m]["m29"]["p50"]["mean"]
    cvals = [cached[s][m]["plausibility_mean"] for s in cached
             if m in cached.get(s, {}) and "plausibility_mean" in cached[s].get(m, {})]
    cmean = float(np.mean(cvals)) if cvals else float("nan")
    d = rec - cmean
    flag = "" if (cvals and abs(d) < 0.01) else "  <-- CHECK"
    out["verification"][m] = {"recomputed_m29_p50": rec, "cached": cmean, "delta": d}
    print(f"{m:16s} {rec:11.4f} {cmean:9.4f} {d:+9.4f}{flag}")

print(f"\n{'='*70}\nNEW 21-mask plausibility (p50, cross-seed mean ± std)\n{'='*70}")
for m in models:
    b = out["cross_seed"][m]["m21"]["p50"]
    print(f"{m:16s} {b['mean']:.4f} +/- {b['std']:.4f}   per-seed {b['per_seed']}")

json.dump(out, open(RES / "_pass2_plausibility.json", "w"), indent=2)
print(f"\nsaved -> {RES / '_pass2_plausibility.json'}")
