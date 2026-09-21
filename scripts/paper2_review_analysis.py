"""Paper 2, revision analysis (referee questions), from cached attributions.

Adds to paper2_analysis.py / ztf_analysis.py:
  * exact (hypergeometric) chance level, no Monte Carlo -- and the MC error of
    the 200-permutation estimator used in the first submission
  * support-restricted null and ceiling ("support-aware rescaled score")
  * bootstrap CIs over test objects and seed spread for the rescaled score
  * alternative per-slot harmonisations: rank, rank on a held-out reference
    half, z-score, robust (median/MAD), quantile-to-normal
  * threshold-agnostic summary: mean rescaled score over p in [25, 90]
  * alternative binarisations: fixed k = K, global (dataset-level) percentile

Writes paper2/review_tables.json.  Runs in minutes (vectorised).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.plausibility_calibration import MASK_BOOL as MASK60  # noqa: E402
from src.xai.masks_ztf import build_expert_mask_ztf, get_feature_names_ztf  # noqa: E402

OUT = PROJECT_ROOT / "paper2"
RES = PROJECT_ROOT / "scripts" / "results"
PERCS = [25, 50, 60, 70, 75, 80, 90]
rng = np.random.default_rng(42)


# ----------------------------------------------------------------- core
def select_matrix(A: np.ndarray, q: float) -> np.ndarray:
    """Per-sample percentile binarisation (Eq. 1), vectorised. A: (N, D) >= 0."""
    A = np.abs(A)
    sel = np.zeros_like(A, dtype=bool)
    for i, a in enumerate(A):
        pos = a[a > 0]
        if len(pos):
            sel[i] = a >= np.percentile(pos, q)
    return sel


def iou_rows(sel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inter = (sel & mask).sum(1); union = (sel | mask).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def chance_exact(k: np.ndarray, K: int, D: int) -> np.ndarray:
    """E[IoU] of a uniformly random k-subset of D against a K-mask, per sample."""
    out = np.zeros(len(k))
    for kk in np.unique(k):
        if kk == 0:
            continue
        x = np.arange(0, min(kk, K) + 1)
        pmf = hypergeom.pmf(x, D, K, kk)
        out[k == kk] = (pmf * x / (kk + K - x)).sum()
    return out


def ceiling(k: np.ndarray, K: int) -> np.ndarray:
    return np.where(k > 0, np.minimum(k, K) / np.maximum(np.maximum(k, K), 1), 0.0)


def calibrate(A: np.ndarray, mask: np.ndarray, q: float, support_null: bool = False) -> dict:
    """Per-sample observed / chance / ceiling / rescaled.  With support_null the
    null and ceiling are computed inside each sample's non-zero support."""
    sel = select_matrix(A, q); k = sel.sum(1)
    obs = iou_rows(sel, mask)
    if support_null:
        supp = np.abs(A) > 0
        D_s = supp.sum(1); K_s = (supp & mask).sum(1)
        # a random k-subset of the support hits K_s mask features inside it;
        # the mask features outside the support always count in the union
        K_out = mask.sum() - K_s
        cha = np.zeros(len(k)); ce = np.zeros(len(k))
        for i in range(len(k)):
            if k[i] == 0:
                continue
            x = np.arange(0, min(k[i], K_s[i]) + 1)
            pmf = hypergeom.pmf(x, D_s[i], K_s[i], k[i])
            cha[i] = (pmf * x / (k[i] + K_s[i] + K_out[i] - x)).sum()
            m = min(k[i], K_s[i])
            ce[i] = m / (k[i] + K_s[i] + K_out[i] - m)
    else:
        cha = chance_exact(k, int(mask.sum()), len(mask)); ce = ceiling(k, int(mask.sum()))
    ok = ce > cha
    resc = np.where(ok, (obs - cha) / np.where(ok, ce - cha, 1), np.nan)
    return dict(obs=obs, chance=cha, ceil=ce, resc=resc, k=k)


def summary(c: dict) -> dict:
    o, ch, ce = c["obs"].mean(), c["chance"].mean(), c["ceil"].mean()
    return dict(observed=float(o), chance=float(ch), ceiling=float(ce),
                rescaled=float((o - ch) / (ce - ch)) if ce > ch else float("nan"), n_sel=float(c["k"].mean()))


def boot_ci(c: dict, n_boot: int = 1000) -> tuple[float, float]:
    """Bootstrap over test objects of the aggregate rescaled score."""
    N = len(c["obs"]); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        o, ch, ce = c["obs"][idx].mean(), c["chance"][idx].mean(), c["ceil"][idx].mean()
        vals.append((o - ch) / (ce - ch) if ce > ch else np.nan)
    return tuple(float(x) for x in np.nanpercentile(vals, [2.5, 97.5]))


# ----------------------------------------------------- harmonisations
def harmonise(A: np.ndarray, n_bands: int, n_stats: int, how: str, ref: np.ndarray | None = None) -> np.ndarray:
    """Per-slot scale harmonisation of the per-band block of A (N, >= n_bands*n_stats).
    ref: optional reference matrix whose slot distributions define the map."""
    A = np.abs(A).astype(float).copy(); R = np.abs(ref).astype(float) if ref is not None else A
    nb = n_bands * n_stats
    blk = A[:, :nb].reshape(len(A), n_bands, n_stats); rb = R[:, :nb].reshape(len(R), n_bands, n_stats)
    for s in range(n_stats):
        v = blk[:, :, s]; r = rb[:, :, s]; pos = v > 0; rpos = r[r > 0]
        if rpos.size == 0 or pos.sum() == 0:
            continue
        if how == "rank":
            srt = np.sort(rpos); v[pos] = (np.searchsorted(srt, v[pos], "left") + np.searchsorted(srt, v[pos], "right")) / (2 * len(srt))
        elif how == "zscore":
            v[pos] = (v[pos] - rpos.mean()) / (rpos.std() + 1e-12); v[pos] = v[pos] - v[pos].min() + 1e-9
        elif how == "robust":
            med = np.median(rpos); mad = np.median(np.abs(rpos - med)) + 1e-12
            v[pos] = (v[pos] - med) / mad; v[pos] = v[pos] - v[pos].min() + 1e-9
        elif how == "quantile":
            srt = np.sort(rpos); u = (np.searchsorted(srt, v[pos], "left") + np.searchsorted(srt, v[pos], "right") + 1) / (2 * (len(srt) + 1))
            v[pos] = norm.ppf(u) - norm.ppf(1 / (len(srt) + 1)) + 1e-9
        else:
            raise ValueError(how)
        blk[:, :, s] = v
    A[:, :nb] = blk.reshape(len(A), nb)
    return A


# ----------------------------------------------- alternative binarisations
def fixed_k(A: np.ndarray, K: int) -> np.ndarray:
    A = np.abs(A); sel = np.zeros_like(A, dtype=bool)
    for i, a in enumerate(A):
        nz = np.flatnonzero(a); top = nz[np.argsort(-a[nz])[:K]]; sel[i, top] = True
    return sel


def global_percentile(A: np.ndarray, q: float) -> np.ndarray:
    A = np.abs(A); thr = np.percentile(A[A > 0], q); return A >= thr


def calib_from_sel(sel: np.ndarray, mask: np.ndarray) -> dict:
    k = sel.sum(1); obs = iou_rows(sel, mask)
    cha = chance_exact(k, int(mask.sum()), len(mask)); ce = ceiling(k, int(mask.sum()))
    return dict(obs=obs, chance=cha, ceil=ce, resc=np.full(len(k), np.nan), k=k)


# ------------------------------------------------------------------ data
def load_plasticc() -> dict[str, dict[str, np.ndarray]]:
    seeds = {}
    for d in sorted(RES.glob("seed_*")):
        p = d / "_pass2_meta.pkl"
        if p.exists():
            seeds[d.name] = {m: np.asarray(a) for m, a in pickle.load(open(p, "rb"))["meta"].items()}
    return seeds


def load_ztf():
    seeds = {}
    for d in sorted((RES / "ztf").glob("seed_*")):
        if (d / "meta.pkl").exists():
            seeds[d.name] = pickle.load(open(d / "meta.pkl", "rb"))
    return seeds


def to_basis(A, names_from, names_to):
    out = np.zeros((len(A), len(names_to)))
    for j, n in enumerate(names_to):
        if n in names_from:
            out[:, j] = A[:, names_from.index(n)]
    return out


# ------------------------------------------------------------------ main
def main():
    R: dict = {}
    P = load_plasticc(); Z = load_ztf()
    print("PLAsTiCC seeds", list(P), " ZTF seeds", list(Z))
    PROJ = {"LSTM", "CNN1D", "CNN2D", "ViT"}
    PM = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT", "Avocado", "ParSNIP"]

    # ---- 1. exact chance vs 200-permutation MC, and MC error (PLAsTiCC seed 42)
    A42 = P["seed_42"]
    mc = {}
    for m in ["XGBoost", "CNN1D", "Avocado"]:
        for q in (50, 90):
            sel = select_matrix(A42[m], q); k = sel.sum(1)
            exact = chance_exact(k, 21, 60)
            ests = []
            for n_perm in (50, 200, 1000):
                vals = []
                for i, a in enumerate(np.abs(A42[m])):
                    vals.append(np.mean([iou_rows(select_matrix(a[rng.permutation(60)][None], q), MASK60)[0]
                                         for _ in range(n_perm)]))
                ests.append((n_perm, float(np.mean(vals)), float(np.std(np.array(vals) - exact) / np.sqrt(len(vals)))))
            mc[f"{m}_p{q}"] = dict(exact=float(exact.mean()), mc=ests)
            print(f"MC check {m} p{q}: exact={exact.mean():.4f}", ests)
    R["mc_check"] = mc

    # ---- 2. PLAsTiCC: full-basis vs support-restricted null, bootstrap CI, seed spread
    tab = {}
    for m in PM:
        per_seed = {"full": [], "supp": []}; cis = []
        for sname, meta in P.items():
            A = meta[m]
            for key, sn in (("full", False), ("supp", True)):
                c = calibrate(A, MASK60, 50, support_null=sn); per_seed[key].append(summary(c))
                if key == "full" and sname == "seed_42":
                    cis.append(boot_ci(c))
        tab[m] = {k: {f: (float(np.mean([r[f] for r in v])), float(np.std([r[f] for r in v]))) for f in v[0]}
                  for k, v in per_seed.items()}
        tab[m]["boot95_full_seed42"] = cis[0]
        print(f"{m:<14} full resc={tab[m]['full']['rescaled'][0]:+.2f}±{tab[m]['full']['rescaled'][1]:.2f} "
              f"CI95={cis[0][0]:+.2f}..{cis[0][1]:+.2f}   supp resc={tab[m]['supp']['rescaled'][0]:+.2f} "
              f"(chance {tab[m]['supp']['chance'][0]:.3f} ceil {tab[m]['supp']['ceiling'][0]:.3f})")
    R["plasticc_nulls"] = tab

    # ---- 3. harmonisation alternatives (projected models, 5 seeds, p50 and p90)
    harm = {}
    for m in ["LSTM", "CNN1D", "CNN2D", "ViT"]:
        harm[m] = {}
        for how in ("raw", "rank", "rank_ref", "zscore", "robust", "quantile"):
            rows = {50: [], 90: []}
            for meta in P.values():
                A = meta[m]
                if how == "raw":
                    H = np.abs(A)
                elif how == "rank_ref":
                    # ranks defined on one random half, applied to the other half
                    idx = rng.permutation(len(A)); h = len(A) // 2
                    H = harmonise(A[idx[h:]], 6, 10, "rank", ref=A[idx[:h]])
                else:
                    H = harmonise(A, 6, 10, how)
                for q in (50, 90):
                    rows[q].append(summary(calibrate(H, MASK60, q)))
            harm[m][how] = {q: {f: float(np.mean([r[f] for r in v])) for f in v[0]} for q, v in rows.items()}
        print(m, {how: (round(v[50]["rescaled"], 2), round(v[90]["rescaled"], 2)) for how, v in harm[m].items()})
    R["harmonisation"] = harm

    # ---- 4. threshold-agnostic summary: mean rescaled over p in [25, 90] (5 seeds)
    auc = {}
    for m in PM:
        for how in (("raw",) if m not in PROJ else ("raw", "rank")):
            vals = []
            for meta in P.values():
                A = meta[m] if how == "raw" else harmonise(meta[m], 6, 10, "rank")
                vals.append(np.mean([summary(calibrate(A, MASK60, q))["rescaled"] for q in PERCS]))
            auc[f"{m}|{how}"] = (float(np.mean(vals)), float(np.std(vals)))
    print("AUC-p:", {k: f"{v[0]:+.2f}±{v[1]:.2f}" for k, v in auc.items()})
    R["auc_p"] = auc

    # ---- 5. alternative binarisations (seed 42, raw and rank for projected)
    binz = {}
    for m in PM:
        A = A42[m] if m not in PROJ else harmonise(A42[m], 6, 10, "rank")
        binz[m] = {}
        for name, sel in (("percentile_p50", select_matrix(A, 50)), ("fixed_k21", fixed_k(A, 21)),
                          ("global_p50", global_percentile(A, 50))):
            binz[m][name] = summary(calib_from_sel(sel, MASK60))
        print(m, {n: (round(v["observed"], 3), round(v["rescaled"], 2), round(v["n_sel"], 1)) for n, v in binz[m].items()})
    R["binarisation"] = binz

    # ---- 6. ZTF: support-restricted null under Mask B, bootstrap, seed spread
    ztf = {}
    for m in ["XGBoost-min", "XGBoost-bts", "XGBoost-z", "CNN1D", "LSTM"]:
        if m not in Z["seed_42"]["meta"]:
            continue
        ztf[m] = {}
        for which in ("A", "B"):
            per = {"full": [], "supp": []}; ci = None
            for sname, blob in Z.items():
                A, b = np.asarray(blob["meta"][m]), blob["basis"][m]
                if m in ("CNN1D", "LSTM"):
                    A = harmonise(A, 2, 10, "rank")
                names = get_feature_names_ztf("minimal") if which == "A" else \
                    get_feature_names_ztf("bts", redshift_aware=("r_peak_absmag" in b))
                mask = build_expert_mask_ztf(names, "minimal" if which == "A" else "bts") > 0
                X = to_basis(A, b, names)
                for key, sn in (("full", False), ("supp", True)):
                    c = calibrate(X, mask, 50, support_null=sn); per[key].append(summary(c))
                    if key == "full" and sname == "seed_42":
                        ci = boot_ci(c)
            ztf[m][which] = {k: {f: (float(np.mean([r[f] for r in v])), float(np.std([r[f] for r in v]))) for f in v[0]}
                             for k, v in per.items()}
            ztf[m][which]["boot95_full_seed42"] = ci
            print(f"ZTF {m:<12} mask{which} full={ztf[m][which]['full']['rescaled'][0]:+.2f}±{ztf[m][which]['full']['rescaled'][1]:.2f} "
                  f"CI={ci[0]:+.2f}..{ci[1]:+.2f}  supp={ztf[m][which]['supp']['rescaled'][0]:+.2f}")
    R["ztf"] = ztf
    if "LSTM" in Z["seed_42"]["metrics"]:
        R["ztf_metrics"] = {m: {k: (float(np.mean([b["metrics"][m][k] for b in Z.values()])),
                                    float(np.std([b["metrics"][m][k] for b in Z.values()])))
                                for k in Z["seed_42"]["metrics"][m]} for m in Z["seed_42"]["metrics"]}

    json.dump(R, open(OUT / "review_tables.json", "w"), indent=1, default=float)
    print("written", OUT / "review_tables.json")


if __name__ == "__main__":
    main()
