"""Paper 2, second review round: denominator safeguard (Q9) and complexity
re-ranking with sparsity-invariant measures (Q10).  Cached attributions only.

Writes paper2/review2_tables.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.paper2_review_analysis import (MASK60, calibrate, harmonise, load_plasticc,  # noqa: E402
                                            load_ztf, to_basis)
from src.xai.masks_ztf import build_expert_mask_ztf, get_feature_names_ztf  # noqa: E402
from src.xai.metrics import explanation_complexity  # noqa: E402

OUT = PROJECT_ROOT / "paper2"
PERCS = [25, 50, 60, 70, 75, 80, 90]
PM = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT", "Avocado", "ParSNIP"]
PROJ = {"LSTM", "CNN1D", "CNN2D", "ViT"}


# ------------------------------------------------------------ Q9: denominator
def denominators(P, Z):
    """Per-sample and mean (ceil - chance) for every model / threshold / mask."""
    rows = {}
    for m in PM:
        A = P["seed_42"][m]
        A = harmonise(A, 6, 10, "rank") if m in PROJ else A
        for q in PERCS:
            c = calibrate(A, MASK60, q)
            d = c["ceil"] - c["chance"]
            rows[f"PLAsTiCC|{m}|p{q}"] = dict(mean=float(d.mean()), min=float(d.min()),
                                              frac_below_0p05=float((d < 0.05).mean()), k=float(c["k"].mean()))
    for m in ["XGBoost-min", "XGBoost-bts", "XGBoost-z", "CNN1D", "LSTM"]:
        blob = Z["seed_42"]; A, b = np.asarray(blob["meta"][m]), blob["basis"][m]
        if m in ("CNN1D", "LSTM"):
            A = harmonise(A, 2, 10, "rank")
        for which in ("A", "B"):
            names = get_feature_names_ztf("minimal") if which == "A" else get_feature_names_ztf("bts", redshift_aware=("r_peak_absmag" in b))
            mask = build_expert_mask_ztf(names, "minimal" if which == "A" else "bts") > 0
            X = to_basis(A, b, names)
            for q in PERCS:
                c = calibrate(X, mask, q); d = c["ceil"] - c["chance"]
                rows[f"ZTF|{m}|mask{which}|p{q}"] = dict(mean=float(d.mean()), min=float(d.min()),
                                                         frac_below_0p05=float((d < 0.05).mean()), k=float(c["k"].mean()))
    return rows


# ------------------------------------------------------------ Q10: complexity
def gini(v: np.ndarray) -> float:
    """Gini index of |v| (1 = one feature carries everything, 0 = uniform), sparsity-invariant
    in the sense of Hurley & Rickard (2009): zeros are kept, N fixed."""
    x = np.sort(np.abs(v).astype(float)); n = len(x); s = x.sum()
    if s <= 0:
        return 0.0
    k = np.arange(1, n + 1)
    return float(1 - 2 * np.sum((x / s) * (n - k + 0.5) / n))


def entropy_logD(v: np.ndarray) -> float:
    """Normalised entropy of |v| with the normaliser log D (D fixed), zeros included."""
    x = np.abs(v).astype(float); s = x.sum()
    if s <= 0:
        return 0.0
    p = x / s; p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(v)))


def complexity_table(P):
    out = {}
    for m in PM:
        per = {"paper1": [], "logD": [], "gini": [], "band_paper1": [], "band_gini": [], "n_eff": []}
        for meta in P.values():
            A = np.abs(np.asarray(meta[m]))
            per["paper1"].append(np.mean([explanation_complexity(a) for a in A]))
            per["logD"].append(np.mean([entropy_logD(a) for a in A]))
            per["gini"].append(np.mean([gini(a) for a in A]))
            B = A.reshape(len(A), 6, 10).sum(2)                      # band-aggregated (6-d)
            per["band_paper1"].append(np.mean([explanation_complexity(b) for b in B]))
            per["band_gini"].append(np.mean([gini(b) for b in B]))
            per["n_eff"].append(float((A > 0).sum(1).mean()))
        out[m] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per.items()}
    return out


def main():
    P, Z = load_plasticc(), load_ztf()
    R = {"denominators": denominators(P, Z), "complexity": complexity_table(P)}
    print("== denominator (ceil - chance), seed 42, standardised projections")
    for k, v in R["denominators"].items():
        if v["mean"] < 0.25 or v["frac_below_0p05"] > 0:
            print(f"  {k:<40} mean={v['mean']:.3f} min={v['min']:.3f} frac<0.05={v['frac_below_0p05']:.2f} |A|={v['k']:.1f}")
    small = min(R["denominators"].items(), key=lambda t: t[1]["mean"])
    print("  smallest mean:", small)
    print("\n== complexity (5 seeds)")
    print(f"{'model':<14}{'N_eff':>7}{'paper1':>9}{'logD':>8}{'gini':>8}{'band_p1':>9}{'band_gini':>10}")
    for m, r in R["complexity"].items():
        print(f"{m:<14}{r['n_eff'][0]:>7.1f}{r['paper1'][0]:>9.3f}{r['logD'][0]:>8.3f}{r['gini'][0]:>8.3f}{r['band_paper1'][0]:>9.3f}{r['band_gini'][0]:>10.3f}")
    for key in ("paper1", "logD", "gini", "band_paper1", "band_gini"):
        order = sorted(PM, key=lambda m: R["complexity"][m][key][0])
        print(f"  rank by {key:<12}: " + " < ".join(order))
    json.dump(R, open(OUT / "review2_tables.json", "w"), indent=1)
    print("written", OUT / "review2_tables.json")


if __name__ == "__main__":
    main()
