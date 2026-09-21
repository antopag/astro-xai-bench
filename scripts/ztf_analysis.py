"""Paper 2, Sect. 6: calibrated plausibility on ZTF BTS under Mask A and Mask B.

Reads scripts/results/ztf/seed_*/meta.pkl (from run_ztf.py).  For each model
and mask: observed / chance / ceiling / rescaled / in-mask / |A| at p50 (five
seeds), the percentile sweep (seed 42), raw and after per-slot standardisation
for the projected model.  Writes paper2/ztf_tables.json and
paper2/figures/fig_ztf_sweep.pdf.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.xai.masks_ztf import build_expert_mask_ztf, get_feature_names_ztf  # noqa: E402

RES = PROJECT_ROOT / "scripts" / "results" / "ztf"
OUT = PROJECT_ROOT / "paper2"
N_PERM = 200
PERCS = [25, 50, 60, 70, 75, 80, 90]
MODELS = ["XGBoost-min", "XGBoost-bts", "XGBoost-z", "CNN1D", "LSTM"]


def select(a, q):
    a = np.abs(a); pos = a[a > 0]
    return a >= np.percentile(pos, q) if len(pos) else np.zeros_like(a, bool)


def iou(sel, mask):
    u = (sel | mask).sum()
    return float((sel & mask).sum() / u) if u else 0.0


def analyse(attr, mask, q, rng, n_perm=N_PERM):
    K = int(mask.sum()); D = len(mask)
    obs, cha, frac, ceil, ks = [], [], [], [], []
    for a in attr:
        sel = select(a, q); k = int(sel.sum())
        obs.append(iou(sel, mask)); ks.append(k)
        frac.append((sel & mask).sum() / k if k else 0.0)
        ceil.append(min(k, K) / max(k, K) if k else 0.0)
        cha.append(np.mean([iou(select(a[rng.permutation(D)], q), mask) for _ in range(n_perm)]))
    o, c, ce = map(float, (np.mean(obs), np.mean(cha), np.mean(ceil)))
    return dict(observed=o, chance=c, ceiling=ce, in_mask=float(np.mean(frac)),
                rescaled=(o - c) / (ce - c) if ce > c else float("nan"), n_sel=float(np.mean(ks)), K=K, D=D)


def standardise(a: np.ndarray, n_bands: int = 2, n_stats: int = 10) -> np.ndarray:
    """Per-slot rank standardisation with mid-ranks for ties (a constant slot
    maps to 0.5 and can never dominate the selection), scaled to (0, 1].  Zeros stay
    zero.  a: (N, D) with the per-band block first."""
    x = np.abs(a).astype(float).copy()
    nb = n_bands * n_stats
    blk = x[:, :nb].reshape(len(x), n_bands, n_stats)
    for s in range(n_stats):
        v = blk[:, :, s]; pos = v > 0
        if pos.sum() == 0:
            continue
        srt = np.sort(v[pos])
        v[pos] = (np.searchsorted(srt, v[pos], "left") + np.searchsorted(srt, v[pos], "right")) / (2 * len(srt))
        blk[:, :, s] = v
    x[:, :nb] = blk.reshape(len(x), nb)
    return x


def to_basis(attr, names_from, names_to):
    """Re-index an attribution matrix onto another basis (missing slots -> 0)."""
    out = np.zeros((len(attr), len(names_to)))
    for j, n in enumerate(names_to):
        if n in names_from:
            out[:, j] = attr[:, names_from.index(n)]
    return out


def mask_for(model_basis, which):
    """Mask A on the minimal basis; Mask B on the BTS basis of the model (26 or 27)."""
    if which == "A":
        names = get_feature_names_ztf("minimal")
    else:
        names = get_feature_names_ztf("bts", redshift_aware=("r_peak_absmag" in model_basis))
    return names, build_expert_mask_ztf(names, "minimal" if which == "A" else "bts") > 0


def main():
    rng = np.random.default_rng(42)
    seeds = {d.name: pickle.load(open(d / "meta.pkl", "rb")) for d in sorted(RES.glob("seed_*")) if (d / "meta.pkl").exists()}
    print("seeds:", list(seeds))
    tables, sweep = {}, {}
    for which in ("A", "B"):
        for mode in ("raw", "std"):
            key = f"mask{which}_{mode}"; per = {}
            for sname, blob in seeds.items():
                for m in MODELS:
                    if m not in blob["meta"]:
                        continue
                    a, b = np.asarray(blob["meta"][m]), blob["basis"][m]
                    if mode == "std" and m in ("CNN1D", "LSTM"):
                        a = standardise(a)
                    elif mode == "std":
                        continue
                    names, mask = mask_for(b, which)
                    per.setdefault(m, []).append(analyse(to_basis(a, b, names), mask, 50, rng))
            tables[key] = {m: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]} for m, runs in per.items()}
            print(f"\n== {key} ({len(seeds)} seeds, p50)")
            for m, r in tables[key].items():
                print(f"{m:<12} K/D={r['K']:.0f}/{r['D']:.0f} obs={r['observed']:.3f} ch={r['chance']:.3f} "
                      f"ce={r['ceiling']:.3f} resc={r['rescaled']:+.2f} in={r['in_mask']:.3f} |A|={r['n_sel']:.1f}")
    # accuracy summary
    acc = {}
    for sname, blob in seeds.items():
        for m, v in blob["metrics"].items():
            acc.setdefault(m, []).append(v)
    tables["metrics"] = {m: {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in runs[0]} for m, runs in acc.items()}
    print("\n== classification (mean, std over seeds)")
    for m, r in tables["metrics"].items():
        print(f"{m:<12}" + "".join(f"  {k}={v[0]:.3f}±{v[1]:.3f}" for k, v in r.items()))

    # sweep, seed 42
    blob = seeds["seed_42"]
    fig, axes = plt.subplots(2, len(MODELS), figsize=(3 * len(MODELS), 5), sharex=True, sharey=True)
    for col, m in enumerate(MODELS):
        a, b = np.asarray(blob["meta"][m]), blob["basis"][m]
        for row, which in enumerate(("A", "B")):
            names, mask = mask_for(b, which); ax = axes[row, col]
            for mode, ls in (("raw", "-"), ("std", "--")):
                if mode == "std" and m not in ("CNN1D", "LSTM"):
                    continue
                aa = standardise(a) if mode == "std" else a
                rs = [analyse(to_basis(aa, b, names), mask, q, rng, n_perm=30) for q in PERCS]
                sweep[f"{m}_mask{which}_{mode}"] = {str(q): r for q, r in zip(PERCS, rs)}
                ax.plot(PERCS, [r["observed"] for r in rs], ls, color="C0", marker="o", ms=3, label="observed" + ("" if mode == "raw" else " (std.)"))
                ax.plot(PERCS, [r["chance"] for r in rs], ls, color="C3", marker="s", ms=3, label="chance" + ("" if mode == "raw" else " (std.)"))
            ax.set_title(f"{m}, Mask {which}", fontsize=9); ax.grid(alpha=.3)
    for ax in axes[1]: ax.set_xlabel("binarisation percentile $p$")
    axes[0, 0].set_ylabel("IoU"); axes[1, 0].set_ylabel("IoU"); axes[0, -1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "fig_ztf_sweep.pdf"); plt.close(fig)
    json.dump({"tables": tables, "sweep": sweep}, open(OUT / "ztf_tables.json", "w"), indent=1)
    print("\nwritten", OUT / "ztf_tables.json")


if __name__ == "__main__":
    main()
