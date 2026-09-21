"""Paper 2 analysis: calibrated plausibility, per-slot standardisation, figures.

Runs on cached attributions only (scripts/results/seed_*/_pass2_meta.pkl).
Outputs JSON tables and PDF figures into paper2/.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.plausibility_calibration import (get_feature_names, MASK_BOOL, select, iou,
                                              RESULTS, BASE6, N_PERM)  # noqa

OUT = PROJECT_ROOT / "paper2"
FIG = OUT / "figures"
NAMES = get_feature_names()
STATS = ["mean","std","amplitude","median","n_obs","skewness","kurtosis","mean_snr","slope","frac_above_mean"]
MODELS = ["Random Forest","XGBoost","LSTM","CNN1D","CNN2D","ViT","Avocado","ParSNIP"]
PROJECTED = {"LSTM","CNN1D","CNN2D","ViT"}
PERCS = [25, 50, 60, 70, 75, 80, 90]


def standardise(a: np.ndarray, n_bands: int = 6, n_stats: int = 10) -> np.ndarray:
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


def analyse(attr, q, rng, n_perm=N_PERM):
    obs, cha, frac, ceil, k_all = [], [], [], [], []
    for a in attr:
        sel = select(a, q); k = int(sel.sum())
        obs.append(iou(sel)); k_all.append(k)
        frac.append((sel & MASK_BOOL).sum() / k if k else 0.0)
        ceil.append(min(k, 21) / max(k, 21) if k else 0.0)
        cha.append(np.mean([iou(select(a[rng.permutation(60)], q)) for _ in range(n_perm)]))
    o, c, ce = map(float, (np.mean(obs), np.mean(cha), np.mean(ceil)))
    return dict(observed=o, chance=c, ceiling=ce, in_mask=float(np.mean(frac)),
                rescaled=(o - c) / (ce - c) if ce > c else float("nan"), n_sel=float(np.mean(k_all)))


def main():
    rng = np.random.default_rng(42)
    seeds = {}
    for d in sorted(RESULTS.glob("seed_*")):
        p = d / "_pass2_meta.pkl"
        if p.exists():
            seeds[d.name] = pickle.load(open(p, "rb"))

    # ---------- Table: raw vs standardised, p50, five seeds ----------
    table = {}
    for mode in ("raw", "std"):
        per = {}
        for sname, blob in seeds.items():
            meta, w = blob["meta"], np.asarray(blob.get("w"))
            this = {}
            for m in MODELS:
                a = np.asarray(meta[m])
                if mode == "std" and m in PROJECTED:
                    a = standardise(a)
                this[m] = analyse(a, 50, rng)
            this["Ensemble"] = {k: float((w * np.array([this[b][k] for b in BASE6])).sum())
                                for k in this["LSTM"]}
            for m, r in this.items():
                per.setdefault(m, []).append(r)
        table[mode] = {m: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
                       for m, runs in per.items()}
        print(f"\n== {mode} (p50, {len(seeds)} seeds)")
        for m, r in table[mode].items():
            print(f"{m:<14} obs={r['observed']:.3f} ch={r['chance']:.3f} ce={r['ceiling']:.3f} "
                  f"resc={r['rescaled']:+.2f} in={r['in_mask']:.3f} |A|={r['n_sel']:.1f}")
    json.dump(table, open(OUT / "table_calibration.json", "w"), indent=1)

    # ---------- Sweep: percentile curves (seed 42, all samples, fewer perms) ----------
    meta42 = seeds["seed_42"]["meta"]
    sweep = {}
    for mode in ("raw", "std"):
        sweep[mode] = {}
        for m in MODELS:
            a = np.asarray(meta42[m])
            if mode == "std" and m in PROJECTED:
                a = standardise(a)
            sweep[mode][m] = {str(q): analyse(a, q, rng, n_perm=30) for q in PERCS}
    json.dump(sweep, open(OUT / "sweep_percentile.json", "w"), indent=1)

    # ---------- F1: observed and chance vs percentile ----------
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, MODELS):
        for mode, ls in (("raw", "-"), ("std", "--")):
            if mode == "std" and m not in PROJECTED:
                continue
            o = [sweep[mode][m][str(q)]["observed"] for q in PERCS]
            c = [sweep[mode][m][str(q)]["chance"] for q in PERCS]
            lab = "" if mode == "raw" else " (standardised)"
            ax.plot(PERCS, o, ls, color="C0", marker="o", ms=3, label="observed" + lab)
            ax.plot(PERCS, c, ls, color="C3", marker="s", ms=3, label="chance" + lab)
        ax.set_title(m); ax.grid(alpha=.3)
    axes[1, 0].set_xlabel("binarisation percentile $p$"); axes[0, 0].set_ylabel("IoU")
    for ax in axes[1]: ax.set_xlabel("binarisation percentile $p$")
    axes[0, 2].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig_threshold_sweep.pdf"); plt.close(fig)

    # ---------- F2: selection-frequency heatmap at p50 and p90 (raw) ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, q in zip(axes, (50, 90)):
        F = np.stack([np.stack([select(x, q) for x in np.asarray(meta42[m])]).mean(0) for m in MODELS])
        im = ax.imshow(F, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(60)); ax.set_xticklabels(NAMES, rotation=90, fontsize=5)
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=8)
        for i, inm in enumerate(MASK_BOOL):
            if inm: ax.axvspan(i - .5, i + .5, color="w", alpha=.12, lw=0)
        ax.set_title(f"selection frequency, $p={q}$")
    fig.colorbar(im, ax=axes, fraction=.02, pad=.01)
    fig.savefig(FIG / "fig_selection_frequency.pdf", bbox_inches="tight"); plt.close(fig)

    # ---------- F3: per-slot scale ----------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    scales = {}
    for i, m in enumerate(MODELS):
        a = np.abs(np.asarray(meta42[m])).reshape(-1, 6, 10)
        med = np.median(a, axis=(0, 1)); scales[m] = med.tolist()
        med = np.where(med > 0, med, np.nan)
        ax.plot(range(10), med, marker="o", ms=4, label=m, ls="-" if m in PROJECTED else ":")
    ax.set_yscale("log"); ax.set_xticks(range(10)); ax.set_xticklabels(STATS, rotation=35, ha="right")
    ax.set_ylabel("median $|a|$ over samples and bands"); ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig_slot_scale.pdf"); plt.close(fig)
    json.dump({"stats": STATS, "scales": scales}, open(OUT / "slot_scales.json", "w"), indent=1)
    print("\nfigures written to", FIG)


if __name__ == "__main__":
    main()
