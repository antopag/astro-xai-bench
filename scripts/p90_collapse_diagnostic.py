"""Why does plausibility IoU collapse to ~0 at p90 for the gradient-based models?

For each model (seed 42, full sample) and each percentile, count how often each
of the 60 features enters the selected set, split in-mask / out-of-mask, and
report the top selected features.  Runs on cached attributions only.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.plausibility_calibration import get_feature_names, MASK_BOOL, select  # noqa

NAMES = get_feature_names()
with open(PROJECT_ROOT / "scripts/results/seed_42/_pass2_meta.pkl", "rb") as fh:
    meta = pickle.load(fh)["meta"]

print("mask features:", [n for n, m in zip(NAMES, MASK_BOOL) if m])
for q in (50, 75, 90):
    print(f"\n================ p{q}")
    for name, a in meta.items():
        a = np.asarray(a)
        sel = np.stack([select(x, q) for x in a])          # (N, 60)
        freq = sel.mean(0)
        k = sel.sum(1)
        in_frac = (sel & MASK_BOOL).sum(1) / np.maximum(k, 1)
        # sparsity of the attribution itself
        nz = (np.abs(a) > 0).sum(1).mean()
        top = np.argsort(-freq)[:8]
        stats_by_band = {}
        for i in np.where(freq > 0)[0]:
            b, s = NAMES[i].split("_", 1)
            stats_by_band[b] = stats_by_band.get(b, 0) + freq[i]
        print(f"{name:<14} |A|={k.mean():5.1f} nonzero={nz:5.1f} in-mask={in_frac.mean():.3f}"
              f"  band-share={ {b: round(v/sum(stats_by_band.values()),2) for b,v in stats_by_band.items()} }")
        print("     top:", ", ".join(f"{NAMES[i]}{'*' if MASK_BOOL[i] else ''}({freq[i]:.2f})" for i in top))
print("\n(* = in expert mask)")


# ---- scale of each projected statistic slot (median over samples and bands)
STATS = ["mean","std","amplitude","median","n_obs","skewness","kurtosis","mean_snr","slope","frac_above_mean"]
print("\n================ median |value| per statistic slot (over samples x bands)")
print(f"{'model':<14}" + "".join(f"{s:>12}" for s in STATS))
for name, a in meta.items():
    a = np.abs(np.asarray(a)).reshape(len(a), 6, 10)
    med = np.median(a, axis=(0, 1))
    print(f"{name:<14}" + "".join(f"{v:>12.2e}" for v in med))
