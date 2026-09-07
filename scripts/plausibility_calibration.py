"""Calibration analysis for the cross-representation plausibility metric.

This is the analysis behind the Chance / Ceil. / Resc. columns of Table 5 of
Pagliaro (2026), and the starting point for the follow-up paper on metric
calibration.

The plausibility IoU thresholds each sample's |attribution| at the median of its
*positive* entries. The size of the selected set therefore depends on how sparse
the attribution is, which makes both the chance level and the reachable ceiling
model-dependent -- so raw IoU values are not comparable across models. This
script quantifies that:

  * observed  : the published plausibility, recomputed as a reproduction check
  * chance    : the same estimator on a random permutation of each attribution,
                which preserves sparsity and hence the selected-set size
  * ceiling   : the largest IoU reachable at that selected-set size
  * rescaled  : (observed - chance) / (ceiling - chance), the only quantity of
                the four that may be compared across model families
  * in-mask   : fraction of the selected features falling inside the mask,
                against the 21/60 = 0.35 expected of a random selection

It also sweeps the binarisation percentile, which is where the chance-relative
ordering of tree-based and time-series models inverts (p75).

Runs on cached attributions only: no GPU, no torch, no shap.

Usage:  python scripts/plausibility_calibration.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.metrics import build_expert_mask_tabular, plausibility_score  # noqa: E402

RESULTS = PROJECT_ROOT / "scripts" / "results"
N_PERM = 200
QUANTILES = (25, 50, 75, 90)
BASE6 = ["Random Forest", "XGBoost", "LSTM", "CNN1D", "CNN2D", "ViT"]

# Table 5, for the reproduction check.
PUBLISHED = {
    "Random Forest": 0.229, "XGBoost": 0.224, "LSTM": 0.381, "CNN1D": 0.398,
    "CNN2D": 0.149, "ViT": 0.257, "Avocado": 0.399, "ParSNIP": 0.215,
    "Ensemble": 0.250,
}


def get_feature_names(n_bands: int = 6) -> list[str]:
    """Copy of src/xai/shap_analysis.py::get_feature_names, inlined so that this
    script does not import shap."""
    bands = ["u", "g", "r", "i", "z", "Y"][:n_bands]
    stats = ["mean", "std", "amplitude", "median", "n_obs",
             "skewness", "kurtosis", "mean_snr", "slope", "frac_above_mean"]
    return [f"{b}_{s}" for b in bands for s in stats]


MASK = build_expert_mask_tabular(get_feature_names())
MASK_BOOL = MASK > 0
assert MASK.sum() == 21


def select(attr: np.ndarray, q: float) -> np.ndarray:
    a = np.abs(attr)
    pos = a[a > 0]
    if len(pos) == 0:
        return np.zeros_like(a, dtype=bool)
    return a >= np.percentile(pos, q)


def iou(sel: np.ndarray) -> float:
    union = (sel | MASK_BOOL).sum()
    return float((sel & MASK_BOOL).sum() / union) if union else 0.0


def analyse(attr: np.ndarray, q: float, rng: np.random.Generator) -> dict:
    obs, cha, frac, ceil = [], [], [], []
    for a in attr:
        sel = select(a, q)
        k = int(sel.sum())
        obs.append(iou(sel))
        frac.append(float((sel & MASK_BOOL).sum() / k) if k else 0.0)
        ceil.append(min(k, 21) / max(k, 21) if k else 0.0)
        cha.append(np.mean([iou(select(a[rng.permutation(len(a))], q))
                            for _ in range(N_PERM)]))
    o, c, ce = float(np.mean(obs)), float(np.mean(cha)), float(np.mean(ceil))
    return {"observed": o, "chance": c, "ceiling": ce, "in_mask": float(np.mean(frac)),
            "rescaled": (o - c) / (ce - c) if ce > c else float("nan"),
            "n_sel": float(np.mean([select(a, q).sum() for a in attr]))}


def main() -> None:
    rng = np.random.default_rng(42)
    per_model: dict[str, list[dict]] = {}

    for seed_dir in sorted(RESULTS.glob("seed_*")):
        pkl = seed_dir / "_pass2_meta.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as fh:
            blob = pickle.load(fh)
        meta, w = blob["meta"], blob.get("w")
        this = {name: analyse(np.asarray(a), 50, rng) for name, a in meta.items()}
        for name, res in this.items():
            per_model.setdefault(name, []).append(res)
        # The published Ensemble plausibility is the coefficient-weighted mean of
        # the base scores, not the score of a reconstructed attribution, so its
        # chance and ceiling are formed the same way.
        if w and all(b in this for b in BASE6):
            w = np.asarray(w)
            per_model.setdefault("Ensemble", []).append({
                k: float((w * np.array([this[b][k] for b in BASE6])).sum())
                for k in ("observed", "chance", "ceiling", "in_mask", "rescaled", "n_sel")})

    print(f"{'Model':<16}{'obs':>8}{'pub':>8}{'chance':>9}{'ceil':>8}"
          f"{'resc':>8}{'in-mask':>9}{'|A|':>7}")
    print("-" * 73)
    for name, runs in per_model.items():
        m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        pub = PUBLISHED.get(name, float("nan"))
        flag = "" if abs(m["observed"] - pub) < 0.002 else "  <<< non riproduce"
        print(f"{name:<16}{m['observed']:>8.3f}{pub:>8.3f}{m['chance']:>9.3f}"
              f"{m['ceiling']:>8.3f}{m['rescaled']:>8.2f}{m['in_mask']:>9.3f}"
              f"{m['n_sel']:>7.1f}{flag}")
    print("  in-mask atteso da selezione casuale: 21/60 = 0.350")

    print(f"\nsweep sui percentili (seed 42, osservato/caso, + sopra il caso):")
    with open(RESULTS / "seed_42" / "_pass2_meta.pkl", "rb") as fh:
        meta42 = pickle.load(fh)["meta"]
    print(f"{'Model':<16}" + "".join(f"{'p'+str(q):>17}" for q in QUANTILES))
    for name, a in meta42.items():
        cells = []
        for q in QUANTILES:
            r = analyse(np.asarray(a)[:50], q, rng)
            cells.append(f"{r['observed']:.3f}/{r['chance']:.3f}"
                         f"{'+' if r['observed'] > r['chance'] else '-'}")
        print(f"{name:<16}" + "".join(f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
