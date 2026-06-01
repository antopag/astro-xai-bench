"""Generate the GAF pipeline schematic (Figure A.1) for the revision.

Visualises the four stages that turn a 1-D light curve into a Gramian Angular
(Summation) Field image: raw series -> Piecewise Aggregate Approximation (PAA)
+ min-max scaling to [-1, 1] -> angular encoding phi = arccos(x) -> outer
product GASF[i,j] = cos(phi_i + phi_j).

Spyder: %runfile scripts/make_gaf_schematic.py --wdir
Paths are anchored to PROJECT_ROOT (not cwd) per the project convention.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- a synthetic single-band transient light curve (schematic only) ----
t = np.linspace(0, 1, 120)
flux = (np.exp(-((t - 0.32) ** 2) / 0.004)               # fast rise
        + 0.55 * np.exp(-((t - 0.5) / 0.18) ** 2)         # broad decline
        + 0.05 * np.sin(2 * np.pi * 5 * t))               # small wiggle
flux = flux / flux.max()

# ---- stage 2: PAA to n_paa segments ----
n_paa = 16
paa = flux[: (len(flux) // n_paa) * n_paa].reshape(n_paa, -1).mean(axis=1)
# min-max scale to [-1, 1]
scaled = 2 * (paa - paa.min()) / (paa.max() - paa.min()) - 1
scaled = np.clip(scaled, -1, 1)

# ---- stage 3: angular encoding ----
phi = np.arccos(scaled)                                   # in [0, pi]

# ---- stage 4: GASF ----
gasf = np.cos(phi[:, None] + phi[None, :])                # [-1, 1]

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
fig.subplots_adjust(wspace=0.45, left=0.05, right=0.97, bottom=0.18, top=0.84)

# (a) raw
ax = axes[0]
ax.plot(t, flux, color="#1f77b4", lw=1.6)
ax.set_title("(a) Light curve", fontsize=11)
ax.set_xlabel("time")
ax.set_ylabel("normalised flux")
ax.set_yticks([0, 0.5, 1.0])

# (b) PAA + scaling
ax = axes[1]
centres = (np.arange(n_paa) + 0.5) / n_paa
ax.step(np.concatenate([[0], centres, [1]]),
        np.concatenate([[scaled[0]], scaled, [scaled[-1]]]),
        where="mid", color="#d62728", lw=1.6)
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_title(f"(b) PAA ($n={n_paa}$),\nscaled to $[-1,1]$", fontsize=11)
ax.set_xlabel("time")
ax.set_ylabel(r"$\tilde{x}_i$")
ax.set_ylim(-1.15, 1.15)

# (c) angular encoding on the unit circle
ax = axes[2]
ax.set_aspect("equal")
th = np.linspace(0, np.pi, 200)
ax.plot(np.cos(th), np.sin(th), color="grey", lw=0.8)
ax.scatter(np.cos(phi), np.sin(phi), c=np.arange(n_paa), cmap="viridis", s=28, zorder=3)
for k in (0, n_paa // 2, n_paa - 1):
    ax.plot([0, np.cos(phi[k])], [0, np.sin(phi[k])], color="grey", lw=0.6, ls="--")
ax.set_title(r"(c) Angular encoding" + "\n" + r"$\phi_i=\arccos(\tilde{x}_i)$", fontsize=11)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-0.15, 1.15)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([0, 1])

# (d) GASF matrix
ax = axes[3]
im = ax.imshow(gasf, cmap="rainbow", vmin=-1, vmax=1, origin="upper")
ax.set_title(r"(d) GASF" + "\n" + r"$G_{ij}=\cos(\phi_i+\phi_j)$", fontsize=11)
ax.set_xlabel("$j$")
ax.set_ylabel("$i$")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_ticks([-1, 0, 1])

# arrows between panels
for x in (0.265, 0.505, 0.745):
    fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
        (x, 0.5), (x + 0.022, 0.5), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, color="black", lw=1.0))

out = FIG_DIR / "gaf_schematic.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
