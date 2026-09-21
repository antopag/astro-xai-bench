"""Expert masks and feature bases for ZTF Bright Transient Survey (g, r).

Two masks are defined, to be used together (paper 2, Sect. 6):

* **minimal** -- the PLAsTiCC rule of ``build_expert_mask_tabular`` applied to the
  20-feature basis (10 per-band statistics x 2 bands).  Controlled replica: the
  only change with respect to PLAsTiCC is the number of passbands.
* **bts** -- the 20 per-band statistics plus 6 cross-band / timescale features
  that the BTS classification literature actually uses (Perley et al. 2020;
  Fremling et al. 2020), with a mask that follows that literature.  Robustness
  test: does the set of models above chance depend on the mask?

Chance and ceiling of the calibrated plausibility depend only on (|A_p|, K, D),
so the two masks are directly comparable once rescaled.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

ZTF_BANDS = ["g", "r"]
STATS = ["mean", "std", "amplitude", "median", "n_obs",
         "skewness", "kurtosis", "mean_snr", "slope", "frac_above_mean"]

#: Cross-band and timescale features appended in the BTS basis, in this order.
BTS_EXTRA = [
    "gr_colour_peak",     # g - r (in mag) at r-band peak
    "gr_colour_slope",    # d(g - r)/dt over the observed span, mag/day
    "r_rise_time",        # days from first detection to r peak
    "r_fade_time",        # days from r peak to last detection
    "r_peak_mag",         # apparent r magnitude at peak
    "r_half_peak_width",  # days above half of r peak flux
]

#: Appended only for the redshift-aware model (BTS spectroscopic z available).
BTS_ABSMAG = "r_peak_absmag"


def get_feature_names_ztf(basis: str = "minimal", redshift_aware: bool = False) -> list[str]:
    """Feature names of the ZTF tabular basis.

    Args:
        basis: ``"minimal"`` (20 features) or ``"bts"`` (26 features).
        redshift_aware: With ``basis="bts"``, append the absolute peak magnitude
            (27 features).  Only for models that take the redshift as input;
            flux-only models keep the apparent magnitude alone.

    Returns:
        Ordered list of feature names.
    """
    names = [f"{b}_{s}" for b in ZTF_BANDS for s in STATS]
    if basis == "bts":
        names += BTS_EXTRA
        if redshift_aware:
            names.append(BTS_ABSMAG)
    elif basis != "minimal":
        raise ValueError(f"unknown basis {basis!r}")
    return names


def build_expert_mask_ztf(feature_names: list[str], mask: str = "minimal") -> np.ndarray:
    """Build the ZTF expert mask.

    ``minimal`` applies the PLAsTiCC rule literally: amplitude and std in every
    band; slope, skewness and mean_snr in the "good" bands, which for ZTF are
    both.  K = 10 of D = 20.

    ``bts`` keeps the same per-band core (amplitude, std, slope in g and r:
    brightness, variability, evolution rate -- what a BTS classifier reads off a
    light curve first) and adds the five cross-band / timescale features that
    Perley et al. (2020) use to separate SN Ia, SN II, SN Ib/c, SLSN and TDE:
    colour at peak, colour evolution, rise time, fade time, and peak
    magnitude.  Skewness and mean_snr are dropped: the former is superseded by
    the explicit rise/fade asymmetry, the latter is a detectability proxy that
    BTS's magnitude cut (< 18.5 at peak) makes uninformative.  The half-peak
    width is excluded as redundant with rise + fade.  K = 11 of D = 26.
    If the basis carries the absolute peak magnitude (redshift-aware model),
    it replaces the apparent one in the mask, so K stays 11 of D = 27.

    Args:
        feature_names: Output of :func:`get_feature_names_ztf` for the matching basis.
        mask: ``"minimal"`` or ``"bts"``.

    Returns:
        Binary array of shape ``(len(feature_names),)``.
    """
    if mask == "minimal":
        keep = {f"{b}_{s}" for b in ZTF_BANDS
                for s in ("amplitude", "std", "slope", "skewness", "mean_snr")}
    elif mask == "bts":
        keep = {f"{b}_{s}" for b in ZTF_BANDS for s in ("amplitude", "std", "slope")}
        keep |= {"gr_colour_peak", "gr_colour_slope", "r_rise_time", "r_fade_time"}
        keep.add(BTS_ABSMAG if BTS_ABSMAG in feature_names else "r_peak_mag")
    else:
        raise ValueError(f"unknown mask {mask!r}")

    missing = keep - set(feature_names)
    if missing:
        raise ValueError(f"mask {mask!r} needs features absent from basis: {sorted(missing)}")
    out = np.array([1.0 if n in keep else 0.0 for n in feature_names])
    logger.info(f"ZTF expert mask [{mask}]: {int(out.sum())}/{len(out)} features selected")
    return out


def extract_bts_extra(light_curves: np.ndarray, times: np.ndarray,
                      zeropoint: float = 27.5) -> np.ndarray:
    """Compute the six BTS cross-band / timescale features.

    Args:
        light_curves: ``(N, 2, T, 2)`` flux and flux_err, band order (g, r);
            zero flux marks a missing epoch, as in the PLAsTiCC arrays.
        times: ``(N, 2, T)`` observation times in days (any zero-point).
        zeropoint: AB zero-point for the flux-to-magnitude conversion.

    Returns:
        ``(N, 6)`` array in the order of :data:`BTS_EXTRA`.  Undefined values
        (no detections, non-positive flux) are 0.
    """
    n = light_curves.shape[0]
    out = np.zeros((n, len(BTS_EXTRA)), dtype=np.float32)
    for i in range(n):
        fg, fr = light_curves[i, 0, :, 0], light_curves[i, 1, :, 0]
        tg, tr = times[i, 0], times[i, 1]
        mg, mr = fg > 0, fr > 0
        if mr.sum() < 2:
            continue
        tr_, fr_ = tr[mr], fr[mr]
        k = int(np.argmax(fr_))
        t_peak, f_peak = tr_[k], fr_[k]
        r_peak_mag = -2.5 * np.log10(f_peak) + zeropoint
        rise = t_peak - tr_.min()
        fade = tr_.max() - t_peak
        half = float((fr_ >= 0.5 * f_peak).sum()) * (np.ptp(tr_) / max(len(tr_) - 1, 1))
        colour_peak = colour_slope = 0.0
        if mg.sum() >= 1:
            tg_, fg_ = tg[mg], fg[mg]
            j = int(np.argmin(np.abs(tg_ - t_peak)))
            colour_peak = (-2.5 * np.log10(fg_[j]) + zeropoint) - r_peak_mag
            if mg.sum() >= 2 and np.ptp(tg_) > 0:
                # colour at each g epoch against r interpolated at that epoch
                r_interp = np.interp(tg_, tr_, fr_)
                ok = r_interp > 0
                if ok.sum() >= 2:
                    gr = (-2.5 * np.log10(fg_[ok]) + zeropoint) - (-2.5 * np.log10(r_interp[ok]) + zeropoint)
                    colour_slope = float(np.polyfit(tg_[ok], gr, 1)[0])
        out[i] = [colour_peak, colour_slope, rise, fade, r_peak_mag, half]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def absolute_peak_mag(r_peak_mag: np.ndarray, redshift: np.ndarray,
                      H0: float = 70.0, Om0: float = 0.3) -> np.ndarray:
    """Absolute r magnitude at peak from apparent magnitude and spectroscopic z.

    Flat LCDM luminosity distance; no K-correction (BTS z < 0.1, the term is
    below the photometric scatter).  Zero apparent magnitude (undefined) maps
    to zero.

    Args:
        r_peak_mag: ``(N,)`` apparent magnitudes.
        redshift: ``(N,)`` spectroscopic redshifts.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    z = np.asarray(redshift, dtype=np.float64)
    ok = (np.asarray(r_peak_mag) != 0) & (z > 0)
    out = np.zeros_like(np.asarray(r_peak_mag, dtype=np.float32))
    if ok.any():
        dm = cosmo.distmod(z[ok]).value
        out[ok] = np.asarray(r_peak_mag)[ok] - dm
    return out
