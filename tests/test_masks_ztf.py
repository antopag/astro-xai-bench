"""Tests for the ZTF expert masks and BTS extra features."""
import numpy as np
import pytest

from src.xai.masks_ztf import (BTS_ABSMAG, BTS_EXTRA, absolute_peak_mag, build_expert_mask_ztf, extract_bts_extra,
                               get_feature_names_ztf)


def test_minimal_basis_and_mask():
    names = get_feature_names_ztf("minimal")
    assert len(names) == 20
    m = build_expert_mask_ztf(names, "minimal")
    assert m.sum() == 10
    assert m[names.index("g_amplitude")] == 1 and m[names.index("r_kurtosis")] == 0


def test_bts_basis_and_mask():
    names = get_feature_names_ztf("bts")
    assert len(names) == 26 and names[-6:] == BTS_EXTRA
    m = build_expert_mask_ztf(names, "bts")
    assert m.sum() == 11
    assert m[names.index("gr_colour_peak")] == 1
    assert m[names.index("r_half_peak_width")] == 0
    assert m[names.index("g_skewness")] == 0


def test_bts_mask_requires_bts_basis():
    with pytest.raises(ValueError):
        build_expert_mask_ztf(get_feature_names_ztf("minimal"), "bts")


def test_extract_bts_extra_on_synthetic_sn():
    t = np.linspace(0, 60, 31)
    # r peaks at day 20; g is bluer before peak and reddens after
    fr = 100 * np.exp(-0.5 * ((t - 20) / 8) ** 2) + 1
    fg = fr * np.where(t < 20, 1.3, 0.7)
    lc = np.zeros((1, 2, 31, 2)); lc[0, 0, :, 0] = fg; lc[0, 1, :, 0] = fr
    times = np.stack([t, t])[None]
    x = extract_bts_extra(lc, times)
    colour_peak, colour_slope, rise, fade, peak_mag, half = x[0]
    assert abs(rise - 20) < 1e-6 and abs(fade - 40) < 1e-6
    assert colour_slope > 0            # reddening with time
    assert 0 < half < 60
    assert abs(peak_mag - (-2.5 * np.log10(101) + 27.5)) < 1e-4


def test_extract_bts_extra_no_detections_is_zero():
    lc = np.zeros((1, 2, 10, 2)); times = np.zeros((1, 2, 10))
    assert np.all(extract_bts_extra(lc, times) == 0)


def test_redshift_aware_basis_swaps_peak_mag_in_mask():
    names = get_feature_names_ztf("bts", redshift_aware=True)
    assert len(names) == 27 and names[-1] == BTS_ABSMAG
    m = build_expert_mask_ztf(names, "bts")
    assert m.sum() == 11
    assert m[names.index(BTS_ABSMAG)] == 1 and m[names.index("r_peak_mag")] == 0


def test_absolute_peak_mag():
    M = absolute_peak_mag(np.array([18.0, 0.0]), np.array([0.05, 0.05]))
    assert -19.5 < M[0] < -18.0 and M[1] == 0
