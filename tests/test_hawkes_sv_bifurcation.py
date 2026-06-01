"""Synthetic positive/negative controls for the repositioned Theorem 4 discriminator.

These tests are the honest replacement for the deleted n_SV ~1e-15 "verification":
instead of confirming a definition, they confirm that an OBSERVABLE statistic (a
sharp narrow-band spectral LINE at an interior frequency, rising above the smooth
red-noise / critical-slowing-down background) separates the two strata on simulator
data with KNOWN ground-truth coupling kappa / frequency omega.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.theory.hawkes_sv_bifurcation import (
    _phase_randomize,
    classify_stratum,
    simulate_sv_limit,
)


def test_positive_control_hopf_edge_is_oscillatory():
    # complex eigenvalue pair near the boundary: small kappa, nonzero omega -> Hopf side
    series = simulate_sv_limit(kappa=0.02, omega=0.30, n_steps=4096, noise=0.02, seed=1)
    res = classify_stratum(series)
    assert res.stratum == "hopf_edge", res
    assert res.peak_frequency > 0.0
    # a prominent line over the smooth background, clear of the default threshold (4.0)
    # and far above the real-edge continuum ceiling (~2.4 on the controls)
    assert res.line_prominence > 4.0
    # the oscillation frequency should sit near omega/(2*pi) cycles per sample
    expected = 0.30 / (2 * np.pi)
    assert abs(res.peak_frequency - expected) < 0.02


def test_negative_control_real_edge_is_nonoscillatory():
    # real eigenvalue near zero: small kappa, omega == 0 -> saddle-node / Hardiman side
    series = simulate_sv_limit(kappa=0.02, omega=0.0, n_steps=4096, noise=0.02, seed=2)
    res = classify_stratum(series)
    assert res.stratum == "real_edge", res
    # critically: NOT flagged as oscillatory even though it is near criticality
    assert res.branching_proxy >= 0.3
    # its strongest interior "line" is just continuum -- well below the Hopf threshold
    assert res.line_prominence < 4.0


def test_stable_region_not_flagged_as_edge():
    # far from the boundary: large kappa -> fast mean reversion, low autocorrelation
    series = simulate_sv_limit(kappa=2.0, omega=0.0, n_steps=4096, noise=1.0, seed=3)
    res = classify_stratum(series)
    assert res.stratum == "stable", res
    assert res.branching_proxy < 0.3


def test_discriminating_power_across_seeds():
    """Power proxy: across independent seeds the discriminator should call Hopf on the
    Hopf-side sims and NOT call Hopf on the saddle-node-side sims, most of the time."""
    hopf_calls = 0
    real_false_positives = 0
    n = 12
    for s in range(n):
        hopf_series = simulate_sv_limit(
            kappa=0.02, omega=0.30, n_steps=4096, noise=0.02, seed=100 + s
        )
        real_series = simulate_sv_limit(
            kappa=0.02, omega=0.0, n_steps=4096, noise=0.02, seed=200 + s
        )
        if classify_stratum(hopf_series).stratum == "hopf_edge":
            hopf_calls += 1
        if classify_stratum(real_series).stratum == "hopf_edge":
            real_false_positives += 1
    # high power on the Hopf side, low false-positive rate on the real side
    assert hopf_calls >= int(0.8 * n), f"power too low: {hopf_calls}/{n}"
    assert real_false_positives <= int(0.2 * n), (
        f"false-positive rate too high: {real_false_positives}/{n}"
    )


def test_classify_rejects_short_series():
    with pytest.raises(ValueError):
        classify_stratum(np.ones(8))


def test_phase_randomize_preserves_power_spectrum():
    # the exposed surrogate helper must preserve the magnitude spectrum (and hence
    # variance) while destroying phase -- it is the wrong null for peak significance
    # (which is precisely why classify_stratum does not use it), but it must be sound.
    rng = np.random.default_rng(0)
    x = simulate_sv_limit(kappa=0.02, omega=0.30, n_steps=512, noise=0.02, seed=5)
    surr = _phase_randomize(x, rng)
    assert surr.shape == x.shape
    px = np.abs(np.fft.rfft(x - x.mean()))
    ps = np.abs(np.fft.rfft(surr - surr.mean()))
    # magnitude spectra match to numerical precision (phase-only change)
    assert np.allclose(px, ps, atol=1e-8)


def test_odd_length_series_classifies():
    # exercise the odd-length branch of the surrogate / FFT machinery end-to-end
    series = simulate_sv_limit(kappa=0.02, omega=0.30, n_steps=4097, noise=0.02, seed=6)
    res = classify_stratum(series)
    assert res.stratum in {"hopf_edge", "real_edge", "stable"}


def test_short_series_uses_single_segment_welch_path():
    # a series too short for n_segments segments forces the single-segment Welch
    # fallback (the `if not starts` branch); it must still classify without error.
    series = simulate_sv_limit(kappa=0.02, omega=0.30, n_steps=20, noise=0.02, seed=7)
    res = classify_stratum(series, n_segments=8)
    assert res.stratum in {"hopf_edge", "real_edge", "stable"}
    assert res.peak_frequency >= 0.0


def test_constant_series_zero_variance_guards():
    # a constant series exercises the zero-variance autocorrelation guard and the
    # zero/degenerate background fallback; lag-1 autocorr is 0 -> "stable".
    res = classify_stratum(np.full(256, 3.0))
    assert res.stratum == "stable"
    assert res.branching_proxy == 0.0


def test_high_rolloff_skip_falls_back_to_full_band():
    # rolloff_skip_frac large enough to skip the whole band must fall back to skip=1
    # (the `if skip >= freqs.size` guard) and still return a valid classification.
    series = simulate_sv_limit(kappa=0.02, omega=0.30, n_steps=4096, noise=0.02, seed=8)
    res = classify_stratum(series, rolloff_skip_frac=5.0)
    assert res.stratum in {"hopf_edge", "real_edge", "stable"}
