"""Tests for the kappa per-USD <-> dimensionless change-of-variables.

Validates (1) drift invariance of the change of variables to machine precision,
(2) round-trip of the forward/inverse maps, (3) the characteristic-scale reader
agreeing with the raw aggregator, and (4) the honest INDETERMINACY result:
across plausible SPX (S0, OI) assumptions the dimensionless image of
kappa_0 ~ 5e-12 spans several orders of magnitude and brackets kappa* = 17.81.
"""

from __future__ import annotations

import numpy as np

from reflexive_options.simulator.gamma_aggregator import (
    GammaAggregator,
    GammaAggregatorConfig,
)
from reflexive_options.theory.kappa_rescaling import (
    characteristic_gamma_scale,
    dimensionless_kappa_band,
    dimensionless_to_per_usd,
    flat_oi_characteristic_scale,
    per_usd_to_dimensionless,
)
from reflexive_options.types import OpenInterestGrid, SurfaceGrid

KAPPA0 = 5e-12  # per-USD literature anchor (theory.md L627 / 6.5)
KAPPA_STAR = 17.8065068  # dimensionless Hopf threshold (theory.md 4.3.5)


def _flat_aggregator(spot: float, oi_per_cell: float, iv: float = 0.20) -> GammaAggregator:
    logm = np.linspace(-0.15, 0.15, 7)
    mats = np.array([1.0 / 12, 3.0 / 12, 6.0 / 12])
    grid = SurfaceGrid(log_moneyness=logm.astype(np.float64), maturities=mats.astype(np.float64))
    oi = OpenInterestGrid(grid=grid, contracts_open=np.full((7, 3), float(oi_per_cell)))
    return GammaAggregator(
        oi_grid=oi,
        risk_free_rate=0.0,
        config=GammaAggregatorConfig(multiplier=100.0, fixed_iv=iv),
    )


def test_change_of_variables_preserves_drift_exactly() -> None:
    """kappa_per_usd * G == kappa_dimensionless * (G / G_char) for every state."""
    iv = 0.20
    agg = _flat_aggregator(spot=5000.0, oi_per_cell=1e6, iv=iv)
    g_char = characteristic_gamma_scale(agg, spot=5000.0, variance=iv * iv)
    kappa_dim = per_usd_to_dimensionless(KAPPA0, g_char)

    for x in (-0.10, -0.03, 0.0, 0.03, 0.10):
        g_usd = agg.compute(5000.0 * float(np.exp(x)), iv * iv, 0.0)
        drift_per_usd = KAPPA0 * g_usd
        drift_dimensionless = kappa_dim * (g_usd / g_char)
        # The map is algebraically exact; the only residual is double-precision
        # rounding from the multiply/divide by g_char (~1e-15 relative).
        assert abs(drift_per_usd - drift_dimensionless) <= 1e-12 * abs(drift_per_usd)


def test_forward_inverse_round_trip() -> None:
    g_char = 1.86e12
    kd = per_usd_to_dimensionless(KAPPA0, g_char)
    back = dimensionless_to_per_usd(kd, g_char)
    assert abs(back - KAPPA0) <= 1e-24


def test_characteristic_scale_matches_raw_aggregator_peak() -> None:
    iv = 0.20
    agg = _flat_aggregator(spot=5000.0, oi_per_cell=1e6, iv=iv)
    g_char = characteristic_gamma_scale(agg, spot=5000.0, variance=iv * iv, n_scan=61)
    xs = np.linspace(-0.15, 0.15, 61)
    raw_peak = max(abs(agg.compute(5000.0 * float(np.exp(x)), iv * iv, 0.0)) for x in xs)
    assert g_char == raw_peak
    assert g_char > 0.0


def test_characteristic_scale_grows_linearly_in_spot() -> None:
    """G_char ~ S0 on the level. The aggregator forms m*S^2*sum(q*Gamma_BS), and
    per-share BS gamma carries a 1/S, so the net dollar-gamma level scales as S^1:
    a decade in S0 ~ one decade in G_char."""
    iv = 0.20
    g_lo = characteristic_gamma_scale(
        _flat_aggregator(spot=100.0, oi_per_cell=5e4, iv=iv), spot=100.0, variance=iv * iv
    )
    g_hi = characteristic_gamma_scale(
        _flat_aggregator(spot=1000.0, oi_per_cell=5e4, iv=iv), spot=1000.0, variance=iv * iv
    )
    ratio = g_hi / g_lo
    # one decade from S^1 scaling (the moneyness shape is scale-invariant in log-spot).
    assert 9.0 <= ratio <= 11.0


def test_flat_oi_convenience_matches_manual_build() -> None:
    iv = 0.20
    manual = characteristic_gamma_scale(
        _flat_aggregator(spot=5000.0, oi_per_cell=1e6, iv=iv), spot=5000.0, variance=iv * iv
    )
    conv = flat_oi_characteristic_scale(spot=5000.0, oi_per_cell=1e6, iv=iv)
    assert abs(conv - manual) <= 1e-6 * manual


def test_dimensionless_image_is_indeterminate_band_bracketing_kappa_star() -> None:
    """THE honest result: across plausible SPX (S0, OI) the dimensionless image of
    kappa_0 spans many orders of magnitude AND brackets kappa* = 17.81, so the
    dimensionless threshold alone cannot place the market 'near the Hopf'."""
    iv = 0.20
    # plausible extremes: small fixture (S0=100, OI=5e4) .. realistic SPX (S0=5000, OI~2e6)
    g_lo = flat_oi_characteristic_scale(spot=100.0, oi_per_cell=5e4, iv=iv)
    g_hi = flat_oi_characteristic_scale(spot=5000.0, oi_per_cell=2e6, iv=iv)
    band = dimensionless_kappa_band(KAPPA0, g_lo, g_hi, kappa_star=KAPPA_STAR)

    # The band is wide: several orders of magnitude of pure modelling freedom.
    assert band.orders_of_magnitude >= 3.0
    # kappa* sits inside it -> position relative to Hopf is undetermined pre-data.
    assert band.contains_kappa_star
    assert band.kappa_dim_lo < KAPPA_STAR < band.kappa_dim_hi


def test_naive_no_rescale_comparison_is_meaningless() -> None:
    """Sanity: comparing kappa_0 (5e-12) directly to kappa* (17.81) without rescaling
    is off by ~12 orders of magnitude — the exact incoherence we are fixing."""
    naive_gap = np.log10(KAPPA_STAR / KAPPA0)
    assert naive_gap > 11.0  # ~12.55 decades of spurious distance
