"""Tests for the sliced-Wasserstein-2 metric on arbitrage-filtered windows.

Covers (per the task brief):
1. Closed-form 1D W2 hand-check.
2. Sliced approximation convergence to analytical W2 between Gaussians.
3. Symmetry: SW2(A, B) == SW2(B, A) under shared rng.
4. Identity at zero: SW2(A, A) ≈ 0 for finite N_slices.
5. Window builder shapes.
6. Arbitrage filter integration with hand-injected violations.
7. End-to-end Heston-vs-Heston (same params): distance is small.
8. End-to-end Heston-vs-Heston (shifted theta): ordering monotone in shift.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from reflexive_options.surface.generator import make_pre_reg_grid
from reflexive_options.surface.wasserstein import (
    SlicedW2Result,
    _flatten_windows,
    evaluate_sliced_w2_on_surface_windows,
    filter_arbitrage_free_windows,
    make_rolling_windows,
    sliced_wasserstein_2,
)
from reflexive_options.types import SurfaceGrid

# ---------------------------------------------------------------------------
# Test 1 — 1D closed-form sanity check
# ---------------------------------------------------------------------------


def test_sliced_w2_1d_closed_form() -> None:
    """In d=1 sliced-W2 == |1D W2| because θ ∈ {-1, +1}."""
    a = np.array([[1.0], [2.0], [3.0], [4.0]])
    b = np.array([[5.0], [6.0], [7.0], [8.0]])
    # 1D W2 between sorted (1,2,3,4) and (5,6,7,8) = sqrt(mean(16, 16, 16, 16)) = 4.0
    rng = np.random.default_rng(0)
    out = sliced_wasserstein_2(a, b, n_slices=100, rng=rng)
    assert out == pytest.approx(4.0, abs=1e-9)


def test_sliced_w2_1d_unequal_size() -> None:
    """Equal-quantile resampling for unequal sample sizes recovers the true 1D W2."""
    # Two empirical CDFs supported on [0, 1] vs [10, 11]: true W2 = 10.
    rng = np.random.default_rng(7)
    a = rng.uniform(0.0, 1.0, size=(200, 1))
    b = rng.uniform(10.0, 11.0, size=(73, 1))
    out = sliced_wasserstein_2(a, b, n_slices=200, rng=rng)
    assert out == pytest.approx(10.0, abs=0.05)


# ---------------------------------------------------------------------------
# Test 2 — sliced approximation convergence
# ---------------------------------------------------------------------------


def _analytical_sw2_between_shifted_gaussians(mean_diff: float, d: int) -> float:
    """Sliced-W2 between N(0, I_d) and N(μ·e_1, I_d).

    For each θ ∼ Unif(S^{d-1}) the projected pair is N(0, 1) vs N(μ·θ_1, 1)
    with 1D W2² = (μ·θ_1)². Averaging E[θ_1²] = 1/d gives SW2 = |μ|/√d.
    """
    return float(abs(mean_diff) / np.sqrt(d))


def test_sliced_w2_converges_to_truth() -> None:
    """SW2 converges to the analytical SW2 between two isotropic Gaussians as
    N_slices grows. We check that by 10k slices we are within ~5% of the target
    *given* the finite-sample MC noise from the n=1000 empirical draw.
    """
    d = 4
    n = 1000
    mu = 2.0
    sigma = 1.0
    rng = np.random.default_rng(123)
    x = rng.standard_normal((n, d)) * sigma
    y = rng.standard_normal((n, d)) * sigma
    y[:, 0] += mu

    target = _analytical_sw2_between_shifted_gaussians(mu, d)  # = 1.0 for d=4, μ=2
    estimates: list[float] = []
    for n_slices in (10, 100, 1000, 10000):
        est = sliced_wasserstein_2(x, y, n_slices=n_slices, rng=np.random.default_rng(0))
        estimates.append(est)
    # The N_slices=10000 estimate should be within 5% of the analytical SW2.
    assert estimates[-1] == pytest.approx(target, rel=0.05), (
        f"final SW2={estimates[-1]:.4f} not close to target SW2={target:.4f}; sequence: {estimates}"
    )
    # And the N_slices=1000 estimate within 10%.
    assert estimates[2] == pytest.approx(target, rel=0.10)
    # Variance of the estimator decreases with N_slices: |est_10000 - target| < |est_10 - target|.
    assert abs(estimates[-1] - target) <= abs(estimates[0] - target) + 1e-6


# ---------------------------------------------------------------------------
# Test 3 — symmetry
# ---------------------------------------------------------------------------


def test_sliced_w2_symmetric() -> None:
    """SW2(A, B) == SW2(B, A) when fed the same θ draws."""
    rng_a = np.random.default_rng(11)
    a = rng_a.standard_normal((40, 6))
    rng_b = np.random.default_rng(22)
    b = rng_b.standard_normal((40, 6)) + 1.0
    # Equal-size case is exactly symmetric (sorting is order-agnostic).
    rng_left = np.random.default_rng(0)
    rng_right = np.random.default_rng(0)
    sw_ab = sliced_wasserstein_2(a, b, n_slices=200, rng=rng_left)
    sw_ba = sliced_wasserstein_2(b, a, n_slices=200, rng=rng_right)
    assert sw_ab == pytest.approx(sw_ba, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 4 — identity at zero
# ---------------------------------------------------------------------------


def test_sliced_w2_identity() -> None:
    """SW2(A, A) is exactly zero — sorted projections are identical per slice."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 8))
    out = sliced_wasserstein_2(a, a, n_slices=10000, rng=np.random.default_rng(1))
    # Equal samples, equal sort order ⇒ identically zero (no MC noise).
    assert out == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 5 — rolling-window builder
# ---------------------------------------------------------------------------


def _synthetic_daily_series(n_days: int, n_K: int = 11, n_T: int = 7) -> np.ndarray:
    """Tile a flat 0.20 IV surface so the arbitrage filter is happy by default."""
    return np.full((n_days, n_K, n_T), 0.20, dtype=np.float64)


def test_make_rolling_windows_50_day_series() -> None:
    series = _synthetic_daily_series(50)
    windows = make_rolling_windows(series, window_length=21, stride=1)
    assert windows.shape == (30, 21, 11, 7)
    # Spot-check first / last window correspond to days [0..20] and [29..49].
    np.testing.assert_array_equal(windows[0], series[0:21])
    np.testing.assert_array_equal(windows[-1], series[29:50])


def test_make_rolling_windows_short_series_returns_empty() -> None:
    series = _synthetic_daily_series(10)
    windows = make_rolling_windows(series, window_length=21, stride=1)
    assert windows.shape == (0, 21, 11, 7)


def test_make_rolling_windows_stride_2() -> None:
    series = _synthetic_daily_series(50)
    windows = make_rolling_windows(series, window_length=21, stride=2)
    # n_windows = (50 - 21) // 2 + 1 = 15
    assert windows.shape == (15, 21, 11, 7)


def test_make_rolling_windows_rejects_bad_args() -> None:
    series = _synthetic_daily_series(50)
    with pytest.raises(ValueError):
        make_rolling_windows(series[0], window_length=21, stride=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_rolling_windows(series, window_length=0, stride=1)
    with pytest.raises(ValueError):
        make_rolling_windows(series, window_length=21, stride=0)


# ---------------------------------------------------------------------------
# Test 6 — arbitrage filter integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pre_reg_grid() -> SurfaceGrid:
    return make_pre_reg_grid()


def _butterfly_violator(grid: SurfaceGrid) -> np.ndarray:
    """Surface that fails the butterfly check at an interior strike."""
    iv = np.full(grid.shape, 0.20, dtype=np.float64)
    # Drop a single strike's IV by 5 vol points → the centred 2nd-difference
    # of the call price at the neighbouring strikes goes negative.
    iv[5, 3] -= 0.05
    return iv


def test_arbitrage_filter_drops_windows_containing_violator(pre_reg_grid: SurfaceGrid) -> None:
    n_days = 50
    series = _synthetic_daily_series(
        n_days, n_K=pre_reg_grid.n_strikes, n_T=pre_reg_grid.n_maturities
    )
    bad = _butterfly_violator(pre_reg_grid)
    series[5] = bad
    series[25] = bad

    windows = make_rolling_windows(series, window_length=21, stride=1)
    assert windows.shape[0] == 30
    kept, mask = filter_arbitrage_free_windows(
        windows, pre_reg_grid, spot=100.0, rate=0.0, dividend=0.0
    )
    # Day 5 is in windows starting at days 0..5 (since window covers [start, start+20]).
    # Window w covers days [w, w+20]. Day 5 ∈ window w iff w ≤ 5 ≤ w+20 ⇒ w ∈ [0, 5].
    # Day 25 ∈ window w iff w ∈ [5, 25]. Union: w ∈ [0, 25] = 26 windows dropped.
    # Surviving windows: w ∈ [26, 29] = 4.
    expected_drops = set(range(0, 26))
    rejected_idx = set(np.where(~mask)[0].tolist())
    assert rejected_idx == expected_drops
    assert kept.shape == (30 - 26, 21, 11, 7)


def test_filter_rejects_bad_dim() -> None:
    bogus = np.zeros((3, 11, 7))  # 3D, not 4D
    grid = make_pre_reg_grid()
    with pytest.raises(ValueError):
        filter_arbitrage_free_windows(bogus, grid, spot=100.0, rate=0.0, dividend=0.0)  # type: ignore[arg-type]


def test_filter_handles_empty() -> None:
    empty = np.zeros((0, 21, 11, 7))
    grid = make_pre_reg_grid()
    kept, mask = filter_arbitrage_free_windows(empty, grid, spot=100.0, rate=0.0, dividend=0.0)
    assert kept.shape == (0, 21, 11, 7)
    assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# End-to-end on a Heston-like synthetic surface generator
#
# We use a self-rolled OU-driven IV-surface generator (instead of QL-Heston) to
# keep the test deterministic, fast, and free of QL inverse-IV NaN issues at
# the extreme strikes of the pre-reg grid. The surface is arbitrage-free by
# construction (smile is a quadratic in k with positive curvature; total
# variance is monotone in T because ATM IV is bounded; Lee bounds satisfied at
# the locked Δk = 0.04).
# ---------------------------------------------------------------------------


def _heston_like_surface(
    grid: SurfaceGrid,
    *,
    atm_iv: float,
    skew: float,
    smile_curvature: float,
) -> np.ndarray:
    """Quadratic smile in k, mild term-structure flattening in T.

    σ(k, τ) = atm_iv * (1 + smile_curvature * k² + skew * k) * (1 - 0.05 * (τ - 0.5))

    Stays comfortably arbitrage-free for atm_iv ∈ (0.1, 0.4), curvature ≤ 1.5,
    |skew| ≤ 0.3 on the (Δk = 0.04, k ∈ [-0.2, 0.2]) × (7d…1y) pre-reg grid.
    """
    k = grid.log_moneyness[:, None]
    T = grid.maturities[None, :]
    smile = 1.0 + smile_curvature * (k**2) + skew * k
    term = 1.0 - 0.05 * (T - 0.5)
    return atm_iv * smile * term


def _heston_like_daily_series(
    n_days: int,
    grid: SurfaceGrid,
    *,
    seed: int,
    atm_mean: float = 0.20,
    atm_drift: float = 0.0,
) -> np.ndarray:
    """OU process on ATM IV, deterministic skew/curvature."""
    rng = np.random.default_rng(seed)
    atm = np.empty(n_days, dtype=np.float64)
    atm[0] = atm_mean
    kappa_ou = 5.0 / 252.0  # mean-reversion per day
    sigma_ou = 0.005
    for d in range(1, n_days):
        atm[d] = (
            atm[d - 1]
            + kappa_ou * (atm_mean + atm_drift - atm[d - 1])
            + sigma_ou * rng.standard_normal()
        )
        atm[d] = float(np.clip(atm[d], 0.05, 0.60))
    surfaces = np.empty((n_days, grid.n_strikes, grid.n_maturities), dtype=np.float64)
    for d in range(n_days):
        surfaces[d] = _heston_like_surface(grid, atm_iv=atm[d], skew=-0.4, smile_curvature=1.0)
    return surfaces


@pytest.mark.slow
def test_e2e_same_model_small_distance(pre_reg_grid: SurfaceGrid) -> None:
    """Two seeds of the same OU-IV process produce a small SW2 distance."""
    n_days = 100
    surf_a = _heston_like_daily_series(n_days, pre_reg_grid, seed=11)
    surf_b = _heston_like_daily_series(n_days, pre_reg_grid, seed=22)

    res = evaluate_sliced_w2_on_surface_windows(
        surf_a,
        surf_b,
        pre_reg_grid,
        spot=100.0,
        rate=0.0,
        dividend=0.0,
        n_slices=300,
        rng=np.random.default_rng(0),
    )
    assert isinstance(res, SlicedW2Result)
    # Most windows survive — the synthetic surface is built to be arbitrage-free.
    assert res.n_windows_left >= 60
    assert res.n_windows_right >= 60
    assert np.isfinite(res.distance)
    # Same model, two seeds — distance should be tiny relative to the IV scale.
    assert res.distance < 0.10, f"unexpectedly large distance for same-model pair: {res.distance}"


@pytest.mark.slow
def test_e2e_shifted_drift_increases_distance(pre_reg_grid: SurfaceGrid) -> None:
    """Larger ATM-IV drift ⇒ larger SW2 from the unshifted reference."""
    n_days = 100
    surf_ref = _heston_like_daily_series(n_days, pre_reg_grid, seed=33)
    surf_small = _heston_like_daily_series(n_days, pre_reg_grid, seed=44, atm_drift=0.01)
    surf_large = _heston_like_daily_series(n_days, pre_reg_grid, seed=44, atm_drift=0.10)

    res_small = evaluate_sliced_w2_on_surface_windows(
        surf_ref,
        surf_small,
        pre_reg_grid,
        spot=100.0,
        n_slices=300,
        rng=np.random.default_rng(0),
    )
    res_large = evaluate_sliced_w2_on_surface_windows(
        surf_ref,
        surf_large,
        pre_reg_grid,
        spot=100.0,
        n_slices=300,
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(res_small.distance)
    assert np.isfinite(res_large.distance)
    assert res_large.distance > res_small.distance, (
        f"expected SW2 ordering: small_shift {res_small.distance:.4f} "
        f"< large_shift {res_large.distance:.4f}"
    )


# ---------------------------------------------------------------------------
# Other coverage checks
# ---------------------------------------------------------------------------


def test_evaluate_returns_nan_with_warning_on_too_few_windows(pre_reg_grid: SurfaceGrid) -> None:
    """< 30 surviving windows ⇒ NaN distance + warning."""
    n_days = 25  # only 5 raw windows
    series = _synthetic_daily_series(
        n_days, n_K=pre_reg_grid.n_strikes, n_T=pre_reg_grid.n_maturities
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = evaluate_sliced_w2_on_surface_windows(
            series, series, pre_reg_grid, spot=100.0, n_slices=10
        )
    assert np.isnan(res.distance)
    assert any("insufficient surviving windows" in str(item.message) for item in w)
    assert res.n_windows_left < 30
    assert res.n_windows_right < 30


def test_evaluate_reports_rejection_fraction(pre_reg_grid: SurfaceGrid) -> None:
    """End-to-end pipeline reports the arbitrage rejection fraction faithfully."""
    n_days = 80
    series_clean = _synthetic_daily_series(
        n_days, n_K=pre_reg_grid.n_strikes, n_T=pre_reg_grid.n_maturities
    )
    bad = _butterfly_violator(pre_reg_grid)
    series_dirty = series_clean.copy()
    series_dirty[40] = bad  # corrupts windows w ∈ [20, 40] = 21 of 60 windows

    res = evaluate_sliced_w2_on_surface_windows(
        series_clean,
        series_dirty,
        pre_reg_grid,
        spot=100.0,
        n_slices=20,
        rng=np.random.default_rng(0),
    )
    assert res.rejected_left_frac == pytest.approx(0.0, abs=1e-12)
    assert res.rejected_right_frac > 0.0
    expected_dirty_drops = 21 / 60
    assert res.rejected_right_frac == pytest.approx(expected_dirty_drops, abs=1e-12)


def test_sliced_w2_rejects_dim_mismatch() -> None:
    a = np.zeros((10, 4))
    b = np.zeros((10, 5))
    with pytest.raises(ValueError):
        sliced_wasserstein_2(a, b)


def test_sliced_w2_rejects_bad_n_slices() -> None:
    a = np.zeros((10, 4))
    with pytest.raises(ValueError):
        sliced_wasserstein_2(a, a, n_slices=0)


def test_sliced_w2_rejects_non_2d() -> None:
    a = np.zeros((10,))
    b = np.zeros((10, 4))
    with pytest.raises(ValueError):
        sliced_wasserstein_2(a, b)  # type: ignore[arg-type]


def test_sliced_w2_returns_nan_on_empty() -> None:
    a = np.zeros((0, 4))
    b = np.zeros((10, 4))
    out = sliced_wasserstein_2(a, b, n_slices=5)
    assert np.isnan(out)


def test_make_pre_reg_grid_matches_spec() -> None:
    grid = make_pre_reg_grid()
    assert grid.shape == (11, 7)
    np.testing.assert_allclose(
        grid.maturities,
        np.array([7, 14, 30, 60, 90, 180, 365], dtype=np.float64) / 365.0,
    )
    np.testing.assert_allclose(grid.log_moneyness[0], -0.20, atol=1e-12)
    np.testing.assert_allclose(grid.log_moneyness[-1], 0.20, atol=1e-12)
    np.testing.assert_allclose(np.diff(grid.log_moneyness), 0.04, atol=1e-12)


def test_flatten_windows_round_trip() -> None:
    rng = np.random.default_rng(0)
    win = rng.standard_normal((4, 21, 11, 7))
    flat = _flatten_windows(win)
    assert flat.shape == (4, 21 * 11 * 7)
    # Verify C-order: last axis varies fastest.
    np.testing.assert_array_equal(flat[0, :7], win[0, 0, 0, :])
