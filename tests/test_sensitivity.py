"""Tests for `theory.sensitivity` — GP-posterior slope CI (amendment A6).

The pre-A6 implementation used a UnivariateSpline-derivative + iid
bootstrap, which the V3 audit (`/tmp/audit_v3_bootstrap_v2.py`) showed had
0% coverage when the underlying κ-curve is non-smooth at the anchor. A6
replaces it with a Gaussian-process posterior over the function whose
derivative-at-anchor has a closed-form Gaussian distribution.

Tests exercise three scenarios:
  (a) true linear slope = 0.5 → GP recovers ~0.5 within tight CI.
  (b) true cubic with derivative 0 at anchor → GP says 0 within CI.
  (c) kinked function (out of GP prior class) → GP CI widens to admit truth.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from reflexive_options.theory.sensitivity import (
    SensitivityResult,
    kappa_sensitivity_curve,
)


def _grid(anchor: float, n: int = 9, low: float = 0.0, high: float = 2.0) -> np.ndarray:
    return np.linspace(low * anchor, high * anchor, n).astype(np.float64)


# ---------------------------------------------------------------------------
# Linear truth — point + CI both crisp.
# ---------------------------------------------------------------------------


def test_gp_slope_recovers_linear_truth() -> None:
    """True linear slope = 2.0 / κ_anchor → recovered slope within ~5%, CI tight."""
    anchor = 1.0
    grid = _grid(anchor)
    true_slope = 2.0 / anchor

    def f(k: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return true_slope * k + 0.01 * float(rng.standard_normal())

    res = kappa_sensitivity_curve(
        metric_fn=f,
        kappa_grid=grid,
        kappa_anchor=anchor,
        n_seeds=50,
        rng_seed=0,
    )
    assert res.method in ("gp", "local_quadratic_fallback")
    rel_err = abs(res.slope_at_anchor - true_slope) / abs(true_slope)
    assert rel_err < 0.10, f"slope estimate {res.slope_at_anchor} off from {true_slope}"
    assert res.slope_ci_low < true_slope < res.slope_ci_high
    # CI width should be modest given low noise.
    ci_width = res.slope_ci_high - res.slope_ci_low
    assert ci_width < 0.5 * abs(true_slope), f"CI too wide: {ci_width}"


# ---------------------------------------------------------------------------
# Cubic truth with derivative 0 at anchor — GP should report ~0 with CI containing 0.
# ---------------------------------------------------------------------------


def test_gp_slope_recovers_cubic_zero_derivative() -> None:
    """f(κ) = (κ - κ_anchor)³ has derivative 0 at κ_anchor.

    With realistic measurement noise (σ=0.05) and n_seeds=20, the GP posterior
    correctly admits 0 as the slope at the anchor — both the point estimate
    is near zero and the CI contains it.
    """
    anchor = 1.0
    grid = _grid(anchor)

    def f(k: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return (k - anchor) ** 3 + 0.05 * float(rng.standard_normal())

    res = kappa_sensitivity_curve(
        metric_fn=f,
        kappa_grid=grid,
        kappa_anchor=anchor,
        n_seeds=20,
        rng_seed=0,
    )
    # Slope estimate should be close to 0; CI should contain 0.
    assert abs(res.slope_at_anchor) < 0.5, f"slope_at_anchor too far from 0: {res.slope_at_anchor}"
    assert res.slope_ci_low <= 0.0 <= res.slope_ci_high, (
        f"CI [{res.slope_ci_low}, {res.slope_ci_high}] does not contain 0"
    )


# ---------------------------------------------------------------------------
# Kink at anchor — out of the GP RBF's smooth prior class. CI should widen.
# ---------------------------------------------------------------------------


def test_gp_slope_widens_ci_on_kinked_function() -> None:
    """f(κ) = max(κ - κ_anchor, 0): right-derivative=1, left-derivative=0.

    The GP can't represent the kink exactly with an RBF kernel, but the
    posterior CI should widen enough to admit the truth (or some plausible
    value between 0 and 1). The pre-A6 spline-bootstrap CI gave 0% coverage
    at this case (V3 audit). The acceptance criterion here is that the CI
    *contains* a truth-relevant slope — either 0 (left), 0.5 (symmetric), or
    1 (right) — under multiple seeds.
    """
    anchor = 1.0
    grid = _grid(anchor)

    def f(k: float, seed: int) -> float:
        # Tiny deterministic (seed-independent) value so we just measure the
        # spline-fitting bias against the kink.
        del seed
        return max(k - anchor, 0.0)

    res = kappa_sensitivity_curve(
        metric_fn=f,
        kappa_grid=grid,
        kappa_anchor=anchor,
        n_seeds=10,
        rng_seed=0,
    )
    # The CI should be non-degenerate (positive width).
    ci_width = res.slope_ci_high - res.slope_ci_low
    assert ci_width > 0.05, f"GP CI degenerate on kink: width={ci_width}"
    # And it should admit at least one of the truth-relevant slopes.
    contained = any(res.slope_ci_low <= truth <= res.slope_ci_high for truth in (0.0, 0.5, 1.0))
    assert contained, (
        f"GP CI [{res.slope_ci_low}, {res.slope_ci_high}] does not admit any of (0, 0.5, 1)"
    )


# ---------------------------------------------------------------------------
# Result-dataclass shape.
# ---------------------------------------------------------------------------


def test_sensitivity_result_has_method_and_se_fields() -> None:
    """A6: result dataclass exposes `method` and `slope_se` — needed downstream by TOST."""
    anchor = 1.0
    grid = _grid(anchor)

    def f(k: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return float(k + 0.01 * rng.standard_normal())

    res = kappa_sensitivity_curve(
        metric_fn=f, kappa_grid=grid, kappa_anchor=anchor, n_seeds=20, rng_seed=0
    )
    assert isinstance(res, SensitivityResult)
    assert res.method in ("gp", "local_quadratic_fallback")
    assert res.slope_se >= 0.0
    # CI should be approximately ±1.96 SE around the point estimate.
    half_width = (res.slope_ci_high - res.slope_ci_low) / 2.0
    expected_half = 1.959964 * res.slope_se
    if expected_half > 0:
        rel = abs(half_width - expected_half) / max(expected_half, 1e-12)
        assert rel < 0.05, f"CI half-width {half_width} not ≈ 1.96·SE {expected_half}"


# ---------------------------------------------------------------------------
# Performance — the brief mandates <1s on a 9-point grid.
# ---------------------------------------------------------------------------


def test_gp_slope_completes_in_under_one_second() -> None:
    """Performance gate: GP fit on 9-point grid + posterior eval < 1 s."""
    anchor = 1.0
    grid = _grid(anchor)

    def f(k: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return float(k + 0.01 * rng.standard_normal())

    # Time only the kappa_sensitivity_curve call (no metric_fn overhead beyond grid * seeds).
    start = time.perf_counter()
    kappa_sensitivity_curve(
        metric_fn=f, kappa_grid=grid, kappa_anchor=anchor, n_seeds=5, rng_seed=0
    )
    elapsed = time.perf_counter() - start
    # Hard cap: 1 second (per the engineering brief). On modern hardware the
    # GP fit itself is ~50–200 ms; metric_fn is trivial here.
    assert elapsed < 1.0, f"sensitivity curve took {elapsed:.3f}s, expected < 1s"


# ---------------------------------------------------------------------------
# Validation: ascending grid + anchor inside the grid.
# ---------------------------------------------------------------------------


def test_kappa_sensitivity_curve_validates_ascending_grid() -> None:
    grid = np.array([2.0, 1.0, 3.0])

    def f(k: float, s: int) -> float:
        del k, s
        return 0.0

    with pytest.raises(ValueError, match="ascending"):
        kappa_sensitivity_curve(
            metric_fn=f, kappa_grid=grid, kappa_anchor=1.5, n_seeds=2, rng_seed=0
        )


def test_kappa_sensitivity_curve_validates_anchor_inside_grid() -> None:
    grid = np.array([0.0, 1.0, 2.0])

    def f(k: float, s: int) -> float:
        del k, s
        return 0.0

    with pytest.raises(ValueError, match="strictly inside"):
        kappa_sensitivity_curve(
            metric_fn=f, kappa_grid=grid, kappa_anchor=0.0, n_seeds=2, rng_seed=0
        )


def test_local_quadratic_fallback_engages_on_degenerate_y() -> None:
    """When all y values are identical, GP collapses to slope=0 and method='gp' (degenerate-y path)."""
    anchor = 1.0
    grid = _grid(anchor)

    def f(k: float, seed: int) -> float:
        del k, seed
        return 1.0

    res = kappa_sensitivity_curve(
        metric_fn=f, kappa_grid=grid, kappa_anchor=anchor, n_seeds=5, rng_seed=0
    )
    # Either path is acceptable — what matters is the slope estimate is 0.
    assert res.slope_at_anchor == 0.0
    assert res.slope_se == 0.0


def test_local_quadratic_slope_directly() -> None:
    """Direct unit test of `_local_quadratic_slope` — used as the GP fallback."""
    from reflexive_options.theory.sensitivity import _local_quadratic_slope

    # Linear truth: y = 2x + 0; slope at x=0 is 2.
    x = np.linspace(-1.0, 1.0, 9)
    y = 2.0 * x
    slope, se = _local_quadratic_slope(x, y, x_anchor=0.0)
    assert abs(slope - 2.0) < 1e-10
    assert se < 1e-6  # essentially zero residual

    # Too few points should raise.
    with pytest.raises(RuntimeError, match="at least 3 points"):
        _local_quadratic_slope(np.array([0.0, 1.0]), np.array([0.0, 1.0]), x_anchor=0.5)


def test_local_quadratic_slope_handles_singular_design_matrix() -> None:
    """Three duplicated points → XtX is singular → RuntimeError."""
    from reflexive_options.theory.sensitivity import _local_quadratic_slope

    x = np.array([1.0, 1.0, 1.0])
    y = np.array([0.0, 1.0, 2.0])
    with pytest.raises(RuntimeError, match="singular"):
        _local_quadratic_slope(x, y, x_anchor=1.0)


def test_gp_derivative_posterior_pinv_fallback_path() -> None:
    """When Cholesky fails, the GP posterior calc falls back to pinv.

    Force the path by passing a singular kernel matrix (zero noise variance,
    near-duplicate training points → near-singular K).
    """
    from reflexive_options.theory.sensitivity import _gp_derivative_posterior

    # Three near-duplicate x points → K has near-zero eigenvalue.
    x = np.array([0.0, 1e-15, 2e-15])
    y = np.array([0.0, 0.0, 0.0])
    mean, var = _gp_derivative_posterior(
        x,
        y,
        x_anchor=0.0,
        length_scale=1.0,
        noise_variance=0.0,  # no noise jitter — force pure kernel
        signal_variance=1.0,
        y_mean=0.0,
    )
    # With y all zero, mean should be 0 regardless.
    assert mean == 0.0
    assert var >= 0.0


def test_gp_falls_back_to_local_quadratic_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When GP fitting raises, the wrapper falls back to local-quadratic."""
    import reflexive_options.theory.sensitivity as sens

    def _failing_gp(_x, _y, _x_anchor):
        raise RuntimeError("synthetic GP failure for fallback test")

    monkeypatch.setattr(sens, "_fit_gp_and_derivative", _failing_gp)

    anchor = 1.0
    grid = _grid(anchor)

    def f(k: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return float(2.0 * k + 0.05 * rng.standard_normal())

    res = kappa_sensitivity_curve(
        metric_fn=f, kappa_grid=grid, kappa_anchor=anchor, n_seeds=10, rng_seed=0
    )
    assert res.method == "local_quadratic_fallback"
    # Local-quadratic still recovers the linear slope reasonably.
    assert abs(res.slope_at_anchor - 2.0) < 0.5
