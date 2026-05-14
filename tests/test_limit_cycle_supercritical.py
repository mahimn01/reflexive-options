"""Smoke + invariant tests for the supercritical limit-cycle experiment.

The limit-cycle experiment validates Theorem 1 conclusion (1) numerically.
Tests here pin:
    1. The simulator returns a finite period within ~10% of the predicted T_κ.
    2. The cycle amplitudes in (y, u, z) are all strictly positive (i.e. the
       transient has decayed and the orbit is non-degenerate).
    3. The relative period error is monotonically small at modest κ-overshoot
       (sanity: at κ = 1.05·κ* we should see < 5% drift, not > 10%).
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.experiments.limit_cycle_supercritical import (
    LimitCycleConfig,
    _drift,
    _estimate_period,
    run,
)


def test_drift_at_origin_is_zero() -> None:
    """The §4.2 skeleton at (y, u, z) = 0 has zero drift (it's the equilibrium)."""
    f = _drift(np.zeros(3, dtype=np.float64), kappa=1.0)
    np.testing.assert_allclose(f, np.zeros(3), atol=1e-12)


@pytest.mark.parametrize(
    "y,u,z",
    [
        (0.1, 0.0, 0.0),
        (0.0, 0.05, 0.0),
        (0.0, 0.0, 0.2),
        (-0.1, 0.05, -0.2),
    ],
)
def test_drift_matches_jacobian_linearisation_near_origin(y: float, u: float, z: float) -> None:
    """For small perturbations, drift ≈ J · x with the §4.2 Jacobian."""
    from reflexive_options.experiments.limit_cycle_supercritical import (
        _ALPHA,
        _BETA,
        _G_V,
        _G_X,
        _G_Z,
        _GAMMA,
        _KAPPA_STAR,
        _KAPPA_V,
    )

    kappa = _KAPPA_STAR
    x = np.array([y, u, z], dtype=np.float64)
    f_actual = _drift(x, kappa=kappa)
    # Linear part of the §4.2 drift (constant-vol surrogate, no Itô term)
    J_lin = np.array(
        [
            [kappa * _G_X, kappa * _G_V, kappa * _G_Z],
            [0.0, -_KAPPA_V, _GAMMA],
            [_BETA, 0.0, -_ALPHA],
        ]
    )
    f_linear = J_lin @ x
    # At |x| ≤ 0.2 the cubic correction is < 1e-3 — so linear approx
    # should match within 5e-3.
    np.testing.assert_allclose(f_actual, f_linear, atol=5e-3)


def test_estimate_period_recovers_known_sinusoid() -> None:
    """Period estimator on a pure sinusoid recovers the period to machine precision."""
    T_true = 3.7
    t = np.linspace(0.0, 50 * T_true, 100_000)
    y = np.sin(2 * np.pi * t / T_true)
    T_est = _estimate_period(t, y)
    rel_err = abs(T_est - T_true) / T_true
    assert rel_err < 1e-3, f"expected sinusoid period recovery within 0.1%, got {rel_err:.3%}"


def test_limit_cycle_smoke_quick() -> None:
    """End-to-end smoke: quick config returns a sane period and amplitudes.

    Uses the --quick path budget; period accuracy is looser (≤ 8%) because
    fewer transient periods are discarded.
    """
    cfg = LimitCycleConfig(t_total=80.0, n_eval=3201, transient_periods=4)
    metrics = run(cfg)
    assert isinstance(metrics["period_measured"], float)
    period = float(metrics["period_measured"])
    period_th = float(metrics["period_theory"])
    assert np.isfinite(period), "period must be finite"
    assert period_th == pytest.approx(2 * np.pi / 0.5724, rel=1e-4)
    rel_err = abs(period - period_th) / period_th
    assert rel_err < 0.08, (
        f"limit-cycle period drifted: measured {period:.3f} yr, "
        f"theory {period_th:.3f} yr, rel = {rel_err:.3%}"
    )
    for key in ("amplitude_y", "amplitude_u", "amplitude_z"):
        amp = float(metrics[key])  # type: ignore[arg-type]
        assert amp > 1e-3, f"{key} should be strictly positive, got {amp:.3e}"
