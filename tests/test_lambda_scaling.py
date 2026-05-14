"""Smoke + invariant tests for the empirical Λ-vs-(ρξ) scaling experiment.

The lambda-scaling experiment fits log|Λ| = log|A| + B · log|ρξ| over a
6×6 (ξ, ρ) grid at the §4.2 canonical regime. Tests here pin:
    1. The OLS fit produces finite (A, B) with a meaningful CI.
    2. The per-cell Λ values are finite and within order-of-magnitude bounds.
    3. The bootstrap CI brackets the OLS estimate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reflexive_options.experiments.lambda_scaling import (
    CellResult,
    LambdaScalingConfig,
    _fit_power_law,
    _ols_fit,
    _scan_cells,
)


def test_ols_fit_recovers_synthetic_power_law() -> None:
    """Fit recovers (log|A|, B) for a noiseless synthetic Λ ~ A·|ρξ|^B."""
    rng = np.random.default_rng(20260514)
    log_x = rng.uniform(-3.5, -0.5, size=20)
    log_A_true = -1.0
    B_true = 0.6
    log_y = log_A_true + B_true * log_x
    a, b, resid_std = _ols_fit(log_x, log_y)
    assert abs(a - log_A_true) < 1e-10
    assert abs(b - B_true) < 1e-10
    assert resid_std < 1e-10


def test_ols_fit_with_noise_recovers_within_se() -> None:
    """With Gaussian noise σ_y = 0.1, fit recovers parameters within ~0.05."""
    rng = np.random.default_rng(99)
    n = 100
    log_x = rng.uniform(-3.0, 0.0, size=n)
    log_A_true = -2.0
    B_true = 0.667
    log_y = log_A_true + B_true * log_x + rng.normal(0.0, 0.1, size=n)
    a, b, _ = _ols_fit(log_x, log_y)
    assert abs(a - log_A_true) < 0.10
    assert abs(b - B_true) < 0.05


def test_fit_power_law_bootstrap_brackets_point_estimate() -> None:
    """Bootstrap CI for B contains the OLS point estimate."""
    rng = np.random.default_rng(7)
    cells = []
    log_A_true = -2.5
    B_true = 0.3
    for _ in range(40):
        rho = rng.choice([-0.7, -0.3, 0.3, 0.7])
        xi = rng.choice([0.1, 0.3, 0.5, 1.0])
        rho_xi = rho * xi
        log_lam = log_A_true + B_true * math.log(abs(rho_xi)) + rng.normal(0.0, 0.15)
        lam = math.exp(log_lam) * (1.0 if rng.random() > 0.5 else -1.0)
        cells.append(
            CellResult(
                xi=float(xi),
                rho=float(rho),
                rho_xi=float(rho_xi),
                lambda_value=float(lam),
                abs_lambda=float(abs(lam)),
                log_abs_rho_xi=float(math.log(abs(rho_xi))),
                log_abs_lambda=float(math.log(abs(lam))),
            )
        )
    fit = _fit_power_law(
        cells,
        n_bootstrap=200,
        bootstrap_seed=42,
        predicted_exponent=2.0 / 3.0,
    )
    assert fit.B_ci_low <= fit.B <= fit.B_ci_high
    assert fit.log_A_ci_low <= fit.log_A <= fit.log_A_ci_high
    # B should be near 0.3 within bootstrap noise
    assert abs(fit.B - B_true) < 0.15


@pytest.mark.slow
def test_scan_cells_quick_smoke() -> None:
    """End-to-end smoke at very low fidelity — just check we get finite Λ values."""
    cfg = LambdaScalingConfig(
        xi_grid=(0.1, 0.5),
        rho_grid=(-0.7, 0.3),
        n_paths=200,
        n_steps=1_000,
    )
    cells = _scan_cells(cfg)
    assert len(cells) == 4
    for c in cells:
        assert math.isfinite(c.lambda_value), f"non-finite Λ at (ξ={c.xi}, ρ={c.rho})"
        assert -1.0 <= c.lambda_value <= 1.0, (
            f"|Λ| = {abs(c.lambda_value):.3e} outside [0, 1] at (ξ={c.xi}, ρ={c.rho})"
        )


def test_lambda_scaling_config_defaults_sane() -> None:
    """Default config has the right §4.2 skeleton parameters and grid sizes."""
    cfg = LambdaScalingConfig()
    assert cfg.kappa_v == 2.0
    assert cfg.theta_v == 0.04
    assert cfg.alpha == 0.5
    assert cfg.beta == 1.0
    assert cfg.gamma == 0.5
    assert cfg.coupling_at_kappa_star == pytest.approx(0.8964, abs=1e-4)
    assert len(cfg.xi_grid) == 6
    assert len(cfg.rho_grid) == 6
    assert cfg.predicted_exponent == pytest.approx(2.0 / 3.0)
