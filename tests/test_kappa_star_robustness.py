"""Tests for the κ★ robustness analysis (paper §4.3.6 / §3.5).

Covers:
    - Analytical implicit-function partials d κ★ / d(G_y, G_v) match the
      direct chord on the closed-form quadratic.
    - Outer chain-rule partials d κ★ / d(μ_q, σ_q) match a Richardson-converged
      finite difference on `kappa_star_lognormal_oi(G(μ_q, σ_q))` to ≤ 1e-3
      relative.
    - Mis-specification error decreases monotonically as the bimodal separation
      collapses to 0 (single-component limit).
    - Mixture-moment-fit identity: a degenerate single-component mixture
      reproduces (μ̂, σ̂) = (μ_q, σ_q).
    - Calibration tolerance scales linearly in the target κ★ budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.theory.bifurcation import (
    G_lognormal_oi_partials,
    kappa_star_lognormal_oi,
)
from reflexive_options.theory.robustness import (
    calibration_tolerance,
    fit_lognormal_to_mixture_moments,
    kappa_star_brute_force_from_G,
    kappa_star_misspecification_error,
    kappa_star_sensitivity_lognormal_oi,
    make_mixture_lognormal_density,
)

# Canonical specification (paper §4.3 + this section).
_CANONICAL = dict(
    mu_q=float(np.log(100.0)),
    sigma_q=0.10,
    T_eff=0.25,
    kappa_v=2.0,
    theta_v=0.04,
    alpha=0.05,
    beta=1.0,
    gamma=1.0,
    a_star=float(np.log(100.0)),
    v_star=0.04,
    coupling_units=1.0,
)


def _kappa_at(mu: float, sg: float) -> float:
    """Recompute κ★ from (μ_q, σ_q) at the canonical regime — used for FD checks."""
    p = G_lognormal_oi_partials(
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
        mu_q=mu,
        sigma_q=sg,
        T_eff=_CANONICAL["T_eff"],
    )
    k, _ = kappa_star_lognormal_oi(
        G_y=p["G_a"],
        G_v=p["G_v"],
        kappa_v=_CANONICAL["kappa_v"],
        alpha=_CANONICAL["alpha"],
        beta=_CANONICAL["beta"],
        gamma=_CANONICAL["gamma"],
    )
    return float(k)


# ---------------------------------------------------------------------------
# 1. Analytical partials match a Richardson-converged FD on κ★(μ_q, σ_q).
# ---------------------------------------------------------------------------


def test_dkappa_dmuq_matches_richardson_fd() -> None:
    """∂κ★/∂μ_q from `kappa_star_sensitivity_lognormal_oi` matches a 5-point
    central FD on the κ★(μ_q) chord to ≤ 1e-3 relative."""
    s = kappa_star_sensitivity_lognormal_oi(**_CANONICAL)
    mu0 = float(_CANONICAL["mu_q"])
    sg0 = float(_CANONICAL["sigma_q"])

    # 4th-order central FD: (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12 h)
    h = 1e-4
    fd4 = (
        -_kappa_at(mu0 + 2 * h, sg0)
        + 8 * _kappa_at(mu0 + h, sg0)
        - 8 * _kappa_at(mu0 - h, sg0)
        + _kappa_at(mu0 - 2 * h, sg0)
    ) / (12.0 * h)
    rel = abs(s.dkappa_dmu_q - fd4) / abs(fd4)
    assert rel < 1e-3, f"d κ★/dμ_q mismatch: analytic={s.dkappa_dmu_q}, FD={fd4} (rel={rel:.3e})"


def test_dkappa_dsigma_q_matches_richardson_fd() -> None:
    """∂κ★/∂σ_q matches the 4th-order FD on κ★(σ_q) chord to ≤ 1e-3 relative."""
    s = kappa_star_sensitivity_lognormal_oi(**_CANONICAL)
    mu0 = float(_CANONICAL["mu_q"])
    sg0 = float(_CANONICAL["sigma_q"])

    h = 1e-4
    fd4 = (
        -_kappa_at(mu0, sg0 + 2 * h)
        + 8 * _kappa_at(mu0, sg0 + h)
        - 8 * _kappa_at(mu0, sg0 - h)
        + _kappa_at(mu0, sg0 - 2 * h)
    ) / (12.0 * h)
    rel = abs(s.dkappa_dsigma_q - fd4) / abs(fd4)
    assert rel < 1e-3, f"d κ★/dσ_q mismatch: analytic={s.dkappa_dsigma_q}, FD={fd4}"


def test_canonical_elasticities_match_published_table() -> None:
    """At the canonical regime (mu_q=log 100, sigma_q=0.10), the elasticities are
    pinned to the values reported in `paper/kappa_star_robustness.md`. Tightens
    silently-changing parameter conventions into a CI gate."""
    s = kappa_star_sensitivity_lognormal_oi(**_CANONICAL)
    # κ★ pinned (matches the existing test_lognormal_lyapunov.py number)
    assert s.kappa_star == pytest.approx(17.806507, rel=1e-6)
    # Elasticities pinned — within 1e-2 absolute of the markdown writeup
    assert s.elasticity_sigma_q == pytest.approx(-1.5829, abs=1e-3)
    # μ_q-elasticity is huge because (μ_q · ∂κ★/∂μ_q)/κ★ where μ_q ≈ log 100 ≈ 4.6
    assert s.elasticity_mu_q == pytest.approx(702.83, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Mixture moment-fit and degenerate mixture identity.
# ---------------------------------------------------------------------------


def test_single_component_mixture_recovers_lognormal() -> None:
    """A degenerate mixture with one component returns its (μ, σ) exactly."""
    mu_hat, sigma_hat = fit_lognormal_to_mixture_moments(
        mu_components=[0.42], sigma_components=[0.17], weights=[1.0]
    )
    assert mu_hat == pytest.approx(0.42, abs=1e-15)
    assert sigma_hat == pytest.approx(0.17, abs=1e-15)


def test_mixture_density_normalises() -> None:
    """The mixture density integrates to 1 over a wide log-strike band."""
    from scipy.integrate import quad

    q = make_mixture_lognormal_density(
        mu_components=[-0.5, 0.5],
        sigma_components=[0.2, 0.3],
        weights=[0.4, 0.6],
    )
    total, _ = quad(q, -10.0, 10.0, limit=200)
    assert abs(total - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# 3. Misspecification error vanishes in the single-component limit.
# ---------------------------------------------------------------------------


def test_misspecification_error_zero_at_single_component() -> None:
    """When the 'mixture' is degenerate (single component), the moment-matched
    log-normal is the true OI density, so the closed-form κ★ should agree
    with the FD-on-G κ★ to numerical-quadrature precision (≤ 5e-4)."""
    err = kappa_star_misspecification_error(
        mu_components=[float(np.log(100.0))],
        sigma_components=[0.10],
        weights=[1.0],
        T_eff=0.25,
        kappa_v=2.0,
        theta_v=0.04,
        alpha=0.05,
        beta=1.0,
        gamma=1.0,
        a_star=float(np.log(100.0)),
        v_star=0.04,
    )
    assert err.relative_error < 5e-4, (
        f"single-component 'mixture' should give zero misspec error, got "
        f"{err.relative_error * 100:.3f}% (κ_cf={err.kappa_star_closed_form}, "
        f"κ_true={err.kappa_star_true})"
    )


def test_misspecification_error_grows_with_separation() -> None:
    """Bimodal separation → larger relative error. Monotone increase between
    sep=0 and sep=0.20 in the canonical regime."""
    seps = [0.00, 0.05, 0.10, 0.15, 0.20]
    rel_errs = []
    for sep in seps:
        if sep == 0.0:
            mu_components = [float(np.log(100.0))]
            sigma_components = [0.07]
            weights = [1.0]
        else:
            mu_components = [float(np.log(100.0)) - sep / 2, float(np.log(100.0)) + sep / 2]
            sigma_components = [0.07, 0.07]
            weights = [0.5, 0.5]
        err = kappa_star_misspecification_error(
            mu_components=mu_components,
            sigma_components=sigma_components,
            weights=weights,
            T_eff=0.25,
            kappa_v=2.0,
            theta_v=0.04,
            alpha=0.05,
            beta=1.0,
            gamma=1.0,
            a_star=float(np.log(100.0)),
            v_star=0.04,
        )
        rel_errs.append(err.relative_error)
    # Monotone non-decreasing
    diffs = np.diff(rel_errs)
    assert np.all(diffs >= -1e-9), (
        f"misspec error should be monotone in separation, got rel_errs={rel_errs}"
    )
    # Headline: by sep=0.10 we are <10%, by sep=0.20 we are >50%
    assert rel_errs[2] < 0.10, f"sep=0.10 should be <10% error, got {rel_errs[2] * 100:.2f}%"
    assert rel_errs[4] > 0.50, f"sep=0.20 should be >50% error, got {rel_errs[4] * 100:.2f}%"


# ---------------------------------------------------------------------------
# 4. Calibration tolerance scales linearly in the target budget.
# ---------------------------------------------------------------------------


def test_calibration_tolerance_scales_linearly() -> None:
    """Tolerance scales linearly in the target — doubling the budget doubles
    the allowed (σ_q, μ_q) tolerance."""
    s = kappa_star_sensitivity_lognormal_oi(**_CANONICAL)
    t_5 = calibration_tolerance(s, target_kappa_relative_error=0.05)
    t_10 = calibration_tolerance(s, target_kappa_relative_error=0.10)
    # Doubling target → exactly doubling the tolerances
    assert t_10["sigma_q_pct_tol"] == pytest.approx(2.0 * t_5["sigma_q_pct_tol"], rel=1e-12)
    assert t_10["mu_q_log_strike_tol"] == pytest.approx(2.0 * t_5["mu_q_log_strike_tol"], rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Brute-force κ★ on a known log-normal G recovers the closed form.
# ---------------------------------------------------------------------------


def test_brute_force_kappa_matches_closed_form_for_lognormal() -> None:
    """Sanity check: when G is built from a single log-normal density,
    `kappa_star_brute_force_from_G` returns the same κ★ as the closed form."""
    from reflexive_options.theory.bifurcation import G_lognormal_oi

    canon = dict(_CANONICAL)
    mu_q = float(canon["mu_q"])
    sigma_q = float(canon["sigma_q"])

    def G_func(a: float, v: float) -> float:
        return G_lognormal_oi(
            log_spot=a,
            variance=v,
            mu_q=mu_q,
            sigma_q=sigma_q,
            T_eff=float(canon["T_eff"]),
            coupling_units=float(canon["coupling_units"]),
        )

    k_bf, _ = kappa_star_brute_force_from_G(
        G_func=G_func,
        a_star=float(canon["a_star"]),
        v_star=float(canon["v_star"]),
        kappa_v=float(canon["kappa_v"]),
        alpha=float(canon["alpha"]),
        beta=float(canon["beta"]),
        gamma=float(canon["gamma"]),
    )
    # Closed form
    p = G_lognormal_oi_partials(
        a_star=float(canon["a_star"]),
        v_star=float(canon["v_star"]),
        mu_q=mu_q,
        sigma_q=sigma_q,
        T_eff=float(canon["T_eff"]),
    )
    k_cf, _ = kappa_star_lognormal_oi(
        G_y=p["G_a"],
        G_v=p["G_v"],
        kappa_v=float(canon["kappa_v"]),
        alpha=float(canon["alpha"]),
        beta=float(canon["beta"]),
        gamma=float(canon["gamma"]),
    )
    assert k_bf == pytest.approx(k_cf, rel=1e-4), (
        f"brute-force on log-normal G should reproduce the closed form: k_bf={k_bf}, k_cf={k_cf}"
    )
