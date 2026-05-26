"""Tests for the McKean-Vlasov mean-field limit (paper §4 / Appendix B).

Covers:
    - Closed-form $C(T)$ propagation-of-chaos constant.
    - 4D Hopf-threshold correction (v0.3.6): exact match to canonical closed
      form, regression pins at the auditor's 5-point table, infinite-theta_G
      limit recovers the single-dealer threshold.
    - Particle-system smoke test: simulator runs and produces finite output.
    - $1/\\sqrt n$ scaling: fitted log-log slope close to 1 within 30%.
    - Lipschitz / input-validation: rejects non-positive theta_G, sigma_G,
      n_particles, T, n_steps; rejects non-monotonic n_grid.
    - Mean-field limit ODE matches stationary value at long horizons.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.theory.mckean_vlasov import (
    KAPPA_STAR_SINGLE_CANONICAL,
    mckean_vlasov_jacobian_4d,
    mckean_vlasov_kappa_star,
    mckean_vlasov_kappa_star_canonical_closed_form,
    mckean_vlasov_kappa_star_shift,
    mean_field_limit_trajectory,
    propagation_of_chaos_constant,
    propagation_of_chaos_error,
    propagation_of_chaos_scaling,
    simulate_n_dealer_system,
)

# Auditor's canonical regime — short-gamma (G_y > 0) parameters at which the
# v0.3.5 heuristic formula was wrong in sign.
CANONICAL_REGIME = dict(
    G_y=0.5,
    G_v=-0.5,
    G_z=-0.5,
    alpha=0.5,
    beta=1.0,
    gamma=0.5,
    kappa_v=2.0,
    sigma2_y=0.0,
    sigma2_v=0.0,
)

# ---------------------------------------------------------------------------
# Closed-form helpers.
# ---------------------------------------------------------------------------


def test_propagation_of_chaos_constant_stationary_regime() -> None:
    """C(T) recovers the stationary OU variance when Var(G_0) = 0 and T -> infty.

    Stationary variance of an OU process is $\\sigma^2 / (2 \\theta)$.  At
    Var(G_0) = 0 and large $T$, the term $(1 - e^{-2 \\theta T})$ saturates
    at 1 so the bound matches the stationary variance.
    """
    theta_G = 10.0
    sigma_G = 0.2
    expected_stationary = sigma_G * sigma_G / (2.0 * theta_G)
    C_T = propagation_of_chaos_constant(
        theta_G=theta_G,
        sigma_G=sigma_G,
        var_G0=0.0,
        T=10.0,  # >> 1/theta_G = 0.1
    )
    assert pytest.approx(expected_stationary, rel=1e-6) == C_T


def test_propagation_of_chaos_constant_initial_dominates() -> None:
    """When Var(G_0) >> stationary, the bound saturates at Var(G_0)."""
    theta_G = 10.0
    sigma_G = 0.0  # no diffusion at all
    var_G0 = 0.5
    C_T = propagation_of_chaos_constant(
        theta_G=theta_G,
        sigma_G=sigma_G,
        var_G0=var_G0,
        T=0.1,
    )
    assert pytest.approx(var_G0, rel=1e-6) == C_T


def test_propagation_of_chaos_constant_rejects_invalid_args() -> None:
    """Non-positive theta_G, sigma_G < 0, var_G0 < 0, T <= 0 are rejected."""
    with pytest.raises(ValueError, match="theta_G"):
        propagation_of_chaos_constant(theta_G=0.0, sigma_G=0.1, var_G0=0.0, T=1.0)
    with pytest.raises(ValueError, match="sigma_G"):
        propagation_of_chaos_constant(theta_G=1.0, sigma_G=-0.1, var_G0=0.0, T=1.0)
    with pytest.raises(ValueError, match="var_G0"):
        propagation_of_chaos_constant(theta_G=1.0, sigma_G=0.1, var_G0=-1.0, T=1.0)
    with pytest.raises(ValueError, match="T"):
        propagation_of_chaos_constant(theta_G=1.0, sigma_G=0.1, var_G0=0.0, T=0.0)


# ---------------------------------------------------------------------------
# Hopf-threshold shift (CORRECTED v0.3.6 — 4D extended Jacobian).
#
# The v0.3.5 closed-form sqrt(1 + (omega/theta_G)^2) was incorrect:
# the auditor's numerical 4D Hopf computation showed the ratio is < 1
# (destabilising) at the canonical short-gamma regime, not > 1.
# See paper/mv_hopf_corrected.md.
# ---------------------------------------------------------------------------


def test_kappa_star_canonical_closed_form_matches_numerical() -> None:
    """The closed-form $\\kappa^\\star_\\mathrm{MV}(\\theta_G)$ at the canonical regime
    agrees with the 4D numerical Hopf solver to machine precision."""
    for theta_G in [0.1, 0.5, 1.0, 5.0, 50.0, 500.0]:
        k_closed = mckean_vlasov_kappa_star_canonical_closed_form(theta_G)
        k_num, _ = mckean_vlasov_kappa_star(theta_G=theta_G, **CANONICAL_REGIME)
        assert k_num == pytest.approx(k_closed, rel=1e-6), (
            f"theta_G={theta_G}: numerical {k_num} vs closed-form {k_closed}"
        )


def test_kappa_star_mv_canonical_audit_table() -> None:
    """Auditor's regression table at the canonical regime (Wave: external audit).

    The 4D Hopf threshold at short-gamma canonical params, verified independently:

        theta_G | kappa_star_MV
        0.5     | 0.536
        1.0     | 0.619
        5.0     | 0.800
        50      | 0.885
        500     | 0.895

    These are pinned to 3 decimals (closed-form is exact, so this is just a
    floor on the audit reproduction).
    """
    expected = [
        (0.5, 0.536),
        (1.0, 0.619),
        (5.0, 0.800),
        (50.0, 0.885),
        (500.0, 0.895),
    ]
    for theta_G, k_audit in expected:
        result = mckean_vlasov_kappa_star_shift(theta_G=theta_G, **CANONICAL_REGIME)
        assert result.kappa_star_mv == pytest.approx(k_audit, abs=0.001), (
            f"theta_G={theta_G}: computed {result.kappa_star_mv:.6f} vs audit {k_audit}"
        )


def test_kappa_star_mv_canonical_theta_5_pin() -> None:
    """Pinned 4-decimal regression: at theta_G=5 the MV threshold is 0.7996.

    Direct anchor against the auditor's report (4-decimal precision).
    """
    result = mckean_vlasov_kappa_star_shift(theta_G=5.0, **CANONICAL_REGIME)
    assert result.kappa_star_mv == pytest.approx(0.79955, abs=5e-5)


def test_kappa_star_mv_canonical_ratio_below_one() -> None:
    """At the canonical short-gamma regime the MV correction LOWERS the threshold."""
    for theta_G in [0.5, 1.0, 5.0, 50.0]:
        result = mckean_vlasov_kappa_star_shift(theta_G=theta_G, **CANONICAL_REGIME)
        assert result.ratio < 1.0, (
            f"theta_G={theta_G}: MV ratio {result.ratio:.4f} >= 1 — the v0.3.5 sign error has returned"
        )


def test_kappa_star_mv_instantaneous_hedging_limit() -> None:
    """As theta_G -> infty (instantaneous hedging), the MV ratio -> 1.

    The 4D system's $g$ channel collapses onto its target instantaneously,
    decoupling the dealer mode from the price/variance dynamics.
    """
    result = mckean_vlasov_kappa_star_shift(theta_G=1e6, **CANONICAL_REGIME)
    assert result.ratio == pytest.approx(1.0, abs=1e-5), f"large-theta_G ratio {result.ratio} != 1"
    # Single-dealer asymptote
    assert result.kappa_star_single == pytest.approx(KAPPA_STAR_SINGLE_CANONICAL, rel=1e-6)


def test_kappa_star_mv_frozen_dealer_limit_finite() -> None:
    """As theta_G -> 0+ the MV threshold approaches 8/21 (FINITE, not divergent).

    Unlike the v0.3.5 heuristic which blew up as 1/theta_G, the correct 4D
    closed-form has $\\kappa^\\star_\\mathrm{MV}(0^+) = 8/21 \\approx 0.381$
    at the canonical regime. The dealer state freezes but the 4D system
    still has a Hopf at a smaller threshold than the single-dealer model.
    """
    k_closed_small = mckean_vlasov_kappa_star_canonical_closed_form(1e-6)
    assert k_closed_small == pytest.approx(8.0 / 21.0, rel=1e-3)


def test_kappa_star_mv_long_gamma_regime_above_one() -> None:
    """The log-normal-OI long-gamma regime (G_y < 0) gives ratio > 1 — the
    regime-dependence of the corrected formula is a real feature.

    This test guards against accidentally re-applying a "ratio < 1 always"
    assumption taken from the short-gamma audit table.
    """
    # Approximate log-normal-OI calibration values from §4.3.
    result = mckean_vlasov_kappa_star_shift(
        theta_G=50.0,
        G_y=-0.035,
        G_v=-0.177,
        G_z=0.0,
        kappa_v=2.0,
        alpha=0.05,
        beta=1.0,
        gamma=1.0,
        sigma2_y=0.0,
        sigma2_v=1.0,
        kappa_min=0.1,
        kappa_max=1e5,
    )
    assert result.ratio > 1.0, f"long-gamma regime should have ratio > 1, got {result.ratio:.4f}"


def test_kappa_star_shift_legacy_omega_star_emits_deprecation_warning() -> None:
    """Passing the deprecated omega_star kwarg emits a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="omega_star is deprecated"):
        mckean_vlasov_kappa_star_shift(
            theta_G=5.0,
            omega_star=0.5,
            **CANONICAL_REGIME,
        )


def test_kappa_star_shift_rejects_invalid_theta_G() -> None:
    """theta_G must be > 0."""
    with pytest.raises(ValueError, match="theta_G"):
        mckean_vlasov_kappa_star_shift(theta_G=0.0, **CANONICAL_REGIME)


def test_kappa_star_shift_rejects_missing_jacobian_args() -> None:
    """The corrected API requires the full 4D Jacobian structure."""
    with pytest.raises(TypeError, match="missing keyword arguments"):
        mckean_vlasov_kappa_star_shift(theta_G=5.0)


def test_jacobian_4d_construction_shape_and_structure() -> None:
    """The 4D Jacobian is 4x4 with the expected sparsity pattern."""
    J = mckean_vlasov_jacobian_4d(kappa=1.0, theta_G=5.0, **CANONICAL_REGIME)
    assert J.shape == (4, 4)
    # Spot row: only the dealer-state coupling is nonzero (sigma2_y = sigma2_v = 0 at canonical).
    assert J[0, 0] == 0.0
    assert J[0, 1] == 0.0
    assert J[0, 2] == 0.0
    assert J[0, 3] == 1.0  # = kappa
    # Dealer row: theta_G * G_y, theta_G * G_v, theta_G * G_z, -theta_G
    assert J[3, 0] == pytest.approx(5.0 * 0.5)
    assert J[3, 1] == pytest.approx(5.0 * -0.5)
    assert J[3, 2] == pytest.approx(5.0 * -0.5)
    assert J[3, 3] == pytest.approx(-5.0)


def test_jacobian_4d_rejects_invalid_args() -> None:
    """theta_G, kappa_v, alpha must all be > 0."""
    base = dict(kappa=1.0, **CANONICAL_REGIME)
    with pytest.raises(ValueError, match="theta_G"):
        mckean_vlasov_jacobian_4d(theta_G=0.0, **base)
    base2 = {**base, "kappa_v": 0.0}
    with pytest.raises(ValueError, match="kappa_v"):
        mckean_vlasov_jacobian_4d(theta_G=1.0, **base2)
    base3 = {**base, "alpha": -1.0}
    with pytest.raises(ValueError, match="alpha"):
        mckean_vlasov_jacobian_4d(theta_G=1.0, **base3)


def test_kappa_star_canonical_closed_form_rejects_invalid_theta_G() -> None:
    with pytest.raises(ValueError, match="theta_G"):
        mckean_vlasov_kappa_star_canonical_closed_form(0.0)


def test_kappa_star_mv_solver_raises_when_no_hopf_in_bracket() -> None:
    """If no sign change of max(Re lambda) appears in [kappa_min, kappa_max],
    the solver raises rather than silently returning a bad value."""
    # Pick a tiny bracket that excludes the Hopf at canonical regime.
    with pytest.raises(RuntimeError, match="no 4D MV Hopf"):
        mckean_vlasov_kappa_star(
            theta_G=5.0,
            kappa_min=1e-5,
            kappa_max=1e-4,
            **CANONICAL_REGIME,
        )


# ---------------------------------------------------------------------------
# Mean-field limit trajectory.
# ---------------------------------------------------------------------------


def test_mean_field_limit_relaxes_to_target() -> None:
    """For constant target g and long horizon, $E[G(t)]$ converges to g."""
    target = 0.7
    _, traj = mean_field_limit_trajectory(
        theta_G=20.0,
        g_target=lambda _t: target,
        G_bar_inf_0=0.0,
        T=2.0,  # >> 1/theta_G = 0.05
        n_steps=400,
    )
    assert traj[-1] == pytest.approx(target, rel=1e-3)


def test_mean_field_limit_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="theta_G"):
        mean_field_limit_trajectory(
            theta_G=0.0, g_target=lambda _t: 0.0, G_bar_inf_0=0.0, T=1.0, n_steps=10
        )
    with pytest.raises(ValueError, match="T"):
        mean_field_limit_trajectory(
            theta_G=1.0, g_target=lambda _t: 0.0, G_bar_inf_0=0.0, T=0.0, n_steps=10
        )
    with pytest.raises(ValueError, match="n_steps"):
        mean_field_limit_trajectory(
            theta_G=1.0, g_target=lambda _t: 0.0, G_bar_inf_0=0.0, T=1.0, n_steps=0
        )


# ---------------------------------------------------------------------------
# Particle-system simulator.
# ---------------------------------------------------------------------------


def test_simulate_n_dealer_system_smoke() -> None:
    """Smoke: simulator runs at moderate n and returns a sensible-shape output."""
    n_particles = 64
    n_steps = 200

    def G0_dist(rng: np.random.Generator, n: int) -> np.ndarray:
        return 0.5 + 0.05 * rng.standard_normal(n)

    t_grid, paths = simulate_n_dealer_system(
        n_particles=n_particles,
        theta_G=20.0,
        sigma_G=0.1,
        g_target=lambda _t: 0.5,
        G0_distribution=G0_dist,
        T=0.5,
        n_steps=n_steps,
        seed=42,
    )
    assert t_grid.shape == (n_steps + 1,)
    assert paths.shape == (n_steps + 1, n_particles)
    assert np.all(np.isfinite(paths))
    # OU stationary mean is 0.5; the empirical mean over particles & late
    # times should be near it.
    assert paths[-100:].mean() == pytest.approx(0.5, abs=0.05)


def test_simulate_n_dealer_system_rejects_invalid_args() -> None:
    """Validate guards on the simulator inputs (Lipschitz-condition prerequisites)."""

    def G0_dist(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.standard_normal(n)

    base_kwargs = dict(
        theta_G=10.0,
        sigma_G=0.1,
        g_target=lambda _t: 0.0,
        G0_distribution=G0_dist,
        T=0.1,
        n_steps=10,
    )
    with pytest.raises(ValueError, match="n_particles"):
        simulate_n_dealer_system(n_particles=0, **base_kwargs)
    with pytest.raises(ValueError, match="theta_G"):
        simulate_n_dealer_system(
            n_particles=10,
            **{**base_kwargs, "theta_G": -1.0},
        )
    with pytest.raises(ValueError, match="sigma_G"):
        simulate_n_dealer_system(
            n_particles=10,
            **{**base_kwargs, "sigma_G": -0.5},
        )
    with pytest.raises(ValueError, match="T"):
        simulate_n_dealer_system(
            n_particles=10,
            **{**base_kwargs, "T": 0.0},
        )
    with pytest.raises(ValueError, match="n_steps"):
        simulate_n_dealer_system(
            n_particles=10,
            **{**base_kwargs, "n_steps": 0},
        )


# ---------------------------------------------------------------------------
# Propagation-of-chaos error.
# ---------------------------------------------------------------------------


def test_propagation_of_chaos_error_below_theoretical_bound() -> None:
    """The empirical RMSE is bounded by sqrt(C(T) / n) up to a small factor.

    Sznitman / Méléard give an *upper* bound; the empirical RMSE should
    sit at or below sqrt(C(T)/n) modulo the slack constant.  We verify
    the bound holds at a 2x slack margin (typical for this OU-target
    structure where the bound is tight).
    """
    theta_G = 10.0
    sigma_G = 0.1
    G0_std = 0.05
    T = 0.25
    var_G0 = G0_std * G0_std
    C_T = propagation_of_chaos_constant(
        theta_G=theta_G,
        sigma_G=sigma_G,
        var_G0=var_G0,
        T=T,
    )
    n_particles = 200
    err = propagation_of_chaos_error(
        n_particles=n_particles,
        theta_G=theta_G,
        sigma_G=sigma_G,
        g_target=lambda _t: 0.5,
        G0_mean=0.5,
        G0_std=G0_std,
        T=T,
        n_steps=200,
        n_replicates=32,
        seed=20260514,
    )
    bound = float(np.sqrt(C_T / n_particles))
    assert err.l2_error_sup < 2.0 * bound, (
        f"empirical RMSE {err.l2_error_sup:.3e} exceeds 2x Sznitman bound {bound:.3e}"
    )


# ---------------------------------------------------------------------------
# 1/sqrt(n) scaling — the headline numerical result.
# ---------------------------------------------------------------------------


def test_propagation_of_chaos_scaling_slope_near_one() -> None:
    """Fitted log-log slope vs 1/sqrt(n) is close to 1.0 (the Sznitman rate).

    With 5 n-points spanning two decades and 32 replicates, the slope
    estimator has finite-sample noise of $\\sim 0.1$.  We assert the
    fitted slope lies in [0.7, 1.3] — comfortably within tolerance for
    the bound to be considered numerically validated, and structurally
    excludes both the no-scaling ($a = 0$) and overestimate ($a >> 1$)
    failure modes.
    """
    n_grid = np.array([20, 60, 200, 600, 2000], dtype=np.int64)
    scaling = propagation_of_chaos_scaling(
        n_grid=n_grid,
        theta_G=10.0,
        sigma_G=0.1,
        g_target=lambda _t: 0.5,
        G0_mean=0.5,
        G0_std=0.05,
        T=0.25,
        n_steps=100,
        n_replicates=24,
        seed=20260514,
    )
    # Strict expected slope is 1.0 (RMSE ∝ 1/sqrt(n)); allow ±0.30 slack.
    assert 0.7 <= scaling.fitted_slope <= 1.3, (
        f"fitted slope {scaling.fitted_slope:.3f} out of [0.7, 1.3]; 1/sqrt(n) scaling is broken"
    )


def test_mckean_vlasov_validation_experiment_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Smoke: the experiment runner executes end-to-end and writes artifacts.

    Uses --quick mode (3 n-points, 16 replicates) so wall-clock is < 5s.
    Validates that:
        - run() returns a metrics dict with the expected keys.
        - The figure is rendered to paper/figures/.
        - The fitted slope is in a sane range (0.5..1.5; broader than the
          dedicated scaling test above because --quick uses fewer replicates).
    """
    from reflexive_options.experiments import mckean_vlasov_validation as mv_exp

    metrics = mv_exp.run(quick=True)
    assert "kappa_star_single" in metrics
    assert "kappa_star_mv" in metrics
    assert "kappa_star_shift_ratio" in metrics
    assert "fitted_slope_log_inv_sqrt_n" in metrics
    assert "C_T_theoretical" in metrics
    slope = float(metrics["fitted_slope_log_inv_sqrt_n"])  # type: ignore[arg-type]
    assert 0.5 <= slope <= 1.5, f"experiment slope {slope:.3f} out of [0.5, 1.5]"
    # The log-normal-OI calibration (G_y < 0) is a long-gamma regime where the
    # corrected v0.3.6 MV ratio is > 1. Short-gamma canonical regimes give
    # ratio < 1; both are covered in the unit tests above.
    assert float(metrics["kappa_star_shift_ratio"]) > 1.0  # type: ignore[arg-type]


def test_propagation_of_chaos_error_rejects_invalid_replicates() -> None:
    """n_replicates must be >= 1."""
    with pytest.raises(ValueError, match="n_replicates"):
        propagation_of_chaos_error(
            n_particles=10,
            theta_G=1.0,
            sigma_G=0.1,
            g_target=lambda _t: 0.0,
            G0_mean=0.0,
            G0_std=0.1,
            T=0.1,
            n_steps=10,
            n_replicates=0,
        )


def test_simulate_n_dealer_system_seed_none_runs() -> None:
    """Smoke: passing seed=None defaults to fresh entropy and runs cleanly."""

    def G0_dist(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.standard_normal(n)

    _, paths = simulate_n_dealer_system(
        n_particles=4,
        theta_G=1.0,
        sigma_G=0.1,
        g_target=lambda _t: 0.0,
        G0_distribution=G0_dist,
        T=0.05,
        n_steps=5,
        seed=None,
    )
    assert paths.shape == (6, 4)
    assert np.all(np.isfinite(paths))


def test_mckean_vlasov_validation_main_quick(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Cover the experiment's main() entrypoint via argparse."""
    from reflexive_options.experiments import mckean_vlasov_validation as mv_exp

    monkeypatch.setattr("sys.argv", ["mckean_vlasov_validation", "--quick"])
    mv_exp.main()


def test_propagation_of_chaos_scaling_rejects_invalid_grid() -> None:
    """Non-monotonic and singleton n_grid are rejected."""
    with pytest.raises(ValueError, match="ascending"):
        propagation_of_chaos_scaling(
            n_grid=np.array([100, 10], dtype=np.int64),
            theta_G=1.0,
            sigma_G=0.1,
            g_target=lambda _t: 0.0,
            G0_mean=0.0,
            G0_std=0.1,
            T=0.1,
            n_steps=10,
            n_replicates=4,
        )
    with pytest.raises(ValueError, match=">= 2"):
        propagation_of_chaos_scaling(
            n_grid=np.array([100], dtype=np.int64),
            theta_G=1.0,
            sigma_G=0.1,
            g_target=lambda _t: 0.0,
            G0_mean=0.0,
            G0_std=0.1,
            T=0.1,
            n_steps=10,
            n_replicates=4,
        )
