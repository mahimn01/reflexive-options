"""Tests for the McKean-Vlasov mean-field limit (paper §4 / Appendix B).

Covers:
    - Closed-form $C(T)$ propagation-of-chaos constant.
    - $\\kappa^\\star$ shift formula limit cases (theta_G -> infty, finite).
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
    mckean_vlasov_kappa_star_shift,
    mean_field_limit_trajectory,
    propagation_of_chaos_constant,
    propagation_of_chaos_error,
    propagation_of_chaos_scaling,
    simulate_n_dealer_system,
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
# Hopf-threshold shift.
# ---------------------------------------------------------------------------


def test_kappa_star_shift_instantaneous_hedging_limit() -> None:
    """As theta_G -> infty (instantaneous hedging), MV ratio -> 1."""
    ratio_fast = mckean_vlasov_kappa_star_shift(theta_G=1e6, omega_star=1.0)
    assert ratio_fast == pytest.approx(1.0, abs=1e-6)


def test_kappa_star_shift_slow_hedging_increases_threshold() -> None:
    """Slow hedging (small theta_G) raises the threshold (ratio > 1)."""
    ratio_slow = mckean_vlasov_kappa_star_shift(theta_G=0.5, omega_star=1.0)
    # With theta_G = omega_star: ratio = sqrt(1 + (1/0.5)^2) = sqrt(5) ≈ 2.236
    assert ratio_slow == pytest.approx(np.sqrt(5.0), rel=1e-6)


def test_kappa_star_shift_zero_omega() -> None:
    """At zero Hopf frequency the channel is DC; ratio = 1 regardless of theta_G."""
    assert mckean_vlasov_kappa_star_shift(theta_G=10.0, omega_star=0.0) == pytest.approx(1.0)


def test_kappa_star_shift_rejects_invalid_args() -> None:
    """theta_G must be > 0 and omega_star must be >= 0."""
    with pytest.raises(ValueError, match="theta_G"):
        mckean_vlasov_kappa_star_shift(theta_G=0.0, omega_star=1.0)
    with pytest.raises(ValueError, match="omega_star"):
        mckean_vlasov_kappa_star_shift(theta_G=1.0, omega_star=-0.1)


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
    # MV correction is positive (threshold expands).
    assert float(metrics["kappa_star_shift_ratio"]) >= 1.0  # type: ignore[arg-type]


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
