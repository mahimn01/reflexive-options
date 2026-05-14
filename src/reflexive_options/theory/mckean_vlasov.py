r"""McKean-Vlasov mean-field limit of the dealer-gamma channel.

The single-representative-dealer SDE in the §2 model implicitly assumes
either (a) perfect coordination across dealers, or (b) the law-of-large-
numbers limit where idiosyncratic dealer noise washes out.  This module
formalises (b) as a McKean-Vlasov (MV) SDE coupled to the *law* of the
dealer-gamma process, and provides a finite-particle simulator that
validates Sznitman (1991) / Méléard (1996)'s propagation-of-chaos $1/n$
rate.

Setup (n-dealer system).  Dealer $i \in \{1, \ldots, n\}$ holds gamma
exposure $G_i$ that obeys

    dG_i = -theta_G (G_i - g(S, v)) dt + sigma_G dW^i_G,

with $\{W^i_G\}_i$ independent Brownian motions and $g(S, v)$ the
"target" dealer-gamma map (e.g. the closed-form log-normal-OI aggregator
of paper §4.3, evaluated at the current spot/variance).  $\theta_G > 0$
is the dealer-hedging speed (equivalently $\tau_G := 1/\theta_G$ is the
autocorrelation time of the gamma deviation).  Aggregating into the
spot dynamics:

    dS / S = (mu + kappa * G_bar_n) dt + sigma(S, v) dW^S,
    G_bar_n := (1/n) sum_i G_i.

In the limit $n \to \infty$, propagation of chaos gives weak convergence
of the empirical measure $\bar\mu_n^t = (1/n)\sum_i \delta_{G_i^t}$ to
the deterministic measure $\mu^t = \mathrm{Law}(G^t)$ where $G$ solves
the MV SDE

    dG = -theta_G (G - g(S, v)) dt + sigma_G dW_G,
    dS / S = (mu + kappa * G_bar_inf) dt + sigma(S, v) dW^S,
    G_bar_inf(t) = E[G^t | F_t^{S, v}] = \int g\, d\mu^t(g).

For the OU-target dynamics here, $G_bar_inf(t) = E[G(t)]$ which obeys
the mean ODE $\dot{\bar G}_\infty = -\theta_G (\bar G_\infty - g(S, v))$ —
the conditional expectation matches the deterministic relaxation of
$G$ towards its target.

Key quantitative outputs:

1. Propagation-of-chaos $L^2$ error bound (Sznitman 1991, Théorème I.1.4;
   Méléard 1996 Prop 2.5; Carmona-Delarue 2018 Vol I Thm 2.12).  Under
   Lipschitz $b, sigma_G$ and finite second moment of $G_i^0$,

       sup_{t <= T} E[(G_bar_n(t) - G_bar_inf(t))^2] <= C(T) / n.

   For our linear-in-$G$ drift this constant is explicit:

       C(T) = sigma_G^2 / (2 theta_G) * (1 - exp(-2 theta_G T))
              + Var(G^0) * exp(-2 theta_G T)
            <= max(sigma_G^2 / (2 theta_G), Var(G^0)).

2. Hopf-threshold shift.  The MV system's $G$-channel acts as a
   first-order low-pass filter on the target $g(S, v)$ with time
   constant $\tau_G = 1/\theta_G$.  At the Hopf frequency $\omega^\star$
   the effective coupling becomes
   $\kappa_\text{eff} = \kappa \cdot \theta_G / \sqrt{\theta_G^2 + (\omega^\star)^2}$,
   so the threshold shifts to

       kappa_star_MV / kappa_star_single
           = sqrt(theta_G^2 + omega_star^2) / theta_G
           = sqrt(1 + (omega_star * tau_G)^2)
           = 1 + 0.5 * (omega_star * tau_G)^2 + O((omega_star tau_G)^4).

   For instantaneous hedging $\theta_G \to \infty$ (i.e. $\tau_G \to 0$)
   the correction vanishes: MV agrees with single-dealer.  For slow
   hedging the threshold is *higher* — the dealer ensemble damps the
   feedback channel, requiring stronger coupling to destabilise.

This module implements:

* `mckean_vlasov_kappa_star_shift` — closed-form ratio above.
* `simulate_n_dealer_system` — Euler-Maruyama on the n-particle SDE.
* `propagation_of_chaos_error` — measures $|G_bar_n - G_bar_inf|$ and
  the $L^2$ supremum over a path.
* `propagation_of_chaos_scaling` — sweeps $n$ and reports the fitted
  log-log slope (should be $\approx 0.5$ for $\sqrt{1/n}$ RMSE).

References:
    Sznitman, A.-S. (1991) "Topics in propagation of chaos." Lecture
        Notes in Math. 1464.  Théorème I.1.4 (the canonical $C/n$ bound).
    Méléard, S. (1996) "Asymptotic behaviour of some interacting particle
        systems; McKean-Vlasov and Boltzmann models." Lecture Notes in
        Math. 1627, Prop 2.5.
    Carmona, R. & Delarue, F. (2018) Probabilistic Theory of Mean Field
        Games with Applications I-II. Springer.  Vol I, Thm 2.12.
    Lacker, D. (2018) "Mean field games via controlled martingale problems."
        SPA 128.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Closed-form Hopf-threshold shift (Deliverable 3).
# ---------------------------------------------------------------------------


def mckean_vlasov_kappa_star_shift(
    *,
    theta_G: float,
    omega_star: float,
) -> float:
    """Closed-form ratio kappa_star_MV / kappa_star_single.

    The dealer-ensemble OU channel is a low-pass filter of bandwidth
    $\\theta_G$ acting on the spot/variance perturbation that drives the
    target gamma $g(S, v)$.  At the Hopf frequency $\\omega^\\star$, the
    transfer-function gain is $\\theta_G / \\sqrt{\\theta_G^2 + \\omega^{\\star 2}}$,
    so the effective coupling shrinks by that factor and the threshold
    expands by its reciprocal.

    Returns:
        kappa_star_MV / kappa_star_single = sqrt(1 + (omega_star / theta_G)^2).

        For theta_G -> infty (instantaneous hedging, tau_G -> 0) the ratio
        is 1; for slow hedging the threshold is strictly higher.

    Args:
        theta_G: dealer-hedging speed > 0 (autocorrelation time tau_G = 1/theta_G).
        omega_star: deterministic Hopf frequency at the single-dealer threshold.
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if omega_star < 0.0:
        raise ValueError(f"omega_star must be >= 0, got {omega_star}")
    return float(np.sqrt(1.0 + (omega_star / theta_G) ** 2))


def propagation_of_chaos_constant(
    *,
    theta_G: float,
    sigma_G: float,
    var_G0: float,
    T: float,
) -> float:
    """Closed-form $C(T)$ in $\\sup_t E[(G_bar_n - G_bar_inf)^2] \\le C(T)/n$.

    For OU dynamics $dG_i = -\\theta_G(G_i - g) dt + \\sigma_G dW^i_G$ with
    a *common* (non-particle-dependent) target $g = g(S, v)$, the
    deviations $\\delta_i := G_i - g(S, v)$ inherit the same OU and the
    cross-correlations $E[\\delta_i \\delta_j]$ for $i \\ne j$ vanish under
    independent Brownian motions.  The variance of the empirical mean is

        Var(G_bar_n - G_bar_inf) = (1/n) Var(G_i - g)
                                 = (1/n) [Var(G^0 - g^0) e^{-2 theta_G t}
                                          + (sigma_G^2 / 2 theta_G)
                                            * (1 - e^{-2 theta_G t})].

    The supremum over $t \\in [0, T]$ is bounded by
    max(Var(G^0), sigma_G^2 / (2 theta_G)) at the worst-case $t$.

    Returns:
        C(T) such that sup_{t<=T} E[(G_bar_n - G_bar_inf)^2] <= C(T) / n.
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if sigma_G < 0.0:
        raise ValueError(f"sigma_G must be >= 0, got {sigma_G}")
    if var_G0 < 0.0:
        raise ValueError(f"var_G0 must be >= 0, got {var_G0}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    decay = float(np.exp(-2.0 * theta_G * T))
    stationary = sigma_G * sigma_G / (2.0 * theta_G)
    # The two extreme regimes give the supremum over t in [0, T]:
    #   - If Var(G^0) > stationary : C(T) = Var(G^0) (decay-dominated, sup at t=0).
    #   - If Var(G^0) < stationary : C(T) = stationary * (1 - exp(-2 theta_G T))
    #     plus the residual decay term — sup at t=T.
    sup_t = max(var_G0, stationary * (1.0 - decay) + var_G0 * decay)
    return float(sup_t)


# ---------------------------------------------------------------------------
# Particle-system simulator (Deliverable 4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosErrorResult:
    """Output of `propagation_of_chaos_error` at a single particle count.

    `n_particles`: number of dealers in the simulation.
    `t_grid`: time points used for the Euler discretisation.
    `mean_traj_n`: $\\bar G_n(t)$ across the simulation horizon.
    `mean_traj_inf`: $\\bar G_\\infty(t)$ — the deterministic OU mean.
    `l2_error_sup`: $\\sup_t \\sqrt{E[(G_n - G_\\infty)^2]}$ over `n_replicates`
        independent draws (RMSE supremum).
    `n_replicates`: number of independent particle-system replicates used
        to estimate the expectation.
    """

    n_particles: int
    t_grid: NDArray[np.float64]
    mean_traj_n: NDArray[np.float64]  # one representative replicate, for plotting
    mean_traj_inf: NDArray[np.float64]
    l2_error_sup: float
    n_replicates: int


def simulate_n_dealer_system(
    *,
    n_particles: int,
    theta_G: float,
    sigma_G: float,
    g_target: Callable[[float], float],
    G0_distribution: Callable[[np.random.Generator, int], NDArray[np.float64]],
    T: float,
    n_steps: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Euler-Maruyama on the n-dealer system at a frozen $g(S, v)$ trajectory.

    For the propagation-of-chaos validation we hold the spot/variance path
    deterministic — the question is purely about how well $\\bar G_n$
    tracks $\\bar G_\\infty$ at finite $n$, irrespective of the spot
    feedback (which is identical between the two systems by construction).

    Args:
        n_particles: number of dealers ($n$).
        theta_G: hedging speed > 0.
        sigma_G: idiosyncratic noise scale >= 0.
        g_target: callable $t \\mapsto g(S(t), v(t))$ — the common target.
        G0_distribution: callable (rng, n) -> initial G values, length n.
        T: horizon (years).
        n_steps: Euler-Maruyama steps over [0, T].
        seed: optional RNG seed.

    Returns:
        (t_grid, G_paths) where t_grid is shape (n_steps+1,) and G_paths
        is shape (n_steps+1, n_particles).
    """
    if n_particles <= 0:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if sigma_G < 0.0:
        raise ValueError(f"sigma_G must be >= 0, got {sigma_G}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = float(np.sqrt(dt))
    t_grid = np.linspace(0.0, T, n_steps + 1, dtype=np.float64)

    G = np.asarray(G0_distribution(rng, n_particles), dtype=np.float64).reshape(n_particles)
    paths = np.zeros((n_steps + 1, n_particles), dtype=np.float64)
    paths[0] = G

    for k in range(1, n_steps + 1):
        t = float(t_grid[k - 1])
        g_t = float(g_target(t))
        dW = rng.standard_normal(n_particles) * sqrt_dt
        # OU drift towards the common target; independent noise per particle.
        G = G + (-theta_G * (G - g_t)) * dt + sigma_G * dW
        paths[k] = G
    return t_grid, paths


def mean_field_limit_trajectory(
    *,
    theta_G: float,
    g_target: Callable[[float], float],
    G_bar_inf_0: float,
    T: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Deterministic ODE for $\\bar G_\\infty(t)$ in the MV limit.

    For the OU dynamics $dG = -\\theta_G(G - g)dt + \\sigma_G dW$, taking
    expectation gives $\\dot E[G] = -\\theta_G(E[G] - g(S, v))$ — a first-
    order linear ODE with closed form

        E[G(t)] = E[G(0)] e^{-theta_G t}
                + theta_G \\int_0^t e^{-theta_G (t - s)} g(S(s), v(s)) ds.

    We integrate with the explicit Euler scheme matching the particle
    simulator's discretisation so the comparison is clean.
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1, dtype=np.float64)
    traj = np.zeros(n_steps + 1, dtype=np.float64)
    traj[0] = G_bar_inf_0
    for k in range(1, n_steps + 1):
        t = float(t_grid[k - 1])
        traj[k] = traj[k - 1] + (-theta_G * (traj[k - 1] - float(g_target(t)))) * dt
    return t_grid, traj


def propagation_of_chaos_error(
    *,
    n_particles: int,
    theta_G: float,
    sigma_G: float,
    g_target: Callable[[float], float],
    G0_mean: float,
    G0_std: float,
    T: float,
    n_steps: int,
    n_replicates: int = 32,
    seed: int | None = None,
) -> ChaosErrorResult:
    """Estimate $\\sup_t \\sqrt{E[(G_bar_n - G_bar_inf)^2]}$ over `n_replicates`.

    The expectation is over the joint Brownian noise + initial-condition
    randomness of the n-particle system; $G_bar_inf$ is the deterministic
    OU mean (closed form via `mean_field_limit_trajectory`).

    Args:
        n_particles: dealers per replicate.
        theta_G, sigma_G, T, n_steps: SDE config.
        g_target: deterministic target trajectory.
        G0_mean, G0_std: initial Gaussian distribution moments for $G_i^0$.
        n_replicates: independent simulations for the expectation estimate.
        seed: optional RNG seed (each replicate uses seed + replicate_idx).

    Returns:
        `ChaosErrorResult` with the path-supremum RMSE.
    """
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")

    # Mean-field limit: starts at G0_mean (the expectation of G_i^0).
    t_grid, traj_inf = mean_field_limit_trajectory(
        theta_G=theta_G,
        g_target=g_target,
        G_bar_inf_0=G0_mean,
        T=T,
        n_steps=n_steps,
    )

    def G0_dist(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return G0_mean + G0_std * rng.standard_normal(n)

    # Track squared error at every time step across replicates.
    squared_errors = np.zeros((n_replicates, n_steps + 1), dtype=np.float64)
    representative_traj_n = None
    for r in range(n_replicates):
        replicate_seed = None if seed is None else seed + r
        _, paths = simulate_n_dealer_system(
            n_particles=n_particles,
            theta_G=theta_G,
            sigma_G=sigma_G,
            g_target=g_target,
            G0_distribution=G0_dist,
            T=T,
            n_steps=n_steps,
            seed=replicate_seed,
        )
        mean_traj_n = paths.mean(axis=1)  # (n_steps+1,)
        squared_errors[r] = (mean_traj_n - traj_inf) ** 2
        if r == 0:
            representative_traj_n = mean_traj_n

    # Per-time MSE then sup over time, then sqrt — i.e. sup_t RMSE_t.
    per_time_mse = squared_errors.mean(axis=0)
    rmse_sup = float(np.sqrt(np.max(per_time_mse)))

    assert representative_traj_n is not None
    return ChaosErrorResult(
        n_particles=n_particles,
        t_grid=t_grid,
        mean_traj_n=representative_traj_n,
        mean_traj_inf=traj_inf,
        l2_error_sup=rmse_sup,
        n_replicates=n_replicates,
    )


@dataclass(frozen=True)
class ChaosScalingResult:
    """Output of `propagation_of_chaos_scaling` over a sweep of $n$.

    `n_grid`: array of particle counts swept (ascending).
    `rmse_sup`: $\\sup_t \\sqrt{E[(G_n - G_\\infty)^2]}$ at each $n$.
    `fitted_slope`: log-log slope of `rmse_sup` vs `1/sqrt(n)`; should be
        $\\approx 1$ when plotted against $1/\\sqrt n$ (i.e. $\\approx -1/2$
        when plotted vs $n$).
    `fitted_intercept`: log-log intercept; intercept $\\approx \\log\\sqrt{C(T)}$
        with $C(T)$ from `propagation_of_chaos_constant`.
    `theoretical_constant`: $C(T)$ from the closed form.
    """

    n_grid: NDArray[np.int64]
    rmse_sup: NDArray[np.float64]
    fitted_slope: float
    fitted_intercept: float
    theoretical_constant: float


def propagation_of_chaos_scaling(
    *,
    n_grid: NDArray[np.int64],
    theta_G: float,
    sigma_G: float,
    g_target: Callable[[float], float],
    G0_mean: float,
    G0_std: float,
    T: float,
    n_steps: int,
    n_replicates: int = 32,
    seed: int | None = None,
) -> ChaosScalingResult:
    """Sweep $n \\in n_grid$ and verify the $1/\\sqrt n$ RMSE scaling.

    Returns the empirical RMSEs plus a least-squares fit of
    $\\log(\\text{RMSE}) = a \\cdot \\log(1/\\sqrt n) + b$.  The Sznitman
    bound predicts $a \\approx 1$ (i.e. RMSE $\\propto 1/\\sqrt n$).
    """
    if n_grid.ndim != 1 or len(n_grid) < 2:
        raise ValueError("n_grid must be 1D with >= 2 entries")
    if not np.all(np.diff(n_grid) > 0):
        raise ValueError("n_grid must be strictly ascending")

    rmses = np.zeros(len(n_grid), dtype=np.float64)
    for i, n in enumerate(n_grid):
        # Use a different seed per n so noise patterns aren't correlated
        # across particle-count buckets.
        n_seed = None if seed is None else int(seed + 1000 * i)
        result = propagation_of_chaos_error(
            n_particles=int(n),
            theta_G=theta_G,
            sigma_G=sigma_G,
            g_target=g_target,
            G0_mean=G0_mean,
            G0_std=G0_std,
            T=T,
            n_steps=n_steps,
            n_replicates=n_replicates,
            seed=n_seed,
        )
        rmses[i] = result.l2_error_sup

    # Least-squares fit on log(RMSE) vs log(1/sqrt(n)).
    x = np.log(1.0 / np.sqrt(n_grid.astype(np.float64)))
    y = np.log(rmses)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    C_T = propagation_of_chaos_constant(
        theta_G=theta_G,
        sigma_G=sigma_G,
        var_G0=G0_std * G0_std,
        T=T,
    )

    return ChaosScalingResult(
        n_grid=n_grid.astype(np.int64),
        rmse_sup=rmses,
        fitted_slope=float(slope),
        fitted_intercept=float(intercept),
        theoretical_constant=float(C_T),
    )
