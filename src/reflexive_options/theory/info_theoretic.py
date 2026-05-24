"""Information-theoretic reflexivity — Theorem 5 (excess entropy at the Hopf boundary).

This module operationalises **Theorem 5** from paper §3.10: the linearised
3D reflexive SDE is a multivariate Ornstein-Uhlenbeck process, and the
conditional mutual information

    E_tau(kappa) := I(F_(-inf, 0]^y ; R_tau | u_0, z_0),

with R_tau := y_tau - y_0 = ∫_0^tau (dS_s / S_s) the integrated future log-return
and (u_0, z_0) the present variance- and memory-deviations, is computable in
closed form via the Lyapunov stationary covariance and the matrix exponential.

The key structural result (proven in paper/theory.md §10):

    E_tau(kappa) = (1/2) log(1 + (v_1^2 * sigma_y|u,z^2) / m_y(tau))

where

    v_1   := (e_1^T (exp(J(kappa) tau) - I))_1
    sigma_y|u,z^2 := P_11 - P_{1,2:} P_{2:,2:}^{-1} P_{2:,1}    (Schur complement of P)
    m_y(tau) := (P - exp(J tau) P exp(J^T tau))_{1,1}
    P solves J P + P J^T + Sigma Sigma^T = 0.

The structural / surprising finding (vs the naive "MI diverges at criticality"
prior): E_tau(kappa^*) is FINITE, not divergent. The slow-mode collapse near
the Hopf bifurcation is coherent across (y, u, z) — it lives in a 1D subspace
— so conditioning on the (u, z) projection of the present state removes the
divergent contribution to Var(y | u, z) exactly. The past-spot information
that survives the conditioning has a bounded contribution and saturates to a
finite asymptote.

What IS true at criticality:
  * E_tau(0+) = 0 (Markov closure: the kappa = 0 SDE has y as a frozen mode).
  * E_tau is monotonically increasing in kappa on (0, kappa^*) at the canonical
    regime (verified numerically — proven on (kappa_NS, kappa^*) where the
    slow mode is a complex pair, via Hopf transversality + implicit-function
    theorem applied to the Lyapunov equation; see §10.3 in theory.md).
  * The approach to E_tau(kappa^*) is LINEAR in (kappa^* - kappa) near the
    boundary (mean-field critical exponent beta = 1):
        E_tau(kappa) = E_tau(kappa^*) - C * (kappa^* - kappa) + O((kappa^* - kappa)^2).
  * Empirical proxy: the transfer entropy from a (calibrated) dealer-gamma
    series G_t to next-step returns r_{t+1}, measured under an IAAFT-surrogate
    null, gives a model-free analogue testable in Phase 4. We supply a
    Schreiber-1985-style empirical estimator and the IAAFT-null calibration.

Tests in `tests/test_info_theoretic.py` cover: the Markov limit
E_tau(0+) -> 0, monotonicity on the canonical kappa-grid, the linear
critical-exponent fit (beta = 1) at the boundary, IAAFT-null calibration of
the empirical transfer entropy estimator, and a Stuart-Landau-like
positive-control smoke test.

References:
    Schreiber (2000), "Measuring Information Transfer", Phys Rev Lett 85: 461.
    Crutchfield & Feldman (2003), "Regularities Unseen, Randomness Observed:
        Levels of Entropy Convergence", Chaos 13: 25.
    Crutchfield (2012), "Between Order and Chaos", Nature Phys 8: 17.
    Lizier, Prokopenko & Zomaya (2012), "Local Measures of Information
        Storage in Complex Distributed Computation", Information Sciences 208: 39.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, solve_continuous_lyapunov

from reflexive_options.theory.spectral import iaaft_surrogate


@dataclass(frozen=True)
class ExcessEntropyCurveResult:
    """Output of `excess_entropy_curve`.

    Attributes:
        kappa_grid: ascending kappa values used for the scan.
        tau: time horizon (years) the excess entropy is evaluated at.
        excess_entropy: E_tau(kappa) at each grid point (nats). NaN entries
            mark non-Hurwitz kappa values where the stationary covariance
            P does not exist (past kappa_star, typically).
        is_monotone: True iff E_tau is non-decreasing across the entire
            evaluated grid (defined-entries only).
        kappa_star_estimated: smallest kappa in the grid at which J(kappa)
            stops being Hurwitz (the first non-finite excess-entropy entry,
            or None if the grid stays Hurwitz throughout).
    """

    kappa_grid: NDArray[np.float64]
    tau: float
    excess_entropy: NDArray[np.float64]
    is_monotone: bool
    kappa_star_estimated: float | None


def _stationary_covariance(
    jacobian: NDArray[np.float64],
    diffusion_outer: NDArray[np.float64],
    *,
    hurwitz_tol: float = 1e-12,
) -> NDArray[np.float64] | None:
    """Solve J P + P J^T + Σ Σ^T = 0 for the stationary covariance.

    Returns None if J is not Hurwitz (some eigenvalue has real part ≥
    -hurwitz_tol), in which case no stationary covariance exists.
    """
    eigs = np.linalg.eigvals(jacobian)
    if not np.all(eigs.real < -hurwitz_tol):
        return None
    # solve_continuous_lyapunov solves J X + X J^T = -Q for X given Q.
    # We want J P + P J^T = -SS, so pass Q = SS.
    p = solve_continuous_lyapunov(jacobian, -diffusion_outer)
    # Symmetrise to clean numerical asymmetry.
    return np.asarray(0.5 * (p + p.T), dtype=np.float64)


def excess_entropy_linear(
    jacobian: NDArray[np.float64],
    diffusion_outer: NDArray[np.float64],
    tau: float,
    *,
    observed_index: int = 0,
    conditioned_indices: tuple[int, ...] = (1, 2),
    hurwitz_tol: float = 1e-12,
) -> float:
    """Closed-form Gaussian conditional mutual information of the linearised SDE.

    Computes

        E_tau = I(x_0[observed] ; (x_tau - x_0)[observed] | x_0[conditioned])

    for the multivariate OU dx = J x dt + Σ dW with J Hurwitz, treating the
    `observed_index` component as the "spot" (y) and the `conditioned_indices`
    as the "present non-spot state" (variance- and memory-deviations).

    Derivation in paper/theory.md §10.2: under stationarity the conditional MI
    reduces to (1/2) log(1 + v_1^2 σ²_{y|u,z} / m_y(τ)) where
        v_1 = (e_obs^T (exp(J τ) - I))_obs,
        σ²_{y|u,z} = Schur complement of the stationary covariance P,
        m_y(τ) = (P - exp(J τ) P exp(J^T τ))_{obs,obs}.

    Args:
        jacobian: (d, d) Jacobian of the linearised drift at the equilibrium.
        diffusion_outer: (d, d) outer product Σ Σ^T of the diffusion matrix.
        tau: time horizon (must be > 0).
        observed_index: index of the "spot" component (default 0 = log S).
        conditioned_indices: indices of the components conditioned on (default
            (1, 2) = variance, memory).
        hurwitz_tol: J is considered Hurwitz iff max Re(eig) < -hurwitz_tol.

    Returns:
        E_tau in nats. Returns NaN if J is not Hurwitz (no stationary
        covariance exists).

    Raises:
        ValueError: on shape / index mismatches or non-positive tau.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    j = np.asarray(jacobian, dtype=np.float64)
    ss = np.asarray(diffusion_outer, dtype=np.float64)
    if j.ndim != 2 or j.shape[0] != j.shape[1]:
        raise ValueError(f"jacobian must be square, got shape {j.shape}")
    if ss.shape != j.shape:
        raise ValueError(f"diffusion_outer shape {ss.shape} must match jacobian shape {j.shape}")
    d = j.shape[0]
    if not 0 <= observed_index < d:
        raise ValueError(f"observed_index {observed_index} out of range for d={d}")
    cond = tuple(int(i) for i in conditioned_indices)
    if any(not 0 <= i < d for i in cond):
        raise ValueError(f"conditioned_indices {cond} out of range for d={d}")
    if observed_index in cond:
        raise ValueError(
            f"observed_index {observed_index} cannot also be in conditioned_indices {cond}"
        )

    p = _stationary_covariance(j, ss, hurwitz_tol=hurwitz_tol)
    if p is None:
        return float("nan")

    # Conditional variance of x_0[observed] given x_0[conditioned] — Schur
    # complement of P[cond,cond] in P[{obs}∪cond, {obs}∪cond].
    cond_arr = np.array(cond, dtype=np.int64)
    p_obs_obs = float(p[observed_index, observed_index])
    p_obs_cond = p[observed_index, cond_arr]
    p_cond_cond = p[np.ix_(cond_arr, cond_arr)]
    schur = float(p_obs_cond @ np.linalg.solve(p_cond_cond, p_obs_cond))
    var_y_cond = p_obs_obs - schur
    if var_y_cond <= 0.0:
        # Numerical degeneracy: the conditional variance should be > 0
        # for a non-degenerate stationary OU. Treat as zero MI (the
        # observed component is a deterministic function of the conditioned
        # ones, so there is nothing to predict from).
        return 0.0

    # v_1 = (e_obs^T (exp(J τ) - I))_obs.
    exp_jt = expm(j * float(tau))
    v_obs = float(exp_jt[observed_index, observed_index] - 1.0)

    # m_y(τ) = (P - exp(J τ) P exp(J^T τ))_{obs,obs}.
    cond_cov = p - exp_jt @ p @ exp_jt.T
    m_obs = float(cond_cov[observed_index, observed_index])
    if m_obs <= 0.0:
        # Same degeneracy guard: a zero conditional variance of the future
        # increment means the increment is deterministic given the present —
        # the MI is then either +inf (informative determinism) or 0
        # (uninformative determinism). Both edge cases require care; for our
        # Hurwitz OU with non-singular Σ this should not occur.
        return float("nan")

    ratio = v_obs * v_obs * var_y_cond / m_obs
    return 0.5 * float(np.log1p(ratio))


def excess_entropy_curve(
    kappa_grid: NDArray[np.float64],
    jacobian_at: Callable[[float], NDArray[np.float64]],
    diffusion_outer: NDArray[np.float64],
    tau: float,
    *,
    observed_index: int = 0,
    conditioned_indices: tuple[int, ...] = (1, 2),
    monotone_tol: float = 1e-10,
) -> ExcessEntropyCurveResult:
    """Sweep `excess_entropy_linear` over a kappa grid.

    Args:
        kappa_grid: strictly ascending kappa values.
        jacobian_at: callable kappa ↦ Jacobian matrix.
        diffusion_outer: (d, d) outer product Σ Σ^T (kappa-independent under
            the constant-vol surrogate of paper §4.2).
        tau: time horizon (years).
        observed_index, conditioned_indices: see `excess_entropy_linear`.
        monotone_tol: tolerance for the monotonicity check (an entry-to-entry
            decrease of magnitude ≤ monotone_tol is treated as flat).

    Returns:
        ExcessEntropyCurveResult with the per-kappa E_tau values, the
        monotonicity flag, and the located kappa_star (first NaN, if any).
    """
    if not np.all(np.diff(kappa_grid) > 0):
        raise ValueError("kappa_grid must be strictly ascending")
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")

    n = len(kappa_grid)
    e_vals = np.zeros(n, dtype=np.float64)
    for i, k in enumerate(kappa_grid):
        e_vals[i] = excess_entropy_linear(
            jacobian_at(float(k)),
            diffusion_outer,
            tau,
            observed_index=observed_index,
            conditioned_indices=conditioned_indices,
        )

    # Locate kappa_star as the smallest kappa whose excess entropy is NaN.
    nan_mask = np.isnan(e_vals)
    kappa_star = float(kappa_grid[int(np.argmax(nan_mask))]) if nan_mask.any() else None

    finite_mask = ~nan_mask
    if finite_mask.sum() >= 2:
        finite_vals = e_vals[finite_mask]
        diffs = np.diff(finite_vals)
        is_monotone = bool(np.all(diffs >= -monotone_tol))
    else:
        is_monotone = True

    return ExcessEntropyCurveResult(
        kappa_grid=np.asarray(kappa_grid, dtype=np.float64),
        tau=float(tau),
        excess_entropy=e_vals,
        is_monotone=is_monotone,
        kappa_star_estimated=kappa_star,
    )


# ---------------------------------------------------------------------------
# Empirical transfer entropy — Schreiber (2000) directional information flow,
# with IAAFT-surrogate null calibration. Used in paper §3.10 Corollary
# (Phase-4 prediction): T_{G -> r} should be statistically significant under
# IAAFT on calibrated SPX windows.
# ---------------------------------------------------------------------------


def _bin_series(
    x: NDArray[np.float64],
    n_bins: int,
    *,
    bin_edges: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Equiquantile-bin a 1D series into `n_bins` integer levels.

    Equiquantile binning is robust to heavy tails (every bin has ~equal
    occupancy, so the empirical histogram is well-conditioned) and is the
    standard practice for empirical transfer entropy on financial returns.

    Returns (binned_int_codes, bin_edges).
    """
    if bin_edges is None:
        # Equiquantile bins.
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(x, quantiles)
        # Stretch the outermost edges by epsilon so np.digitize captures them.
        eps = 1e-12 * max(abs(bin_edges[-1] - bin_edges[0]), 1.0)
        bin_edges[0] -= eps
        bin_edges[-1] += eps
    codes = np.digitize(x, bin_edges[1:-1], right=False)
    return codes.astype(np.int64), bin_edges


def transfer_entropy_simulated(
    source_series: NDArray[np.float64],
    target_series: NDArray[np.float64],
    *,
    lag: int = 1,
    n_bins: int = 8,
) -> float:
    """Empirical transfer entropy T_{source -> target} (Schreiber 2000).

    Computes

        T_{X -> Y} = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

    via plug-in histogram estimation on equiquantile-binned codes. Lag-1 by
    default — the canonical "next-step" predictive directional MI.

    Args:
        source_series: 1D source X (e.g., the dealer-gamma series G_t).
        target_series: 1D target Y (e.g., the spot log-return r_t).
        lag: number of steps the source leads the target prediction by.
            lag = 1 is the standard "X_{t-1} predicts Y_t given Y_{t-1}".
        n_bins: number of equiquantile bins for both series.

    Returns:
        Empirical T_{source -> target} in nats. A non-zero value indicates
        the source's history adds information about Y_t beyond what is
        already in Y's own history. Schreiber's plug-in estimator is biased
        towards positive values on finite samples; calibrate against a
        surrogate null via `transfer_entropy_iaaft_pvalue` for inference.

    Raises:
        ValueError: on incompatible shapes or non-positive lag / n_bins.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    x = np.asarray(source_series, dtype=np.float64).ravel()
    y = np.asarray(target_series, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"source/target shapes differ: {x.shape} vs {y.shape}")
    if len(x) <= lag + 1:
        raise ValueError(f"series too short for lag {lag}: len = {len(x)}")

    x_codes, _ = _bin_series(x, n_bins)
    y_codes, _ = _bin_series(y, n_bins)

    # Form the joint sample (Y_t, Y_{t-1}, X_{t-lag}).
    y_t = y_codes[lag:]
    y_prev = y_codes[lag - 1 : -1] if lag >= 1 else y_codes[lag:]  # = y_codes[lag-1:-1]
    x_past = x_codes[:-lag]
    # Truncate to common length (defensive).
    n_samples = min(len(y_t), len(y_prev), len(x_past))
    y_t = y_t[:n_samples]
    y_prev = y_prev[:n_samples]
    x_past = x_past[:n_samples]

    # Joint histograms.
    p_yt_yp_xp, _ = np.histogramdd(
        np.stack([y_t, y_prev, x_past], axis=-1).astype(np.int64),
        bins=(n_bins, n_bins, n_bins),
        range=((0, n_bins), (0, n_bins), (0, n_bins)),
    )
    p_yt_yp_xp = p_yt_yp_xp / max(p_yt_yp_xp.sum(), 1.0)
    p_yp_xp = p_yt_yp_xp.sum(axis=0)
    p_yt_yp = p_yt_yp_xp.sum(axis=2)
    p_yp = p_yt_yp.sum(axis=0)

    # T_{X->Y} = Σ p(y_t, y_p, x_p) log [ p(y_t | y_p, x_p) / p(y_t | y_p) ]
    te = 0.0
    for iy in range(n_bins):
        for ip in range(n_bins):
            for ix in range(n_bins):
                pjoint = p_yt_yp_xp[iy, ip, ix]
                if pjoint <= 0.0:
                    continue
                num = pjoint / p_yp_xp[ip, ix] if p_yp_xp[ip, ix] > 0.0 else 0.0
                den = p_yt_yp[iy, ip] / p_yp[ip] if p_yp[ip] > 0.0 else 0.0
                if num <= 0.0 or den <= 0.0:
                    continue
                te += pjoint * float(np.log(num / den))
    return float(te)


@dataclass(frozen=True)
class TransferEntropyIAAFTResult:
    """Output of `transfer_entropy_iaaft_pvalue`.

    Attributes:
        observed: T_{source -> target} on the original series (nats).
        surrogate_quantile_95: 95th percentile of surrogate T values (the
            informal one-sided critical value at alpha = 0.05).
        p_value: Phipson-Smyth corrected empirical p-value
            (1 + #{surrogate_T >= observed}) / (1 + n_surrogates).
        n_surrogates: number of IAAFT draws used for the null calibration.
    """

    observed: float
    surrogate_quantile_95: float
    p_value: float
    n_surrogates: int


def transfer_entropy_iaaft_pvalue(
    source_series: NDArray[np.float64],
    target_series: NDArray[np.float64],
    *,
    lag: int = 1,
    n_bins: int = 8,
    n_surrogates: int = 200,
    rng: np.random.Generator | None = None,
) -> TransferEntropyIAAFTResult:
    """Empirical transfer entropy with IAAFT-null p-value.

    The IAAFT surrogate (Schreiber-Schmitz 1996) preserves the source's
    marginal distribution AND its linear autocorrelation, randomising only the
    nonlinear cross-coupling structure between source and target. This is the
    appropriate null for "is the dealer-gamma feedback channel statistically
    informative beyond what its own linear autocorrelation predicts?", and
    matches the H4 PSD-peak detector's null (paper pre-registration
    amendment A5).

    Args:
        source_series: 1D source (e.g. calibrated G_t over an event window).
        target_series: 1D target (e.g. observed log-returns r_t).
        lag: prediction lag in steps (default 1).
        n_bins: equiquantile bin count for the plug-in TE estimator.
        n_surrogates: IAAFT replicate count.
        rng: numpy Generator for reproducible surrogates.

    Returns:
        TransferEntropyIAAFTResult with the observed TE, surrogate 95th
        percentile, and the Phipson-Smyth-corrected p-value.
    """
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates must be >= 1, got {n_surrogates}")
    if rng is None:
        rng = np.random.default_rng()

    observed = transfer_entropy_simulated(source_series, target_series, lag=lag, n_bins=n_bins)

    surrogate_te = np.zeros(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        # Surrogate the SOURCE (preserves source's marginal + linear ACF;
        # destroys nonlinear cross-coupling to target). This matches the
        # standard convention in the directional-information-flow literature
        # (Schreiber 2000 §IV, Lizier 2014 §3.3).
        source_surr = iaaft_surrogate(source_series, rng=rng)
        surrogate_te[i] = transfer_entropy_simulated(
            source_surr, target_series, lag=lag, n_bins=n_bins
        )

    n_ge = int(np.sum(surrogate_te >= observed))
    # Phipson-Smyth (2010) +1/+1 correction — strictly positive p.
    p_value = (1.0 + n_ge) / (1.0 + n_surrogates)
    q95 = float(np.quantile(surrogate_te, 0.95))

    return TransferEntropyIAAFTResult(
        observed=float(observed),
        surrogate_quantile_95=q95,
        p_value=float(p_value),
        n_surrogates=int(n_surrogates),
    )


# ---------------------------------------------------------------------------
# Critical-exponent fit at the boundary — the Theorem 5 (c) numerical anchor.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticalExponentFit:
    """Output of `fit_critical_exponent`.

    The fit assumes the mean-field saturation ansatz

        E_tau(kappa) = E_inf - C * (kappa^* - kappa)^beta + O((kappa^* - kappa)^(beta+1))

    near kappa^*. Theorem 5(c) predicts beta = 1 (linear approach), so the
    fit returns the empirical beta and the saturation E_inf for sanity-checking
    against the prediction.

    Attributes:
        beta: fitted critical exponent. The Theorem 5(c) mean-field prediction
            is beta = 1.
        e_infinity: fitted saturation value E_tau(kappa^*).
        coefficient: fitted C in the ansatz.
        residual_std: residual standard deviation of the fit (nats).
        n_points: number of (kappa, E) points used in the fit.
    """

    beta: float
    e_infinity: float
    coefficient: float
    residual_std: float
    n_points: int


def fit_critical_exponent(
    jacobian_at: Callable[[float], NDArray[np.float64]],
    diffusion_outer: NDArray[np.float64],
    tau: float,
    kappa_star: float,
    *,
    deltas: NDArray[np.float64] | None = None,
    observed_index: int = 0,
    conditioned_indices: tuple[int, ...] = (1, 2),
) -> CriticalExponentFit:
    """Fit the saturation ansatz E(κ) = E_inf - C·(κ★ - κ)^β near κ★.

    Performs a log-log linear regression of E_inf - E(κ★ - δ) on δ across
    a geometric grid of small δ values, after first extrapolating E_inf via
    Richardson-style limit at δ → 0. Theorem 5(c) predicts β = 1
    (mean-field linear approach to the saturation), so the fit returns the
    empirical β for direct comparison.

    The fit uses the `jacobian_at` callable directly (not a precomputed grid)
    so we can sample at arbitrarily small δ, where the linear regime is
    cleanly resolved. The default δ-grid is a logarithmic sweep from 1e-2
    down to 1e-5 (4 decades, 10 points).

    Args:
        jacobian_at: callable κ ↦ J(κ).
        diffusion_outer: Σ Σ^T (kappa-independent under the constant-vol
            surrogate).
        tau: time horizon (years).
        kappa_star: the Hopf threshold (anchors the boundary fit).
        deltas: explicit δ grid (κ - κ★ values, all negative when subtracted
            from κ★). Default = np.geomspace(1e-2, 1e-5, 10).
        observed_index, conditioned_indices: see `excess_entropy_linear`.

    Returns:
        CriticalExponentFit. Mean-field prediction: β ≈ 1.

    Raises:
        ValueError: if the δ-grid produces fewer than 3 valid finite E_tau
            values.
    """
    if deltas is None:
        deltas = np.geomspace(1e-2, 1e-5, 10)
    deltas = np.asarray(deltas, dtype=np.float64)
    if np.any(deltas <= 0.0):
        raise ValueError("deltas must be strictly positive")

    e_vals = np.array(
        [
            excess_entropy_linear(
                jacobian_at(kappa_star - float(d)),
                diffusion_outer,
                tau,
                observed_index=observed_index,
                conditioned_indices=conditioned_indices,
            )
            for d in deltas
        ],
        dtype=np.float64,
    )
    finite = ~np.isnan(e_vals)
    d_fit = deltas[finite]
    e_fit = e_vals[finite]
    if len(d_fit) < 3:
        raise ValueError(
            f"insufficient finite excess-entropy values for fit: {len(d_fit)} of "
            f"{len(deltas)}; check that κ★ - max(deltas) lies inside the Hurwitz region"
        )

    # Extrapolate E_inf via a very small δ (3 orders of magnitude below the
    # smallest fit-grid δ). This is well inside the mean-field linear regime.
    d_extrap = float(d_fit.min()) * 1e-3
    e_inf_kappa = kappa_star - d_extrap
    e_inf = excess_entropy_linear(
        jacobian_at(e_inf_kappa),
        diffusion_outer,
        tau,
        observed_index=observed_index,
        conditioned_indices=conditioned_indices,
    )
    if not np.isfinite(e_inf):
        # Fall back to the largest-stable κ in the fit window.
        e_inf = float(e_fit[np.argmin(d_fit)]) + 1e-12

    gap = e_inf - e_fit
    # Drop any nonpositive gap entries (numerical noise can push them across zero
    # at the smallest δ).
    valid = gap > 0.0
    if valid.sum() < 3:
        raise ValueError(
            f"after dropping nonpositive-gap entries only {int(valid.sum())} "
            f"points remain; E_inf extrapolation may be unreliable"
        )
    log_gap = np.log(gap[valid])
    log_d = np.log(d_fit[valid])
    slope, intercept = np.polyfit(log_d, log_gap, 1)
    beta = float(slope)
    coefficient = float(np.exp(intercept))
    predicted = e_inf - coefficient * (d_fit[valid]) ** beta
    residual_std = float(np.std(e_fit[valid] - predicted))

    return CriticalExponentFit(
        beta=beta,
        e_infinity=float(e_inf),
        coefficient=coefficient,
        residual_std=residual_std,
        n_points=int(valid.sum()),
    )
