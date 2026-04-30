"""Fokker-Planck stationary marginal density of the reflexive simulator.

The 3D Fokker-Planck PDE for π(S, v, z, t) admits no general closed-form
stationary solution. We compute π*(S) numerically via Monte-Carlo: simulate
long trajectories past mixing time, kernel-density-estimate the marginal.

For analytical comparison to Heston, the variance under standard Heston has a
known stationary chi-squared / gamma density (Feller 1951): with parameters
(κ_v, θ_v, ξ),

    π*_Heston(v) = (2κ_v/ξ²)^a / Γ(a) · v^(a-1) · exp(-2κ_v v / ξ²),
    a = 2 κ_v θ_v / ξ².

The spot S has no stationary distribution under either measure (it drifts).
The proper comparator is therefore log-deviations from drift, which is what
this module standardises (mean-centring the log-spot samples) before any
direct cross-model comparison.

Pre-registered hypotheses (paper/theory.md §7):
    H_tail:   tail index of π*_reflexive(log S) ≤ tail index of π*_Heston (heavier).
    H_skew:   skew(π*_reflexive) sign tracks sign(G_x) at equilibrium.
    H_bimod:  approaching κ → κ*, π*_reflexive(log S) develops bimodality.

Implementation: see paper/theory.md §7 for the substantiated numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import diptest
import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats

from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.types import HestonParams, SimulatorProtocol

Component = Literal["log_spot", "variance", "memory"]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationaryDensity:
    """Empirical stationary marginal density estimate."""

    grid: NDArray[np.float64]  # 1D grid over the marginal variable
    density: NDArray[np.float64]  # KDE-estimated density on the grid
    samples: NDArray[np.float64]  # raw stationary samples (for moments / bootstrap)

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    @property
    def variance(self) -> float:
        return float(self.samples.var(ddof=1))

    @property
    def skewness(self) -> float:
        return float(stats.skew(self.samples))

    @property
    def excess_kurtosis(self) -> float:
        return float(stats.kurtosis(self.samples))

    def tail_index_hill(self, k_largest: int = 100) -> float:
        """Hill estimator of the right-tail index (Pareto-α). Higher = heavier tail.

        Returns α from F̄(x) ≈ x^{-α}: a *smaller* α means a *heavier* right tail.
        We work on |samples - median| to make the test scale-invariant and robust to
        sign of the marginal (e.g. centred log-returns).
        """
        if len(self.samples) < k_largest * 2:
            raise ValueError(
                f"need at least {2 * k_largest} samples for Hill estimator,"
                f" got {len(self.samples)}"
            )
        # Hill is for positive variates: take absolute deviations from the median
        x = np.abs(self.samples - float(np.median(self.samples)))
        x = x[x > 0]
        sorted_descending = np.sort(x)[::-1]
        top = sorted_descending[:k_largest]
        threshold = sorted_descending[k_largest]
        if threshold <= 0.0:
            raise ValueError("Hill threshold non-positive; samples too concentrated")
        return 1.0 / float(np.mean(np.log(top / threshold)))


# ---------------------------------------------------------------------------
# Reflexive (or any SimulatorProtocol) MC stationary density
# ---------------------------------------------------------------------------


def solve_stationary(
    sim: SimulatorProtocol,
    *,
    n_paths: int = 10_000,
    burn_in_steps: int = 50_000,
    sample_steps: int = 50_000,
    dt: float = 1.0 / (252 * 390),
    seed: int | None = 42,
    grid_size: int = 200,
    component: Component = "log_spot",
) -> StationaryDensity:
    """Estimate the stationary marginal density of `component` from long simulations.

    Args:
        sim: any SimulatorProtocol-compliant simulator.
        component: "log_spot" | "variance" | "memory".
        burn_in_steps: discarded as transient.
        sample_steps: kept for the density estimate.

    Returns:
        StationaryDensity with KDE on a uniform grid spanning [μ - 5σ, μ + 5σ].
    """
    total_steps = burn_in_steps + sample_steps
    spots, variances = sim.simulate(n_paths=n_paths, n_steps=total_steps, dt=dt, seed=seed)

    if component == "log_spot":
        samples = np.log(spots[:, burn_in_steps:].ravel())
    elif component == "variance":
        samples = variances[:, burn_in_steps:].ravel()
    elif component == "memory":
        raise NotImplementedError(
            "memory-variable extraction requires the 3D state from the simulator's"
            " `step` interface; use the per-path memory log directly"
        )
    else:
        raise ValueError(f"unknown component {component!r}")

    samples = samples[np.isfinite(samples)]
    mean, std = float(samples.mean()), float(samples.std(ddof=1))
    grid = np.linspace(mean - 5 * std, mean + 5 * std, grid_size).astype(np.float64)
    kde = stats.gaussian_kde(samples)
    density = np.asarray(kde(grid), dtype=np.float64)

    return StationaryDensity(grid=grid, density=density, samples=samples)


# ---------------------------------------------------------------------------
# Heston analytical helpers
# ---------------------------------------------------------------------------


def heston_stationary_variance_density(
    grid: NDArray[np.float64],
    params: HestonParams,
) -> NDArray[np.float64]:
    """Closed-form Feller stationary density of the variance under standard Heston.

    The CIR variance process

        dv = κ(θ - v) dt + ξ √v dW

    has stationary gamma density

        π*(v) = (2κ/ξ²)^a / Γ(a) · v^{a-1} · exp(-2κ v / ξ²),  a = 2κθ/ξ².

    Strictly positive on (0, ∞) iff Feller's condition 2κθ > ξ² holds (a > 1).
    For 0 < a ≤ 1 the density still integrates to 1 but blows up at zero.
    """
    if params.kappa <= 0.0 or params.theta <= 0.0 or params.xi <= 0.0:
        raise ValueError("Heston (κ, θ, ξ) must all be strictly positive")
    a = 2.0 * params.kappa * params.theta / (params.xi * params.xi)
    scale = params.xi * params.xi / (2.0 * params.kappa)
    return np.asarray(stats.gamma.pdf(grid, a=a, scale=scale), dtype=np.float64)


def _heston_log_return_cf(
    u: NDArray[np.complex128],
    *,
    params: HestonParams,
    dt: float,
    drift: float,
) -> NDArray[np.complex128]:
    """Heston characteristic function of the log-return X_τ = log(S_τ/S_0).

    Schoutens-trap / "Little Heston Trap" form (Albrecher, Mayer, Schoutens, Tistaert
    2007) — numerically stable across the branch cut. Risk-neutral drift = drift,
    initial variance v_0 = params.v0.
    """
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi * xi * (iu + u * u))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    exp_minus_d_dt = np.exp(-d * dt)
    C = drift * iu * dt + (kappa * theta / (xi * xi)) * (
        (kappa - rho * xi * iu - d) * dt
        - 2.0 * np.log((1.0 - g * exp_minus_d_dt) / (1.0 - g))
    )
    D = (
        (kappa - rho * xi * iu - d)
        / (xi * xi)
        * (1.0 - exp_minus_d_dt)
        / (1.0 - g * exp_minus_d_dt)
    )
    return np.asarray(np.exp(C + D * v0), dtype=np.complex128)


def heston_log_return_cdf(
    x: NDArray[np.float64],
    *,
    params: HestonParams,
    dt: float,
    drift: float = 0.0,
    u_max: float = 200.0,
) -> NDArray[np.float64]:
    """CDF of the Heston log-return at horizon `dt` via Gil-Pelaez inversion.

        F(x) = 1/2 - (1/π) ∫_0^∞ Im[ e^{-iux} φ(u) ] / u  du

    with φ the characteristic function of X_dt = log(S_dt / S_0) - drift_term.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x_arr)
    for i, xi_val in enumerate(x_arr):
        def integrand(u: float, xv: float = float(xi_val)) -> float:
            phi = _heston_log_return_cf(
                np.asarray([u], dtype=np.complex128),
                params=params,
                dt=dt,
                drift=drift,
            )[0]
            return float(np.imag(np.exp(-1j * u * xv) * phi) / u)

        val, _ = integrate.quad(integrand, 1e-8, u_max, limit=200)
        out[i] = 0.5 - val / np.pi
    return out


def heston_log_return_quantiles(
    params: HestonParams,
    dt: float,
    quantiles: NDArray[np.float64],
    *,
    drift: float = 0.0,
    bracket: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Quantiles of the Heston log-return at horizon `dt`.

    Inverts the Gil-Pelaez CDF by bisection. `quantiles` must lie strictly in (0, 1).
    """
    q = np.asarray(quantiles, dtype=np.float64)
    if not np.all((q > 0.0) & (q < 1.0)):
        raise ValueError("quantiles must lie strictly in (0, 1)")

    if bracket is None:
        sigma = float(np.sqrt(params.theta * dt))
        lo, hi = -8.0 * sigma + drift * dt, 8.0 * sigma + drift * dt
    else:
        lo, hi = bracket

    def cdf(x: float) -> float:
        return float(heston_log_return_cdf(np.array([x]), params=params, dt=dt, drift=drift)[0])

    out = np.empty_like(q)
    for i, qi in enumerate(q):
        a, b = lo, hi
        fa, fb = cdf(a) - float(qi), cdf(b) - float(qi)
        # Expand if needed
        n_expand = 0
        while fa * fb > 0.0 and n_expand < 6:
            a *= 2.0
            b *= 2.0
            fa, fb = cdf(a) - float(qi), cdf(b) - float(qi)
            n_expand += 1
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = cdf(m) - float(qi)
            if abs(fm) < 1e-7 or 0.5 * (b - a) < 1e-9:
                a = b = m
                break
            if fa * fm < 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        out[i] = 0.5 * (a + b)
    return out


# ---------------------------------------------------------------------------
# Bimodality detector (Hartigan's dip)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BimodalityResult:
    """Hartigan dip-test outcome."""

    dip_statistic: float
    p_value: float
    is_bimodal: bool  # True iff p_value < 0.05 (default Hartigan-style threshold)


def detect_bimodality(samples: NDArray[np.float64], *, alpha: float = 0.05) -> BimodalityResult:
    """Hartigan & Hartigan (1985) dip statistic + bootstrap p-value.

    H_0: samples drawn from a unimodal distribution.
    Reject (i.e. flag as multimodal) when p-value < `alpha`.
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        raise ValueError("dip test requires at least 4 finite samples")
    dip, pval = diptest.diptest(x)
    return BimodalityResult(
        dip_statistic=float(dip),
        p_value=float(pval),
        is_bimodal=bool(pval < alpha),
    )


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


def _heston_simulator_from_params(
    params: HestonParams,
    *,
    spot0: float,
    drift: float,
) -> SimulatorProtocol:
    """Build the standard HestonSimulator from a HestonParams set.

    Imported lazily to avoid a hard dependency on QuantLib at module load time
    (the analytic IV path in `baselines.heston` pulls QuantLib in transitively).
    """
    from reflexive_options.baselines.heston import HestonSimulator

    return HestonSimulator(
        regimes=[params],
        breakpoints=[],
        spot0=spot0,
        drift=drift,
    )


def compare_to_heston(
    reflexive_sim: ReflexiveSimulator,
    heston_params: HestonParams,
    *,
    dt: float = 1.0 / 252.0,
    n_paths: int = 4_000,
    burn_in_steps: int = 2_000,
    sample_steps: int = 4_000,
    hill_k: int = 100,
    seed: int | None = 7,
) -> dict[str, float]:
    """Compare reflexive vs Heston stationary marginal of log S.

    Procedure:
        1. Both simulators are run with matched (κ_v, θ_v, ξ, ρ, v_0) via
           `heston_params` (the *base* parameters of `reflexive_sim` should match
           or the comparison is apples-to-oranges; this is the caller's contract).
        2. Long simulation past `burn_in_steps`; centre log-spot samples at
           the empirical mean to remove drift (no martingale property under
           the physical measure).
        3. Compute mean / variance / skewness / excess kurtosis / Hill tail index.
        4. Anderson-Darling against the empirical Heston distribution as null.
        5. Return a dict of differences (reflexive - heston) and effect sizes.
    """
    drift = reflexive_sim.params.drift
    s0 = reflexive_sim.initial_spot

    refl_density = solve_stationary(
        reflexive_sim,
        n_paths=n_paths,
        burn_in_steps=burn_in_steps,
        sample_steps=sample_steps,
        dt=dt,
        seed=seed,
        component="log_spot",
    )
    heston_sim = _heston_simulator_from_params(heston_params, spot0=s0, drift=drift)
    heston_density = solve_stationary(
        heston_sim,
        n_paths=n_paths,
        burn_in_steps=burn_in_steps,
        sample_steps=sample_steps,
        dt=dt,
        seed=None if seed is None else seed + 1,
        component="log_spot",
    )

    # Centre both samples at their respective means (remove drift / S_0 anchor)
    refl_centred = refl_density.samples - refl_density.mean
    heston_centred = heston_density.samples - heston_density.mean

    refl_hill = StationaryDensity(
        grid=refl_density.grid,
        density=refl_density.density,
        samples=refl_centred,
    ).tail_index_hill(k_largest=hill_k)
    heston_hill = StationaryDensity(
        grid=heston_density.grid,
        density=heston_density.density,
        samples=heston_centred,
    ).tail_index_hill(k_largest=hill_k)

    # Two-sample Anderson-Darling: H_0 reflexive samples and Heston samples come
    # from the same distribution. We subsample to keep the computation tractable.
    rng = np.random.default_rng(0 if seed is None else seed)
    n_ad = min(2_000, refl_centred.size, heston_centred.size)
    ad = stats.anderson_ksamp(
        [
            rng.choice(refl_centred, size=n_ad, replace=False),
            rng.choice(heston_centred, size=n_ad, replace=False),
        ],
        variant="midrank",
    )

    return {
        "mean_reflexive": float(np.mean(refl_centred)),
        "mean_heston": float(np.mean(heston_centred)),
        "variance_reflexive": float(np.var(refl_centred, ddof=1)),
        "variance_heston": float(np.var(heston_centred, ddof=1)),
        "skewness_reflexive": float(stats.skew(refl_centred)),
        "skewness_heston": float(stats.skew(heston_centred)),
        "excess_kurtosis_reflexive": float(stats.kurtosis(refl_centred)),
        "excess_kurtosis_heston": float(stats.kurtosis(heston_centred)),
        "tail_index_reflexive": float(refl_hill),
        "tail_index_heston": float(heston_hill),
        "delta_variance": float(np.var(refl_centred, ddof=1) - np.var(heston_centred, ddof=1)),
        "delta_skewness": float(stats.skew(refl_centred) - stats.skew(heston_centred)),
        "delta_excess_kurtosis": float(
            stats.kurtosis(refl_centred) - stats.kurtosis(heston_centred)
        ),
        "delta_tail_index": float(refl_hill - heston_hill),
        "anderson_darling_statistic": float(ad.statistic),
        "anderson_darling_pvalue": float(ad.pvalue),
    }


# ---------------------------------------------------------------------------
# Tail-index curve vs κ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailIndexCurve:
    """Hill tail index of π*(log S) vs κ for the reflexive simulator."""

    kappa_grid: NDArray[np.float64]
    tail_indices: NDArray[np.float64]
    excess_kurtoses: NDArray[np.float64]


def tail_index_vs_kappa_curve(
    reflexive_factory: Callable[[float], ReflexiveSimulator],
    kappa_grid: NDArray[np.float64],
    *,
    dt: float = 1.0 / 252.0,
    n_paths: int = 2_000,
    burn_in_steps: int = 1_000,
    sample_steps: int = 4_000,
    hill_k: int = 100,
    seed: int | None = 11,
) -> TailIndexCurve:
    """Sweep κ and compute the Hill tail index + excess kurtosis at each level.

    Args:
        reflexive_factory: callable κ → ReflexiveSimulator. The caller is
            responsible for keeping all *other* parameters fixed across κ.
        kappa_grid: ascending sequence of κ values. κ = 0 reduces to Heston.

    Returns:
        TailIndexCurve over `kappa_grid`.
    """
    if not np.all(np.diff(kappa_grid) >= 0):
        raise ValueError("kappa_grid must be non-decreasing")

    n_k = len(kappa_grid)
    tails = np.zeros(n_k, dtype=np.float64)
    kurts = np.zeros(n_k, dtype=np.float64)

    for i, k in enumerate(kappa_grid):
        sim = reflexive_factory(float(k))
        density = solve_stationary(
            sim,
            n_paths=n_paths,
            burn_in_steps=burn_in_steps,
            sample_steps=sample_steps,
            dt=dt,
            seed=None if seed is None else seed + i,
            component="log_spot",
        )
        # Centre to remove drift
        centred = density.samples - density.mean
        tails[i] = StationaryDensity(
            grid=density.grid, density=density.density, samples=centred
        ).tail_index_hill(k_largest=hill_k)
        kurts[i] = float(stats.kurtosis(centred))

    return TailIndexCurve(
        kappa_grid=np.asarray(kappa_grid, dtype=np.float64),
        tail_indices=tails,
        excess_kurtoses=kurts,
    )
