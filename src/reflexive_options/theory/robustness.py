"""Robustness of the closed-form Hopf threshold $\\kappa^\\star$ (paper §4.3 Eq. 18)
to OI distribution misspecification.

Two complementary robustness results:

    1. **Analytical sensitivity** of $\\kappa^\\star$ to the log-normal OI
       parameters $(\\mu_q, \\sigma_q)$, via implicit differentiation of the
       quadratic Routh–Hurwitz polynomial $H(\\kappa) = A_2\\kappa^2 + A_1\\kappa + A_0$
       evaluated at the closed-form root. Returns the elasticities
       $\\eta_{\\sigma_q} := (\\sigma_q/\\kappa^\\star)\\,\\partial\\kappa^\\star/\\partial\\sigma_q$
       and $\\eta_{\\mu_q}$ at the canonical regime.

    2. **Mis-specification error** when the true OI is a mixture of log-normals
       (a synthetic SPX-like bimodal grid) but the closed-form $\\kappa^\\star$
       is computed from a single log-normal $\\mathcal{N}(\\hat\\mu_q, \\hat\\sigma_q^2)$
       fitted to the true mixture. The "true" $\\kappa^\\star$ is computed via
       the FD-tensor pipeline applied directly to the mixture density (no
       log-normal assumption). Returns the relative error
       $|\\kappa^\\star_{\\text{cf}} - \\kappa^\\star_{\\text{true}}| / \\kappa^\\star_{\\text{true}}$
       as a function of mixture-separation severity.

Used by `experiments/kappa_star_robustness.py` and `tests/test_kappa_star_robustness.py`.
Writeup at `paper/kappa_star_robustness.md` (and `paper/theory.md` §4.3.6).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from reflexive_options.theory.bifurcation import (
    G_lognormal_oi_partials,
    hopf_scan,
    jacobian_3d,
    kappa_star_lognormal_oi,
)

__all__ = [
    "KappaStarSensitivityResult",
    "MisspecificationError",
    "fit_lognormal_to_mixture_moments",
    "kappa_star_brute_force_from_G",
    "kappa_star_misspecification_error",
    "kappa_star_sensitivity_lognormal_oi",
    "make_mixture_lognormal_density",
]


# ---------------------------------------------------------------------------
# Deliverable 1: analytical sensitivity ∂κ★/∂(σ_q, μ_q)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KappaStarSensitivityResult:
    """Output of `kappa_star_sensitivity_lognormal_oi`.

    All quantities evaluated at the supplied (canonical) parameter point.
    """

    kappa_star: float
    omega_star: float
    G_y: float
    G_v: float
    # Implicit-function partials d κ★ / dG_y, dG_v
    dkappa_dGy: float
    dkappa_dGv: float
    # Outer chain-rule partials of G_y, G_v wrt OI parameters (μ_q, σ_q),
    # evaluated by central FD on the analytic G_lognormal_oi_partials machinery.
    dGy_dmu: float
    dGy_dsigma: float
    dGv_dmu: float
    dGv_dsigma: float
    # Final partials of κ★ wrt (μ_q, σ_q)
    dkappa_dmu_q: float
    dkappa_dsigma_q: float
    # Elasticities (dimensionless)
    elasticity_mu_q: float  # (μ_q / κ★) * ∂κ★/∂μ_q     — note μ_q can be 0; we report
    #                                                     the absolute partial too
    elasticity_sigma_q: float  # (σ_q / κ★) * ∂κ★/∂σ_q
    # "Per-percent-misspec" multipliers: these are the practical Phase-4 numbers.
    # %-change in κ★ per 1% increase in σ_q:
    pct_dkappa_per_pct_sigma_q: float  # = elasticity_sigma_q
    # %-change in κ★ per absolute log-strike unit shift in μ_q:
    pct_dkappa_per_unit_mu_q: float  # = (1/κ★) * ∂κ★/∂μ_q · 100


def _G_partials_wrapper(
    *,
    mu_q: float,
    sigma_q: float,
    a_star: float,
    v_star: float,
    T_eff: float,
    coupling_units: float,
    rate: float,
    dividend: float,
) -> tuple[float, float]:
    """Return (G_y, G_v) at the given OI parameters via the closed-form partials."""
    p = G_lognormal_oi_partials(
        a_star=a_star,
        v_star=v_star,
        mu_q=mu_q,
        sigma_q=sigma_q,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    return float(p["G_a"]), float(p["G_v"])


def kappa_star_sensitivity_lognormal_oi(
    *,
    mu_q: float,
    sigma_q: float,
    T_eff: float,
    kappa_v: float,
    theta_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    a_star: float,
    v_star: float | None = None,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    fd_step_mu: float = 1e-5,
    fd_step_sigma: float = 1e-5,
) -> KappaStarSensitivityResult:
    """Analytical sensitivity of $\\kappa^\\star$ to the log-normal OI parameters.

    Uses implicit differentiation of $H(\\kappa^\\star; G_y, G_v) = 0$ to get
    $\\partial\\kappa^\\star/\\partial G_\\bullet$ in closed form, then central FD
    on the *closed-form* $G$-partials to get $\\partial G_\\bullet/\\partial(\\mu_q, \\sigma_q)$
    — the FD step is on a smooth analytic function so the resulting partials
    are accurate to O(fd_step^2) without truncation/roundoff issues until
    fd_step ~ 1e-7.

    The implicit-function pieces (with $A := \\alpha + \\kappa_v$, $L := \\beta\\gamma$):

        $H = A_2 \\kappa^2 + A_1 \\kappa + A_0$,
        $A_2 = G_y^2 A,\\; A_1 = G_v L - G_y A^2,\\; A_0 = \\alpha\\kappa_v A - L/2$,
        $\\partial H / \\partial \\kappa = 2 A_2 \\kappa + A_1$,
        $\\partial H / \\partial G_y = 2 G_y A \\kappa^2 - A^2 \\kappa$,
        $\\partial H / \\partial G_v = L \\kappa$,
        $\\partial \\kappa^\\star / \\partial G_y = -\\frac{2 G_y A \\kappa^{\\star 2} - A^2 \\kappa^\\star}{2 A_2 \\kappa^\\star + A_1}$,
        $\\partial \\kappa^\\star / \\partial G_v = -\\frac{L \\kappa^\\star}{2 A_2 \\kappa^\\star + A_1}$.

    Args (all canonical-regime defaults assumed by caller):
        mu_q, sigma_q, T_eff: log-normal OI params and representative maturity.
        kappa_v, theta_v, alpha, beta, gamma: SDE parameters (paper §1).
        a_star, v_star: equilibrium location. v_star defaults to theta_v.
        coupling_units, rate, dividend: G-aggregator constants.
        fd_step_mu, fd_step_sigma: central-difference step sizes for the outer
            chain-rule derivatives. Default 1e-5 — at this scale, central-FD
            truncation error is O(h^2) ≲ 1e-10 and double-precision roundoff
            ε/h is ≲ 1e-11 for the smooth analytic G(μ_q, σ_q) map (verified
            via a Richardson extrapolation scan in the test suite).

    Returns:
        KappaStarSensitivityResult bundling κ★, the implicit-function partials,
        the outer-FD partials, the chain-rule partials, and the dimensionless
        elasticities $\\eta_{\\mu_q}, \\eta_{\\sigma_q}$ for direct integration into
        the Phase-4 calibration tolerance equation.

    Raises:
        ValueError if no positive Hopf root exists at the canonical point or if
        the sensitivity denominator $2 A_2 \\kappa^\\star + A_1$ is zero
        (degenerate Bautin-like regime — caller should treat this as "infinite
        sensitivity" and re-pose the question).
    """
    if v_star is None:
        v_star = theta_v
    if fd_step_mu <= 0 or fd_step_sigma <= 0:
        raise ValueError("FD step sizes must be > 0")

    # Step 1 — closed-form κ★ at the canonical point.
    G_y, G_v = _G_partials_wrapper(
        mu_q=mu_q,
        sigma_q=sigma_q,
        a_star=a_star,
        v_star=v_star,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    kappa_star, omega_star = kappa_star_lognormal_oi(
        G_y=G_y, G_v=G_v, kappa_v=kappa_v, alpha=alpha, beta=beta, gamma=gamma
    )

    # Step 2 — implicit-function partials d κ★ / dG_•.
    A = alpha + kappa_v
    L = beta * gamma
    A2 = G_y * G_y * A
    A1 = G_v * L - G_y * A * A

    denom = 2.0 * A2 * kappa_star + A1
    if abs(denom) < 1e-300:
        raise ValueError(
            f"sensitivity denominator dH/dκ = {denom:.3e} ≈ 0 at κ★ = {kappa_star}; "
            "degenerate Bautin-like point — sensitivity is unbounded"
        )

    dHdGy = 2.0 * G_y * A * kappa_star**2 - A * A * kappa_star
    dHdGv = L * kappa_star

    dkappa_dGy = -dHdGy / denom
    dkappa_dGv = -dHdGv / denom

    # Step 3 — outer chain-rule partials ∂G_y/∂μ_q, ∂G_y/∂σ_q, etc., via
    # central FD on the analytic G_lognormal_oi_partials. The function is
    # smooth in (μ_q, σ_q) so a 4th-order accurate Richardson sweep is
    # unnecessary; standard 2nd-order central FD is good to 1e-8 for step 1e-4.
    Gy_pm, Gv_pm = _G_partials_wrapper(
        mu_q=mu_q + fd_step_mu,
        sigma_q=sigma_q,
        a_star=a_star,
        v_star=v_star,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    Gy_mm, Gv_mm = _G_partials_wrapper(
        mu_q=mu_q - fd_step_mu,
        sigma_q=sigma_q,
        a_star=a_star,
        v_star=v_star,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    Gy_ps, Gv_ps = _G_partials_wrapper(
        mu_q=mu_q,
        sigma_q=sigma_q + fd_step_sigma,
        a_star=a_star,
        v_star=v_star,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    Gy_ms, Gv_ms = _G_partials_wrapper(
        mu_q=mu_q,
        sigma_q=sigma_q - fd_step_sigma,
        a_star=a_star,
        v_star=v_star,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    dGy_dmu = (Gy_pm - Gy_mm) / (2.0 * fd_step_mu)
    dGv_dmu = (Gv_pm - Gv_mm) / (2.0 * fd_step_mu)
    dGy_dsigma = (Gy_ps - Gy_ms) / (2.0 * fd_step_sigma)
    dGv_dsigma = (Gv_ps - Gv_ms) / (2.0 * fd_step_sigma)

    # Step 4 — chain rule.
    dkappa_dmu_q = dkappa_dGy * dGy_dmu + dkappa_dGv * dGv_dmu
    dkappa_dsigma_q = dkappa_dGy * dGy_dsigma + dkappa_dGv * dGv_dsigma

    elasticity_mu_q = (mu_q / kappa_star) * dkappa_dmu_q if mu_q != 0.0 else 0.0
    elasticity_sigma_q = (sigma_q / kappa_star) * dkappa_dsigma_q

    return KappaStarSensitivityResult(
        kappa_star=float(kappa_star),
        omega_star=float(omega_star),
        G_y=float(G_y),
        G_v=float(G_v),
        dkappa_dGy=float(dkappa_dGy),
        dkappa_dGv=float(dkappa_dGv),
        dGy_dmu=float(dGy_dmu),
        dGy_dsigma=float(dGy_dsigma),
        dGv_dmu=float(dGv_dmu),
        dGv_dsigma=float(dGv_dsigma),
        dkappa_dmu_q=float(dkappa_dmu_q),
        dkappa_dsigma_q=float(dkappa_dsigma_q),
        elasticity_mu_q=float(elasticity_mu_q),
        elasticity_sigma_q=float(elasticity_sigma_q),
        pct_dkappa_per_pct_sigma_q=float(elasticity_sigma_q),
        pct_dkappa_per_unit_mu_q=float(100.0 * dkappa_dmu_q / kappa_star),
    )


# ---------------------------------------------------------------------------
# Deliverables 2/3: numerical scan + multi-modal misspecification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MisspecificationError:
    """Result of comparing closed-form κ★ on a fitted log-normal OI to the
    "true" κ★ computed via FD-Jacobian on a multi-modal density."""

    separation: float  # the mixture separation parameter (e.g. modal split in log-strike)
    mu_hat: float
    sigma_hat: float
    kappa_star_closed_form: float
    kappa_star_true: float
    relative_error: float


def make_mixture_lognormal_density(
    *,
    mu_components: Sequence[float] | NDArray[np.float64],
    sigma_components: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
) -> Callable[[float], float]:
    """Return q(log K) — a finite mixture of Gaussians in log-strike.

    Args:
        mu_components: list of component means (log-strike units).
        sigma_components: list of component std-devs (log-strike units), each > 0.
        weights: non-negative mixing weights summing to 1.

    Returns:
        A python callable q(ell) computing the mixture density at log-strike ell.
    """
    mu = np.asarray(mu_components, dtype=np.float64)
    sg = np.asarray(sigma_components, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if mu.shape != sg.shape or mu.shape != w.shape:
        raise ValueError("mu_components, sigma_components, weights must have same length")
    if np.any(sg <= 0.0):
        raise ValueError(f"all sigma_components must be > 0, got {sg}")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be ≥ 0, got {w}")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to > 0")
    w_norm = w / total

    inv2pi = 1.0 / np.sqrt(2.0 * np.pi)

    def q(ell: float) -> float:
        z = (ell - mu) / sg
        return float(np.sum(w_norm * inv2pi * np.exp(-0.5 * z * z) / sg))

    return q


def fit_lognormal_to_mixture_moments(
    *,
    mu_components: Sequence[float] | NDArray[np.float64],
    sigma_components: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
) -> tuple[float, float]:
    """Fit a single log-normal $\\mathcal{N}(\\hat\\mu, \\hat\\sigma^2)$ to a mixture
    by matching the first two moments (in log-strike).

    For a mixture with components $(\\mu_i, \\sigma_i^2)$ and weights $w_i$:

        $\\hat\\mu = \\sum_i w_i \\mu_i$,
        $\\hat\\sigma^2 = \\sum_i w_i (\\sigma_i^2 + \\mu_i^2) - \\hat\\mu^2$
                  $= \\sum_i w_i \\sigma_i^2 + \\sum_i w_i (\\mu_i - \\hat\\mu)^2$.

    Returns (mu_hat, sigma_hat).
    """
    mu = np.asarray(mu_components, dtype=np.float64)
    sg = np.asarray(sigma_components, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if mu.shape != sg.shape or mu.shape != w.shape:
        raise ValueError("inputs must have matching shapes")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be ≥ 0, got {w}")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to > 0")
    w_norm = w / total

    mu_hat = float(np.sum(w_norm * mu))
    var_hat = float(np.sum(w_norm * sg * sg) + np.sum(w_norm * (mu - mu_hat) ** 2))
    if var_hat <= 0.0:
        raise ValueError(f"fitted variance non-positive: {var_hat:.3e}")
    return mu_hat, float(np.sqrt(var_hat))


def _G_from_density(
    *,
    a: float,
    v: float,
    q_density: Callable[[float], float],
    T_eff: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    integration_halfwidth: float = 8.0,
    log_strike_center: float = 0.0,
    integration_limit: int = 200,
) -> float:
    """Compute G(a, v) by direct quadrature for an arbitrary OI density q(log K).

    Mirrors the structure of `tests/test_lognormal_lyapunov.py::_numerical_G`
    but takes a generic density callable rather than assuming log-normal form.

    The integration window is centred on `log_strike_center` (defaults to 0;
    pass mu_q for the log-normal case) with half-width `integration_halfwidth`
    log-strike units — wide enough to capture all relevant mass for the
    canonical regime (σ_q ~ 0.1–0.3).
    """
    if v <= 0.0:
        raise ValueError(f"v must be > 0, got {v}")
    if T_eff <= 0.0:
        raise ValueError(f"T_eff must be > 0, got {T_eff}")
    sigma = float(np.sqrt(v))
    sqrt_T = float(np.sqrt(T_eff))
    spot = float(np.exp(a))

    def integrand(ell: float) -> float:
        K = float(np.exp(ell))
        d1 = (np.log(spot / K) + (rate - dividend + 0.5 * sigma * sigma) * T_eff) / (sigma * sqrt_T)
        gamma_bs = (
            float(np.exp(-dividend * T_eff))
            * float(np.exp(-0.5 * d1 * d1))
            / float(np.sqrt(2.0 * np.pi))
            / (spot * sigma * sqrt_T)
        )
        return q_density(ell) * gamma_bs

    val, _ = quad(
        integrand,
        log_strike_center - integration_halfwidth,
        log_strike_center + integration_halfwidth,
        limit=integration_limit,
    )
    return float(coupling_units * val)


def kappa_star_brute_force_from_G(
    *,
    G_func: Callable[[float, float], float],
    a_star: float,
    v_star: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    fd_step_a: float = 5e-4,
    fd_step_v: float = 5e-4,
    kappa_grid: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """Compute κ★ via the deterministic-skeleton Jacobian for an arbitrary G(a, v).

    Procedure:
        1. Numerically differentiate G at (a*, v*) via central FD to get G_y, G_v.
           (G is smooth — quadrature integral of two Gaussians times any density.)
        2. Build the 3×3 Jacobian (3) of paper/theory.md, recognising that
           σ² = v ⇒ b(κ) = κ G_v − 1/2, a(κ) = κ G_y, G_z = 0.
        3. Either: (i) solve the closed-form quadratic in κ given (G_y, G_v),
           OR (ii) call hopf_scan to bracket the bifurcation. We use (i) since
           the quadratic is *exact* once (G_y, G_v) are computed — the only
           approximation is the FD on G itself, which is O(fd_step^2) on a
           smooth integrand.

    This is the "true" κ★ for an arbitrary OI density (no log-normal assumption).
    The hopf_scan path is exposed via the `kappa_grid` argument as a sanity check.

    Returns:
        (kappa_star, omega_star).

    Raises:
        ValueError if no Hopf threshold exists for this G.
    """
    G_y_num = (G_func(a_star + fd_step_a, v_star) - G_func(a_star - fd_step_a, v_star)) / (
        2.0 * fd_step_a
    )
    G_v_num = (G_func(a_star, v_star + fd_step_v) - G_func(a_star, v_star - fd_step_v)) / (
        2.0 * fd_step_v
    )

    kappa_star_qf, omega_star = kappa_star_lognormal_oi(
        G_y=float(G_y_num),
        G_v=float(G_v_num),
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    # Optional double-check via numerical hopf_scan on the same Jacobian — the two
    # should agree to root-finding precision because the analytic quadratic is
    # exact for σ² = v Heston backbone with G_z = 0 (the only structure that
    # matters for H(κ)).
    if kappa_grid is not None:

        def jac(k: float) -> NDArray[np.float64]:
            return jacobian_3d(
                kappa=k,
                a_kappa=k * float(G_y_num),
                b_kappa=k * float(G_v_num) - 0.5,
                G_z=0.0,
                kappa_v=kappa_v,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

        scan = hopf_scan(np.asarray(kappa_grid, dtype=np.float64), jac)
        # Use the brentq-located one if the closed-form quadratic disagrees
        # (will only happen in pathological numerics — fall back to the more
        # robust eigenvalue-crossing root).
        if scan.kappa_star is not None and not np.isclose(
            scan.kappa_star, kappa_star_qf, rtol=1e-3
        ):
            return float(scan.kappa_star), float(scan.omega_at_crossing or omega_star)
    return float(kappa_star_qf), float(omega_star)


def kappa_star_misspecification_error(
    *,
    mu_components: Sequence[float] | NDArray[np.float64],
    sigma_components: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
    T_eff: float,
    kappa_v: float,
    theta_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    a_star: float,
    v_star: float | None = None,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    integration_halfwidth: float = 8.0,
) -> MisspecificationError:
    """Compute the relative error of the closed-form κ★ when the *true* OI is
    a mixture of log-normals but a single log-normal is used for calibration.

    Procedure:
        1. Fit a single log-normal $(\\hat\\mu, \\hat\\sigma)$ to the mixture by
           moment matching (`fit_lognormal_to_mixture_moments`).
        2. Closed-form $\\kappa^\\star_{\\text{cf}}$ from
           `kappa_star_lognormal_oi(G_y, G_v from analytic partials at the fitted
           log-normal)`.
        3. "True" $\\kappa^\\star_{\\text{true}}$ via `kappa_star_brute_force_from_G`
           where G is built from numerical quadrature against the actual mixture
           density.
        4. Report `(separation, mu_hat, sigma_hat, kappa_cf, kappa_true, rel_err)`.

    The "separation" returned is $(\\max\\mu - \\min\\mu)$ — a one-number summary
    of how non-log-normal the true OI is. For a single-mode mixture (all
    components at the same μ), separation = 0 and the closed form should be
    exact.
    """
    if v_star is None:
        v_star = theta_v

    mu = np.asarray(mu_components, dtype=np.float64)
    sg = np.asarray(sigma_components, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    mu_hat, sigma_hat = fit_lognormal_to_mixture_moments(
        mu_components=mu, sigma_components=sg, weights=w
    )

    # Closed-form κ★ at the fitted log-normal.
    G_y_cf, G_v_cf = _G_partials_wrapper(
        mu_q=mu_hat,
        sigma_q=sigma_hat,
        a_star=a_star,
        v_star=v_star,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    kappa_cf, _ = kappa_star_lognormal_oi(
        G_y=G_y_cf, G_v=G_v_cf, kappa_v=kappa_v, alpha=alpha, beta=beta, gamma=gamma
    )

    # True κ★ from the full mixture density.
    q_mix = make_mixture_lognormal_density(mu_components=mu, sigma_components=sg, weights=w)

    def G_mix(a: float, v: float) -> float:
        return _G_from_density(
            a=a,
            v=v,
            q_density=q_mix,
            T_eff=T_eff,
            coupling_units=coupling_units,
            rate=rate,
            dividend=dividend,
            integration_halfwidth=integration_halfwidth,
            log_strike_center=mu_hat,  # centre window on the fitted mean for tightest quadrature
        )

    kappa_true, _ = kappa_star_brute_force_from_G(
        G_func=G_mix,
        a_star=a_star,
        v_star=v_star,
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    rel_err = abs(kappa_cf - kappa_true) / kappa_true
    separation = float(mu.max() - mu.min())

    return MisspecificationError(
        separation=separation,
        mu_hat=float(mu_hat),
        sigma_hat=float(sigma_hat),
        kappa_star_closed_form=float(kappa_cf),
        kappa_star_true=float(kappa_true),
        relative_error=float(rel_err),
    )


# ---------------------------------------------------------------------------
# Calibration tolerance helper for Phase 4
# ---------------------------------------------------------------------------


def calibration_tolerance(
    sensitivity: KappaStarSensitivityResult,
    *,
    target_kappa_relative_error: float,
) -> dict[str, float]:
    """Convert a target |Δκ★|/κ★ budget into a tolerance budget for (μ_q, σ_q).

    Allocate the κ★ budget equally between the two parameters in quadrature
    (independent error sources):

        $(\\Delta\\kappa^\\star/\\kappa^\\star)^2 = \\eta_{\\sigma_q}^2 (\\Delta\\sigma_q/\\sigma_q)^2
                                            + (\\Delta\\mu_q \\cdot \\partial_\\mu \\kappa^\\star/\\kappa^\\star)^2$

    Allocating $\\frac{1}{\\sqrt 2}$ of the budget to each parameter:

        $|\\Delta\\sigma_q/\\sigma_q| = (\\text{budget}/\\sqrt{2}) / |\\eta_{\\sigma_q}|$
        $|\\Delta\\mu_q| = (\\text{budget}/\\sqrt{2}) / |\\partial_\\mu \\kappa^\\star/\\kappa^\\star|$
                                                       (in log-strike units).

    Returns a dict with keys:
        target_kappa_rel_err, sigma_q_rel_tol, mu_q_abs_tol_log_strike,
        sigma_q_pct_tol, mu_q_log_strike_tol.

    Used by the `paper/kappa_star_robustness.md` calibration table.
    """
    if target_kappa_relative_error <= 0.0:
        raise ValueError(f"target must be > 0, got {target_kappa_relative_error}")

    half_budget = target_kappa_relative_error / np.sqrt(2.0)
    eta_s = abs(sensitivity.elasticity_sigma_q)
    eta_mu_dimless = abs(sensitivity.dkappa_dmu_q / sensitivity.kappa_star)

    sigma_q_rel_tol = float(half_budget / eta_s) if eta_s > 0 else float("inf")
    mu_q_abs_tol = float(half_budget / eta_mu_dimless) if eta_mu_dimless > 0 else float("inf")

    return {
        "target_kappa_rel_err": float(target_kappa_relative_error),
        "sigma_q_rel_tol": sigma_q_rel_tol,
        "mu_q_abs_tol_log_strike": mu_q_abs_tol,
        "sigma_q_pct_tol": 100.0 * sigma_q_rel_tol,
        "mu_q_log_strike_tol": mu_q_abs_tol,
    }
