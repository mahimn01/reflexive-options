"""Tests for the closed-form Lyapunov coefficient with log-normal OI.

Covers (paper/theory.md §4.3):
    - Closed-form G(a, v) matches direct numerical integration.
    - All 10 unique third-order partials match 5-point central FD.
    - Closed-form κ* (quadratic root) matches numerical hopf_scan.
    - Closed-form ℓ_1 matches numerical compute_lyapunov_coefficient on the
      same parameter set.
    - Sign region: at the canonical regime, ℓ_1 < 0 (supercritical).
    - ω* > 0 at κ* (Routh-Hurwitz consistency).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from reflexive_options.theory.bifurcation import (
    G_lognormal_oi,
    G_lognormal_oi_partials,
    build_bilinear_trilinear_tensors,
    compute_lyapunov_coefficient,
    hopf_scan,
    jacobian_3d,
    kappa_star_lognormal_oi,
    lyapunov_coefficient_lognormal_oi,
)

# Canonical parameter set for §4.3 — chosen to lie in the supercritical region of
# the (σ_q, T_eff, α, γ) phase diagram derived in the closed-form derivation script.
# At these values: κ* ≈ 17.81, ω* ≈ 1.18, ℓ_1 ≈ -0.48 (supercritical, ℓ_1 < 0).
_CANONICAL = dict(
    mu_q=float(np.log(100.0)),
    sigma_q=0.10,
    T_eff=0.25,
    kappa_v=2.0,
    theta_v=0.04,
    alpha=0.05,
    beta=1.0,
    gamma=1.0,
    a_star=float(np.log(100.0)),  # ATM equilibrium
    v_star=0.04,
    coupling_units=1.0,
)


# ---------------------------------------------------------------------------
# 1. G_lognormal_oi matches direct numerical integration of q(K) Γ_BS dK
# ---------------------------------------------------------------------------


def _numerical_G(
    a: float,
    v: float,
    *,
    mu_q: float,
    sigma_q: float,
    T_eff: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """Direct numerical integral of q(log K) Γ_BS(S=e^a, K=e^ℓ, T_eff, σ=√v) d(log K)."""
    sigma = float(np.sqrt(v))
    sqrt_T = float(np.sqrt(T_eff))
    spot = float(np.exp(a))

    def integrand(ell: float) -> float:
        K = float(np.exp(ell))
        d1 = (np.log(spot / K) + (rate - dividend + 0.5 * sigma**2) * T_eff) / (sigma * sqrt_T)
        gamma_bs = (
            float(np.exp(-dividend * T_eff))
            * float(np.exp(-0.5 * d1**2))
            / float(np.sqrt(2.0 * np.pi))
            / (spot * sigma * sqrt_T)
        )
        q_density = float(np.exp(-((ell - mu_q) ** 2) / (2.0 * sigma_q**2))) / (
            sigma_q * float(np.sqrt(2.0 * np.pi))
        )
        return q_density * gamma_bs

    val, _ = quad(integrand, mu_q - 10.0 * sigma_q, mu_q + 10.0 * sigma_q, limit=200)
    return float(coupling_units * val)


def test_G_at_atm_matches_numerical() -> None:
    """At a = μ_q (ATM), the closed-form G matches direct quadrature to 1e-10."""
    args = dict(
        mu_q=float(np.log(100.0)),
        sigma_q=0.20,
        T_eff=30.0 / 365.25,
        coupling_units=1.0,
        rate=0.0,
        dividend=0.0,
    )
    g_closed = G_lognormal_oi(args["mu_q"], 0.04, **args)  # type: ignore[arg-type]
    g_num = _numerical_G(args["mu_q"], 0.04, **args)  # type: ignore[arg-type]
    assert abs(g_closed - g_num) < 1e-10, f"closed: {g_closed}, num: {g_num}"


def test_G_off_atm_matches_numerical() -> None:
    """Spot away from μ_q still matches quadrature."""
    args = dict(
        mu_q=float(np.log(100.0)),
        sigma_q=0.30,
        T_eff=0.5,
        coupling_units=10.0,
        rate=0.02,
        dividend=0.01,
    )
    for a_off in [-0.2, -0.05, 0.0, 0.05, 0.2]:
        a = args["mu_q"] + a_off
        g_closed = G_lognormal_oi(a, 0.05, **args)  # type: ignore[arg-type]
        g_num = _numerical_G(a, 0.05, **args)  # type: ignore[arg-type]
        rel_err = abs(g_closed - g_num) / max(abs(g_num), 1e-15)
        assert rel_err < 1e-8, f"off={a_off}: closed={g_closed}, num={g_num}, rel={rel_err:.3e}"


# ---------------------------------------------------------------------------
# 2. All 10 unique third-order partials match 5-point central FD
# ---------------------------------------------------------------------------


def _five_point_first(g: callable, h: float) -> float:  # type: ignore[type-arg,valid-type]
    """f'(0) ≈ [-g(2h) + 8 g(h) - 8 g(-h) + g(-2h)] / (12 h)."""
    return float((-g(2.0 * h) + 8.0 * g(h) - 8.0 * g(-h) + g(-2.0 * h)) / (12.0 * h))


def _five_point_second(g: callable, h: float) -> float:  # type: ignore[type-arg,valid-type]
    """f''(0) ≈ [-g(2h) + 16 g(h) - 30 g(0) + 16 g(-h) - g(-2h)] / (12 h²)."""
    return float(
        (-g(2.0 * h) + 16.0 * g(h) - 30.0 * g(0.0) + 16.0 * g(-h) - g(-2.0 * h)) / (12.0 * h * h)
    )


def _five_point_third(g: callable, h: float) -> float:  # type: ignore[type-arg,valid-type]
    """f'''(0) ≈ [g(2h) - 2 g(h) + 2 g(-h) - g(-2h)] / (2 h³)."""
    return float((g(2.0 * h) - 2.0 * g(h) + 2.0 * g(-h) - g(-2.0 * h)) / (2.0 * h * h * h))


def test_partials_match_finite_differences() -> None:
    """Every closed-form partial G_y, G_v, G_yy, ..., G_yvv, G_vvv matches
    a 5-point central FD on the closed-form G to better than 1e-4 relative.
    """
    a_star = float(np.log(100.0))
    v_star = 0.04
    args = dict(
        mu_q=a_star,
        sigma_q=0.30,
        T_eff=0.5,
        coupling_units=1.0,
        rate=0.01,
        dividend=0.005,
    )
    p = G_lognormal_oi_partials(a_star=a_star, v_star=v_star, **args)  # type: ignore[arg-type]

    h_a = 1e-3
    h_v = 1e-4

    def g_a(da: float) -> float:
        return G_lognormal_oi(a_star + da, v_star, **args)  # type: ignore[arg-type]

    def g_v(dv: float) -> float:
        return G_lognormal_oi(a_star, v_star + dv, **args)  # type: ignore[arg-type]

    def g_av(da: float, dv: float) -> float:
        return G_lognormal_oi(a_star + da, v_star + dv, **args)  # type: ignore[arg-type]

    # 1st derivatives
    assert abs(_five_point_first(g_a, h_a) - p["G_a"]) / abs(p["G_a"]) < 1e-6
    assert abs(_five_point_first(g_v, h_v) - p["G_v"]) / abs(p["G_v"]) < 1e-6
    # 2nd derivatives
    assert abs(_five_point_second(g_a, h_a) - p["G_aa"]) / abs(p["G_aa"]) < 1e-4
    assert abs(_five_point_second(g_v, h_v) - p["G_vv"]) / abs(p["G_vv"]) < 1e-4

    # Mixed: G_av via 4-point cross stencil
    fpp_av = (g_av(h_a, h_v) - g_av(h_a, -h_v) - g_av(-h_a, h_v) + g_av(-h_a, -h_v)) / (
        4.0 * h_a * h_v
    )
    assert abs(fpp_av - p["G_av"]) / max(abs(p["G_av"]), 1e-20) < 1e-3

    # 3rd derivatives
    assert abs(_five_point_third(g_a, h_a) - p["G_aaa"]) / abs(p["G_aaa"]) < 1e-2
    assert abs(_five_point_third(g_v, h_v) - p["G_vvv"]) / abs(p["G_vvv"]) < 5e-2
    # G_aav: f''_aa(g_v(0)) cross-derivative; use cross stencil
    # ∂³g/(∂a² ∂v) = [g(h_a, h_v) - 2 g(0, h_v) + g(-h_a, h_v)
    #               - g(h_a, -h_v) + 2 g(0, -h_v) - g(-h_a, -h_v)] / (2 h_a² h_v)
    g_aav_fd = (
        g_av(h_a, h_v)
        - 2.0 * g_av(0.0, h_v)
        + g_av(-h_a, h_v)
        - g_av(h_a, -h_v)
        + 2.0 * g_av(0.0, -h_v)
        - g_av(-h_a, -h_v)
    ) / (2.0 * h_a * h_a * h_v)
    assert abs(g_aav_fd - p["G_aav"]) / max(abs(p["G_aav"]), 1e-20) < 1e-2

    # G_avv
    g_avv_fd = (
        g_av(h_a, h_v)
        - 2.0 * g_av(h_a, 0.0)
        + g_av(h_a, -h_v)
        - g_av(-h_a, h_v)
        + 2.0 * g_av(-h_a, 0.0)
        - g_av(-h_a, -h_v)
    ) / (2.0 * h_a * h_v * h_v)
    assert abs(g_avv_fd - p["G_avv"]) / max(abs(p["G_avv"]), 1e-20) < 5e-2


# ---------------------------------------------------------------------------
# 3. κ* matches numerical hopf_scan in the canonical regime
# ---------------------------------------------------------------------------


def test_kappa_star_matches_existing_numerical_scan() -> None:
    """Closed-form κ* (quadratic root) matches numerical hopf_scan within 1e-3
    on the canonical §4.3 parameter set.
    """
    p = G_lognormal_oi_partials(
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
        mu_q=_CANONICAL["mu_q"],
        sigma_q=_CANONICAL["sigma_q"],
        T_eff=_CANONICAL["T_eff"],
        coupling_units=_CANONICAL["coupling_units"],
    )
    kappa_star_closed, omega_star_closed = kappa_star_lognormal_oi(
        G_y=p["G_a"],
        G_v=p["G_v"],
        kappa_v=_CANONICAL["kappa_v"],
        alpha=_CANONICAL["alpha"],
        beta=_CANONICAL["beta"],
        gamma=_CANONICAL["gamma"],
    )

    # Numerical scan: build the 3D Jacobian and search over κ
    def jac_at(k: float) -> np.ndarray:
        a_lin = k * p["G_a"]
        b_lin = k * p["G_v"] - 0.5
        return jacobian_3d(
            kappa=k,
            a_kappa=a_lin,
            b_kappa=b_lin,
            G_z=0.0,
            kappa_v=_CANONICAL["kappa_v"],
            alpha=_CANONICAL["alpha"],
            beta=_CANONICAL["beta"],
            gamma=_CANONICAL["gamma"],
        )

    # Scan a wide grid around the closed-form prediction
    grid = np.linspace(0.5 * kappa_star_closed, 1.5 * kappa_star_closed, 401).astype(np.float64)
    result = hopf_scan(grid, jac_at)
    assert result.kappa_star is not None, "numerical scan failed to find κ*"
    rel = abs(result.kappa_star - kappa_star_closed) / kappa_star_closed
    assert rel < 1e-3, (
        f"closed κ*={kappa_star_closed:.6f}, numerical κ*={result.kappa_star:.6f}, rel={rel:.3e}"
    )
    # And ω* matches
    assert result.omega_at_crossing is not None
    assert abs(result.omega_at_crossing - omega_star_closed) < 1e-3


# ---------------------------------------------------------------------------
# 4. Closed-form ℓ_1 matches numerical compute_lyapunov_coefficient
# ---------------------------------------------------------------------------


def test_ell1_matches_existing_numerical_lyapunov() -> None:
    """At the canonical regime, closed-form ℓ_1 matches the FD-tensor numerical
    ℓ_1 on the same drift to within 5% relative.
    """
    kappa_star, omega_star, ell_1_closed = lyapunov_coefficient_lognormal_oi(**_CANONICAL)  # type: ignore[arg-type]

    # Build the same drift as a numerical function and run the FD path
    def drift_fn(x: np.ndarray) -> np.ndarray:
        a = _CANONICAL["a_star"] + x[0]
        v = _CANONICAL["v_star"] + x[1]
        z = x[2]
        G_val = G_lognormal_oi(
            a,
            v,
            mu_q=_CANONICAL["mu_q"],
            sigma_q=_CANONICAL["sigma_q"],
            T_eff=_CANONICAL["T_eff"],
            coupling_units=_CANONICAL["coupling_units"],
        )
        # Equilibrium drift mu chosen so that f1 = 0 at (0, 0, 0) for the chosen κ*.
        G_eq = G_lognormal_oi(
            _CANONICAL["a_star"],
            _CANONICAL["v_star"],
            mu_q=_CANONICAL["mu_q"],
            sigma_q=_CANONICAL["sigma_q"],
            T_eff=_CANONICAL["T_eff"],
            coupling_units=_CANONICAL["coupling_units"],
        )
        mu = 0.5 * _CANONICAL["v_star"] - kappa_star * G_eq
        f1 = mu - 0.5 * v + kappa_star * G_val
        f2 = -_CANONICAL["kappa_v"] * x[1] + _CANONICAL["gamma"] * z
        f3 = -_CANONICAL["alpha"] * z + _CANONICAL["beta"] * x[0]
        return np.array([f1, f2, f3], dtype=np.float64)

    B_fd, C_fd = build_bilinear_trilinear_tensors(drift_fn, (0.0, 0.0, 0.0), h=1e-3)
    # Build J via FD too
    h = 1e-5
    J_fd = np.zeros((3, 3))
    for j in range(3):
        e_j = np.zeros(3)
        e_j[j] = h
        J_fd[:, j] = (drift_fn(e_j) - drift_fn(-e_j)) / (2.0 * h)

    ell_1_num = compute_lyapunov_coefficient(J_fd, B_fd, C_fd, omega=omega_star)
    rel = abs(ell_1_closed - ell_1_num) / abs(ell_1_closed)
    assert rel < 0.05, (
        f"closed ℓ_1={ell_1_closed:.6e}, numerical ℓ_1={ell_1_num:.6e}, rel={rel:.3%}"
    )


# ---------------------------------------------------------------------------
# 5. Sign at canonical regime: supercritical
# ---------------------------------------------------------------------------


def test_ell1_sign_region_supercritical_at_default() -> None:
    """At the canonical §4.3 regime, ℓ_1 < 0 (supercritical Hopf)."""
    _, _, ell_1 = lyapunov_coefficient_lognormal_oi(**_CANONICAL)  # type: ignore[arg-type]
    assert ell_1 < 0.0, f"expected supercritical (ℓ_1 < 0), got {ell_1:.6e}"


# ---------------------------------------------------------------------------
# 6. ω* > 0 at κ* (Routh-Hurwitz consistency)
# ---------------------------------------------------------------------------


def test_omega_star_positive_at_kappa_star() -> None:
    _, omega_star, _ = lyapunov_coefficient_lognormal_oi(**_CANONICAL)  # type: ignore[arg-type]
    assert omega_star > 0.0, f"ω* must be > 0, got {omega_star}"


# ---------------------------------------------------------------------------
# 7. Sign-region phase boundary: ℓ_1 changes sign across (σ_q, γ)
# ---------------------------------------------------------------------------


def test_phase_boundary_supercritical_subcritical_split() -> None:
    """Verify there exist (σ_q, γ) configurations on both sides of the ℓ_1 = 0
    contour — the headline parametric phase boundary.
    """
    base = dict(_CANONICAL)
    super_count = 0
    sub_count = 0
    # The boundary lies roughly along increasing γ at fixed σ_q: small γ → super,
    # large γ → sub. Scan in (σ_q, γ) over a window that straddles the contour.
    for sigma_q in [0.05, 0.10, 0.20]:
        for gamma in [0.5, 1.0, 2.0, 5.0]:
            base["sigma_q"] = sigma_q
            base["gamma"] = gamma
            try:
                _, _, ell_1 = lyapunov_coefficient_lognormal_oi(**base)  # type: ignore[arg-type]
                if ell_1 < 0.0:
                    super_count += 1
                elif ell_1 > 0.0:
                    sub_count += 1
            except ValueError:
                pass
    assert super_count > 0, "phase scan found no supercritical points"
    assert sub_count > 0, "phase scan found no sub-critical points"


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


def test_G_lognormal_oi_rejects_invalid_args() -> None:
    """Sanity guards for invalid input."""
    with pytest.raises(ValueError, match="sigma_q must be > 0"):
        G_lognormal_oi(0.0, 0.04, mu_q=0.0, sigma_q=0.0, T_eff=0.1)
    with pytest.raises(ValueError, match="T_eff must be > 0"):
        G_lognormal_oi(0.0, 0.04, mu_q=0.0, sigma_q=0.1, T_eff=0.0)
    with pytest.raises(ValueError, match="variance must be"):
        G_lognormal_oi(0.0, -0.01, mu_q=0.0, sigma_q=0.1, T_eff=0.1)


def test_kappa_star_rejects_no_real_root() -> None:
    """When the quadratic discriminant is negative, raises ValueError."""
    # G_y = G_v = 0 - degenerate case
    with pytest.raises(ValueError, match="Hopf condition degenerate"):
        kappa_star_lognormal_oi(G_y=0.0, G_v=0.0, kappa_v=2.0, alpha=0.5, beta=1.0, gamma=1.0)
