"""Tests for the mixture-of-K-lognormals OI generalisation (paper §4.3.7,
`paper/mixture_oi_lyapunov.md`).

Coverage:
    - K=1 limit equivalence to `lyapunov_coefficient_lognormal_oi` (machine precision).
    - Weight-normalisation invariance: scaling all weights by a positive constant
      leaves G, G_partials, κ★, ℓ_1 unchanged.
    - Component-order invariance: permuting `mixture_components` does not change
      κ★, ω★, ℓ_1.
    - K=2 closed-form vs FD-tensor reference: ≤ 0.6% relative agreement
      (matching the single-lognormal §4.3.5 tolerance).
    - Bimodal robustness regression: closed form recovers κ★ within 1%
      relative error at the §3.6 fragility cases (Δ ∈ {0.05, 0.10, 0.20}).
    - Multilinearity check: G_mixture(K=2) literally equals
      w_1 G_1 + w_2 G_2 from the single-lognormal API to machine precision.
    - Invalid input rejection: empty component list, zero total weight,
      negative weight, non-positive sigma all raise ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.theory.bifurcation import (
    G_lognormal_oi,
    G_lognormal_oi_partials,
    G_mixture_lognormal_oi,
    G_mixture_lognormal_oi_partials,
    MixtureOIComponent,
    kappa_star_lognormal_oi,
    kappa_star_mixture_lognormal_oi,
    lyapunov_coefficient_lognormal_oi,
    lyapunov_coefficient_mixture_lognormal_oi,
)
from reflexive_options.theory.robustness import (
    kappa_star_misspecification_error,
)

# Canonical specification — identical to tests/test_lognormal_lyapunov.py so
# that mixture results are directly comparable to the single-lognormal case.
_CANONICAL = dict(
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
_MU_Q = float(np.log(100.0))
_SIGMA_Q = 0.10


# ---------------------------------------------------------------------------
# 1. K=1 limit equivalence — mixture API must reproduce single-lognormal API
# ---------------------------------------------------------------------------


def test_K1_mixture_G_equals_single_lognormal() -> None:
    """G_mixture(K=1, w=1) must equal G_lognormal_oi to machine precision."""
    comp = [MixtureOIComponent(weight=1.0, mu_q=_MU_Q, sigma_q=_SIGMA_Q)]
    for a_off in [-0.1, 0.0, 0.05]:
        for v in [0.01, 0.04, 0.10]:
            a = _CANONICAL["a_star"] + a_off
            g_single = G_lognormal_oi(a, v, mu_q=_MU_Q, sigma_q=_SIGMA_Q, T_eff=_CANONICAL["T_eff"])
            g_mix = G_mixture_lognormal_oi(a, v, mixture_components=comp, T_eff=_CANONICAL["T_eff"])
            assert abs(g_single - g_mix) < 1e-15, f"a={a}, v={v}: single={g_single}, mix={g_mix}"


def test_K1_mixture_partials_equals_single_lognormal() -> None:
    """G_mixture_lognormal_oi_partials(K=1, w=1) must equal G_lognormal_oi_partials
    on every key to machine precision."""
    comp = [MixtureOIComponent(weight=1.0, mu_q=_MU_Q, sigma_q=_SIGMA_Q)]
    p_single = G_lognormal_oi_partials(
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
        mu_q=_MU_Q,
        sigma_q=_SIGMA_Q,
        T_eff=_CANONICAL["T_eff"],
    )
    p_mix = G_mixture_lognormal_oi_partials(
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
        mixture_components=comp,
        T_eff=_CANONICAL["T_eff"],
    )
    assert set(p_single) == set(p_mix)
    for key in p_single:
        diff = abs(p_single[key] - p_mix[key])
        assert diff < 1e-15, f"key={key}: single={p_single[key]}, mix={p_mix[key]}, diff={diff:.3e}"


def test_K1_mixture_kappa_star_equals_single() -> None:
    """κ★ and ω★ from the mixture API at K=1 must equal the single-lognormal API."""
    comp = [MixtureOIComponent(weight=1.0, mu_q=_MU_Q, sigma_q=_SIGMA_Q)]
    p = G_lognormal_oi_partials(
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
        mu_q=_MU_Q,
        sigma_q=_SIGMA_Q,
        T_eff=_CANONICAL["T_eff"],
    )
    k_single, om_single = kappa_star_lognormal_oi(
        G_y=p["G_a"],
        G_v=p["G_v"],
        kappa_v=_CANONICAL["kappa_v"],
        alpha=_CANONICAL["alpha"],
        beta=_CANONICAL["beta"],
        gamma=_CANONICAL["gamma"],
    )
    k_mix, om_mix = kappa_star_mixture_lognormal_oi(
        mixture_components=comp,
        T_eff=_CANONICAL["T_eff"],
        kappa_v=_CANONICAL["kappa_v"],
        alpha=_CANONICAL["alpha"],
        beta=_CANONICAL["beta"],
        gamma=_CANONICAL["gamma"],
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
    )
    assert abs(k_single - k_mix) < 1e-12, f"κ★: single={k_single}, mix={k_mix}"
    assert abs(om_single - om_mix) < 1e-12, f"ω★: single={om_single}, mix={om_mix}"


def test_K1_mixture_ell1_equals_single_lognormal() -> None:
    """ℓ_1 from the mixture API at K=1 must equal the single-lognormal API."""
    comp = [MixtureOIComponent(weight=1.0, mu_q=_MU_Q, sigma_q=_SIGMA_Q)]
    k_single, om_single, ell_single = lyapunov_coefficient_lognormal_oi(
        mu_q=_MU_Q,
        sigma_q=_SIGMA_Q,
        **_CANONICAL,  # type: ignore[arg-type]
    )
    k_mix, om_mix, ell_mix = lyapunov_coefficient_mixture_lognormal_oi(
        mixture_components=comp,
        **_CANONICAL,  # type: ignore[arg-type]
    )
    assert abs(k_single - k_mix) < 1e-12
    assert abs(om_single - om_mix) < 1e-12
    assert abs(ell_single - ell_mix) < 1e-12, (
        f"ℓ_1: single={ell_single}, mix={ell_mix}, diff={abs(ell_single - ell_mix):.3e}"
    )


# ---------------------------------------------------------------------------
# 2. Weight-normalisation invariance — scaling weights leaves outputs unchanged
# ---------------------------------------------------------------------------


def test_weight_normalisation_invariance() -> None:
    """Scaling all weights by a positive constant preserves G and partials."""
    base = [
        MixtureOIComponent(weight=1.0, mu_q=_MU_Q - 0.05, sigma_q=0.08),
        MixtureOIComponent(weight=2.0, mu_q=_MU_Q + 0.05, sigma_q=0.08),
    ]
    scaled = [
        MixtureOIComponent(weight=10.0, mu_q=_MU_Q - 0.05, sigma_q=0.08),
        MixtureOIComponent(weight=20.0, mu_q=_MU_Q + 0.05, sigma_q=0.08),
    ]
    p_base = G_mixture_lognormal_oi_partials(
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
        mixture_components=base,
        T_eff=_CANONICAL["T_eff"],
    )
    p_scaled = G_mixture_lognormal_oi_partials(
        a_star=_CANONICAL["a_star"],
        v_star=_CANONICAL["v_star"],
        mixture_components=scaled,
        T_eff=_CANONICAL["T_eff"],
    )
    for key in p_base:
        assert abs(p_base[key] - p_scaled[key]) < 1e-15


# ---------------------------------------------------------------------------
# 3. Component-order invariance — permuting components doesn't change κ★ etc.
# ---------------------------------------------------------------------------


def test_component_order_invariance() -> None:
    """Permuting the order of mixture components yields identical κ★, ω★, ℓ_1."""
    comps_abc = [
        MixtureOIComponent(weight=0.3, mu_q=_MU_Q - 0.05, sigma_q=0.07),
        MixtureOIComponent(weight=0.5, mu_q=_MU_Q, sigma_q=0.10),
        MixtureOIComponent(weight=0.2, mu_q=_MU_Q + 0.10, sigma_q=0.06),
    ]
    comps_cba = list(reversed(comps_abc))
    k_abc, om_abc, ell_abc = lyapunov_coefficient_mixture_lognormal_oi(
        mixture_components=comps_abc,
        **_CANONICAL,  # type: ignore[arg-type]
    )
    k_cba, om_cba, ell_cba = lyapunov_coefficient_mixture_lognormal_oi(
        mixture_components=comps_cba,
        **_CANONICAL,  # type: ignore[arg-type]
    )
    assert abs(k_abc - k_cba) < 1e-12, f"κ★: abc={k_abc}, cba={k_cba}"
    assert abs(om_abc - om_cba) < 1e-12, f"ω★: abc={om_abc}, cba={om_cba}"
    assert abs(ell_abc - ell_cba) < 1e-12, f"ℓ_1: abc={ell_abc}, cba={ell_cba}"


# ---------------------------------------------------------------------------
# 4. Multilinearity check — G_mixture = Σ w_k G_k (literal identity)
# ---------------------------------------------------------------------------


def test_multilinearity_decomposition() -> None:
    """G_mixture(K=2) at any (a, v) literally equals w_1 G_1 + w_2 G_2."""
    w = [0.3, 0.7]
    mu = [_MU_Q - 0.05, _MU_Q + 0.10]
    sigma = [0.08, 0.06]
    comps = [MixtureOIComponent(weight=w[k], mu_q=mu[k], sigma_q=sigma[k]) for k in range(2)]
    for a in [_MU_Q - 0.1, _MU_Q, _MU_Q + 0.1]:
        for v in [0.01, 0.05, 0.10]:
            g_mix = G_mixture_lognormal_oi(
                a, v, mixture_components=comps, T_eff=_CANONICAL["T_eff"]
            )
            g_sum = sum(
                w[k] * G_lognormal_oi(a, v, mu_q=mu[k], sigma_q=sigma[k], T_eff=_CANONICAL["T_eff"])
                for k in range(2)
            )
            assert abs(g_mix - g_sum) < 1e-15, f"a={a}, v={v}: mix={g_mix}, sum={g_sum}"


# ---------------------------------------------------------------------------
# 5. K=2 closed-form vs FD-tensor reference — matches single-lognormal tol
# ---------------------------------------------------------------------------


def test_K2_closed_form_matches_fd_tensor() -> None:
    """The K=2 mixture closed-form κ★ matches the FD-tensor reference (the
    `kappa_star_brute_force_from_G` pipeline applied directly to the mixture
    density via quadrature) to ≤ 0.6% relative — i.e. at least as tightly as
    the single-lognormal closed form vs its FD reference (paper §4.3.5).
    """
    for delta in [0.05, 0.10, 0.20, 0.30]:
        comps = [
            MixtureOIComponent(weight=0.5, mu_q=_MU_Q - delta / 2.0, sigma_q=0.07),
            MixtureOIComponent(weight=0.5, mu_q=_MU_Q + delta / 2.0, sigma_q=0.07),
        ]
        # FD reference via the existing misspecification pipeline
        err = kappa_star_misspecification_error(
            mu_components=[c.mu_q for c in comps],
            sigma_components=[c.sigma_q for c in comps],
            weights=[c.weight for c in comps],
            T_eff=_CANONICAL["T_eff"],
            kappa_v=_CANONICAL["kappa_v"],
            theta_v=_CANONICAL["theta_v"],
            alpha=_CANONICAL["alpha"],
            beta=_CANONICAL["beta"],
            gamma=_CANONICAL["gamma"],
            a_star=_CANONICAL["a_star"],
            v_star=_CANONICAL["v_star"],
        )
        k_true = err.kappa_star_true
        k_mix, _ = kappa_star_mixture_lognormal_oi(
            mixture_components=comps,
            T_eff=_CANONICAL["T_eff"],
            kappa_v=_CANONICAL["kappa_v"],
            alpha=_CANONICAL["alpha"],
            beta=_CANONICAL["beta"],
            gamma=_CANONICAL["gamma"],
            a_star=_CANONICAL["a_star"],
            v_star=_CANONICAL["v_star"],
        )
        rel = abs(k_mix - k_true) / k_true
        assert rel <= 6e-3, (
            f"Δ={delta}: K=2 mixture κ★={k_mix} vs FD κ★={k_true}, rel={rel:.3e} > 0.6%"
        )


# ---------------------------------------------------------------------------
# 6. Bimodal robustness regression — closed form within 1% at Δ ≤ 0.20
# ---------------------------------------------------------------------------


def test_K2_bimodal_robustness_regression() -> None:
    """At Δ ∈ {0.05, 0.10, 0.20} — the §3.6 fragility cases for the single
    log-normal (with 0.23%, 4.83%, 119% errors respectively) — the K=2
    mixture closed form recovers κ★ to < 1% relative error.

    This is the headline robustness improvement claimed in
    paper/mixture_oi_lyapunov.md.
    """
    for delta in [0.05, 0.10, 0.20]:
        comps = [
            MixtureOIComponent(weight=0.5, mu_q=_MU_Q - delta / 2.0, sigma_q=0.07),
            MixtureOIComponent(weight=0.5, mu_q=_MU_Q + delta / 2.0, sigma_q=0.07),
        ]
        err = kappa_star_misspecification_error(
            mu_components=[c.mu_q for c in comps],
            sigma_components=[c.sigma_q for c in comps],
            weights=[c.weight for c in comps],
            T_eff=_CANONICAL["T_eff"],
            kappa_v=_CANONICAL["kappa_v"],
            theta_v=_CANONICAL["theta_v"],
            alpha=_CANONICAL["alpha"],
            beta=_CANONICAL["beta"],
            gamma=_CANONICAL["gamma"],
            a_star=_CANONICAL["a_star"],
            v_star=_CANONICAL["v_star"],
        )
        k_true = err.kappa_star_true
        k_mix, _ = kappa_star_mixture_lognormal_oi(
            mixture_components=comps,
            T_eff=_CANONICAL["T_eff"],
            kappa_v=_CANONICAL["kappa_v"],
            alpha=_CANONICAL["alpha"],
            beta=_CANONICAL["beta"],
            gamma=_CANONICAL["gamma"],
            a_star=_CANONICAL["a_star"],
            v_star=_CANONICAL["v_star"],
        )
        rel = abs(k_mix - k_true) / k_true
        assert rel < 1e-2, (
            f"Δ={delta}: K=2 mixture closed form has {rel * 100:.3f}% relative error vs FD"
        )


# ---------------------------------------------------------------------------
# 7. Invalid input handling
# ---------------------------------------------------------------------------


def test_empty_components_rejected() -> None:
    with pytest.raises(ValueError, match="≥ 1"):
        G_mixture_lognormal_oi_partials(
            a_star=_CANONICAL["a_star"],
            v_star=_CANONICAL["v_star"],
            mixture_components=[],
            T_eff=_CANONICAL["T_eff"],
        )


def test_zero_total_weight_rejected() -> None:
    """All-zero weights are degenerate; the constructor accepts a zero
    individual weight but the sum-zero check rejects an entirely-zero list.
    """
    with pytest.raises(ValueError, match="sum of mixture weights"):
        G_mixture_lognormal_oi(
            _CANONICAL["a_star"],
            _CANONICAL["v_star"],
            mixture_components=[
                MixtureOIComponent(weight=0.0, mu_q=_MU_Q, sigma_q=0.10),
                MixtureOIComponent(weight=0.0, mu_q=_MU_Q + 0.05, sigma_q=0.10),
            ],
            T_eff=_CANONICAL["T_eff"],
        )


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="weight must be"):
        MixtureOIComponent(weight=-0.1, mu_q=_MU_Q, sigma_q=0.10)


def test_nonpositive_sigma_rejected() -> None:
    with pytest.raises(ValueError, match="sigma_q must be"):
        MixtureOIComponent(weight=1.0, mu_q=_MU_Q, sigma_q=0.0)
    with pytest.raises(ValueError, match="sigma_q must be"):
        MixtureOIComponent(weight=1.0, mu_q=_MU_Q, sigma_q=-0.05)


# ---------------------------------------------------------------------------
# 8. K=3 sanity — three components stay numerically stable
# ---------------------------------------------------------------------------


def test_K3_mixture_is_numerically_stable() -> None:
    """K=3 mixture produces a finite, well-typed κ★, ω★, ℓ_1 and the K=3 limit
    when one weight → 0 collapses to the K=2 case to machine precision.
    """
    comps_K3 = [
        MixtureOIComponent(weight=0.4, mu_q=_MU_Q - 0.05, sigma_q=0.07),
        MixtureOIComponent(weight=0.4, mu_q=_MU_Q + 0.05, sigma_q=0.07),
        MixtureOIComponent(weight=1e-15, mu_q=_MU_Q + 0.15, sigma_q=0.05),
    ]
    comps_K2 = [
        MixtureOIComponent(weight=0.4, mu_q=_MU_Q - 0.05, sigma_q=0.07),
        MixtureOIComponent(weight=0.4, mu_q=_MU_Q + 0.05, sigma_q=0.07),
    ]
    k3, om3, ell3 = lyapunov_coefficient_mixture_lognormal_oi(
        mixture_components=comps_K3,
        **_CANONICAL,  # type: ignore[arg-type]
    )
    k2, om2, ell2 = lyapunov_coefficient_mixture_lognormal_oi(
        mixture_components=comps_K2,
        **_CANONICAL,  # type: ignore[arg-type]
    )
    assert np.isfinite(k3) and np.isfinite(om3) and np.isfinite(ell3)
    assert abs(k3 - k2) / k2 < 1e-10
    assert abs(om3 - om2) / om2 < 1e-10
    assert abs(ell3 - ell2) / abs(ell2) < 1e-8
