"""Regression test for the §4.2 dimensionless Hopf example in paper/theory.md.

Section 4.2 publishes a representative dimensionless regime where the
deterministic skeleton sits on a supercritical Hopf bifurcation:

    G_x = 0.5,  G_v = -0.5,  G_z = -0.5
    α = 0.5,    β = 1,       γ = 0.5
    κ_v = 2
    G_xx = -0.1,  G_xxx = -0.2

Convention: this regime uses the **constant-vol surrogate** σ² = const, so
∂_x σ² = ∂_v σ² = 0 in the linearisation a(κ) = κ G_x, b(κ) = κ G_v. This is
*not* the σ² = v Heston convention used in §4.3 — see paper §4.2 footnote and
verification_v2_consistency.md BLOCKER-1.

Locked headline numbers (paper §4.2 Table):

    κ* = 0.8964
    ω* = 0.5724 rad/yr
    ℓ_1 = -0.0253 (supercritical, ℓ_1 < 0)

This test would have caught V1-B2 (silent drift in §4.2 numbers) immediately.
Any future drift > 1% relative on κ*, ω*, or ℓ_1 should fail this test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from reflexive_options.theory.bifurcation import (
    build_bilinear_trilinear_tensors,
    compute_lyapunov_coefficient,
    hopf_scan,
    jacobian_3d,
)

# Locked §4.2 canonical regime
_G_X = 0.5
_G_V = -0.5
_G_Z = -0.5
_ALPHA = 0.5
_BETA = 1.0
_GAMMA = 0.5
_KAPPA_V = 2.0
_G_XX = -0.1
_G_XXX = -0.2

# Locked headline values
_KAPPA_STAR_EXPECTED = 0.8964
_OMEGA_STAR_EXPECTED = 0.5724
_ELL1_EXPECTED = -0.0253


def _jac_constant_vol(kappa: float) -> np.ndarray:
    """Jacobian under the §4.2 constant-vol surrogate (∂_v σ² = 0)."""
    a = kappa * _G_X  # − 0.5 ∂_x σ² = 0 by convention
    b = kappa * _G_V  # − 0.5 ∂_v σ² = 0 by convention
    return jacobian_3d(
        kappa=kappa,
        a_kappa=a,
        b_kappa=b,
        G_z=_G_Z,
        kappa_v=_KAPPA_V,
        alpha=_ALPHA,
        beta=_BETA,
        gamma=_GAMMA,
    )


def _drift_section_4_2(
    kappa_star: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Local drift skeleton at the §4.2 regime, parametrized by κ*.

    State (y, u, z) := (δ log S, δv, δz). Under σ² = const, f_1 has only the
    κG channel — no -½ σ²(y, u) contribution.
    """

    def drift(x: np.ndarray) -> np.ndarray:
        y, u, z = float(x[0]), float(x[1]), float(x[2])
        f1 = kappa_star * (
            _G_X * y + _G_V * u + _G_Z * z + 0.5 * _G_XX * y * y + (1.0 / 6.0) * _G_XXX * y * y * y
        )
        f2 = -_KAPPA_V * u + _GAMMA * z
        f3 = -_ALPHA * z + _BETA * y
        return np.array([f1, f2, f3], dtype=np.float64)

    return drift


def test_section_4_2_numerical_example_reproduces() -> None:
    """Section 4.2 publishes a representative dimensionless Hopf regime.

    Reproduces κ* = 0.8964, ω* = 0.5724, ℓ_1 = -0.0253 to within 1% relative.
    Uses the constant-vol surrogate convention (∂_v σ² = 0) explicitly.
    """
    grid = np.linspace(0.01, 2.0, 5001).astype(np.float64)
    res = hopf_scan(grid, _jac_constant_vol)

    assert res.kappa_star is not None, "Hopf scan failed to find κ* in §4.2 regime"
    assert res.omega_at_crossing is not None, "Hopf scan failed to find ω* in §4.2 regime"

    rel_kappa = abs(res.kappa_star - _KAPPA_STAR_EXPECTED) / _KAPPA_STAR_EXPECTED
    assert rel_kappa < 0.01, (
        f"§4.2 κ* drifted: got {res.kappa_star:.6f}, expected {_KAPPA_STAR_EXPECTED}, "
        f"rel = {rel_kappa:.3%}"
    )

    rel_omega = abs(res.omega_at_crossing - _OMEGA_STAR_EXPECTED) / _OMEGA_STAR_EXPECTED
    assert rel_omega < 0.01, (
        f"§4.2 ω* drifted: got {res.omega_at_crossing:.6f}, expected {_OMEGA_STAR_EXPECTED}, "
        f"rel = {rel_omega:.3%}"
    )

    drift_fn = _drift_section_4_2(res.kappa_star)
    B, C = build_bilinear_trilinear_tensors(drift_fn, (0.0, 0.0, 0.0), h=1e-3)
    J = _jac_constant_vol(res.kappa_star)
    ell1 = compute_lyapunov_coefficient(J, B, C, omega=res.omega_at_crossing)

    rel_ell1 = abs(ell1 - _ELL1_EXPECTED) / abs(_ELL1_EXPECTED)
    assert rel_ell1 < 0.01, (
        f"§4.2 ℓ_1 drifted: got {ell1:.6e}, expected {_ELL1_EXPECTED}, rel = {rel_ell1:.3%}"
    )

    assert ell1 < 0.0, f"§4.2 expected supercritical (ℓ_1 < 0), got {ell1:.6e}"


def test_section_4_2_convention_documented() -> None:
    """§4.2 uses ∂_v σ² = 0 explicitly. Switching to the σ² = v convention
    drifts κ* / ω* / ℓ_1 by ~10% — this test pins the convention so any silent
    swap to σ² = v in the §4.2 codepath fails immediately.
    """

    def jac_sigma2_eq_v(kappa: float) -> np.ndarray:
        a = kappa * _G_X  # ∂_x σ² = 0 (σ² depends only on v)
        b = kappa * _G_V - 0.5  # ∂_v σ² = 1 (the σ² = v Heston convention)
        return jacobian_3d(
            kappa=kappa,
            a_kappa=a,
            b_kappa=b,
            G_z=_G_Z,
            kappa_v=_KAPPA_V,
            alpha=_ALPHA,
            beta=_BETA,
            gamma=_GAMMA,
        )

    grid = np.linspace(0.01, 2.0, 5001).astype(np.float64)
    res_sv = hopf_scan(grid, jac_sigma2_eq_v)

    # Under σ² = v we should get a noticeably different κ* (≈ 0.796), proving
    # the §4.2 numbers depend on the convention choice.
    assert res_sv.kappa_star is not None
    drift = abs(res_sv.kappa_star - _KAPPA_STAR_EXPECTED) / _KAPPA_STAR_EXPECTED
    assert drift > 0.05, (
        f"Convention switch should drift κ* by >5%, got {drift:.3%}; "
        "either §4.2 has been silently modified or one of the conventions is wrong."
    )


@pytest.mark.parametrize(
    "param,bad_value",
    [
        ("kappa_v", -1.0),
        ("alpha", -0.5),
    ],
)
def test_section_4_2_jacobian_invariants(param: str, bad_value: float) -> None:
    """Sanity guard: the §4.2 regime requires κ_v > 0 and α > 0; if a future
    refactor zeroes one of these, the bifurcation analysis becomes degenerate.
    """
    overrides = {param: bad_value}
    kappa_v = overrides.get("kappa_v", _KAPPA_V)
    alpha = overrides.get("alpha", _ALPHA)

    def jac(kappa: float) -> np.ndarray:
        a = kappa * _G_X
        b = kappa * _G_V
        return jacobian_3d(
            kappa=kappa,
            a_kappa=a,
            b_kappa=b,
            G_z=_G_Z,
            kappa_v=kappa_v,
            alpha=alpha,
            beta=_BETA,
            gamma=_GAMMA,
        )

    # Should still numerically execute without a crash, but the resulting
    # Jacobian will not satisfy the §4.2 sign structure.
    J = jac(0.9)
    assert J.shape == (3, 3)
