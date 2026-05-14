"""Tests for `theory/hawkes_equivalence.py` — Theorem 2 (paper §3.7).

Three core properties of the n_SV(κ) construction:

1. **Criticality endpoint is exact.** n_SV(κ★) = 1 by construction, to
   numerical precision when κ★ sits exactly on the input grid (and to
   linear-interpolation accuracy when it does not).
2. **Monotonicity on the post-node-spiral interval.** Once the slow
   mode is a complex pair (typically κ ≳ 0.13 in the §4.2 regime),
   n_SV(κ) is non-decreasing in κ up to κ★. This is the rigorous
   regime where the Hawkes interpretation holds.
3. **Input-validation.** β₀ ≤ 0 in `n_sv_from_eigenvalues` raises;
   non-ascending κ-grids in `hawkes_branching_ratio_curve` raise.

The §4.2 canonical regime (G_x = 0.5, G_v = -0.5, G_z = -0.5,
α = 0.5, β = 1, γ = 0.5, κ_v = 2) is the locked anchor — same as
`tests/test_paper_section_4_2.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from reflexive_options.theory.bifurcation import jacobian_3d
from reflexive_options.theory.hawkes_equivalence import (
    hawkes_branching_ratio_curve,
    n_sv_at_kappa,
    n_sv_from_eigenvalues,
)

# Locked §4.2 canonical regime
_G_X = 0.5
_G_V = -0.5
_G_Z = -0.5
_ALPHA = 0.5
_BETA = 1.0
_GAMMA = 0.5
_KAPPA_V = 2.0
_KAPPA_STAR_PAPER = 0.8964


def _jac_constant_vol(kappa: float) -> NDArray[np.float64]:
    """§4.2 constant-vol surrogate Jacobian (∂_v σ² = 0)."""
    a = kappa * _G_X
    b = kappa * _G_V
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


def test_n_sv_at_kappa_star_equals_one() -> None:
    """Theorem 2's headline: n_SV(κ★) = 1 by construction.

    Place κ★ exactly on the grid (within 1e-6) so the located
    `result.kappa_star` equals the paper value, and check that the
    n_SV value at that grid index is 1 to ~1e-3 (the residual is a
    numerical artefact of the eigenvalue λ_max(κ★) being only ≈ 1e-5
    rather than exactly zero on a fine but discrete κ grid).
    """
    # Use a grid that contains κ★ to numerical precision.
    pre = np.linspace(0.0, _KAPPA_STAR_PAPER, 5001)
    post = np.linspace(_KAPPA_STAR_PAPER, 2.0 * _KAPPA_STAR_PAPER, 5000)[1:]
    kappa_grid = np.concatenate([pre, post]).astype(np.float64)
    result = hawkes_branching_ratio_curve(kappa_grid, _jac_constant_vol)

    # The grid contains κ★ exactly; the located κ★ should match the paper.
    assert result.kappa_star is not None
    assert abs(result.kappa_star - _KAPPA_STAR_PAPER) < 5e-3, (
        f"located κ★ = {result.kappa_star} drifted from paper {_KAPPA_STAR_PAPER}"
    )
    # n_SV at the located κ★ should be ≈ 1.
    n_at_star = float(np.interp(_KAPPA_STAR_PAPER, result.kappa_grid, result.n_sv))
    assert abs(n_at_star - 1.0) < 1e-3, (
        f"Theorem 2 critical-endpoint identity failed: n_SV(κ★) = {n_at_star} ≠ 1; "
        f"residual = {abs(n_at_star - 1.0):.3e}"
    )

    # Cross-check via the single-κ wrapper.
    n_via_wrapper = n_sv_at_kappa(_KAPPA_STAR_PAPER, _jac_constant_vol, beta_zero=result.beta_zero)
    assert abs(n_via_wrapper - 1.0) < 1e-3, f"n_sv_at_kappa wrapper disagrees: {n_via_wrapper} ≠ 1"


def test_n_sv_monotonic_in_complex_pair_regime() -> None:
    """n_SV is monotonically non-decreasing on [κ_node-spiral, κ★].

    The complex-pair (slow) mode forms past a node-spiral transition
    (κ ≈ 0.13 in the §4.2 regime). Past that point, the implicit
    function theorem applied to the characteristic polynomial gives
    ∂λ_max/∂κ > 0 monotonically up to κ★, so n_SV(κ) is monotonic
    increasing on the interval.

    This is the rigorous Hawkes-equivalent regime — outside it the
    slow mode is purely real and eigenvalue ordering can swap.
    """
    kappa_grid = np.linspace(0.2, _KAPPA_STAR_PAPER, 401).astype(np.float64)
    result = hawkes_branching_ratio_curve(kappa_grid, _jac_constant_vol)

    diffs = np.diff(result.n_sv)
    # Allow a tiny numerical-noise tolerance (eigenvalue computation gives
    # ~1e-12 jitter on a dense grid).
    assert np.all(diffs >= -1e-9), f"n_SV not monotonic on [0.2, κ★]: min diff = {diffs.min():.3e}"

    # n_SV(0.2) should be well below 1 (system is far from criticality);
    # n_SV(κ★) should be ≈ 1.
    assert result.n_sv[0] < 0.5, f"n_SV at κ = 0.2 should be far below 1, got {result.n_sv[0]:.3f}"
    assert abs(result.n_sv[-1] - 1.0) < 5e-3, f"n_SV at κ★ should be ≈ 1, got {result.n_sv[-1]:.3f}"


def test_n_sv_input_validation() -> None:
    """β₀ ≤ 0 and non-ascending κ-grids must raise informative errors."""
    # β₀ ≤ 0 → raises.
    with pytest.raises(ValueError, match="beta_zero must be > 0"):
        n_sv_from_eigenvalues(np.array([-0.1, -0.2]), beta_zero=0.0)
    with pytest.raises(ValueError, match="beta_zero must be > 0"):
        n_sv_from_eigenvalues(np.array([-0.1, -0.2]), beta_zero=-1.0)

    # Non-ascending grid → raises.
    bad_grid = np.array([0.5, 0.3, 0.1], dtype=np.float64)
    with pytest.raises(ValueError, match="strictly ascending"):
        hawkes_branching_ratio_curve(bad_grid, _jac_constant_vol)

    # Single-element grid trivially passes ascending check (np.diff is empty,
    # so np.all(empty > 0) is True). We don't test that — it's an edge case
    # the caller is unlikely to hit, and the function returns a valid (if
    # uninformative) HawkesEquivalenceResult.

    # n_sv_from_eigenvalues with a positive β₀ should return the right
    # arithmetic.
    out = n_sv_from_eigenvalues(np.array([-1.0, -0.5, 0.0, 0.25]), beta_zero=1.0)
    np.testing.assert_allclose(out, np.array([0.0, 0.5, 1.0, 1.25]))
