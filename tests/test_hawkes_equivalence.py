"""Tests for `theory/hawkes_equivalence.py` (paper §3.11 / amendment A10).

These are self-consistency regression tests of the λ_max(κ) eigenvalue
track and the *definitional* n_SV(κ) rescaling — NOT verification of a
paper theorem. The v0.3.9 reposition (A10) demoted the n_SV equivalence to
a definitional identity; the operative falsifiable construct is the spectral
discriminator in `theory/hawkes_sv_bifurcation.py`. Three properties:

1. **Criticality endpoint holds by construction.** n_SV(κ★) = 1 is a
   definitional identity of the β₀ gauge (a tautology, not a result); the
   test confirms the arithmetic to numerical precision when κ★ sits exactly
   on the input grid (and to linear-interpolation accuracy when it does not).
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
from scipy.optimize import brentq

from reflexive_options.theory.bifurcation import (
    jacobian_3d,
    jacobian_eigenvalues,
    routh_hurwitz_H,
)
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
    """Definitional identity: n_SV(κ★) = 1 by construction of the β₀ gauge.

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
        f"definitional critical-endpoint identity failed: n_SV(κ★) = {n_at_star} ≠ 1; "
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


def test_n_sv_at_brent_root_is_machine_epsilon() -> None:
    """Self-consistency: the Routh-Hurwitz Hopf root and the eigenvalue-derived
    n_SV agree to machine precision.

    Two *independent* criteria locate the Hopf threshold: the Routh-Hurwitz
    condition H(κ) = c_1·c_2 - c_0 = 0, and the leading-eigenvalue condition
    λ_max(κ) = 0 (which makes the definitional n_SV = 1 + λ_max/β₀ equal 1).
    This test confirms they coincide:

      (a) at the published 4-decimal κ★_4 = 0.8964, |n_SV - 1| ≈ 3.85e-5 — a
          *truncation* artefact of rounding κ★ to 4 decimals, not solver noise;
      (b) at the higher-precision Brent root of H (xtol=1e-14, rtol=1e-15),
          the residual drops below 1e-12, confirming λ_max(κ★) = 0 there to
          machine ε.

    A failure at (b) would mean the Routh-Hurwitz root and the eigenvalue zero
    disagree — i.e. a numerical bug in the eigenvalue machinery. (n_SV(κ★) = 1
    is itself definitional, not a result; see §3.11 / amendment A10.)
    """

    # Solve H(κ) = c_1·c_2 - c_0 = 0 via Brent at machine-precision tolerances.
    def H(k: float) -> float:
        eig = jacobian_eigenvalues(_jac_constant_vol(k))
        _, _, _, h = routh_hurwitz_H(eig)
        return h

    # H changes sign across κ★ ≈ 0.896 — bracket it well inside the canonical
    # window. H(0.85) > 0 (stable), H(0.90) < 0 (past Hopf).
    assert H(0.85) > 0.0, "expected H(0.85) > 0 in canonical regime"
    assert H(0.90) < 0.0, "expected H(0.90) < 0 in canonical regime"

    kappa_star_brent = float(brentq(H, 0.85, 0.90, xtol=1e-14, rtol=1e-15))

    # Sanity: Brent root must round-to-4-decimals to 0.8964.
    assert round(kappa_star_brent, 4) == _KAPPA_STAR_PAPER, (
        f"Brent root {kappa_star_brent:.10f} does not round to {_KAPPA_STAR_PAPER}"
    )

    # Compute β₀ from the canonical scan (same as the experiment runner).
    grid = np.linspace(0.0, 2.0 * _KAPPA_STAR_PAPER, 1001).astype(np.float64)
    result = hawkes_branching_ratio_curve(grid, _jac_constant_vol)
    beta_zero = result.beta_zero
    assert beta_zero > 0.0

    # (a) Published 4-decimal residual: |n_SV(0.8964) - 1| < 1e-3.
    n_at_paper = n_sv_at_kappa(_KAPPA_STAR_PAPER, _jac_constant_vol, beta_zero=beta_zero)
    residual_paper = abs(n_at_paper - 1.0)
    assert residual_paper < 1e-3, (
        f"published 4-decimal residual exceeded bound: |n_SV({_KAPPA_STAR_PAPER}) - 1| = "
        f"{residual_paper:.3e}, expected < 1e-3"
    )
    # Also check it is *not* tiny (it should be ~3.85e-5, the truncation in 0.8964).
    assert residual_paper > 1e-6, (
        f"published 4-decimal residual unexpectedly tiny ({residual_paper:.3e}); "
        "either the truncation-vs-noise framing is wrong or β₀ has shifted"
    )

    # (b) High-precision Brent residual: |n_SV(κ★_brent) - 1| < 1e-12 (tight).
    n_at_brent = n_sv_at_kappa(kappa_star_brent, _jac_constant_vol, beta_zero=beta_zero)
    residual_brent = abs(n_at_brent - 1.0)
    assert residual_brent < 1e-12, (
        f"Routh-Hurwitz/eigenvalue self-consistency FAILS: "
        f"|n_SV(κ★_brent={kappa_star_brent:.12f}) - 1| = {residual_brent:.3e}, "
        "expected < 1e-12 — λ_max should vanish at the H(κ)=0 root, so this "
        "indicates a numerical bug in the eigenvalue machinery."
    )

    # Truncation dominance: the high-precision residual is many orders of
    # magnitude smaller than the published-4-decimal residual.
    assert residual_brent < residual_paper / 1e6, (
        f"residual_brent ({residual_brent:.3e}) not strictly dominated by truncation "
        f"in 4-decimal κ★ ({residual_paper:.3e}); truncation-vs-solver-noise "
        "self-consistency weakened"
    )


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
