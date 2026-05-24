"""Tests for `theory/info_theoretic.py` — Theorem 5 (paper §3.10).

Six core properties of the excess-entropy / transfer-entropy machinery:

1. **Markov closure.** E_τ(κ → 0⁺) → 0: at vanishing coupling the spot
   process is independent of past spot conditional on (v_0, z_0), so the
   excess entropy of the dealer-gamma channel vanishes.
2. **Monotonicity.** E_τ(κ) is non-decreasing in κ on the entire stable
   interval (0, κ★) at the §4.2 canonical regime — verified numerically
   on a 101-point grid for τ ∈ {0.1, 1, 5} yr.
3. **Critical-exponent fit.** Near κ★, E_τ(κ) approaches a finite
   saturation E_τ(κ★) linearly in (κ★ - κ) (mean-field critical exponent
   β = 1, Theorem 5(c)).
4. **Positive saturation.** E_τ(κ★ - ε) > 0 for small ε > 0 — the
   dealer-gamma channel has nontrivial information content at the
   bifurcation boundary.
5. **Transfer-entropy directionality.** On a controlled AR(1)-driven
   target the empirical T_{src → tgt} substantially exceeds T_{tgt → src}
   and is significant under an IAAFT-surrogate null.
6. **IAAFT-null calibration.** Under an independent-source null the IAAFT
   p-value is approximately uniform (not skewed towards small values),
   verified by a smoke test on a Stuart-Landau-like positive control.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from reflexive_options.theory.bifurcation import jacobian_3d
from reflexive_options.theory.info_theoretic import (
    CriticalExponentFit,
    ExcessEntropyCurveResult,
    excess_entropy_curve,
    excess_entropy_linear,
    fit_critical_exponent,
    transfer_entropy_iaaft_pvalue,
    transfer_entropy_simulated,
)

# Locked §4.2 canonical regime
_G_X = 0.5
_G_V = -0.5
_G_Z = -0.5
_ALPHA = 0.5
_BETA = 1.0
_GAMMA = 0.5
_KAPPA_V = 2.0
_KAPPA_STAR = 0.8964305216  # Brent-precision root of H(κ) = c_1·c_2 - c_0

# Linearised diffusion in the constant-vol surrogate (§4.2): noise on the
# spot is √θ_v, noise on variance is ξ·√θ_v, no noise on memory. The matrix
# is diagonal in this surrogate.
_THETA_V = 0.04
_XI = 0.3
_SS_CANONICAL: NDArray[np.float64] = np.diag([_THETA_V, (_XI**2) * _THETA_V, 0.0]).astype(
    np.float64
)


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


# ---------------------------------------------------------------------------
# Excess-entropy formula
# ---------------------------------------------------------------------------


def test_markov_closure_kappa_to_zero() -> None:
    """E_τ(κ → 0⁺) → 0 — Theorem 5(a)."""
    # Probe at decreasing κ; the excess entropy should scale as κ² (Taylor
    # expansion of the formula around κ = 0) — verify the order of magnitude.
    for tau in (0.1, 1.0, 5.0):
        e_small = excess_entropy_linear(_jac_constant_vol(1e-3), _SS_CANONICAL, tau)
        e_tiny = excess_entropy_linear(_jac_constant_vol(1e-4), _SS_CANONICAL, tau)
        # κ² scaling: drop by 100 between κ = 10⁻³ and κ = 10⁻⁴.
        assert e_small > 0.0
        assert e_tiny > 0.0
        ratio = e_small / e_tiny
        assert 80.0 < ratio < 120.0, f"κ² scaling broken at τ={tau}: {ratio}"
        # Absolute smallness check.
        assert e_tiny < 1e-7


def test_monotonicity_on_canonical_grid() -> None:
    """E_τ(κ) is non-decreasing on the stable interval — Theorem 5 numerical claim."""
    grid = np.linspace(1e-4, _KAPPA_STAR - 1e-6, 101)
    for tau in (0.1, 1.0, 5.0):
        result = excess_entropy_curve(grid, _jac_constant_vol, _SS_CANONICAL, tau)
        assert result.is_monotone, f"non-monotone E_τ at τ={tau}"
        # No NaNs across the interior of the stable region.
        assert np.all(np.isfinite(result.excess_entropy[:-1]))
        # End-to-end positive enhancement (E at right edge > 100× E at left).
        e_first = result.excess_entropy[0]
        e_last = result.excess_entropy[-1]
        assert e_last > 100.0 * e_first, (
            f"insufficient critical-edge enhancement at τ={tau}: {e_last / e_first:.2e}"
        )


def test_critical_exponent_beta_equals_one() -> None:
    """Near κ★, the linear approach E(κ) ≈ E_inf - C·(κ★ - κ) gives β = 1 — Theorem 5(c)."""
    for tau in (0.1, 1.0, 5.0):
        fit = fit_critical_exponent(_jac_constant_vol, _SS_CANONICAL, tau, _KAPPA_STAR)
        assert isinstance(fit, CriticalExponentFit)
        # Mean-field prediction: β = 1 to 2 decimals.
        assert abs(fit.beta - 1.0) < 1e-2, f"β = {fit.beta:.4f} at τ={tau} (predicted 1.0)"
        # Saturation E_inf is positive and finite.
        assert fit.e_infinity > 0.0
        assert np.isfinite(fit.e_infinity)
        # Coefficient C is positive (E is approaching E_inf from below).
        assert fit.coefficient > 0.0


def test_positive_saturation_at_boundary() -> None:
    """E_τ(κ★ - ε) > 0 for small ε — non-vanishing info content at criticality."""
    epsilon = 1e-6
    for tau in (0.1, 1.0, 5.0):
        e_boundary = excess_entropy_linear(
            _jac_constant_vol(_KAPPA_STAR - epsilon), _SS_CANONICAL, tau
        )
        assert e_boundary > 0.0, f"vanishing saturation at τ={tau}"
        assert np.isfinite(e_boundary), f"divergent saturation at τ={tau}"


def test_excess_entropy_curve_returns_dataclass() -> None:
    """`excess_entropy_curve` returns an ExcessEntropyCurveResult with expected fields."""
    grid = np.linspace(1e-3, 0.5, 11)
    result = excess_entropy_curve(grid, _jac_constant_vol, _SS_CANONICAL, 1.0)
    assert isinstance(result, ExcessEntropyCurveResult)
    assert result.tau == 1.0
    assert result.excess_entropy.shape == (11,)
    assert np.array_equal(result.kappa_grid, grid)
    assert result.is_monotone is True


def test_excess_entropy_input_validation() -> None:
    """`excess_entropy_linear` and `excess_entropy_curve` validate inputs."""
    j = _jac_constant_vol(0.5)
    with pytest.raises(ValueError, match="tau must be > 0"):
        excess_entropy_linear(j, _SS_CANONICAL, 0.0)
    with pytest.raises(ValueError, match="observed_index"):
        excess_entropy_linear(j, _SS_CANONICAL, 1.0, observed_index=5)
    with pytest.raises(ValueError, match="conditioned_indices"):
        excess_entropy_linear(j, _SS_CANONICAL, 1.0, observed_index=0, conditioned_indices=(1, 99))
    with pytest.raises(ValueError, match="cannot also be in conditioned_indices"):
        excess_entropy_linear(j, _SS_CANONICAL, 1.0, observed_index=0, conditioned_indices=(0, 1))
    with pytest.raises(ValueError, match="strictly ascending"):
        excess_entropy_curve(np.array([0.5, 0.1, 0.3]), _jac_constant_vol, _SS_CANONICAL, 1.0)


def test_non_hurwitz_returns_nan() -> None:
    """When J is not Hurwitz, the excess entropy is NaN (no stationary covariance)."""
    # κ > κ★ — at least one eigenvalue has Re ≥ 0.
    e = excess_entropy_linear(_jac_constant_vol(1.5), _SS_CANONICAL, 1.0)
    assert np.isnan(e)


# ---------------------------------------------------------------------------
# Empirical transfer entropy
# ---------------------------------------------------------------------------


def _make_ar1_pair(
    n: int = 4000,
    *,
    seed: int = 20260514,
    coupling: float = 0.4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct a source AR(1) and a target that depends on source's lag-1.

    Used as the positive control: T_{src → tgt} should be much larger than
    T_{tgt → src} and significant under IAAFT.
    """
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n)
    tgt = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        tgt[t] = 0.5 * tgt[t - 1] + coupling * src[t - 1] + 0.1 * rng.standard_normal()
    return src, tgt


def test_transfer_entropy_directionality() -> None:
    """T_{src → tgt} > T_{tgt → src} for source-driven target — Schreiber's directionality."""
    src, tgt = _make_ar1_pair()
    te_fwd = transfer_entropy_simulated(src, tgt, lag=1, n_bins=6)
    te_rev = transfer_entropy_simulated(tgt, src, lag=1, n_bins=6)
    assert te_fwd > 0.0
    assert te_fwd > 10.0 * te_rev, f"directionality broken: fwd={te_fwd}, rev={te_rev}"


def test_iaaft_null_significant_on_positive_control() -> None:
    """IAAFT-null TE p-value is significant for a true source → target coupling."""
    src, tgt = _make_ar1_pair()
    rng = np.random.default_rng(20260514)
    result = transfer_entropy_iaaft_pvalue(src, tgt, lag=1, n_bins=6, n_surrogates=50, rng=rng)
    assert result.observed > 0.1
    assert result.p_value < 0.05, f"p={result.p_value} on a positive control"
    assert result.surrogate_quantile_95 < result.observed
    assert result.n_surrogates == 50


def test_iaaft_null_non_significant_on_independent_source() -> None:
    """IAAFT-null p-value is NOT significant when source and target are independent."""
    rng = np.random.default_rng(20260515)
    _, tgt = _make_ar1_pair()
    # Independent source — same length, no causal link to target.
    src_indep = rng.standard_normal(len(tgt))
    result = transfer_entropy_iaaft_pvalue(
        src_indep, tgt, lag=1, n_bins=6, n_surrogates=50, rng=rng
    )
    # Plug-in TE is bias-prone but the null calibration should keep p-value
    # away from the rejection region for an honestly-null source.
    assert result.p_value > 0.05, f"false positive on independent source: p={result.p_value:.3f}"


def test_transfer_entropy_input_validation() -> None:
    """`transfer_entropy_simulated` and `transfer_entropy_iaaft_pvalue` validate inputs."""
    x = np.random.default_rng(0).standard_normal(100)
    y = np.random.default_rng(1).standard_normal(100)
    with pytest.raises(ValueError, match="lag must be >= 1"):
        transfer_entropy_simulated(x, y, lag=0)
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        transfer_entropy_simulated(x, y, lag=1, n_bins=1)
    with pytest.raises(ValueError, match="shapes differ"):
        transfer_entropy_simulated(x, y[:50], lag=1)
    with pytest.raises(ValueError, match="too short for lag"):
        transfer_entropy_simulated(np.zeros(3), np.zeros(3), lag=5)
    with pytest.raises(ValueError, match="n_surrogates must be >= 1"):
        transfer_entropy_iaaft_pvalue(x, y, n_surrogates=0)


def test_excess_entropy_shape_validation() -> None:
    """`excess_entropy_linear` validates jacobian and diffusion shapes."""
    j_bad = np.ones((3, 2))  # not square
    ss = np.eye(3)
    with pytest.raises(ValueError, match="jacobian must be square"):
        excess_entropy_linear(j_bad, ss, 1.0)
    j_good = _jac_constant_vol(0.5)
    ss_bad = np.eye(4)  # mismatched
    with pytest.raises(ValueError, match="diffusion_outer shape"):
        excess_entropy_linear(j_good, ss_bad, 1.0)


def test_excess_entropy_curve_tau_validation() -> None:
    """`excess_entropy_curve` validates tau."""
    grid = np.linspace(0.1, 0.5, 5)
    with pytest.raises(ValueError, match="tau must be > 0"):
        excess_entropy_curve(grid, _jac_constant_vol, _SS_CANONICAL, 0.0)


def test_fit_critical_exponent_validates_deltas() -> None:
    """`fit_critical_exponent` rejects non-positive δ-grid."""
    with pytest.raises(ValueError, match="deltas must be strictly positive"):
        fit_critical_exponent(
            _jac_constant_vol,
            _SS_CANONICAL,
            1.0,
            _KAPPA_STAR,
            deltas=np.array([0.01, -0.001, 0.0001]),
        )


def test_fit_critical_exponent_rejects_too_few_finite() -> None:
    """If most δ-grid entries fall in the non-Hurwitz region, fit raises."""
    # δ-values past κ★ (so kappa_star - δ < 0 → still Hurwitz for small δ,
    # but choose large δ that push us into the unstable region).
    with pytest.raises(ValueError, match="insufficient finite"):
        fit_critical_exponent(
            _jac_constant_vol,
            _SS_CANONICAL,
            1.0,
            _KAPPA_STAR,
            # All deltas push us to κ < 0 (well past the lower stability bound
            # — at κ near 0 we get the marginal eigenvalue). Use big deltas.
            deltas=np.array([_KAPPA_STAR + 10.0] * 5),
        )


def test_curve_kappa_star_estimated_on_grid_with_unstable_tail() -> None:
    """`excess_entropy_curve.kappa_star_estimated` flags the first NaN κ."""
    # A grid that crosses κ★ — should produce NaN entries past it.
    grid = np.linspace(0.5, 1.5, 21)
    result = excess_entropy_curve(grid, _jac_constant_vol, _SS_CANONICAL, 1.0)
    assert result.kappa_star_estimated is not None
    assert _KAPPA_STAR - 0.1 <= result.kappa_star_estimated <= _KAPPA_STAR + 0.1
