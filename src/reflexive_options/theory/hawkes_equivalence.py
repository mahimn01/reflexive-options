"""Hawkes branching ratio ↔ SV-Jacobian eigenvalue mapping.

This module computes the leading-eigenvalue real part λ_max(κ) of the 3D
reflexive skeleton's Jacobian and the *definitional* Hawkes-branching-ratio
rescaling n_SV(κ) := 1 + λ_max(κ)/β₀ that maps Hardiman-Bercot-Bouchaud
(2013)'s branching ratio into the SV eigenvalue language.

⚠ v0.3.9 reposition (pre-data amendment A10). The identity n_SV(κ★) = 1 is
**definitional, not a result** — it holds by construction of the β₀ gauge,
so any check that it equals 1 "to machine precision" is verifying arithmetic
self-consistency, not a theorem. The paper's repositioned Theorem 4 (§3.11)
no longer claims a scalar n_SV equivalence: it places Hardiman's n ≈ 1 at the
*real-eigenvalue (saddle-node)* stratum and the model's Hopf threshold κ★ as
the strictly-stronger oscillatory cell beyond it. The operative, falsifiable
construct is now the spectral discriminator in `theory/hawkes_sv_bifurcation.py`
(classify_stratum). This module is retained for the genuinely useful pieces —
the λ_max(κ) computation and the BDHM-2013 diffusive-limit mapping below — not
as evidence for any n_SV claim.

Derivation of the mapping (the BDHM diffusive-limit chain is exact; the n_SV
rescaling is definitional — see paper §3.11 / amendment A10 for the reposition):

1. **Hawkes baseline.** A 1D self-exciting Hawkes intensity λ(t) = μ +
   Σᵢ φ(t − tᵢ) with kernel φ(s) = α e^{−βs} has branching ratio
   n := ∫₀^∞ φ(s) ds = α/β. Stability requires n < 1.

2. **Diffusive limit (Bacry-Delattre-Hoffmann-Muzy 2013, Theorem 2 of
   the SPA paper).** Under T → ∞ rescaling t → t/T, λ → T λ̄, the
   centred intensity λ̄ satisfies the SDE
        dλ̄ = −β(1 − n) λ̄ dt + σ_λ dW,
   so the deterministic *relaxation rate* of the rescaled intensity is
   β(1 − n). Equivalently, the leading eigenvalue of the linearised
   intensity dynamics is λ_max(Hawkes) = −β(1 − n).

3. **Inversion.** n = 1 + λ_max/β. Criticality (n = 1) ⇔ λ_max = 0.

4. **Multivariate generalisation (Bacry-Mastromatteo-Muzy 2015, §2.4).**
   For a d-variate Hawkes with integrated kernel matrix Φ ∈ R^{d×d}, n
   is replaced by the spectral radius ρ(Φ). Stability requires ρ(Φ) < 1
   and the slow-mode relaxation rate is 1 − ρ(Φ) (in natural units).
   The endpoint ρ(Φ) = 1 ⇔ leading eigenvalue of the linearised
   generator hits zero is universal across kernel shapes.

5. **SV equivalence.** Our 3D reflexive skeleton has Jacobian J(κ) with
   leading-eigenvalue real part λ_max(κ) ≤ 0 for κ ∈ [0, κ★] and
   λ_max(κ★) = 0 (Hopf threshold). Identify the slow-mode relaxation
   rate β(1 − n) ↔ −λ_max(κ). Choose the natural "bare relaxation rate"
   β₀ as the maximum of |λ_max| over the stable interval — i.e. the
   most-stable point of the slow mode, which is the SV analogue of the
   Hawkes baseline rate β at n = 0. Then
        n_SV(κ) := 1 + λ_max(κ) / β₀.
   By construction n_SV(κ★) = 1 (criticality) and n_SV is monotonically
   non-decreasing on the interval [κ_ref, κ★] where κ_ref achieves β₀.

**Honest limitations.**

- The mapping is **exact** in two senses: (i) the criticality endpoint
  n_SV(κ★) = 1 is exact by construction, and (ii) for the 1D Hawkes
  exponential-kernel case the formula n = 1 + λ_max/β is the exact
  diffusive-limit identity of Bacry et al. (2013).
- It is **approximate as a global Hawkes equivalence**: Hardiman's empirical
  n is the L¹ norm of a fitted multivariate Hawkes kernel on order-flow
  events, not a direct mapping from the continuous SV state. The
  identification "Hardiman n ≈ 1 ⇔ market sits near κ★" rests on the
  universality of the n = 1 ⇔ λ_max = 0 boundary across continuous-time
  reflexive systems, NOT on a path-by-path identity between event-counting
  Hawkes processes and our diffusion.
- The 3D system has a **gauge zero eigenvalue at κ = 0** in the spot
  direction (no σ²-correction, so log S is a frozen mode in the
  noiseless skeleton). This is why we cannot use the naive normalisation
  n_SV(κ) = 1 − λ_max(κ)/λ_max(0) — λ_max(0) = 0 makes that singular.
  Using β₀ := max_{κ ∈ [0, κ★]} (−λ_max(κ)) is the gauge-invariant
  replacement.
- Monotonicity is **provable** on [κ_node-spiral, κ★] (the interval where
  the slow mode is a complex pair), where the implicit-function theorem
  applied to the characteristic polynomial gives ∂λ_max/∂κ > 0.
  Outside that interval — between κ = 0 and the node-spiral transition
  — the slow mode is purely real and the eigenvalue ordering can swap;
  we report n_SV in this regime but flag it as a "pre-Hopf" region where
  the Hawkes interpretation is weaker.

References:
    Hardiman, Bercot & Bouchaud (2013), "Critical Reflexivity in Financial
        Markets: A Hawkes Process Analysis", EPJ B 86:442.
    Bacry, Delattre, Hoffmann & Muzy (2013), "Some limit theorems for Hawkes
        processes and application to financial statistics", SPA 123(7).
    Bacry, Mastromatteo & Muzy (2015), "Hawkes Processes in Finance",
        Market Microstructure and Liquidity 1(1).
    Filimonov & Sornette (2012), Phys Rev E 85:056108.

Implemented in `theory/hawkes_equivalence.py`. Wired into the canonical
§4.2 regime via `experiments/hawkes_sv_equivalence.py`. Tests in
`tests/test_hawkes_equivalence.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reflexive_options.theory.bifurcation import jacobian_eigenvalues


@dataclass(frozen=True)
class HawkesEquivalenceResult:
    """Output of `hawkes_branching_ratio_curve`.

    Attributes:
        kappa_grid: ascending grid of κ values used for the scan.
        lambda_max_real: leading-eigenvalue real part of J(κ) at each κ
            (≤ 0 in the stable regime, 0 at κ★).
        beta_zero: gauge-invariant baseline relaxation rate
            β₀ := max_{κ ∈ grid ∩ [0, κ★]} (−λ_max(κ)). The SV analogue
            of the Hawkes baseline rate β at n = 0.
        n_sv: SV-equivalent branching ratio
            n_SV(κ) = 1 + λ_max(κ) / β₀,
            satisfying n_SV(κ★) = 1 (criticality).
        kappa_at_beta_zero: the κ value attaining β₀ — the "most-stable"
            point of the slow mode, where n_SV(κ) ≈ 0.
        kappa_star: the located Hopf threshold (smallest κ with
            λ_max(κ) ≥ 0). May be None if no crossing in the grid.
    """

    kappa_grid: NDArray[np.float64]
    lambda_max_real: NDArray[np.float64]
    beta_zero: float
    n_sv: NDArray[np.float64]
    kappa_at_beta_zero: float
    kappa_star: float | None


def n_sv_from_eigenvalues(
    lambda_max_real: NDArray[np.float64],
    beta_zero: float,
) -> NDArray[np.float64]:
    """Compute the SV-equivalent branching ratio from a leading-eigenvalue
    array and a baseline relaxation rate.

    Implements the definitional rescaling n_SV(κ) = 1 + λ_max(κ) / β₀
    (paper §3.11 / amendment A10; n_SV(κ★) = 1 holds by construction).

    Args:
        lambda_max_real: leading-eigenvalue real-parts at each κ. Should
            be ≤ 0 for stability; values > 0 correspond to past-Hopf
            regimes where n_SV > 1.
        beta_zero: baseline relaxation rate β₀ > 0 (the SV analogue of
            the Hawkes baseline rate). Must be strictly positive — a
            zero β₀ means the slow mode never relaxes at any κ in the
            sampled interval, which makes the Hawkes-equivalent
            branching ratio ill-defined.

    Returns:
        Array of n_SV values. n_SV = 0 at the κ achieving β₀ (the most
        stable slow-mode point), n_SV = 1 at κ★ (Hopf criticality),
        n_SV > 1 past κ★.

    Raises:
        ValueError: if `beta_zero` is not strictly positive.
    """
    if beta_zero <= 0.0:
        raise ValueError(
            f"beta_zero must be > 0 to define n_SV; got {beta_zero}. "
            "This typically means the slow-mode relaxation rate is identically "
            "zero on the scanned interval — check that the Jacobian has a "
            "well-defined stable eigenvalue at some κ in the grid."
        )
    return np.asarray(1.0 + lambda_max_real / beta_zero, dtype=np.float64)


def hawkes_branching_ratio_curve(
    kappa_grid: NDArray[np.float64],
    jacobian_at: Callable[[float], NDArray[np.float64]],
) -> HawkesEquivalenceResult:
    """Compute the SV-equivalent Hawkes branching ratio n_SV(κ) over a κ grid.

    Computes the λ_max(κ) eigenvalue track and the definitional n_SV(κ)
    rescaling (paper §3.11 / amendment A10). Pipeline:

    1. Evaluate the leading-eigenvalue real part λ_max(κ) on the grid.
    2. Locate the Hopf threshold κ★ as the smallest κ with λ_max(κ) ≥ 0.
    3. Set β₀ := max_{κ ∈ grid ∩ [0, κ★]} (−λ_max(κ)) — the
       gauge-invariant baseline relaxation rate. (The naive
       normalisation by |λ_max(0)| fails in our 3D system because the
       spot mode has a structural zero eigenvalue at κ = 0; see the
       module docstring.)
    4. Return n_SV(κ) = 1 + λ_max(κ) / β₀ across the grid.

    Args:
        kappa_grid: strictly ascending κ values.
        jacobian_at: callable κ ↦ Jacobian matrix (e.g., a partial of
            `theory.bifurcation.jacobian_3d`).

    Returns:
        HawkesEquivalenceResult bundling λ_max, β₀, n_SV, κ_ref, κ★.

    Raises:
        ValueError: if `kappa_grid` is not strictly ascending or if no
            stable point exists on the grid (β₀ would be 0).
    """
    if not np.all(np.diff(kappa_grid) > 0):
        raise ValueError("kappa_grid must be strictly ascending")

    n = len(kappa_grid)
    lambda_max_real = np.zeros(n, dtype=np.float64)
    for i, k in enumerate(kappa_grid):
        eig = jacobian_eigenvalues(jacobian_at(float(k)))
        lambda_max_real[i] = float(eig.real.max())

    # Locate κ★ as the first **sign-change** from strictly negative to
    # non-negative — not the first non-negative index. The distinction
    # matters because the 3D system has a structural zero eigenvalue at
    # κ = 0 (the gauge-zero spot mode) which would falsely trigger a
    # "κ★ at grid origin" if we used `lambda_max ≥ 0` directly.
    kappa_star: float | None = None
    # Treat |λ| ≤ tol as "zero" to avoid floating-point misclassification of
    # the κ = 0 gauge zero. The eigenvalue solver's typical noise floor for
    # 3×3 matrices is ~1e-14; 1e-12 keeps us well above that.
    tol = 1e-12
    is_strictly_negative = lambda_max_real < -tol
    if not is_strictly_negative.any():
        # No κ in the grid produces a strictly negative leading eigenvalue.
        # The Hawkes equivalence is ill-defined (β₀ = 0).
        stable_slice = lambda_max_real
    else:
        first_strict_neg_idx = int(np.argmax(is_strictly_negative))  # first True
        # Look for the first non-negative AFTER we've entered the stable
        # region. This skips the κ = 0 gauge zero.
        post_stable = np.where(lambda_max_real[first_strict_neg_idx:] >= -tol)[0]
        # `post_stable[0]` is 0 itself (the entry-into-stable-region index);
        # we want the first element where lambda_max comes back up to ≥ 0.
        post_back_to_nonneg = post_stable[
            lambda_max_real[first_strict_neg_idx + post_stable] >= 0.0
        ]
        if post_back_to_nonneg.size == 0:
            # Whole post-stable interval stays negative — no κ★ on this grid.
            stable_slice = lambda_max_real
        else:
            idx_kappa_star = first_strict_neg_idx + int(post_back_to_nonneg[0])
            kappa_star = float(kappa_grid[idx_kappa_star])
            stable_slice = lambda_max_real[: idx_kappa_star + 1]

    # β₀ := max(−λ_max) on the stable interval. Gauge-invariant choice.
    decay_rates = -stable_slice
    beta_zero = float(decay_rates.max())
    kappa_at_beta_zero = float(kappa_grid[int(np.argmax(decay_rates))])

    if beta_zero <= 0.0:
        raise ValueError(
            "β₀ ≤ 0: the leading eigenvalue is non-negative across the entire "
            "stable interval. The Hawkes-equivalent branching ratio is "
            "ill-defined. Check that the Jacobian has a stable equilibrium "
            "for some κ in the grid (e.g. κ near 0 with σ² > 0)."
        )

    n_sv = n_sv_from_eigenvalues(lambda_max_real, beta_zero)

    return HawkesEquivalenceResult(
        kappa_grid=kappa_grid,
        lambda_max_real=lambda_max_real,
        beta_zero=beta_zero,
        n_sv=n_sv,
        kappa_at_beta_zero=kappa_at_beta_zero,
        kappa_star=kappa_star,
    )


def n_sv_at_kappa(
    kappa: float,
    jacobian_at: Callable[[float], NDArray[np.float64]],
    *,
    beta_zero: float,
) -> float:
    """Evaluate n_SV at a single κ given a precomputed β₀ baseline.

    Convenience wrapper around `n_sv_from_eigenvalues` for callers that
    already have β₀ pinned (e.g., from a paper-locked canonical regime).

    Args:
        kappa: single κ value.
        jacobian_at: callable κ ↦ Jacobian.
        beta_zero: precomputed baseline relaxation rate (must be > 0).

    Returns:
        n_SV(κ) = 1 + λ_max(κ) / β₀.
    """
    eig = jacobian_eigenvalues(jacobian_at(float(kappa)))
    lambda_max = float(eig.real.max())
    arr = n_sv_from_eigenvalues(np.array([lambda_max], dtype=np.float64), beta_zero)
    return float(arr[0])
