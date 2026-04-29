"""Numerical Hopf bifurcation analysis of the reflexive 3D SDE.

The deterministic skeleton (paper/theory.md §2) has Jacobian (3) with
characteristic polynomial $P(\\lambda; \\kappa) = \\lambda^3 + c_2 \\lambda^2 + c_1 \\lambda + c_0$.
A Hopf bifurcation occurs when $H(\\kappa) := c_1 c_2 - c_0 = 0$ with
$c_2, c_0 > 0$ and $H'(\\kappa) \\neq 0$ (Liu's criterion / Routh-Hurwitz).

This module performs the numerical scan, computes the first Lyapunov coefficient
$\\ell_1$ that fixes super- vs sub-criticality (Kuznetsov 2004 eq. 3.20), and the
stochastic-Hopf shift $\\Lambda(\\kappa)$ (Engel-Lamb-Rasmussen 2024) via a
Khasminskii sphere process for the top Lyapunov exponent of the linearised SDE.

The analytical results live in paper/theory.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

DriftFn3D = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""x \\in R^3 -> f(x) \\in R^3 — pure deterministic drift of the 3D skeleton."""

DiffusionFn3D = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""x \\in R^3 -> Σ(x) \\in R^{3x3} — diffusion matrix s.t. dx = f dt + Σ dW."""


@dataclass(frozen=True)
class HopfScanResult:
    """Output of `hopf_scan` over a κ grid."""

    kappa_grid: NDArray[np.float64]
    eigenvalues: NDArray[np.complex128]  # shape (n_kappa, 3)
    H_values: NDArray[np.float64]  # H(κ) = c_1 c_2 - c_0; zeros are Hopf candidates
    real_parts_max: NDArray[np.float64]  # max(Re(λ)) at each κ; zero crossing = D-bifurcation
    omega_at_crossing: float | None  # Hopf frequency at κ* if found
    kappa_star: float | None  # the bifurcation point


def jacobian_3d(
    kappa: float,
    a_kappa: float,  # κ G_x - 0.5 ∂_x σ²
    b_kappa: float,  # κ G_v - 0.5 ∂_v σ²
    G_z: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Jacobian (3) of the 3D reflexive skeleton at the equilibrium.

    See paper/theory.md §2 equation (3) for the symbolic form.
    """
    return np.array(
        [
            [a_kappa, b_kappa, kappa * G_z],
            [0.0, -kappa_v, gamma],
            [beta, 0.0, -alpha],
        ],
        dtype=np.float64,
    )


def jacobian_eigenvalues(jacobian: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Numerical eigenvalues of the Jacobian. Sorted by descending real part."""
    eig = np.linalg.eigvals(jacobian)
    order = np.argsort(-eig.real)
    return eig[order]


def routh_hurwitz_H(eigenvalues: NDArray[np.complex128]) -> tuple[float, float, float, float]:
    """Compute (c_2, c_1, c_0, H) from eigenvalues of a 3x3 Jacobian.

    P(λ) = (λ - λ_1)(λ - λ_2)(λ - λ_3) = λ³ + c_2 λ² + c_1 λ + c_0
    with c_2 = -Σλ_i, c_1 = Σλ_i λ_j, c_0 = -Πλ_i.
    H = c_1 c_2 - c_0.
    """
    if eigenvalues.shape != (3,):
        raise ValueError(f"expected 3 eigenvalues, got shape {eigenvalues.shape}")
    l1, l2, l3 = eigenvalues
    c_2 = float(-(l1 + l2 + l3).real)
    c_1 = float((l1 * l2 + l1 * l3 + l2 * l3).real)
    c_0 = float(-(l1 * l2 * l3).real)
    return c_2, c_1, c_0, c_1 * c_2 - c_0


def hopf_scan(
    kappa_grid: NDArray[np.float64],
    jacobian_at: Callable[[float], NDArray[np.float64]],
) -> HopfScanResult:
    """Scan κ over `kappa_grid`, locate any Hopf bifurcation.

    Args:
        kappa_grid: ascending values of κ to scan.
        jacobian_at: callable κ -> Jacobian matrix (e.g. partial of `jacobian_3d`).

    Returns:
        HopfScanResult with eigenvalues, H(κ), and the located κ* if any.
    """
    if not np.all(np.diff(kappa_grid) > 0):
        raise ValueError("kappa_grid must be strictly ascending")

    n = len(kappa_grid)
    eigs = np.zeros((n, 3), dtype=np.complex128)
    H_values = np.zeros(n, dtype=np.float64)
    real_parts_max = np.zeros(n, dtype=np.float64)

    for i, k in enumerate(kappa_grid):
        J = jacobian_at(float(k))
        eig = jacobian_eigenvalues(J)
        eigs[i] = eig
        _, _, _, H_values[i] = routh_hurwitz_H(eig)
        real_parts_max[i] = float(eig.real.max())

    kappa_star: float | None = None
    omega_star: float | None = None

    sign_change = np.where(np.diff(np.sign(real_parts_max)) != 0)[0]
    if len(sign_change) > 0:
        idx = int(sign_change[0])
        a, b = float(kappa_grid[idx]), float(kappa_grid[idx + 1])

        def f(k: float) -> float:
            return float(jacobian_eigenvalues(jacobian_at(k)).real.max())

        try:
            kappa_star = float(brentq(f, a, b, xtol=1e-10))
            J_star = jacobian_at(kappa_star)
            eig_star = jacobian_eigenvalues(J_star)
            complex_pair = eig_star[np.argsort(np.abs(eig_star.real))][:2]
            omega_star = float(np.abs(complex_pair[0].imag))
        except ValueError:
            pass

    return HopfScanResult(
        kappa_grid=kappa_grid,
        eigenvalues=eigs,
        H_values=H_values,
        real_parts_max=real_parts_max,
        omega_at_crossing=omega_star,
        kappa_star=kappa_star,
    )


# ---------------------------------------------------------------------------
# First Lyapunov coefficient (Kuznetsov 2004, Elements of Applied Bifurcation
# Theory, eq. 3.20). Sign of ℓ_1 fixes super- (ℓ_1 < 0) vs sub-critical
# (ℓ_1 > 0) Hopf.
# ---------------------------------------------------------------------------


def _hopf_eigenvectors(
    jacobian: NDArray[np.float64],
    omega: float,
    *,
    tol: float = 1e-6,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Right and left eigenvectors q, p of J for the imaginary pair ±iω.

    Conventions (Kuznetsov 2004 §3.5):
        J q = i ω q,
        J^T p = -i ω p,
        ⟨p, q⟩ := \\sum_i p̄_i q_i = 1.
    """
    eigvals, eigvecs = np.linalg.eig(jacobian)
    target = 1j * omega
    distances = np.abs(eigvals - target)
    idx = int(np.argmin(distances))
    if distances[idx] > tol * max(1.0, omega):
        raise ValueError(
            f"no eigenvalue of J near +i·{omega}: closest is {eigvals[idx]} "
            f"(distance {distances[idx]:.3e}); check Jacobian / Hopf threshold"
        )
    q = eigvecs[:, idx].astype(np.complex128)
    q = q / np.linalg.norm(q)

    eigvals_T, eigvecs_T = np.linalg.eig(jacobian.T)
    target_T = -1j * omega
    distances_T = np.abs(eigvals_T - target_T)
    idx_T = int(np.argmin(distances_T))
    if distances_T[idx_T] > tol * max(1.0, omega):
        raise ValueError(
            f"no eigenvalue of J^T near -i·{omega}: closest is {eigvals_T[idx_T]} "
            f"(distance {distances_T[idx_T]:.3e})"
        )
    p = eigvecs_T[:, idx_T].astype(np.complex128)

    # Normalize ⟨p, q⟩ = sum_i conj(p_i) * q_i = 1
    inner = np.vdot(p, q)
    if abs(inner) < 1e-14:
        raise ValueError(
            f"left/right eigenvectors are orthogonal (⟨p,q⟩={inner:.3e}); "
            "indicates a near-defective Jacobian — Hopf normal form ill-defined"
        )
    p = p / np.conj(inner)
    return q, p


def _bilinear(
    B: NDArray[np.float64],
    u: NDArray[np.complex128],
    v: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Apply the symmetric bilinear form B(u, v)_i = Σ_jk B_ijk u_j v_k."""
    return cast(NDArray[np.complex128], np.einsum("ijk,j,k->i", B, u, v))


def _trilinear(
    C: NDArray[np.float64],
    u: NDArray[np.complex128],
    v: NDArray[np.complex128],
    w: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Apply the symmetric trilinear form C(u, v, w)_i = Σ_jkl C_ijkl u_j v_k w_l."""
    return cast(NDArray[np.complex128], np.einsum("ijkl,j,k,l->i", C, u, v, w))


def compute_lyapunov_coefficient(
    jacobian: NDArray[np.float64],
    B_tensor: NDArray[np.float64],
    C_tensor: NDArray[np.float64],
    *,
    omega: float | None = None,
) -> float:
    """First Lyapunov coefficient ℓ_1 at a Hopf point (Kuznetsov 2004 eq. 3.20).

    Given the Jacobian J at the bifurcation (with a pair of pure-imaginary
    eigenvalues ±iω) and the symmetric bilinear / trilinear Taylor tensors

        f_i(x) = J_ij x_j + 1/2 B_ijk x_j x_k + 1/6 C_ijkl x_j x_k x_l + O(|x|^4),

    the first Lyapunov coefficient is

        ℓ_1 = (1/(2ω)) Re[ ⟨p, C(q,q,q̄)⟩
                          - 2 ⟨p, B(q, J^{-1} B(q,q̄))⟩
                          + ⟨p, B(q̄, (2iω I − J)^{-1} B(q,q))⟩ ]

    where J q = iω q, J^T p = -iω p, ⟨p, q⟩ = 1.

    Sign convention: ℓ_1 < 0 → supercritical (stable limit cycle for κ > κ*),
    ℓ_1 > 0 → subcritical (unstable cycle, hysteresis).

    Args:
        jacobian: 3x3 Jacobian at the Hopf point.
        B_tensor: 3x3×3 symmetric bilinear tensor B[i,j,k] = ∂²f_i/∂x_j∂x_k.
        C_tensor: 3x3×3x3 symmetric trilinear tensor C[i,j,k,l] = ∂³f_i/∂x_j∂x_k∂x_l.
        omega: Hopf frequency ω* > 0. If None, inferred from the Jacobian's
            eigenvalues (largest |Im λ| among the closest-to-imaginary pair).
    """
    if jacobian.shape != (3, 3):
        raise ValueError(f"jacobian must be 3x3, got {jacobian.shape}")
    if B_tensor.shape != (3, 3, 3):
        raise ValueError(f"B_tensor must be 3x3×3, got {B_tensor.shape}")
    if C_tensor.shape != (3, 3, 3, 3):
        raise ValueError(f"C_tensor must be 3x3×3x3, got {C_tensor.shape}")

    if omega is None:
        eigvals = np.linalg.eigvals(jacobian)
        # The Hopf pair is the one with the smallest |Re| (ideally ≈ 0).
        order = np.argsort(np.abs(eigvals.real))
        omega_inferred = float(np.abs(eigvals[order[0]].imag))
        if omega_inferred < 1e-10:
            raise ValueError(
                "could not infer ω from Jacobian — no near-imaginary eigenvalue pair"
            )
        omega = omega_inferred

    q, p = _hopf_eigenvectors(jacobian, omega)
    q_bar = np.conj(q)

    eye3 = np.eye(3, dtype=np.complex128)
    J_complex = jacobian.astype(np.complex128)

    # Resolvents needed for the second and third terms.
    # First term needs no resolvent.
    J_inv = np.linalg.inv(J_complex)
    resolvent_2iw = np.linalg.inv(2j * omega * eye3 - J_complex)

    # Term 1: ⟨p, C(q, q, q̄)⟩
    term1 = np.vdot(p, _trilinear(C_tensor, q, q, q_bar))

    # Term 2: -2 ⟨p, B(q, J^{-1} B(q, q̄))⟩
    Bqqb = _bilinear(B_tensor, q, q_bar)
    inner2 = J_inv @ Bqqb
    term2 = -2.0 * np.vdot(p, _bilinear(B_tensor, q, inner2))

    # Term 3: ⟨p, B(q̄, (2iω I − J)^{-1} B(q, q))⟩
    Bqq = _bilinear(B_tensor, q, q)
    inner3 = resolvent_2iw @ Bqq
    term3 = np.vdot(p, _bilinear(B_tensor, q_bar, inner3))

    ell1 = (1.0 / (2.0 * omega)) * float(np.real(term1 + term2 + term3))
    return ell1


# ---------------------------------------------------------------------------
# Finite-difference construction of B and C tensors from a drift function.
# Chosen step size h is a balance between truncation error (~ h^2 for a 4-point
# central scheme) and roundoff (~ ε_machine / h^k for k-th derivative). For
# double precision a step around (eps)^(1/(k+2)) is near-optimal:
#   k=2  →  h ≈ 1e-4
#   k=3  →  h ≈ 1e-3
# We pick a single h ≈ 1e-3 that is acceptable for both — accuracy ~1e-6 to ~1e-4.
# ---------------------------------------------------------------------------


def build_bilinear_trilinear_tensors(
    drift_fn: DriftFn3D,
    equilibrium: tuple[float, float, float] | NDArray[np.float64],
    *,
    h: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct B (3x3×3) and C (3x3×3x3) tensors via finite differences.

    For a smooth f: R^3 → R^3 with f(x*) = 0, the Taylor expansion at the
    equilibrium is

        f_i(x* + ξ) = J_ij ξ_j + (1/2) B_ijk ξ_j ξ_k + (1/6) C_ijkl ξ_j ξ_k ξ_l + ...

    so

        B_ijk  = ∂²f_i/∂x_j ∂x_k        (symmetric in j, k)
        C_ijkl = ∂³f_i/∂x_j ∂x_k ∂x_l   (symmetric in j, k, l)

    Mixed central-difference stencils are used; results are explicitly
    symmetrised over the lower indices. Step size `h` defaults to 1e-3.

    Args:
        drift_fn: callable x ∈ R^3 → f(x) ∈ R^3.
        equilibrium: x* such that drift_fn(x*) ≈ 0.
        h: finite-difference step.
    """
    x0 = np.asarray(equilibrium, dtype=np.float64).reshape(3)

    def f(dx: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(drift_fn(x0 + dx), dtype=np.float64).reshape(3)

    e = np.eye(3, dtype=np.float64) * h
    B = np.zeros((3, 3, 3), dtype=np.float64)
    C = np.zeros((3, 3, 3, 3), dtype=np.float64)

    # Diagonal second derivatives: ∂²f/∂x_j² = (f(+h e_j) − 2 f(0) + f(-h e_j)) / h²
    f0 = f(np.zeros(3))
    f_p: dict[int, NDArray[np.float64]] = {}
    f_m: dict[int, NDArray[np.float64]] = {}
    for j in range(3):
        f_p[j] = f(e[j])
        f_m[j] = f(-e[j])

    for j in range(3):
        B[:, j, j] = (f_p[j] - 2.0 * f0 + f_m[j]) / (h * h)

    # Off-diagonal second derivatives via 4-point cross stencil:
    #   ∂²f/∂x_j∂x_k = [f(+h_j+h_k) − f(+h_j−h_k) − f(−h_j+h_k) + f(−h_j−h_k)] / (4 h²)
    for j in range(3):
        for k in range(j + 1, 3):
            f_pp = f(e[j] + e[k])
            f_pm = f(e[j] - e[k])
            f_mp = f(-e[j] + e[k])
            f_mm = f(-e[j] - e[k])
            mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            B[:, j, k] = mixed
            B[:, k, j] = mixed

    # Third derivatives.
    #   Diagonal:  ∂³f/∂x_j³ = [f(+2h e_j) − 2 f(+h e_j) + 2 f(−h e_j) − f(−2h e_j)] / (2 h³)
    #   Off-diag (jjk):  ∂³f/(∂x_j² ∂x_k) = [(f(+h_j+h_k) − 2 f(+h_k) + f(−h_j+h_k))
    #                                      − (f(+h_j−h_k) − 2 f(−h_k) + f(−h_j−h_k))] / (2 h³)
    #   Off-diag (jkl) all distinct: 8-point stencil.
    f_2p = {j: f(2.0 * e[j]) for j in range(3)}
    f_2m = {j: f(-2.0 * e[j]) for j in range(3)}

    # Third diagonals
    for j in range(3):
        C[:, j, j, j] = (f_2p[j] - 2.0 * f_p[j] + 2.0 * f_m[j] - f_2m[j]) / (2.0 * h**3)

    # ∂³f/∂x_j² ∂x_k for j != k, applying full symmetrisation
    for j in range(3):
        for k in range(3):
            if j == k:
                continue
            f_pp = f(e[j] + e[k])
            f_pm = f(e[j] - e[k])
            f_mp = f(-e[j] + e[k])
            f_mm = f(-e[j] - e[k])
            f_pk = f(e[k])
            f_mk = f(-e[k])
            d3 = (
                (f_pp - 2.0 * f_pk + f_mp) - (f_pm - 2.0 * f_mk + f_mm)
            ) / (2.0 * h**3)
            # Symmetric in the two j's: assign all (jjk, jkj, kjj) variants
            C[:, j, j, k] = d3
            C[:, j, k, j] = d3
            C[:, k, j, j] = d3

    # ∂³f/∂x_j ∂x_k ∂x_m for the unique all-distinct triple (0, 1, 2) — 8-point stencil.
    j, k, m = 0, 1, 2
    f_ppp = f(e[j] + e[k] + e[m])
    f_ppm = f(e[j] + e[k] - e[m])
    f_pmp = f(e[j] - e[k] + e[m])
    f_pmm = f(e[j] - e[k] - e[m])
    f_mpp = f(-e[j] + e[k] + e[m])
    f_mpm = f(-e[j] + e[k] - e[m])
    f_mmp = f(-e[j] - e[k] + e[m])
    f_mmm = f(-e[j] - e[k] - e[m])
    d3_jkm = (
        f_ppp - f_ppm - f_pmp - f_mpp + f_pmm + f_mpm + f_mmp - f_mmm
    ) / (8.0 * h**3)
    # All 6 permutations of (j, k, m)
    for a, b, c in [(j, k, m), (j, m, k), (k, j, m), (k, m, j), (m, j, k), (m, k, j)]:
        C[:, a, b, c] = d3_jkm

    return B, C


# ---------------------------------------------------------------------------
# Stochastic Hopf shift Λ(κ) via Khasminskii's sphere process and Benettin
# renormalisation for the top Lyapunov exponent of the linearised SDE.
#
# Implementation notes:
#   - We linearise the SDE in deviation variables ξ = x - x* around the
#     equilibrium: dξ = J ξ dt + Σ(x*) dW, with Σ multiplied by ε.
#   - Rather than sphere-projecting the whole flow (which adds projection
#     drift terms), we run the linearised flow ξ_t directly and renormalise
#     its norm every Δt steps (Benettin et al. 1980). This yields the same
#     top Lyapunov exponent in the additive-noise limit and is robust enough
#     for the small multiplicative-noise regime needed here.
#   - Cost: O(n_paths · n_steps) per ε. Defaults are conservative; a serious
#     run wants n_paths ≥ 10^4 and n_steps · dt ≥ 10/|λ|.
# ---------------------------------------------------------------------------


def top_lyapunov_exponent_linearised(
    jacobian: NDArray[np.float64],
    diffusion_matrix: NDArray[np.float64],
    *,
    epsilon: float,
    n_paths: int = 200,
    n_steps: int = 5_000,
    dt: float = 1e-2,
    renorm_every: int = 50,
    seed: int | None = None,
) -> float:
    """Top Lyapunov exponent λ_1 of the linearised SDE dξ = J ξ dt + ε Σ dW.

    Uses Benettin renormalisation: integrate Euler-Maruyama, renormalise
    ξ → ξ / ‖ξ‖ every `renorm_every` steps, accumulate log-stretches.

    The linear SDE is degenerate at ξ=0 in the multiplicative sense; we treat
    the noise as additive ε Σ(x*) dW (the linearisation is around the fixed
    equilibrium so the diffusion is evaluated at the equilibrium and is
    constant in ξ — this is the standard assumption for the small-noise
    expansion λ_1 = α + ε^2 Λ + O(ε^4)).

    Returns the time-average of (1/T) Σ log‖ξ(t_i + Δt)‖/‖ξ(t_i)‖ over paths.
    """
    if jacobian.shape != (3, 3):
        raise ValueError(f"jacobian must be 3x3, got {jacobian.shape}")
    if diffusion_matrix.shape != (3, 3):
        raise ValueError(f"diffusion_matrix must be 3x3, got {diffusion_matrix.shape}")
    if n_steps % renorm_every != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a multiple of renorm_every ({renorm_every})"
        )

    rng = np.random.default_rng(seed)
    sqrt_dt = float(np.sqrt(dt))
    eps_sigma = epsilon * diffusion_matrix
    J = jacobian
    # Random initial directions on the unit sphere
    xi = rng.standard_normal((n_paths, 3))
    xi = xi / np.linalg.norm(xi, axis=1, keepdims=True)

    log_stretches = np.zeros(n_paths, dtype=np.float64)

    for step in range(1, n_steps + 1):
        dW = rng.standard_normal((n_paths, 3)) * sqrt_dt
        # ξ_{t+dt} = ξ_t + J ξ_t dt + ε Σ dW
        drift = xi @ J.T  # (n_paths, 3)
        noise = dW @ eps_sigma.T
        xi = xi + drift * dt + noise

        if step % renorm_every == 0:
            norms = np.linalg.norm(xi, axis=1)
            # Guard against zero norms (extremely rare); replace with previous direction.
            norms = np.where(norms < 1e-300, 1.0, norms)
            log_stretches += np.log(norms)
            xi = xi / norms[:, None]

    T = n_steps * dt
    lam_per_path = log_stretches / T
    return float(np.mean(lam_per_path))


def stochastic_hopf_shift_numeric(
    jacobian: NDArray[np.float64],
    diffusion_matrix: NDArray[np.float64],
    *,
    epsilon_low: float = 0.05,
    epsilon_high: float = 0.20,
    n_paths: int = 200,
    n_steps: int = 5_000,
    dt: float = 1e-2,
    renorm_every: int = 50,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Estimate Λ in λ_1(ε) = α + ε² Λ + O(ε⁴) by two-point finite differences.

    Returns (Lambda, lambda_low, lambda_high) so the caller can sanity-check
    α ≈ lambda_low - ε_low² · Λ ≈ lambda_high - ε_high² · Λ.

    Compute cost: ~O(n_paths · n_steps) per ε — defaults give ~2×10^6 RHS
    evaluations total, runs in a few seconds. For a publication-grade Λ,
    use n_paths ≥ 10^4 and n_steps ≥ 10^4 (~10^8 evals, several minutes).
    """
    if not (epsilon_low > 0 and epsilon_high > epsilon_low):
        raise ValueError("require 0 < epsilon_low < epsilon_high")

    seed_low = seed
    seed_high = None if seed is None else seed + 1

    lam_low = top_lyapunov_exponent_linearised(
        jacobian,
        diffusion_matrix,
        epsilon=epsilon_low,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        renorm_every=renorm_every,
        seed=seed_low,
    )
    lam_high = top_lyapunov_exponent_linearised(
        jacobian,
        diffusion_matrix,
        epsilon=epsilon_high,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        renorm_every=renorm_every,
        seed=seed_high,
    )
    Lambda = (lam_high - lam_low) / (epsilon_high**2 - epsilon_low**2)
    return float(Lambda), float(lam_low), float(lam_high)


def compute_lambda_correction(
    simulator: object,
    kappa: float,
    *,
    equilibrium: tuple[float, float, float] | None = None,
    epsilon_low: float = 0.05,
    epsilon_high: float = 0.20,
    n_paths: int = 200,
    n_steps: int = 5_000,
    dt: float = 1e-2,
    renorm_every: int = 50,
    seed: int | None = None,
) -> float:
    """High-level wrapper: extract J and Σ from a `ReflexiveSimulator` at the
    equilibrium and run the two-point Λ estimator.

    The simulator is duck-typed (we only need `.drift`, `.params`, and
    `.gamma_aggregator`) so this function does not couple bifurcation theory
    to the simulator class hierarchy.

    For the noise structure: the variance equation has multiplicative noise
    ξ √v dW, but at the equilibrium v* = θ_v this is constant ξ √θ_v, so the
    additive-noise treatment of `top_lyapunov_exponent_linearised` is the
    correct leading-order linearisation. Cross-correlation ρ between dW^S and
    dW^v is folded into the diffusion matrix via Cholesky.

    Args:
        simulator: a ReflexiveSimulator-like object with `.drift(s, v, z)`,
            `.params.coupling`, `.params.base.{theta, xi, rho, v0}`, and
            `.initial_spot`.
        kappa: the coupling κ at which to evaluate Λ. Caller must have set
            simulator.params.coupling = kappa beforehand (or rely on a
            simulator that's already configured at this κ).
        equilibrium: x* = (log S*, v*, z*). If None, defaults to
            (log S_0, θ_v, 0) — the trivial equilibrium when G(S_0, θ_v, 0) = 0.
    """
    del kappa  # informational; the simulator's coupling is what matters
    sim_any: object = simulator
    base_theta = float(sim_any.params.base.theta)  # type: ignore[attr-defined]
    base_xi = float(sim_any.params.base.xi)  # type: ignore[attr-defined]
    base_rho = float(sim_any.params.base.rho)  # type: ignore[attr-defined]
    initial_spot = float(sim_any.initial_spot)  # type: ignore[attr-defined]

    if equilibrium is None:
        log_s_star = float(np.log(initial_spot))
        x_star = (log_s_star, base_theta, 0.0)
    else:
        x_star = equilibrium

    # Build Jacobian via finite differences from the simulator's drift.
    def drift(x: NDArray[np.float64]) -> NDArray[np.float64]:
        # x = (log S, v, z); simulator.drift takes (S, v, z)
        s = float(np.exp(x[0]))
        v = float(x[1])
        z = float(x[2])
        return np.asarray(sim_any.drift(s, v, z), dtype=np.float64)  # type: ignore[attr-defined]

    h = 1e-4
    x0 = np.asarray(x_star, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        e_j = np.zeros(3)
        e_j[j] = h
        J[:, j] = (drift(x0 + e_j) - drift(x0 - e_j)) / (2.0 * h)

    # Diffusion matrix at the equilibrium x* = (log S*, θ_v, 0):
    #   d(log S) ≈ √v dW^S       — coefficient √θ_v on independent BM 1
    #   dv ≈ ξ √v dW^v          — correlated with dW^S via ρ
    #   dz has no Brownian term  — row of zeros
    sqrt_theta = float(np.sqrt(max(base_theta, 0.0)))
    sigma_S = sqrt_theta
    sigma_v = base_xi * sqrt_theta
    # Cholesky of [[1, ρ], [ρ, 1]]: dW^S = dB1, dW^v = ρ dB1 + √(1-ρ²) dB2.
    sqrt_one_minus_rho2 = float(np.sqrt(max(1.0 - base_rho * base_rho, 0.0)))
    Sigma = np.array(
        [
            [sigma_S, 0.0, 0.0],
            [sigma_v * base_rho, sigma_v * sqrt_one_minus_rho2, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    Lambda, _, _ = stochastic_hopf_shift_numeric(
        J,
        Sigma,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        renorm_every=renorm_every,
        seed=seed,
    )
    return Lambda
