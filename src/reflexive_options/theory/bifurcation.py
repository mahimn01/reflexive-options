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
    # eigvals returns float64 for symmetric inputs and complex128 otherwise.
    # Cast to complex128 unconditionally so downstream arithmetic on .imag is well-typed.
    eig = np.asarray(np.linalg.eigvals(jacobian), dtype=np.complex128)
    order = np.argsort(-eig.real)
    return np.asarray(eig[order], dtype=np.complex128)


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
            raise ValueError("could not infer ω from Jacobian — no near-imaginary eigenvalue pair")
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
            d3 = ((f_pp - 2.0 * f_pk + f_mp) - (f_pm - 2.0 * f_mk + f_mm)) / (2.0 * h**3)
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
    d3_jkm = (f_ppp - f_ppm - f_pmp - f_mpp + f_pmm + f_mpm + f_mmp - f_mmm) / (8.0 * h**3)
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
        raise ValueError(f"n_steps ({n_steps}) must be a multiple of renorm_every ({renorm_every})")

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


# ---------------------------------------------------------------------------
# Closed-form Lyapunov coefficient for log-normal OI in moneyness
# (paper/theory.md §4.3). The aggregator G(a, v) — with a = log S — admits
# the closed form (Briggs 2003 Erf-Gaussian identity)
#
#   G(a, v) = (κ_u e^{-q T} / sqrt(2π τ²)) · e^{-a} · exp(-(a - m)² / (2 τ²))
#   τ²(v)   = σ_q² + v T
#   m(v)    = μ_q − (r − q + v/2) T
#
# because the Gaussian OI density in log-strike multiplied by the BS gamma
# (itself Gaussian in log-strike at fixed v, T) integrates analytically.
#
# All third-order partials of G at an arbitrary (a*, v*) are then explicit
# rational expressions, which we expose as `G_lognormal_oi_partials`. These
# replace the finite-difference tensor builder for the parametric case and
# eliminate the ℓ_1 numerical-noise issue.
#
# G has no z-dependence, so G_z = 0 and Jacobian (3) loses one entry. With
# σ² = v in the Heston backbone (∂_y σ² = 0, ∂_v σ² = 1), the characteristic
# polynomial reduces to a quadratic in κ:
#
#   H(κ) := c_1 c_2 − c_0 = G_y² (α + κ_v) κ²
#                          + (G_v β γ − G_y (α + κ_v)²) κ
#                          + (α κ_v (α + κ_v) − β γ / 2) = 0,
#
# whose positive root is the closed-form Hopf threshold κ*.
# ---------------------------------------------------------------------------


def _norm_pdf(x: float) -> float:
    return float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def G_lognormal_oi(
    log_spot: float,
    variance: float,
    *,
    mu_q: float,
    sigma_q: float,
    T_eff: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """Closed-form aggregate dealer-gamma G(a, v) for log-normal OI in moneyness.

    Integrates the log-normal OI density q(log K) = N(μ_q, σ_q²) against the
    Black-Scholes gamma at maturity T_eff, vol σ = √v, sign convention s ≡ +1
    (SqueezeMetrics SPX default). The product of two Gaussians in log K
    integrates analytically to a Gaussian in a := log S, modulated by 1/S = e^{-a}.

    Args:
        log_spot: a = log S.
        variance: v ≥ 0.
        mu_q: log-normal OI center in log-strike.
        sigma_q: log-normal OI std in log-strike (must be > 0).
        T_eff: representative maturity in years (must be > 0).
        coupling_units: the κ_units constant outside the integral.
        rate: risk-free rate r.
        dividend: dividend yield q (note: collides with OI density q; we use 'dividend').

    Returns:
        G(log_spot, variance) as a float in USD-per-unit-return.
    """
    if sigma_q <= 0.0:
        raise ValueError(f"sigma_q must be > 0, got {sigma_q}")
    if T_eff <= 0.0:
        raise ValueError(f"T_eff must be > 0, got {T_eff}")
    if variance < 0.0:
        raise ValueError(f"variance must be ≥ 0, got {variance}")

    tau2 = sigma_q * sigma_q + variance * T_eff
    m = mu_q - (rate - dividend + 0.5 * variance) * T_eff
    prefactor = coupling_units * np.exp(-dividend * T_eff - log_spot) / np.sqrt(2.0 * np.pi * tau2)
    return float(prefactor * np.exp(-((log_spot - m) ** 2) / (2.0 * tau2)))


def G_lognormal_oi_partials(
    *,
    a_star: float,
    v_star: float,
    mu_q: float,
    sigma_q: float,
    T_eff: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> dict[str, float]:
    """All partials of G(a, v) at (a_star, v_star) up to third order.

    Returned dict keys (numbered 'a' ↔ y in paper notation):
        'G', 'G_a', 'G_v', 'G_z',
        'G_aa', 'G_av', 'G_vv', 'G_az', 'G_vz', 'G_zz',
        'G_aaa', 'G_aav', 'G_avv', 'G_vvv',
        'G_aaz', 'G_avz', 'G_vvz', 'G_azz', 'G_vzz', 'G_zzz'.

    All z-partials are identically 0 (G has no z-dependence in this
    parameterization — see paper/theory.md §4.3).

    Derivation: G(a, v) = C(v) · e^{-(a - m(v))² / (2 τ²(v))} with
        C(v)   = κ_u e^{-q T} / sqrt(2π τ²(v))
        m(v)   = μ_q − (r − q + v/2) T
        τ²(v)  = σ_q² + v T
    Partials in a are Hermite-polynomial multiples of G; partials in v use
    chain rule through (m, τ²). Verified symbolically against sympy in
    notebooks/closed_form_ell1_derivation.py.
    """
    if sigma_q <= 0.0:
        raise ValueError(f"sigma_q must be > 0, got {sigma_q}")
    if T_eff <= 0.0:
        raise ValueError(f"T_eff must be > 0, got {T_eff}")

    # Shorthand
    Tt = T_eff
    sq2 = sigma_q * sigma_q
    tau2 = sq2 + v_star * Tt
    inv_tau2 = 1.0 / tau2
    m = mu_q - (rate - dividend + 0.5 * v_star) * Tt
    delta = a_star - m  # appears everywhere
    G0 = G_lognormal_oi(
        a_star,
        v_star,
        mu_q=mu_q,
        sigma_q=sigma_q,
        T_eff=Tt,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )

    # --- a-derivatives at fixed v ---
    # f(a) = -a - (a - m)² / (2 τ²)  (the log of G as function of a alone)
    # f'(a) = -1 - (a - m)/τ²
    # f''(a) = -1/τ²
    # f'''(a) = 0.
    # G_a = G * f'(a); G_aa = G * (f'² + f''); G_aaa = G * (f'³ + 3 f' f'' + f''').
    fp = -1.0 - delta * inv_tau2
    fpp = -inv_tau2
    fppp = 0.0
    G_a = G0 * fp
    G_aa = G0 * (fp * fp + fpp)
    G_aaa = G0 * (fp**3 + 3.0 * fp * fpp + fppp)

    # --- v-derivatives via chain rule ---
    # log G = log C(v) + g(a, v),  g(a, v) = -a - (a - m(v))² / (2 τ²(v))
    # Let h(v) = log C(v) = -q T - 0.5 log(2π τ²(v)) + log κ_u
    # h'(v)   = -0.5 · T / τ²        (since dτ²/dv = T)
    # h''(v)  = +0.5 · T² / τ⁴
    # h'''(v) = -T³ / τ⁶
    # m'(v)   = -T/2;  m''(v) = m'''(v) = 0.
    Tt2 = Tt * Tt
    Tt3 = Tt2 * Tt
    inv_tau4 = inv_tau2 * inv_tau2
    inv_tau6 = inv_tau4 * inv_tau2

    hp = -0.5 * Tt * inv_tau2
    hpp = 0.5 * Tt2 * inv_tau4
    hppp = -Tt3 * inv_tau6

    # g(a, v) = -a - δ²/(2 τ²) where δ = a - m(v); δ_v = +T/2; (τ²)_v = T.
    # All v-partials below are derived term-by-term and verified against sympy
    # in `notebooks/closed_form_ell1_derivation.py`.
    #
    # g_v   = -δ T / (2 τ²)  +  δ² T / (2 τ⁴)
    # g_vv  = -T² / (4 τ²)   +  δ T² / τ⁴   −  δ² T² / τ⁶
    # g_vvv = (3/4) T³ / τ⁴  −  3 δ T³ / τ⁶  +  3 δ² T³ / τ⁸
    inv_tau8 = inv_tau4 * inv_tau4
    gp_v = -0.5 * delta * Tt * inv_tau2 + 0.5 * delta * delta * Tt * inv_tau4
    gpp_v = -0.25 * Tt2 * inv_tau2 + delta * Tt2 * inv_tau4 - delta * delta * Tt2 * inv_tau6
    gppp_v = (
        0.75 * Tt3 * inv_tau4 - 3.0 * delta * Tt3 * inv_tau6 + 3.0 * delta * delta * Tt3 * inv_tau8
    )

    # log G v-partials L_k := ∂^k log G / ∂v^k at (a*, v*).
    L1 = hp + gp_v
    L2 = hpp + gpp_v
    L3 = hppp + gppp_v

    G_v = G0 * L1
    G_vv = G0 * (L1 * L1 + L2)
    G_vvv = G0 * (L1**3 + 3.0 * L1 * L2 + L3)

    # --- mixed a, v derivatives ---
    # ∂g/∂a = fp.
    # ∂²g/(∂a ∂v) = d/dv[fp] = -[δ_v / τ² + δ · (-T) / τ⁴] = -T/(2 τ²) + δ T / τ⁴.
    # ∂³g/(∂a² ∂v) = d/dv[fpp] = d/dv[-1/τ²] = T / τ⁴.
    # ∂³g/(∂a ∂v²) = d/dv[g_av] = T²/(2 τ⁴) + T (T/2 / τ⁴ - 2 δ T / τ⁶)
    #             = T² / τ⁴ - 2 T² δ / τ⁶.
    g_av = -0.5 * Tt * inv_tau2 + delta * Tt * inv_tau4
    g_aav = Tt * inv_tau4
    g_avv = Tt2 * inv_tau4 - 2.0 * Tt2 * delta * inv_tau6

    # G mixed via the product rule on log G = log C + g  (note ∂h/∂a = 0):
    #   G_av  = G [ (∂a log G)(∂v log G) + ∂²log G/(∂a ∂v) ]
    #         = G [ fp · L1 + g_av ]
    #   G_aav = G [ (∂a log G)² ∂v log G + 2 ∂a log G · ∂²log G/(∂a ∂v)
    #             + ∂² log G/∂a² · ∂v log G + ∂³ log G/(∂a² ∂v) ]
    #         = G [ fp² L1 + 2 fp g_av + fpp L1 + g_aav ]
    #   G_avv = G [ ∂a log G · (∂v log G)² + 2 ∂v log G · ∂²log G/(∂a ∂v)
    #             + ∂a log G · ∂² log G/∂v² + ∂³log G/(∂a ∂v²) ]
    #         = G [ fp L1² + 2 L1 g_av + fp L2 + g_avv ]
    G_av = G0 * (fp * L1 + g_av)
    G_aav = G0 * (fp * fp * L1 + 2.0 * fp * g_av + fpp * L1 + g_aav)
    G_avv = G0 * (fp * L1 * L1 + 2.0 * L1 * g_av + fp * L2 + g_avv)

    # G_z and all z-mixed partials are zero
    return {
        "G": G0,
        "G_a": G_a,
        "G_v": G_v,
        "G_z": 0.0,
        "G_aa": G_aa,
        "G_av": G_av,
        "G_vv": G_vv,
        "G_az": 0.0,
        "G_vz": 0.0,
        "G_zz": 0.0,
        "G_aaa": G_aaa,
        "G_aav": G_aav,
        "G_avv": G_avv,
        "G_vvv": G_vvv,
        "G_aaz": 0.0,
        "G_avz": 0.0,
        "G_vvz": 0.0,
        "G_azz": 0.0,
        "G_vzz": 0.0,
        "G_zzz": 0.0,
    }


def kappa_star_lognormal_oi(
    *,
    G_y: float,
    G_v: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float]:
    """Closed-form Hopf threshold κ* for the log-normal OI parameterization.

    With G_z = 0 and σ² = v (Heston backbone), the Routh-Hurwitz discriminant
    H(κ) = c_1 c_2 − c_0 reduces to a quadratic in κ:

        H(κ) = A_2 κ² + A_1 κ + A_0,
        A_2  = G_y² · (α + κ_v),
        A_1  = G_v · β γ − G_y · (α + κ_v)²,
        A_0  = α κ_v (α + κ_v) − β γ / 2.

    Positive real roots, if any, are returned together with the corresponding
    Hopf frequency ω* = √c_1(κ*).

    Liu's full Hopf criterion requires three things at the candidate κ*:
        (i)   H(κ*) = 0                  ← the quadratic root
        (ii)  c_2(κ*) > 0                ← positive linear coefficient
        (iii) c_0(κ*) > 0                ← positive constant coefficient

    Failing (ii) or (iii) means the candidate sits at an unstable equilibrium
    where the eigenvalue real-parts cross zero in a non-Hopf manner. We raise
    `ValueError` in either case so a bad candidate cannot silently propagate
    into the Lyapunov-coefficient pipeline.

    Args:
        G_y: ∂G/∂a at the equilibrium (= ∂G/∂y in deviation variables).
        G_v: ∂G/∂v at the equilibrium.
        kappa_v: Heston mean-reversion speed (must be > 0).
        alpha: memory-channel decay (must be > 0).
        beta: memory-channel intake.
        gamma: leverage feedback (≥ 0).

    Returns:
        (kappa_star, omega_star). If no positive real root with ω*² > 0 exists,
        OR Routh-Hurwitz positivity (c_2 > 0 ∧ c_0 > 0) fails at the candidate,
        raises ValueError.
    """
    if kappa_v <= 0.0:
        raise ValueError(f"kappa_v must be > 0, got {kappa_v}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    A_total = alpha + kappa_v  # Σ of decay rates
    L = beta * gamma  # leverage flux through (z, v) loop
    if abs(G_y) < 1e-300:
        # H(κ) collapses to G_v L κ + (α κ_v A − L/2) = 0  (linear in κ)
        if abs(G_v * L) < 1e-300:
            raise ValueError("Hopf condition degenerate: G_y = 0 and G_v β γ = 0")
        kappa_star = (0.5 * L - alpha * kappa_v * A_total) / (G_v * L)
        if kappa_star <= 0.0:
            raise ValueError(f"linear Hopf root non-positive: κ* = {kappa_star}")
    else:
        A2 = G_y * G_y * A_total
        A1 = G_v * L - G_y * A_total * A_total
        A0 = alpha * kappa_v * A_total - 0.5 * L
        disc = A1 * A1 - 4.0 * A2 * A0
        if disc < 0.0:
            raise ValueError(
                f"no real Hopf root: discriminant {disc:.3e} < 0; check G_y, G_v signs"
            )
        sqrt_disc = float(np.sqrt(disc))
        roots = ((-A1 - sqrt_disc) / (2.0 * A2), (-A1 + sqrt_disc) / (2.0 * A2))
        positive = [r for r in roots if r > 0.0]
        if not positive:
            raise ValueError(f"no positive Hopf root: roots = {roots}")
        # Prefer the smallest positive root (first crossing as κ ramps up).
        kappa_star = float(min(positive))

    # Routh-Hurwitz positivity at the candidate κ* — Liu's criterion (V1-W2).
    # With G_z = 0 and σ² = v: a(κ) = κ G_y, b(κ) = κ G_v − ½.
    #   c_2 = -trace(J) = -a + κ_v + α
    #   c_0 = -det(J)   = -a κ_v α + β(κ G_z κ_v − b γ)
    #               = -a κ_v α − b β γ            (since G_z = 0)
    a_at_star = kappa_star * G_y
    b_at_star = kappa_star * G_v - 0.5
    c_2 = -a_at_star + kappa_v + alpha
    c_0 = -a_at_star * kappa_v * alpha - b_at_star * beta * gamma
    if c_2 <= 0.0:
        raise ValueError(
            f"κ* candidate violates Routh-Hurwitz positivity: c_2 = {c_2:.3e} ≤ 0 "
            f"at κ* = {kappa_star} (Liu's criterion)"
        )
    if c_0 <= 0.0:
        raise ValueError(
            f"κ* candidate violates Routh-Hurwitz positivity: c_0 = {c_0:.3e} ≤ 0 "
            f"at κ* = {kappa_star} (Liu's criterion)"
        )

    # ω*² = c_1(κ*) = -a κ_v - a α + κ_v α
    omega_sq = -a_at_star * (kappa_v + alpha) + kappa_v * alpha
    if omega_sq <= 0.0:
        raise ValueError(
            f"ω*² = {omega_sq:.3e} ≤ 0 at κ* = {kappa_star}; "
            "Hopf candidate is in a stable region (Routh-Hurwitz violated)"
        )
    return kappa_star, float(np.sqrt(omega_sq))


def _build_lognormal_tensors(
    partials: dict[str, float],
    *,
    kappa: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Assemble (J, B, C) for the 3D reflexive skeleton at coupling κ.

    State order is (y, u, z) := (a − a*, v − v*, z − z*).

    Drift components (paper/theory.md eq. 5):
        f_1(y, u, z) = μ - σ²(y, u)/2 + κ G(y, u, z)        ≈ κ G − u/2 + ...
        f_2(y, u, z) = -κ_v u + γ z
        f_3(y, u, z) = -α z + β y

    With σ² = v (so σ²(y, u) = θ_v + u, contributing only the constant -θ_v/2
    that the equilibrium balances and the linear term -u/2 already in J), the
    quadratic / cubic Taylor coefficients of f_1 come entirely from κ G(y, u, z).
    f_2 and f_3 are linear so contribute zero to B and C.

    Returns (J, B, C) where:
        J[i, j]       = ∂f_i / ∂x_j               at x = 0
        B[i, j, k]    = ∂² f_i / ∂x_j ∂x_k        at x = 0  (symmetric in j,k)
        C[i, j, k, l] = ∂³ f_i / ∂x_j ∂x_k ∂x_l   at x = 0  (symmetric in j,k,l)
    """
    G_a = partials["G_a"]
    G_v = partials["G_v"]
    G_aa = partials["G_aa"]
    G_av = partials["G_av"]
    G_vv = partials["G_vv"]
    G_aaa = partials["G_aaa"]
    G_aav = partials["G_aav"]
    G_avv = partials["G_avv"]
    G_vvv = partials["G_vvv"]
    # All G_z, G_zz, etc. are 0 by construction.

    # Jacobian of (y, u, z): linearisation of f_1, f_2, f_3.
    a_lin = kappa * G_a  # ∂f_1/∂y
    b_lin = kappa * G_v - 0.5  # ∂f_1/∂u  (the -1/2 from -∂_v σ²/2 = -1/2)
    J = np.array(
        [
            [a_lin, b_lin, 0.0],
            [0.0, -kappa_v, gamma],
            [beta, 0.0, -alpha],
        ],
        dtype=np.float64,
    )

    # B: only f_1 contributes; second partials of f_1 = κ · second partials of G,
    # and σ² = v has no quadratic terms.
    B = np.zeros((3, 3, 3), dtype=np.float64)
    B[0, 0, 0] = kappa * G_aa  # ∂²f_1/∂y² = κ G_aa
    B[0, 0, 1] = B[0, 1, 0] = kappa * G_av  # ∂²f_1/∂y∂u
    B[0, 1, 1] = kappa * G_vv  # ∂²f_1/∂u²
    # All other entries (involving z, or in f_2/f_3) are 0.

    # C: only f_1 contributes.
    C = np.zeros((3, 3, 3, 3), dtype=np.float64)
    C[0, 0, 0, 0] = kappa * G_aaa  # ∂³f_1/∂y³
    # ∂³f_1/∂y² ∂u = κ G_aav (3 perms: (y,y,u), (y,u,y), (u,y,y))
    for j, k, m in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        C[0, j, k, m] = kappa * G_aav
    # ∂³f_1/∂y ∂u² = κ G_avv (3 perms)
    for j, k, m in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        C[0, j, k, m] = kappa * G_avv
    C[0, 1, 1, 1] = kappa * G_vvv  # ∂³f_1/∂u³

    return J, B, C


def lyapunov_coefficient_lognormal_oi(
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
) -> tuple[float, float, float]:
    """Closed-form first Lyapunov coefficient ℓ_1 at the Hopf point κ*.

    Pipeline:
        1. Compute G(a*, v*) and all third-order partials in closed form
           via `G_lognormal_oi_partials`.
        2. Solve the (now-quadratic) Routh-Hurwitz condition H(κ*) = 0 for
           the smallest positive root via `kappa_star_lognormal_oi`.
        3. Assemble (J, B, C) at κ* (`_build_lognormal_tensors`) and feed
           them to the existing Kuznetsov 2004 formula via
           `compute_lyapunov_coefficient`.

    Defaulting v_star = θ_v matches the equilibrium of the variance-OU
    when γ z* = 0 (i.e. y* = 0 is the natural ATM equilibrium).

    Returns:
        (kappa_star, omega_star, ell_1).

    Sign convention: ℓ_1 < 0 ⇒ supercritical (stable limit cycle for κ > κ*),
    ℓ_1 > 0 ⇒ sub-critical (unstable cycle, hysteresis).
    """
    if v_star is None:
        v_star = theta_v

    partials = G_lognormal_oi_partials(
        a_star=a_star,
        v_star=v_star,
        mu_q=mu_q,
        sigma_q=sigma_q,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )

    kappa_star, omega_star = kappa_star_lognormal_oi(
        G_y=partials["G_a"],
        G_v=partials["G_v"],
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    J, B, C = _build_lognormal_tensors(
        partials,
        kappa=kappa_star,
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    ell_1 = compute_lyapunov_coefficient(J, B, C, omega=omega_star)
    return kappa_star, omega_star, ell_1


# ---------------------------------------------------------------------------
# Codim-2 bifurcation analysis (paper §3.6).
#
# Two codim-2 phenomena live on the boundary of the Hopf region:
#
#   1. Bautin (degenerate Hopf): ℓ_1(σ_q, γ) = 0 along a 1-dimensional locus
#      in (σ_q, γ) space at fixed (μ_q, T_eff, α, β, κ_v, θ_v). Crossing this
#      locus toggles the supercritical → sub-critical transition. Operationally:
#      stable limit cycle → hysteresis + abrupt jumps. Local 2D normal form
#      after centre-manifold reduction (Kuznetsov 2004 §8.3):
#
#          ẋ = β_1 x − ω y + a₁ x (x² + y²) + b₁ x (x² + y²)²,
#          ẏ = ω x + β_1 y + a₁ y (x² + y²) + b₁ y (x² + y²)²,
#
#      with ℓ_1 = a₁/ω. The second Lyapunov coefficient ℓ_2 = b₁/ω fixes the
#      sign of the cusp.
#
#   2. Bogdanov-Takens (BT): saddle-node curve coalesces with the Hopf curve.
#      Equivalently, the Routh-Hurwitz polynomial H(κ) = 0 AND c_0(κ) = 0
#      simultaneously at the same κ (the Jacobian acquires a double-zero
#      eigenvalue). With c_0 linear in κ in the closed-form parameterization
#      (G_z = 0, σ² = v), the saddle-node κ is
#
#          κ_SN = ½ β γ / (G_y α κ_v + G_v β γ),
#
#      and BT is the codim-2 condition H(κ_SN) = 0 ∧ κ_SN > 0 in (σ_q, γ).
#      Local normal form (Kuznetsov 2004 §8.4):
#
#          ẋ = y,    ẏ = β_1 + β_2 x + x² ± xy,
#
#      generating saddle-node, Hopf, and homoclinic curves emanating from the
#      BT point — the canonical "burst-relax" generator. For the canonical
#      log-normal-OI specification, κ_SN < 0 throughout the physical
#      (σ_q, γ) range because G_v < 0 dominates the denominator, so BT does
#      not occur at any (σ_q, γ) > 0 — the dealer-gamma + leverage parameter
#      regime is structurally Hopf-only.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BautinScanResult:
    """Output of `bautin_curve_scan` over a (σ_q, γ) grid.

    `ell_1_grid` carries NaN where no Hopf root exists (discriminant < 0 or
    Routh-Hurwitz positivity violated). `regime_grid` encodes the four
    codim-2 regions:

        0  no Hopf (no positive real root of H(κ) or RH positivity violated)
        1  supercritical Hopf (ℓ_1 < 0)
        2  Bautin tube (|ℓ_1| ≤ bautin_tol — the numerical degenerate locus)
        3  sub-critical Hopf (ℓ_1 > 0)
    """

    sigma_q_grid: NDArray[np.float64]
    gamma_grid: NDArray[np.float64]
    ell_1_grid: NDArray[np.float64]  # shape (n_gamma, n_sigma_q)
    kappa_star_grid: NDArray[np.float64]
    omega_star_grid: NDArray[np.float64]
    regime_grid: NDArray[np.int8]


def kappa_saddle_node_lognormal_oi(
    *,
    G_y: float,
    G_v: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Saddle-node coupling κ_SN for the closed-form log-normal-OI Jacobian.

    The constant Routh-Hurwitz coefficient is

        c_0(κ) = -κ G_y α κ_v − (κ G_v − ½) β γ
               = -κ (G_y α κ_v + G_v β γ) + ½ β γ,

    linear in κ. A saddle-node bifurcation at the equilibrium occurs where
    c_0(κ_SN) = 0 (the Jacobian determinant vanishes), giving

        κ_SN = ½ β γ / (G_y α κ_v + G_v β γ).

    For the natural log-normal OI specification (G_y > 0, G_v < 0), the
    denominator can be of either sign; if G_v β γ dominates G_y α κ_v in
    magnitude, κ_SN < 0 and the saddle-node is unphysical.

    Returns:
        κ_SN. Note this may be negative or non-finite — caller must check
        sign before treating as a valid bifurcation locus.

    Raises:
        ValueError if the denominator is exactly zero (degenerate parameter
        configuration).
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if kappa_v <= 0.0:
        raise ValueError(f"kappa_v must be > 0, got {kappa_v}")
    denom = G_y * alpha * kappa_v + G_v * beta * gamma
    if abs(denom) < 1e-300:
        raise ValueError(
            f"saddle-node denominator vanishes (G_y α κ_v + G_v β γ = {denom:.3e}); "
            "BT-degenerate parameter configuration"
        )
    return 0.5 * beta * gamma / denom


def bogdanov_takens_residual_lognormal_oi(
    *,
    sigma_q: float,
    gamma: float,
    mu_q: float,
    T_eff: float,
    kappa_v: float,
    theta_v: float,
    alpha: float,
    beta: float,
    a_star: float,
    v_star: float | None = None,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> tuple[float, float]:
    """Joint BT residual at (σ_q, γ): (κ_SN, H(κ_SN)).

    Bogdanov-Takens occurs iff κ_SN > 0 AND H(κ_SN) = 0 simultaneously. Both
    are needed: a positive saddle-node coupling that also annihilates the
    Routh-Hurwitz quadratic.

    Returns:
        (kappa_SN, H_at_kappa_SN). `kappa_SN <= 0` means the saddle-node curve
        is unphysical at this (σ_q, γ); `H_at_kappa_SN = 0` is the BT
        condition itself.
    """
    if v_star is None:
        v_star = theta_v
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
    G_y = p["G_a"]
    G_v = p["G_v"]
    try:
        k_sn = kappa_saddle_node_lognormal_oi(
            G_y=G_y, G_v=G_v, kappa_v=kappa_v, alpha=alpha, beta=beta, gamma=gamma
        )
    except ValueError:
        return float("nan"), float("nan")

    A_total = alpha + kappa_v
    L = beta * gamma
    A2 = G_y * G_y * A_total
    A1 = G_v * L - G_y * A_total * A_total
    A0 = alpha * kappa_v * A_total - 0.5 * L
    H_at = A2 * k_sn * k_sn + A1 * k_sn + A0
    return float(k_sn), float(H_at)


def bautin_curve_scan(
    *,
    sigma_q_grid: NDArray[np.float64],
    gamma_grid: NDArray[np.float64],
    mu_q: float,
    T_eff: float,
    kappa_v: float,
    theta_v: float,
    alpha: float,
    beta: float,
    a_star: float,
    v_star: float | None = None,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    bautin_tol: float = 1e-3,
) -> BautinScanResult:
    """Sweep ℓ_1 over a (σ_q, γ) grid and classify each cell into the four
    codim-2 regions of paper §3.6.

    Args:
        sigma_q_grid: ascending σ_q values (1D).
        gamma_grid: ascending γ values (1D).
        bautin_tol: cells with |ℓ_1| ≤ bautin_tol are flagged as belonging to
            the (numerically thickened) Bautin curve. The default 1e-3 is a
            sensible visual width at the canonical scale; reduce for a thinner
            curve, increase to highlight the codim-1 transition zone.
        Other args: as in `lyapunov_coefficient_lognormal_oi`.

    Returns:
        BautinScanResult with shape (n_gamma, n_sigma_q) arrays and a
        regime classification suitable for direct rendering as a phase
        diagram.

    Raises:
        ValueError if the grids are not strictly ascending.
    """
    sq = np.asarray(sigma_q_grid, dtype=np.float64)
    gam = np.asarray(gamma_grid, dtype=np.float64)
    if sq.ndim != 1 or gam.ndim != 1:
        raise ValueError("sigma_q_grid and gamma_grid must be 1D")
    if not (np.all(np.diff(sq) > 0) and np.all(np.diff(gam) > 0)):
        raise ValueError("sigma_q_grid and gamma_grid must be strictly ascending")
    if bautin_tol <= 0:
        raise ValueError(f"bautin_tol must be > 0, got {bautin_tol}")

    n_g = len(gam)
    n_s = len(sq)
    ell = np.full((n_g, n_s), np.nan, dtype=np.float64)
    ks = np.full((n_g, n_s), np.nan, dtype=np.float64)
    om = np.full((n_g, n_s), np.nan, dtype=np.float64)
    regime = np.zeros((n_g, n_s), dtype=np.int8)

    for i, g in enumerate(gam):
        for j, s in enumerate(sq):
            try:
                k_star, omega_star, ell_1 = lyapunov_coefficient_lognormal_oi(
                    mu_q=mu_q,
                    sigma_q=float(s),
                    T_eff=T_eff,
                    kappa_v=kappa_v,
                    theta_v=theta_v,
                    alpha=alpha,
                    beta=beta,
                    gamma=float(g),
                    a_star=a_star,
                    v_star=v_star,
                    coupling_units=coupling_units,
                    rate=rate,
                    dividend=dividend,
                )
            except ValueError:
                regime[i, j] = 0  # no Hopf
                continue
            ks[i, j] = k_star
            om[i, j] = omega_star
            ell[i, j] = ell_1
            if abs(ell_1) <= bautin_tol:
                regime[i, j] = 2  # Bautin tube
            elif ell_1 < 0:
                regime[i, j] = 1  # supercritical
            else:
                regime[i, j] = 3  # sub-critical

    return BautinScanResult(
        sigma_q_grid=sq,
        gamma_grid=gam,
        ell_1_grid=ell,
        kappa_star_grid=ks,
        omega_star_grid=om,
        regime_grid=regime,
    )


def find_bautin_anchors(
    scan: BautinScanResult,
    *,
    n_anchors: int = 6,
) -> list[tuple[float, float, float]]:
    """Extract anchor (σ_q, γ, κ★) triples on the Bautin curve ℓ_1 = 0.

    For each γ row of the scan grid, locate the σ_q at which ℓ_1 crosses zero
    via linear interpolation between adjacent grid cells of opposite sign.
    Returns up to `n_anchors` anchors evenly spaced over the γ range that
    contains a sign change.

    The returned κ★ at each anchor is also linearly interpolated, giving a
    convenient summary table of the codim-2 locus.
    """
    sq = scan.sigma_q_grid
    gam = scan.gamma_grid
    ell = scan.ell_1_grid
    ks = scan.kappa_star_grid

    crossings: list[tuple[float, float, float]] = []
    for i, g in enumerate(gam):
        row = ell[i, :]
        # Find first sign change in this row (ignore NaN cells)
        for j in range(len(sq) - 1):
            a, b = row[j], row[j + 1]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if a == 0.0:
                crossings.append((float(sq[j]), float(g), float(ks[i, j])))
                break
            if (a < 0) != (b < 0):  # sign change
                t = -a / (b - a)
                sq_cross = float(sq[j] + t * (sq[j + 1] - sq[j]))
                k_a, k_b = ks[i, j], ks[i, j + 1]
                if np.isfinite(k_a) and np.isfinite(k_b):
                    k_cross = float(k_a + t * (k_b - k_a))
                else:
                    k_cross = float("nan")
                crossings.append((sq_cross, float(g), k_cross))
                break

    if not crossings:
        return []

    # Subsample to n_anchors evenly across the γ range that has crossings.
    if len(crossings) <= n_anchors:
        return crossings
    idx = np.linspace(0, len(crossings) - 1, n_anchors).round().astype(int)
    return [crossings[i] for i in idx]


# ---------------------------------------------------------------------------
# Mixture-of-K-lognormals generalization of the §4.3 closed form
# (paper/theory.md §4.3.7, paper/mixture_oi_lyapunov.md).
#
# Empirical SPX open-interest grids are multi-modal — ATM concentration plus
# round-strike spikes, calendar-near-expiry clustering, dealer-portfolio-driven
# OTM tails. The single-lognormal closed form of §4.3 is only robust to mild
# bimodality (≤ Δ=0.10); §3.6 reports 119% relative error at Δ=0.20.
#
# Generalization. Take q(log K) = Σ_k w_k · N(log K; μ_k, σ_k²) with w_k ≥ 0
# and Σ_k w_k = 1. By linearity of the dealer-gamma aggregator G in the OI
# density (Eq. 14), the aggregate is itself a mixture:
#
#     G(a, v) = Σ_k w_k · G_k(a, v)
#
# where each G_k is the closed-form single-lognormal aggregator (Eq. 15a) with
# its own (μ_k, σ_k). All partials of G are linear combinations of the
# single-component partials, so the same Routh-Hurwitz quadratic (Eq. 16)
# applies with (G_y, G_v) replaced by the mixture sums Σ_k w_k (G_{y,k}, G_{v,k}).
# The Hopf threshold κ★ is the smallest positive root of that quadratic and the
# first Lyapunov coefficient ℓ_1 is obtained by passing the multilinear (B, C)
# tensors — themselves linear in {w_k} — into the existing Kuznetsov pipeline.
#
# Mathematically this is just "use the mixture partials in place of the
# single-component partials". The closed-form structure of κ★ is preserved,
# and ℓ_1 remains a rational expression in {w_k, μ_k, σ_k}_{k=1}^K and
# (κ_v, α, β, γ).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureOIComponent:
    """One log-normal component of a mixture-OI density q(log K).

    All fields are in log-strike units (so `mu_q` ≈ log S for ATM, `sigma_q`
    is the per-component standard deviation in log-strike). Weights are
    interpreted in the same scale as the other components — they need not
    sum to 1, the mixture machinery normalises internally.
    """

    weight: float
    mu_q: float
    sigma_q: float

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"weight must be ≥ 0, got {self.weight}")
        if self.sigma_q <= 0.0:
            raise ValueError(f"sigma_q must be > 0, got {self.sigma_q}")


def _normalize_components(
    components: list[MixtureOIComponent] | tuple[MixtureOIComponent, ...],
) -> list[MixtureOIComponent]:
    """Return a copy of `components` with weights summing to 1 (no other changes)."""
    if len(components) == 0:
        raise ValueError("mixture_components must contain ≥ 1 component")
    total = sum(c.weight for c in components)
    if total <= 0.0:
        raise ValueError(f"sum of mixture weights must be > 0, got {total}")
    return [
        MixtureOIComponent(weight=c.weight / total, mu_q=c.mu_q, sigma_q=c.sigma_q)
        for c in components
    ]


# Keys of the partials dictionary returned by `G_lognormal_oi_partials` — used
# below to drive the linear mixture combination in a single loop.
_PARTIAL_KEYS: tuple[str, ...] = (
    "G",
    "G_a",
    "G_v",
    "G_z",
    "G_aa",
    "G_av",
    "G_vv",
    "G_az",
    "G_vz",
    "G_zz",
    "G_aaa",
    "G_aav",
    "G_avv",
    "G_vvv",
    "G_aaz",
    "G_avz",
    "G_vvz",
    "G_azz",
    "G_vzz",
    "G_zzz",
)


def G_mixture_lognormal_oi(
    log_spot: float,
    variance: float,
    *,
    mixture_components: list[MixtureOIComponent] | tuple[MixtureOIComponent, ...],
    T_eff: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """Closed-form aggregate dealer-gamma G(a, v) for a mixture-of-K-lognormals OI.

    By linearity of the dealer-gamma aggregator in the OI density,

        G(a, v) = Σ_k w_k · G_k(a, v),

    where each G_k is the single-lognormal closed form `G_lognormal_oi` with
    component parameters (μ_k, σ_k). Weights are normalised internally.

    Args:
        log_spot, variance: state at which to evaluate G.
        mixture_components: list of `MixtureOIComponent`. K ≥ 1.
        T_eff, coupling_units, rate, dividend: as in `G_lognormal_oi`.

    Returns:
        G(log_spot, variance).
    """
    comps = _normalize_components(list(mixture_components))
    total = 0.0
    for c in comps:
        total += c.weight * G_lognormal_oi(
            log_spot,
            variance,
            mu_q=c.mu_q,
            sigma_q=c.sigma_q,
            T_eff=T_eff,
            coupling_units=coupling_units,
            rate=rate,
            dividend=dividend,
        )
    return float(total)


def G_mixture_lognormal_oi_partials(
    *,
    a_star: float,
    v_star: float,
    mixture_components: list[MixtureOIComponent] | tuple[MixtureOIComponent, ...],
    T_eff: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> dict[str, float]:
    """All partials of the mixture G(a, v) at (a_star, v_star) up to third order.

    Multi-linearity of differentiation gives

        ∂^|α| G / ∂x^α = Σ_k w_k · ∂^|α| G_k / ∂x^α

    for every multi-index α. We therefore delegate to `G_lognormal_oi_partials`
    for each component and form the weighted sum. K=1 collapses to the
    single-lognormal closed form to machine precision.

    Args:
        a_star, v_star: equilibrium location.
        mixture_components: list of `MixtureOIComponent`. K ≥ 1.
        T_eff, coupling_units, rate, dividend: as in `G_lognormal_oi_partials`.

    Returns:
        Dict with the same keys as `G_lognormal_oi_partials`. All z-partials
        are 0 since each component is z-independent.
    """
    comps = _normalize_components(list(mixture_components))
    out: dict[str, float] = {k: 0.0 for k in _PARTIAL_KEYS}
    for c in comps:
        pk = G_lognormal_oi_partials(
            a_star=a_star,
            v_star=v_star,
            mu_q=c.mu_q,
            sigma_q=c.sigma_q,
            T_eff=T_eff,
            coupling_units=coupling_units,
            rate=rate,
            dividend=dividend,
        )
        for key in _PARTIAL_KEYS:
            out[key] += c.weight * pk[key]
    return out


def kappa_star_mixture_lognormal_oi(
    *,
    mixture_components: list[MixtureOIComponent] | tuple[MixtureOIComponent, ...],
    T_eff: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    a_star: float,
    v_star: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> tuple[float, float]:
    """Closed-form Hopf threshold κ* for the mixture-OI parameterization.

    Because the Routh-Hurwitz coefficients are linear in (G_y, G_v) and these
    in turn are linear combinations of the per-component partials, the
    *same* quadratic Eq. 16 governs the mixture case — only the (G_y, G_v)
    fed into `kappa_star_lognormal_oi` change:

        G_y(mix) = Σ_k w_k · G_{y,k},
        G_v(mix) = Σ_k w_k · G_{v,k}.

    K=1 recovers the single-lognormal `kappa_star_lognormal_oi` to machine
    precision (verified in tests).

    Returns:
        (kappa_star, omega_star). Same failure modes as
        `kappa_star_lognormal_oi` — propagates ValueError on degenerate /
        non-Hopf regimes.
    """
    p = G_mixture_lognormal_oi_partials(
        a_star=a_star,
        v_star=v_star,
        mixture_components=mixture_components,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )
    return kappa_star_lognormal_oi(
        G_y=p["G_a"],
        G_v=p["G_v"],
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def lyapunov_coefficient_mixture_lognormal_oi(
    *,
    mixture_components: list[MixtureOIComponent] | tuple[MixtureOIComponent, ...],
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
) -> tuple[float, float, float]:
    """Closed-form first Lyapunov coefficient ℓ_1 for the mixture-OI case.

    Pipeline (mirrors `lyapunov_coefficient_lognormal_oi`):
        1. Compute the mixture partials of G via
           `G_mixture_lognormal_oi_partials` (linear in the {w_k}).
        2. Solve the quadratic Routh-Hurwitz condition H(κ*) = 0 via
           `kappa_star_lognormal_oi` on the mixture (G_y, G_v).
        3. Assemble (J, B, C) at κ* via `_build_lognormal_tensors` — the same
           tensor assembly works because every multilinear tensor inherits the
           same linearity in the {w_k}.
        4. Feed (J, B, C, ω*) to the Kuznetsov 2004 formula via
           `compute_lyapunov_coefficient`.

    K=1 collapses to `lyapunov_coefficient_lognormal_oi` to machine precision.

    Returns:
        (kappa_star, omega_star, ell_1). Sign convention: ℓ_1 < 0 supercritical,
        ℓ_1 > 0 sub-critical (cf. Kuznetsov 2004 §3.5).
    """
    if v_star is None:
        v_star = theta_v

    partials = G_mixture_lognormal_oi_partials(
        a_star=a_star,
        v_star=v_star,
        mixture_components=mixture_components,
        T_eff=T_eff,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )

    kappa_star, omega_star = kappa_star_lognormal_oi(
        G_y=partials["G_a"],
        G_v=partials["G_v"],
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    J, B, C = _build_lognormal_tensors(
        partials,
        kappa=kappa_star,
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    ell_1 = compute_lyapunov_coefficient(J, B, C, omega=omega_star)
    return kappa_star, omega_star, ell_1


# ---------------------------------------------------------------------------
# No-Hopf-wedge taxonomy (paper §3.5 closing-the-bifurcation analysis).
#
# The §3.5 closed-form quadratic H(κ) = A_2 κ^2 + A_1 κ + A_0 has discriminant
#
#     D := (G_v L − G_y A^2)^2 − 4 G_y^2 A (M A − L/2),    A := α + κ_v,
#                                                          M := α κ_v,
#                                                          L := β γ,
#
# and a "no-Hopf wedge" defined operationally as
#
#     W_NH := {(σ_q, γ) : H(κ) = 0 has no positive root in κ ∈ [0, ∞)}.
#
# Two sub-cases generate the wedge: (i) D < 0 (no real root of H, the strict
# §3.5 wedge) or (ii) D ≥ 0 but the smallest positive root is non-positive
# (both real roots negative). The L2-T § 3.7 BT analysis showed
# κ_SN(σ_q, γ) < 0 throughout the canonical scan window, so the saddle-node
# locus that they checked is also empty.
#
# The open question is: in W_NH, what bifurcation (if any) is accessible as
# κ ramps up from 0? Theorem 6 (no-Hopf-wedge taxonomy) answers it: with the
# canonical closed-form partials (G_y, G_v) held fixed at the trivial
# equilibrium (a^* = μ_q, v^* = θ_v), all three Routh-Hurwitz inequalities
# (c_2 > 0, c_0 > 0, H > 0) hold *strictly for every κ ≥ 0* throughout W_NH.
# The equilibrium is therefore globally asymptotically stable on the
# physical κ-half-line in the wedge — no codim-1 bifurcation of any kind.
#
# Sufficient condition (closed-form, verified in `is_in_no_hopf_wedge` and
# enforced by `bifurcations_in_no_hopf_wedge`):
#
#     (S1) G_y ≤ 0                            ⇒ c_2(κ) = -κ G_y + A > 0 ∀κ ≥ 0
#     (S2) G_y · α κ_v + G_v · β γ ≤ 0        ⇒ c_0(κ) ≥ L/2 > 0 ∀κ ≥ 0
#     (S3) D < 0,  OR  (D ≥ 0 ∧ A_0 > 0 ∧ A_1 > 0)
#                                              ⇒ H(κ) > 0 ∀κ ≥ 0
#
# (S1)+(S2)+(S3) ⇒ Liu's full Routh-Hurwitz holds strictly for every κ ≥ 0,
# so the spectral abscissa of J(κ) stays strictly negative — globally
# asymptotically stable equilibrium on κ ∈ [0, ∞).
# ---------------------------------------------------------------------------


def is_in_no_hopf_wedge(
    *,
    sigma_q: float,
    gamma: float,
    mu_q: float,
    T_eff: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    a_star: float,
    v_star: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> bool:
    """Classify $(\\sigma_q, \\gamma)$ as belonging to the no-Hopf wedge.

    The no-Hopf wedge is defined operationally as the set of $(\\sigma_q, \\gamma)$
    parameter pairs for which the closed-form Routh-Hurwitz quadratic
    $H(\\kappa) = A_2 \\kappa^2 + A_1 \\kappa + A_0$ admits NO positive real root.
    Two sub-cases generate it: (i) discriminant $D < 0$ (no real root at all —
    the strict §3.5 wedge), or (ii) $D \\geq 0$ but both real roots non-positive.

    Returns True if $(\\sigma_q, \\gamma)$ is in the wedge, False if a positive
    Hopf root exists.

    Args mirror `kappa_star_lognormal_oi` except that the OI partials are
    recomputed from $(\\mu_q, \\sigma_q, T_{eff}, a^\\star, v^\\star)$ here.
    """
    if sigma_q <= 0.0:
        raise ValueError(f"sigma_q must be > 0, got {sigma_q}")
    if T_eff <= 0.0:
        raise ValueError(f"T_eff must be > 0, got {T_eff}")
    if kappa_v <= 0.0:
        raise ValueError(f"kappa_v must be > 0, got {kappa_v}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

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
    G_y = p["G_a"]
    G_v = p["G_v"]
    A_tot = alpha + kappa_v
    M = alpha * kappa_v
    L = beta * gamma

    if abs(G_y) < 1e-300:
        # H collapses to linear: G_v L κ + (M A − L/2)
        if abs(G_v * L) < 1e-300:
            return True  # H is a non-zero constant, no Hopf root
        kappa_lin = (0.5 * L - M * A_tot) / (G_v * L)
        return kappa_lin <= 0.0

    A2 = G_y * G_y * A_tot
    A1 = G_v * L - G_y * A_tot * A_tot
    A0 = M * A_tot - 0.5 * L
    disc = A1 * A1 - 4.0 * A2 * A0
    if disc < 0.0:
        return True  # strict §3.5 wedge: no real root
    sqrt_disc = float(np.sqrt(disc))
    r1 = (-A1 - sqrt_disc) / (2.0 * A2)
    r2 = (-A1 + sqrt_disc) / (2.0 * A2)
    return not (r1 > 0.0 or r2 > 0.0)


@dataclass(frozen=True)
class NoHopfBifurcationResult:
    """Output of `bifurcations_in_no_hopf_wedge`.

    Encodes the §3.5 wedge taxonomy at a single $(\\sigma_q, \\gamma)$:

        - `is_in_wedge`: whether $(\\sigma_q, \\gamma)$ admits no positive Hopf root.
        - `is_globally_stable`: True iff the canonical sufficient conditions
          (S1)+(S2)+(S3) of Theorem 6 hold — Routh-Hurwitz strict for all
          $\\kappa \\geq 0$, equilibrium globally asymptotically stable.
        - `kappa_sn`: positive κ at which the constant Routh-Hurwitz coefficient
          $c_0(\\kappa) = 0$ (saddle-node onset via $\\det J = 0$). `None` if
          $c_0$ stays positive on $[0, \\kappa_{\\max}]$.
        - `kappa_c2_zero`: positive κ at which $c_2(\\kappa) = 0$ (an additional
          eigenvalue-crossing route the §3.7 closed form does NOT cover).
          `None` if $c_2 > 0$ on $[0, \\kappa_{\\max}]$.
        - `kappa_H_zero`: positive κ at which $H(\\kappa) = 0$ — should always be
          `None` inside the wedge by construction, but reported for sanity.
        - `kappa_max_scanned`: the upper end of the scanned κ-interval.
        - `spectral_abscissa_max`: $\\max_{\\kappa \\in [0, \\kappa_{\\max}]} \\max_i
          \\mathrm{Re}\\,\\lambda_i(J(\\kappa))$, sampled. Negative ⇒ stable.
        - `G_y`, `G_v`: the canonical partials used.
    """

    is_in_wedge: bool
    is_globally_stable: bool
    kappa_sn: float | None
    kappa_c2_zero: float | None
    kappa_H_zero: float | None
    kappa_max_scanned: float
    spectral_abscissa_max: float
    G_y: float
    G_v: float


def bifurcations_in_no_hopf_wedge(
    *,
    sigma_q: float,
    gamma: float,
    mu_q: float,
    T_eff: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    a_star: float,
    v_star: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    kappa_max: float = 1.0e3,
    n_kappa_samples: int = 200,
) -> NoHopfBifurcationResult:
    """Scan κ ∈ [0, κ_max] for any codim-1 bifurcation at fixed $(\\sigma_q, \\gamma)$
    inside the no-Hopf wedge.

    Uses the L2-T canonical-equilibrium partials (G_y, G_v frozen at the
    trivial equilibrium $a^* = \\mu_q$, $v^* = \\theta_v$) and inspects three
    closed-form codim-1 indicators:

        - $c_0(\\kappa) = 0$  (saddle-node via $\\det J = 0$)
        - $c_2(\\kappa) = 0$  (real eigenvalue crossing into RHP via trace flip)
        - $H(\\kappa) = 0$    (Hopf — should be excluded by the wedge definition)

    Also samples the eigenvalues of $J(\\kappa)$ on a log-uniform κ-grid up to
    $\\kappa_{\\max}$ and reports $\\max_\\kappa \\max_i \\mathrm{Re}\\,\\lambda_i$
    as a numerical sanity-check against the closed-form claim.

    Returns a `NoHopfBifurcationResult`. The honest interpretation of the
    output:

        - `is_in_wedge=True` ∧ `is_globally_stable=True`
          ⇒ Theorem 6(a): no bifurcation on the physical κ-half-line.

        - `is_in_wedge=True` ∧ `is_globally_stable=False`
          ⇒ Theorem 6(b/c): some codim-1 bifurcation exists; inspect
          `kappa_sn`, `kappa_c2_zero`, `kappa_H_zero` for the mechanism.

    Args:
        sigma_q, gamma: parameter-plane location.
        kappa_max: upper end of the scanned interval (default 1000, large
            enough to cover the §3.5 / §3.7 canonical κ★ scales).
        n_kappa_samples: number of κ samples for the numerical eigenvalue
            sweep (default 200; cost is dominated by 200 × 3×3 eigvals
            — milliseconds).
        Other args: as in `lyapunov_coefficient_lognormal_oi`.
    """
    if kappa_max <= 0.0:
        raise ValueError(f"kappa_max must be > 0, got {kappa_max}")
    if n_kappa_samples < 2:
        raise ValueError(f"n_kappa_samples must be >= 2, got {n_kappa_samples}")

    in_wedge = is_in_no_hopf_wedge(
        sigma_q=sigma_q,
        gamma=gamma,
        mu_q=mu_q,
        T_eff=T_eff,
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        a_star=a_star,
        v_star=v_star,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
    )

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
    G_y = p["G_a"]
    G_v = p["G_v"]

    A_tot = alpha + kappa_v
    M = alpha * kappa_v
    L = beta * gamma

    # Closed-form bifurcation indicators (all linear or quadratic in κ).
    #
    # c_2(κ) = -κ G_y + A:  zero at κ = A / G_y if G_y > 0 (and only then).
    if G_y > 0.0:
        kappa_c2_zero: float | None = A_tot / G_y
        if cast(float, kappa_c2_zero) > kappa_max:
            kappa_c2_zero = None
    else:
        kappa_c2_zero = None

    # c_0(κ) = -κ (G_y M + G_v L) + L/2: zero at κ = (L/2) / (G_y M + G_v L)
    # if denominator is positive.
    denom_c0 = G_y * M + G_v * L
    if denom_c0 > 0.0:
        kappa_sn_candidate = 0.5 * L / denom_c0
        kappa_sn: float | None = (
            kappa_sn_candidate if 0.0 < kappa_sn_candidate <= kappa_max else None
        )
    else:
        kappa_sn = None

    # H(κ) = A2 κ² + A1 κ + A0 zeros (Hopf). Should be excluded if in wedge.
    A2 = G_y * G_y * A_tot
    A1 = G_v * L - G_y * A_tot * A_tot
    A0 = M * A_tot - 0.5 * L
    kappa_H_zero: float | None = None
    if abs(A2) > 1e-300:
        disc = A1 * A1 - 4.0 * A2 * A0
        if disc >= 0.0:
            sqrt_disc = float(np.sqrt(disc))
            r1 = (-A1 - sqrt_disc) / (2.0 * A2)
            r2 = (-A1 + sqrt_disc) / (2.0 * A2)
            positive_roots = [r for r in (r1, r2) if 0.0 < r <= kappa_max]
            if positive_roots:
                kappa_H_zero = float(min(positive_roots))
    elif abs(A1) > 1e-300:
        # Linear: A1 κ + A0 = 0
        kappa_lin = -A0 / A1
        if 0.0 < kappa_lin <= kappa_max:
            kappa_H_zero = kappa_lin

    # Numerical sanity-check: sample max Re(λ) on a κ-grid.
    # Use log spacing so we cover small and large κ.
    kappa_samples = np.concatenate(
        [
            np.array([0.0]),
            np.logspace(
                np.log10(max(1e-6, kappa_max / 1e6)),
                np.log10(kappa_max),
                n_kappa_samples - 1,
            ),
        ]
    )
    max_re_overall = -np.inf
    for k in kappa_samples:
        a_lin = float(k) * G_y
        b_lin = float(k) * G_v - 0.5
        J = np.array(
            [
                [a_lin, b_lin, 0.0],
                [0.0, -kappa_v, gamma],
                [beta, 0.0, -alpha],
            ],
            dtype=np.float64,
        )
        eig_re = float(np.max(np.linalg.eigvals(J).real))
        if eig_re > max_re_overall:
            max_re_overall = eig_re

    # Global-stability verdict (Theorem 6 sufficient conditions).
    cond_c2 = G_y <= 0.0
    cond_c0 = denom_c0 <= 0.0
    cond_H = in_wedge  # H > 0 ∀κ ≥ 0 follows from the wedge definition + A2 > 0
    is_globally_stable = bool(in_wedge and cond_c2 and cond_c0 and cond_H)

    return NoHopfBifurcationResult(
        is_in_wedge=in_wedge,
        is_globally_stable=is_globally_stable,
        kappa_sn=kappa_sn,
        kappa_c2_zero=kappa_c2_zero,
        kappa_H_zero=kappa_H_zero,
        kappa_max_scanned=float(kappa_max),
        spectral_abscissa_max=float(max_re_overall),
        G_y=float(G_y),
        G_v=float(G_v),
    )


def scan_no_hopf_wedge(
    *,
    sigma_q_grid: NDArray[np.float64],
    gamma_grid: NDArray[np.float64],
    mu_q: float,
    T_eff: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    a_star: float,
    v_star: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    kappa_max: float = 1.0e3,
    n_kappa_samples: int = 50,
) -> dict[str, NDArray[np.float64]]:
    """Vectorised wedge scan over a $(\\sigma_q, \\gamma)$ grid.

    For each $(\\sigma_q, \\gamma)$ cell, call `bifurcations_in_no_hopf_wedge`
    and store the four headline scalars as arrays of shape (n_gamma, n_sigma_q):

        - `in_wedge_grid`  (bool):  the wedge classifier output.
        - `globally_stable_grid` (bool):  Theorem 6 sufficient conditions met.
        - `kappa_sn_grid`  (float): NaN where no positive SN; else the κ value.
        - `spectral_abscissa_grid` (float): the numerical max Re(λ) over κ-scan.

    Cost: ~`n_sigma_q × n_gamma × n_kappa_samples` 3×3 eigvals — on the
    51×51 canonical grid with `n_kappa_samples=50`, runs in ~3 s on M-series.
    """
    sq = np.asarray(sigma_q_grid, dtype=np.float64)
    gam = np.asarray(gamma_grid, dtype=np.float64)
    if sq.ndim != 1 or gam.ndim != 1:
        raise ValueError("sigma_q_grid and gamma_grid must be 1D")
    if not (np.all(np.diff(sq) > 0) and np.all(np.diff(gam) > 0)):
        raise ValueError("grids must be strictly ascending")

    n_g = len(gam)
    n_s = len(sq)
    in_wedge_grid = np.zeros((n_g, n_s), dtype=bool)
    globally_stable_grid = np.zeros((n_g, n_s), dtype=bool)
    kappa_sn_grid = np.full((n_g, n_s), np.nan, dtype=np.float64)
    spectral_abscissa_grid = np.full((n_g, n_s), np.nan, dtype=np.float64)

    for i, g in enumerate(gam):
        for j, s in enumerate(sq):
            r = bifurcations_in_no_hopf_wedge(
                sigma_q=float(s),
                gamma=float(g),
                mu_q=mu_q,
                T_eff=T_eff,
                kappa_v=kappa_v,
                alpha=alpha,
                beta=beta,
                a_star=a_star,
                v_star=v_star,
                coupling_units=coupling_units,
                rate=rate,
                dividend=dividend,
                kappa_max=kappa_max,
                n_kappa_samples=n_kappa_samples,
            )
            in_wedge_grid[i, j] = r.is_in_wedge
            globally_stable_grid[i, j] = r.is_globally_stable
            if r.kappa_sn is not None:
                kappa_sn_grid[i, j] = r.kappa_sn
            spectral_abscissa_grid[i, j] = r.spectral_abscissa_max

    return {
        "sigma_q_grid": sq,
        "gamma_grid": gam,
        "in_wedge_grid": in_wedge_grid,
        "globally_stable_grid": globally_stable_grid,
        "kappa_sn_grid": kappa_sn_grid,
        "spectral_abscissa_grid": spectral_abscissa_grid,
    }


# ---------------------------------------------------------------------------
# Task-spec packaging: `NoHopfWedgeScanResult` + `scan_no_hopf_wedge_bifurcations`
#
# Thin typed wrapper around `scan_no_hopf_wedge` that returns a dataclass with
# scalar summary statistics (cell counts, max spectral abscissa, etc.) alongside
# the underlying grids. The summary scalars are the "headline numbers" that
# Theorem 6 and the §3.5-extension narrative in the paper refer to directly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoHopfWedgeScanResult:
    """Aggregate output of `scan_no_hopf_wedge_bifurcations` over a (σ_q, γ) grid.

    Underlying per-cell verdict comes from `bifurcations_in_no_hopf_wedge`; this
    dataclass packages the grids together with the four summary scalars used by
    Theorem 6's bookkeeping.

    Attributes:
        sigma_q_grid: 1D ascending grid of σ_q values, shape (n_sigma_q,).
        gamma_grid: 1D ascending grid of γ values, shape (n_gamma,).
        in_wedge_grid: bool (n_gamma, n_sigma_q) — wedge classifier per cell.
        globally_stable_grid: bool (n_gamma, n_sigma_q) — Theorem 6 sufficient
            conditions (S1)+(S2)+(S3) hold at this cell.
        kappa_sn_grid: float (n_gamma, n_sigma_q) — positive saddle-node κ
            (c_0 = 0) inside (0, κ_max]; NaN where no positive SN root.
        spectral_abscissa_grid: float (n_gamma, n_sigma_q) — sampled
            max_κ max_i Re λ_i(J(κ)) on the κ-grid.
        n_wedge_cells: number of cells classified as in-wedge.
        n_globally_stable_cells: number of wedge cells with Theorem 6
            sufficient conditions verified.
        n_positive_saddle_node_cells: number of cells with a positive κ_SN ≤ κ_max.
        wedge_max_spectral_abscissa: max over wedge cells of the spectral
            abscissa (numerical sanity-check on the closed-form verdict; should
            be < 0 if Theorem 6(a) holds globally).
        kappa_max_scanned: upper end of the κ-interval scanned.
    """

    sigma_q_grid: NDArray[np.float64]
    gamma_grid: NDArray[np.float64]
    in_wedge_grid: NDArray[np.bool_]
    globally_stable_grid: NDArray[np.bool_]
    kappa_sn_grid: NDArray[np.float64]
    spectral_abscissa_grid: NDArray[np.float64]
    n_wedge_cells: int
    n_globally_stable_cells: int
    n_positive_saddle_node_cells: int
    wedge_max_spectral_abscissa: float
    kappa_max_scanned: float


def scan_no_hopf_wedge_bifurcations(
    *,
    sigma_q_grid: NDArray[np.float64],
    gamma_grid: NDArray[np.float64],
    mu_q: float,
    T_eff: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    a_star: float,
    v_star: float,
    coupling_units: float = 1.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    kappa_max: float = 100.0,
    n_kappa_samples: int = 80,
) -> NoHopfWedgeScanResult:
    """Scan (σ_q, γ) over a grid, classify each cell against the §3.5 wedge,
    and search κ ∈ (0, κ_max] for any codim-1 bifurcation.

    Thin wrapper around `scan_no_hopf_wedge` that packages the result as a
    `NoHopfWedgeScanResult` with the four summary scalars referenced by
    Theorem 6 in `paper/saddle_node_no_hopf.md` and `paper/theory.md §4.4.4`.

    Args:
        sigma_q_grid, gamma_grid: 1D ascending grids.
        mu_q, T_eff, a_star, v_star, coupling_units, rate, dividend: log-normal
            OI parameters and equilibrium anchor; see `G_lognormal_oi_partials`.
        kappa_v, alpha, beta: the deterministic-skeleton triangle.
        kappa_max: upper end of the κ-interval scanned per cell (default 100).
        n_kappa_samples: number of κ samples used in the numerical eigenvalue
            sanity-check (default 80).

    Returns:
        `NoHopfWedgeScanResult`. The headline verdict is
        `n_positive_saddle_node_cells == 0` ∧ `wedge_max_spectral_abscissa < 0`
        ⇒ Theorem 6(a) — wedge is globally asymptotically stable on
        κ ∈ [0, κ_max].
    """
    grids = scan_no_hopf_wedge(
        sigma_q_grid=sigma_q_grid,
        gamma_grid=gamma_grid,
        mu_q=mu_q,
        T_eff=T_eff,
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        a_star=a_star,
        v_star=v_star,
        coupling_units=coupling_units,
        rate=rate,
        dividend=dividend,
        kappa_max=kappa_max,
        n_kappa_samples=n_kappa_samples,
    )
    in_wedge = np.asarray(grids["in_wedge_grid"], dtype=np.bool_)
    gs = np.asarray(grids["globally_stable_grid"], dtype=np.bool_)
    ksn = np.asarray(grids["kappa_sn_grid"], dtype=np.float64)
    abs_max = np.asarray(grids["spectral_abscissa_grid"], dtype=np.float64)

    n_wedge = int(np.sum(in_wedge))
    n_gs = int(np.sum(gs))
    n_sn = int(np.sum(np.isfinite(ksn)))
    wedge_abs_max = float(np.max(abs_max[in_wedge])) if n_wedge > 0 else float("nan")

    return NoHopfWedgeScanResult(
        sigma_q_grid=grids["sigma_q_grid"],
        gamma_grid=grids["gamma_grid"],
        in_wedge_grid=in_wedge,
        globally_stable_grid=gs,
        kappa_sn_grid=ksn,
        spectral_abscissa_grid=abs_max,
        n_wedge_cells=n_wedge,
        n_globally_stable_cells=n_gs,
        n_positive_saddle_node_cells=n_sn,
        wedge_max_spectral_abscissa=wedge_abs_max,
        kappa_max_scanned=float(kappa_max),
    )
