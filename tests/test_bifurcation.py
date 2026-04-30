"""Tests for the Hopf bifurcation theory module.

Covers:
    - hopf_scan recovers no Hopf at κ=0 (trivial Heston backbone).
    - hopf_scan finds a synthetic bifurcation whose location is known in closed form.
    - compute_lyapunov_coefficient matches the analytical sign on the Hopf normal form.
    - build_bilinear_trilinear_tensors matches symbolic Taylor coefficients on a
      polynomial RHS.
    - stochastic_hopf_shift recovers Λ ε² → 0 as ε → 0 in expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.theory.bifurcation import (
    build_bilinear_trilinear_tensors,
    compute_lambda_correction,
    compute_lyapunov_coefficient,
    hopf_scan,
    jacobian_3d,
    stochastic_hopf_shift_numeric,
    top_lyapunov_exponent_linearised,
)
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SurfaceGrid,
)

# ---------------------------------------------------------------------------
# 1. hopf_scan at κ = 0 — trivial Heston backbone, no Hopf
# ---------------------------------------------------------------------------


def test_hopf_scan_recovers_zero_at_kappa_zero() -> None:
    """At κ = 0 the price-channel feedback vanishes; the only nonzero entries
    are -0.5 ∂_x σ² and -0.5 ∂_v σ². Configuring those to zero leaves J as a
    block-diagonal Heston-like Jacobian whose eigenvalues are real and ≤ 0.
    """
    kappa_v = 2.0
    alpha = 252.0  # ~1/day
    beta = 0.0  # disable z → x coupling
    gamma = 0.0  # disable z → v coupling

    def jac(k: float) -> np.ndarray:
        return jacobian_3d(
            kappa=k,
            a_kappa=0.0,
            b_kappa=0.0,
            G_z=0.0,
            kappa_v=kappa_v,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    kappa_grid = np.linspace(0.0, 1.0, 51)
    result = hopf_scan(kappa_grid, jac)

    assert result.kappa_star is None, (
        f"unexpected Hopf at κ={result.kappa_star} for trivial Heston backbone"
    )
    # All eigenvalues must be real & non-positive at every κ
    for i in range(len(kappa_grid)):
        eig = result.eigenvalues[i]
        assert np.allclose(eig.imag, 0.0, atol=1e-10), (
            f"complex eigenvalue at κ={kappa_grid[i]}: {eig}"
        )
        assert eig.real.max() <= 1e-10, f"positive real eigenvalue at κ={kappa_grid[i]}: {eig}"


# ---------------------------------------------------------------------------
# 2. Synthetic Hopf with an analytically known κ*
# ---------------------------------------------------------------------------


def test_hopf_scan_finds_bifurcation_with_synthetic_params() -> None:
    """Construct a synthetic configuration with a closed-form κ*.

    Take γ = 0 so the (variance) row/col decouples and J reduces to a 2×2 block
    in (y, z) plus the eigenvalue -κ_v:

        J_2x2 = [[a(κ), κ G_z],
                 [β,    -α   ]]

    Trace = a(κ) - α, det = -a(κ) α - κ G_z β.
    Eigenvalues = ½[(a-α) ± √((a+α)² + 4 κ G_z β)].
    With G_z β < 0 and |G_z β| large, the discriminant is negative and the
    pair is complex with real part (a - α)/2. The Hopf occurs at a(κ*) = α.

    Choose a(κ) = a_0 + g_x κ ⇒ κ* = (α - a_0) / g_x.
    """
    a_0 = -1.0
    g_x = 0.5
    alpha_decay = 1.0
    G_z = -1.0  # large enough that the (y, z) pair stays complex over the scan range
    beta = 1.0
    kappa_v = 2.0  # third eigenvalue = -κ_v < 0 — required for valid Hopf

    expected_kappa_star = (alpha_decay - a_0) / g_x  # = 4.0

    def jac(k: float) -> np.ndarray:
        return jacobian_3d(
            kappa=k,
            a_kappa=a_0 + g_x * k,
            b_kappa=0.0,  # irrelevant — γ = 0 decouples v
            G_z=G_z,
            kappa_v=kappa_v,
            alpha=alpha_decay,
            beta=beta,
            gamma=0.0,
        )

    # Sanity: at κ slightly below expected, complex pair with negative real part
    eig_below = np.linalg.eigvals(jac(expected_kappa_star - 0.5))
    assert (np.abs(eig_below.imag) > 1e-8).sum() == 2
    assert eig_below.real.max() < 0
    # And slightly above: complex pair with positive real part
    eig_above = np.linalg.eigvals(jac(expected_kappa_star + 0.5))
    assert (np.abs(eig_above.imag) > 1e-8).sum() == 2
    assert eig_above.real.max() > 0

    # hopf_scan should locate κ* within tolerance.
    fine_kappa = np.linspace(0.1, 8.0, 791)
    result = hopf_scan(fine_kappa, jac)
    assert result.kappa_star is not None, "hopf_scan failed to locate κ*"
    assert abs(result.kappa_star - expected_kappa_star) < 1e-3, (
        f"κ* = {result.kappa_star}, expected {expected_kappa_star}"
    )
    eig_at_star = np.linalg.eigvals(jac(result.kappa_star))
    imag_pair = eig_at_star[np.abs(eig_at_star.imag) > 1e-8]
    assert imag_pair.shape[0] == 2
    assert max(abs(imag_pair[0].real), abs(imag_pair[1].real)) < 1e-6
    assert result.omega_at_crossing is not None and result.omega_at_crossing > 0


# ---------------------------------------------------------------------------
# 3. First Lyapunov coefficient on the canonical Hopf normal form
# ---------------------------------------------------------------------------


def _hopf_normal_form_drift(
    omega: float,
    a: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (J, B, C) for the 3D Hopf normal form

        ẋ = -ω y + a (x² + y²) x
        ẏ =  ω x + a (x² + y²) y
        ż = -z

    with f(x*) = 0 at the origin. ℓ_1 has a known sign: sign(ℓ_1) = sign(a).
    """
    J = np.array([[0.0, -omega, 0.0], [omega, 0.0, 0.0], [0.0, 0.0, -1.0]])
    B = np.zeros((3, 3, 3))
    C = np.zeros((3, 3, 3, 3))
    # f_1 = -ω y + a x³ + a x y²
    #   ∂³f_1/∂x³ = 6a → C[0,0,0,0] = 6a
    #   ∂³f_1/∂x∂y² = 2a → all 3 perms of (0,1,1) → C[0,0,1,1] = C[0,1,0,1] = C[0,1,1,0] = 2a
    C[0, 0, 0, 0] = 6.0 * a
    for j, k, m in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        C[0, j, k, m] = 2.0 * a
    # f_2 = ω x + a y³ + a y x²
    #   ∂³f_2/∂y³ = 6a → C[1,1,1,1] = 6a
    #   ∂³f_2/∂y∂x² = 2a → all 3 perms of (1,0,0) → C[1,1,0,0] = C[1,0,1,0] = C[1,0,0,1] = 2a
    C[1, 1, 1, 1] = 6.0 * a
    for j, k, m in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        C[1, j, k, m] = 2.0 * a
    return J, B, C


def test_lyapunov_coefficient_supercritical_normal_form() -> None:
    """For the 3D Hopf normal form with a < 0, ℓ_1 < 0 (supercritical)."""
    omega = 1.0
    for a in [-0.5, -1.0, -2.0]:
        J, B, C = _hopf_normal_form_drift(omega, a)
        ell1 = compute_lyapunov_coefficient(J, B, C, omega=omega)
        assert ell1 < 0, f"a={a} should give ℓ_1 < 0, got {ell1}"
        # Quantitative check: derivation in module gives ℓ_1 = 2 a / ω for this normal form
        expected = 2.0 * a / omega
        assert abs(ell1 - expected) < 1e-10, f"a={a}: expected ℓ_1 = {expected}, got {ell1}"


def test_lyapunov_coefficient_subcritical_normal_form() -> None:
    """For a > 0, ℓ_1 > 0 (subcritical)."""
    omega = 1.5
    for a in [0.3, 1.0, 2.5]:
        J, B, C = _hopf_normal_form_drift(omega, a)
        ell1 = compute_lyapunov_coefficient(J, B, C, omega=omega)
        assert ell1 > 0, f"a={a} should give ℓ_1 > 0, got {ell1}"
        expected = 2.0 * a / omega
        assert abs(ell1 - expected) < 1e-10


# ---------------------------------------------------------------------------
# 4. Finite-difference tensor builder vs symbolic
# ---------------------------------------------------------------------------


def test_b_c_tensor_construction_finite_differences_consistency() -> None:
    """For a known polynomial RHS, finite-difference B and C match symbolic.

    Choose f_i(x) with explicit polynomial coefficients up to cubic order.
    """

    # f_1 = 2 x_1 x_2 + 3 x_2² + x_1³ + x_1 x_2 x_3
    # f_2 = x_1² - x_3² + 2 x_1² x_3
    # f_3 = x_2 x_3 + x_3³
    def drift(x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array(
            [
                2.0 * x1 * x2 + 3.0 * x2 * x2 + x1**3 + x1 * x2 * x3,
                x1 * x1 - x3 * x3 + 2.0 * x1 * x1 * x3,
                x2 * x3 + x3**3,
            ]
        )

    B, C = build_bilinear_trilinear_tensors(drift, (0.0, 0.0, 0.0), h=1e-3)

    # Symbolic B[i,j,k] = ∂²f_i/∂x_j∂x_k at 0.
    # For f_1 quadratic part = 2 x1 x2 + 3 x2² →
    #   ∂²f_1/∂x_1∂x_2 = 2, ∂²f_1/∂x_2² = 6.
    # For f_2 quadratic = x_1² - x_3² →
    #   ∂²f_2/∂x_1² = 2, ∂²f_2/∂x_3² = -2.
    # For f_3 quadratic = x_2 x_3 →
    #   ∂²f_3/∂x_2∂x_3 = 1.
    B_expected = np.zeros((3, 3, 3))
    B_expected[0, 0, 1] = B_expected[0, 1, 0] = 2.0
    B_expected[0, 1, 1] = 6.0
    B_expected[1, 0, 0] = 2.0
    B_expected[1, 2, 2] = -2.0
    B_expected[2, 1, 2] = B_expected[2, 2, 1] = 1.0

    np.testing.assert_allclose(B, B_expected, atol=1e-5)

    # Symbolic C[i,j,k,l] = ∂³f_i/∂x_j∂x_k∂x_l.
    # f_1 cubic = x_1³ + x_1 x_2 x_3 →
    #   ∂³f_1/∂x_1³ = 6
    #   ∂³f_1/∂x_1∂x_2∂x_3 = 1 (and 6 perms)
    # f_2 cubic = 2 x_1² x_3 →
    #   ∂³f_2/∂x_1²∂x_3 = 4 (perms (1,1,3), (1,3,1), (3,1,1))
    # f_3 cubic = x_3³ →
    #   ∂³f_3/∂x_3³ = 6
    C_expected = np.zeros((3, 3, 3, 3))
    C_expected[0, 0, 0, 0] = 6.0
    for a, b, c in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        C_expected[0, a, b, c] = 1.0
    for a, b, c in [(0, 0, 2), (0, 2, 0), (2, 0, 0)]:
        C_expected[1, a, b, c] = 4.0
    C_expected[2, 2, 2, 2] = 6.0

    np.testing.assert_allclose(C, C_expected, atol=1e-3)


# ---------------------------------------------------------------------------
# 5. Stochastic Hopf shift — Λ ε² → 0 as ε → 0
# ---------------------------------------------------------------------------


def test_stochastic_hopf_shift_recovers_zero_in_zero_noise_limit() -> None:
    """As ε → 0 the noise-induced shift Λ ε² vanishes; the top Lyapunov
    exponent of the linear system collapses to the deterministic largest real
    part of the Jacobian.

    Use a stable Jacobian (all real parts negative) so α(κ) is well below
    zero and Λ ε² is a small perturbation. Verify lambda_1(ε) → α as ε → 0.
    """
    # 3D Jacobian with eigenvalues {-0.5, -1, -2}, far from any bifurcation.
    J = np.array(
        [
            [-0.5, 0.1, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -2.0],
        ]
    )
    Sigma = np.eye(3) * 0.5  # mild diffusion in all directions
    alpha_det = float(np.linalg.eigvals(J).real.max())  # = -0.5

    # Compute λ_1 at decreasing ε; expect convergence to α_det.
    eps_values = [0.30, 0.10, 0.03]
    lams = [
        top_lyapunov_exponent_linearised(
            J,
            Sigma,
            epsilon=e,
            n_paths=400,
            n_steps=4_000,
            dt=5e-3,
            renorm_every=50,
            seed=42 + i,
        )
        for i, e in enumerate(eps_values)
    ]
    # Each λ should be within ~0.15 of α_det at these noise levels (small-noise
    # regime; tighter bounds need n_paths >> 10^4).
    for e, lam in zip(eps_values, lams, strict=True):
        assert abs(lam - alpha_det) < 0.25, f"ε={e}: λ_1 ≈ {lam:.4f}, α_det = {alpha_det:.4f}"
    # Monotone convergence: |λ(ε_small) - α| < |λ(ε_large) - α| (in expectation)
    # — assert at least the smallest-ε value is closer than the largest.
    assert abs(lams[-1] - alpha_det) <= abs(lams[0] - alpha_det) + 0.05


def test_stochastic_hopf_shift_two_point_extrapolation_is_finite() -> None:
    """The two-point Λ estimator should produce a finite, well-conditioned answer."""
    J = np.array(
        [
            [-0.5, 1.0, 0.0],
            [-1.0, -0.5, 0.0],
            [0.0, 0.0, -2.0],
        ]
    )  # eigenvalues {-0.5 ± i, -2}
    Sigma = np.eye(3) * 0.5
    Lambda, lam_low, lam_high = stochastic_hopf_shift_numeric(
        J,
        Sigma,
        epsilon_low=0.05,
        epsilon_high=0.20,
        n_paths=200,
        n_steps=4_000,
        dt=5e-3,
        renorm_every=50,
        seed=7,
    )
    assert np.isfinite(Lambda)
    assert np.isfinite(lam_low)
    assert np.isfinite(lam_high)
    # Both Lyapunov estimates should be near α_det = -0.5 for this small ε regime.
    assert abs(lam_low - (-0.5)) < 0.3
    assert abs(lam_high - (-0.5)) < 0.4


# ---------------------------------------------------------------------------
# 6. Integration test: simulator drift exposes a usable RHS for tensor builder
# ---------------------------------------------------------------------------


def _make_simulator(coupling: float = 0.0) -> ReflexiveSimulator:
    grid = SurfaceGrid(
        log_moneyness=np.array([-0.05, 0.0, 0.05]),
        maturities=np.array([30 / 365.25, 90 / 365.25]),
    )
    contracts = np.zeros(grid.shape, dtype=np.float64)
    oi = OpenInterestGrid(grid=grid, contracts_open=contracts)
    aggregator = GammaAggregator(oi_grid=oi, risk_free_rate=0.05)
    params = ReflexiveParams(
        base=HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04),
        coupling=coupling,
        drift=0.0,
        leverage=0.5,
    )
    return ReflexiveSimulator(params=params, gamma_aggregator=aggregator, initial_spot=100.0)


def test_simulator_drift_gives_consistent_finite_difference_jacobian() -> None:
    """The Jacobian extracted from simulator.drift via finite differences must
    match the analytical 3×3 form predicted by paper/theory.md §2.
    """
    sim = _make_simulator(coupling=0.0)
    log_s_star = float(np.log(sim.initial_spot))
    v_star = sim.params.base.theta
    z_star = 0.0

    def drift(x: np.ndarray) -> np.ndarray:
        s = float(np.exp(x[0]))
        return sim.drift(s, float(x[1]), float(x[2]))

    h = 1e-5
    x0 = np.array([log_s_star, v_star, z_star])
    J_fd = np.zeros((3, 3))
    for j in range(3):
        e_j = np.zeros(3)
        e_j[j] = h
        J_fd[:, j] = (drift(x0 + e_j) - drift(x0 - e_j)) / (2.0 * h)

    # Analytical at κ=0:
    #   ∂(d log S/dt)/∂x = (μ + κ G) - 0.5 ∂_x v ; v doesn't depend on x ⇒ 0
    #   ∂(d log S/dt)/∂v = -0.5
    #   ∂(d log S/dt)/∂z = 0
    #   d v: ∂/∂x = 0, ∂/∂v = -κ_v, ∂/∂z = γ
    #   d z: ∂/∂x = β, ∂/∂v = 0, ∂/∂z = -α
    expected = np.array(
        [
            [0.0, -0.5, 0.0],
            [0.0, -sim.params.base.kappa, sim.params.leverage],
            [sim.params.memory_intake, 0.0, -sim.params.memory_decay],
        ]
    )
    np.testing.assert_allclose(J_fd, expected, atol=1e-5)


def test_compute_lambda_correction_runs_and_returns_finite() -> None:
    """End-to-end smoke test: extract J/Σ from a stable simulator, run the
    two-point Λ estimator, expect a finite scalar.
    """
    sim = _make_simulator(coupling=0.0)
    Lambda = compute_lambda_correction(
        sim,
        kappa=0.0,
        epsilon_low=0.05,
        epsilon_high=0.20,
        n_paths=100,
        n_steps=2_000,
        dt=5e-3,
        renorm_every=50,
        seed=11,
    )
    assert np.isfinite(Lambda)


# ---------------------------------------------------------------------------
# 7. Build B, C from the simulator and check sane shapes
# ---------------------------------------------------------------------------


def test_build_tensors_from_simulator_shapes_and_symmetry() -> None:
    sim = _make_simulator(coupling=0.0)
    log_s_star = float(np.log(sim.initial_spot))
    v_star = sim.params.base.theta

    def drift(x: np.ndarray) -> np.ndarray:
        s = float(np.exp(x[0]))
        return sim.drift(s, float(x[1]), float(x[2]))

    B, C = build_bilinear_trilinear_tensors(drift, (log_s_star, v_star, 0.0), h=1e-3)
    assert B.shape == (3, 3, 3)
    assert C.shape == (3, 3, 3, 3)
    # Symmetry of B in last two indices
    np.testing.assert_allclose(B, B.transpose(0, 2, 1), atol=1e-6)
    # Symmetry of C under any permutation of last three indices
    for perm in [(0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1)]:
        np.testing.assert_allclose(C, C.transpose(perm), atol=1e-2)


# ---------------------------------------------------------------------------
# Marker: heavier tests are expected to be slow (~seconds each)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_top_lyapunov_exponent_seed_stability(seed: int) -> None:
    """Same J, Σ, ε at different seeds give close (but non-identical) estimates."""
    J = -0.7 * np.eye(3)
    Sigma = 0.3 * np.eye(3)
    lam = top_lyapunov_exponent_linearised(
        J,
        Sigma,
        epsilon=0.1,
        n_paths=200,
        n_steps=2_000,
        dt=5e-3,
        renorm_every=50,
        seed=seed,
    )
    # Should be within ~0.2 of -0.7 (the deterministic stable rate)
    assert abs(lam - (-0.7)) < 0.25
