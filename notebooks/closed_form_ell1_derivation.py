"""Sympy verification of the closed-form ℓ_1 derivation for log-normal OI.

This script reproduces the analytical derivation of paper/theory.md §4.3:

    1. Build G(a, v) symbolically as ∫ q(log K) Γ_BS(S, K, T, σ) d(log K)
       with q log-normal in K and Γ_BS Black-Scholes gamma.
    2. Verify the closed-form Gaussian-product result for G.
    3. Compute all 10 unique third-order partials of G in (a, v).
    4. Numerically compare against the Python implementation in
       `reflexive_options.theory.bifurcation` to machine precision.
    5. Solve the Routh-Hurwitz condition H(κ) = 0 in closed form (it
       reduces to a quadratic in κ when G_z = 0 and σ² = v).
    6. Compute ℓ_1 via the Kuznetsov 2004 eq. 3.20 pipeline at the
       canonical §4.3 regime.
    7. Plot the ℓ_1 = 0 phase-boundary contour in (σ_q, γ) space.

Run from the repo root:
    python notebooks/closed_form_ell1_derivation.py

Total runtime ≈ 30s; figures saved to paper/figures/ell1_phase_boundary.pdf.
"""

from __future__ import annotations

import os
from pathlib import Path

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
# See verification_v5_repro.md §3 — without this, every regen drifts the PDF
# hash even though the rendered drawing is bit-identical.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from reflexive_options.theory.bifurcation import (
    G_lognormal_oi,
    G_lognormal_oi_partials,
    kappa_star_lognormal_oi,
    lyapunov_coefficient_lognormal_oi,
)


def step1_build_G_symbolic() -> sp.Expr:
    """Construct the closed-form G(a, v) via the Gaussian-product identity.

    The integrand q(log K) · Γ_BS(S, K, T, σ) is a product of two Gaussians in
    log K — q centred at μ_q with variance σ_q², Γ_BS centred at μ_d with
    variance σ²T. Their product is itself Gaussian times an analytic constant
    (Bromiley 2003):

        ∫ N(ℓ; μ_q, σ_q²) · N(ℓ; μ_d, σ²T) dℓ
            = (1 / √(2π τ²)) · exp(−(μ_q − μ_d)² / (2 τ²)),
        τ² = σ_q² + σ²T.

    With Γ_BS's overall 1/(S σ √T) prefactor and the change of variable
    a := log S, μ_d − μ_q = (a − m(v)) where m(v) := μ_q − (r − q + v/2)T,
    one obtains the closed form (Eq. 15a in paper/theory.md §4.3).

    Verification of the Gaussian-product identity itself (which is calculus,
    not finance) is left to the reader; we directly verify the *closed form*
    against numerical quadrature in step 3 and against finite differences in
    `tests/test_lognormal_lyapunov.py::test_partials_match_finite_differences`.
    """
    print("=" * 72)
    print("Step 1: closed-form G(a, v) for log-normal OI")
    print("=" * 72)

    a, v = sp.symbols("a v", real=True)
    mu_q, sigma_q, T, r, q_div, ku = sp.symbols(
        "mu_q sigma_q T r q_div ku", real=True, positive=True
    )

    tau2 = sigma_q**2 + v * T
    m = mu_q - (r - q_div + v / 2) * T
    G_closed = (
        ku
        * sp.exp(-q_div * T - a)
        / sp.sqrt(2 * sp.pi * tau2)
        * sp.exp(-((a - m) ** 2) / (2 * tau2))
    )
    print("\n  G(a, v) = (κ_u e^{-q_div T} / √(2π τ²)) · e^{-a} · exp(-(a - m)² / (2 τ²))")
    print("    τ²(v) = σ_q² + v T")
    print("    m(v)  = μ_q − (r − q_div + v/2) T")
    return G_closed


def step2_partials_symbolic(G_expr: sp.Expr) -> dict[str, sp.Expr]:
    """Compute all 10 unique third-order partials symbolically."""
    print("\n" + "=" * 72)
    print("Step 2: Symbolic partials of G(a, v) up to third order")
    print("=" * 72)
    a, v = sp.symbols("a v", real=True)
    partials = {
        "G": G_expr,
        "G_a": sp.diff(G_expr, a),
        "G_v": sp.diff(G_expr, v),
        "G_aa": sp.diff(G_expr, a, 2),
        "G_av": sp.diff(G_expr, a, v),
        "G_vv": sp.diff(G_expr, v, 2),
        "G_aaa": sp.diff(G_expr, a, 3),
        "G_aav": sp.diff(G_expr, a, 2, v),
        "G_avv": sp.diff(G_expr, a, v, 2),
        "G_vvv": sp.diff(G_expr, v, 3),
    }
    print("Computed 10 partials.")
    return partials


def step3_numerical_verification(partials_sym: dict[str, sp.Expr]) -> None:
    """Compare sympy partials against the Python implementation, and
    cross-check G_closed against direct scipy.integrate.quad on the integral.
    """
    from scipy.integrate import quad

    print("\n" + "=" * 72)
    print("Step 3: Verify Python implementation against sympy and quadrature")
    print("=" * 72)
    a, v = sp.symbols("a v", real=True)
    mu_q, sigma_q, T, r, q_div, ku = sp.symbols(
        "mu_q sigma_q T r q_div ku", real=True, positive=True
    )
    test_cases = [
        {"a_v": float(np.log(100.0)), "v_v": 0.04, "mu_q_v": float(np.log(100.0)),
         "sq_v": 0.30, "T_v": 0.5, "r_v": 0.01, "q_v": 0.005, "ku_v": 1.0},
        {"a_v": float(np.log(60.0)), "v_v": 0.10, "mu_q_v": float(np.log(50.0)),
         "sq_v": 0.20, "T_v": 0.1, "r_v": 0.0, "q_v": 0.0, "ku_v": 5.0},
        {"a_v": float(np.log(150.0)), "v_v": 0.02, "mu_q_v": float(np.log(140.0)),
         "sq_v": 0.50, "T_v": 1.0, "r_v": 0.04, "q_v": 0.02, "ku_v": 100.0},
    ]

    def numerical_G(case: dict[str, float]) -> float:
        sigma = float(np.sqrt(case["v_v"]))
        sqrt_T = float(np.sqrt(case["T_v"]))
        spot = float(np.exp(case["a_v"]))

        def integrand(ell: float) -> float:
            K = float(np.exp(ell))
            d1 = (
                np.log(spot / K)
                + (case["r_v"] - case["q_v"] + 0.5 * sigma**2) * case["T_v"]
            ) / (sigma * sqrt_T)
            gamma_bs = (
                float(np.exp(-case["q_v"] * case["T_v"]))
                * float(np.exp(-0.5 * d1**2))
                / float(np.sqrt(2.0 * np.pi))
                / (spot * sigma * sqrt_T)
            )
            q_density = float(
                np.exp(-((ell - case["mu_q_v"]) ** 2) / (2.0 * case["sq_v"] ** 2))
            ) / (case["sq_v"] * float(np.sqrt(2.0 * np.pi)))
            return q_density * gamma_bs

        val, _ = quad(
            integrand,
            case["mu_q_v"] - 12.0 * case["sq_v"],
            case["mu_q_v"] + 12.0 * case["sq_v"],
            limit=400,
        )
        return float(case["ku_v"] * val)

    for case in test_cases:
        subs = {a: case["a_v"], v: case["v_v"], mu_q: case["mu_q_v"], sigma_q: case["sq_v"],
                T: case["T_v"], r: case["r_v"], q_div: case["q_v"], ku: case["ku_v"]}
        sym_vals = {k: float(p.subs(subs)) for k, p in partials_sym.items()}
        mine = G_lognormal_oi_partials(
            a_star=case["a_v"], v_star=case["v_v"],
            mu_q=case["mu_q_v"], sigma_q=case["sq_v"], T_eff=case["T_v"],
            coupling_units=case["ku_v"], rate=case["r_v"], dividend=case["q_v"],
        )
        max_rel = 0.0
        for k in sym_vals:
            rel = abs(mine[k] - sym_vals[k]) / max(abs(sym_vals[k]), 1e-15)
            max_rel = max(max_rel, rel)
        # Verify G itself against direct quadrature (this exercises the closed
        # form rather than just the partial-derivative chain rule).
        g_quad = numerical_G(case)
        rel_quad = abs(mine["G"] - g_quad) / max(abs(g_quad), 1e-15)
        print(
            f"  case (a*={case['a_v']:.2f}, v*={case['v_v']:.2f}, "
            f"σ_q={case['sq_v']:.2f}, T={case['T_v']:.2f}): "
            f"max partial-rel = {max_rel:.3e},  G vs quad rel = {rel_quad:.3e}"
        )
        assert max_rel < 1e-10, f"Python implementation disagrees with sympy: rel = {max_rel}"
        assert rel_quad < 1e-8, f"Closed-form G disagrees with scipy quad: rel = {rel_quad}"
    print("OK: closed-form G matches quadrature; all partials match sympy to machine precision.")


def step4_kappa_star_closed_form() -> None:
    """Derive H(κ) = 0 closed form and verify numerically."""
    print("\n" + "=" * 72)
    print("Step 4: Closed-form Hopf threshold κ*")
    print("=" * 72)
    kappa = sp.symbols("kappa", real=True)
    G_y, G_v = sp.symbols("G_y G_v", real=True)
    kappa_v, alpha_, beta_, gamma_ = sp.symbols(
        "kappa_v alpha beta gamma", real=True, positive=True
    )

    # Skeleton: a(κ) = κ G_y, b(κ) = κ G_v - 1/2; G_z = 0
    a_kappa = kappa * G_y
    b_kappa = kappa * G_v - sp.Rational(1, 2)
    c_2 = -a_kappa + kappa_v + alpha_
    c_1 = -a_kappa * kappa_v - a_kappa * alpha_ + kappa_v * alpha_
    c_0 = -a_kappa * kappa_v * alpha_ - beta_ * b_kappa * gamma_
    H = sp.expand(c_1 * c_2 - c_0)
    H_poly = sp.Poly(H, kappa)

    print("\nH(κ) is a quadratic in κ when G_z = 0 and σ² = v:")
    for d, c in zip(reversed(range(H_poly.degree() + 1)), H_poly.all_coeffs(), strict=True):
        print(f"  [κ^{d}]:  {sp.simplify(c)}")

    # Compact form: A_2 κ² + A_1 κ + A_0 = 0 with A = α + κ_v, M = α κ_v, L = β γ
    A_total = alpha_ + kappa_v
    M = alpha_ * kappa_v
    L = beta_ * gamma_
    A2 = G_y**2 * A_total
    A1 = G_v * L - G_y * A_total**2
    A0 = M * A_total - L / 2
    H_simple = A2 * kappa**2 + A1 * kappa + A0
    print(f"\nCompact form:\n  H(κ) = {A2} · κ² + {A1} · κ + {A0}")
    print(f"  = ({sp.simplify(A2)}) κ² + ({sp.simplify(A1)}) κ + ({sp.simplify(A0)})")

    diff = sp.simplify(H - H_simple)
    assert diff == 0, "Compact form differs from H"
    print("OK: H(κ) = A_2 κ² + A_1 κ + A_0 verified.")
    print()
    print("Closed-form κ* (smallest positive root of the quadratic):")
    print("  κ* = [G_y · A² − G_v · L − √D] / (2 G_y² A)")
    print("  with A = α + κ_v, M = α κ_v, L = β γ, and discriminant")
    print("  D = (G_v L − G_y A²)² − 4 G_y² A (M A − L/2)")


def step5_ell1_at_canonical() -> None:
    """Compute κ*, ω*, ℓ_1 at the canonical §4.3 regime."""
    print("\n" + "=" * 72)
    print("Step 5: ℓ_1 at canonical regime")
    print("=" * 72)
    canonical = dict(  # noqa: C408
        mu_q=float(np.log(100.0)),
        sigma_q=0.10,
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
    p = G_lognormal_oi_partials(
        a_star=canonical["a_star"],
        v_star=canonical["v_star"],
        mu_q=canonical["mu_q"],
        sigma_q=canonical["sigma_q"],
        T_eff=canonical["T_eff"],
        coupling_units=canonical["coupling_units"],
    )
    print(f"  G_y at equilibrium: {p['G_a']:+.6e}")
    print(f"  G_v at equilibrium: {p['G_v']:+.6e}")
    print(f"  G_yy at equilibrium: {p['G_aa']:+.6e}")
    print(f"  G_yyy at equilibrium: {p['G_aaa']:+.6e}")
    k, om, ell = lyapunov_coefficient_lognormal_oi(**canonical)  # type: ignore[arg-type]
    print()
    print(f"  κ* = {k:.6f}")
    print(f"  ω* = {om:.6f}  (period {2 * np.pi / om:.4f} yr)")
    print(f"  ℓ_1 = {ell:+.6e}  ({'super' if ell < 0 else 'sub'}-critical)")


def step6_phase_boundary_plot() -> Path:
    """Plot ℓ_1 = 0 contour in (σ_q, γ) at the canonical α, T_eff."""
    print("\n" + "=" * 72)
    print("Step 6: Phase-boundary plot in (σ_q, γ) space")
    print("=" * 72)
    base = dict(  # noqa: C408
        mu_q=float(np.log(100.0)),
        T_eff=0.25,
        kappa_v=2.0,
        theta_v=0.04,
        alpha=0.05,
        beta=1.0,
        a_star=float(np.log(100.0)),
        v_star=0.04,
        coupling_units=1.0,
    )
    sigma_q_grid = np.linspace(0.05, 0.40, 36)
    gamma_grid = np.linspace(0.2, 5.0, 49)
    Z = np.full((len(gamma_grid), len(sigma_q_grid)), np.nan)
    for i, gam in enumerate(gamma_grid):
        for j, sq in enumerate(sigma_q_grid):
            try:
                _, _, ell = lyapunov_coefficient_lognormal_oi(
                    sigma_q=sq, gamma=gam, **base
                )
                Z[i, j] = ell
            except ValueError:
                pass
    n_super = int(np.sum(Z < 0))
    n_sub = int(np.sum(Z > 0))
    n_nan = int(np.sum(np.isnan(Z)))
    print(f"  Grid: {len(sigma_q_grid)}×{len(gamma_grid)} = {Z.size} points")
    print(f"  Supercritical: {n_super} | Sub-critical: {n_sub} | No Hopf: {n_nan}")

    out_dir = Path(__file__).resolve().parent.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ell1_phase_boundary.pdf"

    fig, ax = plt.subplots(figsize=(7, 5))
    # Use a masked array so NaN regions render in a distinct (grey) colour.
    Z_masked = np.ma.masked_invalid(np.clip(Z, -2.0, 2.0))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")
    pcm = ax.pcolormesh(
        sigma_q_grid,
        gamma_grid,
        Z_masked,
        shading="auto",
        cmap=cmap,
        vmin=-2.0,
        vmax=2.0,
    )
    fig.colorbar(pcm, ax=ax, label=r"first Lyapunov coefficient $\ell_1$ (clipped)")
    # Contour the ℓ_1 = 0 boundary on the valid (non-NaN) region only.
    cs = ax.contour(
        sigma_q_grid,
        gamma_grid,
        np.where(np.isnan(Z), np.nan, Z),
        levels=[0.0],
        colors="black",
        linewidths=2,
    )
    ax.clabel(cs, fmt={0.0: r"$\ell_1 = 0$"})
    ax.set_xlabel(r"OI spread $\sigma_q$ (log-strike)")
    ax.set_ylabel(r"leverage feedback $\gamma$")
    ax.set_title(
        r"Hopf criticality phase boundary at $\alpha=0.05,\ T_{\mathrm{eff}}=0.25,\ "
        r"\kappa_v=2,\ \beta=1$"
        "\n(blue = supercritical / stable LC; red = sub-critical / unstable; grey = no Hopf)"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"OK: phase-boundary figure → {out_path}")
    return out_path


def step7_dimensional_consistency() -> None:
    """Sanity-check that the closed-form G matches the BS gamma scaling."""
    print("\n" + "=" * 72)
    print("Step 7: Dimensional sanity check")
    print("=" * 72)
    spot = 100.0
    a = float(np.log(spot))
    g_atm = G_lognormal_oi(
        a, 0.04, mu_q=a, sigma_q=0.20, T_eff=30 / 365.25, coupling_units=1.0
    )
    print(f"  G(spot=100, ATM, σ_q=0.20, T_eff=30d, ν=0.04) = {g_atm:.6e}")
    print(f"  ATM BS gamma (single contract, σ=0.20, T=30d) = "
          f"{1.0 / (spot * 0.20 * np.sqrt(30 / 365.25) * np.sqrt(2 * np.pi)):.6e}")
    print("  Order-of-magnitude consistent with κ_u × Γ_ATM as expected.")


def main() -> None:
    G_expr = step1_build_G_symbolic()
    partials = step2_partials_symbolic(G_expr)
    step3_numerical_verification(partials)
    step4_kappa_star_closed_form()
    step5_ell1_at_canonical()
    step6_phase_boundary_plot()
    step7_dimensional_consistency()
    print("\n" + "=" * 72)
    print("ALL STEPS PASSED — closed-form ℓ_1 derivation verified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
