"""Hawkes-SV equivalence numerical anchor (Theorem 2, paper §3.7).

Computes the SV-equivalent branching ratio n_SV(κ) over the §4.2
canonical regime κ ∈ [0, 2·κ★] and produces:

  * `runs/hawkes_sv_equivalence/<timestamp>/{config,metrics}.json`
  * `paper/figures/hawkes_sv_equivalence.pdf`

The figure shows the n_SV(κ) curve with the κ★ anchor at n_SV = 1
(Hardiman-empirical critical value), matching the headline claim of
Theorem 2: the Hopf threshold IS the n = 1 boundary in the
continuous-time SV language.

Invocation::

    python -m reflexive_options.experiments.hawkes_sv_equivalence
    python -m reflexive_options.experiments.hawkes_sv_equivalence --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from functools import partial

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.theory.bifurcation import (
    jacobian_3d,
    jacobian_eigenvalues,
    routh_hurwitz_H,
)
from reflexive_options.theory.hawkes_equivalence import (
    HawkesEquivalenceResult,
    hawkes_branching_ratio_curve,
    n_sv_at_kappa,
)

# ---------------------------------------------------------------------------
# §4.2 canonical regime — locked to paper/theory.md §4.2 and
# tests/test_paper_section_4_2.py. Constant-vol surrogate convention.
# ---------------------------------------------------------------------------
_G_X = 0.5
_G_V = -0.5
_G_Z = -0.5
_ALPHA = 0.5
_BETA = 1.0
_GAMMA = 0.5
_KAPPA_V = 2.0

# Locked headline values from §4.2 Table.
_KAPPA_STAR = 0.8964


@dataclass(frozen=True)
class HawkesSVConfig:
    """Configuration for the n_SV(κ) scan."""

    kappa_min: float = 0.0
    kappa_max_multiplier: float = 2.0  # scan κ ∈ [0, 2·κ★]
    n_kappa: int = 1001


def _jacobian_at(kappa: float) -> NDArray[np.float64]:
    """Jacobian under the §4.2 constant-vol surrogate (∂_v σ² = 0)."""
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


def run(cfg: HawkesSVConfig) -> tuple[HawkesEquivalenceResult, dict[str, object]]:
    """Run the n_SV scan and return (result, metrics-dict)."""
    kappa_max = cfg.kappa_max_multiplier * _KAPPA_STAR
    kappa_grid = np.linspace(cfg.kappa_min, kappa_max, cfg.n_kappa).astype(np.float64)
    # The endpoint κ = 0 produces a structural zero eigenvalue (spot mode is
    # marginal). The branching ratio formula treats this gracefully because we
    # use β₀ = max_{κ ∈ [0,κ★]} (-λ_max(κ)), not |λ_max(0)| (which is zero).
    with timed("hawkes_sv_n_scan"):
        result = hawkes_branching_ratio_curve(kappa_grid, _jacobian_at)

    n_at_kappa_star = (
        float(np.interp(_KAPPA_STAR, result.kappa_grid, result.n_sv))
        if result.kappa_star is not None
        else float("nan")
    )
    n_at_2_kappa_star = float(np.interp(2.0 * _KAPPA_STAR, result.kappa_grid, result.n_sv))

    # High-precision κ★ via Brent on H(κ) = c_1·c_2 - c_0. The grid-located
    # κ★ is bounded by the κ-grid resolution (≈ 1.79e-3 at n_kappa=1001), so
    # for the §3.9 truncation-vs-noise audit we record the Brent root too.
    def _H(k: float) -> float:
        eig = jacobian_eigenvalues(_jacobian_at(k))
        _, _, _, h = routh_hurwitz_H(eig)
        return h

    kappa_star_brent: float | None
    n_sv_at_kappa_star_brent: float | None
    criticality_residual_brent: float | None
    try:
        kappa_star_brent = float(brentq(_H, 0.85, 0.90, xtol=1e-14, rtol=1e-15))
        n_sv_at_kappa_star_brent = n_sv_at_kappa(
            kappa_star_brent, _jacobian_at, beta_zero=result.beta_zero
        )
        criticality_residual_brent = abs(n_sv_at_kappa_star_brent - 1.0)
    except (ValueError, RuntimeError):
        kappa_star_brent = None
        n_sv_at_kappa_star_brent = None
        criticality_residual_brent = None

    metrics: dict[str, object] = {
        "kappa_star_paper": _KAPPA_STAR,
        "kappa_star_grid": result.kappa_star,
        "kappa_star_brent": kappa_star_brent,
        "beta_zero": result.beta_zero,
        "kappa_at_beta_zero": result.kappa_at_beta_zero,
        "n_sv_at_kappa_star": n_at_kappa_star,
        "n_sv_at_kappa_star_brent": n_sv_at_kappa_star_brent,
        "n_sv_at_2_kappa_star": n_at_2_kappa_star,
        "criticality_residual": abs(n_at_kappa_star - 1.0),
        "criticality_residual_brent": criticality_residual_brent,
        "n_kappa": cfg.n_kappa,
        "kappa_max": kappa_max,
    }
    print(f"  κ★ (paper)             = {_KAPPA_STAR:.6f}")
    print(f"  κ★ (grid)              = {result.kappa_star}")
    print(f"  κ★ (Brent, machine ε)  = {kappa_star_brent}")
    print(f"  β₀                     = {result.beta_zero:.6f}")
    print(f"  κ at β₀                = {result.kappa_at_beta_zero:.6f}")
    print(f"  n_SV(κ★)               = {n_at_kappa_star:.6f}  (target 1.0)")
    print(f"  n_SV(κ★_brent)         = {n_sv_at_kappa_star_brent}  (target 1.0)")
    print(f"  n_SV(2·κ★)             = {n_at_2_kappa_star:.6f}  (past Hopf)")
    print(f"  |n_SV(κ★) − 1|          = {metrics['criticality_residual']:.3e}")
    print(f"  |n_SV(κ★_brent) − 1|    = {criticality_residual_brent}")
    return result, metrics


def render_figure(result: HawkesEquivalenceResult, out_path: str) -> None:
    """Render n_SV(κ) curve with the κ★ anchor at n = 1."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)

    # Top: leading eigenvalue real part vs κ.
    ax_lam = axes[0]
    ax_lam.plot(
        result.kappa_grid,
        result.lambda_max_real,
        color="C0",
        linewidth=1.4,
        label=r"$\lambda_{\max}^{\mathrm{Re}}(\kappa)$",
    )
    ax_lam.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    if result.kappa_star is not None:
        ax_lam.axvline(
            _KAPPA_STAR,
            color="C3",
            linestyle=":",
            linewidth=1.0,
            label=rf"$\kappa^\star = {_KAPPA_STAR:.4f}$",
        )
    ax_lam.axvline(
        result.kappa_at_beta_zero,
        color="C2",
        linestyle="-.",
        linewidth=1.0,
        label=rf"$\kappa$ @ $\beta_0$ ({result.kappa_at_beta_zero:.3f})",
    )
    ax_lam.set_ylabel(r"$\mathrm{Re}\,\lambda_{\max}$")
    ax_lam.set_title("Leading-eigenvalue real part of $J(\\kappa)$")
    ax_lam.grid(True, alpha=0.3)
    ax_lam.legend(loc="lower left", fontsize=9)

    # Bottom: n_SV(κ) curve.
    ax_n = axes[1]
    ax_n.plot(
        result.kappa_grid,
        result.n_sv,
        color="C0",
        linewidth=1.4,
        label=r"$n_{\mathrm{SV}}(\kappa) = 1 + \lambda_{\max}/\beta_0$",
    )
    ax_n.axhline(
        1.0,
        color="C3",
        linestyle="--",
        linewidth=0.9,
        label=r"Hardiman 2013: $n \approx 1$",
    )
    ax_n.axhline(0.0, color="grey", linestyle=":", linewidth=0.6)
    if result.kappa_star is not None:
        ax_n.axvline(
            _KAPPA_STAR,
            color="C3",
            linestyle=":",
            linewidth=1.0,
            label=rf"$\kappa^\star = {_KAPPA_STAR:.4f}$",
        )
    ax_n.set_xlabel(r"$\kappa$ (reflexive coupling)")
    ax_n.set_ylabel(r"$n_{\mathrm{SV}}(\kappa)$")
    ax_n.set_title(r"SV-equivalent Hawkes branching ratio (Theorem 2, paper §3.7)")
    ax_n.grid(True, alpha=0.3)
    ax_n.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Hawkes-SV equivalence at the Hopf boundary "
        r"($n_{\mathrm{SV}}(\kappa^\star) = 1$ by construction)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"OK: hawkes-sv-equivalence figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduce n_kappa for CI smoke")
    args = parser.parse_args()

    cfg = HawkesSVConfig(n_kappa=201) if args.quick else HawkesSVConfig()

    run_dir = make_run_dir("hawkes_sv_equivalence")
    save_config(run_dir, cfg)

    result, metrics = run(cfg)
    metrics["config"] = asdict(cfg)
    save_metrics(run_dir, metrics)
    print(f"Wrote results to: {run_dir}")

    # Spot for posterity.
    np.savez(
        run_dir / "n_sv_curve.npz",
        kappa_grid=result.kappa_grid,
        lambda_max_real=result.lambda_max_real,
        n_sv=result.n_sv,
        beta_zero=result.beta_zero,
    )

    # Use a partial to keep the function signature regular for the unused-arg
    # warning, even though we don't currently parametrise the renderer.
    render = partial(render_figure, result)
    out_path = FIGURES_DIR / "hawkes_sv_equivalence.pdf"
    render(str(out_path))


if __name__ == "__main__":
    main()
