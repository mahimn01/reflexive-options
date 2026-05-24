"""No-Hopf-wedge bifurcation taxonomy at the canonical specification (paper §3.5 extension).

Scans (σ_q, γ) on a grid that covers the §3.5 no-Hopf wedge with margin, classifies
each cell via `is_in_no_hopf_wedge`, and for every wedge cell sweeps κ ∈ [0, κ_max]
looking for ANY codim-1 bifurcation (Hopf, saddle-node via c_0 = 0, or trace flip
via c_2 = 0). Tests the open question left by §3.5 + §3.7: when the Hopf is
forbidden, what bifurcation IS accessible? Answer (Theorem 6): none — in the wedge
the equilibrium is globally asymptotically stable on the physical κ-half-line.

Output: paper/figures/saddle_node_wedge.pdf, runs/saddle_node_wedge/<ts>/.

Run:
    python -m reflexive_options.experiments.saddle_node_wedge
    python -m reflexive_options.experiments.saddle_node_wedge --quick   # CI
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.theory.bifurcation import (
    NoHopfWedgeScanResult,
    scan_no_hopf_wedge_bifurcations,
)


@dataclass(frozen=True)
class SaddleNodeWedgeConfig:
    """Canonical §4.3 / §3.6 specification, plus the (σ_q, γ) scan window.

    The window is wider than the §3.5 task-spec window so that the wedge is
    actually populated at the canonical (α, κ_v, β) triangle (where H(0) > 0
    requires γ < 2 α κ_v (α + κ_v)/β ≈ 0.41).
    """

    mu_q: float = float(np.log(100.0))
    T_eff: float = 0.25
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.05
    beta: float = 1.0
    coupling_units: float = 1.0
    sigma_q_min: float = 0.02
    sigma_q_max: float = 0.40
    n_sigma_q: int = 41
    gamma_min: float = 0.05
    gamma_max: float = 5.0
    n_gamma: int = 41
    kappa_max: float = 100.0
    n_kappa_samples: int = 80


def run(*, config: SaddleNodeWedgeConfig, quick: bool = False) -> NoHopfWedgeScanResult:
    sigma_q_grid = np.linspace(config.sigma_q_min, config.sigma_q_max, config.n_sigma_q)
    gamma_grid = np.linspace(config.gamma_min, config.gamma_max, config.n_gamma)
    return scan_no_hopf_wedge_bifurcations(
        sigma_q_grid=sigma_q_grid,
        gamma_grid=gamma_grid,
        mu_q=config.mu_q,
        T_eff=config.T_eff,
        kappa_v=config.kappa_v,
        alpha=config.alpha,
        beta=config.beta,
        a_star=config.mu_q,
        v_star=config.theta_v,
        coupling_units=config.coupling_units,
        kappa_max=10.0 if quick else config.kappa_max,
        n_kappa_samples=10 if quick else config.n_kappa_samples,
    )


def render_figure(result: NoHopfWedgeScanResult, *, out_path: Path) -> None:
    """Two-panel figure: wedge classifier (left), spectral abscissa in wedge (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sq = result.sigma_q_grid
    gam = result.gamma_grid

    # Left: 3-class classifier. 0 = globally stable wedge; 1 = Hopf admitted; 2 = wedge but not globally stable.
    classifier = np.full(result.in_wedge_grid.shape, 1, dtype=np.int_)
    classifier[result.globally_stable_grid] = 0
    anomaly = result.in_wedge_grid & ~result.globally_stable_grid
    classifier[anomaly] = 2
    cmap = ListedColormap(["#2ca02c", "#d3d3d3", "#d62728"])
    ax1.pcolormesh(sq, gam, classifier, cmap=cmap, vmin=0, vmax=2, shading="auto")
    ax1.set_xlabel(r"OI spread $\sigma_q$ (log-strike)")
    ax1.set_ylabel(r"leverage feedback $\gamma$")
    ax1.set_title(
        f"No-Hopf-wedge taxonomy ({result.n_wedge_cells}/{result.in_wedge_grid.size} cells in wedge)"
    )
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#2ca02c", label="Globally stable (Theorem 6(a))"),
        Patch(facecolor="#d3d3d3", label=r"Hopf admitted ($\exists\,\kappa^\star > 0$)"),
        Patch(facecolor="#d62728", label="Wedge ∩ not-globally-stable (anomaly)"),
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)

    # Right: spectral abscissa, masked outside wedge.
    abs_grid = result.spectral_abscissa_grid.copy()
    abs_grid[~result.in_wedge_grid] = np.nan
    pc = ax2.pcolormesh(sq, gam, abs_grid, cmap="viridis", shading="auto")
    ax2.set_xlabel(r"OI spread $\sigma_q$ (log-strike)")
    ax2.set_ylabel(r"leverage feedback $\gamma$")
    ax2.set_title(rf"Spectral abscissa in wedge (max = {result.wedge_max_spectral_abscissa:.2e})")
    cb = fig.colorbar(pc, ax=ax2)
    cb.set_label(r"$\max_{\kappa\in[0, \kappa_{\max}]}\,\max_i\,\mathrm{Re}\,\lambda_i(J(\kappa))$")

    fig.suptitle(
        f"Theorem 6 (no-Hopf-wedge): all {result.n_globally_stable_cells} wedge cells "
        f"are globally asymptotically stable (κ_max = {result.kappa_max_scanned:.0f})",
        fontsize=11,
    )
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="CI smoke-test mode (small grid).")
    args = parser.parse_args()

    config = SaddleNodeWedgeConfig()
    if args.quick:
        config = SaddleNodeWedgeConfig(n_sigma_q=11, n_gamma=11)

    run_dir = make_run_dir("saddle_node_wedge")
    save_config(run_dir, config)
    with timed("saddle_node_wedge scan"):
        result = run(config=config, quick=args.quick)

    metrics = {
        "n_total_cells": int(result.in_wedge_grid.size),
        "n_wedge_cells": int(result.n_wedge_cells),
        "n_globally_stable_cells": int(result.n_globally_stable_cells),
        "n_positive_saddle_node_cells": int(result.n_positive_saddle_node_cells),
        "wedge_max_spectral_abscissa": float(result.wedge_max_spectral_abscissa),
        "kappa_max_scanned": float(result.kappa_max_scanned),
        "verdict": (
            "Theorem 6(a) — globally stable"
            if (
                result.n_positive_saddle_node_cells == 0
                and (
                    not np.isfinite(result.wedge_max_spectral_abscissa)
                    or result.wedge_max_spectral_abscissa < 0
                )
            )
            else "regime-dependent — see kappa_sn_grid"
        ),
    }
    save_metrics(run_dir, metrics)
    np.savez(
        run_dir / "scan.npz",
        sigma_q_grid=result.sigma_q_grid,
        gamma_grid=result.gamma_grid,
        in_wedge_grid=result.in_wedge_grid,
        globally_stable_grid=result.globally_stable_grid,
        kappa_sn_grid=result.kappa_sn_grid,
        spectral_abscissa_grid=result.spectral_abscissa_grid,
    )

    out_pdf = FIGURES_DIR / "saddle_node_wedge.pdf"
    render_figure(result, out_path=out_pdf)
    print(f"wrote {out_pdf}")
    print(f"wrote {run_dir / 'metrics.json'}")
    print(f"verdict: {metrics['verdict']}")


if __name__ == "__main__":
    main()
