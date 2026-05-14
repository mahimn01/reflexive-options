"""Codim-2 bifurcation analysis at the boundary of the Hopf region (paper §3.6).

Sweeps the (σ_q, γ) parameter plane at the canonical log-normal-OI specification
of paper/theory.md §4.3 and renders:

    1. The four codim-2 regions: no-Hopf, supercritical (ℓ_1 < 0), Bautin
       tube (|ℓ_1| ≤ ε), sub-critical (ℓ_1 > 0).
    2. The Bautin curve (ℓ_1 = 0) extracted via row-wise sign-change
       interpolation, tabulated at 5–6 anchor points.
    3. The Bogdanov-Takens (BT) residual map: at every (σ_q, γ) we compute
       the saddle-node coupling κ_SN(σ_q, γ) and report whether κ_SN > 0
       and H(κ_SN) = 0 simultaneously. For the canonical regime, κ_SN is
       uniformly negative — the BT locus is empty in the physical
       (σ_q, γ) > 0 quadrant. The script logs this as an empirical theorem.

Output: paper/figures/codim2_phase_diagram.pdf, runs/codim2_analysis/<ts>/.

Run:
    python -m reflexive_options.experiments.codim2_analysis
    python -m reflexive_options.experiments.codim2_analysis --quick   # CI
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
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
    bautin_curve_scan,
    bogdanov_takens_residual_lognormal_oi,
    find_bautin_anchors,
)


@dataclass(frozen=True)
class Codim2Config:
    """Canonical specification matching paper §4.3 / §3.6.

    The (σ_q, γ) grid spans the same envelope as `ell1_phase_boundary.pdf`
    (notebooks/closed_form_ell1_derivation.py) but is denser to resolve the
    Bautin curve cleanly. `bautin_tol` controls the visual width of the
    Bautin band.
    """

    mu_q: float = float(np.log(100.0))
    T_eff: float = 0.25
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.05
    beta: float = 1.0
    a_star: float = float(np.log(100.0))
    v_star: float = 0.04
    coupling_units: float = 1.0
    sigma_q_min: float = 0.05
    sigma_q_max: float = 0.40
    n_sigma_q: int = 71
    gamma_min: float = 0.20
    gamma_max: float = 5.00
    n_gamma: int = 97
    bautin_tol: float = 5e-2
    n_anchors: int = 6


def render_phase_diagram(
    scan: object,
    bt_residual_grid: np.ndarray,
    bt_kappa_sn_grid: np.ndarray,
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    """Render the codim-2 phase diagram with the four regions and BT overlay."""
    sq = scan.sigma_q_grid  # type: ignore[attr-defined]
    gam = scan.gamma_grid  # type: ignore[attr-defined]
    ell = scan.ell_1_grid  # type: ignore[attr-defined]
    regime = scan.regime_grid  # type: ignore[attr-defined]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: 4-region phase diagram ---
    region_cmap = ListedColormap(
        ["#bdbdbd", "#3b7ec1", "#111111", "#d65f5f"]
    )  # 0 no-Hopf grey, 1 super blue, 2 Bautin black, 3 sub red
    ax_left.pcolormesh(
        sq,
        gam,
        regime.astype(np.float64),
        shading="auto",
        cmap=region_cmap,
        vmin=-0.5,
        vmax=3.5,
    )
    # Overlay the ℓ_1 = 0 sign-change contour (the Bautin curve).
    ell_for_contour = np.where(np.isnan(ell), 0.0, ell)
    cs = ax_left.contour(
        sq,
        gam,
        ell_for_contour,
        levels=[0.0],
        colors="white",
        linewidths=2.0,
        linestyles="--",
    )
    if cs.allsegs and cs.allsegs[0]:
        ax_left.clabel(cs, fmt={0.0: r"Bautin: $\ell_1 = 0$"}, fontsize=9)
    ax_left.set_xlabel(r"OI spread $\sigma_q$ (log-strike)")
    ax_left.set_ylabel(r"leverage feedback $\gamma$")
    ax_left.set_title("Codim-2 regions")

    # Synthetic legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#bdbdbd", edgecolor="black", label="No Hopf"),
        Patch(facecolor="#3b7ec1", edgecolor="black", label=r"Supercritical ($\ell_1 < 0$)"),
        Patch(
            facecolor="#111111", edgecolor="black", label=r"Bautin tube ($|\ell_1| \leq \epsilon$)"
        ),
        Patch(facecolor="#d65f5f", edgecolor="black", label=r"Sub-critical ($\ell_1 > 0$)"),
    ]
    ax_left.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.92)

    # --- Right panel: BT residual map (sign of κ_SN) ---
    # Show -1 if κ_SN <= 0 (no physical SN), else log10(|H(κ_SN)|).
    sn_positive = bt_kappa_sn_grid > 0
    overlay = np.where(sn_positive, np.log10(np.abs(bt_residual_grid) + 1e-300), np.nan)
    if np.any(sn_positive):
        pcm = ax_right.pcolormesh(sq, gam, overlay, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax_right, label=r"$\log_{10}|H(\kappa_{\mathrm{SN}})|$")
        ax_right.set_title("BT residual where $\\kappa_{\\mathrm{SN}} > 0$")
    else:
        ax_right.imshow(
            np.zeros_like(bt_kappa_sn_grid),
            extent=[float(sq.min()), float(sq.max()), float(gam.min()), float(gam.max())],
            origin="lower",
            aspect="auto",
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        ax_right.text(
            0.5,
            0.5,
            r"$\kappa_{\mathrm{SN}} \leq 0$ everywhere"
            "\n"
            r"$\Rightarrow$ no Bogdanov-Takens point"
            "\n"
            r"in the physical $(\sigma_q, \gamma) > 0$ range",
            transform=ax_right.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="black"),
        )
    ax_right.set_xlabel(r"OI spread $\sigma_q$ (log-strike)")
    ax_right.set_ylabel(r"leverage feedback $\gamma$")

    fig.suptitle(
        r"Codim-2 bifurcation structure in $(\sigma_q, \gamma)$ at canonical "
        r"$(\alpha, \kappa_v, T_{\mathrm{eff}}, \beta) = (0.05, 2, 0.25, 1)$"
        f"{title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(*, quick: bool = False) -> dict[str, object]:
    """Execute the codim-2 sweep and write all artifacts.

    Returns the metrics dict (also persisted to disk).
    """
    cfg = Codim2Config()
    if quick:
        cfg = Codim2Config(n_sigma_q=21, n_gamma=21, n_anchors=4)

    sq_grid = np.linspace(cfg.sigma_q_min, cfg.sigma_q_max, cfg.n_sigma_q)
    gam_grid = np.linspace(cfg.gamma_min, cfg.gamma_max, cfg.n_gamma)

    with timed("bautin_curve_scan"):
        scan = bautin_curve_scan(
            sigma_q_grid=sq_grid,
            gamma_grid=gam_grid,
            mu_q=cfg.mu_q,
            T_eff=cfg.T_eff,
            kappa_v=cfg.kappa_v,
            theta_v=cfg.theta_v,
            alpha=cfg.alpha,
            beta=cfg.beta,
            a_star=cfg.a_star,
            v_star=cfg.v_star,
            coupling_units=cfg.coupling_units,
            bautin_tol=cfg.bautin_tol,
        )

    # BT residual scan: compute κ_SN(σ_q, γ) and H(κ_SN) at every grid point.
    bt_kappa_sn = np.full_like(scan.ell_1_grid, np.nan)
    bt_H_residual = np.full_like(scan.ell_1_grid, np.nan)
    with timed("bt_residual_scan"):
        for i, g in enumerate(gam_grid):
            for j, s in enumerate(sq_grid):
                k_sn, H_at = bogdanov_takens_residual_lognormal_oi(
                    sigma_q=float(s),
                    gamma=float(g),
                    mu_q=cfg.mu_q,
                    T_eff=cfg.T_eff,
                    kappa_v=cfg.kappa_v,
                    theta_v=cfg.theta_v,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    a_star=cfg.a_star,
                    v_star=cfg.v_star,
                    coupling_units=cfg.coupling_units,
                )
                bt_kappa_sn[i, j] = k_sn
                bt_H_residual[i, j] = H_at

    n_bt_physical = int(np.sum(bt_kappa_sn > 0))
    bt_locus_empty = n_bt_physical == 0

    # Anchor extraction on the Bautin curve.
    anchors = find_bautin_anchors(scan, n_anchors=cfg.n_anchors)

    # Render figure.
    fig_path = FIGURES_DIR / "codim2_phase_diagram.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    render_phase_diagram(scan, bt_H_residual, bt_kappa_sn, fig_path)

    # Persist artifacts.
    run_dir = make_run_dir("codim2_analysis")
    save_config(run_dir, asdict(cfg))
    metrics = {
        "n_sigma_q": int(cfg.n_sigma_q),
        "n_gamma": int(cfg.n_gamma),
        "n_no_hopf": int(np.sum(scan.regime_grid == 0)),
        "n_supercritical": int(np.sum(scan.regime_grid == 1)),
        "n_bautin_tube": int(np.sum(scan.regime_grid == 2)),
        "n_subcritical": int(np.sum(scan.regime_grid == 3)),
        "bautin_anchors": [
            {"sigma_q": float(s), "gamma": float(g), "kappa_star": float(k)} for s, g, k in anchors
        ],
        "bt_locus_empty_in_physical_range": bt_locus_empty,
        "n_bt_physical_kappa_sn_positive_cells": n_bt_physical,
        "kappa_sn_min": float(np.nanmin(bt_kappa_sn)),
        "kappa_sn_max": float(np.nanmax(bt_kappa_sn)),
        "figure_path": str(fig_path),
    }
    save_metrics(run_dir, metrics)
    np.savez_compressed(
        run_dir / "scan.npz",
        sigma_q_grid=scan.sigma_q_grid,
        gamma_grid=scan.gamma_grid,
        ell_1_grid=scan.ell_1_grid,
        kappa_star_grid=scan.kappa_star_grid,
        omega_star_grid=scan.omega_star_grid,
        regime_grid=scan.regime_grid,
        bt_kappa_sn=bt_kappa_sn,
        bt_H_residual=bt_H_residual,
    )

    # Reproducibility README.
    readme = run_dir / "README.md"
    readme.write_text(_readme_text(metrics, cfg))

    print(json.dumps(metrics, indent=2))
    print(f"figure -> {fig_path}")
    print(f"run dir -> {run_dir}")
    return metrics


def _readme_text(metrics: dict[str, object], cfg: Codim2Config) -> str:
    anchors = cast("list[dict[str, float]]", metrics["bautin_anchors"])
    anchors_md = "\n".join(
        f"| {i + 1} | {a['sigma_q']:.4f} | {a['gamma']:.4f} | {a['kappa_star']:.4f} |"
        for i, a in enumerate(anchors)
    )
    bt_status = (
        "**EMPTY in the physical range.** "
        f"All {cfg.n_sigma_q * cfg.n_gamma} grid cells have "
        f"$\\kappa_{{\\mathrm{{SN}}}} \\leq 0$ "
        f"(min {metrics['kappa_sn_min']:.3f}, max {metrics['kappa_sn_max']:.3f})."
        if metrics["bt_locus_empty_in_physical_range"]
        else f"NON-EMPTY: {metrics['n_bt_physical_kappa_sn_positive_cells']} cells "
        f"have $\\kappa_{{\\mathrm{{SN}}}} > 0$."
    )
    return f"""# Codim-2 bifurcation analysis — reproducibility

This run produces the §3.6 codim-2 phase diagram in (σ_q, γ) at the
canonical log-normal-OI specification.

## Reproduce

```bash
python -m reflexive_options.experiments.codim2_analysis
```

Outputs:
- `paper/figures/codim2_phase_diagram.pdf`
- `runs/codim2_analysis/<ts>/{{config,metrics}}.json`
- `runs/codim2_analysis/<ts>/scan.npz`  (full grids)

## Bautin curve anchors (ℓ_1 = 0 in (σ_q, γ))

| # | σ_q | γ | κ★ at the crossing |
|---|---|---|---|
{anchors_md}

## Bogdanov-Takens locus

{bt_status}

The structural reason: for the closed-form log-normal-OI parameterization,
c_0(κ) is linear in κ and the saddle-node coupling is

  κ_SN = ½ β γ / (G_y α κ_v + G_v β γ).

At the canonical equilibrium (a★ = log 100, v★ = 0.04, μ_q = log 100), the
log-normal-OI partial G_v < 0 dominates G_y α κ_v in magnitude over the
entire scanned (σ_q, γ) > 0 quadrant, so the denominator is negative and
κ_SN < 0. The Bogdanov-Takens bifurcation cannot occur at any positive
coupling for this parameter family — economically, the dealer-gamma +
leverage parameter regime is structurally Hopf-only.

## Region counts (N = {cfg.n_sigma_q * cfg.n_gamma} cells)

- No Hopf:        {metrics["n_no_hopf"]}
- Supercritical:  {metrics["n_supercritical"]}
- Bautin tube:    {metrics["n_bautin_tube"]}  (|ℓ_1| ≤ {cfg.bautin_tol})
- Sub-critical:   {metrics["n_subcritical"]}

Implementation: `src/reflexive_options/theory/bifurcation.py`
(`bautin_curve_scan`, `kappa_saddle_node_lognormal_oi`,
`bogdanov_takens_residual_lognormal_oi`, `find_bautin_anchors`).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Coarse grid (21x21) for CI; default is the 71x97 production grid.",
    )
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
