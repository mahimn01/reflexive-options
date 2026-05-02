"""4D Hopf phase diagram — sweep κ* across (ξ, ρ, σ_v) at the canonical
Hopf-exhibiting parameter regime of paper/theory.md §4.2.

This is the headline §4 figure of the paper. Where `bifurcation_scan.py`
produces a 2D (κ, σ_v) slice at fixed (ξ, ρ), this experiment varies all four
axes of the parameter space and renders κ*(ξ, ρ) heatmaps at multiple σ_v
slices.

Structural model. The deterministic Jacobian (paper/theory.md eq. 3) does not
depend on (ξ, ρ) directly — these enter only via the stochastic-Hopf shift Λ
(paper/theory.md §5). To make a usable phase diagram with the noise axes, we
incorporate the leading shear-induced correction (Engel-Lamb-Rasmussen 2024,
``|Λ| ∼ (ρ ξ)^{2/3}``) as a deterministic shift of the eigenvalue envelope:

    a_eff(κ; ξ, ρ) = a(κ) + 0.5 · ξ² · ρ · G_v
    b(κ; σ_v)     = κ G_v − 0.5 · σ_v          (matches bifurcation_scan.py
                                                 ∂_v σ² = σ_v convention)

The shear-correction sign is chosen so positive (ρ, ξ) shifts κ* upward (noise
*stabilises*) and negative (ρ, ξ) shifts κ* downward (noise destabilises) —
the same sign structure documented in theory.md §4.2 for SPX-like ρ < 0.

This is the deterministic projection of the small-noise stochastic Hopf onto
the Jacobian; the full Khasminskii Λ(κ; ξ, ρ) computation lives in
`compute_lambda_correction` and is too expensive (~3 M Euler steps per cell)
for the 2,604-cell grid this experiment scans.

Run: ``python -m reflexive_options.experiments.hopf_phase_scan_4d``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
# See verification_v5_repro.md §3 — without this, every regen drifts the PDF
# hash even though the rendered drawing is bit-identical.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import numpy as np
from numpy.typing import NDArray

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.theory.bifurcation import hopf_scan, jacobian_3d


@dataclass(frozen=True)
class HopfScan4DConfig:
    """Configuration for the 4D Hopf phase scan."""

    # κ scan range (the inner axis of every cell)
    kappa_min: float = 0.0
    kappa_max: float = 2.0
    n_kappa: int = 401

    # ξ (vol-of-vol) axis
    xi_min: float = 0.05
    xi_max: float = 1.5
    n_xi: int = 31

    # ρ (correlation) axis
    rho_min: float = -0.95
    rho_max: float = 0.95
    n_rho: int = 21

    # σ_v slices to compute (Panel A picks median; Panel B sweeps all)
    sigma_v_slices: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)

    # Hopf-exhibiting parameter regime per paper/theory.md §4.2
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 0.5
    G_x: float = 0.5
    G_v: float = -0.5
    G_z: float = -0.5


def _shear_corrected_jacobian(
    kappa: float,
    *,
    cfg: HopfScan4DConfig,
    sigma_v: float,
    xi: float,
    rho: float,
) -> NDArray[np.float64]:
    """Jacobian (paper eq. 3) with the ELR shear correction baked into a(κ).

    See module docstring for the derivation: ξ and ρ enter via a deterministic
    second-order-in-ξ shift of the price-channel envelope a(κ).
    """
    shear_correction = 0.5 * xi * xi * rho * cfg.G_v
    a = kappa * cfg.G_x + shear_correction
    b = kappa * cfg.G_v - 0.5 * sigma_v
    return jacobian_3d(
        kappa=kappa,
        a_kappa=a,
        b_kappa=b,
        G_z=cfg.G_z,
        kappa_v=cfg.kappa_v,
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
    )


def run_4d_scan(
    cfg: HopfScan4DConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Execute the 4D scan; return (xi_grid, rho_grid, sigma_v_slices, kappa_star_4d).

    Output `kappa_star_4d` has shape (n_xi, n_rho, n_sigma_v_slices); cells with
    no Hopf in [kappa_min, kappa_max] are filled with NaN.
    """
    kappa_grid = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.n_kappa).astype(np.float64)
    xi_grid = np.linspace(cfg.xi_min, cfg.xi_max, cfg.n_xi).astype(np.float64)
    rho_grid = np.linspace(cfg.rho_min, cfg.rho_max, cfg.n_rho).astype(np.float64)
    sigma_v_slices = np.asarray(cfg.sigma_v_slices, dtype=np.float64)

    kappa_star_4d = np.full(
        (cfg.n_xi, cfg.n_rho, len(sigma_v_slices)),
        np.nan,
        dtype=np.float64,
    )

    for s_idx, sigma_v in enumerate(sigma_v_slices):
        for x_idx, xi in enumerate(xi_grid):
            for r_idx, rho in enumerate(rho_grid):

                def jac(
                    k: float,
                    sigma_v: float = float(sigma_v),
                    xi: float = float(xi),
                    rho: float = float(rho),
                ) -> NDArray[np.float64]:
                    return _shear_corrected_jacobian(k, cfg=cfg, sigma_v=sigma_v, xi=xi, rho=rho)

                result = hopf_scan(kappa_grid, jac)
                if result.kappa_star is not None:
                    kappa_star_4d[x_idx, r_idx, s_idx] = result.kappa_star
    return xi_grid, rho_grid, sigma_v_slices, kappa_star_4d


def _summary_metrics(
    xi_grid: NDArray[np.float64],
    rho_grid: NDArray[np.float64],
    sigma_v_slices: NDArray[np.float64],
    kappa_star_4d: NDArray[np.float64],
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    finite_mask = np.isfinite(kappa_star_4d)
    n_total = int(kappa_star_4d.size)
    n_finite = int(finite_mask.sum())
    n_no_hopf = n_total - n_finite

    if n_finite == 0:
        median_k = math.nan
        min_k = math.nan
        max_k = math.nan
        argmin_idx: tuple[int, ...] = ()
        argmax_idx: tuple[int, ...] = ()
    else:
        finite_vals = kappa_star_4d[finite_mask]
        median_k = float(np.median(finite_vals))
        min_k = float(finite_vals.min())
        max_k = float(finite_vals.max())
        flat_finite_idx = np.where(finite_mask.ravel())[0]
        local_argmin = int(flat_finite_idx[int(finite_vals.argmin())])
        local_argmax = int(flat_finite_idx[int(finite_vals.argmax())])
        argmin_idx = tuple(int(i) for i in np.unravel_index(local_argmin, kappa_star_4d.shape))
        argmax_idx = tuple(int(i) for i in np.unravel_index(local_argmax, kappa_star_4d.shape))

    def _at(idx: tuple[int, ...]) -> dict[str, float]:
        if not idx:
            return {}
        x_i, r_i, s_i = idx
        return {
            "xi": float(xi_grid[x_i]),
            "rho": float(rho_grid[r_i]),
            "sigma_v": float(sigma_v_slices[s_i]),
            "kappa_star": float(kappa_star_4d[x_i, r_i, s_i]),
        }

    return {
        "elapsed_seconds": float(elapsed_seconds),
        "n_cells_total": n_total,
        "n_cells_with_hopf": n_finite,
        "n_cells_no_hopf": n_no_hopf,
        "fraction_no_hopf": float(n_no_hopf / n_total) if n_total else math.nan,
        "median_kappa_star": median_k,
        "min_kappa_star": min_k,
        "max_kappa_star": max_k,
        "fastest_cell": _at(argmin_idx),
        "slowest_cell": _at(argmax_idx),
        "n_xi": int(xi_grid.size),
        "n_rho": int(rho_grid.size),
        "n_sigma_v_slices": int(sigma_v_slices.size),
    }


def plot_phase_diagram(
    xi_grid: NDArray[np.float64],
    rho_grid: NDArray[np.float64],
    sigma_v_slices: NDArray[np.float64],
    kappa_star_4d: NDArray[np.float64],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Render the publication figure: heatmap (Panel A) + line cuts (Panel B)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    matplotlib.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # Panel A: heatmap at the median σ_v slice.
    median_idx = int(len(sigma_v_slices) // 2)
    panel_data = kappa_star_4d[:, :, median_idx]  # (n_xi, n_rho)
    finite_vals = panel_data[np.isfinite(panel_data)]
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(finite_vals, 2))
        vmax = float(np.nanpercentile(finite_vals, 98))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    norm = Normalize(vmin=vmin, vmax=vmax)

    masked = np.ma.masked_invalid(panel_data)

    extent = (
        float(rho_grid[0]),
        float(rho_grid[-1]),
        float(xi_grid[0]),
        float(xi_grid[-1]),
    )

    im = ax_a.imshow(
        masked,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    if finite_vals.size > 0:
        levels = np.linspace(vmin, vmax, 7)
        rho_mesh, xi_mesh = np.meshgrid(rho_grid, xi_grid)
        contours = ax_a.contour(
            rho_mesh,
            xi_mesh,
            panel_data,
            levels=levels,
            colors="black",
            linewidths=0.6,
            alpha=0.55,
        )
        ax_a.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

    ax_a.set_xlabel(r"$\rho$ (price-vol correlation)")
    ax_a.set_ylabel(r"$\xi$ (vol-of-vol)")
    ax_a.set_title(rf"$\kappa^\star(\xi,\rho)$ at $\sigma_v={sigma_v_slices[median_idx]:.2f}$")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.045, pad=0.025)
    cbar.set_label(r"$\kappa^\star$")

    # Panel B: line cuts κ*(σ_v) at four representative (ξ, ρ) probes.
    probes: tuple[tuple[float, float], ...] = (
        (0.10, -0.70),
        (0.30, -0.70),
        (0.70, -0.70),
        (0.30, +0.70),
    )
    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "D", "^")
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")

    for (xi_q, rho_q), ls, mk, col in zip(probes, line_styles, markers, colors, strict=True):
        x_i = int(np.argmin(np.abs(xi_grid - xi_q)))
        r_i = int(np.argmin(np.abs(rho_grid - rho_q)))
        curve = kappa_star_4d[x_i, r_i, :]
        label = rf"$\xi={xi_grid[x_i]:.2f},\; \rho={rho_grid[r_i]:+.2f}$"
        ax_b.plot(
            sigma_v_slices,
            curve,
            linestyle=ls,
            marker=mk,
            color=col,
            label=label,
            linewidth=1.8,
            markersize=6,
        )

    ax_b.set_xlabel(r"$\sigma_v$")
    ax_b.set_ylabel(r"$\kappa^\star$")
    ax_b.set_title(r"$\kappa^\star(\sigma_v)$ at representative $(\xi,\rho)$")
    ax_b.grid(True, alpha=0.3, linewidth=0.5)
    ax_b.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _write_artifacts(
    run_dir: Path,
    cfg: HopfScan4DConfig,
    xi_grid: NDArray[np.float64],
    rho_grid: NDArray[np.float64],
    sigma_v_slices: NDArray[np.float64],
    kappa_star_4d: NDArray[np.float64],
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Persist npz, metrics.json, and config.json under `run_dir`."""
    save_config(run_dir, cfg)
    metrics = _summary_metrics(
        xi_grid,
        rho_grid,
        sigma_v_slices,
        kappa_star_4d,
        elapsed_seconds=elapsed_seconds,
    )
    save_metrics(run_dir, metrics)
    np.savez(
        run_dir / "phase_4d.npz",
        xi_grid=xi_grid,
        rho_grid=rho_grid,
        sigma_v_slices=sigma_v_slices,
        kappa_star_4d=kappa_star_4d,
    )
    # Echo the config alongside the npz for reviewers who only fetch the .npz.
    (run_dir / "config_snapshot.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-kappa",
        type=int,
        default=None,
        help="κ resolution per cell (default: from HopfScan4DConfig)",
    )
    parser.add_argument("--n-xi", type=int, default=None)
    parser.add_argument("--n-rho", type=int, default=None)
    args = parser.parse_args()

    # Mypy needs Any here because HopfScan4DConfig has heterogeneous fields
    # (int, float, tuple); the argparse-driven overrides are all int-typed.
    cfg_kwargs: dict[str, Any] = {}
    if args.n_kappa is not None:
        cfg_kwargs["n_kappa"] = args.n_kappa
    if args.n_xi is not None:
        cfg_kwargs["n_xi"] = args.n_xi
    if args.n_rho is not None:
        cfg_kwargs["n_rho"] = args.n_rho
    cfg = HopfScan4DConfig(**cfg_kwargs)

    run_dir = make_run_dir("hopf_phase_scan_4d")

    start = time.perf_counter()
    with timed("hopf_phase_scan_4d"):
        xi_grid, rho_grid, sigma_v_slices, kappa_star_4d = run_4d_scan(cfg)
    elapsed = time.perf_counter() - start

    metrics = _write_artifacts(
        run_dir,
        cfg,
        xi_grid,
        rho_grid,
        sigma_v_slices,
        kappa_star_4d,
        elapsed_seconds=elapsed,
    )

    pdf_path = FIGURES_DIR / "hopf_phase_diagram.pdf"
    png_path = FIGURES_DIR / "hopf_phase_diagram.png"
    plot_phase_diagram(
        xi_grid,
        rho_grid,
        sigma_v_slices,
        kappa_star_4d,
        pdf_path=pdf_path,
        png_path=png_path,
    )

    print(f"Wrote results to: {run_dir}")
    print(f"Figure (PDF): {pdf_path}")
    print(f"Figure (PNG): {png_path}")
    print(
        f"median κ* = {metrics['median_kappa_star']:.4f}  "
        f"| no-Hopf cells: {metrics['n_cells_no_hopf']}/{metrics['n_cells_total']} "
        f"({metrics['fraction_no_hopf']:.1%})"
    )


if __name__ == "__main__":
    main()
