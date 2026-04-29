"""Numerical Hopf bifurcation scan — locate κ* via Routh-Hurwitz.

Implements Algorithm 1 from hopf_bifurcation_brief.md §6. Produces a phase
diagram in (κ, σ_v) space showing where the linearised reflexive system
crosses from stable equilibrium to limit cycle.

Run: python -m reflexive_options.experiments.bifurcation_scan
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial

import numpy as np

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.theory.bifurcation import hopf_scan, jacobian_3d


@dataclass(frozen=True)
class BifurcationConfig:
    """Configuration for the Hopf scan."""

    kappa_min: float = 0.0
    kappa_max: float = 1.0
    n_kappa: int = 401
    sigma_v_min: float = 0.05
    sigma_v_max: float = 1.5
    n_sigma_v: int = 31

    # Default Heston-side parameters (calibrated from a representative regime)
    kappa_v: float = 2.0
    theta_v: float = 0.04

    # Memory-channel parameters (paper/theory.md §1)
    alpha: float = 252.0  # ~1-day decay
    beta: float = 1.0
    gamma: float = 0.5

    # Linearization-coefficient priors (would come from G(S, z, v) at equilibrium)
    G_x: float = -0.001  # negative: dealers lean against price moves (long-gamma regime)
    G_v: float = -0.0005
    G_z: float = 0.001
    sigma2_x: float = 0.0  # ∂_x σ²(x*, v*); zero for vanilla Heston backbone
    sigma2_v: float = 1.0  # ∂_v σ² = 1 since σ² = v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-kappa", type=int, default=401)
    parser.add_argument("--n-sigma-v", type=int, default=31)
    args = parser.parse_args()

    cfg = BifurcationConfig(n_kappa=args.n_kappa, n_sigma_v=args.n_sigma_v)
    run_dir = make_run_dir("bifurcation_scan")
    save_config(run_dir, cfg)

    kappa_grid = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.n_kappa)
    sigma_v_grid = np.linspace(cfg.sigma_v_min, cfg.sigma_v_max, cfg.n_sigma_v)

    kappa_star_vs_sigma_v: list[float | None] = []

    with timed("hopf_scan_grid"):
        for sv in sigma_v_grid:
            jac_at = partial(
                _jacobian_for_kappa_at_sigma,
                cfg=cfg,
                sigma_v=sv,
            )
            result = hopf_scan(kappa_grid, jac_at)
            kappa_star_vs_sigma_v.append(result.kappa_star)

    metrics = {
        "kappa_min": cfg.kappa_min,
        "kappa_max": cfg.kappa_max,
        "n_kappa": cfg.n_kappa,
        "sigma_v_grid": sigma_v_grid.tolist(),
        "kappa_star_curve": [None if k is None else float(k) for k in kappa_star_vs_sigma_v],
        "n_bifurcations_found": int(sum(k is not None for k in kappa_star_vs_sigma_v)),
    }
    save_metrics(run_dir, metrics)

    np.savez(
        run_dir / "phase_diagram.npz",
        kappa_grid=kappa_grid,
        sigma_v_grid=sigma_v_grid,
        kappa_star_curve=np.array(
            [np.nan if k is None else k for k in kappa_star_vs_sigma_v]
        ),
    )

    print(f"Wrote results to: {run_dir}")
    print(f"Phase diagram: {FIGURES_DIR}/bifurcation_phase_diagram.pdf (TODO: plot)")


def _jacobian_for_kappa_at_sigma(
    kappa: float,
    *,
    cfg: BifurcationConfig,
    sigma_v: float,
) -> np.ndarray:
    a = kappa * cfg.G_x - 0.5 * cfg.sigma2_x
    b = kappa * cfg.G_v - 0.5 * cfg.sigma2_v * sigma_v
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


if __name__ == "__main__":
    main()
