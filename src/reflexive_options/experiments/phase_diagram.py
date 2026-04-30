"""(κ, σ_v) phase diagram — empirical companion to the bifurcation scan.

Where bifurcation_scan.py works on the LINEARIZED system (Jacobian
eigenvalues), this experiment runs full nonlinear simulations and classifies
each (κ, σ_v) cell into one of {calm, vol-cluster, limit-cycle, blow-up}
based on observed trajectory statistics.

Run: python -m reflexive_options.experiments.phase_diagram
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from reflexive_options.experiments._common import (
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)


@dataclass(frozen=True)
class PhaseDiagramConfig:
    kappa_min: float = 0.0
    kappa_max: float = 1.0
    n_kappa: int = 21
    sigma_v_min: float = 0.05
    sigma_v_max: float = 1.5
    n_sigma_v: int = 16
    n_paths_per_cell: int = 1_000
    n_steps_per_path: int = 252
    seed: int = 42


def classify_regime(
    spots: np.ndarray,
    variances: np.ndarray,
    *,
    initial_spot: float,
) -> str:
    """Classify a batch of paths into one of {blow_up, limit_cycle, vol_cluster, calm}."""
    if (
        not np.all(np.isfinite(spots))
        or not np.all(np.isfinite(variances))
        or np.any(spots > 100 * initial_spot)
    ):
        return "blow_up"

    abs_log_returns = np.abs(np.diff(np.log(spots), axis=1))
    if abs_log_returns.shape[1] >= 2:
        flat = abs_log_returns.reshape(-1)
        ac1 = float(np.corrcoef(flat[:-1], flat[1:])[0, 1]) if len(flat) > 200 else 0.0
    else:
        ac1 = 0.0

    var_of_var = float(variances.var(axis=1).mean())

    if var_of_var > 0.5:
        return "limit_cycle"
    if ac1 > 0.2:
        return "vol_cluster"
    return "calm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-kappa", type=int, default=21)
    parser.add_argument("--n-sigma-v", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PhaseDiagramConfig(
        n_kappa=args.n_kappa,
        n_sigma_v=args.n_sigma_v,
        seed=args.seed,
    )
    run_dir = make_run_dir("phase_diagram", seed=cfg.seed)
    save_config(run_dir, cfg)

    kappa_grid = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.n_kappa)
    sigma_v_grid = np.linspace(cfg.sigma_v_min, cfg.sigma_v_max, cfg.n_sigma_v)

    regime_grid = np.empty((cfg.n_kappa, cfg.n_sigma_v), dtype=object)
    blowup_fraction = np.zeros((cfg.n_kappa, cfg.n_sigma_v))

    with timed("phase_grid"):
        for i, _k in enumerate(kappa_grid):
            for j, _sv in enumerate(sigma_v_grid):
                # TODO(post-implementation, blocked on simulator task #13):
                #   1. Build ReflexiveSimulator(coupling=_k, base.xi=_sv, ...)
                #   2. Run n_paths_per_cell × n_steps_per_path Monte Carlo
                #   3. Apply detect_blowup → blowup_fraction[i, j]
                #   4. Classify surviving paths → regime_grid[i, j]
                regime_grid[i, j] = "calm"  # stub
                blowup_fraction[i, j] = 0.0

    save_metrics(
        run_dir,
        {
            "kappa_grid": kappa_grid.tolist(),
            "sigma_v_grid": sigma_v_grid.tolist(),
            "regime_grid": regime_grid.tolist(),
            "blowup_fraction": blowup_fraction.tolist(),
        },
    )
    np.savez(
        run_dir / "phase_diagram.npz",
        kappa_grid=kappa_grid,
        sigma_v_grid=sigma_v_grid,
        regime_grid=regime_grid,
        blowup_fraction=blowup_fraction,
    )
    print(f"Wrote results to: {run_dir}")


if __name__ == "__main__":
    main()
