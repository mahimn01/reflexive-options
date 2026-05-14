"""Empirical power-law scan of Λ vs (ρ ξ) at the §4.2 canonical regime.

The Engel–Lamb–Rasmussen (2024) shear-induced expansion predicts, for an
additive-noise Hopf system perturbed by a small shear b, that the leading
correction to the top Lyapunov exponent scales as |Λ| ~ b^(2/3). Adapted
to the Heston multiplicative-noise setup of paper §1, the natural shear
proxy is the product ρ · ξ (correlation × vol-of-vol).

V1 audit (`~/Documents/reflexivity-research/verification_v1_math.md`)
found the implemented Λ across 5 (ξ, ρ) probes does *not* satisfy
Λ / (ρ ξ)^(2/3) = const — the ratio drifted ~2.6× across the probes. The
v0.3.1 paper softened §5 to "we do not claim to have validated the
asymptotic scaling." This script replaces that softness with an actual
power-law fit and a quantitative comparison to the predicted exponent
B = 2/3.

Method:
    1. Hold the §4.2 dimensionless skeleton fixed (memory channel:
       α=0.5, β=1, γ=0.5, κ_v=2, θ_v=0.04). Vary (ξ, ρ) on a 6×6 grid.
    2. At each cell, compute Λ via `compute_lambda_correction` (Khasminskii
       sphere process / Benettin renormalisation of the linearised SDE)
       at a high path budget for low MC noise.
    3. Fit  log|Λ| = log|A| + B · log|ρ ξ|  via OLS on the union of
       valid cells.
    4. Bootstrap (B_bootstrap resamples, fixed seed) the (A, B) sample
       distribution and report 95% CI.
    5. Compare B vs predicted B = 2/3 (one-sample test against a fixed
       value: |B - 2/3| / SE(B); flag stat-significance at p < 0.01).
    6. Save (ξ, ρ, Λ) table as CSV; render log|Λ| vs log|ρ ξ| with fit
       line and 95% CI band as `paper/figures/lambda_scaling_loglog.pdf`.

Invocation::

    python -m reflexive_options.experiments.lambda_scaling
    python -m reflexive_options.experiments.lambda_scaling --quick

Outputs:
    runs/lambda_scaling/<timestamp>/{config.json, metrics.json,
                                    lambda_grid.csv}
    paper/figures/lambda_scaling_loglog.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import asdict, dataclass, field

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.theory.bifurcation import compute_lambda_correction
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SurfaceGrid,
)


@dataclass(frozen=True)
class LambdaScalingConfig:
    """Configuration for the (ξ, ρ) Λ-scaling scan."""

    # §4.2 dimensionless deterministic-skeleton parameters
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 0.5
    coupling_at_kappa_star: float = 0.8964

    # Scan grid (ρ, ξ)
    xi_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    rho_grid: tuple[float, ...] = (-0.95, -0.7, -0.5, -0.3, 0.3, 0.7)

    # Khasminskii estimator — high path budget for low MC noise
    epsilon_low: float = 0.05
    epsilon_high: float = 0.20
    n_paths: int = 10_000
    n_steps: int = 10_000
    dt: float = 1.0e-2
    renorm_every: int = 50
    seed: int = 20260514  # locked

    # Bootstrap CI for (A, B)
    n_bootstrap: int = 2_000
    bootstrap_seed: int = 20260514_2

    # Predicted exponent (Engel–Lamb–Rasmussen 2024 leading order, additive)
    predicted_exponent: float = 2.0 / 3.0


@dataclass(frozen=True)
class CellResult:
    """One (ξ, ρ) cell of the Λ scan."""

    xi: float
    rho: float
    rho_xi: float
    lambda_value: float
    abs_lambda: float
    log_abs_rho_xi: float
    log_abs_lambda: float


@dataclass
class FitResult:
    """OLS + bootstrap fit of log|Λ| = log|A| + B · log|ρ ξ|."""

    log_A: float
    B: float
    log_A_ci_low: float
    log_A_ci_high: float
    B_ci_low: float
    B_ci_high: float
    n_cells_fit: int
    residual_std: float
    distinguishable_from_two_thirds: bool
    z_score_vs_two_thirds: float
    bootstrap_log_A: NDArray[np.float64] = field(repr=False)
    bootstrap_B: NDArray[np.float64] = field(repr=False)


def _build_simulator(cfg: LambdaScalingConfig, *, xi: float, rho: float) -> ReflexiveSimulator:
    """Build the §4.2 simulator with the given (ξ, ρ).

    OI grid is held trivially zero so the price-channel feedback vanishes
    from the Jacobian and we measure the bare Heston-with-memory linearised
    Λ — the same setup as `lambda_correction_canonical.py`.
    """
    grid = SurfaceGrid(
        log_moneyness=np.array([-0.05, 0.0, 0.05], dtype=np.float64),
        maturities=np.array([30 / 365.25, 90 / 365.25], dtype=np.float64),
    )
    contracts = np.zeros(grid.shape, dtype=np.float64)
    oi = OpenInterestGrid(grid=grid, contracts_open=contracts)
    aggregator = GammaAggregator(oi_grid=oi, risk_free_rate=0.0)
    params = ReflexiveParams(
        base=HestonParams(
            kappa=cfg.kappa_v,
            theta=cfg.theta_v,
            xi=xi,
            rho=rho,
            v0=cfg.theta_v,
        ),
        coupling=cfg.coupling_at_kappa_star,
        drift=0.0,
        memory_decay=cfg.alpha,
        memory_intake=cfg.beta,
        leverage=cfg.gamma,
    )
    return ReflexiveSimulator(params=params, gamma_aggregator=aggregator, initial_spot=100.0)


def _scan_cells(cfg: LambdaScalingConfig) -> list[CellResult]:
    """Compute Λ at each (ξ, ρ) cell. Returns the populated cell list."""
    cells: list[CellResult] = []
    n_total = len(cfg.xi_grid) * len(cfg.rho_grid)
    cell_idx = 0
    # Deterministic per-cell seed offset so a re-run with the same cfg.seed
    # produces identical Λ values cell-by-cell (no global RNG state coupling).
    for i_xi, xi in enumerate(cfg.xi_grid):
        for i_rho, rho in enumerate(cfg.rho_grid):
            cell_idx += 1
            cell_seed = cfg.seed + 1000 * i_xi + i_rho
            sim = _build_simulator(cfg, xi=xi, rho=rho)
            with timed(f"Λ cell {cell_idx}/{n_total} (ξ={xi:.2f}, ρ={rho:+.2f})"):
                lam = compute_lambda_correction(
                    sim,
                    kappa=cfg.coupling_at_kappa_star,
                    epsilon_low=cfg.epsilon_low,
                    epsilon_high=cfg.epsilon_high,
                    n_paths=cfg.n_paths,
                    n_steps=cfg.n_steps,
                    dt=cfg.dt,
                    renorm_every=cfg.renorm_every,
                    seed=cell_seed,
                )
            rho_xi = rho * xi
            abs_lambda = abs(lam)
            cells.append(
                CellResult(
                    xi=float(xi),
                    rho=float(rho),
                    rho_xi=float(rho_xi),
                    lambda_value=float(lam),
                    abs_lambda=float(abs_lambda),
                    log_abs_rho_xi=float(math.log(abs(rho_xi))) if rho_xi != 0.0 else float("nan"),
                    log_abs_lambda=float(math.log(abs_lambda))
                    if abs_lambda > 0.0
                    else float("nan"),
                )
            )
            print(f"    Λ = {lam:+.4e}  |Λ| = {abs_lambda:.4e}  ρξ = {rho_xi:+.4e}")
    return cells


def _ols_fit(
    log_x: NDArray[np.float64],
    log_y: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Plain OLS: y = a + b·x. Returns (a, b, residual_std)."""
    n = len(log_x)
    x_mean = float(np.mean(log_x))
    y_mean = float(np.mean(log_y))
    x_centered = log_x - x_mean
    y_centered = log_y - y_mean
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 0.0:
        raise ValueError("OLS singular: zero variance in log_x")
    b = float(np.sum(x_centered * y_centered) / denom)
    a = y_mean - b * x_mean
    residuals = log_y - (a + b * log_x)
    if n > 2:
        residual_std = float(np.sqrt(np.sum(residuals * residuals) / (n - 2)))
    else:
        residual_std = float("nan")
    return a, b, residual_std


def _fit_power_law(
    cells: list[CellResult],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    predicted_exponent: float,
) -> FitResult:
    """Fit log|Λ| = log|A| + B · log|ρξ|; bootstrap (A, B) for 95% CI."""
    valid = [c for c in cells if not (math.isnan(c.log_abs_lambda) or math.isnan(c.log_abs_rho_xi))]
    if len(valid) < 4:
        raise ValueError(f"need ≥ 4 valid cells for fit, got {len(valid)}")

    log_x = np.array([c.log_abs_rho_xi for c in valid], dtype=np.float64)
    log_y = np.array([c.log_abs_lambda for c in valid], dtype=np.float64)
    a, b, residual_std = _ols_fit(log_x, log_y)

    # Non-parametric bootstrap on cell tuples
    rng = np.random.default_rng(bootstrap_seed)
    n = len(valid)
    boot_a = np.empty(n_bootstrap, dtype=np.float64)
    boot_b = np.empty(n_bootstrap, dtype=np.float64)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            ak, bk, _ = _ols_fit(log_x[idx], log_y[idx])
        except ValueError:
            # bootstrap sample with no x-variance — re-draw
            ak, bk = a, b
        boot_a[k] = ak
        boot_b[k] = bk
    a_low, a_high = float(np.quantile(boot_a, 0.025)), float(np.quantile(boot_a, 0.975))
    b_low, b_high = float(np.quantile(boot_b, 0.025)), float(np.quantile(boot_b, 0.975))

    # Compare B to predicted exponent (one-sample): z-like score using bootstrap SE
    b_se = float(np.std(boot_b, ddof=1)) if n_bootstrap > 1 else float("nan")
    z = abs(b - predicted_exponent) / b_se if b_se > 0.0 else float("nan")
    # Distinguishable at α≈0.01 if |z| > 2.576 (normal approx) — also check the CI.
    distinguishable = bool(b_low > predicted_exponent or b_high < predicted_exponent)

    return FitResult(
        log_A=float(a),
        B=float(b),
        log_A_ci_low=a_low,
        log_A_ci_high=a_high,
        B_ci_low=b_low,
        B_ci_high=b_high,
        n_cells_fit=n,
        residual_std=residual_std,
        distinguishable_from_two_thirds=distinguishable,
        z_score_vs_two_thirds=float(z),
        bootstrap_log_A=boot_a,
        bootstrap_B=boot_b,
    )


def _save_csv(cells: list[CellResult], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["xi", "rho", "rho_xi", "lambda", "abs_lambda", "log_abs_rho_xi", "log_abs_lambda"]
        )
        for c in cells:
            writer.writerow(
                [
                    f"{c.xi:.6f}",
                    f"{c.rho:.6f}",
                    f"{c.rho_xi:.6e}",
                    f"{c.lambda_value:.6e}",
                    f"{c.abs_lambda:.6e}",
                    f"{c.log_abs_rho_xi:.6f}",
                    f"{c.log_abs_lambda:.6f}",
                ]
            )


def _render_figure(
    cells: list[CellResult],
    fit: FitResult,
    predicted_exponent: float,
    out_path: str,
) -> None:
    valid = [c for c in cells if not math.isnan(c.log_abs_lambda)]
    log_x = np.array([c.log_abs_rho_xi for c in valid], dtype=np.float64)
    log_y = np.array([c.log_abs_lambda for c in valid], dtype=np.float64)
    rho_signs = np.array([1.0 if c.rho > 0 else -1.0 for c in valid], dtype=np.float64)

    x_dense = np.linspace(log_x.min() - 0.2, log_x.max() + 0.2, 200)

    # Bootstrap predictions: y(x) = log_A + B · x for each (boot_log_A, boot_B)
    y_boot = fit.bootstrap_log_A[:, None] + fit.bootstrap_B[:, None] * x_dense[None, :]
    y_lo = np.quantile(y_boot, 0.025, axis=0)
    y_hi = np.quantile(y_boot, 0.975, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pos_mask = rho_signs > 0
    neg_mask = ~pos_mask
    ax.scatter(
        log_x[neg_mask],
        log_y[neg_mask],
        color="C3",
        s=40,
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        label=r"$\rho < 0$",
    )
    ax.scatter(
        log_x[pos_mask],
        log_y[pos_mask],
        color="C0",
        s=40,
        marker="s",
        edgecolor="black",
        linewidth=0.5,
        label=r"$\rho > 0$",
    )

    y_fit = fit.log_A + fit.B * x_dense
    ax.plot(
        x_dense,
        y_fit,
        color="black",
        linewidth=1.5,
        label=rf"OLS fit: $\log|\Lambda| = {fit.log_A:.3f} + {fit.B:.3f}\,\log|\rho\xi|$",
    )
    ax.fill_between(x_dense, y_lo, y_hi, color="grey", alpha=0.22, label="95% bootstrap CI")

    # Reference line at the predicted exponent (rooted at the fit intercept so
    # the slopes can be visually compared at the same baseline).
    y_ref = fit.log_A + predicted_exponent * x_dense
    ax.plot(
        x_dense,
        y_ref,
        color="C2",
        linestyle="--",
        linewidth=1.3,
        label=r"ELR prediction $B = 2/3$ (slope only; intercept matched)",
    )

    ax.set_xlabel(r"$\log|\rho\,\xi|$")
    ax.set_ylabel(r"$\log|\Lambda|$")
    distinguish_str = (
        "distinguishable from 2/3 at 95%"
        if fit.distinguishable_from_two_thirds
        else "consistent with 2/3 at 95%"
    )
    ax.set_title(
        rf"Empirical $\Lambda \sim |\rho\xi|^B$ scan over §4.2 canonical regime"
        "\n"
        rf"$B = {fit.B:.3f}$ (95% CI $[{fit.B_ci_low:.3f}, {fit.B_ci_high:.3f}]$); "
        rf"{distinguish_str};  $z = {fit.z_score_vs_two_thirds:.2f}$"
    )
    ax.legend(loc="best", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"OK: lambda-scaling figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduce path budget and grid for CI-fast smoke run",
    )
    args = parser.parse_args()

    if args.quick:
        cfg = LambdaScalingConfig(
            xi_grid=(0.1, 0.3, 0.7),
            rho_grid=(-0.7, -0.3, 0.3),
            n_paths=400,
            n_steps=2_000,
            n_bootstrap=200,
        )
    else:
        cfg = LambdaScalingConfig()

    run_dir = make_run_dir("lambda_scaling")
    save_config(run_dir, cfg)

    cells = _scan_cells(cfg)
    fit = _fit_power_law(
        cells,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
        predicted_exponent=cfg.predicted_exponent,
    )

    print()
    print("=" * 72)
    print("Empirical power-law fit Λ ~ A · |ρ ξ|^B (OLS + bootstrap CI)")
    print("=" * 72)
    print(f"  N cells fit:            {fit.n_cells_fit}")
    print(
        f"  log|A| (intercept):     {fit.log_A:+.4f}  "
        f"95% CI [{fit.log_A_ci_low:+.4f}, {fit.log_A_ci_high:+.4f}]"
    )
    print(f"  |A| (intercept):        {math.exp(fit.log_A):.4e}")
    print(
        f"  B (exponent):           {fit.B:+.4f}  "
        f"95% CI [{fit.B_ci_low:+.4f}, {fit.B_ci_high:+.4f}]"
    )
    print(f"  Residual std (log y):   {fit.residual_std:.4f}")
    print(f"  Predicted ELR exponent: {cfg.predicted_exponent:.4f}")
    print(f"  |B - 2/3| / SE(B):      {fit.z_score_vs_two_thirds:.3f}")
    print(f"  CI excludes 2/3?        {fit.distinguishable_from_two_thirds}")

    csv_path = run_dir / "lambda_grid.csv"
    _save_csv(cells, str(csv_path))

    metrics = {
        "n_cells_total": len(cells),
        "n_cells_fit": fit.n_cells_fit,
        "fit_log_A": fit.log_A,
        "fit_B": fit.B,
        "fit_log_A_ci": [fit.log_A_ci_low, fit.log_A_ci_high],
        "fit_B_ci": [fit.B_ci_low, fit.B_ci_high],
        "fit_residual_std": fit.residual_std,
        "predicted_exponent": cfg.predicted_exponent,
        "z_score_vs_two_thirds": fit.z_score_vs_two_thirds,
        "distinguishable_from_two_thirds": fit.distinguishable_from_two_thirds,
        "cells": [
            {
                "xi": c.xi,
                "rho": c.rho,
                "rho_xi": c.rho_xi,
                "lambda": c.lambda_value,
                "abs_lambda": c.abs_lambda,
            }
            for c in cells
        ],
        "config": asdict(cfg),
    }
    save_metrics(run_dir, metrics)
    print(f"Wrote results to: {run_dir}")
    print(f"CSV table: {csv_path}")

    out_path = FIGURES_DIR / "lambda_scaling_loglog.pdf"
    _render_figure(cells, fit, cfg.predicted_exponent, str(out_path))


if __name__ == "__main__":
    main()
