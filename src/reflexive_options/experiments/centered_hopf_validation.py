"""Reproduce the revised paper's fixed-equilibrium Hopf example.

The output uses the actual centered Gaussian dealer-book functional.  No
hand-selected quadratic or cubic coefficients are inserted into the drift.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
)
from reflexive_options.theory.centered_model import (
    GaussianBookParams,
    canonical_centered_configuration,
    centered_drift_gaussian_book,
    centered_jacobian,
    gaussian_book_hopf_point,
    normalized_gaussian_book_partials,
)


@dataclass(frozen=True)
class CenteredHopfValidationConfig:
    """Numerical settings; time is measured in years."""

    coupling_multiplier: float = 1.02
    total_time: float = 50.0
    transient_time: float = 40.0
    samples: int = 50_001
    initial_x: float = 1.0e-3
    initial_variance_offset: float = 0.0
    initial_chi: float = 0.0
    rtol: float = 1.0e-9
    atol: float = 1.0e-11


def simulate(
    cfg: CenteredHopfValidationConfig,
) -> tuple[dict[str, float], dict[str, NDArray[np.float64]]]:
    """Integrate the actual nonlinear deterministic skeleton."""

    model, book = canonical_centered_configuration()
    point = gaussian_book_hopf_point(model, book)
    kappa = cfg.coupling_multiplier * point.kappa

    def rhs(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return centered_drift_gaussian_book(state, kappa=kappa, model=model, book=book)

    initial = np.array(
        [cfg.initial_x, model.theta_v + cfg.initial_variance_offset, cfg.initial_chi],
        dtype=np.float64,
    )
    times = np.linspace(0.0, cfg.total_time, cfg.samples)
    solution = solve_ivp(
        rhs,
        (0.0, cfg.total_time),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=0.005,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    kept = solution.t >= cfg.transient_time
    trajectory = {
        "t": solution.t[kept],
        "x": solution.y[0, kept],
        "v": solution.y[1, kept],
        "chi": solution.y[2, kept],
    }
    metrics = {
        "kappa_star": point.kappa,
        "omega_star": point.omega,
        "period_years": point.period_years,
        "transversality": point.transversality,
        "first_lyapunov": point.first_lyapunov,
        "simulation_kappa": kappa,
        "x_min": float(np.min(trajectory["x"])),
        "x_max": float(np.max(trajectory["x"])),
        "variance_min": float(np.min(trajectory["v"])),
        "variance_max": float(np.max(trajectory["v"])),
        "chi_min": float(np.min(trajectory["chi"])),
        "chi_max": float(np.max(trajectory["chi"])),
    }
    if metrics["variance_min"] <= 0.0:
        raise RuntimeError("illustrative orbit left the non-negative variance state space")
    return metrics, trajectory


def _phase_map() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Map valid first Hopf thresholds; NaN means no valid local Hopf root."""

    model, canonical_book = canonical_centered_configuration()
    means = np.linspace(-0.10, 0.16, 53)
    sigmas = np.linspace(0.06, 0.34, 57)
    thresholds = np.full((sigmas.size, means.size), np.nan)
    for row, sigma in enumerate(sigmas):
        for column, mean in enumerate(means):
            book = GaussianBookParams(
                mean_moneyness=float(mean),
                sigma_moneyness=float(sigma),
                effective_maturity=canonical_book.effective_maturity,
                dealer_sign=canonical_book.dealer_sign,
            )
            try:
                point = gaussian_book_hopf_point(model, book)
            except ValueError:
                continue
            thresholds[row, column] = point.kappa
    return means, sigmas, thresholds


def render(
    metrics: dict[str, float],
    trajectory: dict[str, NDArray[np.float64]],
    output: Path,
) -> None:
    """Render eigenvalue crossing, nonlinear orbit, and local-Hopf map."""

    model, book = canonical_centered_configuration()
    partials = normalized_gaussian_book_partials(model, book)
    kappa_star = metrics["kappa_star"]
    kappas = np.linspace(0.75 * kappa_star, 1.25 * kappa_star, 251)
    eigenvalues = np.array(
        [
            np.linalg.eigvals(
                centered_jacobian(
                    kappa=float(kappa),
                    model=model,
                    G_x=partials["G_a"],
                    G_v=partials["G_v"],
                )
            )
            for kappa in kappas
        ]
    )
    ordered_real = np.sort(eigenvalues.real, axis=1)
    means, sigmas, thresholds = _phase_map()

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    for column in range(3):
        ax.plot(kappas, ordered_real[:, column], linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axvline(kappa_star, color="C3", linestyle="--", linewidth=1.0)
    ax.set(xlabel=r"coupling $\kappa$ (yr$^{-1}$)", ylabel=r"$\mathrm{Re}\,\lambda$ (yr$^{-1}$)")
    ax.set_title("(a) Transversal eigenvalue crossing")

    ax = axes[0, 1]
    final_year = trajectory["t"] >= trajectory["t"][-1] - 1.0
    ax.plot(
        trajectory["t"][final_year] - trajectory["t"][final_year][0],
        trajectory["v"][final_year],
        color="C1",
        linewidth=1.0,
    )
    ax.axhline(model.theta_v, color="black", linewidth=0.7, linestyle=":")
    ax.set(xlabel="time in final year", ylabel="annualized variance $v$")
    ax.set_title(r"(b) Nonlinear orbit at $1.02\,\kappa^\star$")

    ax = axes[1, 0]
    ax.plot(trajectory["x"], trajectory["v"], color="C0", linewidth=0.8)
    ax.scatter([0.0], [model.theta_v], color="black", s=18, zorder=3)
    ax.set(xlabel=r"detrended log-price $X$", ylabel="annualized variance $v$")
    ax.set_title("(c) Attracting cycle; equilibrium is the dot")

    ax = axes[1, 1]
    masked = np.ma.masked_invalid(thresholds)
    image = ax.pcolormesh(means, sigmas, masked, shading="auto", cmap="viridis_r")
    ax.scatter(
        [book.mean_moneyness],
        [book.sigma_moneyness],
        marker="*",
        color="white",
        edgecolor="black",
        s=100,
    )
    ax.set(xlabel=r"book mean $\mu_q$", ylabel=r"book dispersion $\sigma_q$")
    ax.set_title("(d) First valid local Hopf threshold")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(r"$\kappa^\star$ (yr$^{-1}$)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short numerical smoke run")
    arguments = parser.parse_args()
    cfg = CenteredHopfValidationConfig()
    if arguments.quick:
        cfg = CenteredHopfValidationConfig(total_time=2.0, transient_time=1.0, samples=2_001)
    metrics, trajectory = simulate(cfg)
    output = FIGURES_DIR / "centered_hopf_validation.pdf"
    render(metrics, trajectory, output)
    run_dir = make_run_dir("centered_hopf_validation")
    save_config(run_dir, asdict(cfg))
    save_metrics(run_dir, metrics)
    print(f"wrote {output}")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
