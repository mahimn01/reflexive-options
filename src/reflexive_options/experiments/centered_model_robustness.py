"""Robustness and numerical-convergence audit for the centered Hopf model.

This experiment uses only the corrected fixed-equilibrium model.  It checks
the local square-root amplitude law on the actual nonlinear Gaussian book,
the period limit, solver and initial-condition convergence, one-at-a-time
parameter sensitivity, first-Lyapunov signs over the book-shape plane, and a
small set of gross-normalized signed Gaussian mixtures.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
)
from reflexive_options.theory.bifurcation import (
    build_bilinear_trilinear_tensors,
    compute_lyapunov_coefficient,
)
from reflexive_options.theory.centered_model import (
    GaussianBookComponent,
    GaussianBookParams,
    canonical_centered_configuration,
    centered_drift_gaussian_book,
    centered_jacobian,
    gaussian_book_hopf_point,
    gaussian_mixture_hopf_point,
    normalized_gaussian_book_partials,
    normalized_gaussian_mixture_partials,
)


@dataclass(frozen=True)
class RobustnessConfig:
    """Deterministic integration and grid settings; time is in years."""

    relative_distances: tuple[float, ...] = (
        0.000625,
        0.00125,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.04,
        0.06,
    )
    final_window: float = 5.0
    final_samples: int = 20_001
    decay_time_constants: float = 12.0
    minimum_total_time: float = 80.0
    max_step: float = 0.005
    rtol: float = 1.0e-9
    atol: float = 1.0e-11
    oat_points: int = 41
    book_grid_points: int = 45


@dataclass(frozen=True)
class CycleSummary:
    relative_distance: float
    total_time: float
    x_amplitude: float
    variance_amplitude: float
    period: float
    variance_min: float
    variance_max: float


def _cycle_summary(
    relative_distance: float,
    *,
    cfg: RobustnessConfig,
    initial: NDArray[np.float64] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> CycleSummary:
    model, book = canonical_centered_configuration()
    point = gaussian_book_hopf_point(model, book)
    if relative_distance <= 0.0:
        raise ValueError("cycle validation requires a coupling above the Hopf point")
    linear_growth = point.transversality * point.kappa * relative_distance
    if linear_growth <= 0.0:
        raise ValueError("canonical crossing must destabilize above threshold")
    total_time = max(cfg.minimum_total_time, cfg.decay_time_constants / linear_growth)
    evaluation_times = np.linspace(
        total_time - cfg.final_window,
        total_time,
        cfg.final_samples,
    )
    if initial is None:
        initial = np.array([0.01, model.theta_v, 0.0], dtype=np.float64)

    def rhs(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return centered_drift_gaussian_book(
            state,
            kappa=(1.0 + relative_distance) * point.kappa,
            model=model,
            book=book,
        )

    solution = solve_ivp(
        rhs,
        (0.0, total_time),
        np.asarray(initial, dtype=np.float64),
        t_eval=evaluation_times,
        method="DOP853",
        rtol=cfg.rtol if rtol is None else rtol,
        atol=cfg.atol if atol is None else atol,
        max_step=cfg.max_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    x = solution.y[0]
    variance = solution.y[1]
    if float(np.min(variance)) <= 0.0:
        raise RuntimeError("robustness trajectory left the positive-variance state space")
    peaks, _ = find_peaks(x, prominence=max(1.0e-10, 0.05 * float(np.ptp(x))))
    if peaks.size < 10:
        raise RuntimeError(f"too few peaks to estimate a period: {peaks.size}")
    periods = np.diff(solution.t[peaks])
    return CycleSummary(
        relative_distance=relative_distance,
        total_time=float(total_time),
        x_amplitude=0.5 * float(np.ptp(x)),
        variance_amplitude=0.5 * float(np.ptp(variance)),
        period=float(np.mean(periods)),
        variance_min=float(np.min(variance)),
        variance_max=float(np.max(variance)),
    )


def _oat_sensitivity(
    cfg: RobustnessConfig,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    model, book = canonical_centered_configuration()
    base = gaussian_book_hopf_point(model, book)
    multipliers = np.linspace(0.6, 1.4, cfg.oat_points)
    curves: dict[str, NDArray[np.float64]] = {}
    model_fields = ("delta", "kappa_v", "theta_v", "alpha", "beta", "gamma")
    for name in model_fields:
        values = np.full(multipliers.shape, np.nan)
        base_value = float(getattr(model, name))
        for index, multiplier in enumerate(multipliers):
            candidate = replace(model, **{name: base_value * float(multiplier)})
            try:
                values[index] = gaussian_book_hopf_point(candidate, book).kappa / base.kappa
            except ValueError:
                continue
        curves[name] = values
    for name in ("sigma_moneyness", "effective_maturity"):
        values = np.full(multipliers.shape, np.nan)
        base_value = float(getattr(book, name))
        for index, multiplier in enumerate(multipliers):
            perturbed = base_value * float(multiplier)
            candidate_book = (
                replace(book, sigma_moneyness=perturbed)
                if name == "sigma_moneyness"
                else replace(book, effective_maturity=perturbed)
            )
            try:
                values[index] = gaussian_book_hopf_point(model, candidate_book).kappa / base.kappa
            except ValueError:
                continue
        curves[name] = values
    return multipliers, curves


def _lyapunov_classification_map(
    cfg: RobustnessConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, int]]:
    model, book = canonical_centered_configuration()
    means = np.linspace(-0.10, 0.16, cfg.book_grid_points)
    sigmas = np.linspace(0.06, 0.34, cfg.book_grid_points)
    classification = np.full((sigmas.size, means.size), np.nan)
    counts = {"no_valid_hopf": 0, "supercritical": 0, "subcritical": 0}
    for row, sigma in enumerate(sigmas):
        for column, mean in enumerate(means):
            candidate = replace(
                book,
                mean_moneyness=float(mean),
                sigma_moneyness=float(sigma),
            )
            try:
                point = gaussian_book_hopf_point(model, candidate)
            except ValueError:
                counts["no_valid_hopf"] += 1
                continue
            if point.first_lyapunov < 0.0:
                classification[row, column] = -1.0
                counts["supercritical"] += 1
            else:
                classification[row, column] = 1.0
                counts["subcritical"] += 1
    return means, sigmas, classification, counts


def _mixture_scenarios() -> dict[str, tuple[GaussianBookComponent, ...]]:
    _, canonical_book = canonical_centered_configuration()
    return {
        "canonical_single": (GaussianBookComponent(1.0, canonical_book),),
        "nearby_same_sign": (
            GaussianBookComponent(0.5, GaussianBookParams(0.04, 0.18, 0.20, 1)),
            GaussianBookComponent(0.5, GaussianBookParams(0.08, 0.22, 0.30, 1)),
        ),
        "dispersed_same_sign": (
            GaussianBookComponent(0.6, GaussianBookParams(0.02, 0.12, 0.10, 1)),
            GaussianBookComponent(0.4, GaussianBookParams(0.10, 0.24, 0.50, 1)),
        ),
        "offsetting_signs": (
            GaussianBookComponent(0.7, GaussianBookParams(0.02, 0.12, 0.10, 1)),
            GaussianBookComponent(0.3, GaussianBookParams(0.10, 0.24, 0.50, -1)),
        ),
        "reversed_single": (
            GaussianBookComponent(
                1.0,
                replace(canonical_book, dealer_sign=-1),
            ),
        ),
    }


def _mixture_metrics() -> dict[str, dict[str, float | str]]:
    model, _ = canonical_centered_configuration()
    output: dict[str, dict[str, float | str]] = {}
    for name, components in _mixture_scenarios().items():
        partials = normalized_gaussian_mixture_partials(model, components)
        row: dict[str, float | str] = {
            "G_x": partials["G_a"],
            "G_v": partials["G_v"],
        }
        try:
            point = gaussian_mixture_hopf_point(model, components)
        except ValueError as error:
            row["classification"] = "no_valid_local_hopf"
            row["reason"] = str(error)
        else:
            row.update(
                {
                    "kappa_star": point.kappa,
                    "period_years": point.period_years,
                    "first_lyapunov": point.first_lyapunov,
                    "transversality": point.transversality,
                    "classification": (
                        "supercritical" if point.first_lyapunov < 0.0 else "subcritical"
                    ),
                }
            )
        output[name] = row
    return output


def _independent_lyapunov_check() -> dict[str, float]:
    model, book = canonical_centered_configuration()
    point = gaussian_book_hopf_point(model, book)
    partials = normalized_gaussian_book_partials(model, book)
    jacobian = centered_jacobian(
        kappa=point.kappa,
        model=model,
        G_x=partials["G_a"],
        G_v=partials["G_v"],
    )

    def drift(state: NDArray[np.float64]) -> NDArray[np.float64]:
        return centered_drift_gaussian_book(
            state,
            kappa=point.kappa,
            model=model,
            book=book,
        )

    B, C = build_bilinear_trilinear_tensors(
        drift,
        np.array([0.0, model.theta_v, 0.0], dtype=np.float64),
        h=2.0e-4,
    )
    finite_difference = compute_lyapunov_coefficient(jacobian, B, C, omega=point.omega)
    return {
        "analytic": point.first_lyapunov,
        "finite_difference": finite_difference,
        "absolute_difference": abs(finite_difference - point.first_lyapunov),
        "relative_difference": abs(finite_difference / point.first_lyapunov - 1.0),
    }


def compute(cfg: RobustnessConfig) -> tuple[dict[str, object], dict[str, object]]:
    model, book = canonical_centered_configuration()
    point = gaussian_book_hopf_point(model, book)
    cycles = [_cycle_summary(distance, cfg=cfg) for distance in cfg.relative_distances]
    distances = np.array([summary.relative_distance for summary in cycles])
    amplitudes = np.array([summary.x_amplitude for summary in cycles])
    log_slope, log_intercept = np.polyfit(np.log(distances), np.log(amplitudes), 1)
    fitted_log = log_intercept + log_slope * np.log(distances)
    log_r_squared = 1.0 - float(
        np.sum((np.log(amplitudes) - fitted_log) ** 2)
        / np.sum((np.log(amplitudes) - np.mean(np.log(amplitudes))) ** 2)
    )
    local = distances <= 0.02
    local_log_slope, local_log_intercept = np.polyfit(
        np.log(distances[local]),
        np.log(amplitudes[local]),
        1,
    )
    local_fitted_log = local_log_intercept + local_log_slope * np.log(distances[local])
    local_log_r_squared = 1.0 - float(
        np.sum((np.log(amplitudes[local]) - local_fitted_log) ** 2)
        / np.sum((np.log(amplitudes[local]) - np.mean(np.log(amplitudes[local]))) ** 2)
    )

    loose = _cycle_summary(0.02, cfg=cfg, rtol=1.0e-8, atol=1.0e-10)
    tight = _cycle_summary(0.02, cfg=cfg, rtol=1.0e-10, atol=1.0e-12)
    initial_conditions = (
        np.array([0.001, model.theta_v, 0.0]),
        np.array([-0.001, model.theta_v, 0.0]),
        np.array([0.01, model.theta_v + 0.003, 0.0]),
        np.array([-0.01, model.theta_v - 0.003, 0.01]),
    )
    initial_summaries = [
        _cycle_summary(0.02, cfg=cfg, initial=initial) for initial in initial_conditions
    ]
    initial_amplitudes = np.array([summary.x_amplitude for summary in initial_summaries])
    initial_periods = np.array([summary.period for summary in initial_summaries])

    multipliers, oat_curves = _oat_sensitivity(cfg)
    means, sigmas, classification, classification_counts = _lyapunov_classification_map(cfg)
    mixture = _mixture_metrics()
    metrics: dict[str, object] = {
        "canonical": {
            "kappa_star": point.kappa,
            "period_years": point.period_years,
            "first_lyapunov": point.first_lyapunov,
            "transversality": point.transversality,
        },
        "cycle_summaries": [asdict(summary) for summary in cycles],
        "amplitude_log_log_slope": float(log_slope),
        "amplitude_log_log_intercept": float(log_intercept),
        "amplitude_log_log_r_squared": log_r_squared,
        "local_amplitude_log_log_slope_through_2pct": float(local_log_slope),
        "local_amplitude_log_log_intercept_through_2pct": float(local_log_intercept),
        "local_amplitude_log_log_r_squared_through_2pct": local_log_r_squared,
        "solver_convergence": {
            "x_amplitude_relative_difference": abs(loose.x_amplitude / tight.x_amplitude - 1.0),
            "period_relative_difference": abs(loose.period / tight.period - 1.0),
        },
        "initial_condition_convergence": {
            "x_amplitude_coefficient_of_variation": float(
                np.std(initial_amplitudes) / np.mean(initial_amplitudes)
            ),
            "period_coefficient_of_variation": float(
                np.std(initial_periods) / np.mean(initial_periods)
            ),
        },
        "independent_lyapunov_check": _independent_lyapunov_check(),
        "classification_counts": classification_counts,
        "mixture_scenarios": mixture,
        "oat_valid_counts": {
            name: int(np.sum(np.isfinite(values))) for name, values in oat_curves.items()
        },
    }
    arrays: dict[str, object] = {
        "distances": distances,
        "amplitudes": amplitudes,
        "periods": np.array([summary.period for summary in cycles]),
        "multipliers": multipliers,
        "oat_curves": oat_curves,
        "means": means,
        "sigmas": sigmas,
        "classification": classification,
    }
    return metrics, arrays


def render(metrics: dict[str, object], arrays: dict[str, object], output: Path) -> None:
    canonical = metrics["canonical"]
    assert isinstance(canonical, dict)
    distances = np.asarray(arrays["distances"], dtype=np.float64)
    amplitudes = np.asarray(arrays["amplitudes"], dtype=np.float64)
    periods = np.asarray(arrays["periods"], dtype=np.float64)
    multipliers = np.asarray(arrays["multipliers"], dtype=np.float64)
    oat_curves = arrays["oat_curves"]
    assert isinstance(oat_curves, dict)
    means = np.asarray(arrays["means"], dtype=np.float64)
    sigmas = np.asarray(arrays["sigmas"], dtype=np.float64)
    classification = np.asarray(arrays["classification"], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), constrained_layout=True)
    ax = axes[0, 0]
    root_distance = np.sqrt(distances)
    local = distances <= 0.02
    coefficient = float(
        (root_distance[local] @ amplitudes[local]) / (root_distance[local] @ root_distance[local])
    )
    fitted = coefficient * np.linspace(0.0, float(np.max(root_distance)) * 1.05, 200)
    fit_axis = np.linspace(0.0, float(np.max(root_distance)) * 1.05, 200)
    ax.plot(
        fit_axis,
        fitted,
        color="black",
        linewidth=1.0,
        label=r"local fit ($\leq2\%$)",
    )
    ax.scatter(root_distance, amplitudes, color="C0", zorder=3)
    ax.set(
        xlabel=r"distance above threshold $\sqrt{(\kappa-\kappa^\star)/\kappa^\star}$",
        ylabel=r"half peak-to-peak amplitude of $X$",
        title="(a) Local square-root amplitude law",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.axhline(
        float(canonical["period_years"]),
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=r"$2\pi/\omega^\star$",
    )
    ax.plot(100.0 * distances, periods, marker="o", color="C1")
    ax.set(
        xlabel=r"coupling above threshold (percent)",
        ylabel="measured period (years)",
        title="(b) Nonlinear period converges to Hopf period",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    labels = {
        "delta": r"$\delta$",
        "kappa_v": r"$\kappa_v$",
        "theta_v": r"$\theta_v$",
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "gamma": r"$\gamma$",
        "sigma_moneyness": r"$\sigma_q$",
        "effective_maturity": r"$T$",
    }
    for name, values in oat_curves.items():
        ax.plot(multipliers, np.asarray(values), linewidth=1.0, label=labels[name])
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.axvline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.set(
        xlabel="one-at-a-time parameter multiplier",
        ylabel=r"threshold ratio $\kappa^\star/\kappa^\star_0$",
        title="(c) Mechanical threshold sensitivity",
    )
    ax.legend(ncol=4, frameon=False, fontsize=7)

    ax = axes[1, 1]
    cmap = ListedColormap(["#3b6fb6", "#c44e52"])
    norm = BoundaryNorm([-1.5, 0.0, 1.5], cmap.N)
    masked = np.ma.masked_invalid(classification)
    ax.pcolormesh(means, sigmas, masked, cmap=cmap, norm=norm, shading="auto")
    ax.scatter([0.06], [0.20], marker="*", s=100, color="white", edgecolor="black")
    ax.text(
        0.98,
        0.06,
        "blue: supercritical\nred: subcritical\nblank: no valid root",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.set(
        xlabel=r"book mean $\mu_q$",
        ylabel=r"book dispersion $\sigma_q$",
        title=r"(d) Sign of $\ell_1$ is book-shape dependent",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    cfg = RobustnessConfig()
    metrics, arrays = compute(cfg)
    output = FIGURES_DIR / "centered_hopf_robustness.pdf"
    render(metrics, arrays, output)
    run_dir = make_run_dir("centered_model_robustness")
    save_config(run_dir, cfg)
    save_metrics(run_dir, metrics)
    print(f"wrote {output}")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
