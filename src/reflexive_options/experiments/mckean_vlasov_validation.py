"""McKean-Vlasov propagation-of-chaos validation at the canonical regime.

Runs the n-dealer particle simulator at $n \\in \\{10, 100, 1000\\}$ (and
configurable extras), measures $\\sup_t \\sqrt{E[(G_bar_n - G_bar_inf)^2]}$
across replicates, and verifies the Sznitman (1991) $1/n$ rate by fitting
$\\log\\text{RMSE}$ vs $\\log(1/\\sqrt{n})$ — slope should be $\\approx 1$.

Renders `paper/figures/mckean_vlasov_propagation_chaos.pdf` showing the
empirical RMSE against the theoretical $1/\\sqrt{n}$ reference line.

Run:
    python -m reflexive_options.experiments.mckean_vlasov_validation
    python -m reflexive_options.experiments.mckean_vlasov_validation --quick

Outputs:
    paper/figures/mckean_vlasov_propagation_chaos.pdf
    runs/mckean_vlasov_validation/<ts>/{config,metrics}.json
    runs/mckean_vlasov_validation/<ts>/scaling.npz
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

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
    G_lognormal_oi_partials,
    kappa_star_lognormal_oi,
)
from reflexive_options.theory.mckean_vlasov import (
    ChaosScalingResult,
    mckean_vlasov_kappa_star_shift,
    propagation_of_chaos_constant,
    propagation_of_chaos_scaling,
)


@dataclass(frozen=True)
class MVValidationConfig:
    """Configuration for the propagation-of-chaos sweep.

    Defaults match the canonical log-normal-OI specification of
    paper/theory.md §4.3 — same $(\\sigma_q, \\gamma, \\mu_q, T_{eff},
    \\kappa_v, \\theta_v, \\alpha, \\beta)$ used in the codim-2 analysis.
    The OU dealer-hedging parameters $(\\theta_G, \\sigma_G, \\tau_G)$ are
    new to this section; the chosen $\\theta_G = 50$ /yr means
    $\\tau_G \\approx 5$ days, a realistic hedging cycle (intraday +
    overnight for SPX dealers).
    """

    # Canonical log-normal-OI specification (matches §4.3 / §3.6).
    mu_q: float = float(np.log(100.0))
    T_eff: float = 0.25
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.05
    beta: float = 1.0
    gamma: float = 1.0
    sigma_q: float = 0.10
    a_star: float = float(np.log(100.0))
    v_star: float = 0.04
    coupling_units: float = 1.0

    # MV-specific dealer-hedging parameters.
    theta_G: float = 50.0  # 1/yr; tau_G ~ 5 trading days
    sigma_G: float = 0.05  # idiosyncratic noise scale (in units of G)
    G0_std: float = 0.05  # initial-condition spread (matches stationary scale)

    # Particle-system sweep.
    n_grid: tuple[int, ...] = (10, 32, 100, 316, 1000)
    T: float = 0.25  # 3 months horizon (matches T_eff)
    n_steps: int = 250  # ~1 trading day per step
    n_replicates: int = 64
    seed: int = 20260514  # locked


def _g_target_constant(_t: float, *, g_star: float) -> float:
    """Frozen target — constant equilibrium value of g(S, v).

    For the propagation-of-chaos validation we hold the spot/variance
    path deterministic at its equilibrium so the question is purely
    about how well $\\bar G_n$ tracks $\\bar G_\\infty$ at finite $n$.
    The constant target makes $\\bar G_\\infty(t) \\equiv g^\\star$
    after the OU transient (the chosen $G_0$ moments centre on $g^\\star$
    so the transient is zero and the test isolates particle-noise
    averaging from initial-condition decay).
    """
    return g_star


def run(*, quick: bool = False) -> dict[str, object]:
    """Execute the propagation-of-chaos validation and write all artifacts."""
    cfg = MVValidationConfig()
    if quick:
        cfg = MVValidationConfig(
            n_grid=(10, 100, 1000),
            n_replicates=16,
            n_steps=100,
        )

    # Compute the canonical $\\kappa^\\star$ and $\\omega^\\star$ from §4.3
    # so the reported MV shift is in apples-to-apples units.
    partials = G_lognormal_oi_partials(
        a_star=cfg.a_star,
        v_star=cfg.v_star,
        mu_q=cfg.mu_q,
        sigma_q=cfg.sigma_q,
        T_eff=cfg.T_eff,
        coupling_units=cfg.coupling_units,
    )
    G_y, G_v = partials["G_a"], partials["G_v"]
    kappa_star_single, omega_star = kappa_star_lognormal_oi(
        G_y=G_y,
        G_v=G_v,
        kappa_v=cfg.kappa_v,
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
    )
    shift_ratio = mckean_vlasov_kappa_star_shift(
        theta_G=cfg.theta_G,
        omega_star=omega_star,
    )
    kappa_star_mv = kappa_star_single * shift_ratio

    # Closed-form $C(T)$.
    C_T = propagation_of_chaos_constant(
        theta_G=cfg.theta_G,
        sigma_G=cfg.sigma_G,
        var_G0=cfg.G0_std * cfg.G0_std,
        T=cfg.T,
    )

    # Particle-system sweep.
    g_star = float(partials["G"])  # frozen target = the canonical equilibrium G value

    def g_target(t: float) -> float:
        return _g_target_constant(t, g_star=g_star)

    with timed("propagation_of_chaos_scaling"):
        scaling = propagation_of_chaos_scaling(
            n_grid=np.array(cfg.n_grid, dtype=np.int64),
            theta_G=cfg.theta_G,
            sigma_G=cfg.sigma_G,
            g_target=g_target,
            G0_mean=g_star,
            G0_std=cfg.G0_std,
            T=cfg.T,
            n_steps=cfg.n_steps,
            n_replicates=cfg.n_replicates,
            seed=cfg.seed,
        )

    fig_path = FIGURES_DIR / "mckean_vlasov_propagation_chaos.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _render_propagation_of_chaos_figure(scaling, C_T=C_T, out_path=fig_path)

    # Persist artifacts.
    run_dir = make_run_dir("mckean_vlasov_validation")
    save_config(run_dir, asdict(cfg))
    metrics: dict[str, object] = {
        "kappa_star_single": float(kappa_star_single),
        "kappa_star_mv": float(kappa_star_mv),
        "kappa_star_shift_ratio": float(shift_ratio),
        "omega_star": float(omega_star),
        "theta_G": float(cfg.theta_G),
        "tau_G_years": float(1.0 / cfg.theta_G),
        "tau_G_trading_days": float(252.0 / cfg.theta_G),
        "C_T_theoretical": float(C_T),
        "n_grid": [int(n) for n in scaling.n_grid],
        "rmse_sup": [float(x) for x in scaling.rmse_sup],
        "fitted_slope_log_inv_sqrt_n": float(scaling.fitted_slope),
        "fitted_intercept_log_inv_sqrt_n": float(scaling.fitted_intercept),
        "fitted_slope_log_n": float(-0.5 * scaling.fitted_slope),  # vs n directly
        "expected_slope_vs_inv_sqrt_n": 1.0,
        "expected_slope_vs_n": -0.5,
        "figure_path": str(fig_path),
    }
    save_metrics(run_dir, metrics)
    np.savez_compressed(
        run_dir / "scaling.npz",
        n_grid=scaling.n_grid,
        rmse_sup=scaling.rmse_sup,
        fitted_slope=scaling.fitted_slope,
        fitted_intercept=scaling.fitted_intercept,
        C_T=C_T,
    )

    print(json.dumps(metrics, indent=2))
    print(f"figure -> {fig_path}")
    print(f"run dir -> {run_dir}")
    return metrics


def _render_propagation_of_chaos_figure(
    scaling: ChaosScalingResult,
    *,
    C_T: float,
    out_path: Path,
) -> None:
    """Render the $1/\\sqrt{n}$ scaling figure on log-log axes."""
    n_grid = scaling.n_grid
    rmse_sup = scaling.rmse_sup
    slope = scaling.fitted_slope
    intercept = scaling.fitted_intercept

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0))

    # Reference line: theoretical RMSE upper bound = sqrt(C_T / n).
    n_ref = np.geomspace(float(n_grid.min()), float(n_grid.max()), 64)
    rmse_theory = np.sqrt(C_T / n_ref)
    ax.loglog(
        n_ref,
        rmse_theory,
        "--",
        color="#888888",
        linewidth=1.5,
        label=r"Sznitman bound: $\sqrt{C(T)/n}$",
    )

    # Empirical points.
    ax.loglog(
        n_grid,
        rmse_sup,
        "o-",
        color="#2c3e50",
        markersize=8,
        linewidth=1.6,
        label=r"Empirical $\sup_t\sqrt{\mathbb{E}[(\bar G_n - \bar G_\infty)^2]}$",
    )

    # Annotated slope.
    fit_x = np.array([n_grid.min(), n_grid.max()], dtype=np.float64)
    fit_inv_sqrt = 1.0 / np.sqrt(fit_x)
    fit_y = np.exp(intercept) * fit_inv_sqrt**slope
    ax.loglog(
        fit_x,
        fit_y,
        ":",
        color="#d65f5f",
        linewidth=2.0,
        label=rf"LS fit: slope vs $1/\sqrt{{n}}$ = {slope:.3f}",
    )

    ax.set_xlabel(r"number of dealers $n$")
    ax.set_ylabel(r"RMSE $\sup_t\sqrt{\mathbb{E}[(\bar G_n - \bar G_\infty)^2]}$")
    ax.set_title(
        "Propagation-of-chaos for the dealer-gamma channel\n"
        r"(McKean-Vlasov mean-field SDE; canonical $\sigma_q = 0.10$ regime)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Coarse sweep (3 n-points, 16 replicates) for CI; default is the production sweep.",
    )
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
