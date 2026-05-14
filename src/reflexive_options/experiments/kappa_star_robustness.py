"""κ★ robustness analysis — sensitivity to OI mis-specification (paper §4.3.6).

Three quantitative artifacts:

    1. **Analytical elasticities** $\\eta_{\\sigma_q}$, $\\eta_{\\mu_q}$ at the
       canonical regime, via implicit differentiation of the closed-form Hopf
       quadratic (`kappa_star_sensitivity_lognormal_oi`).

    2. **2D sensitivity heatmap** of $\\kappa^\\star(\\sigma_q, \\mu_q)$ over the
       ±30% canonical window, with 5%, 10%, 25% iso-contours of fractional
       deviation. Output: `paper/figures/kappa_star_robustness_heatmap.pdf`.

    3. **Multi-modal mis-specification curve**: relative error of the closed-form
       $\\kappa^\\star_{\\text{cf}}$ (computed at a moment-matched single
       log-normal) vs the "true" $\\kappa^\\star_{\\text{true}}$ (computed via
       FD-on-G against a bimodal mixture density), as a function of bimodality
       severity. Output: `paper/figures/kappa_star_misspecification_curve.pdf`.

Both figures use SOURCE_DATE_EPOCH for byte-stable rebuilds.
Run:
    python -m reflexive_options.experiments.kappa_star_robustness
    python -m reflexive_options.experiments.kappa_star_robustness --quick   # CI
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
from reflexive_options.theory.robustness import (
    calibration_tolerance,
    kappa_star_misspecification_error,
    kappa_star_sensitivity_lognormal_oi,
)


@dataclass(frozen=True)
class RobustnessConfig:
    """Canonical specification matching paper §4.3 / §3.5.

    The (σ_q, μ_q) sweep window is ±30% of the canonical baseline:
        σ_q canonical = 0.10 ⇒ window [0.07, 0.13]
        μ_q canonical = log 100 ⇒ window [0.7·log100, 1.3·log100]
            (in log-strike units; effectively a ±30% strike-mean shift)
    """

    mu_q: float = float(np.log(100.0))
    sigma_q: float = 0.10
    T_eff: float = 0.25
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.05
    beta: float = 1.0
    gamma: float = 1.0
    a_star: float = float(np.log(100.0))
    v_star: float = 0.04
    coupling_units: float = 1.0
    sweep_pct: float = 0.30  # ±30% window
    n_sigma: int = 41
    n_mu: int = 41
    # Mis-specification curve: bimodal mixture with separation in log-strike
    misspec_separations: tuple[float, ...] = (
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
        0.25,
        0.30,
    )
    # Each component has its own σ; default 0.07 keeps the moment-matched
    # log-normal close to the canonical σ_q ≈ 0.10 across the sweep.
    component_sigma: float = 0.07
    target_kappa_rel_errors: tuple[float, ...] = (0.05, 0.10, 0.25)


def _heatmap_data(cfg: RobustnessConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute κ★(σ_q, μ_q) on the ±sweep_pct window. Returns (sigmas, mus, K, k0)."""
    s0, m0 = cfg.sigma_q, cfg.mu_q
    sigmas = np.linspace(s0 * (1.0 - cfg.sweep_pct), s0 * (1.0 + cfg.sweep_pct), cfg.n_sigma)
    # μ_q is positive (= log 100); ±30% scales it.
    mus = np.linspace(m0 * (1.0 - cfg.sweep_pct), m0 * (1.0 + cfg.sweep_pct), cfg.n_mu)
    K = np.full((cfg.n_mu, cfg.n_sigma), np.nan, dtype=np.float64)
    for i, mu in enumerate(mus):
        for j, sg in enumerate(sigmas):
            try:
                p = G_lognormal_oi_partials(
                    a_star=cfg.a_star,
                    v_star=cfg.v_star,
                    mu_q=float(mu),
                    sigma_q=float(sg),
                    T_eff=cfg.T_eff,
                    coupling_units=cfg.coupling_units,
                )
                ks, _ = kappa_star_lognormal_oi(
                    G_y=p["G_a"],
                    G_v=p["G_v"],
                    kappa_v=cfg.kappa_v,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                )
                K[i, j] = float(ks)
            except ValueError:
                # No Hopf at this (σ_q, μ_q) — leave NaN
                pass

    # Canonical κ★
    p0 = G_lognormal_oi_partials(
        a_star=cfg.a_star,
        v_star=cfg.v_star,
        mu_q=cfg.mu_q,
        sigma_q=cfg.sigma_q,
        T_eff=cfg.T_eff,
        coupling_units=cfg.coupling_units,
    )
    k0, _ = kappa_star_lognormal_oi(
        G_y=p0["G_a"],
        G_v=p0["G_v"],
        kappa_v=cfg.kappa_v,
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
    )
    return sigmas, mus, K, float(k0)


def render_heatmap(
    sigmas: np.ndarray,
    mus: np.ndarray,
    K: np.ndarray,
    k0: float,
    out_path: Path,
    *,
    cfg: RobustnessConfig,
) -> None:
    """Render the κ★(σ_q, μ_q) heatmap with iso-contours of fractional deviation."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: raw κ★ heatmap
    pcm = ax_left.pcolormesh(sigmas, mus, K, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax_left, label=r"$\kappa^\star$")
    ax_left.scatter([cfg.sigma_q], [cfg.mu_q], color="white", marker="x", s=100, label=r"canonical")
    ax_left.set_xlabel(r"OI spread $\sigma_q$ (log-strike)")
    ax_left.set_ylabel(r"OI mean $\mu_q$ (log-strike)")
    ax_left.set_title(r"$\kappa^\star(\sigma_q, \mu_q)$ — closed form (Eq. 18)")
    ax_left.legend(loc="upper right", fontsize=9)

    # Right: |Δκ★/κ★| with iso-contours at 5%, 10%, 25%
    rel_dev = np.abs(K / k0 - 1.0)
    pcm2 = ax_right.pcolormesh(
        sigmas, mus, rel_dev, shading="auto", cmap="magma", vmin=0.0, vmax=0.50
    )
    fig.colorbar(pcm2, ax=ax_right, label=r"$|\kappa^\star/\kappa^\star_{\text{canon}} - 1|$")
    cs = ax_right.contour(
        sigmas,
        mus,
        rel_dev,
        levels=[0.05, 0.10, 0.25],
        colors=["#7fffff", "#ffff66", "#ff7777"],
        linewidths=1.6,
    )
    ax_right.clabel(cs, fmt={0.05: "5%", 0.10: "10%", 0.25: "25%"}, inline=True, fontsize=9)
    ax_right.scatter([cfg.sigma_q], [cfg.mu_q], color="white", marker="x", s=100)
    ax_right.set_xlabel(r"OI spread $\sigma_q$ (log-strike)")
    ax_right.set_ylabel(r"OI mean $\mu_q$ (log-strike)")
    ax_right.set_title(r"Fractional deviation with iso-contours")

    fig.suptitle(
        r"Sensitivity of the closed-form Hopf threshold $\kappa^\star$ to "
        r"$(\sigma_q, \mu_q)$ over $\pm$"
        f"{int(cfg.sweep_pct * 100)}% canonical window — "
        r"$\kappa^\star_{\text{canon}} = "
        f"{k0:.3f}$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _misspec_data(cfg: RobustnessConfig) -> list[dict[str, float]]:
    """Compute (separation → relative error) curve for bimodal mis-specification."""
    rows: list[dict[str, float]] = []
    for sep in cfg.misspec_separations:
        sep_f = float(sep)
        if sep_f == 0.0:
            # Degenerate "no bimodality" — use a single component to get an exact comparison
            mu_components = [cfg.mu_q]
            sigma_components = [cfg.component_sigma]
            weights = [1.0]
        else:
            mu_components = [cfg.mu_q - sep_f / 2.0, cfg.mu_q + sep_f / 2.0]
            sigma_components = [cfg.component_sigma, cfg.component_sigma]
            weights = [0.5, 0.5]
        err = kappa_star_misspecification_error(
            mu_components=mu_components,
            sigma_components=sigma_components,
            weights=weights,
            T_eff=cfg.T_eff,
            kappa_v=cfg.kappa_v,
            theta_v=cfg.theta_v,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            a_star=cfg.a_star,
            v_star=cfg.v_star,
            coupling_units=cfg.coupling_units,
        )
        rows.append(asdict(err))
    return rows


def render_misspec_curve(
    rows: list[dict[str, float]],
    out_path: Path,
    *,
    cfg: RobustnessConfig,
) -> None:
    """Render the misspecification severity → κ★ relative error curve."""
    seps = np.array([r["separation"] for r in rows])
    rel_errs = np.array([r["relative_error"] for r in rows])
    k_cf = np.array([r["kappa_star_closed_form"] for r in rows])
    k_tr = np.array([r["kappa_star_true"] for r in rows])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.0))

    ax_left.plot(seps, k_cf, marker="o", label=r"closed-form $\kappa^\star_{\text{cf}}$")
    ax_left.plot(
        seps, k_tr, marker="s", label=r"true $\kappa^\star_{\text{true}}$ (FD on mixture G)"
    )
    ax_left.set_xlabel(r"Bimodal separation in log-strike")
    ax_left.set_ylabel(r"$\kappa^\star$")
    ax_left.set_title(r"Closed-form vs true Hopf threshold")
    ax_left.legend(loc="best", fontsize=9)
    ax_left.grid(alpha=0.3)

    ax_right.plot(seps, rel_errs * 100.0, marker="o", color="#d65f5f")
    ax_right.axhline(5.0, color="#7fffff", linestyle="--", alpha=0.5, label="5%")
    ax_right.axhline(10.0, color="#ffff66", linestyle="--", alpha=0.5, label="10%")
    ax_right.axhline(25.0, color="#ff7777", linestyle="--", alpha=0.5, label="25%")
    ax_right.set_xlabel(r"Bimodal separation in log-strike")
    ax_right.set_ylabel(
        r"$|\kappa^\star_{\text{cf}} - \kappa^\star_{\text{true}}| / \kappa^\star_{\text{true}}$ (%)"
    )
    ax_right.set_title(r"Mis-specification relative error")
    ax_right.legend(loc="best", fontsize=9)
    ax_right.grid(alpha=0.3)

    fig.suptitle(
        r"Closed-form $\kappa^\star$ robustness to non-log-normal OI: "
        r"bimodal mixture with components at $\mu_q \pm \Delta/2$, equal weights, "
        r"$\sigma_{\text{comp}} = "
        f"{cfg.component_sigma}$",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(*, quick: bool = False) -> dict[str, object]:
    """Execute the full robustness sweep and write all artifacts.

    Returns the metrics dict (also persisted to disk).
    """
    cfg = RobustnessConfig()
    if quick:
        cfg = RobustnessConfig(
            n_sigma=11,
            n_mu=11,
            misspec_separations=(0.0, 0.05, 0.10, 0.20),
        )

    run_dir = make_run_dir("kappa_star_robustness")
    save_config(run_dir, cfg)

    # Deliverable 1: analytical sensitivity at the canonical regime.
    sens = kappa_star_sensitivity_lognormal_oi(
        mu_q=cfg.mu_q,
        sigma_q=cfg.sigma_q,
        T_eff=cfg.T_eff,
        kappa_v=cfg.kappa_v,
        theta_v=cfg.theta_v,
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        a_star=cfg.a_star,
        v_star=cfg.v_star,
        coupling_units=cfg.coupling_units,
    )

    # Deliverable 2: 2D heatmap.
    with timed("heatmap_compute"):
        sigmas, mus, K, k0 = _heatmap_data(cfg)
    heatmap_path = FIGURES_DIR / "kappa_star_robustness_heatmap.pdf"
    render_heatmap(sigmas, mus, K, k0, heatmap_path, cfg=cfg)

    # Deliverable 3: misspecification curve.
    with timed("misspec_compute"):
        misspec_rows = _misspec_data(cfg)
    misspec_path = FIGURES_DIR / "kappa_star_misspecification_curve.pdf"
    render_misspec_curve(misspec_rows, misspec_path, cfg=cfg)

    # Deliverable 4: calibration tolerance table.
    tolerance_table = [
        calibration_tolerance(sens, target_kappa_relative_error=t)
        for t in cfg.target_kappa_rel_errors
    ]

    # Compute the rel-deviation summary at ±10% and ±30% σ_q corners
    rel_dev = np.abs(K / k0 - 1.0)
    s0, m0 = cfg.sigma_q, cfg.mu_q
    s_lo = s0 * 0.90
    s_hi = s0 * 1.10
    j_lo = int(np.argmin(np.abs(sigmas - s_lo)))
    j_hi = int(np.argmin(np.abs(sigmas - s_hi)))
    i_mu = int(np.argmin(np.abs(mus - m0)))
    rel_dev_at_10pct_sigma = float(max(rel_dev[i_mu, j_lo], rel_dev[i_mu, j_hi]))
    rel_dev_at_30pct_sigma = float(max(rel_dev[i_mu, 0], rel_dev[i_mu, -1]))

    metrics = {
        "kappa_star_canonical": k0,
        "omega_star_canonical": sens.omega_star,
        "G_y_canonical": sens.G_y,
        "G_v_canonical": sens.G_v,
        "sensitivity": asdict(sens),
        "heatmap": {
            "sigma_grid": sigmas.tolist(),
            "mu_grid": mus.tolist(),
            "kappa_star_grid": K.tolist(),
            "rel_dev_at_10pct_sigma_q_only": rel_dev_at_10pct_sigma,
            "rel_dev_at_30pct_sigma_q_only": rel_dev_at_30pct_sigma,
        },
        "misspec": misspec_rows,
        "calibration_tolerance": tolerance_table,
        "figures": {
            "heatmap": str(heatmap_path),
            "misspec_curve": str(misspec_path),
        },
    }
    save_metrics(run_dir, metrics)

    print(f"\n=== Sensitivity at canonical regime (κ★ = {k0:.4f}) ===")
    print(f"η_σq (elasticity)        = {sens.elasticity_sigma_q:+.4f}")
    print(f"η_μq (elasticity)        = {sens.elasticity_mu_q:+.4f}")
    print(f"∂κ★/∂μ_q                  = {sens.dkappa_dmu_q:+.4f}")
    print(f"∂κ★/∂σ_q                  = {sens.dkappa_dsigma_q:+.4f}")
    print(f"%-Δκ★ per unit Δμ_q       = {sens.pct_dkappa_per_unit_mu_q:+.2f}%")
    print(f"\n=== ±10% σ_q-only sweep: |Δκ★/κ★| = {rel_dev_at_10pct_sigma * 100:.2f}% ===")
    print(f"=== ±30% σ_q-only sweep: |Δκ★/κ★| = {rel_dev_at_30pct_sigma * 100:.2f}% ===")
    print("\n=== Calibration tolerance (independent-error quadrature) ===")
    for t in tolerance_table:
        print(
            f"target |Δκ★/κ★| ≤ {t['target_kappa_rel_err'] * 100:.0f}%  →  "
            f"σ_q to ±{t['sigma_q_pct_tol']:.2f}%, "
            f"μ_q to ±{t['mu_q_log_strike_tol']:.4f} log-strike"
        )
    print("\n=== Multi-modal misspecification: closed-form vs true κ★ ===")
    for row in misspec_rows:
        print(
            f"sep={row['separation']:.3f}  "
            f"σ̂_q={row['sigma_hat']:.4f}  "
            f"κ_cf={row['kappa_star_closed_form']:.3f}  "
            f"κ_true={row['kappa_star_true']:.3f}  "
            f"rel.err={row['relative_error'] * 100:.3f}%"
        )
    print(f"\nFigures written:\n  {heatmap_path}\n  {misspec_path}")
    print(f"Metrics: {run_dir / 'metrics.json'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduce grid + misspec resolution for CI smoke-test",
    )
    args = parser.parse_args()
    metrics = run(quick=args.quick)
    # JSON-dump to stdout in case downstream wants to pipe.
    if os.environ.get("ROBUSTNESS_DUMP_JSON"):
        print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
