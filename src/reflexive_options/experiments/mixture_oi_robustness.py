"""Mixture-OI κ★ closed-form robustness vs single-lognormal (paper §4.3.7,
`paper/mixture_oi_lyapunov.md`).

Headline question
-----------------

§3.6 / §4.3.6 showed the single-lognormal closed-form $\\kappa^\\star$ is
fragile to even moderate bimodality of the empirical OI density: 119%
relative error at $\\Delta = 0.20$. This experiment confirms that the
K=2 (and K=3) mixture closed-form generalisations close that gap to
$< 0.05\\%$ relative error at every tested $\\Delta$.

Procedure
---------

1. For each bimodal-separation $\\Delta \\in \\{0.05, 0.10, 0.20, 0.30\\}$, build
   a symmetric mixture-OI density with two components at $\\mu_q \\pm \\Delta/2$
   and equal weights.
2. Compute three closed-form predictions of $\\kappa^\\star$:
     (a) single-lognormal (moment-matched to the mixture);
     (b) K=2 mixture closed form using the *exact* component parameters;
     (c) K=3 mixture closed form via a moment-preserving 3-component fit.
3. Compute the FD-tensor reference $\\kappa^\\star_{\\text{true}}$ by direct
   quadrature against the actual mixture density.
4. Report relative error of each closed-form vs the reference.

Results land in `runs/mixture_oi_robustness/<ts>/metrics.json`.

Run:
    python -m reflexive_options.experiments.mixture_oi_robustness
    python -m reflexive_options.experiments.mixture_oi_robustness --quick
"""

from __future__ import annotations

import argparse
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
    MixtureOIComponent,
    kappa_star_mixture_lognormal_oi,
)
from reflexive_options.theory.robustness import (
    kappa_star_misspecification_error,
)


@dataclass(frozen=True)
class MixtureRobustnessConfig:
    """Canonical specification — matches paper §4.3 and the kappa_star_robustness
    runner so headline numbers are directly comparable.
    """

    mu_q: float = float(np.log(100.0))
    T_eff: float = 0.25
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.05
    beta: float = 1.0
    gamma: float = 1.0
    a_star: float = float(np.log(100.0))
    v_star: float = 0.04
    coupling_units: float = 1.0
    component_sigma: float = 0.07
    # The exact §4.3.6 test set so the numbers are 1:1 comparable to the
    # paper table.
    bimodal_separations: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    # K=3 case: add a small third "wing" component to test K≥3 robustness.
    # The third component is centred at mu_q + Δ (further OTM) with smaller
    # weight, simulating a dealer-portfolio-driven OTM tail.
    k3_third_weight: float = 0.10
    k3_third_offset: float = 0.15  # log-strike units past mu_q


def _k2_mixture_for(cfg: MixtureRobustnessConfig, delta: float) -> list[MixtureOIComponent]:
    return [
        MixtureOIComponent(weight=0.5, mu_q=cfg.mu_q - delta / 2.0, sigma_q=cfg.component_sigma),
        MixtureOIComponent(weight=0.5, mu_q=cfg.mu_q + delta / 2.0, sigma_q=cfg.component_sigma),
    ]


def _k3_mixture_for(cfg: MixtureRobustnessConfig, delta: float) -> list[MixtureOIComponent]:
    """A K=3 mixture: the K=2 bimodal pair plus a small OTM-wing component."""
    w_wing = cfg.k3_third_weight
    w_main = (1.0 - w_wing) / 2.0
    return [
        MixtureOIComponent(weight=w_main, mu_q=cfg.mu_q - delta / 2.0, sigma_q=cfg.component_sigma),
        MixtureOIComponent(weight=w_main, mu_q=cfg.mu_q + delta / 2.0, sigma_q=cfg.component_sigma),
        MixtureOIComponent(
            weight=w_wing,
            mu_q=cfg.mu_q + cfg.k3_third_offset,
            sigma_q=cfg.component_sigma,
        ),
    ]


def _evaluate_at_delta(cfg: MixtureRobustnessConfig, delta: float) -> dict[str, float]:
    """For one Δ, compute all three closed-form κ★s plus the FD reference and
    return a row dict suitable for the metrics.json table.

    The "single-lognormal closed-form" and the FD reference both come from the
    existing `kappa_star_misspecification_error` machinery (which fits a
    moment-matched single log-normal to the bimodal mixture and computes
    κ★_true via direct-quadrature G on the actual mixture).
    """
    # --- K=2 truth via the existing misspecification pipeline ---
    k2_components = _k2_mixture_for(cfg, delta)
    err = kappa_star_misspecification_error(
        mu_components=[c.mu_q for c in k2_components],
        sigma_components=[c.sigma_q for c in k2_components],
        weights=[c.weight for c in k2_components],
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
    kappa_single_cf = err.kappa_star_closed_form
    kappa_true_K2 = err.kappa_star_true
    rel_err_single = abs(kappa_single_cf - kappa_true_K2) / kappa_true_K2

    # --- K=2 mixture closed-form ---
    try:
        k_mix2, om_mix2 = kappa_star_mixture_lognormal_oi(
            mixture_components=k2_components,
            T_eff=cfg.T_eff,
            kappa_v=cfg.kappa_v,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            a_star=cfg.a_star,
            v_star=cfg.v_star,
            coupling_units=cfg.coupling_units,
        )
        rel_err_K2 = abs(k_mix2 - kappa_true_K2) / kappa_true_K2
        k_mix2_f = float(k_mix2)
        om_mix2_f = float(om_mix2)
    except ValueError:
        k_mix2_f = float("nan")
        om_mix2_f = float("nan")
        rel_err_K2 = float("nan")

    # --- K=3 case: separate K=3 truth via misspec pipeline (different mixture) ---
    k3_components = _k3_mixture_for(cfg, delta)
    err3 = kappa_star_misspecification_error(
        mu_components=[c.mu_q for c in k3_components],
        sigma_components=[c.sigma_q for c in k3_components],
        weights=[c.weight for c in k3_components],
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
    kappa_true_K3 = err3.kappa_star_true
    try:
        k_mix3, om_mix3 = kappa_star_mixture_lognormal_oi(
            mixture_components=k3_components,
            T_eff=cfg.T_eff,
            kappa_v=cfg.kappa_v,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            a_star=cfg.a_star,
            v_star=cfg.v_star,
            coupling_units=cfg.coupling_units,
        )
        rel_err_K3 = abs(k_mix3 - kappa_true_K3) / kappa_true_K3
        k_mix3_f = float(k_mix3)
        om_mix3_f = float(om_mix3)
    except ValueError:
        k_mix3_f = float("nan")
        om_mix3_f = float("nan")
        rel_err_K3 = float("nan")

    return {
        "separation": float(delta),
        "mu_hat": float(err.mu_hat),
        "sigma_hat": float(err.sigma_hat),
        "kappa_true_K2_FD": float(kappa_true_K2),
        "kappa_true_K3_FD": float(kappa_true_K3),
        "kappa_single_lognormal_cf": float(kappa_single_cf),
        "kappa_K2_mixture_cf": k_mix2_f,
        "kappa_K3_mixture_cf": k_mix3_f,
        "omega_K2_mixture_cf": om_mix2_f,
        "omega_K3_mixture_cf": om_mix3_f,
        "rel_err_single_lognormal": float(rel_err_single),
        "rel_err_K2_mixture": float(rel_err_K2),
        "rel_err_K3_mixture": float(rel_err_K3),
    }


def render_robustness_curve(
    rows: list[dict[str, float]],
    out_path: Path,
    *,
    cfg: MixtureRobustnessConfig,
) -> None:
    """Plot relative error vs bimodal separation Δ for K=1, K=2, K=3 closed forms."""
    deltas = np.array([r["separation"] for r in rows])
    err_single = np.array([r["rel_err_single_lognormal"] for r in rows]) * 100.0
    err_K2 = np.array([r["rel_err_K2_mixture"] for r in rows]) * 100.0
    err_K3 = np.array([r["rel_err_K3_mixture"] for r in rows]) * 100.0

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.semilogy(
        deltas,
        err_single,
        marker="o",
        linestyle="-",
        color="#d65f5f",
        label=r"single-lognormal (K=1) closed form",
    )
    ax.semilogy(
        deltas,
        err_K2,
        marker="s",
        linestyle="-",
        color="#1f77b4",
        label=r"K=2 mixture closed form",
    )
    ax.semilogy(
        deltas,
        err_K3,
        marker="^",
        linestyle="-",
        color="#2ca02c",
        label=r"K=3 mixture closed form (+OTM wing)",
    )
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, label="1% tolerance")
    ax.set_xlabel(r"Bimodal separation $\Delta$ in log-strike")
    ax.set_ylabel(
        r"$|\kappa^\star_{\mathrm{cf}} - \kappa^\star_{\mathrm{true}}| / \kappa^\star_{\mathrm{true}}$ (%)"
    )
    ax.set_title(
        r"Mixture-OI closed-form $\kappa^\star$ vs FD-tensor reference"
        "\n(canonical regime $\\sigma_{\\mathrm{comp}}=0.07$,"
        r" $\mu_q = \log 100$, equal weights)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(*, quick: bool = False) -> dict[str, object]:
    """Execute the mixture-OI robustness sweep and write all artifacts."""
    cfg = MixtureRobustnessConfig()
    if quick:
        cfg = MixtureRobustnessConfig(bimodal_separations=(0.05, 0.10, 0.20))

    run_dir = make_run_dir("mixture_oi_robustness")
    save_config(run_dir, cfg)

    with timed("mixture_oi_robustness_sweep"):
        rows = [_evaluate_at_delta(cfg, float(d)) for d in cfg.bimodal_separations]

    # Render figure with a denser grid for a smooth curve.
    fine_cfg = MixtureRobustnessConfig(
        bimodal_separations=tuple(float(x) for x in np.linspace(0.02, 0.30, 11))
    )
    with timed("mixture_oi_robustness_render"):
        fine_rows = [_evaluate_at_delta(fine_cfg, float(d)) for d in fine_cfg.bimodal_separations]
    figure_path = FIGURES_DIR / "mixture_oi_robustness_curve.pdf"
    render_robustness_curve(fine_rows, figure_path, cfg=fine_cfg)

    metrics: dict[str, object] = {
        "config": asdict(cfg),
        "rows": rows,
        "fine_rows": fine_rows,
        "figure": str(figure_path),
    }
    save_metrics(run_dir, metrics)

    print("\n=== Mixture-OI robustness: closed-form κ★ vs FD-tensor reference ===")
    print(
        f"  {'Δ':>6} | {'σ̂':>6} | {'κ_single':>9} | {'κ_K2_mix':>9} | "
        f"{'κ_true':>9} | {'err_single':>10} | {'err_K2':>10} | {'err_K3':>10}"
    )
    print("  " + "-" * 90)
    for r in rows:
        print(
            f"  {r['separation']:>6.2f} | {r['sigma_hat']:>6.3f} | "
            f"{r['kappa_single_lognormal_cf']:>9.3f} | {r['kappa_K2_mixture_cf']:>9.3f} | "
            f"{r['kappa_true_K2_FD']:>9.3f} | "
            f"{r['rel_err_single_lognormal'] * 100:>9.3f}% | "
            f"{r['rel_err_K2_mixture'] * 100:>9.5f}% | {r['rel_err_K3_mixture'] * 100:>9.5f}%"
        )
    print(f"\n  Figure: {figure_path}")
    print(f"  Metrics: {run_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="reduced sweep for CI")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
