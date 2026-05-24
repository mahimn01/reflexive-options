"""Excess entropy numerical anchor (Theorem 5, paper §3.10 / theory.md §10).

Computes the closed-form linearised excess entropy E_tau(kappa) of the 3D
reflexive SDE at the §4.2 canonical regime, on a 101-point kappa grid over
(0, kappa_star) for tau in {0.1, 1, 5} yr. Produces:

  * ``runs/info_theoretic_excess_entropy/<timestamp>/{config,metrics}.json``
  * ``paper/figures/excess_entropy_curve.pdf``

Invocation::

    python -m reflexive_options.experiments.info_theoretic_excess_entropy
    python -m reflexive_options.experiments.info_theoretic_excess_entropy --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

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
from reflexive_options.theory.bifurcation import jacobian_3d
from reflexive_options.theory.info_theoretic import (
    excess_entropy_curve,
    fit_critical_exponent,
)

# ---------------------------------------------------------------------------
# §4.2 canonical regime — locked to paper/theory.md §4.2 and
# tests/test_paper_section_4_2.py. Constant-vol surrogate convention.
# ---------------------------------------------------------------------------
_G_X = 0.5
_G_V = -0.5
_G_Z = -0.5
_ALPHA = 0.5
_BETA = 1.0
_GAMMA = 0.5
_KAPPA_V = 2.0
_KAPPA_STAR = 0.8964305216  # Brent-precision root

# Heston diffusion at the equilibrium variance theta_v = 0.04.
_THETA_V = 0.04
_XI = 0.3
_SS_CANONICAL: NDArray[np.float64] = np.diag([_THETA_V, (_XI**2) * _THETA_V, 0.0]).astype(
    np.float64
)


@dataclass(frozen=True)
class ExcessEntropyExperimentConfig:
    """Configuration for the excess-entropy scan."""

    kappa_min: float = 1e-4
    kappa_max_offset: float = 1e-6  # κ_max = κ★ - kappa_max_offset
    n_kappa: int = 101
    taus: tuple[float, ...] = (0.1, 1.0, 5.0)


def _jacobian_at(kappa: float) -> NDArray[np.float64]:
    """Jacobian under the §4.2 constant-vol surrogate (∂_v σ² = 0)."""
    a = kappa * _G_X
    b = kappa * _G_V
    return jacobian_3d(
        kappa=kappa,
        a_kappa=a,
        b_kappa=b,
        G_z=_G_Z,
        kappa_v=_KAPPA_V,
        alpha=_ALPHA,
        beta=_BETA,
        gamma=_GAMMA,
    )


def run(
    cfg: ExcessEntropyExperimentConfig,
) -> tuple[dict[str, object], dict[str, NDArray[np.float64]], NDArray[np.float64]]:
    """Run the excess-entropy scan and return (metrics, curves, kappa_grid)."""
    kappa_max = _KAPPA_STAR - cfg.kappa_max_offset
    kappa_grid = np.linspace(cfg.kappa_min, kappa_max, cfg.n_kappa).astype(np.float64)

    per_tau: dict[str, dict[str, object]] = {}
    curves: dict[str, NDArray[np.float64]] = {}
    with timed("info_theoretic_scan"):
        for tau in cfg.taus:
            curve = excess_entropy_curve(kappa_grid, _jacobian_at, _SS_CANONICAL, tau)
            fit = fit_critical_exponent(_jacobian_at, _SS_CANONICAL, tau, _KAPPA_STAR)
            e_first = float(curve.excess_entropy[0])
            e_last = float(curve.excess_entropy[-1])
            per_tau[f"tau_{tau}"] = {
                "tau": tau,
                "E_first": e_first,
                "E_last": e_last,
                "enhancement_ratio": e_last / e_first if e_first > 0 else float("inf"),
                "is_monotone": bool(curve.is_monotone),
                "beta_fit": float(fit.beta),
                "e_infinity_fit": float(fit.e_infinity),
                "coefficient_fit": float(fit.coefficient),
                "residual_std_fit": float(fit.residual_std),
                "n_fit_points": int(fit.n_points),
            }
            curves[f"tau_{tau}"] = curve.excess_entropy

    metrics: dict[str, object] = {
        "kappa_star_paper": _KAPPA_STAR,
        "kappa_min": cfg.kappa_min,
        "kappa_max": kappa_max,
        "n_kappa": cfg.n_kappa,
        "per_tau": per_tau,
    }

    print(f"  κ★ (paper)             = {_KAPPA_STAR:.6f}")
    for stats in per_tau.values():
        tau_val = stats["tau"]
        print(
            f"  τ={tau_val!s:>3}: E(κ_min)={stats['E_first']:.3e}  "
            f"E(κ_max)={stats['E_last']:.4f}  "
            f"monotone={stats['is_monotone']}  "
            f"β̂={stats['beta_fit']:.4f}  "
            f"E_∞={stats['e_infinity_fit']:.5f}"
        )
    return metrics, curves, kappa_grid


def render_figure(
    kappa_grid: NDArray[np.float64],
    curves: dict[str, NDArray[np.float64]],
    out_path: str,
) -> None:
    """Render E_tau(κ) vs κ for each τ, with log-log inset showing the β=1 fit."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    tau_values = sorted(float(k.removeprefix("tau_")) for k in curves)
    colors = ["C0", "C1", "C2"]
    for tau, color in zip(tau_values, colors, strict=False):
        key = f"tau_{tau}"
        e = curves[key]
        ax.plot(
            kappa_grid,
            e,
            color=color,
            linewidth=1.5,
            label=rf"$\tau = {tau:.1f}$ yr",
        )

    ax.axvline(
        _KAPPA_STAR,
        color="C3",
        linestyle=":",
        linewidth=1.2,
        label=rf"$\kappa^\star = {_KAPPA_STAR:.4f}$",
    )
    ax.set_xlabel(r"$\kappa$ (reflexive coupling)")
    ax.set_ylabel(r"$E_\tau(\kappa)$ (nats)")
    ax.set_title(
        r"Excess entropy $E_\tau(\kappa) = I(\mathcal{F}_{(-\infty,0]}^y;\, R_\tau \mid v_0, z_0)$"
        "\n"
        r"(Theorem 5, paper §3.10 — finite saturation at $\kappa^\star$, mean-field $\beta = 1$)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # Log-log inset: (E_inf - E) vs (κ★ - κ) for τ = 1 yr, showing β=1 slope.
    inset = fig.add_axes((0.55, 0.18, 0.32, 0.30))
    tau_inset = 1.0
    e = curves[f"tau_{tau_inset}"]
    finite = ~np.isnan(e)
    k_fin = kappa_grid[finite]
    e_fin = e[finite]
    e_inf = float(e_fin[-1]) + 1e-6  # tiny slack so log(gap) is finite at rightmost
    deltas = _KAPPA_STAR - k_fin
    gaps = e_inf - e_fin
    valid = (deltas > 0) & (gaps > 0)
    inset.loglog(deltas[valid], gaps[valid], color="C1", marker="o", markersize=3, linewidth=0.8)
    # Reference line slope = 1
    ref_x = np.array([deltas[valid].min(), deltas[valid].max()])
    ref_y = ref_x * (gaps[valid].max() / deltas[valid].max())
    inset.loglog(ref_x, ref_y, color="gray", linestyle="--", linewidth=0.8, label=r"slope = 1")
    inset.set_xlabel(r"$\kappa^\star - \kappa$", fontsize=8)
    inset.set_ylabel(r"$E_\infty - E_\tau(\kappa)$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.set_title(rf"$\tau = {tau_inset:.0f}$ yr boundary scaling", fontsize=8)
    inset.grid(True, which="both", alpha=0.3)
    inset.legend(fontsize=7, loc="upper left")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"OK: excess-entropy figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduce n_kappa for CI smoke")
    args = parser.parse_args()

    cfg = (
        ExcessEntropyExperimentConfig(n_kappa=21) if args.quick else ExcessEntropyExperimentConfig()
    )

    run_dir = make_run_dir("info_theoretic_excess_entropy")
    save_config(run_dir, cfg)

    metrics, curves, kappa_grid = run(cfg)
    metrics["config"] = asdict(cfg)
    save_metrics(run_dir, metrics)

    npz_payload: dict[str, NDArray[np.float64]] = {"kappa_grid": kappa_grid}
    for key, arr in curves.items():
        npz_payload[key] = arr
    # mypy sees np.savez's second positional arg as `bool` due to a stale
    # numpy stub edge case; ignore to keep mypy strict without runtime change.
    np.savez(run_dir / "excess_entropy_curves.npz", **npz_payload)  # type: ignore[arg-type]

    out_path = FIGURES_DIR / "excess_entropy_curve.pdf"
    render_figure(kappa_grid, curves, str(out_path))

    print(f"Wrote results to: {run_dir}")


if __name__ == "__main__":
    main()
