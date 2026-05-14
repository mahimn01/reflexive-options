"""Numerical validation of the supercritical limit cycle past κ*.

Theorem 1 conclusion (1) of paper §3 / theory.md §4 asserts that for κ
slightly past κ* with ℓ_1 < 0 (supercritical), the deterministic skeleton
admits a unique attracting limit cycle of period

    T_κ = 2π/ω* + O(κ - κ*)

and amplitude proportional to √(κ - κ*). At the §4.2 canonical regime
(κ* ≈ 0.8964, ω* ≈ 0.5724 rad/yr, ℓ_1 ≈ −0.0253), the predicted period is
T_κ ≈ 2π / 0.5724 ≈ 10.98 yr.

This experiment integrates the noiseless 3D ODE at κ = 1.05 · κ* via
scipy.integrate.solve_ivp from a small generic initial condition, lets
transients decay over many predicted periods, then estimates the cycle
period from zero-crossings of the projected y-component. The 3D phase
trajectory (and three pairwise 2D projections) are saved as a single
PDF figure for the paper.

Invocation::

    python -m reflexive_options.experiments.limit_cycle_supercritical
    python -m reflexive_options.experiments.limit_cycle_supercritical --quick

Outputs:
    runs/limit_cycle_supercritical/<timestamp>/{config,metrics}.json
    paper/figures/limit_cycle_supercritical.pdf
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
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)

# ---------------------------------------------------------------------------
# §4.2 canonical regime — locked to paper/theory.md §4.2 and
# tests/test_paper_section_4_2.py. Convention: σ² = const so ∂_x σ² =
# ∂_v σ² = 0 in the linearisation, with quadratic / cubic Taylor
# coefficients G_xx = -0.1, G_xxx = -0.2 of the dealer-gamma functional.
# ---------------------------------------------------------------------------
_G_X = 0.5
_G_V = -0.5
_G_Z = -0.5
_ALPHA = 0.5
_BETA = 1.0
_GAMMA = 0.5
_KAPPA_V = 2.0
_G_XX = -0.1
_G_XXX = -0.2

_KAPPA_STAR = 0.8964  # paper Table §4.2 (validated by test_paper_section_4_2)
_OMEGA_STAR = 0.5724  # rad / yr
_PERIOD_THEORY = 2.0 * float(np.pi) / _OMEGA_STAR  # ≈ 10.98 yr


@dataclass(frozen=True)
class LimitCycleConfig:
    """Configuration for the supercritical limit-cycle simulation."""

    kappa_multiplier: float = 1.05  # κ = 1.05 · κ*
    t_total: float = 220.0  # ≈ 20 predicted periods
    n_eval: int = 8801  # ≈ 40 samples per period
    transient_periods: int = 10  # discard the first N predicted periods
    rtol: float = 1.0e-9
    atol: float = 1.0e-11
    initial_state: tuple[float, float, float] = (0.5, 0.0, 0.0)


def _drift(x: NDArray[np.float64], *, kappa: float) -> NDArray[np.float64]:
    """Local cubic-truncation drift of the §4.2 deterministic skeleton.

    Variables (y, u, z) := (δ log S, δv, δz). Under σ² = const, f_1 has only
    the κ G(y, u, z) channel, expanded to cubic order in y with G_xx, G_xxx.
    The variance and memory equations are linear (no Taylor truncation).
    """
    y, u, z = float(x[0]), float(x[1]), float(x[2])
    f1 = kappa * (
        _G_X * y + _G_V * u + _G_Z * z + 0.5 * _G_XX * y * y + (1.0 / 6.0) * _G_XXX * y * y * y
    )
    f2 = -_KAPPA_V * u + _GAMMA * z
    f3 = -_ALPHA * z + _BETA * y
    return np.array([f1, f2, f3], dtype=np.float64)


def _estimate_period(times: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Period estimated from upward zero-crossings of y(t).

    Uses linear interpolation between adjacent samples to avoid the sampling
    bias of taking raw sample times. Returns the median gap between
    consecutive upward crossings — robust to small-amplitude wobble at the
    transient edge of the kept window.
    """
    if y.size < 4:
        raise ValueError(f"need at least 4 samples for period estimation, got {y.size}")
    sign = np.sign(y)
    # upward crossings: -1 → +1 transitions (handles the y=0 case as part of the prior step)
    upward = np.where((sign[:-1] <= 0) & (sign[1:] > 0))[0]
    if upward.size < 2:
        raise ValueError(
            f"need at least 2 upward zero-crossings for period; got {upward.size}. "
            "Likely the transient hasn't decayed or the trajectory diverged."
        )
    crossings: list[float] = []
    for k in upward:
        # linear interp: t such that y(t) = 0 between samples k and k+1
        t0, t1 = float(times[k]), float(times[k + 1])
        y0, y1 = float(y[k]), float(y[k + 1])
        if y1 == y0:
            crossings.append(0.5 * (t0 + t1))
        else:
            crossings.append(t0 - y0 * (t1 - t0) / (y1 - y0))
    diffs = np.diff(np.asarray(crossings))
    return float(np.median(diffs))


def run(cfg: LimitCycleConfig) -> dict[str, object]:
    """Integrate the §4.2 skeleton at κ = κ_multiplier · κ* and return metrics."""
    kappa = cfg.kappa_multiplier * _KAPPA_STAR

    def rhs(_t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _drift(x, kappa=kappa)

    t_eval = np.linspace(0.0, cfg.t_total, cfg.n_eval)
    with timed(f"solve_ivp κ={kappa:.4f}"):
        sol = solve_ivp(
            rhs,
            (0.0, cfg.t_total),
            list(cfg.initial_state),
            t_eval=t_eval,
            method="RK45",
            rtol=cfg.rtol,
            atol=cfg.atol,
            max_step=0.5,
        )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    transient_t = cfg.transient_periods * _PERIOD_THEORY
    keep_mask = sol.t >= transient_t
    t_kept = sol.t[keep_mask]
    y_kept = sol.y[0, keep_mask]
    u_kept = sol.y[1, keep_mask]
    z_kept = sol.y[2, keep_mask]

    period_est = _estimate_period(t_kept, y_kept)
    period_rel_err = abs(period_est - _PERIOD_THEORY) / _PERIOD_THEORY

    amp_y = float(0.5 * (np.max(y_kept) - np.min(y_kept)))
    amp_u = float(0.5 * (np.max(u_kept) - np.min(u_kept)))
    amp_z = float(0.5 * (np.max(z_kept) - np.min(z_kept)))

    print(f"  κ = {kappa:.4f}  (= {cfg.kappa_multiplier} · κ*)")
    print(f"  predicted T_κ = 2π/ω* = {_PERIOD_THEORY:.4f} yr")
    print(f"  measured  T_κ        = {period_est:.4f} yr")
    print(f"  relative error       = {period_rel_err:.3%}")
    print(f"  amplitudes (y, u, z) = ({amp_y:.4f}, {amp_u:.4f}, {amp_z:.4f})")

    return {
        "kappa": kappa,
        "kappa_star": _KAPPA_STAR,
        "kappa_multiplier": cfg.kappa_multiplier,
        "omega_star": _OMEGA_STAR,
        "period_theory": _PERIOD_THEORY,
        "period_measured": period_est,
        "period_relative_error": period_rel_err,
        "amplitude_y": amp_y,
        "amplitude_u": amp_u,
        "amplitude_z": amp_z,
        "n_kept_samples": int(t_kept.size),
        "transient_discarded_yr": float(transient_t),
        "trajectory": {
            "t": t_kept,
            "y": y_kept,
            "u": u_kept,
            "z": z_kept,
        },
    }


def render_figure(metrics: dict[str, object], out_path: str) -> None:
    """Render the 3D phase trajectory + three 2D projections."""
    traj = metrics["trajectory"]
    if not isinstance(traj, dict):
        raise TypeError(f"metrics['trajectory'] must be dict, got {type(traj)}")
    y = np.asarray(traj["y"], dtype=np.float64)
    u = np.asarray(traj["u"], dtype=np.float64)
    z = np.asarray(traj["z"], dtype=np.float64)

    fig = plt.figure(figsize=(11.0, 8.5))
    # Top-left: 3D phase
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_3d.plot(y, u, z, color="C0", linewidth=0.6, alpha=0.85)
    ax_3d.scatter([0.0], [0.0], [0.0], color="black", s=18, label="equilibrium")
    ax_3d.set_xlabel(r"$y = \delta \log S$")
    ax_3d.set_ylabel(r"$u = \delta v$")
    ax_3d.set_zlabel(r"$z$")
    ax_3d.set_title(r"3D phase $(y, u, z)$")
    ax_3d.legend(loc="upper left", fontsize=8)

    # 2D projections
    proj_axes = [
        (fig.add_subplot(2, 2, 2), y, u, r"$y$", r"$u$", "(y, u) projection"),
        (fig.add_subplot(2, 2, 3), y, z, r"$y$", r"$z$", "(y, z) projection"),
        (fig.add_subplot(2, 2, 4), u, z, r"$u$", r"$z$", "(u, z) projection"),
    ]
    for ax, x_data, y_data, xlab, ylab, title in proj_axes:
        ax.plot(x_data, y_data, color="C0", linewidth=0.6, alpha=0.85)
        ax.scatter([0.0], [0.0], color="black", s=18)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)
        ax.axvline(0.0, color="grey", linewidth=0.5, alpha=0.5)

    kappa = float(metrics["kappa"])  # type: ignore[arg-type]
    period_meas = float(metrics["period_measured"])  # type: ignore[arg-type]
    period_th = float(metrics["period_theory"])  # type: ignore[arg-type]
    rel_err = float(metrics["period_relative_error"])  # type: ignore[arg-type]
    fig.suptitle(
        rf"Supercritical limit cycle at $\kappa = {kappa:.4f}$ "
        rf"$(= 1.05 \cdot \kappa^\star)$"
        "\n"
        rf"predicted $T_\kappa = 2\pi/\omega^\star = {period_th:.3f}$ yr;  "
        rf"measured $T = {period_meas:.3f}$ yr  ({rel_err:.2%} error)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"OK: limit-cycle figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduce t_total and n_eval for CI")
    args = parser.parse_args()

    if args.quick:
        cfg = LimitCycleConfig(t_total=80.0, n_eval=3201, transient_periods=4)
    else:
        cfg = LimitCycleConfig()

    run_dir = make_run_dir("limit_cycle_supercritical")
    save_config(run_dir, cfg)

    metrics = run(cfg)

    # Persist metrics WITHOUT the trajectory blob (json-unfriendly arrays).
    persisted = {k: v for k, v in metrics.items() if k != "trajectory"}
    persisted["config"] = asdict(cfg)
    save_metrics(run_dir, persisted)
    print(f"Wrote results to: {run_dir}")

    out_path = FIGURES_DIR / "limit_cycle_supercritical.pdf"
    render_figure(metrics, str(out_path))


if __name__ == "__main__":
    main()
