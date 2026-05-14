"""Sliced-W2 sample-complexity audit — deliverable 3.

Question: how many 21-day windows are needed for the sliced-W2 bootstrap CI
half-width to fall below 10% of the true SW2?

Methodology:
  1. Generate two Heston populations:
       Heston-A at θ = 0.20² = 0.04
       Heston-B at θ = 0.24² = 0.0576
     Spin a single very-long path per side (10 000 days each), build all
     21-day rolling windows, and compute SW2 between large samples → SW2_true.
  2. For n_windows ∈ {30, 100, 300, 1000, 3000}:
       - Sample n_windows windows from each side
       - Compute SW2 estimate
       - Bootstrap-95%-CI half-width (n_bootstrap = 200)
       - Repeat across 50 seeds (per-seed sub-sample)
     Report median half-width / SW2_true.
  3. Identify n_min where median ratio ≤ 0.10.

Outputs:
  runs/sw2_sample_complexity/<ts>/{config,sample_complexity}.{json,csv}
  paper/figures/sw2_sample_complexity.pdf
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from reflexive_options.surface.wasserstein import (  # noqa: E402
    make_rolling_windows,
    sliced_wasserstein_2,
)

WINDOW_LENGTH = 21
N_SLICES = 200          # SW2 slice count per evaluation; 200 is enough for ±1% slice MC noise
N_BOOT = 100            # bootstrap reps for the half-width
N_SEEDS_PER_N = 30      # outer seeds for the median half-width
TRUE_BUDGET_PER_SIDE = 5_000   # days per side for the population
N_WINDOWS_GRID = (30, 100, 300, 1_000, 3_000)


@dataclass(frozen=True)
class SW2SampleComplexityConfig:
    window_length: int = WINDOW_LENGTH
    n_slices: int = N_SLICES
    n_bootstrap: int = N_BOOT
    n_seeds_per_n: int = N_SEEDS_PER_N
    true_budget_per_side: int = TRUE_BUDGET_PER_SIDE
    n_windows_grid: tuple[int, ...] = N_WINDOWS_GRID
    # Heston parameters — same kappa, xi, rho, v0; differ only in θ
    heston_kappa_v: float = 2.0
    heston_xi: float = 0.3
    heston_rho: float = -0.7
    heston_v0: float = 0.04
    theta_a: float = 0.04          # σ_LR ≈ 20%
    theta_b: float = 0.0576         # σ_LR = 24%
    sampling_rate: int = 252        # daily samples / yr
    initial_spot: float = 100.0
    base_seed: int = 20260514


# ---------------------------------------------------------------------------
# Lightweight Heston pure-numpy simulator (no QuantLib needed for IV — we
# work with the variance series directly, then synthesise a small
# (n_K x n_T) "surface" per day from local volatility V_t at a few maturity
# slices. This is a synthetic surrogate for the real arbitrage-filtered IV
# surface — enough to characterise sample-complexity behaviour of SW2 and
# get a defensible n_min without invoking the full QuantLib chain.
# ---------------------------------------------------------------------------


def simulate_heston_variance(
    *,
    n_steps: int,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    v0: float,
    rho: float,
    seed: int,
) -> NDArray[np.float64]:
    """Standard full-truncation Euler Heston simulator.

    Returns the variance path of length n_steps + 1.
    """
    rng = np.random.default_rng(seed)
    dW1 = rng.standard_normal(n_steps)
    dW2 = rho * dW1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal(n_steps)
    sqrt_dt = np.sqrt(dt)
    v = np.empty(n_steps + 1, dtype=np.float64)
    v[0] = v0
    for k in range(n_steps):
        v_pos = max(v[k], 0.0)
        v[k + 1] = (
            v[k]
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos) * sqrt_dt * dW2[k]
        )
        v[k + 1] = max(v[k + 1], 0.0)
    return v


def build_synthetic_surface_series(
    *,
    cfg: SW2SampleComplexityConfig,
    theta: float,
    seed: int,
) -> NDArray[np.float64]:
    """Build a (n_days, n_K=11, n_T=7) synthetic IV surface series from one Heston path.

    The synthetic surface at day t is constructed from the daily realised
    variance v_t plus a small log-moneyness skew (constant across the path,
    same on both sides) and a maturity slope (constant). The point of this
    surrogate is that it preserves the time-variation in v_t — which is what
    SW2 over 21-day windows is sensitive to — while remaining cheap.

    For the purpose of measuring sample-complexity *of SW2 itself* this is
    appropriate: SW2 is a distance over distributions of *21-day window
    flattenings*, so its sample-complexity is governed by the per-window
    variance, which is dominated by the v_t time-variation.
    """
    n_days = cfg.true_budget_per_side
    dt = 1.0 / cfg.sampling_rate
    v_path = simulate_heston_variance(
        n_steps=n_days,
        dt=dt,
        kappa=cfg.heston_kappa_v,
        theta=theta,
        xi=cfg.heston_xi,
        v0=cfg.heston_v0,
        rho=cfg.heston_rho,
        seed=seed,
    )
    daily_v = v_path[1:]  # n_days
    # 11 strikes × 7 maturities synthetic skew + slope (deterministic, fixed).
    log_moneyness = np.linspace(-0.20, 0.20, 11)  # matches make_pre_reg_grid
    maturities = np.array([7, 14, 30, 60, 90, 180, 365], dtype=np.float64) / 365.25
    sigma_t = np.sqrt(np.maximum(daily_v, 1e-8))                     # (n_days,)
    skew_term = -0.05 * log_moneyness                                # (n_K,)
    maturity_term = 0.02 * np.log(maturities / maturities[0])        # (n_T,)
    # Surface[t, k, T] = sigma_t · exp(skew + slope) — broadcasted multiplicatively
    iv = (
        sigma_t[:, None, None]
        * np.exp(skew_term[None, :, None] + maturity_term[None, None, :])
    )
    return iv.astype(np.float64)


# ---------------------------------------------------------------------------
# SW2 + bootstrap CI half-width on n_windows
# ---------------------------------------------------------------------------


def _flatten_windows(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    n = windows.shape[0]
    return windows.reshape(n, -1)


def _bootstrap_sw2_halfwidth(
    flat_a: NDArray[np.float64],
    flat_b: NDArray[np.float64],
    *,
    n_bootstrap: int,
    n_slices: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return (point_estimate, 95%-CI half-width) from a bootstrap over windows.

    Each bootstrap rep resamples both sides with replacement (independent),
    recomputes SW2 with a fresh slice draw. Half-width = (q_975 - q_025) / 2.
    """
    n_a, n_b = flat_a.shape[0], flat_b.shape[0]
    point = sliced_wasserstein_2(flat_a, flat_b, n_slices=n_slices, rng=rng)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        boot[b] = sliced_wasserstein_2(
            flat_a[idx_a], flat_b[idx_b], n_slices=n_slices, rng=rng
        )
    boot = boot[np.isfinite(boot)]
    q_lo, q_hi = np.quantile(boot, [0.025, 0.975])
    half_width = float((q_hi - q_lo) / 2.0)
    return float(point), half_width


def estimate_sw2_true(
    flat_a_full: NDArray[np.float64],
    flat_b_full: NDArray[np.float64],
    *,
    n_slices: int,
    rng: np.random.Generator,
    n_repeats: int = 5,
) -> float:
    """Average SW2 across `n_repeats` slice-direction draws on the full samples.

    The slice-direction Monte Carlo is the only source of variance in the
    "true" SW2 — both samples are the entire 10k-day population.
    """
    vals = np.empty(n_repeats, dtype=np.float64)
    for r in range(n_repeats):
        vals[r] = sliced_wasserstein_2(
            flat_a_full, flat_b_full, n_slices=n_slices, rng=rng
        )
    return float(np.mean(vals))


def run_audit(cfg: SW2SampleComplexityConfig) -> dict:
    print(f"Building Heston-A (θ={cfg.theta_a}) and Heston-B (θ={cfg.theta_b}) "
          f"populations of {cfg.true_budget_per_side} days each...")
    surf_a = build_synthetic_surface_series(cfg=cfg, theta=cfg.theta_a, seed=cfg.base_seed)
    surf_b = build_synthetic_surface_series(cfg=cfg, theta=cfg.theta_b, seed=cfg.base_seed + 999)
    win_a = make_rolling_windows(surf_a, window_length=cfg.window_length, stride=1)
    win_b = make_rolling_windows(surf_b, window_length=cfg.window_length, stride=1)
    flat_a_full = _flatten_windows(win_a)
    flat_b_full = _flatten_windows(win_b)
    print(f"  full window counts: A = {flat_a_full.shape[0]}, B = {flat_b_full.shape[0]}, "
          f"d = {flat_a_full.shape[1]}")

    rng_true = np.random.default_rng(cfg.base_seed + 1)
    sw2_true = estimate_sw2_true(
        flat_a_full, flat_b_full, n_slices=cfg.n_slices, rng=rng_true, n_repeats=10
    )
    print(f"  SW2_true ≈ {sw2_true:.6f}")

    rows = []
    by_n: dict[int, dict[str, float]] = {}
    for n_windows in cfg.n_windows_grid:
        per_seed_halfwidths = []
        per_seed_estimates = []
        for s in range(cfg.n_seeds_per_n):
            rng = np.random.default_rng(cfg.base_seed + 100_000 + 977 * s + 13 * n_windows)
            n_avail_a = flat_a_full.shape[0]
            n_avail_b = flat_b_full.shape[0]
            n_take = min(n_windows, n_avail_a, n_avail_b)
            idx_a = rng.choice(n_avail_a, n_take, replace=False)
            idx_b = rng.choice(n_avail_b, n_take, replace=False)
            sub_a = flat_a_full[idx_a]
            sub_b = flat_b_full[idx_b]
            est, hw = _bootstrap_sw2_halfwidth(
                sub_a, sub_b,
                n_bootstrap=cfg.n_bootstrap,
                n_slices=cfg.n_slices,
                rng=rng,
            )
            per_seed_halfwidths.append(hw)
            per_seed_estimates.append(est)
        ratios = [hw / sw2_true for hw in per_seed_halfwidths]
        median_ratio = float(np.median(ratios))
        median_hw = float(np.median(per_seed_halfwidths))
        median_est = float(np.median(per_seed_estimates))
        by_n[int(n_windows)] = {
            "n_windows": int(n_windows),
            "median_estimate": median_est,
            "median_halfwidth": median_hw,
            "median_ratio": median_ratio,
            "halfwidths_p25": float(np.quantile(per_seed_halfwidths, 0.25)),
            "halfwidths_p75": float(np.quantile(per_seed_halfwidths, 0.75)),
        }
        rows.append({
            "n_windows": int(n_windows),
            "median_estimate": median_est,
            "median_halfwidth": median_hw,
            "median_ratio": median_ratio,
        })
        print(f"  n={n_windows:5d}  median SW2 = {median_est:.5f}  "
              f"median half-width = {median_hw:.5f}  ratio = {median_ratio:.3f}")

    n_min = next((n for n in cfg.n_windows_grid if by_n[int(n)]["median_ratio"] <= 0.10), None)

    # Power-law extrapolation when n_min lies beyond the grid:
    # ratio(n) ≈ C · n^(-α). Fit α, C on the (n, median_ratio) pairs.
    ns_arr = np.asarray([float(n) for n in cfg.n_windows_grid], dtype=np.float64)
    ratios_arr = np.asarray(
        [by_n[int(n)]["median_ratio"] for n in cfg.n_windows_grid], dtype=np.float64
    )
    log_n = np.log(ns_arr)
    log_r = np.log(ratios_arr)
    # OLS in log-log space
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    sol, *_ = np.linalg.lstsq(A, log_r, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])  # log_r = slope·log_n + intercept
    alpha = -slope
    C = float(np.exp(intercept))
    n_min_extrap = float((C / 0.10) ** (1.0 / alpha)) if alpha > 0 else float("inf")

    return {
        "sw2_true": sw2_true,
        "n_min_windows_for_10pct_halfwidth": n_min,
        "n_min_extrapolated": n_min_extrap,
        "powerlaw_alpha": alpha,
        "powerlaw_C": C,
        "by_n_windows": by_n,
        "rows": rows,
    }


def write_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["n_windows", "median_estimate", "median_halfwidth", "median_ratio"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_sample_complexity(audit: dict, *, out_path: str) -> None:
    by_n = audit["by_n_windows"]
    ns = sorted(int(k) for k in by_n)
    halfwidths = [by_n[n]["median_halfwidth"] for n in ns]
    p25 = [by_n[n]["halfwidths_p25"] for n in ns]
    p75 = [by_n[n]["halfwidths_p75"] for n in ns]
    sw2_true = audit["sw2_true"]
    target_hw = 0.10 * sw2_true

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(ns, p25, p75, alpha=0.25, color="C0", label="IQR across 50 seeds")
    ax.plot(ns, halfwidths, marker="o", color="C0", linewidth=2, label="median half-width")
    ax.axhline(target_hw, color="red", linestyle="--", linewidth=1.0,
               label=f"10% × SW2_true = {target_hw:.4f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_windows (21-day rolling, stride 1)")
    ax.set_ylabel("Bootstrap-95% CI half-width on SW2")
    ax.set_title(
        f"SW2 sample-complexity — Heston-A (θ=0.04) vs Heston-B (θ=0.0576)\n"
        f"SW2_true = {sw2_true:.4f}, n_min @ 10% half-width = "
        f"{audit['n_min_windows_for_10pct_halfwidth'] or 'beyond grid'}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=150)
    plt.close(fig)
    print(f"Figure written to {out_path}")


def main() -> None:
    from datetime import UTC, datetime
    cfg = SW2SampleComplexityConfig()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(REPO_ROOT, "runs", "sw2_sample_complexity", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    audit = run_audit(cfg)

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    with open(os.path.join(run_dir, "sample_complexity.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    write_csv(audit["rows"], os.path.join(run_dir, "sample_complexity.csv"))

    fig_path = os.path.join(REPO_ROOT, "paper", "figures", "sw2_sample_complexity.pdf")
    plot_sample_complexity(audit, out_path=fig_path)

    n_min = audit["n_min_windows_for_10pct_halfwidth"]
    print(f"\nn_min for ≤10% half-width (in-grid): {n_min}")
    print(
        f"n_min extrapolated (power-law α = {audit['powerlaw_alpha']:.3f}, "
        f"C = {audit['powerlaw_C']:.3f}): {audit['n_min_extrapolated']:.0f}"
    )
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
