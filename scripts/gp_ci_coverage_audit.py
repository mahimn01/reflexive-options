"""GP-CI coverage audit — 4 methods × 5 truths × 200 reps.

This is the deliverable-1 audit from the v0.3.2 hardening pass: it compares
four candidate replacements for the v0.3.1 GP-with-pinned-noise CI on the
basket of synthetic truths the V3 / G3 audits flagged.

Methods:
  (a) gp_pinned_rbf  — RBF kernel + pinned WhiteKernel noise (the v0.3.1 default).
  (b) gp_inflated    — gp_pinned_rbf with a basket-calibrated inflation factor κ_inf.
  (c) bca_bootstrap  — local-quadratic point + BCa bootstrap CI.
  (d) gp_matern32    — Matérn-3/2 kernel + pinned WhiteKernel (the v0.3.2 default).

Truths (each defined on the 9-point grid x ∈ linspace(0, 1, 9), anchor x = 0.5):
  quadratic   f(x) = x²,             slope = 2 x at 0.5 = 1.0
  linear      f(x) = 0.5 x,           slope = 0.5
  sin         f(x) = sin(2π x),       slope = 2π cos(π) = −6.283
  quintic     f(x) = x⁵,              slope = 5 x⁴ at 0.5 = 0.3125
  kinked      f(x) = max(x − 0.5, 0), slope ∈ {0, 1} (interval truth)

Each rep evaluates the truth at the 9 grid points with σ_n = 0.10 Gaussian noise.
Coverage = fraction of 200 reps where the 95% CI contains the truth.

Outputs:
  runs/gp_ci_coverage/<ts>/coverage_table.json
  runs/gp_ci_coverage/<ts>/config.json
  prints the 4×5 table to stdout

Run: ``python scripts/gp_ci_coverage_audit.py``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from reflexive_options.theory.sensitivity import _local_quadratic_slope  # noqa: E402

GRID = np.linspace(0.0, 1.0, 9, dtype=np.float64)
X_ANCHOR = 0.5
SIGMA_NOISE = 0.10
N_REPS = 200
N_BOOT = 1_000


@dataclass(frozen=True)
class AuditConfig:
    grid: tuple[float, ...] = tuple(GRID.tolist())
    x_anchor: float = X_ANCHOR
    sigma_noise: float = SIGMA_NOISE
    n_reps: int = N_REPS
    n_boot: int = N_BOOT
    seed: int = 20260514
    n_seeds: int = 1


TRUTHS: dict[
    str, tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], float | tuple[float, float]]
] = {
    "quadratic": (lambda x: x**2, 2.0 * X_ANCHOR),
    "linear": (lambda x: 0.5 * x, 0.5),
    "sin": (lambda x: np.sin(2.0 * np.pi * x), 2.0 * np.pi * np.cos(2.0 * np.pi * X_ANCHOR)),
    "quintic": (lambda x: x**5, 5.0 * X_ANCHOR**4),
    "kinked": (lambda x: np.maximum(x - X_ANCHOR, 0.0), (0.0, 1.0)),
}


def _fit_gp(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    kernel_choice: str,
    *,
    noise_variance_raw: float,
) -> tuple[float, float]:
    """Fit GP with the chosen kernel and pinned WhiteKernel noise; return (slope, SE) at x_anchor.

    - RBF uses the closed-form derivative posterior (analytical).
    - Matern-3/2 uses centred-difference derivative on the GP posterior mean.
    """
    x_scale = float(np.max(np.abs(x))) if np.max(np.abs(x)) > 0 else 1.0
    x_s = (x / x_scale).astype(np.float64)
    x_anchor_s = float(X_ANCHOR / x_scale)
    y_mean = float(np.mean(y))
    y_centred = y - y_mean
    y_std = float(np.std(y_centred, ddof=1))
    if y_std == 0.0:
        return 0.0, 0.0
    y_norm = y_centred / y_std

    grid_span = float(np.max(x_s) - np.min(x_s))
    pinned = max(noise_variance_raw / (y_std**2), 1e-10)
    white = WhiteKernel(noise_level=pinned, noise_level_bounds="fixed")

    if kernel_choice == "rbf":
        signal = RBF(
            length_scale=grid_span / 4.0,
            length_scale_bounds=(grid_span * 1e-2, grid_span * 1e2),
        )
    elif kernel_choice == "matern32":
        signal = Matern(
            length_scale=grid_span / 4.0,
            length_scale_bounds=(grid_span * 1e-2, grid_span * 1e2),
            nu=1.5,
        )
    else:
        raise ValueError(kernel_choice)

    kernel = signal + white
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp.fit(x_s.reshape(-1, 1), y_norm)

    if kernel_choice == "rbf":
        # Closed-form RBF derivative posterior
        ell = float(gp.kernel_.k1.length_scale)
        sigma2 = float(gp.kernel_.k2.noise_level)
        diff = x_anchor_s - x_s
        K = np.exp(-0.5 * (diff[:, None] - diff[None, :]) ** 2 / ell**2)
        # ^ wrong, see below
        # build kernel matrices on x_s correctly:
        K = np.exp(-0.5 * ((x_s[:, None] - x_s[None, :]) / ell) ** 2)
        K_jitter = K + sigma2 * np.eye(len(x_s)) + 1e-12 * np.eye(len(x_s))
        k_anchor = np.exp(-0.5 * ((x_anchor_s - x_s) / ell) ** 2)
        k_grad = -((x_anchor_s - x_s) / ell**2) * k_anchor
        try:
            L = np.linalg.cholesky(K_jitter)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            v = np.linalg.solve(L, k_grad)
            var_norm = float(1.0 / ell**2 - v @ v)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K_jitter)
            alpha = K_inv @ y_norm
            var_norm = float(1.0 / ell**2 - k_grad @ K_inv @ k_grad)
        slope_norm = float(k_grad @ alpha)
        var_norm = max(var_norm, 0.0)
    else:
        h = max(grid_span * 1e-3, 1e-8)
        x_eval = np.array([[x_anchor_s - h], [x_anchor_s + h]])
        mean, cov = gp.predict(x_eval, return_cov=True)
        slope_norm = float((mean[1] - mean[0]) / (2.0 * h))
        var_norm = float((cov[1, 1] + cov[0, 0] - 2.0 * cov[0, 1]) / (4.0 * h * h))
        var_norm = max(var_norm, 0.0)

    slope = slope_norm * y_std / x_scale
    se = float(np.sqrt(var_norm)) * y_std / x_scale
    return slope, se


def _bca_bootstrap(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    n_boot: int,
) -> tuple[float, float, float]:
    n = len(x)
    slope_point, _ = _local_quadratic_slope(x, y, X_ANCHOR)
    boot_slopes = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            sb, _ = _local_quadratic_slope(x[idx], y[idx], X_ANCHOR)
        except Exception:
            sb = np.nan
        boot_slopes[b] = sb
    boot_slopes = boot_slopes[np.isfinite(boot_slopes)]
    if len(boot_slopes) < 50:
        return slope_point, float("nan"), float("nan")

    p0 = float(np.mean(boot_slopes < slope_point))
    p0 = min(max(p0, 1e-3), 1 - 1e-3)
    z0 = float(norm.ppf(p0))

    jk = np.empty(n, dtype=np.float64)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        try:
            s_jk, _ = _local_quadratic_slope(x[keep], y[keep], X_ANCHOR)
        except Exception:
            s_jk = slope_point
        jk[i] = s_jk
    jk_mean = float(jk.mean())
    num = float(np.sum((jk_mean - jk) ** 3))
    den = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
    a_hat = num / den if den > 0 else 0.0

    z_lo, z_hi = norm.ppf(0.025), norm.ppf(0.975)
    a1 = norm.cdf(z0 + (z0 + z_lo) / max(1.0 - a_hat * (z0 + z_lo), 1e-6))
    a2 = norm.cdf(z0 + (z0 + z_hi) / max(1.0 - a_hat * (z0 + z_hi), 1e-6))
    a1 = min(max(float(a1), 0.0), 1.0)
    a2 = min(max(float(a2), 0.0), 1.0)
    ci_low = float(np.quantile(boot_slopes, a1))
    ci_high = float(np.quantile(boot_slopes, a2))
    return slope_point, ci_low, ci_high


def _covered(true_slope, ci_low: float, ci_high: float) -> bool:
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return False
    if isinstance(true_slope, tuple):
        lo, hi = true_slope
        return not (ci_high < lo or ci_low > hi)
    return ci_low <= true_slope <= ci_high


def calibrate_kappa_inf(cfg: AuditConfig) -> float:
    """Calibrate κ_inf so the empirical |z| 95th-percentile across the smooth basket = 1.96."""
    rng = np.random.default_rng(cfg.seed)
    z_pool: list[float] = []
    for name in ("linear", "quadratic", "sin", "quintic"):
        f, ts = TRUTHS[name]
        for _ in range(50):
            noise = rng.standard_normal(len(GRID)) * cfg.sigma_noise
            y = f(GRID) + noise
            try:
                s, se = _fit_gp(GRID, y, "rbf", noise_variance_raw=cfg.sigma_noise**2)
                if se > 1e-15 and not isinstance(ts, tuple):
                    z_pool.append(abs((s - ts) / se))
            except Exception:
                pass
    z = np.asarray(z_pool, dtype=np.float64)
    q = float(np.quantile(z, 0.95))
    return max(q / 1.959964, 1.0)


def _run_single_seed(
    cfg: AuditConfig, *, seed: int, kappa_inf: float
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for truth_name in ("quadratic", "linear", "sin", "quintic", "kinked"):
        f, ts = TRUTHS[truth_name]
        cov_pin = 0
        cov_inf = 0
        cov_bca = 0
        cov_mat = 0
        rng = np.random.default_rng(seed + hash(truth_name) % 1_000_000)
        for _ in range(cfg.n_reps):
            noise = rng.standard_normal(len(GRID)) * cfg.sigma_noise
            y = f(GRID) + noise

            try:
                s, se = _fit_gp(GRID, y, "rbf", noise_variance_raw=cfg.sigma_noise**2)
                if _covered(ts, s - 1.959964 * se, s + 1.959964 * se):
                    cov_pin += 1
                if _covered(ts, s - 1.959964 * se * kappa_inf, s + 1.959964 * se * kappa_inf):
                    cov_inf += 1
            except Exception:
                pass

            try:
                s, se = _fit_gp(GRID, y, "matern32", noise_variance_raw=cfg.sigma_noise**2)
                if _covered(ts, s - 1.959964 * se, s + 1.959964 * se):
                    cov_mat += 1
            except Exception:
                pass

            try:
                _, lo_b, hi_b = _bca_bootstrap(
                    GRID, y, np.random.default_rng(rng.integers(0, 2**31)), n_boot=cfg.n_boot
                )
                if _covered(ts, lo_b, hi_b):
                    cov_bca += 1
            except Exception:
                pass

        table[truth_name] = {
            "gp_pinned_rbf": cov_pin / cfg.n_reps,
            "gp_inflated": cov_inf / cfg.n_reps,
            "bca_bootstrap": cov_bca / cfg.n_reps,
            "gp_matern32": cov_mat / cfg.n_reps,
        }
    return table


def run_audit(cfg: AuditConfig) -> dict[str, object]:
    print(
        f"σ_noise = {cfg.sigma_noise}, n_reps = {cfg.n_reps}, n_seeds = {cfg.n_seeds}, grid = 9 pts on [0,1]"
    )
    kappa_inf = calibrate_kappa_inf(cfg)
    print(f"Calibrated κ_inf for inflated method: {kappa_inf:.3f}")

    methods = ["gp_pinned_rbf", "gp_inflated", "bca_bootstrap", "gp_matern32"]
    truth_names = ("quadratic", "linear", "sin", "quintic", "kinked")

    per_seed: list[dict[str, dict[str, float]]] = []
    for s_idx in range(cfg.n_seeds):
        seed = cfg.seed + s_idx
        print(f"\n[seed {s_idx + 1}/{cfg.n_seeds} = {seed}]")
        sub = _run_single_seed(cfg, seed=seed, kappa_inf=kappa_inf)
        per_seed.append(sub)
        for tname in truth_names:
            print(f"  {tname:10s}: ", sub[tname])

    # Aggregate across seeds: mean and (min, max) per (truth, method).
    table_mean: dict[str, dict[str, float]] = {}
    table_min: dict[str, dict[str, float]] = {}
    table_max: dict[str, dict[str, float]] = {}
    for tname in truth_names:
        table_mean[tname] = {}
        table_min[tname] = {}
        table_max[tname] = {}
        for m in methods:
            vals = [sub[tname][m] for sub in per_seed]
            arr = np.asarray(vals, dtype=np.float64)
            table_mean[tname][m] = float(arr.mean())
            table_min[tname][m] = float(arr.min())
            table_max[tname][m] = float(arr.max())

    print(
        f"\n=== Mean Coverage Across {cfg.n_seeds} Seeds "
        f"(rows=truth, cols=method, n_reps={cfg.n_reps}) ==="
    )
    print(f"{'truth':12s} " + " ".join(f"{m:>16s}" for m in methods))
    for t in truth_names:
        print(f"{t:12s} " + " ".join(f"{table_mean[t][m]:>16.3f}" for m in methods))

    if cfg.n_seeds > 1:
        print("\n=== (min, max) range across seeds ===")
        print(f"{'truth':12s} " + " ".join(f"{m:>16s}" for m in methods))
        for t in truth_names:
            cells = [f"({table_min[t][m]:.2f},{table_max[t][m]:.2f})".rjust(16) for m in methods]
            print(f"{t:12s} " + " ".join(cells))

    print("\n#truths with mean coverage ≥ 0.90:")
    for m in methods:
        n_ok = sum(1 for t in truth_names if table_mean[t][m] >= 0.90)
        print(f"  {m:18s} {n_ok}/5")

    return {
        "kappa_inf": kappa_inf,
        "table": table_mean,
        "table_min": table_min,
        "table_max": table_max,
        "per_seed": per_seed,
    }


def main() -> None:
    from datetime import UTC, datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA_NOISE,
        help=f"Gaussian observation noise σ_n (default {SIGMA_NOISE}).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of top-level seeds to average over (default 1).",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=N_REPS,
        help=f"Replicates per (seed, truth, method) (default {N_REPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260514,
        help="Base seed; per-seed offsets are seed+0, seed+1, ...",
    )
    args = parser.parse_args()

    cfg = AuditConfig(
        sigma_noise=float(args.sigma),
        n_seeds=int(args.n_seeds),
        n_reps=int(args.n_reps),
        seed=int(args.seed),
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = f"sigma{cfg.sigma_noise:.2f}_seeds{cfg.n_seeds}"
    run_dir = os.path.join(REPO_ROOT, "runs", "gp_ci_coverage", f"{timestamp}_{tag}")
    os.makedirs(run_dir, exist_ok=True)

    payload = run_audit(cfg)

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    with open(os.path.join(run_dir, "coverage_table.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nResults written to {run_dir}")


if __name__ == "__main__":
    main()
