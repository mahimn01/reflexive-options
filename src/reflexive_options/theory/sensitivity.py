"""κ-sensitivity helper.

The κ-sensitivity curve is the central novel evaluation in the paper:
train an RL agent at κ = κ₀, deploy across a family of environments at
varying κ, and plot the metric vs κ. The slope at κ₀ is a quantitative
measure of reflexivity-importance.

This module provides the deterministic numerical sensitivity (no RL agent —
just propagate the simulator). The full RL κ-sensitivity experiment lives
in `experiments/reflexive_transfer.py`.

Reference: evaluation_framework_brief.md §3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline


@dataclass(frozen=True)
class SensitivityResult:
    """κ-sensitivity curve data."""

    kappa_grid: NDArray[np.float64]
    metric_values: NDArray[np.float64]  # metric at each κ
    metric_std: NDArray[np.float64]  # std (e.g. across seeds) at each κ
    kappa_anchor: float  # the training-time κ₀ at which slope is reported
    slope_at_anchor: float  # ∂metric/∂κ at κ₀ — the headline scalar
    slope_ci_low: float  # block-bootstrap lower bound on the slope
    slope_ci_high: float


def kappa_sensitivity_curve(
    metric_fn: Callable[[float, int], float],
    *,
    kappa_grid: NDArray[np.float64],
    kappa_anchor: float,
    n_seeds: int = 100,
    n_bootstrap: int = 1_000,
    rng_seed: int = 42,
) -> SensitivityResult:
    """Sweep κ and produce the κ-sensitivity curve with bootstrap CIs on the slope.

    Args:
        metric_fn: f(kappa, seed) -> scalar metric value.
        kappa_grid: ascending grid of κ values; kappa_anchor must lie strictly inside.
        kappa_anchor: the κ₀ at which we evaluate the slope.
        n_seeds: number of seeds per κ.
        n_bootstrap: number of bootstrap resamples for the slope CI.
    """
    if not np.all(np.diff(kappa_grid) > 0):
        raise ValueError("kappa_grid must be strictly ascending")
    if not kappa_grid[0] < kappa_anchor < kappa_grid[-1]:
        raise ValueError("kappa_anchor must be strictly inside kappa_grid")

    rng = np.random.default_rng(rng_seed)
    n_k = len(kappa_grid)
    raw = np.zeros((n_k, n_seeds), dtype=np.float64)
    for i, k in enumerate(kappa_grid):
        for j in range(n_seeds):
            raw[i, j] = metric_fn(float(k), int(rng.integers(0, 2**31 - 1)))

    means = raw.mean(axis=1)
    stds = raw.std(axis=1, ddof=1)

    spline = UnivariateSpline(kappa_grid, means, k=4, s=0)
    slope = float(spline.derivative()(kappa_anchor))

    boot_slopes = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means = raw[:, idx].mean(axis=1)
        try:
            boot_spline = UnivariateSpline(kappa_grid, boot_means, k=4, s=0)
            boot_slopes[b] = float(boot_spline.derivative()(kappa_anchor))
        except (ValueError, RuntimeError):
            boot_slopes[b] = np.nan

    boot_slopes = boot_slopes[np.isfinite(boot_slopes)]
    ci_low = float(np.percentile(boot_slopes, 2.5))
    ci_high = float(np.percentile(boot_slopes, 97.5))

    return SensitivityResult(
        kappa_grid=kappa_grid,
        metric_values=means,
        metric_std=stds,
        kappa_anchor=kappa_anchor,
        slope_at_anchor=slope,
        slope_ci_low=ci_low,
        slope_ci_high=ci_high,
    )
