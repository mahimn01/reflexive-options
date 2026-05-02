"""Sliced Wasserstein-2 distance over arbitrage-filtered IV-surface windows.

Implements the H1 primary metric of `paper/pre_registration.md` §4 and
`evaluation_framework_brief.md` §2.3:

- Roll the daily IV-surface series into 21-day windows (stride = 1).
- Drop a window if *any* of its daily surfaces fails the static-arbitrage
  filter (`surface/arbitrage.check_arbitrage_free`, `passes_all=True`).
- Vectorise each surviving window into R^{1617} (= 21 × 7 × 11).
- Compute sliced-W2 between two empirical distributions of windows via
  `N_slices` random projections drawn uniformly on the unit sphere; per-slice
  1D W2 is closed-form via sorted quantiles.

Cost is `O(N_slices * n * log n)` with `n` = number of surviving windows.

The pre-reg grid (7 maturities × 11 strikes, Δk = 0.04, k ∈ [-0.20, +0.20]) is
re-exported from `surface.generator.make_pre_reg_grid` for callers that import
it through this module — see `paper/pre_registration.md` §4 for the lock.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reflexive_options.surface.arbitrage import check_arbitrage_free
from reflexive_options.surface.generator import make_pre_reg_grid
from reflexive_options.types import SurfaceGrid

WindowArray = NDArray[np.float64]
"""Shape (n_windows, window_length, n_K, n_T) — rolling windows of daily surfaces."""

_MIN_WINDOWS_FOR_DISTANCE = 30


@dataclass(frozen=True)
class SlicedW2Result:
    """Output of `evaluate_sliced_w2_on_surface_windows`."""

    distance: float
    n_slices: int
    n_windows_left: int
    n_windows_right: int
    rejected_left_frac: float
    rejected_right_frac: float


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def make_rolling_windows(
    daily_surfaces: NDArray[np.float64],
    window_length: int = 21,
    stride: int = 1,
) -> WindowArray:
    """Roll a daily-surface series into overlapping windows.

    Args:
        daily_surfaces: shape `(n_days, n_K, n_T)` daily IV surfaces.
        window_length: number of consecutive daily surfaces per window. Pre-reg
            §4 locks this at 21.
        stride: step between successive windows in days. Pre-reg §4 locks at 1.

    Returns:
        `(n_windows, window_length, n_K, n_T)` array of windows. `n_windows`
        equals `(n_days - window_length) // stride + 1`. Empty result if the
        series is shorter than one window.
    """
    if daily_surfaces.ndim != 3:
        raise ValueError(
            f"daily_surfaces must be 3D (n_days, n_K, n_T), got shape {daily_surfaces.shape}"
        )
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    n_days = daily_surfaces.shape[0]
    if n_days < window_length:
        return np.empty(
            (0, window_length, daily_surfaces.shape[1], daily_surfaces.shape[2]),
            dtype=np.float64,
        )

    n_windows = (n_days - window_length) // stride + 1
    starts = np.arange(n_windows) * stride
    # Build via fancy-indexing — explicit copy keeps downstream slicing O(1).
    idx = starts[:, None] + np.arange(window_length)[None, :]
    return daily_surfaces[idx]


# ---------------------------------------------------------------------------
# Arbitrage filter for windows
# ---------------------------------------------------------------------------


def filter_arbitrage_free_windows(
    windows: WindowArray,
    grid: SurfaceGrid,
    *,
    spot: float,
    rate: float,
    dividend: float,
) -> tuple[WindowArray, NDArray[np.bool_]]:
    """Keep only windows whose every daily surface is arbitrage-free.

    A window passes iff `check_arbitrage_free(...).passes_all` is True for
    every one of its `window_length` daily surfaces (so any `is_marginal`
    daily surface drops the window — this matches the existing API contract
    and is the strictest reading of pre-reg §4).

    Returns:
        kept_windows: subset of input windows that survived the filter.
        mask: 1D bool array of length `n_windows`; True = kept.
    """
    if windows.ndim != 4:
        raise ValueError(
            f"windows must be 4D (n_windows, window_length, n_K, n_T), got {windows.shape}"
        )
    n_windows = windows.shape[0]
    if n_windows == 0:
        return windows, np.zeros(0, dtype=np.bool_)

    mask = np.zeros(n_windows, dtype=np.bool_)
    window_length = windows.shape[1]
    for w in range(n_windows):
        all_pass = True
        for d in range(window_length):
            check = check_arbitrage_free(
                windows[w, d], grid, spot=spot, rate=rate, dividend=dividend
            )
            if not check.passes_all:
                all_pass = False
                break
        mask[w] = all_pass

    return windows[mask], mask


# ---------------------------------------------------------------------------
# Sliced Wasserstein-2 on R^d empirical distributions
# ---------------------------------------------------------------------------


def sliced_wasserstein_2(
    samples_left: NDArray[np.float64],
    samples_right: NDArray[np.float64],
    *,
    n_slices: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Sliced-W2 distance between two empirical distributions in R^d.

    For each θ ∼ Unif(S^{d-1}), project both samples onto θ, compute the 1D
    W2 distance via sorted-quantile L2 (closed-form), and average squared
    distances across slices. Returns the square root of the mean.

    Equal-size case: sort both projected vectors, take the L2 distance of the
    sorted arrays. Unequal-size case: align via equal-quantile resampling
    using the larger sample size as the common grid.

    Args:
        samples_left: shape (n_left, d).
        samples_right: shape (n_right, d).
        n_slices: number of random projections (default 1000 per pre-reg §4).
        rng: numpy Generator for reproducibility; default-initialised if None.

    Returns:
        Sliced-W2 distance estimate (scalar). NaN if either sample is empty.
    """
    if samples_left.ndim != 2 or samples_right.ndim != 2:
        raise ValueError(
            "samples_left and samples_right must be 2D (n, d); "
            f"got {samples_left.shape} and {samples_right.shape}"
        )
    d_left = samples_left.shape[1]
    d_right = samples_right.shape[1]
    if d_left != d_right:
        raise ValueError(f"sample dimensions must match: got d_left={d_left}, d_right={d_right}")
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")

    n_left = samples_left.shape[0]
    n_right = samples_right.shape[0]
    if n_left == 0 or n_right == 0:
        return float("nan")

    rng_ = rng if rng is not None else np.random.default_rng()
    d = d_left

    # Random directions: z ~ N(0, I_d), normalise per row to S^{d-1}.
    thetas = rng_.standard_normal((n_slices, d)).astype(np.float64)
    norms = np.linalg.norm(thetas, axis=1, keepdims=True)
    # Reject the (measure-zero) case of an exact zero-vector by replacing with e_0.
    zero_rows = norms.ravel() < 1e-300
    if zero_rows.any():
        thetas[zero_rows] = 0.0
        thetas[zero_rows, 0] = 1.0
        norms = np.linalg.norm(thetas, axis=1, keepdims=True)
    thetas = thetas / norms  # (n_slices, d)

    # Project both samples in one matmul each: (n, d) @ (d, n_slices) -> (n, n_slices).
    proj_left = samples_left @ thetas.T  # (n_left, n_slices)
    proj_right = samples_right @ thetas.T  # (n_right, n_slices)

    if n_left == n_right:
        sl = np.sort(proj_left, axis=0)
        sr = np.sort(proj_right, axis=0)
    else:
        # Resample to a common grid via empirical-CDF inversion.
        n_common = max(n_left, n_right)
        qs = np.linspace(0.0, 1.0, n_common, dtype=np.float64)
        # np.quantile along axis=0 returns shape (n_common, n_slices).
        sl = np.quantile(proj_left, qs, axis=0)
        sr = np.quantile(proj_right, qs, axis=0)

    diff_sq = (sl - sr) ** 2  # (n_common, n_slices)
    # Per-slice 1D W2^2 = mean squared difference of sorted (or equal-quantile) arrays.
    sw2_sq_per_slice = diff_sq.mean(axis=0)  # (n_slices,)
    return float(np.sqrt(sw2_sq_per_slice.mean()))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _flatten_windows(windows: WindowArray) -> NDArray[np.float64]:
    """`(n, L, n_K, n_T)` -> `(n, L*n_K*n_T)` C-order flat per row."""
    n = windows.shape[0]
    return windows.reshape(n, -1)


def evaluate_sliced_w2_on_surface_windows(
    daily_surfaces_left: NDArray[np.float64],
    daily_surfaces_right: NDArray[np.float64],
    grid: SurfaceGrid,
    *,
    spot: float = 100.0,
    rate: float = 0.0,
    dividend: float = 0.0,
    window_length: int = 21,
    n_slices: int = 1000,
    rng: np.random.Generator | None = None,
) -> SlicedW2Result:
    """Roll, arbitrage-filter, flatten, and compute sliced-W2.

    Returns NaN for `distance` (with a warning) if either side has fewer than
    30 surviving windows after the arbitrage filter — the empirical CIs are
    untrustworthy below that, and the metric is locked at the rolling-window
    level by pre-reg §4.
    """
    win_left = make_rolling_windows(daily_surfaces_left, window_length=window_length, stride=1)
    win_right = make_rolling_windows(daily_surfaces_right, window_length=window_length, stride=1)

    n_total_left = win_left.shape[0]
    n_total_right = win_right.shape[0]

    kept_left, _ = filter_arbitrage_free_windows(
        win_left, grid, spot=spot, rate=rate, dividend=dividend
    )
    kept_right, _ = filter_arbitrage_free_windows(
        win_right, grid, spot=spot, rate=rate, dividend=dividend
    )

    n_kept_left = kept_left.shape[0]
    n_kept_right = kept_right.shape[0]

    rejected_left_frac = 0.0 if n_total_left == 0 else (n_total_left - n_kept_left) / n_total_left
    rejected_right_frac = (
        0.0 if n_total_right == 0 else (n_total_right - n_kept_right) / n_total_right
    )

    if n_kept_left < _MIN_WINDOWS_FOR_DISTANCE or n_kept_right < _MIN_WINDOWS_FOR_DISTANCE:
        warnings.warn(
            f"insufficient surviving windows for sliced-W2: "
            f"n_left={n_kept_left}, n_right={n_kept_right} "
            f"(< {_MIN_WINDOWS_FOR_DISTANCE}); returning NaN distance",
            stacklevel=2,
        )
        return SlicedW2Result(
            distance=float("nan"),
            n_slices=n_slices,
            n_windows_left=n_kept_left,
            n_windows_right=n_kept_right,
            rejected_left_frac=rejected_left_frac,
            rejected_right_frac=rejected_right_frac,
        )

    flat_left = _flatten_windows(kept_left)
    flat_right = _flatten_windows(kept_right)

    distance = sliced_wasserstein_2(flat_left, flat_right, n_slices=n_slices, rng=rng)

    return SlicedW2Result(
        distance=distance,
        n_slices=n_slices,
        n_windows_left=n_kept_left,
        n_windows_right=n_kept_right,
        rejected_left_frac=rejected_left_frac,
        rejected_right_frac=rejected_right_frac,
    )


__all__ = [
    "SlicedW2Result",
    "WindowArray",
    "evaluate_sliced_w2_on_surface_windows",
    "filter_arbitrage_free_windows",
    "make_pre_reg_grid",
    "make_rolling_windows",
    "sliced_wasserstein_2",
]
