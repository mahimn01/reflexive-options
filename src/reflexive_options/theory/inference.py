"""Statistical inference primitives for the pre-registered analysis pipeline.

Three independent building blocks live here:

  * `stationary_block_bootstrap` — Politis & Romano (1994) stationary block
    bootstrap, the locked resampling scheme for §6 of `pre_registration.md`.
    The iid bootstrap previously used in `theory.sensitivity` underestimates
    variance for the heavily-overlapped 21-day rolling-window statistics
    that all of H1/H2 build on; the stationary block bootstrap collapses to
    iid as `block_length_mean → 1` and broadens correctly under positive
    autocorrelation.

  * `benjamini_hochberg` — BH (1995) step-up procedure, the locked secondary
    multiple-comparison correction. Standard sort-and-threshold formulation;
    handles the §6 H2/H3/H4 secondary grid and the per-event variants.

  * `tost_equivalence` — Two One-Sided Tests for equivalence within `±margin`
    (Schuirmann 1987). Pairs with the GP-posterior slope estimate from
    `theory.sensitivity` (amendment A6) on the dimensionless elasticity scale
    (amendment A7) so the locked ±0.1 margin in §3 H2 recovers its intended
    "10% effect" interpretation.

All three are deliberately framework-agnostic — they take numpy arrays, not
result dataclasses — so that callers in `theory.sensitivity` (κ-sensitivity
slope CI) and `experiments.h4_validation` (FDR over per-signal p-values) can
share a single tested implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# ---------------------------------------------------------------------------
# Politis–Romano (1994) stationary block bootstrap.
# ---------------------------------------------------------------------------


def stationary_block_bootstrap(
    samples: NDArray[np.float64],
    *,
    block_length_mean: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Politis–Romano (1994) stationary block bootstrap.

    Block lengths drawn iid from Geometric(1 / `block_length_mean`); blocks
    selected by uniform random start indices on the *circular* sample.
    Returns shape ``(n_resamples, len(samples))`` — each row is a resample of
    the same length as the input, formed by concatenating geometric-length
    blocks until the requested length is reached.

    The returned object is a fresh contiguous array (no view aliasing into
    `samples`); callers may mutate it in place.

    The recommended block-length default for the §6 setting (21-day rolling
    windows) is `block_length_mean = 21` — one window length, the natural
    decorrelation scale of overlapping rolling statistics. See `theory.
    sensitivity` for the wired-in default at the call site.

    Args:
        samples: 1D float array of original observations.
        block_length_mean: mean of the geometric block-length distribution.
            Must be > 0. With mean = 1 the bootstrap collapses to iid.
        n_resamples: number of resamples to generate.
        rng: numpy Generator used for both block lengths and start indices.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1D, got shape {samples.shape}")
    n = len(samples)
    if n == 0:
        raise ValueError("samples must be non-empty")
    if block_length_mean <= 0.0:
        raise ValueError(f"block_length_mean must be > 0, got {block_length_mean}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be > 0, got {n_resamples}")

    # Geometric parameter: mean of Geom(p, support {1, 2, ...}) is 1/p.
    p = 1.0 / float(block_length_mean)
    p = min(max(p, np.finfo(np.float64).eps), 1.0)

    out = np.empty((n_resamples, n), dtype=np.float64)
    for r in range(n_resamples):
        filled = 0
        while filled < n:
            start = int(rng.integers(0, n))
            block_len = int(rng.geometric(p))
            block_len = max(block_len, 1)
            take = min(block_len, n - filled)
            # Circular indexing handles wrap-around without a copy of samples.
            idx = (start + np.arange(take)) % n
            out[r, filled : filled + take] = samples[idx]
            filled += take
    return out


def block_bootstrap_ci(
    samples: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float],
    *,
    confidence: float = 0.95,
    block_length_mean: float = 21.0,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Block-bootstrap CI for an arbitrary scalar statistic.

    Returns ``(point_estimate, ci_low, ci_high)``. The point estimate is the
    statistic applied to the original sample; the CI is the percentile
    interval on the bootstrap distribution at the requested confidence level.
    BCa is intentionally not used — it requires a jackknife-style influence
    estimate that's ill-defined for the highly-correlated rolling-window
    statistics this is intended for; the percentile CI is what the pre-reg
    locks in §6.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if rng is None:
        rng = np.random.default_rng()

    samples = np.asarray(samples, dtype=np.float64)
    point = float(statistic(samples))

    resamples = stationary_block_bootstrap(
        samples,
        block_length_mean=block_length_mean,
        n_resamples=n_resamples,
        rng=rng,
    )
    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        boot_stats[i] = float(statistic(resamples[i]))

    finite = boot_stats[np.isfinite(boot_stats)]
    if len(finite) == 0:
        raise RuntimeError("all bootstrap statistics were non-finite")

    alpha = 1.0 - confidence
    lo = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Benjamini–Hochberg (1995) FDR correction.
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> NDArray[np.bool_]:
    """Benjamini–Hochberg (1995) step-up FDR control.

    Returns a boolean array — True where the corresponding hypothesis is
    rejected at the FDR level `alpha`. The procedure controls FDR exactly
    under independent p-values and conservatively under PRDS (positive
    regression dependency on the subset of true nulls) — the relevant case
    for the §6 secondary-hypothesis grid where the test statistics are
    derived from a shared estimation pipeline and are therefore positively
    associated.

    Procedure:
        1. Sort p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m).
        2. Find the largest k such that p_(k) ≤ k * alpha / m.
        3. Reject all H_(1), ..., H_(k); accept the rest.

    Args:
        p_values: 1D array of p-values, each in [0, 1]. NaNs are not allowed.
        alpha: target FDR level in (0, 1).
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1D, got shape {p.shape}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if np.any(~np.isfinite(p)):
        raise ValueError("p_values must all be finite")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must all be in [0, 1]")

    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=np.bool_)

    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = alpha * np.arange(1, m + 1, dtype=np.float64) / float(m)
    below = sorted_p <= thresholds

    rejected = np.zeros(m, dtype=np.bool_)
    if not below.any():
        return rejected
    k = int(np.where(below)[0].max())  # largest index satisfying the condition
    # Reject all sorted positions 0..k inclusive — map back to original order.
    rejected[order[: k + 1]] = True
    return rejected


# ---------------------------------------------------------------------------
# Two One-Sided Tests for equivalence (Schuirmann 1987).
# ---------------------------------------------------------------------------


def tost_equivalence(
    estimate: float,
    standard_error: float,
    *,
    margin: float,
    alpha: float = 0.05,
    df: float | None = None,
) -> tuple[bool, float]:
    """Two One-Sided Tests for equivalence within ±`margin`.

    Returns ``(is_equivalent, max_p_value)``. Equivalence is concluded iff
    BOTH one-sided null hypotheses are rejected at level `alpha`:

        H_{0,upper}: estimate ≥  margin   → reject if (estimate - margin)/SE ≤ -t_{alpha}
        H_{0,lower}: estimate ≤ -margin   → reject if (estimate + margin)/SE ≥  t_{alpha}

    We use a t-distribution with `df` degrees of freedom (defaulting to a
    Gaussian z-test when `df` is None — the limit `df → ∞`). The reported
    p-value is the *larger* of the two one-sided p-values; equivalence at
    level `alpha` is equivalent to that maximum p ≤ `alpha`.

    Args:
        estimate: point estimate of the parameter (e.g. the dimensionless
            elasticity from the GP-posterior slope; see `theory.sensitivity`).
        standard_error: SE of `estimate`. Must be > 0.
        margin: equivalence margin (one-sided width); must be > 0. The test
            asks whether the parameter is in (-margin, +margin).
        alpha: per-test significance level. Equivalence is at the same level
            (the maximum of two one-sided p-values is itself a p-value).
        df: degrees of freedom for the t-distribution; None ⇒ z-test.
    """
    if standard_error <= 0.0:
        raise ValueError(f"standard_error must be > 0, got {standard_error}")
    if margin <= 0.0:
        raise ValueError(f"margin must be > 0, got {margin}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    t_upper = (estimate - margin) / standard_error
    t_lower = (estimate + margin) / standard_error

    if df is None:
        # Gaussian one-sided p-values.
        # H_{0,upper} rejected when (estimate - margin)/SE is far below 0 ⇒ p_upper = P(Z ≤ t_upper).
        # H_{0,lower} rejected when (estimate + margin)/SE is far above 0 ⇒ p_lower = P(Z ≥ t_lower) = 1 - P(Z ≤ t_lower).
        p_upper = float(stats.norm.cdf(t_upper))
        p_lower = float(stats.norm.sf(t_lower))
    else:
        if df <= 0:
            raise ValueError(f"df must be > 0, got {df}")
        p_upper = float(stats.t.cdf(t_upper, df=df))
        p_lower = float(stats.t.sf(t_lower, df=df))

    max_p = max(p_upper, p_lower)
    is_equivalent = max_p <= alpha
    return is_equivalent, max_p
