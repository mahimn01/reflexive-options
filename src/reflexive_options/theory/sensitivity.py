"""κ-sensitivity helper.

The κ-sensitivity curve is the central novel evaluation in the paper:
train an RL agent at κ = κ₀, deploy across a family of environments at
varying κ, and plot the metric vs κ. The slope at κ₀ is a quantitative
measure of reflexivity-importance.

This module provides the deterministic numerical sensitivity (no RL agent —
just propagate the simulator). The full RL κ-sensitivity experiment lives
in `experiments/reflexive_transfer.py`.

**Slope CI methodology — pre-reg amendment A6.** The original locked spec
fit a degree-4 interpolating UnivariateSpline (s=0) through the 9 (κ, mean)
points and bootstrapped seed assignments to get a slope CI. The V3 audit
(`/tmp/audit_v3_bootstrap_v2.py`) showed this has 0% coverage when the
underlying κ-curve is non-smooth at the anchor — a plausible scenario for
an RL agent trained at exactly κ_anchor. Amendment A6 replaces it with a
Gaussian-process posterior over the function whose derivative-at-anchor
has a closed-form Gaussian distribution; the slope CI is the resulting
posterior interval. A local-quadratic-regression fallback handles the rare
case where GP hyperparameter optimisation diverges.

**Noise-variance pinning — v0.3.1 fix to A6.** The G3 statistical audit at
v0.3.0 found ~70% empirical coverage on smooth-truth synthetic cases (vs
the 95% nominal), traced to the WhiteKernel noise MLE collapsing to its
1e-10 lower bound on the n=9 grid. The fix: instead of letting the GP
optimise the noise variance jointly with the length scale, we *pin* the
noise variance to the observed seed-to-seed std (averaged across the κ
grid). This is the natural noise level — it is exactly the variance of
the metric estimator at fixed κ, which is what the GP's WhiteKernel
should represent. With pinned noise, coverage on quadratic / linear /
sin / quintic synthetic ground truths recovers to ≥85% at the locked
n_seeds = 100. See `tests/test_sensitivity.py::test_gp_slope_ci_coverage_with_pinned_noise`.

Reference: evaluation_framework_brief.md §3, paper/pre_registration_amendments.md A6.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

SlopeMethod = Literal["gp", "local_quadratic_fallback"]


@dataclass(frozen=True)
class SensitivityResult:
    """κ-sensitivity curve data.

    The headline scalar is `slope_at_anchor`; the operational test is whether
    `(slope_ci_low, slope_ci_high)` excludes zero and whether the
    dimensionless elasticity (computed downstream) passes a TOST equivalence
    check at the locked ±0.1 margin (pre-reg §3 H2 + amendment A7).
    """

    kappa_grid: NDArray[np.float64]
    metric_values: NDArray[np.float64]  # metric at each κ
    metric_std: NDArray[np.float64]  # std (e.g. across seeds) at each κ
    kappa_anchor: float  # the training-time κ₀ at which slope is reported
    slope_at_anchor: float  # ∂metric/∂κ at κ₀ — the headline scalar
    slope_ci_low: float  # GP-posterior 95% CI lower bound on the slope
    slope_ci_high: float
    slope_se: float  # GP-posterior SE at the anchor (used by TOST downstream)
    method: SlopeMethod  # which estimator produced the slope CI


# ---------------------------------------------------------------------------
# RBF-kernel GP derivative posterior (closed form).
# ---------------------------------------------------------------------------


def _rbf_kernel_matrix(
    x_a: NDArray[np.float64], x_b: NDArray[np.float64], length_scale: float
) -> NDArray[np.float64]:
    """K(x_a, x_b)_{ij} = exp(-(x_a_i - x_b_j)^2 / (2 ℓ^2))."""
    diff = x_a[:, None] - x_b[None, :]
    return np.exp(-0.5 * (diff / length_scale) ** 2)


def _gp_derivative_posterior(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_anchor: float,
    *,
    length_scale: float,
    noise_variance: float,
    signal_variance: float,
    y_mean: float,
) -> tuple[float, float]:
    """Closed-form posterior mean and variance of f'(x_anchor) under an RBF GP.

    For a GP with kernel k(x, x') = signal_variance · exp(-(x - x')² / (2 ℓ²))
    and observation noise σ²I, the posterior over the derivative at x_anchor
    is also Gaussian:

        mean = k_x*(x_anchor, X) (K + σ²I)^{-1} (y - y_mean)
        var  = k_xx*(x_anchor, x_anchor) - k_x*(x_anchor, X) (K + σ²I)^{-1} k_x*(x_anchor, X)^T

    where:
        k_x*(x, x')   = ∂_x k(x, x')   = -(x - x') / ℓ²  · k(x, x')
        k_xx*(x, x')  = ∂_x ∂_{x'} k   = (1/ℓ² - (x - x')² / ℓ⁴) · k(x, x')

    `y_mean` is the constant prior mean already subtracted from `y_train`;
    the derivative of a constant mean is zero, so it doesn't appear in the
    formulas above (it would have appeared as `m'(x_anchor) = 0`).
    """
    n = len(x_train)
    diff = x_anchor - x_train  # shape (n,)
    k_anchor = signal_variance * np.exp(-0.5 * (diff / length_scale) ** 2)
    # ∂_x k(x_anchor, X)_i = -(x_anchor - x_train_i) / ℓ² · k(x_anchor, x_train_i)
    k_grad = -(diff / length_scale**2) * k_anchor  # shape (n,)
    # ∂_x ∂_{x'} k(x_anchor, x_anchor) = signal_variance / ℓ²
    k_grad_grad = signal_variance / length_scale**2

    K = signal_variance * _rbf_kernel_matrix(
        x_train, x_train, length_scale
    ) + noise_variance * np.eye(n)
    K_jittered = K + 1e-12 * np.eye(n)
    # Solve via Cholesky for numerical stability; pinv fallback for the
    # very-ill-conditioned case (Cholesky raises LinAlgError).
    try:
        L = np.linalg.cholesky(K_jittered)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train - y_mean))
        v = np.linalg.solve(L, k_grad)
        var = float(k_grad_grad - v @ v)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K_jittered)
        alpha = K_inv @ (y_train - y_mean)
        var = float(k_grad_grad - k_grad @ K_inv @ k_grad)

    mean = float(k_grad @ alpha)
    var = max(var, 0.0)  # numerical floor for the posterior variance
    return mean, var


def _fit_gp_and_derivative(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_anchor: float,
    *,
    noise_variance_raw: float | None = None,
) -> tuple[float, float]:
    """Fit sklearn GP (RBF + White) on (x, y) then evaluate derivative posterior at x_anchor.

    Returns ``(slope_mean, slope_se)``. Raises RuntimeError on any failure
    so callers can fall back to local-quadratic regression cleanly.

    Args:
        x_train: κ grid (raw scale).
        y_train: per-κ mean metric values (raw scale).
        x_anchor: κ at which to evaluate the slope (raw scale).
        noise_variance_raw: if not None, pin the WhiteKernel noise variance
            to this value (in raw `y_train` units, i.e. `Var(estimator at κ)`).
            This represents the seed-to-seed Monte-Carlo variance of the metric
            mean and is the *natural* noise level for the GP. Pinning fixes
            the under-coverage discovered by the v0.3.0 G3 audit (the noise
            MLE was collapsing to the lower bound on n=9 grid points).
            If None, fall back to the legacy joint-MLE behaviour.
    """
    # Centre and scale x for numerical stability — anchor scales can be 1e-12.
    x_scale = float(np.max(np.abs(x_train))) if np.max(np.abs(x_train)) > 0 else 1.0
    x_train_s = (x_train / x_scale).astype(np.float64)
    x_anchor_s = float(x_anchor / x_scale)

    y_mean = float(np.mean(y_train))
    y_centred = y_train - y_mean
    y_std = float(np.std(y_centred, ddof=1))
    if y_std == 0.0:
        # Truly degenerate — every point has the same metric value.
        return 0.0, 0.0
    y_norm = y_centred / y_std

    grid_span = float(np.max(x_train_s) - np.min(x_train_s))
    if grid_span == 0.0:
        raise RuntimeError("kappa_grid degenerate after scaling")

    if noise_variance_raw is not None and noise_variance_raw > 0.0:
        # Pin noise level on the standardised y scale.
        pinned_noise_level = float(noise_variance_raw / (y_std**2))
        # Floor at machine-precision so the kernel matrix stays positive-definite.
        pinned_noise_level = max(pinned_noise_level, 1e-10)
        white = WhiteKernel(
            noise_level=pinned_noise_level,
            noise_level_bounds="fixed",
        )
    else:
        white = WhiteKernel(
            noise_level=1e-2,
            noise_level_bounds=(1e-10, 1.0),
        )
    kernel = (
        RBF(
            length_scale=grid_span / 4.0,
            length_scale_bounds=(grid_span * 1e-2, grid_span * 1e2),
        )
        + white
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,  # we already centred + standardised
        n_restarts_optimizer=5,
        random_state=0,
    )
    # On the deliberately tiny locked n=9 grid, a rough response can place
    # the fitted RBF length scale exactly at its registered lower bound.  That
    # is an admissible constrained optimum, not a failed fit.  A single L-BFGS
    # restart can also terminate abnormally even when another of the six
    # deterministic starts supplies the finite optimum used by sklearn.
    # Suppress only those two diagnostics inside this multi-start fit; the
    # returned slope and uncertainty remain covered by analytic-truth tests.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "The optimal value found for dimension 0 of parameter "
                "k1__length_scale is close to the specified lower bound.*"
            ),
            category=ConvergenceWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="lbfgs failed to converge after .*",
            category=ConvergenceWarning,
        )
        gp.fit(x_train_s.reshape(-1, 1), y_norm)

    # Pull fitted hyperparameters from the optimised kernel.
    fitted = gp.kernel_
    length_scale = float(fitted.k1.length_scale)
    noise_level = float(fitted.k2.noise_level)
    # We standardised y by y_std; signal variance is then unit (the RBF in
    # sklearn has unit prefactor by convention).
    signal_variance = 1.0
    noise_variance = noise_level

    slope_mean_norm, slope_var_norm = _gp_derivative_posterior(
        x_train_s,
        y_norm,
        x_anchor_s,
        length_scale=length_scale,
        noise_variance=noise_variance,
        signal_variance=signal_variance,
        y_mean=0.0,  # already centred
    )

    # Undo the (y_centred / y_std) and (x / x_scale) rescalings:
    #   slope on (x, y_centred) = slope_mean_norm * y_std / x_scale
    slope_mean = slope_mean_norm * y_std / x_scale
    slope_se = float(np.sqrt(slope_var_norm)) * y_std / x_scale

    if not np.isfinite(slope_mean) or not np.isfinite(slope_se):
        raise RuntimeError("GP derivative posterior produced non-finite output")

    return slope_mean, slope_se


def _local_quadratic_slope(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_anchor: float,
) -> tuple[float, float]:
    """Fallback: closed-form local quadratic regression slope + analytic SE.

    Fit y = β₀ + β₁(x - x_anchor) + β₂(x - x_anchor)² by OLS over the full
    grid. The slope at x_anchor is β₁; its variance is the (1, 1) entry of
    σ² (XᵀX)^{-1}, where σ² is the residual variance estimate.
    """
    n = len(x_train)
    if n < 3:
        raise RuntimeError(f"need at least 3 points for local quadratic, got {n}")
    z = x_train - x_anchor
    X = np.column_stack([np.ones(n), z, z**2])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as err:
        raise RuntimeError(f"local quadratic fit singular: {err}") from err
    beta = XtX_inv @ X.T @ y_train
    residuals = y_train - X @ beta
    dof = max(n - 3, 1)
    sigma2 = float(residuals @ residuals) / dof
    var_beta1 = float(XtX_inv[1, 1] * sigma2)
    return float(beta[1]), float(np.sqrt(max(var_beta1, 0.0)))


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def kappa_sensitivity_curve(
    metric_fn: Callable[[float, int], float],
    *,
    kappa_grid: NDArray[np.float64],
    kappa_anchor: float,
    n_seeds: int = 100,
    n_bootstrap: int = 1_000,
    rng_seed: int = 42,
) -> SensitivityResult:
    """Sweep κ and produce the κ-sensitivity curve with GP-posterior CI on the slope.

    The `n_bootstrap` argument is preserved for back-compat with existing
    callers (`experiments/reflexive_transfer.py`) but is no longer used —
    amendment A6 replaces the bootstrap with a closed-form GP-posterior CI.
    A future minor version will deprecate it.

    Args:
        metric_fn: f(kappa, seed) -> scalar metric value.
        kappa_grid: ascending grid of κ values; kappa_anchor must lie strictly inside.
        kappa_anchor: the κ₀ at which we evaluate the slope.
        n_seeds: number of seeds per κ.
        n_bootstrap: deprecated; retained for back-compat. See module docstring.
        rng_seed: seed for the seed-id generator used inside the metric loop.
    """
    del n_bootstrap  # accepted for back-compat; no longer used (A6)

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
    # Pin the GP noise variance to the seed-mean MC variance, averaged across κ.
    # This is the natural noise level — Var(metric mean | κ) ≈ stds² / n_seeds —
    # and prevents the WhiteKernel MLE from collapsing to its lower bound on
    # small grids (the v0.3.0 G3 audit's coverage bug).
    if n_seeds > 0:
        seed_mean_variances = (stds**2) / float(n_seeds)
        noise_variance_raw = float(np.mean(seed_mean_variances))
    else:
        noise_variance_raw = 0.0

    method: SlopeMethod = "gp"
    try:
        slope, slope_se = _fit_gp_and_derivative(
            kappa_grid,
            means,
            kappa_anchor,
            noise_variance_raw=noise_variance_raw if noise_variance_raw > 0.0 else None,
        )
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        slope, slope_se = _local_quadratic_slope(kappa_grid, means, kappa_anchor)
        method = "local_quadratic_fallback"

    # 95% Gaussian CI from the posterior mean ± 1.96 SE.
    ci_low = float(slope - 1.959964 * slope_se)
    ci_high = float(slope + 1.959964 * slope_se)

    return SensitivityResult(
        kappa_grid=kappa_grid,
        metric_values=means,
        metric_std=stds,
        kappa_anchor=kappa_anchor,
        slope_at_anchor=float(slope),
        slope_ci_low=ci_low,
        slope_ci_high=ci_high,
        slope_se=float(slope_se),
        method=method,
    )
