"""Critical-slowing-down (CSD) early-warning-signal detector.

This module replaces the original H4 spectral-peak test, which was geometrically
impossible: the Hopf limit-cycle period in the reflexive model is 5.3-11 years
(``theory.md`` L144: omega* = 0.5724 rad/yr -> 11.0 yr; L778: omega* = 1.18
rad/yr -> 5.3 yr), so the target frequency sits *below* the lowest non-DC bin of
a 1024-day Welch window, and each +/-60-day event window spans only 4-9% of a
single cycle.  A multi-year oscillation cannot be resolved in <=4 yr of data.

The replacement detects *nearness to* the bifurcation rather than the
post-bifurcation oscillation itself.  This is the standard early-warning-signal
(EWS) methodology for short pre-transition records:

  Scheffer et al. (2009) "Early-warning signals for critical transitions",
    Nature 461:53-59.
  Dakos et al. (2012) "Methods for detecting early warnings of critical
    transitions in time series and spatial data", PLoS ONE 7(7):e41010.
  Lenton (2011) "Early warning of climate tipping points",
    Nature Climate Change 1:201-209.

Mechanism: as the reflexive coupling ``kappa`` approaches ``kappa_star``, the
real part of the leading Jacobian eigenvalue approaches zero, so the system's
recovery rate from perturbations slows ("critical slowing down").  In a SHORT
rolling window this manifests as:

  (i)   rising lag-1 autocorrelation of the volatility proxy,
  (ii)  rising rolling variance of the volatility proxy,
  (iii) (auxiliary) rising fitted AR(1) coefficient.

The monotone trend in each rolling statistic is tested with Kendall's tau, with
significance assessed against an AR(1) / phase-randomised stationary surrogate
null that preserves the linear autocorrelation structure but destroys any trend.

Pure numpy/scipy, typed, deterministic.  The detector imports nothing from the
simulator; ground-truth validation (which DOES drive the simulator / a
mechanistic AR(1)-ramp generator) lives in ``scripts/csd_validation.py`` and the
test module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


# ----------------------------------------------------------------------------
# Detrending
# ----------------------------------------------------------------------------


def detrend_series(x: FloatArray, mode: str = "gaussian") -> FloatArray:
    """Remove a slow trend from ``x`` before computing EWS statistics.

    A Gaussian-kernel smoother (bandwidth = 10% of the record) is the Dakos et
    al. (2012) default; it prevents a deterministic mean shift from being read as
    rising variance.  ``"linear"`` removes an OLS line; ``"none"`` is a no-op.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if mode == "none":
        return x.copy()
    if mode == "linear":
        t = np.arange(n, dtype=np.float64)
        slope, intercept, *_ = stats.linregress(t, x)
        return np.asarray(x - (slope * t + intercept), dtype=np.float64)
    if mode == "gaussian":
        bw = max(2.0, 0.1 * n)
        t = np.arange(n, dtype=np.float64)
        diff = t[:, None] - t[None, :]
        w = np.exp(-0.5 * (diff / bw) ** 2)
        w /= w.sum(axis=1, keepdims=True)
        trend = w @ x
        return np.asarray(x - trend, dtype=np.float64)
    raise ValueError(f"unknown detrend mode: {mode!r}")


# ----------------------------------------------------------------------------
# Pointwise EWS statistics
# ----------------------------------------------------------------------------


def lag1_autocorr(x: FloatArray) -> float:
    """Lag-1 autocorrelation (biased estimator, n in the denominator)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(xc[:-1], xc[1:]) / denom)


def ar1_coefficient(x: FloatArray) -> float:
    """OLS AR(1) slope b in x_t = a + b x_{t-1} + eps."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    y = x[1:]
    z = x[:-1]
    zc = z - z.mean()
    denom = float(np.dot(zc, zc))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(zc, y - y.mean()) / denom)


def rolling_statistic(series: FloatArray, window: int, statistic: str) -> FloatArray:
    """Rolling EWS statistic over ``series`` (right-edge anchored).

    Returns an array of length ``len(series) - window + 1`` (one value per fully
    populated window).  ``statistic`` is one of
    {"variance", "std", "autocorr", "ar1"}.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.size
    if window < 3:
        raise ValueError("window must be >= 3")
    if n < window:
        return np.empty(0, dtype=np.float64)

    n_out = n - window + 1
    out = np.empty(n_out, dtype=np.float64)
    if statistic == "variance":
        for i in range(n_out):
            out[i] = float(np.var(series[i : i + window], ddof=1))
    elif statistic == "std":
        for i in range(n_out):
            out[i] = float(np.std(series[i : i + window], ddof=1))
    elif statistic == "autocorr":
        for i in range(n_out):
            out[i] = lag1_autocorr(series[i : i + window])
    elif statistic == "ar1":
        for i in range(n_out):
            out[i] = ar1_coefficient(series[i : i + window])
    else:
        raise ValueError(f"unknown statistic: {statistic!r}")
    return out


# ----------------------------------------------------------------------------
# Trend statistic
# ----------------------------------------------------------------------------


def kendall_tau_trend(stat_series: FloatArray) -> tuple[float, float]:
    """Kendall's tau of a statistic series against time.

    Returns (tau, asymptotic_two_sided_p).  The operative significance comes from
    the surrogate ensemble, not this asymptotic p.
    """
    s = np.asarray(stat_series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 3:
        return float("nan"), float("nan")
    t = np.arange(s.size, dtype=np.float64)
    res = stats.kendalltau(t, s)
    tau = float(np.asarray(res.statistic).item())
    pval = float(np.asarray(res.pvalue).item())
    return tau, pval


# ----------------------------------------------------------------------------
# Surrogate nulls (stationary: same linear structure, NO trend)
# ----------------------------------------------------------------------------


def _fit_ar1(x: FloatArray) -> tuple[float, float, float]:
    """Fit AR(1): x_t = c + phi x_{t-1} + eps. Returns (c, phi, sigma_eps)."""
    x = np.asarray(x, dtype=np.float64)
    y = x[1:]
    z = x[:-1]
    zc = z - z.mean()
    denom = float(np.dot(zc, zc))
    if denom <= 0.0:
        return float(x.mean()), 0.0, float(np.std(x, ddof=1))
    phi = float(np.dot(zc, y - y.mean()) / denom)
    phi = float(np.clip(phi, -0.999, 0.999))
    c = float(y.mean() - phi * z.mean())
    resid = y - (c + phi * z)
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    return c, phi, sigma


def ar1_surrogate(x: FloatArray, rng: np.random.Generator) -> FloatArray:
    """One stationary AR(1) surrogate matching x's mean, phi, innovation var.

    Preserves the linear stochastic structure (so EWS baselines are calibrated)
    but contains NO trend -- the correct null for "is the observed rising trend
    beyond what a fixed-coupling stationary system produces?".
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    c, phi, sigma = _fit_ar1(x)
    eps = rng.standard_normal(n) * sigma
    y = np.empty(n, dtype=np.float64)
    y[0] = float(x[0])
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + eps[t]
    return y


def phase_randomised_surrogate(x: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Fourier phase-randomised surrogate: preserves spectrum, destroys trend."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mu = float(x.mean())
    fft = np.fft.rfft(x - mu)
    amp = np.abs(fft)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=amp.size)
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    surro = np.fft.irfft(amp * np.exp(1j * phases), n=n)
    return np.asarray(surro + mu, dtype=np.float64)


# ----------------------------------------------------------------------------
# Full CSD test
# ----------------------------------------------------------------------------


@dataclass
class CSDResult:
    """Outcome of a CSD early-warning test on one series."""

    statistic: str
    window: int
    tau: float
    tau_asymptotic_p: float
    p_value: float
    n_surrogates: int
    rolling_values: FloatArray
    surrogate_taus: FloatArray
    significant: bool


def csd_test(
    series: FloatArray,
    window: int,
    statistic: str = "autocorr",
    *,
    detrend: str = "gaussian",
    n_surrogates: int = 1000,
    surrogate: str = "ar1",
    alpha: float = 0.05,
    seed: int | None = None,
) -> CSDResult:
    """Full CSD early-warning test on one volatility-proxy series.

    Procedure
    ---------
    1. Detrend ``series`` (default Gaussian kernel, Dakos 2012).
    2. Compute the rolling EWS ``statistic`` over a ``window``-length window.
    3. Compute Kendall's tau of that rolling statistic vs time.
    4. Build ``n_surrogates`` stationary surrogates of the detrended series,
       recompute the rolling statistic + tau on each.
    5. One-sided surrogate p-value (directional CSD prediction tau > 0):
         p = (#{tau_surr >= tau_obs} + 1) / (N_valid + 1).
    """
    series = np.asarray(series, dtype=np.float64)
    xd = detrend_series(series, detrend)

    rolling = rolling_statistic(xd, window, statistic)
    tau_obs, tau_p = kendall_tau_trend(rolling)

    rng = np.random.default_rng(seed)
    surr_taus = np.empty(n_surrogates, dtype=np.float64)
    for k in range(n_surrogates):
        if surrogate == "ar1":
            s = ar1_surrogate(xd, rng)
        elif surrogate == "phase":
            s = phase_randomised_surrogate(xd, rng)
        else:
            raise ValueError(f"unknown surrogate: {surrogate!r}")
        rs = rolling_statistic(s, window, statistic)
        tau_s, _ = kendall_tau_trend(rs)
        surr_taus[k] = tau_s

    valid = surr_taus[np.isfinite(surr_taus)]
    if not np.isfinite(tau_obs) or valid.size == 0:
        p_value = float("nan")
        significant = False
    else:
        n_ge = int(np.sum(valid >= tau_obs))
        p_value = (n_ge + 1) / (valid.size + 1)
        significant = bool(p_value <= alpha)

    return CSDResult(
        statistic=statistic,
        window=window,
        tau=tau_obs,
        tau_asymptotic_p=tau_p,
        p_value=p_value,
        n_surrogates=n_surrogates,
        rolling_values=rolling,
        surrogate_taus=surr_taus,
        significant=significant,
    )


# ----------------------------------------------------------------------------
# Multiple-testing + proxy helpers
# ----------------------------------------------------------------------------


def benjamini_hochberg(pvals: FloatArray, fdr: float = 0.10) -> NDArray[np.bool_]:
    """Benjamini-Hochberg step-up. Returns the boolean reject mask at ``fdr``."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = fdr * (np.arange(1, m + 1, dtype=np.float64) / m)
    passed = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = int(np.max(np.nonzero(passed)[0]))
        reject[order[: k + 1]] = True
    return reject


def vol_proxy(log_returns: FloatArray, kind: str = "abs") -> FloatArray:
    """Volatility proxy from a log-return series.

    "abs" -> |r_t| (the original H4 target series), "sq" -> r_t^2,
    "logsq" -> log(r_t^2 + eps) (Gaussianised).
    """
    r = np.asarray(log_returns, dtype=np.float64)
    if kind == "abs":
        return np.abs(r)
    if kind == "sq":
        return r * r
    if kind == "logsq":
        return np.log(r * r + 1e-12)
    raise ValueError(f"unknown vol proxy kind: {kind!r}")
