"""Primary redesigned empirical test H1': dealer-gamma (GEX) regression.

This module implements, with NO reinforcement-learning agent and NO
simulator-fit IV surface in the inference loop:

1. A Black-Scholes gamma kernel and a GEX estimator that aggregates an
   end-of-day option open-interest (OI) grid into a single signed dealer
   gamma exposure number per day, using a dealer-sign convention.
2. Realized vol-of-vol and a short-window critical-slowing-down (CSD) proxy
   (rolling lag-1 autocorrelation of |r_t|) computed directly from returns.
3. An ordinary-least-squares estimator with Newey-West (HAC) standard errors
   and a moving-block-bootstrap alternative, plus the H1' decision rule.

The core falsifiable prediction (theory.md, predictions 3-4): when dealers are
net SHORT gamma (G_t < 0), hedging is destabilizing -> next-period realized
vol-of-vol and the CSD signal are ELEVATED; when net LONG gamma, hedging damps
them. In the standardized regression

    y_{t+1} = b0 + b1 * z(G_t) + controls_t + eps_{t+1}

with y = next-day realized vol-of-vol (or the CSD autocorr proxy) and z(G_t)
the standardized signed GEX, the reflexive mechanism predicts b1 < 0 (more
negative GEX -> higher vol-of-vol). The decision rule is: reject H0 in favour
of H1' iff b1 < 0 with HAC AND block-bootstrap one-sided p < alpha after
Benjamini-Hochberg control across the small pre-registered family of outcome
variables, AND the effect is weaker/absent in a quiet-regime control window.

All functions are pure numpy/scipy. No statsmodels dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# 1. Black-Scholes gamma and the GEX estimator                                #
# --------------------------------------------------------------------------- #


def bs_gamma(
    spot: float,
    strike: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Black-Scholes spot gamma d2V/dS2 per unit of underlying.

    Vectorized over arrays of strike, time-to-maturity (years) and implied vol.
    Calls and puts share the same gamma, so no option-type argument is needed.

    gamma = exp(-q*tau) * phi(d1) / (S * sigma * sqrt(tau))
    """
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Guard against degenerate inputs (expiry today, zero vol).
    safe_tau = np.maximum(tau, 1e-8)
    safe_sigma = np.maximum(sigma, 1e-8)
    sqrt_tau = np.sqrt(safe_tau)

    d1 = (np.log(spot / strike) + (r - q + 0.5 * safe_sigma**2) * safe_tau) / (
        safe_sigma * sqrt_tau
    )
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    gamma = np.exp(-q * safe_tau) * phi / (spot * safe_sigma * sqrt_tau)
    # Zero out contracts that are effectively expired.
    out: np.ndarray = np.where(tau <= 0.0, 0.0, gamma)
    return out


@dataclass
class OIGrid:
    """End-of-day SPX option open-interest grid for a single trading day.

    Arrays are flat (one entry per listed contract) and aligned by index.
    """

    spot: float
    strike: np.ndarray
    tau: np.ndarray
    sigma: np.ndarray
    oi: np.ndarray
    is_call: np.ndarray
    contract_multiplier: float = 100.0


def estimate_gex(
    grid: OIGrid,
    convention: str = "squeeze_metrics",
    scale: float = 1e-9,
) -> float:
    """Aggregate dealer gamma exposure (GEX) from an OI grid.

    GEX(t) = sum_k OI_k * gamma_k * sign_k * spot^2 * 0.01 * multiplier * scale

    Dealer-sign conventions:
    - "squeeze_metrics" : dealers long calls (+), short puts (-) (SqueezeMetrics
      / SpotGamma SPX default). Net positive GEX => dealers long gamma.
    - "all_long"        : sign_k = +1 for every contract (diagnostic).
    - "naive_put_call"  : signed but WITHOUT the spot^2 dollar-gamma scaling.

    The spot^2 * 0.01 factor converts BS gamma (per $1) into dollar-gamma (the $
    change in delta per 1% move). Only the SIGN and the standardized value enter
    the regression, so the absolute scale is immaterial.
    """
    g = grid
    gamma = bs_gamma(g.spot, g.strike, g.tau, g.sigma)

    if convention == "squeeze_metrics":
        sign = np.where(g.is_call, 1.0, -1.0)
        dollar = g.spot**2 * 0.01
    elif convention == "all_long":
        sign = np.ones_like(gamma)
        dollar = g.spot**2 * 0.01
    elif convention == "naive_put_call":
        sign = np.where(g.is_call, 1.0, -1.0)
        dollar = 1.0
    else:
        raise ValueError(f"unknown convention: {convention!r}")

    contrib = g.oi * gamma * sign * dollar * g.contract_multiplier
    total = float(np.sum(contrib) * scale)
    # Canonicalize negative zero so sign tests are well-behaved.
    return total + 0.0 if total != 0.0 else 0.0


def estimate_gex_series(
    grids: list[OIGrid],
    convention: str = "squeeze_metrics",
    scale: float = 1e-9,
) -> np.ndarray:
    """Estimate the daily GEX series from a list of per-day OI grids."""
    return np.array([estimate_gex(g, convention=convention, scale=scale) for g in grids])


# --------------------------------------------------------------------------- #
# 2. Outcome variables from returns (no simulator state used)                 #
# --------------------------------------------------------------------------- #


def realized_vol(returns: np.ndarray, window: int = 5) -> np.ndarray:
    """Rolling realized vol (std of log returns) over `window` days.

    Index i holds the std of returns[i-window+1 : i+1]; the first window-1
    entries are NaN.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.std(r[i - window + 1 : i + 1], ddof=1)
    return out


def realized_vol_of_vol(returns: np.ndarray, rv_window: int = 5, vv_window: int = 5) -> np.ndarray:
    """Realized vol-of-vol: rolling std of the realized-vol series.

    This is the primary H1' outcome. Computed purely from returns; no latent
    variance path is used (so it is identically defined in sim and in data).
    """
    rv = realized_vol(returns, window=rv_window)
    n = len(rv)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = i - vv_window + 1
        if lo < 0:
            continue
        w = rv[lo : i + 1]
        if np.any(np.isnan(w)):
            continue
        out[i] = np.std(w, ddof=1)
    return out


def csd_autocorr(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Short-window critical-slowing-down proxy: rolling lag-1 autocorrelation
    of |r_t| (absolute returns) over `window` days.

    Index i holds the lag-1 autocorr of |r| over [i-window+1, i]; first
    window-1 entries are NaN.
    """
    a = np.abs(np.asarray(returns, dtype=float))
    n = len(a)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = a[i - window + 1 : i + 1]
        x0 = w[:-1] - np.mean(w[:-1])
        x1 = w[1:] - np.mean(w[1:])
        denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
        out[i] = float(np.sum(x0 * x1) / denom) if denom > 0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# 3. OLS with Newey-West (HAC) SEs and moving-block bootstrap                  #
# --------------------------------------------------------------------------- #


@dataclass
class RegressionResult:
    """Result of an OLS fit with HAC inference."""

    coef: np.ndarray  # including intercept at index 0
    names: list[str]
    se_ols: np.ndarray
    se_hac: np.ndarray
    tstat_hac: np.ndarray
    pvalue_hac: np.ndarray
    n: int
    r2: float
    resid: np.ndarray = field(repr=False)
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)

    def coef_for(self, name: str) -> float:
        return float(self.coef[self.names.index(name)])

    def pvalue_for(self, name: str) -> float:
        return float(self.pvalue_hac[self.names.index(name)])

    def tstat_for(self, name: str) -> float:
        return float(self.tstat_hac[self.names.index(name)])


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficient vector.

    V = (X'X)^-1 [ S0 + sum_{l=1..L} w_l (S_l + S_l') ] (X'X)^-1
    with Bartlett weights w_l = 1 - l/(L+1) and S_l = sum_t u_t u_{t-l} x_t x_{t-l}'.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    u = resid.reshape(-1, 1)
    Xu = X * u  # (n,k) score contributions
    S = Xu.T @ Xu  # S0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma_lag = Xu[lag:].T @ Xu[:-lag]
        S += weight * (gamma_lag + gamma_lag.T)
    cov: np.ndarray = XtX_inv @ S @ XtX_inv
    return cov


def ols_hac(
    y: np.ndarray,
    X: np.ndarray,
    names: list[str],
    hac_lags: int | None = None,
) -> RegressionResult:
    """OLS of y on X (X must already include an intercept column) with
    Newey-West HAC standard errors.

    hac_lags defaults to the Newey-West rule-of-thumb floor(4*(n/100)^(2/9)).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    # Plain OLS SEs (homoskedastic) for reference.
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_ols = np.sqrt(np.maximum(np.diag(sigma2 * XtX_inv), 0.0))

    if hac_lags is None:
        hac_lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        hac_lags = max(hac_lags, 1)
    cov_hac = _newey_west_cov(X, resid, hac_lags)
    se_hac = np.sqrt(np.maximum(np.diag(cov_hac), 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se_hac > 0, beta / se_hac, 0.0)
    # Two-sided p using t with n-k dof.
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), df=dof)

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionResult(
        coef=beta,
        names=names,
        se_ols=se_ols,
        se_hac=se_hac,
        tstat_hac=tstat,
        pvalue_hac=pvalue,
        n=n,
        r2=r2,
        resid=resid,
        X=X,
        y=y,
    )


def block_bootstrap_pvalue(
    y: np.ndarray,
    X: np.ndarray,
    coef_index: int,
    block_len: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
    alternative: str = "less",
) -> dict[str, float]:
    """Moving-block-bootstrap one-sided p-value for a single coefficient.

    Resamples overlapping blocks of (y, X) rows to preserve serial dependence,
    refits OLS each draw, and builds the bootstrap distribution of the target
    coefficient. For alternative="less" (b1 < 0) the one-sided p-value is the
    fraction of bootstrap coefficients that are non-negative.

    Returns the point estimate, bootstrap SE, a 90% CI, and the p-value.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(y)
    # Clamp block length so a block always fits inside a (possibly short) series.
    block_len = max(1, min(block_len, n))
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len

    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    point = float(beta_hat[coef_index])

    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        yb, Xb = y[idx], X[idx]
        try:
            bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
            boots[b] = bb[coef_index]
        except np.linalg.LinAlgError:
            boots[b] = np.nan

    boots = boots[np.isfinite(boots)]
    boot_se = float(np.std(boots, ddof=1))
    ci_lo, ci_hi = (float(x) for x in np.percentile(boots, [5.0, 95.0]))

    if alternative == "less":
        pvalue = float(np.mean(boots >= 0.0))
    elif alternative == "greater":
        pvalue = float(np.mean(boots <= 0.0))
    else:  # two-sided
        pvalue = 2.0 * min(float(np.mean(boots >= 0.0)), float(np.mean(boots <= 0.0)))
        pvalue = min(pvalue, 1.0)

    return {
        "point": point,
        "boot_se": boot_se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "pvalue": pvalue,
        "n_boot": len(boots),
    }


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up; returns a boolean reject mask at FDR alpha."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if np.any(below):
        kmax = int(np.max(np.where(below)[0]))
        reject_sorted = np.zeros(m, dtype=bool)
        reject_sorted[: kmax + 1] = True
        reject[order] = reject_sorted
    return reject


# --------------------------------------------------------------------------- #
# 4. The H1' design matrix + end-to-end test runner                           #
# --------------------------------------------------------------------------- #


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = float(np.std(x))
    out: np.ndarray = (x - np.mean(x)) / s if s > 0 else x - np.mean(x)
    return out


@dataclass
class Design:
    """Assembled H1' regression design matrix (typed return of build_design)."""

    y: np.ndarray
    X: np.ndarray
    names: list[str]
    valid_mask: np.ndarray


def build_design(
    gex: np.ndarray,
    returns: np.ndarray,
    vix: np.ndarray | None = None,
    rv_window: int = 5,
    vv_window: int = 5,
    outcome: str = "vol_of_vol",
    csd_window: int = 20,
    include_dow: bool = True,
    dow: np.ndarray | None = None,
) -> Design:
    """Assemble the H1' regression: next-day outcome on standardized GEX(t)
    plus controls (VIX level, lagged realized vol, day-of-week dummies).

    The dependent variable is the outcome at t+1; all regressors are dated t,
    so the regression is strictly predictive (GEX_t -> y_{t+1}). This removes
    the mechanical same-day correlation and isolates the forward feedback claim.

    Returns a Design with y, X, names, and valid_mask.
    """
    gex = np.asarray(gex, dtype=float)
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if outcome == "vol_of_vol":
        y_raw = realized_vol_of_vol(returns, rv_window=rv_window, vv_window=vv_window)
    elif outcome == "csd_autocorr":
        y_raw = csd_autocorr(returns, window=csd_window)
    else:
        raise ValueError(f"unknown outcome: {outcome!r}")

    lagged_rv = realized_vol(returns, window=rv_window)

    # Align: y at t+1, regressors at t.  Build for t in [0, n-2].
    names: list[str] = ["const", "gex_z"]
    z_gex = _zscore(gex)

    y = y_raw[1:]  # t+1
    g_t = z_gex[:-1]
    rv_t = lagged_rv[:-1]

    const = np.ones(n - 1)
    cols: list[np.ndarray] = [const, g_t]

    names.append("lagged_rv")
    cols.append(rv_t)

    if vix is not None:
        vix = np.asarray(vix, dtype=float)
        names.append("vix")
        cols.append(_zscore(vix)[:-1])

    if include_dow:
        dow = np.arange(n) % 5 if dow is None else np.asarray(dow, dtype=int)
        # 4 dummies (drop Monday=0 as baseline).
        for d in (1, 2, 3, 4):
            names.append(f"dow_{d}")
            cols.append((dow[:-1] == d).astype(float))

    X = np.column_stack(cols)

    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return Design(y=y[valid], X=X[valid], names=names, valid_mask=valid)


@dataclass
class H1PrimeResult:
    """Outcome of the full H1' test on one window."""

    outcome: str
    n: int
    coef_gex: float
    se_hac: float
    tstat_hac: float
    pvalue_hac: float
    boot: dict[str, float]
    r2: float
    reject_sign_and_sig: bool
    regression: RegressionResult = field(repr=False)


def _decide(coef: float, boot_p: float, hac_two_sided_p: float, alpha: float) -> bool:
    """Conjunctive H1' decision: coef < 0 AND both one-sided p < alpha.

    The HAC p is two-sided, so the one-sided left-tail p (given coef < 0) is
    hac_two_sided_p / 2. Requiring both the moving-block bootstrap and the
    Newey-West HAC to agree keeps the single-window test well-calibrated under
    the null (the bootstrap alone is mildly liberal at n ~ 113).
    """
    hac_one_sided = hac_two_sided_p / 2.0
    return (coef < 0.0) and (boot_p < alpha) and (hac_one_sided < alpha)


def run_h1prime(
    gex: np.ndarray,
    returns: np.ndarray,
    vix: np.ndarray | None = None,
    outcome: str = "vol_of_vol",
    alpha: float = 0.05,
    block_len: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> H1PrimeResult:
    """Run the H1' regression + HAC + block-bootstrap on a single window."""
    d = build_design(gex, returns, vix=vix, outcome=outcome)
    res = ols_hac(d.y, d.X, d.names)
    gi = res.names.index("gex_z")
    boot = block_bootstrap_pvalue(
        d.y,
        d.X,
        coef_index=gi,
        block_len=block_len,
        n_boot=n_boot,
        seed=seed,
        alternative="less",
    )
    coef = res.coef_for("gex_z")
    reject = _decide(coef, boot["pvalue"], res.pvalue_for("gex_z"), alpha)
    return H1PrimeResult(
        outcome=outcome,
        n=res.n,
        coef_gex=coef,
        se_hac=float(res.se_hac[gi]),
        tstat_hac=res.tstat_for("gex_z"),
        pvalue_hac=res.pvalue_for("gex_z"),
        boot=boot,
        r2=res.r2,
        reject_sign_and_sig=bool(reject),
        regression=res,
    )


def run_h1prime_pooled(
    windows: list[dict[str, np.ndarray]],
    outcome: str = "vol_of_vol",
    alpha: float = 0.05,
    block_len: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> H1PrimeResult:
    """Run the H1' regression on a POOLED panel of event windows (PRIMARY).

    Each window contributes one design block built by ``build_design`` (GEX
    standardized WITHIN the window, so cross-event scale differences are
    absorbed); blocks are stacked and per-event intercept dummies are appended
    (event fixed effects). The single pooled GEX slope is identified off
    within-window variation -- exactly the short-vs-long-gamma contrast the
    mechanism predicts.

    windows : list of dicts, each {'gex','returns','vix'(optional),'dow'(optional)}.
    """
    blocks_y: list[np.ndarray] = []
    blocks_X: list[np.ndarray] = []
    blocks_event: list[np.ndarray] = []
    base_names: list[str] | None = None

    for j, w in enumerate(windows):
        d = build_design(
            w["gex"],
            w["returns"],
            vix=w.get("vix"),
            outcome=outcome,
            dow=w.get("dow"),
        )
        if base_names is None:
            base_names = list(d.names)
        blocks_y.append(d.y)
        blocks_X.append(d.X)
        blocks_event.append(np.full(len(d.y), j))

    assert base_names is not None
    y = np.concatenate(blocks_y)
    X = np.vstack(blocks_X)
    event = np.concatenate(blocks_event)

    # Replace the single 'const' with per-event intercepts (event 0 baseline).
    names = list(base_names)
    const_idx = names.index("const")
    X_noconst = np.delete(X, const_idx, axis=1)
    names_noconst = [nm for i, nm in enumerate(names) if i != const_idx]
    n_events = len(windows)
    fe_cols: list[np.ndarray] = [np.ones(len(y))]
    fe_names = ["const"]
    for j in range(1, n_events):
        fe_cols.append((event == j).astype(float))
        fe_names.append(f"event_{j}")
    Xp = np.column_stack(fe_cols + [X_noconst[:, k] for k in range(X_noconst.shape[1])])
    names_p = fe_names + names_noconst

    res = ols_hac(y, Xp, names_p)
    gi = names_p.index("gex_z")
    boot = block_bootstrap_pvalue(
        y,
        Xp,
        coef_index=gi,
        block_len=block_len,
        n_boot=n_boot,
        seed=seed,
        alternative="less",
    )
    coef = res.coef_for("gex_z")
    reject = _decide(coef, boot["pvalue"], res.pvalue_for("gex_z"), alpha)
    return H1PrimeResult(
        outcome=outcome,
        n=res.n,
        coef_gex=coef,
        se_hac=float(res.se_hac[gi]),
        tstat_hac=res.tstat_for("gex_z"),
        pvalue_hac=res.pvalue_for("gex_z"),
        boot=boot,
        r2=res.r2,
        reject_sign_and_sig=bool(reject),
        regression=res,
    )
