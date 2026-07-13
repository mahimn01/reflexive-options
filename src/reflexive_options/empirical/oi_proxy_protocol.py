"""Amendments A13--A16: participant-sign-free open-interest proxy protocol.

The primary objects in this module are observable summaries of an option
open-interest grid.  None is labelled a dealer position.  Convention-signed
GEX series remain secondary measurement sensitivities in the registration.
A14 fixes the contract filters, forward construction, controls, and the rule
for classifying agreement between HAC and block-bootstrap inference. A16
adds fail-closed input requirements and complete-calendar inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reflexive_options.empirical.gex_regression import (
    OIGrid,
    RegressionResult,
    block_bootstrap_pvalue,
    bs_gamma,
    moving_block_bootstrap_indices,
    ols_hac,
)

A16_HAC_LAGS = 22
A16_BLOCK_LENGTH = 22
A14_BOOTSTRAP_DRAWS = 2_000
A14_BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class GammaBookSummary:
    """Four observable, participant-sign-free summaries and attrition counts."""

    unsigned_mass: float
    call_put_balance: float
    mean_log_moneyness: float
    dispersion_log_moneyness: float
    n_input: int | None = None
    n_eligible: int | None = None


def put_call_parity_forward(
    strikes: NDArray[np.float64],
    call_mids: NDArray[np.float64],
    put_mids: NDArray[np.float64],
    *,
    rate: float,
    tau: float,
) -> float:
    """Construct A14's expiration-level forward from the nearest parity pair.

    Inputs must already be uniquely matched call--put pairs that survived the
    quote and settlement filters.  The selected pair minimizes ``abs(C-P)``;
    exact ties are resolved at the smaller strike.
    """

    strikes = np.asarray(strikes, dtype=np.float64)
    call_mids = np.asarray(call_mids, dtype=np.float64)
    put_mids = np.asarray(put_mids, dtype=np.float64)
    if strikes.ndim != 1 or call_mids.ndim != 1 or put_mids.ndim != 1:
        raise ValueError("parity-pair arrays must be one-dimensional")
    if not (strikes.size == call_mids.size == put_mids.size) or strikes.size == 0:
        raise ValueError("parity-pair arrays must be aligned and non-empty")
    if np.any(~np.isfinite(strikes)) or np.any(strikes <= 0.0):
        raise ValueError("parity strikes must be finite and positive")
    if np.any(~np.isfinite(call_mids)) or np.any(call_mids < 0.0):
        raise ValueError("call mids must be finite and non-negative")
    if np.any(~np.isfinite(put_mids)) or np.any(put_mids < 0.0):
        raise ValueError("put mids must be finite and non-negative")
    if not np.isfinite(rate) or not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("rate must be finite and maturity must be positive")
    difference = np.abs(call_mids - put_mids)
    index = int(np.lexsort((strikes, difference))[0])
    forward = strikes[index] + np.exp(rate * tau) * (call_mids[index] - put_mids[index])
    if not np.isfinite(forward) or forward <= 0.0:
        raise ValueError("put--call parity produced a non-positive forward")
    return float(forward)


def interpolate_zero_rate(
    maturity_days: float,
    curve_days: NDArray[np.float64],
    curve_rates: NDArray[np.float64],
) -> tuple[float, bool]:
    """Linearly interpolate A14's continuous zero rate in calendar days.

    Returns the rate and a flag that is true when nearest-endpoint extension,
    rather than interior interpolation, was required.
    """

    days = np.asarray(curve_days, dtype=np.float64)
    rates = np.asarray(curve_rates, dtype=np.float64)
    if days.ndim != 1 or rates.ndim != 1 or days.size != rates.size or days.size < 2:
        raise ValueError("zero-curve arrays must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(days)) or np.any(days <= 0.0) or np.any(np.diff(days) <= 0.0):
        raise ValueError("zero-curve maturities must be finite, positive, and increasing")
    if np.any(~np.isfinite(rates)):
        raise ValueError("zero rates must be finite")
    if not np.isfinite(maturity_days) or maturity_days <= 0.0:
        raise ValueError("target maturity must be finite and positive")
    outside = bool(maturity_days < days[0] or maturity_days > days[-1])
    rate = np.interp(maturity_days, days, rates)
    return float(rate), outside


def gamma_book_summary(
    grid: OIGrid,
    *,
    forward: float | NDArray[np.float64],
    rate: float | NDArray[np.float64],
    dividend: float | NDArray[np.float64],
    scale: float = 1.0,
) -> GammaBookSummary:
    """Apply numerical A14 filters and compute the four A13 summaries.

    ``forward`` may be one maturity-specific value per contract.  It is
    mandatory so registered code cannot silently substitute spot for an A14
    parity or flagged-carry forward.  Rate and dividend inputs are mandatory
    and must match the tuple used for the corresponding forwards.  Root,
    settlement, adjustment, quote, and
    duplicate filters require vendor fields absent from :class:`OIGrid` and
    therefore must be applied upstream before calling this function.
    """

    arrays = [
        np.asarray(grid.strike, dtype=np.float64),
        np.asarray(grid.tau, dtype=np.float64),
        np.asarray(grid.sigma, dtype=np.float64),
        np.asarray(grid.oi, dtype=np.float64),
        np.asarray(grid.is_call),
    ]
    shapes = {array.shape for array in arrays}
    if (
        any(array.ndim != 1 for array in arrays)
        or len(shapes) != 1
        or not shapes
        or next(iter(shapes))[0] == 0
    ):
        raise ValueError("OI-grid arrays must be aligned and non-empty")
    strike, tau, sigma, oi, is_call = arrays
    if is_call.dtype != np.bool_:
        raise ValueError("is_call must be a validated boolean array")
    if (
        not np.isfinite(grid.spot)
        or not np.isfinite(grid.contract_multiplier)
        or grid.spot <= 0.0
        or grid.contract_multiplier != 100.0
    ):
        raise ValueError(
            "spot must be positive and the registered contract multiplier must equal 100"
        )
    if np.any(~np.isfinite(strike)) or np.any(strike <= 0.0):
        raise ValueError("strikes must be finite and positive")
    if np.any(~np.isfinite(tau)) or np.any(~np.isfinite(sigma)):
        raise ValueError("maturities and volatilities must be finite")
    if np.any(~np.isfinite(oi)) or np.any(oi < 0.0):
        raise ValueError("open interest must be finite and non-negative")

    forwards = np.broadcast_to(np.asarray(forward, dtype=np.float64), strike.shape)
    if np.any(~np.isfinite(forwards)) or np.any(forwards <= 0.0):
        raise ValueError("forwards must be finite and positive")

    rates = np.broadcast_to(np.asarray(rate, dtype=np.float64), strike.shape)
    dividends = np.broadcast_to(np.asarray(dividend, dtype=np.float64), strike.shape)
    if np.any(~np.isfinite(rates)) or np.any(~np.isfinite(dividends)):
        raise ValueError("rates and dividends must be finite")
    log_moneyness = np.log(strike / forwards)
    eligible = (
        (tau > 0.0)
        & (tau <= 1.0)
        & (sigma >= 0.01)
        & (sigma <= 5.0)
        & (np.abs(log_moneyness) <= 0.50)
    )
    n_input = int(strike.size)
    n_eligible = int(np.sum(eligible))
    if n_eligible == 0:
        raise ValueError("no contracts survive the registered numerical filters")

    strike = strike[eligible]
    tau = tau[eligible]
    sigma = sigma[eligible]
    oi = oi[eligible]
    is_call = is_call[eligible]
    rates = rates[eligible]
    dividends = dividends[eligible]
    log_moneyness = log_moneyness[eligible]
    gamma = bs_gamma(grid.spot, strike, tau, sigma, r=rates, q=dividends)
    weights = oi * gamma * grid.spot**2 * 0.01 * grid.contract_multiplier
    total_unscaled = float(np.sum(weights))
    if not np.isfinite(total_unscaled) or total_unscaled <= 0.0:
        raise ValueError("gamma-weighted open-interest mass must be positive")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    call_mass = float(np.sum(weights[is_call]))
    put_mass = float(np.sum(weights[~is_call]))
    balance = (call_mass - put_mass) / total_unscaled
    mean = float(np.sum(weights * log_moneyness) / total_unscaled)
    dispersion = float(np.sqrt(np.sum(weights * (log_moneyness - mean) ** 2) / total_unscaled))
    return GammaBookSummary(
        unsigned_mass=scale * total_unscaled,
        call_put_balance=balance,
        mean_log_moneyness=mean,
        dispersion_log_moneyness=dispersion,
        n_input=n_input,
        n_eligible=n_eligible,
    )


def transform_primary_summaries(
    summaries: list[GammaBookSummary],
) -> tuple[NDArray[np.float64], list[str]]:
    """Return the four standardized A13 regressors in registered order."""

    if len(summaries) < 2:
        raise ValueError("at least two daily summaries are required")
    raw = np.array(
        [
            [
                np.log(summary.unsigned_mass),
                summary.call_put_balance,
                summary.mean_log_moneyness,
                np.log(summary.dispersion_log_moneyness + 1.0e-6),
            ]
            for summary in summaries
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(raw)):
        raise ValueError("primary summary transforms must be finite")
    std = np.std(raw, axis=0)
    if np.any(std <= 0.0):
        raise ValueError("every primary summary must vary over the sample")
    transformed = (raw - np.mean(raw, axis=0)) / std
    names = ["log_unsigned_mass", "call_put_balance", "mean_moneyness", "log_dispersion"]
    return transformed, names


@dataclass(frozen=True)
class A13Design:
    """Strictly forward A13--A16 design and its complete-calendar dates."""

    y: NDArray[np.float64]
    X: NDArray[np.float64]
    names: list[str]
    regressor_day: NDArray[np.int64]


def build_a13_design(
    proxy: NDArray[np.float64],
    returns: NDArray[np.float64],
    vix: NDArray[np.float64],
    day_of_week: NDArray[np.int64],
    expiration: NDArray[np.float64],
    *,
    spot: NDArray[np.float64],
    outcome: str = "log_squared_return",
) -> A13Design:
    """Build the A13 equation subject to A14's control locks.

    ``returns`` contains CRSP simple returns transformed upstream as
    ``log1p(sprtrn)``.  Arrays must remain on the complete ordered CRSP market
    calendar: a missing option proxy is represented by NaN and must never be
    removed before leads and lags are constructed.  This ensures ``t+1`` is
    the next trading session rather than the next date with an option chain.
    Day of week is encoded Monday=0 through Friday=4.
    ``expiration`` is exactly one monthly-expiration-session indicator.
    Every regression variable is dated at or before ``t`` and the outcome is
    the immediately following CRSP trading session.
    """

    proxy = np.asarray(proxy, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    vix = np.asarray(vix, dtype=np.float64)
    spot = np.asarray(spot, dtype=np.float64)
    day_of_week_raw = np.asarray(day_of_week)
    if np.any(~np.isfinite(day_of_week_raw)) or np.any(
        day_of_week_raw != np.floor(day_of_week_raw)
    ):
        raise ValueError("day_of_week must be finite and integer-valued")
    day_of_week = day_of_week_raw.astype(np.int64)
    expiration = np.asarray(expiration, dtype=np.float64)
    arrays = (proxy, returns, vix, spot, day_of_week)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("daily arrays must be one-dimensional")
    n = returns.size
    if not (proxy.size == vix.size == spot.size == day_of_week.size == n):
        raise ValueError("daily arrays must have equal lengths")
    if expiration.ndim == 1:
        expiration = expiration[:, None]
    if expiration.ndim != 2 or expiration.shape != (n, 1):
        raise ValueError("expiration must be exactly one daily control")
    if n < 24:
        raise ValueError("at least 24 days are required for 22-day controls and a lead")
    if outcome == "log_squared_return":
        outcome_series = np.log(returns * returns + 1.0e-8)
    elif outcome == "absolute_return":
        outcome_series = np.abs(returns)
    else:
        raise ValueError(f"unknown outcome {outcome!r}")
    if np.any(np.isfinite(vix) & (vix <= 0.0)):
        raise ValueError("VIX must be positive before taking logs")
    if np.any(np.isfinite(spot) & (spot <= 0.0)):
        raise ValueError("spot must be positive before taking logs")
    if np.any((day_of_week < 0) | (day_of_week > 4)):
        raise ValueError("day_of_week must encode Monday=0 through Friday=4")
    if np.any(np.isfinite(expiration) & ~np.isin(expiration, (0.0, 1.0))):
        raise ValueError("monthly expiration control must be binary")

    rows: list[list[float]] = []
    targets: list[float] = []
    regressor_days: list[int] = []
    for t in range(21, n - 1):
        row = [
            1.0,
            float(proxy[t]),
            float(outcome_series[t]),
            float(np.mean(outcome_series[t - 4 : t + 1])),
            float(np.mean(outcome_series[t - 21 : t + 1])),
            float(np.log(vix[t])),
            float(np.log(spot[t])),
            float((t - 0.5 * (n - 1)) / n),
        ]
        # The calendar-known weekday of the outcome session controls its
        # seasonality.  Using DOW_t instead is equivalent on most ordinary
        # weeks but fails around exchange holidays.
        row.extend(float(day_of_week[t + 1] == weekday) for weekday in (1, 2, 3, 4))
        row.extend(float(value) for value in expiration[t])
        if np.isfinite(outcome_series[t + 1]) and np.all(np.isfinite(row)):
            rows.append(row)
            targets.append(float(outcome_series[t + 1]))
            regressor_days.append(t)

    names = [
        "const",
        "proxy",
        "y_lag1",
        "y_mean5",
        "y_mean22",
        "log_vix",
        "log_spot",
        "linear_session_trend",
    ]
    names.extend(f"dow_{weekday}" for weekday in (1, 2, 3, 4))
    names.append("monthly_expiration")
    return A13Design(
        y=np.asarray(targets, dtype=np.float64),
        X=np.asarray(rows, dtype=np.float64),
        names=names,
        regressor_day=np.asarray(regressor_days, dtype=np.int64),
    )


@dataclass(frozen=True)
class A13RegressionResult:
    """Two-sided HAC and bootstrap results; neither is selected ex post."""

    n: int
    coefficient: float
    hac_pvalue: float
    bootstrap: dict[str, float]
    vif: dict[str, float]
    regression: RegressionResult = field(repr=False)


def variance_inflation_factors(design: NDArray[np.float64], names: list[str]) -> dict[str, float]:
    """Compute auxiliary-regression VIFs for every non-intercept column."""

    X = np.asarray(design, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(names) or X.shape[0] == 0:
        raise ValueError("design and names must describe a non-empty matrix")
    if np.any(~np.isfinite(X)):
        raise ValueError("VIF design must be finite")
    output: dict[str, float] = {}
    for index, name in enumerate(names):
        if name == "const":
            continue
        target = X[:, index]
        centered = target - np.mean(target)
        total = float(centered @ centered)
        if total <= np.finfo(float).eps:
            output[name] = float("inf")
            continue
        auxiliary = np.delete(X, index, axis=1)
        coefficient, *_ = np.linalg.lstsq(auxiliary, target, rcond=None)
        residual = target - auxiliary @ coefficient
        r_squared = 1.0 - float(residual @ residual) / total
        denominator = 1.0 - min(max(r_squared, 0.0), 1.0)
        output[name] = float("inf") if denominator <= np.finfo(float).eps else 1.0 / denominator
    return output


def run_a13_regression(
    design: A13Design,
    *,
    bootstrap_indices: NDArray[np.int64] | None = None,
) -> A13RegressionResult:
    """Fit one registered equation with every A14 inference setting fixed."""

    if design.X.ndim != 2 or design.y.ndim != 1 or design.X.shape[0] != design.y.size:
        raise ValueError("A14 design arrays must be aligned")
    n, k = design.X.shape
    if n <= k:
        raise ValueError("primary regression requires more complete rows than coefficients")
    if np.linalg.matrix_rank(design.X) < k:
        raise ValueError("primary regression design is rank deficient")
    regression = ols_hac(
        design.y,
        design.X,
        design.names,
        hac_lags=A16_HAC_LAGS,
        time_index=design.regressor_day,
    )
    proxy_index = design.names.index("proxy")
    bootstrap = block_bootstrap_pvalue(
        design.y,
        design.X,
        coef_index=proxy_index,
        block_len=A16_BLOCK_LENGTH,
        n_boot=A14_BOOTSTRAP_DRAWS,
        seed=A14_BOOTSTRAP_SEED,
        alternative="two-sided",
        confidence=0.95,
        monte_carlo_correction=True,
        time_index=design.regressor_day,
        resample_indices=bootstrap_indices,
    )
    return A13RegressionResult(
        n=regression.n,
        coefficient=regression.coef_for("proxy"),
        hac_pvalue=regression.pvalue_for("proxy"),
        bootstrap=bootstrap,
        vif=variance_inflation_factors(design.X, design.names),
        regression=regression,
    )


def benjamini_hochberg_adjusted(pvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    """Benjamini--Hochberg adjusted p-values in the original input order."""

    pvalues = np.asarray(pvalues, dtype=np.float64)
    if pvalues.ndim != 1 or pvalues.size == 0:
        raise ValueError("pvalues must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(pvalues)) or np.any((pvalues < 0.0) | (pvalues > 1.0)):
        raise ValueError("pvalues must be finite and lie in [0, 1]")
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted_ranked = ranked * pvalues.size / np.arange(1, pvalues.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


@dataclass(frozen=True)
class A14FamilyDecision:
    """Separate BH adjustments and the four locked A14 evidence labels."""

    hac_adjusted_pvalues: NDArray[np.float64]
    bootstrap_adjusted_pvalues: NDArray[np.float64]
    labels: tuple[str, str, str, str]


@dataclass(frozen=True)
class A14FamilyResult:
    """The locked four-regression family and its common-sample diagnostics."""

    regressions: tuple[
        A13RegressionResult,
        A13RegressionResult,
        A13RegressionResult,
        A13RegressionResult,
    ]
    decision: A14FamilyDecision
    proxy_correlation: NDArray[np.float64]
    bootstrap_candidates_attempted: int
    bootstrap_rank_deficient_discarded: int


def classify_a14_family(
    hac_pvalues: NDArray[np.float64],
    bootstrap_pvalues: NDArray[np.float64],
    bootstrap_ci_low: NDArray[np.float64],
    bootstrap_ci_high: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> A14FamilyDecision:
    """Apply A14's non-selective evidence rule to exactly four regressions."""

    inputs = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (hac_pvalues, bootstrap_pvalues, bootstrap_ci_low, bootstrap_ci_high)
    )
    if any(values.ndim != 1 or values.size != 4 for values in inputs):
        raise ValueError("A14 requires exactly four values from each inferential output")
    hac, bootstrap, ci_low, ci_high = inputs
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if np.any(~np.isfinite(ci_low)) or np.any(~np.isfinite(ci_high)):
        raise ValueError("bootstrap interval endpoints must be finite")
    if np.any(ci_low > ci_high):
        raise ValueError("bootstrap interval lower endpoints cannot exceed upper endpoints")

    hac_adjusted = benjamini_hochberg_adjusted(hac)
    bootstrap_adjusted = benjamini_hochberg_adjusted(bootstrap)
    hac_reject = hac_adjusted < alpha
    bootstrap_reject = bootstrap_adjusted < alpha
    interval_excludes_zero = (ci_high < 0.0) | (ci_low > 0.0)
    labels: list[str] = []
    for hac_hit, bootstrap_hit, excludes_zero in zip(
        hac_reject, bootstrap_reject, interval_excludes_zero, strict=True
    ):
        if hac_hit and bootstrap_hit and excludes_zero:
            labels.append("robustly_associated")
        elif bool(hac_hit) ^ bool(bootstrap_hit):
            labels.append("method_sensitive")
        else:
            labels.append("not_detected")
    return A14FamilyDecision(
        hac_adjusted_pvalues=hac_adjusted,
        bootstrap_adjusted_pvalues=bootstrap_adjusted,
        labels=(labels[0], labels[1], labels[2], labels[3]),
    )


def run_a14_family(designs: list[A13Design]) -> A14FamilyResult:
    """Fit exactly four equations after enforcing a common outcome/control sample."""

    if len(designs) != 4:
        raise ValueError("A14 requires exactly four primary designs")
    reference = designs[0]
    proxy_index = reference.names.index("proxy")
    control_indices = [index for index in range(len(reference.names)) if index != proxy_index]
    for design in designs[1:]:
        if design.names != reference.names:
            raise ValueError("all A14 designs must use identical regressor names")
        if not np.array_equal(design.regressor_day, reference.regressor_day):
            raise ValueError("all A14 designs must use identical regressor dates")
        if not np.array_equal(design.y, reference.y):
            raise ValueError("all A14 designs must use an identical outcome")
        if not np.array_equal(design.X[:, control_indices], reference.X[:, control_indices]):
            raise ValueError("all A14 designs must use identical controls")

    candidates = moving_block_bootstrap_indices(
        reference.y.size,
        block_len=A16_BLOCK_LENGTH,
        n_draws=20 * A14_BOOTSTRAP_DRAWS,
        seed=A14_BOOTSTRAP_SEED,
        time_index=reference.regressor_day,
    )
    accepted: list[NDArray[np.int64]] = []
    attempted = 0
    rank_deficient = 0
    for indices in candidates:
        attempted += 1
        if all(np.linalg.matrix_rank(design.X[indices]) == design.X.shape[1] for design in designs):
            accepted.append(indices)
            if len(accepted) == A14_BOOTSTRAP_DRAWS:
                break
        else:
            rank_deficient += 1
    if len(accepted) < A14_BOOTSTRAP_DRAWS:
        raise RuntimeError(
            "could not obtain 2,000 resamples that are full rank for all four A14 designs"
        )
    shared_indices = np.asarray(accepted, dtype=np.int64)
    fitted_list = [
        run_a13_regression(design, bootstrap_indices=shared_indices) for design in designs
    ]
    fitted = (fitted_list[0], fitted_list[1], fitted_list[2], fitted_list[3])
    decision = classify_a14_family(
        hac_pvalues=np.array([result.hac_pvalue for result in fitted]),
        bootstrap_pvalues=np.array([result.bootstrap["pvalue"] for result in fitted]),
        bootstrap_ci_low=np.array([result.bootstrap["ci_lo"] for result in fitted]),
        bootstrap_ci_high=np.array([result.bootstrap["ci_hi"] for result in fitted]),
    )
    proxy_matrix = np.column_stack([design.X[:, design.names.index("proxy")] for design in designs])
    proxy_correlation = np.asarray(np.corrcoef(proxy_matrix, rowvar=False), dtype=np.float64)
    return A14FamilyResult(
        regressions=fitted,
        decision=decision,
        proxy_correlation=proxy_correlation,
        bootstrap_candidates_attempted=attempted,
        bootstrap_rank_deficient_discarded=rank_deficient,
    )
