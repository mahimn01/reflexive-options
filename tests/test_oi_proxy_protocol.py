"""Checks for the pre-extraction A13--A15 open-interest proxy protocol."""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.empirical.gex_regression import OIGrid
from reflexive_options.empirical.oi_proxy_protocol import (
    A13Design,
    GammaBookSummary,
    benjamini_hochberg_adjusted,
    build_a13_design,
    classify_a14_family,
    gamma_book_summary,
    interpolate_zero_rate,
    put_call_parity_forward,
    run_a13_regression,
    run_a14_family,
    transform_primary_summaries,
)


def test_gamma_book_summary_uses_no_dealer_sign() -> None:
    grid = OIGrid(
        spot=100.0,
        strike=np.array([90.0, 110.0]),
        tau=np.array([0.25, 0.25]),
        sigma=np.array([0.20, 0.20]),
        oi=np.array([10.0, 10.0]),
        is_call=np.array([True, False]),
    )
    summary = gamma_book_summary(grid, forward=np.array([100.0, 100.0]))
    assert summary.unsigned_mass > 0.0
    assert -1.0 <= summary.call_put_balance <= 1.0
    assert np.log(0.9) < summary.mean_log_moneyness < np.log(1.1)
    assert summary.dispersion_log_moneyness > 0.0
    assert summary.n_input == 2
    assert summary.n_eligible == 2


def test_gamma_book_summary_applies_a14_numerical_filters_and_rate_tuple() -> None:
    grid = OIGrid(
        spot=100.0,
        strike=np.array([100.0, 100.0, 100.0, 200.0, 100.0, 105.0]),
        tau=np.array([1 / 365, 366 / 365, 0.25, 0.25, 0.25, 0.50]),
        sigma=np.array([0.20, 0.20, 0.009, 0.20, 5.01, 0.25]),
        oi=np.full(6, 10.0),
        is_call=np.array([True, True, False, False, True, False]),
    )
    summary = gamma_book_summary(
        grid,
        forward=np.full(6, 100.0),
        rate=np.full(6, 0.04),
        dividend=np.full(6, 0.015),
    )
    assert summary.n_input == 6
    assert summary.n_eligible == 2
    assert summary.unsigned_mass > 0.0


def test_put_call_parity_forward_uses_nearest_pair() -> None:
    forward = put_call_parity_forward(
        np.array([95.0, 100.0, 105.0]),
        np.array([7.0, 4.2, 2.0]),
        np.array([1.0, 4.0, 6.5]),
        rate=0.04,
        tau=0.25,
    )
    assert forward == pytest.approx(100.0 + np.exp(0.01) * 0.2)


def test_put_call_parity_forward_breaks_exact_tie_at_smaller_strike() -> None:
    forward = put_call_parity_forward(
        np.array([105.0, 95.0]),
        np.array([4.0, 3.0]),
        np.array([3.0, 2.0]),
        rate=0.0,
        tau=0.25,
    )
    assert forward == pytest.approx(96.0)


def test_zero_rate_interpolation_and_endpoint_flag() -> None:
    days = np.array([7.0, 30.0, 90.0])
    rates = np.array([0.03, 0.04, 0.05])
    rate, outside = interpolate_zero_rate(60.0, days, rates)
    assert rate == pytest.approx(0.045)
    assert not outside
    short_rate, outside = interpolate_zero_rate(1.0, days, rates)
    assert short_rate == pytest.approx(0.03)
    assert outside


def test_gamma_book_summary_rejects_zero_mass_and_bad_alignment() -> None:
    zero = OIGrid(
        spot=100.0,
        strike=np.array([100.0]),
        tau=np.array([0.25]),
        sigma=np.array([0.20]),
        oi=np.array([0.0]),
        is_call=np.array([True]),
    )
    with pytest.raises(ValueError, match="mass"):
        gamma_book_summary(zero)
    bad = OIGrid(
        spot=100.0,
        strike=np.array([100.0, 101.0]),
        tau=np.array([0.25]),
        sigma=np.array([0.20]),
        oi=np.array([1.0]),
        is_call=np.array([True]),
    )
    with pytest.raises(ValueError, match="aligned"):
        gamma_book_summary(bad)


def test_primary_summary_transform_order_and_standardization() -> None:
    summaries = [
        GammaBookSummary(100.0 + i, -0.2 + i / 100, -0.1 + i / 200, 0.1 + i / 300)
        for i in range(20)
    ]
    transformed, names = transform_primary_summaries(summaries)
    assert names == ["log_unsigned_mass", "call_put_balance", "mean_moneyness", "log_dispersion"]
    np.testing.assert_allclose(np.mean(transformed, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(transformed, axis=0), 1.0, atol=1e-12)


def test_design_dates_every_outcome_strictly_after_regressors() -> None:
    n = 60
    returns = np.linspace(-0.03, 0.02, n)
    proxy = np.linspace(-1.0, 1.0, n)
    vix = np.linspace(15.0, 25.0, n)
    dow = np.arange(n) % 5
    expiration = np.asarray(np.arange(n) % 21 == 0, dtype=float)
    design = build_a13_design(proxy, returns, vix, dow, expiration)
    assert design.regressor_day[0] == 21
    assert design.regressor_day[-1] == n - 2
    expected = np.log(returns[design.regressor_day + 1] ** 2 + 1e-8)
    np.testing.assert_allclose(design.y, expected)
    np.testing.assert_allclose(
        design.X[:, design.names.index("proxy")], proxy[design.regressor_day]
    )
    for weekday in (1, 2, 3, 4):
        column = design.X[:, design.names.index(f"dow_{weekday}")]
        np.testing.assert_array_equal(column, dow[design.regressor_day + 1] == weekday)


def test_missing_option_day_does_not_compress_the_return_calendar() -> None:
    n = 60
    returns = np.linspace(-0.03, 0.02, n)
    proxy = np.linspace(-1.0, 1.0, n)
    proxy[30] = np.nan
    design = build_a13_design(
        proxy,
        returns,
        np.full(n, 20.0),
        np.arange(n) % 5,
        np.zeros(n),
    )
    # Day 29 remains a valid regressor date and predicts the CRSP return on
    # day 30 even though day 30 has no option proxy.  Only day 30 is excluded
    # as a regressor date; the sequence is never compressed to day 31.
    row = int(np.flatnonzero(design.regressor_day == 29)[0])
    assert design.y[row] == pytest.approx(np.log(returns[30] ** 2 + 1e-8))
    assert 30 not in design.regressor_day


def test_a13_regression_uses_two_sided_inference() -> None:
    rng = np.random.default_rng(8)
    n = 180
    returns = rng.normal(0.0, 0.01, n)
    proxy = rng.normal(size=n)
    design = build_a13_design(
        proxy,
        returns,
        np.linspace(18.0, 24.0, n) + rng.normal(0.0, 0.2, n),
        np.arange(n) % 5,
        np.asarray(np.arange(n) % 21 == 0, dtype=float),
    )
    result = run_a13_regression(design)
    assert 0.0 <= result.hac_pvalue <= 1.0
    assert 0.0 <= result.bootstrap["pvalue"] <= 1.0
    assert result.bootstrap["confidence"] == pytest.approx(0.95)
    assert result.bootstrap["monte_carlo_correction"] == pytest.approx(1.0)
    assert result.bootstrap["n_boot"] == pytest.approx(2_000)
    assert result.bootstrap["seed"] == pytest.approx(42)
    assert set(result.vif) == set(design.names) - {"const"}
    assert all(value >= 1.0 for value in result.vif.values())
    assert result.n == n - 22


def test_a14_family_enforces_common_sample_and_reports_correlations() -> None:
    rng = np.random.default_rng(91)
    n = 90
    returns = rng.normal(0.0, 0.01, n)
    vix = 20.0 + rng.normal(0.0, 1.0, n)
    proxies = rng.normal(size=(n, 4))
    designs = [
        build_a13_design(
            proxies[:, index],
            returns,
            vix,
            np.arange(n) % 5,
            np.asarray(np.arange(n) % 21 == 0, dtype=float),
        )
        for index in range(4)
    ]
    family = run_a14_family(designs)
    assert len(family.regressions) == 4
    assert family.proxy_correlation.shape == (4, 4)
    np.testing.assert_allclose(np.diag(family.proxy_correlation), 1.0)
    assert len(family.decision.labels) == 4

    mismatched = list(designs)
    mismatched[3] = A13Design(
        y=designs[3].y,
        X=designs[3].X,
        names=designs[3].names,
        regressor_day=designs[3].regressor_day + 1,
    )
    with pytest.raises(ValueError, match="dates"):
        run_a14_family(mismatched)


def test_a14_design_rejects_more_than_one_expiration_control() -> None:
    n = 30
    with pytest.raises(ValueError, match="exactly one"):
        build_a13_design(
            np.arange(n, dtype=float),
            np.linspace(-0.01, 0.01, n),
            np.full(n, 20.0),
            np.arange(n) % 5,
            np.zeros((n, 2)),
        )


def test_bh_adjusted_values_are_monotone_in_rank() -> None:
    p = np.array([0.04, 0.001, 0.03, 0.20])
    adjusted = benjamini_hochberg_adjusted(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-15)
    assert adjusted[1] == pytest.approx(0.004)


def test_a14_family_classification_uses_both_adjusted_families_and_interval() -> None:
    decision = classify_a14_family(
        hac_pvalues=np.array([0.001, 0.001, 0.60, 0.001]),
        bootstrap_pvalues=np.array([0.002, 0.80, 0.70, 0.002]),
        bootstrap_ci_low=np.array([0.10, -0.20, -0.30, -0.10]),
        bootstrap_ci_high=np.array([0.40, 0.30, 0.40, 0.20]),
    )
    assert decision.labels == (
        "robustly_associated",
        "method_sensitive",
        "not_detected",
        "not_detected",
    )
    assert decision.hac_adjusted_pvalues.shape == (4,)
    assert decision.bootstrap_adjusted_pvalues.shape == (4,)
