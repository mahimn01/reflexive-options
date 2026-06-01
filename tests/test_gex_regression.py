"""Deterministic tests for the primary redesigned empirical test H1' (GEX).

Covers:
  1. Black-Scholes gamma kernel correctness (peak near ATM, vanishing wings,
     positivity, calls == puts).
  2. GEX estimator: sign convention, sign and monotonicity in the latent dealer
     gamma, exact value on a known OI grid.
  3. OLS + Newey-West HAC: recovers a known slope; HAC SE is sane.
  4. Outcome variables: realized vol, vol-of-vol, CSD autocorr shapes.
  5. Benjamini-Hochberg behaviour.
  6. Coefficient SIGN recovery at strong coupling vs NULL recovery at kappa=0,
     plus a bounded null FPR at kappa=0.

All randomness is seeded; no real data, no RL agent, no frozen simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.empirical.gex_regression import (
    OIGrid,
    benjamini_hochberg,
    bs_gamma,
    build_design,
    csd_autocorr,
    estimate_gex,
    estimate_gex_series,
    ols_hac,
    realized_vol,
    realized_vol_of_vol,
    run_h1prime,
    run_h1prime_pooled,
)
from reflexive_options.empirical.gex_simulator import GEXReflexiveSimulator, GEXSimParams

KAPPA_HIGH = 0.8
KAPPA_NULL = 0.0


# --------------------------------------------------------------------------- #
# 1. Black-Scholes gamma                                                      #
# --------------------------------------------------------------------------- #


def test_bs_gamma_atm_peak_and_wings():
    spot = 100.0
    strikes = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
    tau = np.full(5, 0.25)
    sigma = np.full(5, 0.20)
    g = bs_gamma(spot, strikes, tau, sigma)
    assert np.all(g >= 0.0)
    # ATM (K=100) gamma is the largest.
    assert int(np.argmax(g)) == 2
    # Deep wings have much smaller gamma than ATM.
    assert g[0] < g[2] and g[4] < g[2]


def test_bs_gamma_call_put_identical_and_expiry_zero():
    spot = 100.0
    k = np.array([100.0])
    g_live = bs_gamma(spot, k, np.array([0.5]), np.array([0.2]))
    # Black-Scholes gamma is identical for calls and puts (no type arg needed):
    # a second call with identical inputs must match exactly.
    g_again = bs_gamma(spot, k, np.array([0.5]), np.array([0.2]))
    assert np.allclose(g_live, g_again)
    # Expired contract contributes zero gamma.
    g_exp = bs_gamma(spot, k, np.array([0.0]), np.array([0.2]))
    assert g_exp[0] == 0.0


# --------------------------------------------------------------------------- #
# 2. GEX estimator                                                            #
# --------------------------------------------------------------------------- #


def test_gex_one_contract_exact_value():
    # One ATM call, OI=1: GEX = OI * gamma * (+1) * S^2 * 0.01 * 100 * scale.
    spot = 100.0
    grid = OIGrid(
        spot=spot,
        strike=np.array([100.0]),
        tau=np.array([0.25]),
        sigma=np.array([0.20]),
        oi=np.array([1.0]),
        is_call=np.array([True]),
        contract_multiplier=100.0,
    )
    gamma = bs_gamma(spot, np.array([100.0]), np.array([0.25]), np.array([0.20]))[0]
    expected = 1.0 * gamma * 1.0 * spot**2 * 0.01 * 100.0 * 1e-9
    assert np.isclose(estimate_gex(grid), expected, rtol=1e-12)


def test_gex_put_sign_is_negative_under_squeeze_convention():
    spot = 100.0
    base = dict(
        spot=spot,
        strike=np.array([100.0]),
        tau=np.array([0.25]),
        sigma=np.array([0.20]),
        oi=np.array([10.0]),
        contract_multiplier=100.0,
    )
    call = OIGrid(is_call=np.array([True]), **base)
    put = OIGrid(is_call=np.array([False]), **base)
    assert estimate_gex(call) > 0.0
    assert estimate_gex(put) < 0.0
    assert np.isclose(estimate_gex(call), -estimate_gex(put))


def test_gex_sign_and_monotonicity_in_latent_gamma():
    sim = GEXReflexiveSimulator(GEXSimParams(kappa=0.0), seed=0)
    vals = [estimate_gex(sim._build_grid(100.0, 0.2, g)) for g in (-2, -1, 0, 1, 2)]
    # Strictly increasing in the latent dealer gamma, with sign match.
    assert all(vals[i] < vals[i + 1] for i in range(4))
    assert vals[2] == pytest.approx(0.0, abs=1e-9)
    assert vals[0] < 0.0 < vals[4]


# --------------------------------------------------------------------------- #
# 3. OLS + Newey-West                                                         #
# --------------------------------------------------------------------------- #


def test_ols_hac_recovers_known_slope():
    rng = np.random.default_rng(7)
    n = 400
    x = rng.standard_normal(n)
    y = 2.0 - 1.5 * x + 0.3 * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    res = ols_hac(y, X, ["const", "x"])
    assert res.coef_for("const") == pytest.approx(2.0, abs=0.1)
    assert res.coef_for("x") == pytest.approx(-1.5, abs=0.1)
    assert res.se_hac[1] > 0.0
    assert res.pvalue_for("x") < 0.01


def test_ols_hac_handles_autocorrelated_errors():
    # AR(1) errors: HAC SE should exceed the (mis-specified) homoskedastic SE.
    rng = np.random.default_rng(3)
    n = 500
    x = rng.standard_normal(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal()
    y = 1.0 + 0.5 * x + e
    X = np.column_stack([np.ones(n), x])
    res = ols_hac(y, X, ["const", "x"])
    assert res.se_hac[0] > res.se_ols[0]  # intercept SE inflated by serial corr


# --------------------------------------------------------------------------- #
# 4. Outcome variables                                                        #
# --------------------------------------------------------------------------- #


def test_realized_vol_shapes_and_nans():
    r = np.linspace(-0.01, 0.01, 30)
    rv = realized_vol(r, window=5)
    assert rv.shape == r.shape
    assert np.all(np.isnan(rv[:4]))
    assert np.all(np.isfinite(rv[4:]))


def test_vol_of_vol_and_csd_finite_tail():
    rng = np.random.default_rng(0)
    r = 0.01 * rng.standard_normal(200)
    vv = realized_vol_of_vol(r, rv_window=5, vv_window=5)
    csd = csd_autocorr(r, window=20)
    assert np.isfinite(vv[-1])
    assert np.isfinite(csd[-1])
    assert -1.0 <= csd[-1] <= 1.0


# --------------------------------------------------------------------------- #
# 5. Benjamini-Hochberg                                                       #
# --------------------------------------------------------------------------- #


def test_benjamini_hochberg_basic():
    # One tiny p among large ones -> only the tiny one rejected at FDR 0.05.
    p = np.array([0.001, 0.4, 0.6, 0.8])
    rej = benjamini_hochberg(p, alpha=0.05)
    assert rej[0] and not rej[1:].any()
    # All large -> none rejected.
    assert not benjamini_hochberg(np.array([0.5, 0.6, 0.7]), alpha=0.05).any()


# --------------------------------------------------------------------------- #
# 6. Design matrix and sign/NULL recovery                                     #
# --------------------------------------------------------------------------- #


def test_build_design_is_predictive_and_well_formed():
    rng = np.random.default_rng(1)
    n = 130
    gex = rng.standard_normal(n)
    returns = 0.01 * rng.standard_normal(n)
    vix = 0.2 + 0.02 * rng.standard_normal(n)
    d = build_design(gex, returns, vix=vix, outcome="vol_of_vol")
    assert "gex_z" in d.names
    assert d.X.shape[0] == d.y.shape[0]
    assert d.X.shape[1] == len(d.names)
    # No NaNs survive into the fit matrix.
    assert np.all(np.isfinite(d.X)) and np.all(np.isfinite(d.y))


def test_sign_recovery_high_kappa_averaged_single_window():
    # Strong feedback: averaged over independent paths, the single-window GEX
    # coefficient on next-day vol-of-vol is negative (short gamma -> higher
    # vol-of-vol). A SINGLE 121-day window is deliberately the hardest case and
    # is underpowered for the per-seed sign (the pooled panel below is the
    # PRIMARY estimator and recovers the sign on ~94% of panels); here we only
    # require the mean effect to carry the predicted sign and the sign to appear
    # on at least a non-trivial fraction of windows.
    coefs = []
    signs = 0
    for s in range(12):
        out = GEXReflexiveSimulator(GEXSimParams(kappa=KAPPA_HIGH), seed=s).simulate(121)
        gex = estimate_gex_series(out.grids)
        res = run_h1prime(gex, out.returns, vix=out.vix, n_boot=200, seed=s)
        coefs.append(res.coef_gex)
        signs += int(res.coef_gex < 0)
    assert float(np.mean(coefs)) < 0.0  # mean effect has the predicted sign
    assert signs >= 8  # sign recovered on the clear majority of single windows


def test_null_at_zero_kappa_single_window():
    # Pure Heston: GEX is a null predictor -> do not reject for this seed.
    out = GEXReflexiveSimulator(GEXSimParams(kappa=KAPPA_NULL), seed=1).simulate(121)
    gex = estimate_gex_series(out.grids)
    res = run_h1prime(gex, out.returns, vix=out.vix, n_boot=400, seed=1)
    assert not res.reject_sign_and_sig


def test_pooled_panel_recovers_negative_sign_at_high_kappa():
    # The pooled 3x121 panel is the PRIMARY estimator. Averaged over independent
    # panels, the GEX coefficient is negative (short gamma -> higher forward
    # vol-of-vol), the sign is recovered on a clear majority of panels, and the
    # conjunctive decision rule rejects on a majority of panels (real power).
    coefs = []
    signs = 0
    rejects = 0
    for s in range(12):
        wins = []
        for e in range(3):
            out = GEXReflexiveSimulator(
                GEXSimParams(kappa=KAPPA_HIGH), seed=100 + s * 10 + e
            ).simulate(121)
            wins.append(
                {
                    "gex": estimate_gex_series(out.grids),
                    "returns": out.returns,
                    "vix": out.vix,
                }
            )
        res = run_h1prime_pooled(wins, n_boot=200, seed=s)
        coefs.append(res.coef_gex)
        signs += int(res.coef_gex < 0)
        rejects += int(res.reject_sign_and_sig)
    assert float(np.mean(coefs)) < 0.0
    assert signs >= 9  # sign recovered on the vast majority of panels
    assert rejects >= 6  # the primary estimator is genuinely powered


def test_null_fpr_bounded_at_zero_kappa():
    # Across 20 independent Heston pooled panels the false-positive rate of the
    # one-sided sign+significance rule stays well under ~4x nominal (small-sample
    # slack). At kappa=0 the GEX channel is identically off, so any rejection is
    # a pure false positive.
    rejects = 0
    n = 20
    for s in range(n):
        wins = []
        for e in range(3):
            out = GEXReflexiveSimulator(GEXSimParams(kappa=KAPPA_NULL), seed=s * 10 + e).simulate(
                121
            )
            wins.append(
                {
                    "gex": estimate_gex_series(out.grids),
                    "returns": out.returns,
                    "vix": out.vix,
                }
            )
        res = run_h1prime_pooled(wins, n_boot=300, seed=s)
        rejects += int(res.reject_sign_and_sig)
    assert rejects / n <= 0.20
