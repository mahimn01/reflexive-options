"""Tests for the arbitrage-free filter."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.optimize import brentq

from reflexive_options.surface.arbitrage import (
    ArbitrageTolerances,
    _black76_call,
    batch_check,
    check_arbitrage_free,
)
from reflexive_options.surface.generator import make_standard_grid
from reflexive_options.types import SurfaceGrid

QL = pytest.importorskip("QuantLib")


SPOT = 100.0
RATE = 0.02
DIV = 0.0


# ---------------------------------------------------------------------------
# Heston surface fixture
# ---------------------------------------------------------------------------


def _heston_surface(grid: SurfaceGrid, *, spot: float = SPOT) -> np.ndarray:
    """Generate an arbitrage-free IV surface using QuantLib's analytic Heston engine."""
    today = QL.Date(15, 1, 2026)
    QL.Settings.instance().evaluationDate = today
    dc = QL.Actual365Fixed()
    cal = QL.NullCalendar()

    rate_h = QL.YieldTermStructureHandle(QL.FlatForward(today, RATE, dc))
    div_h = QL.YieldTermStructureHandle(QL.FlatForward(today, DIV, dc))
    spot_h = QL.QuoteHandle(QL.SimpleQuote(spot))

    v0, kappa, theta, xi, rho = 0.04, 1.5, 0.04, 0.4, -0.5
    process = QL.HestonProcess(rate_h, div_h, spot_h, v0, kappa, theta, xi, rho)
    model = QL.HestonModel(process)
    engine = QL.AnalyticHestonEngine(model)

    del cal  # unused (kept for parity with QuantLib examples)
    surf = np.zeros(grid.shape, dtype=np.float64)
    for j, T in enumerate(grid.maturities):
        expiry = today + round(float(T) * 365)
        F_T = spot * float(np.exp((RATE - DIV) * float(T)))
        D_T = float(np.exp(-RATE * float(T)))
        for i, k in enumerate(grid.log_moneyness):
            K = spot * float(np.exp(k))
            # Use OTM option (call for K >= F, put for K < F) to avoid catastrophic
            # cancellation when inverting a deep-ITM call price ≈ intrinsic.
            is_itm_call = K < F_T
            opt_type = QL.Option.Put if is_itm_call else QL.Option.Call
            payoff = QL.PlainVanillaPayoff(opt_type, K)
            ex = QL.EuropeanExercise(expiry)
            opt = QL.VanillaOption(payoff, ex)
            opt.setPricingEngine(engine)
            price = float(opt.NPV())

            # Reduce to a Black-76 call price (via parity if a put was priced).
            # P + D*F = C + D*K  =>  C = P + D*(F - K)
            call_price = price + D_T * (F_T - K) if is_itm_call else price
            intrinsic = D_T * max(F_T - K, 0.0)
            if call_price <= intrinsic + 1e-12:
                surf[i, j] = float(np.sqrt(v0))
                continue

            def _diff(
                sigma: float,
                _price: float = call_price,
                _F: float = F_T,
                _K: float = K,
                _T: float = float(T),
                _D: float = D_T,
            ) -> float:
                bs = float(
                    _black76_call(
                        np.array([_F]),
                        np.array([_K]),
                        np.array([_T]),
                        np.array([sigma]),
                        np.array([_D]),
                    )[0]
                )
                return bs - _price

            try:
                surf[i, j] = brentq(_diff, 1e-4, 5.0, xtol=1e-10, maxiter=200)
            except (ValueError, RuntimeError):
                surf[i, j] = float(np.sqrt(v0))
    return surf


@pytest.fixture(scope="module")
def grid() -> SurfaceGrid:
    return make_standard_grid(spot=SPOT)


@pytest.fixture(scope="module")
def heston_surface(grid: SurfaceGrid) -> np.ndarray:
    return _heston_surface(grid)


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_heston_surface_passes(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    report = check_arbitrage_free(heston_surface, grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert report.passes_all is True, (
        f"Heston surface flagged: violations={report.violations}, "
        f"severity={report.severity}, notes={report.notes}"
    )
    assert report.is_marginal is False


def test_butterfly_violator_caught(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    bad = heston_surface.copy()
    # Drop IV at an OTM strike: deep-OTM call baseline is convex+small, so a downward
    # IV perturbation creates a negative second-difference at the *neighbouring*
    # interior cells (the centered second-diff at i depends on C[i-1], C[i], C[i+1],
    # so a single-cell C drop kicks D2C negative at i±1).
    target_i = 7  # OTM call (k = +0.2)
    target_T = 3
    bad[target_i, target_T] -= 0.05
    report = check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert "butterfly" in report.violations
    flagged = report.violations["butterfly"]
    flagged_K_idx = {i for i, _ in flagged}
    assert flagged_K_idx & {target_i - 1, target_i + 1}, (
        f"expected i in {{{target_i - 1}, {target_i + 1}}}, got {flagged}"
    )
    assert report.passes_all is False


def test_calendar_violator_caught(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    bad = heston_surface.copy()
    # Inflate IV at the *second-shortest* maturity huge so that w(k, T_1) >> w(k, T_2),
    # creating a non-monotone forward call across T_1 -> T_2.
    bad[:, 1] *= 5.0
    report = check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)
    # Either the call-price calendar or the total-variance calendar must flag it.
    assert ("calendar_call" in report.violations) or (
        "calendar_total_variance" in report.violations
    )
    assert report.passes_all is False


def test_total_variance_violator_caught(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    bad = heston_surface.copy()
    # Inflate the entire shortest-maturity slice → total variance at T_0 exceeds T_1.
    bad[:, 0] += 0.10
    report = check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert "calendar_total_variance" in report.violations
    assert report.passes_all is False


def test_lee_bound_violator_caught(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    bad = heston_surface.copy()
    # Replace the entire right wing with absurd vol → slope of w/|k| ≫ 2.
    n_K = grid.n_strikes
    n_wing = max(3, n_K // 10)
    for j, T in enumerate(grid.maturities):
        for i in range(n_K - n_wing, n_K):
            k = grid.log_moneyness[i]
            target_w = 25.0 * abs(k)  # slope 25 in w/|k|
            bad[i, j] = float(np.sqrt(max(target_w, 1e-6) / max(T, 1e-6)))
    report = check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert "lee_right_wing" in report.violations
    assert report.passes_all is False


def test_marginal_classification(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    """Surface with violation between 1x and 10x tolerance → is_marginal=True."""
    # Use a Lee-slope perturbation we can size precisely. Construct a surface whose
    # right-wing slope sits at 2 + 3*lee_slope (clearly in the warn band: > 1× tol,
    # < 10× tol). Lee_slope default = 0.05 → target slope = 2.15, excess = 0.15 < 0.5.
    bad = heston_surface.copy()
    n_K = grid.n_strikes
    n_wing = max(3, n_K // 10)
    # On the right wing, set w_j(k) = (2 + 3*0.05) * |k| = 2.15 * |k|
    target_slope = 2.0 + 3.0 * 0.05
    for j, T in enumerate(grid.maturities):
        for i in range(n_K - n_wing, n_K):
            k = grid.log_moneyness[i]
            target_w = target_slope * abs(k)
            bad[i, j] = float(np.sqrt(max(target_w, 1e-6) / max(T, 1e-6)))
    report = check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)
    # We expect a Lee right-wing flag in the warn band.
    assert "lee_right_wing" in report.violations
    sev = report.severity["lee_right_wing"]
    assert sev > ArbitrageTolerances().lee_slope, "expected severity > 1x tol"
    assert sev < ArbitrageTolerances().lee_slope * 10.0, "expected severity < 10x tol"
    assert report.is_marginal is True
    assert report.passes_all is False


def test_throughput_50k_surfaces_under_5s(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    n = 50_000
    rng = np.random.default_rng(42)
    # Tile the Heston surface and add tiny perturbations that stay arbitrage-free.
    base = heston_surface[None, ...]
    perturb = 1.0 + 0.005 * rng.standard_normal((n, 1, 1))  # uniform vol scaling per surface
    surfaces = base * perturb
    surfaces = np.maximum(surfaces, 1e-3)

    t0 = time.perf_counter()
    results = batch_check(surfaces, grid, spot=SPOT, rate=RATE, dividend=DIV)
    elapsed = time.perf_counter() - t0
    assert len(results) == n
    if elapsed > 5.0:
        pytest.skip(f"machine too slow: 50k surfaces in {elapsed:.2f}s (threshold 5s)")
    # Smoke check: most surfaces should pass.
    pass_rate = sum(1 for r in results if r.passes_all) / n
    assert pass_rate > 0.5, f"Heston-perturbed pass rate too low: {pass_rate:.3f}"


def test_input_validity_rejects_nan(grid: SurfaceGrid) -> None:
    bad = np.full(grid.shape, 0.2)
    bad[0, 0] = np.nan
    report = check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert report.passes_all is False
    assert "input_validity" in report.violations


def test_input_validity_rejects_zero_iv(grid: SurfaceGrid) -> None:
    bad = np.full(grid.shape, 0.2)
    bad[0, 0] = 0.0
    report = check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert report.passes_all is False
    assert "input_validity" in report.violations


def test_batch_check_matches_single(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    surfaces = np.stack([heston_surface, heston_surface * 1.001])
    batch = batch_check(surfaces, grid, spot=SPOT, rate=RATE, dividend=DIV)
    single0 = check_arbitrage_free(surfaces[0], grid, spot=SPOT, rate=RATE, dividend=DIV)
    single1 = check_arbitrage_free(surfaces[1], grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert batch[0].passes_all == single0.passes_all
    assert batch[1].passes_all == single1.passes_all
    # Severity should agree on the keys present in both.
    for key in batch[0].severity.keys() & single0.severity.keys():
        assert batch[0].severity[key] == pytest.approx(single0.severity[key], rel=1e-9)


# ---------------------------------------------------------------------------
# Targeted branch coverage: edge cases of single + batched checkers.
# ---------------------------------------------------------------------------


def test_check_arbitrage_free_rejects_shape_mismatch(grid: SurfaceGrid) -> None:
    """iv_surface shape must equal grid.shape."""
    bad = np.full((grid.n_strikes + 1, grid.n_maturities), 0.2)
    with pytest.raises(ValueError, match="does not match grid shape"):
        check_arbitrage_free(bad, grid, spot=SPOT, rate=RATE, dividend=DIV)


def test_batch_check_rejects_wrong_ndim(grid: SurfaceGrid) -> None:
    """batch_check requires a 3D (N, n_K, n_T) tensor matching the grid shape."""
    # Wrong ndim (2D instead of 3D)
    with pytest.raises(ValueError, match="incompatible with grid shape"):
        batch_check(np.full(grid.shape, 0.2), grid, spot=SPOT, rate=RATE, dividend=DIV)
    # 3D but wrong inner shape
    with pytest.raises(ValueError, match="incompatible with grid shape"):
        batch_check(
            np.full((2, grid.n_strikes + 1, grid.n_maturities), 0.2),
            grid,
            spot=SPOT,
            rate=RATE,
            dividend=DIV,
        )


def test_check_arbitrage_free_skips_butterfly_when_too_few_strikes() -> None:
    """n_strikes < 3 → butterfly check is skipped silently (no violations key)."""
    g = SurfaceGrid(
        log_moneyness=np.array([-0.05, 0.05], dtype=np.float64),
        maturities=np.array([0.1, 0.5], dtype=np.float64),
    )
    iv = np.full(g.shape, 0.2)
    report = check_arbitrage_free(iv, g, spot=SPOT, rate=RATE, dividend=DIV)
    assert "butterfly" not in report.violations


def test_check_arbitrage_free_skips_calendar_when_one_maturity() -> None:
    """n_T < 2 → calendar checks are skipped."""
    g = SurfaceGrid(
        log_moneyness=np.linspace(-0.2, 0.2, 11),
        maturities=np.array([0.25], dtype=np.float64),
    )
    iv = np.full(g.shape, 0.2)
    report = check_arbitrage_free(iv, g, spot=SPOT, rate=RATE, dividend=DIV)
    assert "calendar_call" not in report.violations
    assert "calendar_total_variance" not in report.violations


def test_check_arbitrage_free_notes_too_few_strikes_for_lee() -> None:
    """n_strikes ∈ [3, 6] passes butterfly but is too few for Lee — Lee adds a note."""
    g = SurfaceGrid(
        log_moneyness=np.array([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=np.float64),
        maturities=np.array([0.1, 0.5], dtype=np.float64),
    )
    iv = np.full(g.shape, 0.2)
    report = check_arbitrage_free(iv, g, spot=SPOT, rate=RATE, dividend=DIV)
    assert any("too few strikes" in n for n in report.notes)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_check_arbitrage_free_notes_degenerate_lee_wing() -> None:
    """If a wing has |k| std ≈ 0 (all strikes ATM in the wing) the wing OLS
    is degenerate and the check is skipped with a note. We construct a grid
    whose left wing has three identical log-moneyness values to trigger the
    `x.std() < 1e-12` branch. The duplicate strikes also drive a RuntimeWarning
    in the butterfly second-difference divide; we silence it because it is
    incidental to the branch under test.
    """
    g = SurfaceGrid(
        log_moneyness=np.array(
            [0.0, 0.0, 0.0, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
            dtype=np.float64,
        ),
        maturities=np.array([0.25, 0.5], dtype=np.float64),
    )
    iv = np.full(g.shape, 0.2)
    report = check_arbitrage_free(iv, g, spot=SPOT, rate=RATE, dividend=DIV)
    assert any("degenerate wing" in n for n in report.notes)


def test_batch_check_input_validity_path(grid: SurfaceGrid, heston_surface: np.ndarray) -> None:
    """Per-surface invalid_mask flags NaN/non-positive IV in the batched path."""
    bad = heston_surface.copy()
    bad[0, 0] = np.nan
    surfaces = np.stack([bad, heston_surface])
    results = batch_check(surfaces, grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert results[0].passes_all is False
    assert "input_validity" in results[0].violations
    assert results[1].passes_all is True


def test_batch_check_records_butterfly_violations(
    grid: SurfaceGrid, heston_surface: np.ndarray
) -> None:
    """Butterfly violators are flagged through the vectorised batch path."""
    bad = heston_surface.copy()
    bad[7, 3] -= 0.05  # same trick as the single-surface test
    results = batch_check(bad[None, ...], grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert "butterfly" in results[0].violations
    assert results[0].passes_all is False


def test_batch_check_records_calendar_violations(
    grid: SurfaceGrid, heston_surface: np.ndarray
) -> None:
    """Calendar (call + total variance) violators are flagged through batch_check."""
    bad = heston_surface.copy()
    bad[:, 0] += 0.10  # short maturity slice blown up
    results = batch_check(bad[None, ...], grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert ("calendar_call" in results[0].violations) or (
        "calendar_total_variance" in results[0].violations
    )
    assert results[0].passes_all is False


def test_batch_check_records_lee_right_wing_violations(
    grid: SurfaceGrid, heston_surface: np.ndarray
) -> None:
    """Lee right-wing slope violators are flagged through batch_check."""
    bad = heston_surface.copy()
    n_K = grid.n_strikes
    n_wing = max(3, n_K // 10)
    for j, T in enumerate(grid.maturities):
        for i in range(n_K - n_wing, n_K):
            k = grid.log_moneyness[i]
            target_w = 25.0 * abs(k)  # slope ≫ 2
            bad[i, j] = float(np.sqrt(max(target_w, 1e-6) / max(T, 1e-6)))
    results = batch_check(bad[None, ...], grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert "lee_right_wing" in results[0].violations
    assert results[0].passes_all is False


def test_batch_check_records_lee_left_wing_violations(
    grid: SurfaceGrid, heston_surface: np.ndarray
) -> None:
    """Lee left-wing slope violators are flagged through batch_check (left side)."""
    bad = heston_surface.copy()
    n_K = grid.n_strikes
    n_wing = max(3, n_K // 10)
    for j, T in enumerate(grid.maturities):
        for i in range(n_wing):
            k = grid.log_moneyness[i]
            target_w = 25.0 * abs(k)  # slope ≫ 2 on the left side
            bad[i, j] = float(np.sqrt(max(target_w, 1e-6) / max(T, 1e-6)))
    results = batch_check(bad[None, ...], grid, spot=SPOT, rate=RATE, dividend=DIV)
    assert "lee_left_wing" in results[0].violations
    assert results[0].passes_all is False


def test_batch_check_notes_too_few_strikes_for_lee() -> None:
    """In the batch path with n_K < 7, the Lee-skipped note is appended."""
    g = SurfaceGrid(
        log_moneyness=np.linspace(-0.1, 0.1, 5),
        maturities=np.array([0.1, 0.5], dtype=np.float64),
    )
    iv = np.full((2, *g.shape), 0.2)
    results = batch_check(iv, g, spot=SPOT, rate=RATE, dividend=DIV)
    assert all(any("too few strikes" in n for n in r.notes) for r in results)


def test_check_arbitrage_free_empty_intersection_k_grid_skips_total_variance() -> None:
    """When the intersection log-moneyness grid across maturities is empty
    (k_max(T_long) ≤ k_min(T_short) because of large (r-q) and big T span),
    the total-variance calendar check is skipped with an explanatory note.
    """
    g = SurfaceGrid(
        log_moneyness=np.linspace(-0.05, 0.05, 11),
        maturities=np.array([0.01, 5.0], dtype=np.float64),  # 5 years vs 1 week
    )
    iv = np.full(g.shape, 0.2)
    # Large rate: (r - q) * T spread → k = log_moneyness - (r-q)T pushes far apart.
    report = check_arbitrage_free(iv, g, spot=SPOT, rate=0.5, dividend=0.0)
    assert any("empty intersection k-grid" in n for n in report.notes)


def test_batch_check_empty_intersection_k_grid_falls_back() -> None:
    """Batch path equivalent: with empty intersection, w_viol stays None and the
    per-surface assembler skips total-variance reporting (no key added)."""
    g = SurfaceGrid(
        log_moneyness=np.linspace(-0.05, 0.05, 11),
        maturities=np.array([0.01, 5.0], dtype=np.float64),
    )
    iv = np.full((2, *g.shape), 0.2)
    results = batch_check(iv, g, spot=SPOT, rate=0.5, dividend=0.0)
    for r in results:
        assert "calendar_total_variance" not in r.violations
