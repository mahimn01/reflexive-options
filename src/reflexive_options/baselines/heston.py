"""Time-dependent Heston (piecewise-constant regimes) — primary baseline.

Reduces to standard Heston for `len(regimes) == 1`. With multiple regimes the
analytic IV computation in `implied_surface` uses the active regime's parameters
only — this is a deliberate first-order approximation. A fully time-dependent
analytic engine would integrate the characteristic function piecewise; for v1
the ATM-IV error is bounded by the volatility-of-vol times the regime length and
is acceptable for our purposes.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import QuantLib as ql  # noqa: N813 — QuantLib's canonical alias is `ql`
from numpy.typing import NDArray

from reflexive_options.simulator.integrators import (
    correlated_brownians,
    euler_maruyama_step,
)
from reflexive_options.types import (
    HestonParams,
    PathArray,
    SDEState,
    SurfaceArray,
    SurfaceGrid,
)


@dataclass(frozen=True)
class _HestonRegimeIndex:
    """Cached lookup helper: maps t to the active regime index."""

    breakpoints: tuple[float, ...]

    def index_at(self, t: float) -> int:
        return bisect_right(self.breakpoints, t)


class HestonSimulator:
    """Time-dependent Heston with piecewise-constant regimes."""

    def __init__(
        self,
        regimes: list[HestonParams],
        breakpoints: list[float],
        spot0: float = 100.0,
        drift: float = 0.0,
    ) -> None:
        if len(regimes) != len(breakpoints) + 1:
            raise ValueError(
                f"len(regimes)={len(regimes)} must equal len(breakpoints)+1={len(breakpoints) + 1}"
            )
        if any(b <= 0 for b in breakpoints):
            raise ValueError("breakpoints must be positive years")
        if any(breakpoints[i] >= breakpoints[i + 1] for i in range(len(breakpoints) - 1)):
            raise ValueError("breakpoints must be strictly ascending")
        if spot0 <= 0:
            raise ValueError(f"spot0 must be > 0, got {spot0}")

        self.regimes: tuple[HestonParams, ...] = tuple(regimes)
        self.breakpoints: tuple[float, ...] = tuple(breakpoints)
        self.spot0 = float(spot0)
        self.drift = float(drift)
        self._index = _HestonRegimeIndex(self.breakpoints)

    def regime_at(self, t: float) -> HestonParams:
        return self.regimes[self._index.index_at(t)]

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> tuple[PathArray, PathArray]:
        rng = np.random.default_rng(seed)
        spots = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        spots[:, 0] = self.spot0
        variances[:, 0] = self.regimes[0].v0

        # Pre-generate per-regime correlated Brownians: each regime uses its own ρ.
        # We generate step-by-step using the active regime's ρ to keep this exact.
        # For perf, batch by contiguous regime spans.
        spans = self._regime_spans(n_steps, dt)
        col_offset = 0
        for regime, n_span in spans:
            dW_S, dW_v = correlated_brownians(
                n_paths=n_paths,
                n_steps=n_span,
                rho=regime.rho,
                dt=dt,
                rng=rng,
                antithetic=False,
            )
            for k in range(n_span):
                t = (col_offset + k) * dt
                v = variances[:, col_offset + k]
                s = spots[:, col_offset + k]
                sqrt_v = np.sqrt(v)
                drift_s = self.drift * s
                drift_v = regime.kappa * (regime.theta - v)
                diff_s = sqrt_v * s
                diff_v = regime.xi * sqrt_v
                new_s, new_v = euler_maruyama_step(
                    spot=s,
                    variance=v,
                    t=t,
                    dt=dt,
                    drift_S=drift_s,
                    drift_v=drift_v,
                    diff_S=diff_s,
                    diff_v=diff_v,
                    dW_S=dW_S[:, k],
                    dW_v=dW_v[:, k],
                )
                spots[:, col_offset + k + 1] = new_s
                variances[:, col_offset + k + 1] = new_v
            col_offset += n_span
        return spots, variances

    def _regime_spans(self, n_steps: int, dt: float) -> list[tuple[HestonParams, int]]:
        """Group consecutive steps that share the same regime."""
        spans: list[tuple[HestonParams, int]] = []
        current_idx = self._index.index_at(0.0)
        run_len = 0
        for k in range(n_steps):
            t = k * dt
            idx = self._index.index_at(t)
            if idx != current_idx:
                spans.append((self.regimes[current_idx], run_len))
                current_idx = idx
                run_len = 0
            run_len += 1
        if run_len > 0:
            spans.append((self.regimes[current_idx], run_len))
        return spans

    def step(self, state: SDEState, dt: float, dW: NDArray[np.float64]) -> SDEState:
        if dW.shape != (2,):
            raise ValueError(f"dW must have shape (2,), got {dW.shape}")
        regime = self.regime_at(state.time)
        s_arr = np.array([state.spot], dtype=np.float64)
        v_arr = np.array([state.variance], dtype=np.float64)
        sqrt_v = np.sqrt(v_arr)
        new_s, new_v = euler_maruyama_step(
            spot=s_arr,
            variance=v_arr,
            t=state.time,
            dt=dt,
            drift_S=self.drift * s_arr,
            drift_v=regime.kappa * (regime.theta - v_arr),
            diff_S=sqrt_v * s_arr,
            diff_v=regime.xi * sqrt_v,
            dW_S=np.array([dW[0]]),
            dW_v=np.array([dW[1]]),
        )
        return SDEState(
            spot=float(new_s[0]),
            variance=float(new_v[0]),
            time=state.time + dt,
        )

    def implied_surface(self, state: SDEState, grid: SurfaceGrid) -> SurfaceArray:
        regime = self.regime_at(state.time)
        return _quantlib_heston_iv_surface(
            spot=state.spot,
            v0=state.variance,
            params=regime,
            grid=grid,
            drift=self.drift,
        )


def _quantlib_heston_iv_surface(
    spot: float,
    v0: float,
    params: HestonParams,
    grid: SurfaceGrid,
    drift: float,
) -> SurfaceArray:
    """Closed-form Heston call prices via QuantLib AnalyticHestonEngine, inverted to IV."""
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    risk_free_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, drift, day_count))
    dividend_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

    process = ql.HestonProcess(
        risk_free_curve,
        dividend_curve,
        spot_handle,
        v0,
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    n_strikes = grid.n_strikes
    n_maturities = grid.n_maturities
    iv = np.zeros((n_strikes, n_maturities), dtype=np.float64)

    for j, T in enumerate(grid.maturities):
        maturity_days = max(round(float(T) * 365.0), 1)
        maturity_date = today + maturity_days
        exercise = ql.EuropeanExercise(maturity_date)
        for i, k in enumerate(grid.log_moneyness):
            strike = spot * float(np.exp(k))
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(engine)
            try:
                price = float(option.NPV())
            except RuntimeError:
                iv[i, j] = float("nan")
                continue
            try:
                bs_process = ql.BlackScholesMertonProcess(
                    spot_handle,
                    dividend_curve,
                    risk_free_curve,
                    ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(today, calendar, 0.20, day_count)
                    ),
                )
                vol = option.impliedVolatility(
                    price,
                    bs_process,
                    1e-6,
                    200,
                    1e-4,
                    5.0,
                )
                iv[i, j] = float(vol)
            except RuntimeError:
                iv[i, j] = float("nan")
    return iv
