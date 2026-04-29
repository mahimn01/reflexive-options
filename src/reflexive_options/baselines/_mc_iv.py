"""Shared Monte-Carlo IV helpers — used by LSV and SV32 baselines.

The closed-form Heston engine in QuantLib lets HestonSimulator skip MC for IV.
LSV and 3/2 have no analytic engine, so we Monte-Carlo prices and invert via Brent.
This is slow (~seconds per surface at 20k paths × 252 steps/year) — gate behind
`compute_surface=False` by default in caller.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from reflexive_options.types import PathArray, SDEState, SurfaceArray, SurfaceGrid


class _SimulateOnly(Protocol):
    spot0: float

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> tuple[PathArray, PathArray]: ...


def bs_call_price(spot: float, strike: float, T: float, sigma: float, r: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(spot - strike * math.exp(-r * T), 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return spot * float(norm.cdf(d1)) - strike * math.exp(-r * T) * float(norm.cdf(d2))


def implied_vol_brent(
    price: float,
    spot: float,
    strike: float,
    T: float,
    r: float,
) -> float:
    intrinsic = max(spot - strike * math.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-12:
        return float("nan")
    if price >= spot - 1e-12:
        return float("nan")

    def objective(sigma: float) -> float:
        return bs_call_price(spot, strike, T, sigma, r) - price

    try:
        return float(brentq(objective, 1e-5, 5.0, maxiter=200, xtol=1e-7))
    except (ValueError, RuntimeError):
        return float("nan")


def mc_iv_surface(
    simulator: _SimulateOnly,
    state: SDEState,
    grid: SurfaceGrid,
    drift: float,
    n_paths: int = 20_000,
    n_steps_per_year: int = 252,
    seed: int | None = 0,
) -> SurfaceArray:
    del state  # caller responsibility: state.time should be 0 for surface from inception
    max_T = float(np.max(grid.maturities))
    n_steps = max(math.ceil(max_T * n_steps_per_year), 1)
    dt = max_T / n_steps
    spots, _ = simulator.simulate(n_paths=n_paths, n_steps=n_steps, dt=dt, seed=seed)

    iv = np.full(grid.shape, np.nan, dtype=np.float64)
    for j, T in enumerate(grid.maturities):
        step_idx = min(max(round(float(T) / dt), 1), n_steps)
        s_T = spots[:, step_idx]
        for i, k in enumerate(grid.log_moneyness):
            strike = simulator.spot0 * math.exp(float(k))
            payoff = np.maximum(s_T - strike, 0.0)
            disc = math.exp(-drift * float(T))
            price = float(disc * payoff.mean())
            iv[i, j] = implied_vol_brent(price, simulator.spot0, strike, float(T), drift)
    return iv
