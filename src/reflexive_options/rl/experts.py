"""Heuristic experts that produce (obs, action) trajectories for behavioral cloning.

The reflexive-transfer experiment (`experiments/reflexive_transfer.py`) needs an
"anchor" agent π_{κ₀} trained at the calibrated coupling κ₀. The full-fat path is
PPO/EWC over the vendored ATLAS Mamba backbone — ~200 GPU-h per the pre-reg
compute envelope. For v1 we substitute behavioral cloning from a heuristic
short-vol delta-hedger; the BC student inherits the heuristic's structure but is
imperfect, which is exactly what we need to produce a non-trivial κ-sensitivity
slope (a perfect heuristic would have an analytic — and uninteresting — curve).

Expert design — `make_delta_hedged_short_vol_expert`:

    1. **At reset** — short the front-month ATM call by `short_atm_contracts`
       (pure short-vol exposure, vega-positive going against the market).
    2. **Delta-hedge offset** — go long deep-ITM front-month calls (high delta,
       low vega) sized to neutralize the net portfolio delta. This is the
       options-only analogue of "buy ΔS shares" — the env's action space holds
       only call positions on the (strike × maturity) grid, so we use the
       deepest-ITM call available as a synthetic underlying proxy.
    3. **At each step** — recompute target-delta-neutral position using the
       *current* surface's BS deltas; the action returned is the new flat
       target position.
    4. **Near expiry** — when the front-month TTM falls below `close_threshold`
       (in years), close everything (action = zeros). Avoids gamma blow-ups.

The action returned has the same shape as `env.action_space`; downstream BC
just trains a small MLP to mimic it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from reflexive_options.rl.env import OptionsHedgeEnv
from reflexive_options.types import SurfaceArray, SurfaceGrid


def bs_delta_call(spot: float, strike: float, ttm: float, sigma: float, rate: float) -> float:
    """Black-Scholes delta of a European call. Returns 0 for degenerate inputs."""
    if ttm <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return 1.0 if spot > strike else 0.0
    sqrt_t = math.sqrt(ttm)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * ttm) / (sigma * sqrt_t)
    return float(norm.cdf(d1))


def bs_vega_call(spot: float, strike: float, ttm: float, sigma: float, rate: float) -> float:
    """Black-Scholes vega per 1.00 (not per-1%) of European call. Returns 0 for degenerate inputs."""
    if ttm <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(ttm)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * ttm) / (sigma * sqrt_t)
    return float(spot * sqrt_t * norm.pdf(d1))


def _surface_delta_grid(
    spot: float,
    surface: SurfaceArray,
    grid: SurfaceGrid,
    rate: float,
    state_time: float,
) -> NDArray[np.float64]:
    """Return BS-delta grid of shape grid.shape, per-(strike, maturity) call."""
    out = np.zeros(grid.shape, dtype=np.float64)
    for j, T in enumerate(grid.maturities):
        ttm = max(float(T) - float(state_time), 0.0)
        for i, k in enumerate(grid.log_moneyness):
            strike = spot * math.exp(float(k))
            out[i, j] = bs_delta_call(spot, strike, ttm, float(surface[i, j]), rate)
    return out


@dataclass(frozen=True)
class DeltaHedgedShortVolExpert:
    """Stateless heuristic that maps (env, obs) → next-step action.

    The expert is *stateless in policy*: every action is recomputed from the
    current env state via the env's current spot, surface, and time. Storing
    the env reference is a convenience — the policy uses only env config
    (grid, rate, max position) and the live state, never any prior-step
    history. This is what we want for BC: the student MLP only needs the
    observation to reproduce the action.

    Attributes:
        env: the environment to read live state from. Held by reference; the
            expert never mutates it.
        short_atm_contracts: number of front-month ATM calls to short (positive
            number — the position itself is signed negative in the action).
        front_month_idx: which maturity index counts as "front month" for the
            short leg. Defaults to 0 (the shortest-dated column).
        max_position_per_strike: clip on every cell of the action so we never
            exceed the env's `ActionConfig.max_position_per_strike`.
        close_ttm_threshold: front-month TTM (in years) below which the
            expert flattens to zero. Default 1 trading day.
    """

    env: OptionsHedgeEnv
    short_atm_contracts: float = 1.0
    front_month_idx: int = 0
    max_position_per_strike: float = 100.0
    close_ttm_threshold: float = 1.0 / 252.0

    def __post_init__(self) -> None:
        grid = self.env.state_cfg.surface_grid
        if not 0 <= self.front_month_idx < grid.n_maturities:
            raise ValueError(
                f"front_month_idx {self.front_month_idx} out of range [0, {grid.n_maturities})"
            )
        if self.short_atm_contracts <= 0:
            raise ValueError(
                f"short_atm_contracts must be > 0, got {self.short_atm_contracts}"
            )
        if self.max_position_per_strike <= 0:
            raise ValueError(
                f"max_position_per_strike must be > 0, got {self.max_position_per_strike}"
            )

    def act(self) -> NDArray[np.float64]:
        """Return the target position vector (flat, length `action_dim`).

        Reads `env._sde_state` and `env._surface` directly. Caller must have
        invoked `env.reset()` at least once.
        """
        sde_state = self.env._sde_state
        surface = self.env._surface
        if sde_state is None or surface is None:
            raise RuntimeError("expert.act() called before env.reset()")

        grid = self.env.state_cfg.surface_grid
        spot = float(sde_state.spot)
        state_time = float(sde_state.time)
        rate = self.env.risk_free_rate

        front_T = float(grid.maturities[self.front_month_idx])
        front_ttm = max(front_T - state_time, 0.0)

        target = np.zeros(grid.shape, dtype=np.float64)
        if front_ttm <= self.close_ttm_threshold:
            # Avoid expiration-day gamma blow-up: close everything.
            return target.reshape(-1)

        # Pick ATM strike (closest to log_moneyness = 0).
        atm_idx = int(np.argmin(np.abs(grid.log_moneyness)))
        # Pick the deepest-ITM strike available (most negative log-moneyness).
        # Deep-ITM call ≈ delta 1, so it acts as a synthetic long underlying.
        hedge_strike_idx = int(np.argmin(grid.log_moneyness))

        if hedge_strike_idx == atm_idx:
            # Single-strike grid — cannot delta-hedge with calls. Just hold short ATM.
            target[atm_idx, self.front_month_idx] = -self.short_atm_contracts
            return self._clip(target)

        deltas = _surface_delta_grid(spot, surface, grid, rate, state_time)
        short_delta = float(deltas[atm_idx, self.front_month_idx])
        hedge_delta = float(deltas[hedge_strike_idx, self.front_month_idx])

        # Net delta of the short leg: -short_atm_contracts × short_delta.
        # Hedge contracts h chosen so h × hedge_delta + (-short_atm_contracts × short_delta) = 0.
        if hedge_delta <= 1e-6:
            hedge_contracts = 0.0
        else:
            hedge_contracts = self.short_atm_contracts * short_delta / hedge_delta

        target[atm_idx, self.front_month_idx] = -self.short_atm_contracts
        target[hedge_strike_idx, self.front_month_idx] = hedge_contracts
        return self._clip(target)

    def _clip(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(
            position.reshape(-1),
            -self.max_position_per_strike,
            +self.max_position_per_strike,
        )


def make_delta_hedged_short_vol_expert(
    env: OptionsHedgeEnv,
    *,
    short_atm_contracts: float = 1.0,
    front_month_idx: int = 0,
) -> Callable[[], NDArray[np.float64]]:
    """Build a callable `expert()` that returns the next action for the given env.

    Convenience wrapper: most callers want a zero-arg policy closure to pass into
    a generic rollout loop, not the dataclass instance.
    """
    expert = DeltaHedgedShortVolExpert(
        env=env,
        short_atm_contracts=short_atm_contracts,
        front_month_idx=front_month_idx,
        max_position_per_strike=env.action_cfg.max_position_per_strike,
    )
    return expert.act


__all__ = [
    "DeltaHedgedShortVolExpert",
    "bs_delta_call",
    "bs_vega_call",
    "make_delta_hedged_short_vol_expert",
]
