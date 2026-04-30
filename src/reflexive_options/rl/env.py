"""Gymnasium environment wrapping any `SimulatorProtocol`.

Same env class trains agents inside the reflexive simulator and inside every
non-reflexive baseline — this is the architectural commitment that makes the
H1 / H2 comparisons in `paper/pre_registration.md` possible without per-model
plumbing duplication.

Step semantics:

    1. Apply action → new flat (n_K * n_T,) target position. Clip to bounds.
    2. trade = new - current. Charge transaction cost on |trade @ price|.
    3. Sample correlated dW from internal RNG. Call `sim.step(state, dt, dW)`.
    4. Re-price the position at the *new* state's IV surface; P&L = new_value -
       old_value - cash_paid_for_trade.
    5. Generate the new IV surface via `sim.implied_surface(...)`.
    6. Build new observation, compute reward, return (obs, reward, terminated,
       truncated, info).

Marking-to-market uses the simulator's per-step IV surface as the BS volatility
input — i.e., the agent is hedging against options priced inside the same
volatility model that drives the underlying. This is the standard convention
for model-vs-model RL training; it does NOT introduce a survivor or look-ahead
bias since the surface depends only on the *current* state.

For v1 the agent's portfolio is interpreted as European calls; the put leg is
deferred (dealer sign convention is handled in `simulator.gamma_aggregator`,
not here).
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from reflexive_options.baselines._mc_iv import bs_call_price
from reflexive_options.rl.actions import (
    ActionConfig,
    apply_action,
    make_action_space,
)
from reflexive_options.rl.rewards import RewardConfig, compute_reward, rolling_sharpe
from reflexive_options.rl.state import StateConfig, build_observation
from reflexive_options.types import (
    SDEState,
    SimulatorProtocol,
    SurfaceArray,
    SurfaceGrid,
)


def price_option_position(
    spot: float,
    surface: SurfaceArray,
    position: NDArray[np.float64],
    grid: SurfaceGrid,
    rate: float,
    dividend: float,
    state_time: float,
) -> float:
    """Mark-to-market a position grid as European calls under Black-Scholes.

    Args:
        spot: current S_t.
        surface: IV surface at the current state, shape (n_K, n_T).
        position: contracts held per (strike, maturity), same shape (or flat).
            One contract = one underlying-share-equivalent (multiplier left to
            the caller — for v1 we treat each entry as 1 share-equivalent so
            transaction-cost / position-size are in dollars-of-underlying units).
        grid: strike-maturity grid.
        rate: risk-free rate.
        dividend: continuous dividend yield (subtracted from `rate` internally).
        state_time: current simulator time, used to derive time-to-expiry per
            maturity column.

    Returns:
        Total dollar value of the position (long + short, signed).
    """
    pos = np.asarray(position, dtype=np.float64).reshape(grid.shape)
    if surface.shape != grid.shape:
        raise ValueError(f"surface shape {surface.shape} != grid shape {grid.shape}")

    total = 0.0
    discount_rate = rate - dividend
    for j, T in enumerate(grid.maturities):
        ttm = max(float(T) - float(state_time), 0.0)
        for i, k in enumerate(grid.log_moneyness):
            qty = float(pos[i, j])
            if qty == 0.0:
                continue
            strike = spot * math.exp(float(k))
            sigma = float(surface[i, j])
            if not np.isfinite(sigma) or sigma <= 0.0 or ttm <= 0.0:
                # Expired or undefined → intrinsic
                value = max(spot - strike * math.exp(-discount_rate * ttm), 0.0)
            else:
                value = bs_call_price(spot, strike, ttm, sigma, discount_rate)
            total += qty * value
    return total


# Action space is the continuous Box (or MultiDiscrete in ablation mode) — both
# return ndarray samples; using NDArray[np.float64] as the canonical ActType
# matches `apply_action()`'s output and the dominant continuous-mode path.
# See docs/quality_research_brief.md §1 for the gymnasium 1.x typing convention.
ObsArray = npt.NDArray[np.float64]
ActArray = npt.NDArray[np.float64]


class OptionsHedgeEnv(gym.Env[ObsArray, ActArray]):
    """Single-asset options-hedging environment over a (strike × maturity) grid."""

    # gymnasium.Env declares `metadata` as an instance variable; we override it
    # at the class level following the framework convention. Suppress RUF012:
    # the dict is intentionally shared (read-only sentinel — agents must not mutate).
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        sim: SimulatorProtocol,
        state_cfg: StateConfig,
        action_cfg: ActionConfig,
        reward_cfg: RewardConfig,
        *,
        episode_length: int = 252,
        dt: float = 1.0 / 252,
        initial_spot: float = 100.0,
        initial_variance: float = 0.04,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        rho: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if state_cfg.surface_grid.shape != action_cfg.grid.shape:
            raise ValueError("state_cfg.surface_grid and action_cfg.grid must have the same shape")
        if episode_length <= 0:
            raise ValueError(f"episode_length must be > 0, got {episode_length}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {rho}")

        self.sim = sim
        self.state_cfg = state_cfg
        self.action_cfg = action_cfg
        self.reward_cfg = reward_cfg
        self.episode_length = episode_length
        self.dt = dt
        self.initial_spot = float(initial_spot)
        self.initial_variance = float(initial_variance)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)
        self.rho = float(rho)
        self._seed_arg = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.action_space: gym.spaces.Space[npt.NDArray[Any]] = make_action_space(action_cfg)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_cfg.observation_dim,),
            dtype=np.float64,
        )

        # Episode state — populated in reset()
        self._step_count: int = 0
        self._sde_state: SDEState | None = None
        self._surface: SurfaceArray | None = None
        self._position: NDArray[np.float64] = np.zeros(action_cfg.action_dim, dtype=np.float64)
        self._history_buf: NDArray[np.float64] | None = None
        self._pnl_history: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_correlated_dW(self) -> NDArray[np.float64]:
        z1, z2 = self._rng.standard_normal(2)
        sqrt_dt = math.sqrt(self.dt)
        dW_S = float(z1) * sqrt_dt
        dW_v = (self.rho * float(z1) + math.sqrt(1.0 - self.rho * self.rho) * float(z2)) * sqrt_dt
        return np.array([dW_S, dW_v], dtype=np.float64)

    def _push_history(self, surface: SurfaceArray) -> None:
        if self.state_cfg.history_window <= 0:
            return
        assert self._history_buf is not None
        # Roll: drop oldest, append newest at the tail (chronological order).
        self._history_buf = np.concatenate(
            [self._history_buf[1:], surface[None, ...]],
            axis=0,
        )

    def _gross_position_dollars(self, spot: float) -> float:
        """|long-notional| + |short-notional| of the position at current spot.

        For an options book a precise gross notional needs greeks; for the size
        penalty we use |position| × spot, which is the share-equivalent
        notional and the correct unit for the bp transaction-cost model.
        """
        return float(np.sum(np.abs(self._position)) * spot)

    def _trade_notional(
        self,
        trade: NDArray[np.float64],
        spot: float,
    ) -> float:
        """Signed dollar notional traded this step (sum of contract-prices × shares)."""
        return float(np.sum(trade) * spot)

    def _build_obs(self) -> NDArray[np.float64]:
        assert self._sde_state is not None and self._surface is not None
        return build_observation(
            state=self._sde_state,
            surface=self._surface,
            position=self._position,
            history=self._history_buf,
            cfg=self.state_cfg,
            initial_spot=self.initial_spot,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        del options  # unused for v1

        self._step_count = 0
        self._sde_state = SDEState(
            spot=self.initial_spot,
            variance=self.initial_variance,
            time=0.0,
            aggregate_gamma=None,
            memory=0.0,
        )
        # Probe the simulator once at t=0 so any G/z that the simulator carries
        # are reflected from the start.
        zero_dW = np.array([0.0, 0.0], dtype=np.float64)
        probed = self.sim.step(self._sde_state, 0.0, zero_dW)
        # `step(..., dt=0.0)` should be a no-op for time + dynamics but may
        # populate G and z. Fall back to the original state if the simulator
        # does not honour dt=0 cleanly.
        if probed.time == 0.0:
            self._sde_state = probed

        self._surface = self.sim.implied_surface(self._sde_state, self.state_cfg.surface_grid)
        self._position = np.zeros(self.action_cfg.action_dim, dtype=np.float64)
        if self.state_cfg.history_window > 0:
            self._history_buf = np.zeros(
                (
                    self.state_cfg.history_window,
                    self.state_cfg.surface_grid.n_strikes,
                    self.state_cfg.surface_grid.n_maturities,
                ),
                dtype=np.float64,
            )
            # Seed the buffer with the initial surface so the agent doesn't
            # observe a zero-history shock on its first step.
            self._history_buf[-1] = self._surface
        else:
            self._history_buf = None
        self._pnl_history = []

        obs = self._build_obs()
        info: dict[str, Any] = {
            "step": self._step_count,
            "spot": float(self._sde_state.spot),
            "variance": float(self._sde_state.variance),
        }
        return obs, info

    def step(
        self,
        action: ActArray,
    ) -> tuple[ObsArray, float, bool, bool, dict[str, Any]]:
        if self._sde_state is None or self._surface is None:
            raise RuntimeError("step() called before reset()")

        # 1) Resolve action → new target position (clipped).
        new_position = apply_action(self._position, action, self.action_cfg)
        trade = new_position - self._position

        # 2) Mark-to-market BEFORE advancing dynamics so we know what we paid
        #    for the trade at current prices.
        old_state = self._sde_state
        old_surface = self._surface
        spot_before = float(old_state.spot)

        # Value of pre-trade position at current state (held leg).
        held_value_before = price_option_position(
            spot=spot_before,
            surface=old_surface,
            position=self._position,
            grid=self.state_cfg.surface_grid,
            rate=self.risk_free_rate,
            dividend=self.dividend_yield,
            state_time=old_state.time,
        )
        # Value of new position at current state (after the trade).
        new_value_before = price_option_position(
            spot=spot_before,
            surface=old_surface,
            position=new_position,
            grid=self.state_cfg.surface_grid,
            rate=self.risk_free_rate,
            dividend=self.dividend_yield,
            state_time=old_state.time,
        )
        cash_paid_for_trade = new_value_before - held_value_before

        # 3) Advance dynamics with sampled dW.
        dW_vec = self._sample_correlated_dW()
        new_state = self.sim.step(old_state, self.dt, dW_vec)

        # 4) Re-price NEW position at the new state.
        new_surface = self.sim.implied_surface(new_state, self.state_cfg.surface_grid)
        new_value_after = price_option_position(
            spot=float(new_state.spot),
            surface=new_surface,
            position=new_position,
            grid=self.state_cfg.surface_grid,
            rate=self.risk_free_rate,
            dividend=self.dividend_yield,
            state_time=new_state.time,
        )

        # P&L = mark-to-market change in position value − cash paid for the trade.
        # cash_paid_for_trade is positive when buying additional value, and the
        # position-value change includes that additional value, so we subtract it.
        pnl = float(new_value_after - new_value_before)
        # Equivalent formulation: (new_value_after − held_value_before) − cash_paid_for_trade,
        # but new_value_after − new_value_before is the pure dynamics PnL.
        del cash_paid_for_trade  # not used directly; kept above for clarity / future hooks

        # 5) Commit env state and compute reward.
        self._position = new_position
        self._sde_state = new_state
        self._surface = new_surface
        self._push_history(new_surface)

        trade_dollars = self._trade_notional(trade, spot_before)
        gross_dollars = self._gross_position_dollars(float(new_state.spot))
        reward = compute_reward(
            pnl=pnl,
            trade_dollars=trade_dollars,
            gross_position_dollars=gross_dollars,
            pnl_history=self._pnl_history,
            cfg=self.reward_cfg,
        )
        self._pnl_history.append(pnl)

        # 6) Episode termination.
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.episode_length

        obs = self._build_obs()
        info: dict[str, Any] = {
            "step": self._step_count,
            "spot": float(new_state.spot),
            "variance": float(new_state.variance),
            "aggregate_gamma": (
                float(new_state.aggregate_gamma) if new_state.aggregate_gamma is not None else 0.0
            ),
            "memory": float(new_state.memory) if new_state.memory is not None else 0.0,
            "pnl": pnl,
            "trade_dollars": trade_dollars,
            "gross_position_dollars": gross_dollars,
            "rolling_sharpe": rolling_sharpe(self._pnl_history, self.reward_cfg.sharpe_window),
        }
        return obs, float(reward), terminated, truncated, info


__all__ = ["OptionsHedgeEnv", "price_option_position"]
