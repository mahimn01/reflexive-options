"""Reward function for OptionsHedgeEnv.

Per `paper/pre_registration.md` §5: reward = P&L − transaction cost − position-size penalty.

The Sharpe-shaping term is an ablation switch (pre-reg §7 mentions Sharpe as one of the
agent-level reporting metrics). When `sharpe_shaping=True`, the env additionally adds
a rolling-Sharpe term to the reward; the magnitude is small (the term is normalized
to roughly the same scale as P&L so that the unshaped path remains comparable). When
False, the term is computed but not applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights.

    Attributes:
        transaction_cost_bps: cost in *basis points* of the dollar amount traded
            (|trade_dollars|). 1 bp = 0.0001. Charged on the absolute notional
            traded this step.
        position_size_penalty_lambda: per-step penalty proportional to the gross
            (long + |short|) dollar notional held. Encourages the agent to hold
            small books unless the P&L pays for them.
        sharpe_shaping: if True, *add* a rolling-Sharpe shaping term to the
            reward. The Sharpe term is computed regardless and exposed via the
            env's info dict; this flag controls only whether it enters the
            reward signal.
        sharpe_window: lookback window (in steps) for the rolling Sharpe.
    """

    transaction_cost_bps: float = 1.0
    position_size_penalty_lambda: float = 1e-4
    sharpe_shaping: bool = False
    sharpe_window: int = 20

    def __post_init__(self) -> None:
        if self.transaction_cost_bps < 0:
            raise ValueError(f"transaction_cost_bps must be >= 0, got {self.transaction_cost_bps}")
        if self.position_size_penalty_lambda < 0:
            raise ValueError(
                "position_size_penalty_lambda must be >= 0, "
                f"got {self.position_size_penalty_lambda}"
            )
        if self.sharpe_window <= 1:
            raise ValueError(f"sharpe_window must be > 1, got {self.sharpe_window}")


def rolling_sharpe(pnl_history: Sequence[float], window: int) -> float:
    """Annualization-agnostic rolling Sharpe — mean / std on the last `window` returns.

    Returns 0.0 if there are fewer than 2 observations or the window has zero std.
    """
    arr = np.asarray(pnl_history[-window:], dtype=np.float64)
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(arr.mean() / sd)


def compute_reward(
    pnl: float,
    trade_dollars: float,
    gross_position_dollars: float,
    pnl_history: Sequence[float],
    cfg: RewardConfig,
) -> float:
    """Compute the per-step reward.

    Args:
        pnl: marked-to-market P&L for this step in dollars (signed).
        trade_dollars: signed dollar amount transacted this step. Cost is on
            its absolute value.
        gross_position_dollars: |long| + |short| dollar notional currently held,
            *after* the action.
        pnl_history: prior per-step P&L values (this step's value is NOT
            included; the env appends after the call).
        cfg: weights.

    Returns:
        Scalar reward.
    """
    cost = (cfg.transaction_cost_bps * 1e-4) * abs(float(trade_dollars))
    size_penalty = cfg.position_size_penalty_lambda * float(gross_position_dollars)
    base = float(pnl) - cost - size_penalty

    if cfg.sharpe_shaping:
        # Shape with a Sharpe term computed on the history *not yet* including this PnL.
        # Adding `pnl` so the shaping reflects the candidate total path.
        candidate = [*list(pnl_history), float(pnl)]
        shaping = rolling_sharpe(candidate, cfg.sharpe_window)
        base += shaping

    return base


__all__ = ["RewardConfig", "compute_reward", "rolling_sharpe"]
