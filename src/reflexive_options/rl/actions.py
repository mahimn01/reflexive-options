"""Action space + application logic for the OptionsHedgeEnv.

V1: continuous Box action of shape `(n_K * n_T,)`. Each component is interpreted
as the **target** position to move to (in number of contracts, signed), clipped
to `[-max_position_per_strike, +max_position_per_strike]`. The env diffs against
the current position to compute the trade and charge transaction cost.

A discrete-bucketed mode is exposed for ablations / smaller policies — each
contract dimension picks one of `n_buckets` evenly-spaced target positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from reflexive_options.types import SurfaceGrid


@dataclass(frozen=True)
class ActionConfig:
    """Configuration for the agent's action space.

    Attributes:
        grid: the (strike × maturity) grid the action vector is laid out on.
            Action shape = `(grid.n_strikes * grid.n_maturities,)`.
        max_position_per_strike: clip bound (absolute value) on the target position
            per (strike, maturity) cell.
        discrete: if True, use `MultiDiscrete` over `n_buckets` evenly-spaced
            target positions; if False, use a continuous Box.
        n_buckets: number of buckets in the discrete mode. Must be odd so that
            zero is exactly one of the bucket centers.
    """

    grid: SurfaceGrid
    max_position_per_strike: float = 100.0
    discrete: bool = False
    n_buckets: int = 7

    def __post_init__(self) -> None:
        if self.max_position_per_strike <= 0:
            raise ValueError(
                f"max_position_per_strike must be > 0, got {self.max_position_per_strike}"
            )
        if self.discrete and self.n_buckets < 3:
            raise ValueError(f"n_buckets must be >= 3 in discrete mode, got {self.n_buckets}")
        if self.discrete and self.n_buckets % 2 == 0:
            raise ValueError(
                f"n_buckets must be odd so zero is a bucket center, got {self.n_buckets}"
            )

    @property
    def action_dim(self) -> int:
        return self.grid.n_strikes * self.grid.n_maturities


def make_action_space(cfg: ActionConfig) -> gym.Space:
    """Build the gymnasium action space matching `cfg`."""
    if cfg.discrete:
        return gym.spaces.MultiDiscrete(
            np.full(cfg.action_dim, cfg.n_buckets, dtype=np.int64)
        )
    return gym.spaces.Box(
        low=-cfg.max_position_per_strike,
        high=cfg.max_position_per_strike,
        shape=(cfg.action_dim,),
        dtype=np.float64,
    )


def _bucket_centers(cfg: ActionConfig) -> NDArray[np.float64]:
    """Evenly-spaced bucket centers in [-max, +max], symmetric, includes 0."""
    return np.linspace(
        -cfg.max_position_per_strike,
        cfg.max_position_per_strike,
        cfg.n_buckets,
        dtype=np.float64,
    )


def apply_action(
    current_position: NDArray[np.float64],
    action: NDArray,  # type: ignore[type-arg]
    cfg: ActionConfig,
) -> NDArray[np.float64]:
    """Apply `action` and return the new clipped position vector (flat, length `action_dim`).

    For continuous mode the action *is* the target position (after clipping).
    For discrete mode the action is a vector of bucket indices that selects
    targets from `_bucket_centers(cfg)`.

    The previous position is intentionally unused in the mapping itself —
    transaction cost is computed by the env via `new - current`.
    """
    cur = np.asarray(current_position, dtype=np.float64).reshape(-1)
    if cur.size != cfg.action_dim:
        raise ValueError(
            f"current_position has {cur.size} elements, expected {cfg.action_dim}"
        )

    act = np.asarray(action)
    if act.size != cfg.action_dim:
        raise ValueError(f"action has {act.size} elements, expected {cfg.action_dim}")

    if cfg.discrete:
        idx = act.astype(np.int64).reshape(-1)
        if np.any(idx < 0) or np.any(idx >= cfg.n_buckets):
            raise ValueError(
                f"discrete action out of range [0, {cfg.n_buckets}); got {idx.tolist()}"
            )
        target = _bucket_centers(cfg)[idx]
    else:
        target = act.astype(np.float64).reshape(-1)

    return np.clip(
        target,
        -cfg.max_position_per_strike,
        +cfg.max_position_per_strike,
    )


__all__ = ["ActionConfig", "apply_action", "make_action_space"]
