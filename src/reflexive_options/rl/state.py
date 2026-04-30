"""Observation builder for the OptionsHedgeEnv.

The observation vector is a deterministic concatenation; the layout is locked here
so downstream policy networks (Mamba/transformer) can rely on stable slice indices.
The `include_gamma=False` switch implements the A2 ablation in
`paper/pre_registration.md` §7 (strip G from agent state, retrain inside reflexive
sim — tests whether the gain is from the environment or from the observation).

Layout (in order, all components are flat float64):

    [ S_t,                                     # 1
      log(S_t / S_0),                          # 1
      v_t,                                     # 1
      sqrt(v_t),                               # 1
      G_t,                                     # 1   (only if include_gamma)
      z_t,                                     # 1   (only if include_memory)
      surface_flat,                            # n_K * n_T
      position_flat,                           # n_K * n_T
      time_to_expiry_per_maturity,             # n_T
      history_flat ]                           # history_window * n_K * n_T  (only if history_window > 0)

Slice indices are exposed via `StateConfig.observation_layout()` for tests and for
downstream models that need to carve channels out of the flat vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reflexive_options.types import SDEState, SurfaceArray, SurfaceGrid


@dataclass(frozen=True)
class StateConfig:
    """Configuration for the observation builder.

    Attributes:
        surface_grid: the (strike × maturity) grid the agent sees.
        position_dim: number of (strike, maturity) cells the agent can hold a position in.
            Must equal `surface_grid.n_strikes * surface_grid.n_maturities` for v1.
        include_gamma: if False, the G_t scalar is omitted (A2 ablation: strip-G).
        include_memory: if False, the z_t scalar is omitted.
        history_window: number of past surfaces to include. 0 = no history.
            When > 0 the env maintains a rolling buffer; the *flattened* surface
            history is appended at the tail of the observation in chronological
            order (oldest first, newest last).
    """

    surface_grid: SurfaceGrid
    position_dim: int
    include_gamma: bool = True
    include_memory: bool = True
    history_window: int = 20

    def __post_init__(self) -> None:
        expected = self.surface_grid.n_strikes * self.surface_grid.n_maturities
        if self.position_dim != expected:
            raise ValueError(
                f"position_dim={self.position_dim} must equal n_strikes*n_maturities={expected}"
            )
        if self.history_window < 0:
            raise ValueError(f"history_window must be >= 0, got {self.history_window}")

    @property
    def surface_flat_dim(self) -> int:
        return self.surface_grid.n_strikes * self.surface_grid.n_maturities

    @property
    def history_flat_dim(self) -> int:
        return self.history_window * self.surface_flat_dim

    @property
    def observation_dim(self) -> int:
        n = 4  # S, log(S/S0), v, sqrt(v)
        if self.include_gamma:
            n += 1
        if self.include_memory:
            n += 1
        n += self.surface_flat_dim
        n += self.position_dim
        n += self.surface_grid.n_maturities
        n += self.history_flat_dim
        return n

    def observation_layout(self) -> dict[str, slice]:
        """Return the named slice for each component of the observation vector.

        Used by tests to assert slice contents and by downstream models that
        need to route channels to distinct sub-networks.
        """
        layout: dict[str, slice] = {}
        cursor = 0

        layout["spot"] = slice(cursor, cursor + 1)
        cursor += 1
        layout["log_moneyness_to_s0"] = slice(cursor, cursor + 1)
        cursor += 1
        layout["variance"] = slice(cursor, cursor + 1)
        cursor += 1
        layout["sqrt_variance"] = slice(cursor, cursor + 1)
        cursor += 1
        if self.include_gamma:
            layout["aggregate_gamma"] = slice(cursor, cursor + 1)
            cursor += 1
        if self.include_memory:
            layout["memory"] = slice(cursor, cursor + 1)
            cursor += 1
        layout["surface"] = slice(cursor, cursor + self.surface_flat_dim)
        cursor += self.surface_flat_dim
        layout["position"] = slice(cursor, cursor + self.position_dim)
        cursor += self.position_dim
        layout["time_to_expiry"] = slice(cursor, cursor + self.surface_grid.n_maturities)
        cursor += self.surface_grid.n_maturities
        if self.history_window > 0:
            layout["history"] = slice(cursor, cursor + self.history_flat_dim)
            cursor += self.history_flat_dim

        assert cursor == self.observation_dim, (
            f"layout cursor {cursor} != observation_dim {self.observation_dim}"
        )
        return layout


def build_observation(
    state: SDEState,
    surface: SurfaceArray,
    position: NDArray[np.float64],
    history: NDArray[np.float64] | None,
    cfg: StateConfig,
    *,
    initial_spot: float,
) -> NDArray[np.float64]:
    """Assemble the flat observation vector per the layout in this module's docstring.

    Args:
        state: current SDE state (spot, variance, time, optional G and z).
        surface: model-implied IV surface, shape (n_K, n_T) per `cfg.surface_grid`.
        position: agent's current contract holdings, shape (n_K, n_T) or
            flattened (n_K * n_T,). Convention: positive = long.
        history: rolling buffer of past surfaces, shape (history_window, n_K, n_T)
            in chronological order (oldest first). Must be supplied iff
            `cfg.history_window > 0`. Newly-reset envs should pass a zero-filled
            buffer of the correct shape.
        cfg: layout configuration.
        initial_spot: S_0 reference for the log(S_t / S_0) channel. Held by the env.

    Returns:
        1D float64 array of length `cfg.observation_dim`.
    """
    if surface.shape != cfg.surface_grid.shape:
        raise ValueError(f"surface shape {surface.shape} != grid shape {cfg.surface_grid.shape}")
    pos_flat = np.asarray(position, dtype=np.float64).reshape(-1)
    if pos_flat.size != cfg.position_dim:
        raise ValueError(f"position has {pos_flat.size} elements, expected {cfg.position_dim}")
    if cfg.history_window > 0:
        if history is None:
            raise ValueError("history must be provided when cfg.history_window > 0")
        expected_hist_shape = (
            cfg.history_window,
            cfg.surface_grid.n_strikes,
            cfg.surface_grid.n_maturities,
        )
        if history.shape != expected_hist_shape:
            raise ValueError(f"history shape {history.shape} != expected {expected_hist_shape}")

    parts: list[NDArray[np.float64]] = []

    spot = float(state.spot)
    variance = max(float(state.variance), 0.0)
    parts.append(np.array([spot], dtype=np.float64))
    parts.append(np.array([np.log(max(spot, 1e-12) / max(initial_spot, 1e-12))], dtype=np.float64))
    parts.append(np.array([variance], dtype=np.float64))
    parts.append(np.array([np.sqrt(variance)], dtype=np.float64))

    if cfg.include_gamma:
        g = state.aggregate_gamma if state.aggregate_gamma is not None else 0.0
        parts.append(np.array([float(g)], dtype=np.float64))
    if cfg.include_memory:
        z = state.memory if state.memory is not None else 0.0
        parts.append(np.array([float(z)], dtype=np.float64))

    parts.append(np.asarray(surface, dtype=np.float64).reshape(-1))
    parts.append(pos_flat)

    tte = np.maximum(cfg.surface_grid.maturities.astype(np.float64) - float(state.time), 0.0)
    parts.append(tte)

    if cfg.history_window > 0:
        assert history is not None  # for mypy; checked above
        parts.append(np.asarray(history, dtype=np.float64).reshape(-1))

    obs = np.concatenate(parts, axis=0)
    if obs.shape[0] != cfg.observation_dim:
        raise AssertionError(
            f"assembled obs has length {obs.shape[0]}, expected {cfg.observation_dim}"
        )
    return obs


__all__ = ["StateConfig", "build_observation"]
