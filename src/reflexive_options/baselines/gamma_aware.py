"""Gamma-aware Heston — the critical ablation.

Pure-Heston dynamics, but the simulator computes G(S, z, v) and z_t at every step
and packs them into the returned `SDEState` so an RL agent training in this env
sees the *same observation vector* as it would in the reflexive simulator.

This isolates the contribution of the FEEDBACK CHANNEL ITSELF from the contribution
of having a richer state representation. Same state, no feedback ⇒ if the reflexive
sim outperforms this, the win is from the feedback dynamics, not the larger state.

`GammaAggregator` is imported lazily — when the simulator agent has not yet committed
`simulator/gamma_aggregator.py`, we fall back to the local Protocol stub here. The
next pass will reconcile.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
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

try:
    from reflexive_options.simulator.gamma_aggregator import (
        GammaAggregator as _RealGammaAggregator,
    )

    _GAMMA_AGGREGATOR_AVAILABLE = True
except ImportError:
    _RealGammaAggregator = None  # type: ignore[assignment,misc]
    _GAMMA_AGGREGATOR_AVAILABLE = False


@runtime_checkable
class GammaAggregatorProtocol(Protocol):
    """Minimal interface this baseline needs from the dealer-gamma aggregator.

    The real `simulator/gamma_aggregator.py` will implement this. Until then,
    any duck-typed object with `compute(spot, variance, log_memory) -> float`
    is accepted.
    """

    def compute(self, spot: float, variance: float, log_memory: float) -> float: ...


class GammaAwareSimulator:
    """Heston dynamics + state-symmetric (G, z) tracking with NO feedback."""

    def __init__(
        self,
        heston: HestonParams,
        aggregator: GammaAggregatorProtocol,
        memory_decay: float,
        memory_intake: float,
        drift: float = 0.0,
        spot0: float = 100.0,
    ) -> None:
        if memory_decay <= 0:
            raise ValueError(f"memory_decay must be > 0, got {memory_decay}")
        if spot0 <= 0:
            raise ValueError(f"spot0 must be > 0, got {spot0}")

        self.heston = heston
        self.aggregator = aggregator
        self.memory_decay = float(memory_decay)
        self.memory_intake = float(memory_intake)
        self.drift = float(drift)
        self.spot0 = float(spot0)

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> tuple[PathArray, PathArray]:
        rng = np.random.default_rng(seed)
        dW_S, dW_v = correlated_brownians(
            n_paths=n_paths,
            n_steps=n_steps,
            rho=self.heston.rho,
            dt=dt,
            rng=rng,
            antithetic=False,
        )
        spots = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        spots[:, 0] = self.spot0
        variances[:, 0] = self.heston.v0

        for k in range(n_steps):
            t = k * dt
            s = spots[:, k]
            v = variances[:, k]
            sqrt_v = np.sqrt(v)
            new_s, new_v = euler_maruyama_step(
                spot=s,
                variance=v,
                t=t,
                dt=dt,
                drift_S=self.drift * s,
                drift_v=self.heston.kappa * (self.heston.theta - v),
                diff_S=sqrt_v * s,
                diff_v=self.heston.xi * sqrt_v,
                dW_S=dW_S[:, k],
                dW_v=dW_v[:, k],
            )
            spots[:, k + 1] = new_s
            variances[:, k + 1] = new_v
        return spots, variances

    def step(self, state: SDEState, dt: float, dW: NDArray[np.float64]) -> SDEState:
        if dW.shape != (2,):
            raise ValueError(f"dW must have shape (2,), got {dW.shape}")

        prev_z = state.memory if state.memory is not None else 0.0
        s_arr = np.array([state.spot], dtype=np.float64)
        v_arr = np.array([state.variance], dtype=np.float64)
        sqrt_v = np.sqrt(v_arr)
        new_s, new_v = euler_maruyama_step(
            spot=s_arr,
            variance=v_arr,
            t=state.time,
            dt=dt,
            drift_S=self.drift * s_arr,
            drift_v=self.heston.kappa * (self.heston.theta - v_arr),
            diff_S=sqrt_v * s_arr,
            diff_v=self.heston.xi * sqrt_v,
            dW_S=np.array([dW[0]]),
            dW_v=np.array([dW[1]]),
        )
        new_spot = float(new_s[0])
        new_variance = float(new_v[0])

        # z_t evolves the same way as in reflexive (observed-but-not-fed-back).
        log_ratio = float(np.log(new_spot / self.spot0))
        new_z = prev_z + dt * (-self.memory_decay * prev_z + self.memory_intake * log_ratio)

        g = float(self.aggregator.compute(new_spot, new_variance, new_z))

        return SDEState(
            spot=new_spot,
            variance=new_variance,
            time=state.time + dt,
            aggregate_gamma=g,
            memory=new_z,
        )

    def implied_surface(self, state: SDEState, grid: SurfaceGrid) -> SurfaceArray:
        # Pure-Heston dynamics ⇒ Heston analytic surface. Defer to the Heston baseline
        # to avoid duplicating the QuantLib glue.
        from reflexive_options.baselines.heston import _quantlib_heston_iv_surface

        return _quantlib_heston_iv_surface(
            spot=state.spot,
            v0=state.variance,
            params=self.heston,
            grid=grid,
            drift=self.drift,
        )


__all__ = [
    "GAMMA_AGGREGATOR_AVAILABLE",
    "GammaAggregatorProtocol",
    "GammaAwareSimulator",
]

GAMMA_AGGREGATOR_AVAILABLE = _GAMMA_AGGREGATOR_AVAILABLE
