"""Local-stochastic volatility — secondary baseline.

dS/S = μ dt + L(S, t) √v dW_S
dv   = κ_v(θ_v - v) dt + ξ √v dW_v

L(S, t) is a fixed polynomial in (log-moneyness k = log(S/S0), time t):
    L(S, t) = 1 + a1 k + a2 k² + a3 k t

Calibration of L to a target surface (the Guyon–Henry-Labordère particle method
or PDE-projection methods of Engelmann/Lipton) is OUT OF SCOPE for v1 — that is
the standard literature bottleneck and not what this paper studies. v1 takes the
polynomial coefficients as input. The TODO is tracked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from reflexive_options.baselines._mc_iv import mc_iv_surface
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


class LeverageCoeffs(TypedDict, total=False):
    a1: float
    a2: float
    a3: float


@dataclass(frozen=True)
class _LeverageFn:
    a1: float
    a2: float
    a3: float
    spot0: float

    def evaluate(
        self,
        spot: NDArray[np.float64] | float,
        t: float,
    ) -> NDArray[np.float64] | float:
        k = np.log(np.asarray(spot, dtype=np.float64) / self.spot0)
        result = 1.0 + self.a1 * k + self.a2 * k * k + self.a3 * k * t
        if np.ndim(result) == 0:
            return float(np.asarray(result).item())
        return np.asarray(result, dtype=np.float64)


class LSVSimulator:
    """Local-stochastic vol with a fixed polynomial leverage surface."""

    def __init__(
        self,
        heston: HestonParams,
        leverage_coeffs: LeverageCoeffs,
        spot0: float = 100.0,
        drift: float = 0.0,
    ) -> None:
        if spot0 <= 0:
            raise ValueError(f"spot0 must be > 0, got {spot0}")
        self.heston = heston
        self.spot0 = float(spot0)
        self.drift = float(drift)
        self._leverage = _LeverageFn(
            a1=float(leverage_coeffs.get("a1", 0.0)),
            a2=float(leverage_coeffs.get("a2", 0.0)),
            a3=float(leverage_coeffs.get("a3", 0.0)),
            spot0=self.spot0,
        )
        # TODO: implement Guyon–Henry-Labordère particle calibration of L(S,t) to a
        # target IV surface. The MC inversion in `implied_surface` is a placeholder;
        # this is the standard literature bottleneck (~hours for one calibration).

    def leverage(
        self, spot: NDArray[np.float64] | float, t: float
    ) -> NDArray[np.float64] | float:
        return self._leverage.evaluate(spot, t)

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
            lev = np.asarray(self._leverage.evaluate(s, t), dtype=np.float64)
            new_s, new_v = euler_maruyama_step(
                spot=s,
                variance=v,
                t=t,
                dt=dt,
                drift_S=self.drift * s,
                drift_v=self.heston.kappa * (self.heston.theta - v),
                diff_S=lev * sqrt_v * s,
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
        s_arr = np.array([state.spot], dtype=np.float64)
        v_arr = np.array([state.variance], dtype=np.float64)
        sqrt_v = np.sqrt(v_arr)
        lev = np.asarray(self._leverage.evaluate(s_arr, state.time), dtype=np.float64)
        new_s, new_v = euler_maruyama_step(
            spot=s_arr,
            variance=v_arr,
            t=state.time,
            dt=dt,
            drift_S=self.drift * s_arr,
            drift_v=self.heston.kappa * (self.heston.theta - v_arr),
            diff_S=lev * sqrt_v * s_arr,
            diff_v=self.heston.xi * sqrt_v,
            dW_S=np.array([dW[0]]),
            dW_v=np.array([dW[1]]),
        )
        return SDEState(
            spot=float(new_s[0]),
            variance=float(new_v[0]),
            time=state.time + dt,
        )

    def implied_surface(
        self,
        state: SDEState,
        grid: SurfaceGrid,
        compute_surface: bool = False,
        n_paths: int = 20_000,
        n_steps_per_year: int = 252,
        seed: int | None = 0,
    ) -> SurfaceArray:
        if not compute_surface:
            return np.full(grid.shape, np.nan, dtype=np.float64)
        return mc_iv_surface(
            simulator=self,
            state=state,
            grid=grid,
            drift=self.drift,
            n_paths=n_paths,
            n_steps_per_year=n_steps_per_year,
            seed=seed,
        )
