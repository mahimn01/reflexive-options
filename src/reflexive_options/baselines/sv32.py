"""3/2 stochastic volatility — robustness baseline.

dS/S = μ dt + √v dW_S
dv   = κ_v v (θ_v - v) dt + ξ v^{3/2} dW_v

Note the v factor in the variance drift and the v^{3/2} in the diffusion vs. Heston.
The 3/2 model fits short-dated smile shapes that Heston systematically misses
(Lewis 2000, Drimus 2012) and is a standard robustness check.

Numerics: full-truncation Euler with the variance floored at 0. The v^{3/2}
diffusion can be stiff at high v — for production use we recommend dt ≤ 1/(252·390)
(1-minute step) and/or a smaller diffusion clamp; we expose substep_factor as a knob.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reflexive_options.baselines._mc_iv import mc_iv_surface
from reflexive_options.simulator.integrators import (
    correlated_brownians,
    euler_maruyama_step,
)
from reflexive_options.types import (
    PathArray,
    SDEState,
    SurfaceArray,
    SurfaceGrid,
)


class SV32Simulator:
    """3/2 stochastic-vol simulator with full-truncation Euler."""

    def __init__(
        self,
        kappa_v: float,
        theta_v: float,
        xi: float,
        rho: float,
        v0: float,
        drift: float = 0.0,
        spot0: float = 100.0,
    ) -> None:
        if v0 <= 0:
            raise ValueError(f"v0 must be > 0, got {v0}")
        if theta_v <= 0:
            raise ValueError(f"theta_v must be > 0, got {theta_v}")
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {rho}")
        if spot0 <= 0:
            raise ValueError(f"spot0 must be > 0, got {spot0}")

        self.kappa_v = float(kappa_v)
        self.theta_v = float(theta_v)
        self.xi = float(xi)
        self.rho = float(rho)
        self.v0 = float(v0)
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
            rho=self.rho,
            dt=dt,
            rng=rng,
            antithetic=False,
        )
        spots = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        spots[:, 0] = self.spot0
        variances[:, 0] = self.v0

        for k in range(n_steps):
            t = k * dt
            s = spots[:, k]
            v = variances[:, k]
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            v_three_half = v_pos * sqrt_v
            new_s, new_v = euler_maruyama_step(
                spot=s,
                variance=v,
                t=t,
                dt=dt,
                drift_S=self.drift * s,
                drift_v=self.kappa_v * v_pos * (self.theta_v - v_pos),
                diff_S=sqrt_v * s,
                diff_v=self.xi * v_three_half,
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
        v_pos = np.maximum(v_arr, 0.0)
        sqrt_v = np.sqrt(v_pos)
        v_three_half = v_pos * sqrt_v
        new_s, new_v = euler_maruyama_step(
            spot=s_arr,
            variance=v_arr,
            t=state.time,
            dt=dt,
            drift_S=self.drift * s_arr,
            drift_v=self.kappa_v * v_pos * (self.theta_v - v_pos),
            diff_S=sqrt_v * s_arr,
            diff_v=self.xi * v_three_half,
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
