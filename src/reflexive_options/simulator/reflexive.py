"""Reflexive 3D simulator — central artifact of the paper.

Implements the SDE in paper/theory.md §1:
    dS/S = (μ + κ G(S, z, v)) dt + σ(S, v) dW_S
    dv   = (κ_v(θ_v - v) + γ z) dt + ξ √v dW_v
    dz   = (-α z + β log(S/S_0)) dt
with d⟨W_S, W_v⟩ = ρ dt. Reduces to standard Heston when κ = γ = 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.integrators import (
    correlated_brownians,
    euler_maruyama_step,
)
from reflexive_options.types import (
    PathArray,
    ReflexiveParams,
    SDEState,
    SurfaceArray,
    SurfaceGrid,
)


class ReflexiveSimulator:
    """Monte-Carlo + single-step interface for the 3D reflexive SDE."""

    def __init__(
        self,
        params: ReflexiveParams,
        gamma_aggregator: GammaAggregator,
        initial_spot: float,
        surface_grid: SurfaceGrid | None = None,
        variance_floor: float = 1e-8,
        antithetic: bool = True,
    ) -> None:
        self.params = params
        self.gamma_aggregator = gamma_aggregator
        self.initial_spot = initial_spot
        self.surface_grid = surface_grid
        self.variance_floor = variance_floor
        self.antithetic = antithetic
        self._log_s0 = np.log(initial_spot)

    def _g_vectorized(
        self,
        spots: NDArray[np.float64],
        variances: NDArray[np.float64],
        memories: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # gamma_aggregator.compute is path-scalar; loop over paths. n_paths is bounded
        # by the MC budget (~1e4–1e5) so this is acceptable for v1; vectorizing requires
        # rewriting the BS-gamma grid to broadcast over paths × strikes × maturities.
        n_paths = spots.shape[0]
        out = np.empty(n_paths, dtype=np.float64)
        for i in range(n_paths):
            out[i] = self.gamma_aggregator.compute(
                float(spots[i]), float(variances[i]), float(memories[i])
            )
        return out

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> tuple[PathArray, PathArray]:
        rng = np.random.default_rng(seed)
        antithetic = self.antithetic and (n_paths % 2 == 0)
        dW_S, dW_v = correlated_brownians(
            n_paths=n_paths,
            n_steps=n_steps,
            rho=self.params.base.rho,
            dt=dt,
            rng=rng,
            antithetic=antithetic,
        )

        spots = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        memories = np.empty((n_paths, n_steps + 1), dtype=np.float64)

        spots[:, 0] = self.initial_spot
        variances[:, 0] = self.params.base.v0
        memories[:, 0] = 0.0

        kappa_v = self.params.base.kappa
        theta_v = self.params.base.theta
        xi = self.params.base.xi
        kappa = self.params.coupling
        mu = self.params.drift
        gamma = self.params.leverage
        alpha = self.params.memory_decay
        beta = self.params.memory_intake

        # Suppress overflow / invalid-value warnings inside the loop. At extreme
        # κ (e.g. stress-test paths in `test_tail_index_increases_with_kappa`)
        # the SDE legitimately blows up; the resulting +inf / NaN values are
        # caught downstream by `detect_blowup` and `floor_variance`. Letting
        # numpy emit warnings here would just be noise.
        with np.errstate(over="ignore", invalid="ignore"):
            for k in range(n_steps):
                s_k = spots[:, k]
                v_k = variances[:, k]
                z_k = memories[:, k]

                g_vec = (
                    self._g_vectorized(s_k, v_k, z_k)
                    if kappa != 0.0
                    else np.zeros(n_paths, dtype=np.float64)
                )

                sqrt_v = np.sqrt(np.maximum(v_k, 0.0))
                drift_S = s_k * (mu + kappa * g_vec)
                diff_S = s_k * sqrt_v
                drift_v = kappa_v * (theta_v - v_k) + gamma * z_k
                diff_v = xi * sqrt_v

                new_spot, new_variance = euler_maruyama_step(
                    spot=s_k,
                    variance=v_k,
                    t=k * dt,
                    dt=dt,
                    drift_S=drift_S,
                    drift_v=drift_v,
                    diff_S=diff_S,
                    diff_v=diff_v,
                    dW_S=dW_S[:, k],
                    dW_v=dW_v[:, k],
                    floor_variance=self.variance_floor,
                )
                # explicit Euler on the deterministic z-channel
                log_ratio = np.log(np.maximum(s_k, 1e-12)) - self._log_s0
                new_memory = z_k + (-alpha * z_k + beta * log_ratio) * dt

                spots[:, k + 1] = new_spot
                variances[:, k + 1] = new_variance
                memories[:, k + 1] = new_memory

        # memories are the *third* state; the protocol returns (spots, variances).
        # The full trajectory of z is reconstructable deterministically from S,
        # so dropping it preserves the API without information loss.
        return spots, variances

    def drift(self, spot: float, variance: float, memory: float) -> NDArray[np.float64]:
        """Pure deterministic drift of the 3D skeleton at state (S, v, z).

        Returns the RHS f(x) of the noiseless ODE dx/dt = f(x) where the state
        vector is x = (log S, v, z). Working in log-spot removes the multiplicative
        scaling and matches the convention used in paper/theory.md §2 for the
        Jacobian and the Hopf bifurcation analysis.

        Used by `theory.bifurcation.build_bilinear_trilinear_tensors` to extract
        B (quadratic) and C (cubic) Taylor coefficients via finite differences.
        """
        s = max(spot, 1e-12)
        v = max(variance, 0.0)
        kappa = self.params.coupling
        g = self.gamma_aggregator.compute(s, v, memory) if kappa != 0.0 else 0.0
        # d(log S)/dt = μ + κ G - 0.5 v   (Itô correction since σ² ≈ v)
        drift_log_s = self.params.drift + kappa * g - 0.5 * v
        drift_v = (
            self.params.base.kappa * (self.params.base.theta - v) + self.params.leverage * memory
        )
        drift_z = -self.params.memory_decay * memory + self.params.memory_intake * (
            np.log(s) - self._log_s0
        )
        return np.array([drift_log_s, drift_v, drift_z], dtype=np.float64)

    def step(self, state: SDEState, dt: float, dW: NDArray[np.float64]) -> SDEState:
        if dW.shape != (2,):
            raise ValueError(f"dW must have shape (2,), got {dW.shape}")
        s = state.spot
        v = max(state.variance, 0.0)
        z = state.memory if state.memory is not None else 0.0
        sqrt_v = float(np.sqrt(v))

        kappa = self.params.coupling
        g = self.gamma_aggregator.compute(s, v, z) if kappa != 0.0 else 0.0

        drift_S = s * (self.params.drift + kappa * g)
        diff_S = s * sqrt_v
        drift_v = self.params.base.kappa * (self.params.base.theta - v) + self.params.leverage * z
        diff_v = self.params.base.xi * sqrt_v

        new_spot = s + drift_S * dt + diff_S * float(dW[0])
        new_variance = max(v + drift_v * dt + diff_v * float(dW[1]), self.variance_floor)
        new_z = (
            z
            + (
                -self.params.memory_decay * z
                + self.params.memory_intake * (np.log(max(s, 1e-12)) - self._log_s0)
            )
            * dt
        )

        return SDEState(
            spot=float(new_spot),
            variance=float(new_variance),
            time=state.time + dt,
            aggregate_gamma=float(g),
            memory=float(new_z),
        )

    def implied_surface(self, state: SDEState, grid: SurfaceGrid) -> SurfaceArray:
        # TODO: proper surface generator lives in `reflexive_options.surface`. v1 returns a
        # flat IV = √v across the whole grid — adequate for downstream RL state-encoding
        # but not for option-pricing experiments which need re-pricing at every (K, T).
        sigma = float(np.sqrt(max(state.variance, 1e-12)))
        return np.full(grid.shape, sigma, dtype=np.float64)
