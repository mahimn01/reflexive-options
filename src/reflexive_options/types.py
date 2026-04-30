"""Canonical types shared across the package.

All simulators (reflexive + baselines) implement `SimulatorProtocol` so the RL env,
surface generator, and experiments can swap them transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Numeric array aliases
# ---------------------------------------------------------------------------

PathArray = NDArray[np.float64]
"""Shape (n_paths, n_steps + 1) — Monte-Carlo simulator output, indexed by path & time."""

SurfaceArray = NDArray[np.float64]
"""Shape (n_strikes, n_maturities) — IV (or call-price) on a strike-maturity grid."""


@dataclass(frozen=True)
class SurfaceGrid:
    """The strike-maturity grid that surfaces are sampled on."""

    log_moneyness: NDArray[np.float64]  # shape (n_strikes,), centered at 0 = ATM
    maturities: NDArray[np.float64]  # shape (n_maturities,) in years

    @property
    def n_strikes(self) -> int:
        return int(self.log_moneyness.shape[0])

    @property
    def n_maturities(self) -> int:
        return int(self.maturities.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_strikes, self.n_maturities)


@dataclass(frozen=True)
class OpenInterestGrid:
    """Open-interest grid: contracts open per (strike, maturity).

    Used by the dealer-gamma aggregator to compute G(S, t).
    Sign convention for dealer position is handled separately in
    `simulator.gamma_aggregator`.
    """

    grid: SurfaceGrid
    contracts_open: NDArray[np.float64]  # shape (n_strikes, n_maturities)

    def __post_init__(self) -> None:
        if self.contracts_open.shape != self.grid.shape:
            raise ValueError(
                f"contracts_open shape {self.contracts_open.shape} does not match "
                f"grid shape {self.grid.shape}"
            )


@dataclass(frozen=True)
class GreekGrid:
    """Black-Scholes greeks on a (strike, maturity) grid for a single spot/v snapshot."""

    grid: SurfaceGrid
    delta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    theta: NDArray[np.float64]


@dataclass
class SDEState:
    """Instantaneous state of a stochastic-volatility simulator.

    Reflexive simulators carry G and the memory variable z alongside (S, v);
    baselines leave G and z as None.

    The memory variable z (low-pass-filtered log-price) is required for the
    3D Hopf bifurcation analysis — see paper/theory.md and the brief at
    ../reflexivity-research/hopf_bifurcation_brief.md §3.2 for why the bare
    2D (S, v) skeleton cannot Hopf.
    """

    spot: float
    variance: float
    time: float
    aggregate_gamma: float | None = None  # None for non-reflexive baselines
    memory: float | None = None  # z = low-pass(log S); None for baselines

    def to_array(self) -> NDArray[np.float64]:
        g = self.aggregate_gamma if self.aggregate_gamma is not None else 0.0
        z = self.memory if self.memory is not None else 0.0
        return np.array([self.spot, self.variance, self.time, g, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Model parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HestonParams:
    """Time-dependent Heston parameters for one regime.

    Multiple regimes are handled by a list[HestonParams] + breakpoints in baselines.heston.
    """

    kappa: float  # mean-reversion speed of variance
    theta: float  # long-run variance
    xi: float  # vol-of-vol
    rho: float  # correlation between dW_S and dW_v
    v0: float  # initial variance

    def feller_satisfied(self) -> bool:
        """Feller condition: 2 κ θ > ξ². Required to keep variance strictly positive."""
        return 2.0 * self.kappa * self.theta > self.xi**2


@dataclass(frozen=True)
class ReflexiveParams:
    """Parameters for the reflexive 3D simulator.

    State: (S, v, z) where z is a low-pass-filtered log-price (memory channel).

    SDE:
        dS/S = (μ + κ · G(S, z, v)) dt + σ(S, v) dW_S
        dv   = (κ_v(θ_v - v) + γ z) dt + ξ √v dW_v
        dz   = (-α z + β log(S/S_0)) dt
    Reduces to standard time-dep Heston when κ = γ = 0.

    See paper/theory.md and ../reflexivity-research/hopf_bifurcation_brief.md
    for derivation of why the memory channel is necessary for Hopf bifurcation.
    """

    base: HestonParams
    coupling: float  # κ in the drift coupling κ G(S, z, v); literature prior O(1e-12) per USD dollar-gamma
    drift: float = 0.0  # μ; usually risk-neutral or fitted
    memory_decay: float = 1.0  # α in dz = -α z + β log(S/S_0). Units: 1/year. Default ~1 day.
    memory_intake: float = 1.0  # β in dz = -α z + β log(S/S_0). Dimensionless.
    leverage: float = 0.0  # γ in dv = (κ_v(θ_v - v) + γ z) dt + ... — closes the feedback loop

    def __post_init__(self) -> None:
        if self.coupling < 0:
            raise ValueError(f"coupling κ must be ≥ 0, got {self.coupling}")
        if self.memory_decay <= 0:
            raise ValueError(f"memory_decay α must be > 0, got {self.memory_decay}")


# ---------------------------------------------------------------------------
# Simulator interface
# ---------------------------------------------------------------------------


@runtime_checkable
class SimulatorProtocol(Protocol):
    """Common interface for all simulators (reflexive + baselines).

    Critical: identical interface across reflexive/non-reflexive lets the RL env,
    surface generator, and evaluation pipeline be model-agnostic. This is *the*
    architectural commitment of the package.
    """

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> tuple[PathArray, PathArray]:
        """Simulate (spot, variance) paths.

        Returns:
            spots: shape (n_paths, n_steps + 1)
            variances: shape (n_paths, n_steps + 1)
        """
        ...

    def step(self, state: SDEState, dt: float, dW: NDArray[np.float64]) -> SDEState:
        """Advance the state by dt given a 2-vector of correlated Brownian increments.

        Used by the RL env which needs single-step control instead of batch simulate().
        dW shape: (2,) for (dW_S, dW_v).
        """
        ...

    def implied_surface(
        self,
        state: SDEState,
        grid: SurfaceGrid,
    ) -> SurfaceArray:
        """Compute the model-implied IV surface at the given state."""
        ...
