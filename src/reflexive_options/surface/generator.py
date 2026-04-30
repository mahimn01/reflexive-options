"""Surface generator — thin wrapper over `SimulatorProtocol.implied_surface`.

Design rationale: most simulators (Heston, SABR, reflexive analytic skeleton)
have a fast analytic or quasi-analytic IV computation. Monte-Carlo-only
simulators (LSV, 3/2) implement the same `implied_surface` method but route
through their MC IV pipeline. Centralising surface generation here would force
a one-size-fits-all path; instead we delegate and provide grid construction
helpers that every simulator can re-use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reflexive_options.types import SDEState, SimulatorProtocol, SurfaceArray, SurfaceGrid


def generate_surface(
    sim: SimulatorProtocol,
    state: SDEState,
    grid: SurfaceGrid,
    *,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> SurfaceArray:
    """Generate the model-implied IV surface at the given state.

    `rate` and `dividend` are accepted for API symmetry with the arbitrage
    filter; the simulator's own `implied_surface` is responsible for using
    them if relevant. Most analytic engines bake them into the discount
    factor / forward when the surface is constructed.
    """
    del rate, dividend  # reserved for future MC engines that need them at call time
    return sim.implied_surface(state, grid)


_DEFAULT_MATURITIES_YEARS: tuple[float, ...] = (
    7 / 365.0,  # 1 week
    14 / 365.0,  # 2 weeks
    1 / 12.0,  # 1 month
    2 / 12.0,  # 2 months
    3 / 12.0,  # 3 months
    6 / 12.0,  # 6 months
    1.0,  # 1 year
)


def make_standard_grid(
    spot: float,
    *,
    n_strikes: int = 11,
    n_maturities: int = 7,
    sigma_atm: float = 0.20,
    wing_sigmas: float = 2.5,
    maturities: NDArray[np.float64] | None = None,
) -> SurfaceGrid:
    """Canonical 11-strike x 7-maturity grid.

    Strikes are log-moneyness in `[-wing_sigmas * sigma_sqrtT, +wing_sigmas * sigma_sqrtT]`
    where `sigma_sqrtT = sigma_atm * sqrt(T_max)` (the longest maturity). This makes
    the strike range adapt to the volatility regime while keeping the relative
    coverage constant.

    Args:
        spot: only used to validate positivity; log-moneyness is spot-independent.
        n_strikes: number of strikes (default 11).
        n_maturities: number of maturities (default 7); ignored if `maturities` is given.
        sigma_atm: ATM vol used to scale the strike wings.
        wing_sigmas: half-width of the strike grid in sigma*sqrt(T_max) units.
        maturities: explicit maturity vector in years; overrides n_maturities/defaults.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if n_strikes < 3:
        raise ValueError(f"n_strikes must be >= 3, got {n_strikes}")

    if maturities is not None:
        T = np.asarray(maturities, dtype=np.float64)
    elif n_maturities == 7:
        T = np.asarray(_DEFAULT_MATURITIES_YEARS, dtype=np.float64)
    else:
        # Geometric spacing from 1 week to 1 year.
        T = np.geomspace(7 / 365.0, 1.0, n_maturities).astype(np.float64)

    if (T <= 0).any() or not np.all(np.diff(T) > 0):
        raise ValueError("maturities must be strictly positive and increasing")

    half_width = wing_sigmas * sigma_atm * float(np.sqrt(T.max()))
    k = np.linspace(-half_width, half_width, n_strikes)
    return SurfaceGrid(log_moneyness=k.astype(np.float64), maturities=T)


__all__ = ["generate_surface", "make_standard_grid"]
