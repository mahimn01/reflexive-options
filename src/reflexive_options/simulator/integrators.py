"""SDE integrators — Euler-Maruyama with antithetic variates, Milstein optional.

These are model-agnostic — they take drift/diffusion functions and step them.
Used by simulator.reflexive and all baselines.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IntegratorConfig:
    """Configuration for the Monte-Carlo integrator."""

    n_paths: int = 50_000
    dt: float = 1.0 / (252 * 390)  # 1-minute step (390 mins / trading day, 252 days / year)
    antithetic: bool = True  # mirror dW for variance reduction
    seed: int | None = None


def correlated_brownians(
    n_paths: int,
    n_steps: int,
    rho: float,
    dt: float,
    rng: np.random.Generator,
    antithetic: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate two correlated Brownian increment arrays of shape (n_paths, n_steps).

    Returns (dW_S, dW_v) with d⟨W_S, W_v⟩ = ρ dt.
    Antithetic variance reduction halves the effective sample count then mirrors.
    """
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")
        half = n_paths // 2
        z1_half = rng.standard_normal((half, n_steps))
        z2_half = rng.standard_normal((half, n_steps))
        z1 = np.concatenate([z1_half, -z1_half], axis=0)
        z2 = np.concatenate([z2_half, -z2_half], axis=0)
    else:
        z1 = rng.standard_normal((n_paths, n_steps))
        z2 = rng.standard_normal((n_paths, n_steps))

    # Correlate: dW_v = ρ dW_S + √(1-ρ²) dZ
    dW_S = z1 * np.sqrt(dt)
    dW_v = (rho * z1 + np.sqrt(1.0 - rho**2) * z2) * np.sqrt(dt)
    return dW_S, dW_v


DriftFn = Callable[[NDArray[np.float64], NDArray[np.float64], float], NDArray[np.float64]]
"""Signature: (S, v, t) -> drift_S OR drift_v.  Vectorized over paths."""

DiffusionFn = Callable[[NDArray[np.float64], NDArray[np.float64], float], NDArray[np.float64]]
"""Signature: (S, v, t) -> diffusion_S OR diffusion_v.  Vectorized over paths."""


def euler_maruyama_step(
    spot: NDArray[np.float64],
    variance: NDArray[np.float64],
    t: float,
    dt: float,
    drift_S: NDArray[np.float64],
    drift_v: NDArray[np.float64],
    diff_S: NDArray[np.float64],
    diff_v: NDArray[np.float64],
    dW_S: NDArray[np.float64],
    dW_v: NDArray[np.float64],
    floor_variance: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One Euler-Maruyama step for a 2D SDE.

    Floor variance at `floor_variance` to handle Feller-violation paths gracefully
    (full-truncation scheme — Lord, Koekkoek, van Dijk 2010).
    """
    new_spot = spot + drift_S * dt + diff_S * dW_S
    new_variance = variance + drift_v * dt + diff_v * dW_v
    np.maximum(new_variance, floor_variance, out=new_variance)
    return new_spot, new_variance
