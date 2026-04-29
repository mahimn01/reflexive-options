"""κ stability check via linearization eigenvalues.

For the deterministic skeleton of the reflexive SDE
    dS/dt = μ S + κ G(S, t, v) S
    dv/dt = κ_v (θ_v - v)
the Jacobian at an equilibrium has eigenvalues that depend on κ. If the largest
real-part eigenvalue crosses zero as κ increases, the system loses stability —
candidate Hopf bifurcation.

`check_coupling_stable` is a fast pre-flight check before running an expensive simulation.
The full theoretical analysis lives in `theory.bifurcation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CouplingStability:
    """Result of the linearization-based stability check."""

    largest_real_part: float
    """max(Re(λ)) of the Jacobian. Stable iff < 0."""

    eigenvalues: NDArray[np.complex128]
    """All Jacobian eigenvalues, for diagnostic plots."""

    is_stable: bool
    """largest_real_part < tolerance."""

    is_oscillatory: bool
    """True if any eigenvalue has a non-zero imaginary part — implies cycles near equilibrium."""


def check_coupling_stable(
    jacobian: NDArray[np.float64],
    tolerance: float = -1e-8,
) -> CouplingStability:
    """Eigenvalue-based stability of a linearized reflexive system.

    Args:
        jacobian: 2x2 (or n×n) Jacobian matrix of the deterministic skeleton at equilibrium.
        tolerance: max(Re(λ)) < tolerance ⇒ stable. Slightly negative to be conservative.

    The actual Jacobian construction depends on the specific G(S, t, v) used, so this
    function takes the Jacobian as input. See `theory.bifurcation` for end-to-end scans
    over κ.
    """
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError(f"jacobian must be square 2D, got shape {jacobian.shape}")

    eigvals = np.linalg.eigvals(jacobian)
    largest_real = float(np.max(eigvals.real))
    has_imag = bool(np.any(np.abs(eigvals.imag) > 1e-12))

    return CouplingStability(
        largest_real_part=largest_real,
        eigenvalues=eigvals,
        is_stable=largest_real < tolerance,
        is_oscillatory=has_imag,
    )
