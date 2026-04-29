"""Path-rejection rules and blow-up detection.

Reflexive SDEs with positive feedback can explode (S → ∞ or v → ∞ faster than the
diffusion can damp). We need cheap runtime checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StabilityConfig:
    """Thresholds for declaring a path 'blown up' and rejecting it."""

    max_spot_multiplier: float = 100.0
    """If S_t > max_spot_multiplier * S_0, path is rejected. 100x is generous."""

    max_variance: float = 5.0
    """If variance > 5.0 (vol > ~224%/yr), path is rejected. SPX rarely exceeds 100% vol even in crashes."""

    nan_inf_check: bool = True
    """Reject any path containing NaN or inf at any timestep."""


@dataclass(frozen=True)
class StabilityResult:
    """Per-batch stability summary."""

    n_paths: int
    n_rejected: int
    rejection_mask: NDArray[np.bool_]  # shape (n_paths,), True = rejected
    reason_counts: dict[str, int]

    @property
    def survivor_fraction(self) -> float:
        return 1.0 - self.n_rejected / self.n_paths


def detect_blowup(
    spots: NDArray[np.float64],
    variances: NDArray[np.float64],
    initial_spot: float,
    cfg: StabilityConfig | None = None,
) -> StabilityResult:
    """Identify which paths blew up and should be excluded from analysis.

    Args:
        spots: shape (n_paths, n_steps + 1)
        variances: shape (n_paths, n_steps + 1)
        initial_spot: S_0, used for relative spot threshold

    Returns:
        StabilityResult with per-path rejection mask + reason counts.
    """
    cfg = cfg or StabilityConfig()
    n_paths = spots.shape[0]
    rejection = np.zeros(n_paths, dtype=bool)
    reasons: dict[str, int] = {}

    if cfg.nan_inf_check:
        nan_inf = (~np.isfinite(spots)).any(axis=1) | (~np.isfinite(variances)).any(axis=1)
        rejection |= nan_inf
        reasons["nan_or_inf"] = int(nan_inf.sum())

    spot_blowup = (spots > cfg.max_spot_multiplier * initial_spot).any(axis=1)
    rejection |= spot_blowup
    reasons["spot_blowup"] = int(spot_blowup.sum())

    variance_blowup = (variances > cfg.max_variance).any(axis=1)
    rejection |= variance_blowup
    reasons["variance_blowup"] = int(variance_blowup.sum())

    return StabilityResult(
        n_paths=n_paths,
        n_rejected=int(rejection.sum()),
        rejection_mask=rejection,
        reason_counts=reasons,
    )
