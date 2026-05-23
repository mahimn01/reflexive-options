"""Tests for the canonical type dataclasses + simulator protocol.

Covers the validation guards in `__post_init__` and the helper methods that
sit on the dataclasses; these are wiring rather than science but they are
asserted by every downstream caller and so deserve direct coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SDEState,
    SurfaceGrid,
)


def _grid_2x3() -> SurfaceGrid:
    return SurfaceGrid(
        log_moneyness=np.array([-0.1, 0.1], dtype=np.float64),
        maturities=np.array([0.05, 0.25, 1.0], dtype=np.float64),
    )


def test_open_interest_grid_rejects_shape_mismatch() -> None:
    """contracts_open shape must equal grid.shape."""
    grid = _grid_2x3()
    bad = np.zeros((3, 2), dtype=np.float64)  # transposed: incompatible
    with pytest.raises(ValueError, match="contracts_open shape"):
        OpenInterestGrid(grid=grid, contracts_open=bad)


def test_sde_state_to_array_uses_provided_gamma_and_memory() -> None:
    """When aggregate_gamma and memory are not None, to_array packs them
    verbatim into slots 3 and 4 of the returned vector.
    """
    state = SDEState(
        spot=123.45,
        variance=0.04,
        time=0.5,
        aggregate_gamma=2.5,
        memory=-0.7,
    )
    arr = state.to_array()
    assert arr.shape == (5,)
    np.testing.assert_allclose(arr, np.array([123.45, 0.04, 0.5, 2.5, -0.7]))


def test_sde_state_to_array_defaults_when_none() -> None:
    """When aggregate_gamma or memory are None, to_array substitutes zeros."""
    state = SDEState(spot=100.0, variance=0.04, time=0.0)
    arr = state.to_array()
    assert arr.shape == (5,)
    assert arr[3] == 0.0  # aggregate_gamma default
    assert arr[4] == 0.0  # memory default


def test_heston_params_feller_satisfied_true_and_false() -> None:
    """Feller's condition: 2 κ θ > ξ². Direct truth-table coverage."""
    p_yes = HestonParams(kappa=2.0, theta=0.04, xi=0.2, rho=-0.5, v0=0.04)
    # 2 * 2 * 0.04 = 0.16, ξ² = 0.04 — Feller holds
    assert p_yes.feller_satisfied() is True

    p_no = HestonParams(kappa=0.5, theta=0.01, xi=0.5, rho=-0.5, v0=0.01)
    # 2 * 0.5 * 0.01 = 0.01, ξ² = 0.25 — Feller fails
    assert p_no.feller_satisfied() is False


def test_reflexive_params_rejects_negative_coupling() -> None:
    base = HestonParams(kappa=2.0, theta=0.04, xi=0.2, rho=-0.5, v0=0.04)
    with pytest.raises(ValueError, match=r"coupling κ must be ≥ 0"):
        ReflexiveParams(base=base, coupling=-1e-12)


def test_reflexive_params_rejects_nonpositive_memory_decay() -> None:
    base = HestonParams(kappa=2.0, theta=0.04, xi=0.2, rho=-0.5, v0=0.04)
    with pytest.raises(ValueError, match=r"memory_decay α must be > 0"):
        ReflexiveParams(base=base, coupling=0.0, memory_decay=0.0)
