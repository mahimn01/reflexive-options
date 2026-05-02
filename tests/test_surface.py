"""Tests for the surface generator and parquet I/O."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from reflexive_options.surface.generator import (
    generate_surface,
    make_pre_reg_grid,
    make_standard_grid,
)
from reflexive_options.surface.io import load_surfaces, save_surfaces
from reflexive_options.types import SDEState, SurfaceGrid


def test_make_standard_grid_shape() -> None:
    grid = make_standard_grid(spot=5000.0)
    assert grid.shape == (11, 7)
    # Strikes symmetric around ATM.
    np.testing.assert_allclose(grid.log_moneyness[0], -grid.log_moneyness[-1])
    # Maturities strictly increasing, in years.
    assert np.all(np.diff(grid.maturities) > 0)
    # Cover 1w to 1y.
    assert grid.maturities[0] == pytest.approx(7 / 365.0)
    assert grid.maturities[-1] == pytest.approx(1.0)


def test_make_standard_grid_custom_dims() -> None:
    grid = make_standard_grid(spot=100.0, n_strikes=21, n_maturities=5)
    assert grid.shape == (21, 5)


def test_make_standard_grid_rejects_bad_spot() -> None:
    with pytest.raises(ValueError):
        make_standard_grid(spot=-1.0)


def test_make_pre_reg_grid_matches_locked_spec() -> None:
    """Pre-registration §4 locks the grid at 11 log-moneyness points with
    Δk = 0.04 (covering [-0.20, +0.20]) and 7 maturities {7, 14, 30, 60, 90,
    180, 365} days. Verify the factory returns exactly that.
    """
    grid = make_pre_reg_grid()

    # Strike axis
    assert grid.n_strikes == 11
    np.testing.assert_allclose(grid.log_moneyness[0], -0.20)
    np.testing.assert_allclose(grid.log_moneyness[-1], 0.20)
    # Δk = 0.04 between every consecutive pair
    deltas = np.diff(grid.log_moneyness)
    np.testing.assert_allclose(deltas, 0.04, atol=1e-12)

    # Maturity axis
    assert grid.n_maturities == 7
    expected_days = np.array([7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0])
    np.testing.assert_allclose(grid.maturities * 365.0, expected_days, atol=1e-12)

    # Per-window dim = 21 × 7 × 11 = 1617
    rolling_window_len = 21
    per_window_dim = rolling_window_len * grid.n_maturities * grid.n_strikes
    assert per_window_dim == 1617


def test_generate_surface_calls_simulator_method() -> None:
    grid = make_standard_grid(spot=100.0)
    state = SDEState(spot=100.0, variance=0.04, time=0.0)

    sim = MagicMock()
    expected = np.full(grid.shape, 0.2)
    sim.implied_surface.return_value = expected

    out = generate_surface(sim, state, grid, rate=0.01, dividend=0.0)

    sim.implied_surface.assert_called_once_with(state, grid)
    np.testing.assert_array_equal(out, expected)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    surfaces = 0.15 + 0.05 * rng.random((100, 11, 7))
    path = tmp_path / "surfaces.parquet"

    save_surfaces(surfaces, path, metadata={"experiment": "unit-test", "seed": 0})
    loaded, meta = load_surfaces(path)

    np.testing.assert_array_equal(loaded, surfaces)
    assert meta == {"experiment": "unit-test", "seed": 0}


def test_save_rejects_non_finite(tmp_path: Path) -> None:
    bad = np.full((1, 5, 5), np.nan)
    with pytest.raises(ValueError):
        save_surfaces(bad, tmp_path / "x.parquet")


def test_save_rejects_wrong_ndim(tmp_path: Path) -> None:
    bad = np.full((5, 5), 0.2)
    with pytest.raises(ValueError):
        save_surfaces(bad, tmp_path / "x.parquet")


def test_load_rejects_missing_shape_metadata(tmp_path: Path) -> None:
    """Reader requires reflexive_options.shape — a parquet without it is rejected."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"iv": pa.array([0.2, 0.3], type=pa.float64())})
    path = tmp_path / "no_shape.parquet"
    pq.write_table(table, path)  # type: ignore[no-untyped-call]
    with pytest.raises(ValueError, match=r"missing reflexive_options\.shape"):
        load_surfaces(path)


def test_load_rejects_row_count_mismatch(tmp_path: Path) -> None:
    """If metadata shape disagrees with the iv-column row count, raise."""
    import json as _json

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"iv": pa.array([0.2, 0.3, 0.4], type=pa.float64())})
    # Claim shape (1, 5, 5) → 25 rows, but only 3 are present.
    table = table.replace_schema_metadata(
        {b"reflexive_options.shape": _json.dumps([1, 5, 5]).encode("utf-8")}
    )
    path = tmp_path / "mismatch.parquet"
    pq.write_table(table, path)  # type: ignore[no-untyped-call]
    with pytest.raises(ValueError, match="does not match metadata shape"):
        load_surfaces(path)


def test_grid_attributes() -> None:
    g = SurfaceGrid(
        log_moneyness=np.linspace(-0.2, 0.2, 5),
        maturities=np.array([0.1, 0.5, 1.0]),
    )
    assert g.n_strikes == 5
    assert g.n_maturities == 3
    assert g.shape == (5, 3)
