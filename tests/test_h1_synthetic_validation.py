"""Smoketest for the H1 synthetic-validation pipeline.

The full validation runs the BC-MLP at 50 episodes + 100-day rollouts × three
sources and is too heavy for CI. The smoketest here confirms:

  - the experiment module imports and the smoketest CLI path runs end-to-end
    in <60 s on a CPU laptop;
  - the metrics dict has the required structure (per-source SW2 + CIs +
    ordering booleans);
  - the rich pseudo-surface generator and arbitrage filter are wired such that
    the κG-skew shift produces sign discrimination in some seeds (we don't
    assert the smoketest reaches the full a < b < c ordering — the budget is
    too small for tight CIs — only that the pipeline returns the right shape
    and the pseudo-surface is sensitive to κ).

The full ordering result on the production-budget run is reported in
`paper/main.tex` §5.4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reflexive_options.experiments.h1_synthetic_validation import (
    H1ValidationConfig,
    _heston_iv_at_state,
    _kappa_gamma_smile_shift,
    _stationary_block_indices,
    block_bootstrap_sw2,
    render_figure,
    run_h1_synthetic_validation,
)
from reflexive_options.types import HestonParams, SurfaceGrid

# ---------------------------------------------------------------------------
# Pseudo-surface generator + arbitrage-safe smile-shift
# ---------------------------------------------------------------------------


def _grid() -> SurfaceGrid:
    return SurfaceGrid(
        log_moneyness=np.linspace(-0.15, 0.15, 11, dtype=np.float64),
        maturities=np.array([14, 30, 60, 90, 180, 365], dtype=np.float64) / 365.0,
    )


def _base() -> HestonParams:
    return HestonParams(kappa=2.0, theta=0.04, xi=0.20, rho=-0.30, v0=0.04)


def test_heston_iv_at_state_returns_valid_surface() -> None:
    grid = _grid()
    surf = _heston_iv_at_state(100.0, 0.04, grid, base=_base())
    assert surf.shape == grid.shape
    assert np.all(np.isfinite(surf))
    assert (surf > 0.0).all()


def test_heston_iv_at_state_handles_blowup_gracefully() -> None:
    """Non-finite spot/variance returns a NaN surface (caller drops the day)."""
    grid = _grid()
    surf = _heston_iv_at_state(float("nan"), 0.04, grid, base=_base())
    assert surf.shape == grid.shape
    assert np.all(np.isnan(surf))
    surf2 = _heston_iv_at_state(-1.0, 0.04, grid, base=_base())
    assert np.all(np.isnan(surf2))


def test_kappa_gamma_smile_shift_zero_when_kappa_zero() -> None:
    grid = _grid()
    base_surf = _heston_iv_at_state(100.0, 0.04, grid, base=_base())
    shifted = _kappa_gamma_smile_shift(base_surface=base_surf, grid=grid, kappa=0.0, g_value=5.0e9)
    np.testing.assert_array_equal(shifted, base_surf)


def test_kappa_gamma_smile_shift_is_curvature_preserving_skew() -> None:
    """The shift is even in k, so skew (1st moment of IV across k) is preserved."""
    grid = _grid()
    base_surf = _heston_iv_at_state(100.0, 0.04, grid, base=_base())
    shifted = _kappa_gamma_smile_shift(
        base_surface=base_surf, grid=grid, kappa=1e-10, g_value=5.0e9
    )
    # Same value at k=+|k| and k=-|k| in the shift component; the difference
    # should be even, i.e. shifted - base_surf must be symmetric in k.
    delta = shifted - base_surf
    n_k = grid.n_strikes
    for j in range(grid.n_maturities):
        for i in range(n_k // 2):
            mirror = n_k - 1 - i
            assert delta[i, j] == pytest.approx(delta[mirror, j], abs=1e-12), (
                f"shift not symmetric at (i={i}, mirror={mirror}, j={j})"
            )


# ---------------------------------------------------------------------------
# Block-bootstrap SW2 helper
# ---------------------------------------------------------------------------


def test_stationary_block_indices_returns_valid_index_array() -> None:
    rng = np.random.default_rng(0)
    idx = _stationary_block_indices(50, block_length=5, rng=rng)
    assert idx.shape == (50,)
    assert (idx >= 0).all() and (idx < 50).all()


def test_block_bootstrap_sw2_returns_finite_ci() -> None:
    """Bootstrap on two iid samples returns a finite point + CI."""
    rng = np.random.default_rng(0)
    samples_left = rng.standard_normal((40, 8))
    samples_right = rng.standard_normal((40, 8))
    point, lo, hi, reps = block_bootstrap_sw2(
        samples_left,
        samples_right,
        n_bootstrap=20,
        block_length=5,
        n_slices=32,
        rng=rng,
    )
    assert np.isfinite(point)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi
    assert reps.shape == (20,)


# ---------------------------------------------------------------------------
# End-to-end smoketest — < 60 s budget on a CPU laptop
# ---------------------------------------------------------------------------


def test_h1_synthetic_validation_smoketest_runs(tmp_path: Path) -> None:
    """End-to-end smoketest: tiny budgets + just verify the metrics shape."""
    cfg = H1ValidationConfig(
        n_bc_train_episodes=4,
        bc_epochs=2,
        n_paths_per_source=6,
        n_days_per_path=30,
        n_slices=64,
        n_bootstrap=20,
        block_length=5,
        window_length=10,
        seed=11,
    )
    metrics = run_h1_synthetic_validation(cfg, tmp_path)
    # Required keys.
    for key in ("sw2", "ordering_holds", "ci_a_b_disjoint", "ci_b_c_disjoint"):
        assert key in metrics, f"missing top-level key {key!r}"
    sw2 = metrics["sw2"]
    for tag in ("source_a_kappa0_deployed", "source_b_2kappa0", "source_c_heston"):
        entry = sw2[tag]
        assert "distance" in entry
        assert "ci_low" in entry
        assert "ci_high" in entry
        assert entry["ci_low"] <= entry["ci_high"]
    # Persisted to disk.
    assert (tmp_path / "metrics.json").exists()
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert on_disk["config"]["seed"] == 11


def test_h1_synthetic_validation_figure_renders(tmp_path: Path) -> None:
    """`render_figure` writes a non-empty PDF with the three SW2 bars."""
    metrics = {
        "sw2": {
            "source_a_kappa0_deployed": {"distance": 0.005, "ci_low": 0.004, "ci_high": 0.007},
            "source_b_2kappa0": {"distance": 0.020, "ci_low": 0.015, "ci_high": 0.030},
            "source_c_heston": {"distance": 0.050, "ci_low": 0.040, "ci_high": 0.060},
        },
        "ordering_holds": True,
        "ci_a_b_disjoint": True,
        "ci_b_c_disjoint": True,
    }
    out_path = tmp_path / "h1_synthetic_ordering.pdf"
    render_figure(metrics, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000
