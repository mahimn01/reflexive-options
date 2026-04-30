"""Tests for the (κ, σ_v) phase-diagram experiment.

Targets `experiments/phase_diagram.py`. The grid-classification pipeline is
currently stubbed (TODO referencing simulator task #13), but `classify_regime`
is real and `main()` exercises the surrounding plumbing — config persistence,
grid construction, npz dump.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from reflexive_options.experiments import phase_diagram
from reflexive_options.experiments.phase_diagram import (
    PhaseDiagramConfig,
    classify_regime,
    main,
)

# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------


def _flat_paths(spot_value: float, n_paths: int = 4, n_steps: int = 256) -> np.ndarray:
    return np.full((n_paths, n_steps), spot_value, dtype=np.float64)


def test_classify_regime_blowup_on_nonfinite_spots() -> None:
    spots = _flat_paths(100.0)
    spots[0, 10] = np.inf
    variances = np.full_like(spots, 0.04)
    assert classify_regime(spots, variances, initial_spot=100.0) == "blow_up"


def test_classify_regime_blowup_on_nonfinite_variances() -> None:
    spots = _flat_paths(100.0)
    variances = np.full_like(spots, 0.04)
    variances[1, 5] = np.nan
    assert classify_regime(spots, variances, initial_spot=100.0) == "blow_up"


def test_classify_regime_blowup_on_runaway_spot() -> None:
    """Spot crossing 100× the initial value triggers the blowup classifier."""
    spots = _flat_paths(100.0)
    spots[0, 50] = 100.0 * 100.0 + 1.0  # > 100 × initial_spot
    variances = np.full_like(spots, 0.04)
    assert classify_regime(spots, variances, initial_spot=100.0) == "blow_up"


def test_classify_regime_calm_constant_paths() -> None:
    """Constant spots → zero log-returns, zero variance-of-variance → calm."""
    spots = _flat_paths(100.0)
    variances = np.full_like(spots, 0.04)
    assert classify_regime(spots, variances, initial_spot=100.0) == "calm"


def test_classify_regime_limit_cycle_high_var_of_var() -> None:
    """Variance trajectories with var(v) per path > 0.5 → limit_cycle."""
    rng = np.random.default_rng(0)
    n_paths, n_steps = 8, 512
    spots = 100.0 + rng.normal(scale=0.01, size=(n_paths, n_steps)).cumsum(axis=1)
    # Inject a wide-amplitude variance oscillation: var across time ≈ 1.0 ≫ 0.5.
    t = np.arange(n_steps)
    variances = 0.5 + np.sqrt(2.0) * np.sin(t * 0.3)[None, :].repeat(n_paths, axis=0)
    assert classify_regime(spots, variances, initial_spot=100.0) == "limit_cycle"


def test_classify_regime_vol_cluster_on_autocorrelated_returns() -> None:
    """High lag-1 autocorrelation in |log returns| with low var(v) → vol_cluster."""
    rng = np.random.default_rng(3)
    n_paths, n_steps = 4, 512
    # |log return| follows a slow sinusoid (highly autocorrelated by
    # construction), random ±1 signs. Drives ac1 ≫ 0.2 while keeping var-of-var
    # at zero (constant variance path), so the classifier picks vol_cluster
    # over limit_cycle.
    t = np.arange(n_steps - 1)
    magnitude = 0.001 + 0.01 * (0.5 + 0.5 * np.sin(0.01 * t))
    signs = rng.choice([-1.0, 1.0], size=(n_paths, n_steps - 1))
    log_returns = magnitude[None, :] * signs
    log_spots = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1)
    spots = 100.0 * np.exp(log_spots)
    variances = np.full_like(spots, 0.04)  # var-of-var = 0 ⇒ not limit_cycle.
    assert classify_regime(spots, variances, initial_spot=100.0) == "vol_cluster"


def test_classify_regime_short_paths_skip_autocorr() -> None:
    """≤2 return-steps short-circuits the autocorrelation branch (ac1=0) ⇒ calm."""
    spots = np.array([[100.0, 100.5, 100.2]])
    variances = np.full_like(spots, 0.04)
    assert classify_regime(spots, variances, initial_spot=100.0) == "calm"


def test_classify_regime_single_step_paths_no_returns() -> None:
    """n_steps=2 → only 1 log-return per path (shape[1]<2) hits the else-branch ac1=0."""
    spots = np.array([[100.0, 101.0], [100.0, 99.0]])
    variances = np.full_like(spots, 0.04)
    assert classify_regime(spots, variances, initial_spot=100.0) == "calm"


# ---------------------------------------------------------------------------
# PhaseDiagramConfig
# ---------------------------------------------------------------------------


def test_phase_diagram_config_defaults() -> None:
    cfg = PhaseDiagramConfig()
    assert cfg.kappa_min == 0.0
    assert cfg.kappa_max == 1.0
    assert cfg.n_kappa == 21
    assert cfg.n_sigma_v == 16
    assert cfg.seed == 42


def test_phase_diagram_config_is_frozen() -> None:
    cfg = PhaseDiagramConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.seed = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# main() smoketest — exercise the file-writing plumbing on a tiny grid.
# ---------------------------------------------------------------------------


def test_main_writes_config_metrics_and_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() with --n-kappa 2 --n-sigma-v 2 dumps config.json, metrics.json, npz."""
    # Redirect RUNS_DIR away from the repo so the test leaves no on-disk droppings.
    monkeypatch.setattr(phase_diagram, "make_run_dir", _stub_make_run_dir(tmp_path))
    monkeypatch.setattr(sys, "argv", ["phase_diagram", "--n-kappa", "2", "--n-sigma-v", "2"])

    main()

    # Exactly one timestamped run dir was created under tmp_path.
    children = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(children) == 1
    run_dir = children[0]

    # config.json round-trips the dataclass.
    cfg_payload = json.loads((run_dir / "config.json").read_text())
    assert cfg_payload["n_kappa"] == 2
    assert cfg_payload["n_sigma_v"] == 2
    assert cfg_payload["seed"] == 42

    # metrics.json has the expected grids.
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert {"kappa_grid", "sigma_v_grid", "regime_grid", "blowup_fraction"}.issubset(metrics)
    assert len(metrics["kappa_grid"]) == 2
    assert len(metrics["sigma_v_grid"]) == 2

    # npz round-trip.
    with np.load(run_dir / "phase_diagram.npz", allow_pickle=True) as npz:
        assert npz["kappa_grid"].shape == (2,)
        assert npz["sigma_v_grid"].shape == (2,)
        assert npz["regime_grid"].shape == (2, 2)
        assert npz["blowup_fraction"].shape == (2, 2)
        # Stub fills every cell with "calm" / 0.0 — verify so a regression
        # surfaces the moment the simulator wiring lands.
        assert (npz["regime_grid"] == "calm").all()
        np.testing.assert_array_equal(npz["blowup_fraction"], np.zeros((2, 2)))


def test_main_seed_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--seed flag flows into the persisted config and the run-dir suffix."""
    monkeypatch.setattr(phase_diagram, "make_run_dir", _stub_make_run_dir(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["phase_diagram", "--n-kappa", "2", "--n-sigma-v", "2", "--seed", "1234"],
    )
    main()
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert json.loads((run_dir / "config.json").read_text())["seed"] == 1234


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stub_make_run_dir(root: Path) -> Callable[..., Path]:
    """Replacement for experiments._common.make_run_dir that lands under `root`."""
    counter = {"n": 0}

    def _factory(experiment_name: str, *, seed: int | None = None) -> Path:
        counter["n"] += 1
        suffix = f"_seed{seed}" if seed is not None else ""
        run = root / f"{experiment_name}_{counter['n']:04d}{suffix}"
        run.mkdir(parents=True, exist_ok=True)
        return run

    return _factory
