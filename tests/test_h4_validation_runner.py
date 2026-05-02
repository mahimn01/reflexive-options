"""Smoke tests for `experiments.h4_validation` (the H4 detector CLI runner).

The detector itself is exhaustively tested in `test_h4_detector.py`. Here we
exercise the validation-suite scaffolding (config, run loop, plotting,
metrics writeout) with a tiny configuration so the runner doesn't drift
unnoticed if internals change.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np

from reflexive_options.experiments.h4_validation import (
    H4ValidationConfig,
    H4ValidationResult,
    _interp_threshold,
    plot_power_curves,
    run_validation,
)
from reflexive_options.experiments.h4_validation import (
    main as run_main,
)


def test_run_validation_tiny_config_returns_finite_results(tmp_path: Path) -> None:
    """Smallest plausible run of the H4 validation: a tiny grid still completes."""
    cfg = H4ValidationConfig(
        n_seeds_per_point=4,
        n_permutations=20,
        n_h0_seeds=8,
        t_grid=(256, 1024),
        snr_amplitudes=(0.40, 1.60),
        h0_t=512,
    )
    result = run_validation(cfg, base_seed=1)
    assert isinstance(result, H4ValidationResult)
    assert result.t_grid.tolist() == [256, 1024]
    # Per A2: per-signal dicts keyed by signal name.
    for sig in cfg.signals:
        assert result.power_vs_t[sig].shape == (2,)
        assert result.power_vs_snr[sig].shape == (2,)
        assert np.all(np.isfinite(result.power_vs_t[sig]))
        assert np.all(np.isfinite(result.power_vs_snr[sig]))
        assert 0.0 <= result.h0_fpr[sig] <= 1.0
        assert 0.0 <= result.h0_mean_p[sig] <= 1.0
    assert result.joint_power_vs_t.shape == (2,)
    assert result.joint_power_vs_snr.shape == (2,)
    assert 0.0 <= result.h0_joint_fpr <= 1.0
    # At fixed-SNR amplitude 0.4, longer T should give >= power than shorter.
    assert result.joint_power_vs_t[1] >= result.joint_power_vs_t[0] - 0.5
    # Higher SNR should give >= power than lower.
    assert result.joint_power_vs_snr[1] >= result.joint_power_vs_snr[0] - 0.5


def test_plot_power_curves_writes_pdf(tmp_path: Path) -> None:
    """The plotter writes a non-empty PDF to the requested path."""
    cfg = H4ValidationConfig(
        n_seeds_per_point=4,
        n_permutations=20,
        n_h0_seeds=4,
        t_grid=(256, 1024),
        snr_amplitudes=(0.40, 1.60),
        h0_t=512,
    )
    result = run_validation(cfg, base_seed=2)
    out = tmp_path / "h4_power.pdf"
    plot_power_curves(result, out_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 1_000  # PDFs are at minimum a few KB


def test_main_quick_writes_metrics_and_figure(tmp_path: Path, monkeypatch: object) -> None:
    """End-to-end: `python -m reflexive_options.experiments.h4_validation --quick`.

    Uses --signals abs_returns to keep wall-clock under 60 s in CI; the dual-
    signal A2 plumbing is exercised by `test_run_validation_tiny_config_returns_finite_results`.
    """
    args = ["h4_validation", "--quick", "--seed", "999", "--signals", "abs_returns"]
    with mock.patch.object(sys, "argv", args):
        run_main()

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs" / "h4_validation"
    latest = max(runs_dir.glob("*"), key=lambda p: p.stat().st_mtime)

    assert (latest / "config.json").exists()
    assert (latest / "metrics.json").exists()
    assert (latest / "power_curve.npz").exists()
    assert (latest / "h4_detector_power.pdf").exists()

    metrics = json.loads((latest / "metrics.json").read_text())
    assert "t_grid" in metrics
    # A2 schema: per-signal columns + joint columns.
    assert "joint_power_vs_t" in metrics
    assert "joint_power_vs_snr" in metrics
    assert "signals" in metrics
    assert "h0_joint_fpr_at_alpha_per_signal" in metrics
    assert 0.0 <= metrics["h0_joint_fpr_at_alpha_per_signal"] <= 1.0
    for sig in metrics["signals"]:
        assert f"power_vs_t__{sig}" in metrics
        assert f"power_vs_snr__{sig}" in metrics
        assert f"h0_fpr__{sig}" in metrics


def test_interp_threshold_returns_first_crossing() -> None:
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 0.5, 0.9, 1.0])
    # Threshold 0.9: linearly interpolates between (2.0, 0.5) and (3.0, 0.9).
    crossing = _interp_threshold(xs, ys, 0.9)
    assert crossing is not None
    assert abs(crossing - 3.0) < 1e-9


def test_interp_threshold_first_value_already_above() -> None:
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([1.0, 0.5, 0.2])
    crossing = _interp_threshold(xs, ys, 0.9)
    assert crossing == 1.0


def test_interp_threshold_returns_none_when_never_crossed() -> None:
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([0.1, 0.2, 0.3])
    crossing = _interp_threshold(xs, ys, 0.9)
    assert crossing is None


def test_interp_threshold_handles_flat_segment() -> None:
    """When the y-values are equal across the crossing bracket, fall back to x_hi."""
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([0.5, 0.9, 0.9])
    crossing = _interp_threshold(xs, ys, 0.9)
    assert crossing == 2.0


def test_h4_validation_config_defaults_are_pre_reg_aligned() -> None:
    """The default config matches the pre-reg's locked Welch settings."""
    cfg = H4ValidationConfig()
    assert cfg.welch_window == 1024
    assert cfg.welch_overlap == 0.5
    assert cfg.bandwidth_frac == 0.20
    assert cfg.alpha == 0.05


# Suppress the matplotlib "find_font" warnings on minimal CI font sets,
# in case the test runs in a stripped-down environment.
os.environ.setdefault("MPLBACKEND", "Agg")
