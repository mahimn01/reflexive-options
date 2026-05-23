"""Smoke test for the bifurcation_scan experiment runner.

Exercises the CLI entry point at a tiny grid size + redirects artifacts to
tmp_path so we instrument the (otherwise uncovered) main() pipeline without
adding minutes to the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_bifurcation_config_defaults() -> None:
    """Dataclass round-trip + sanity on the canonical scan defaults."""
    from reflexive_options.experiments.bifurcation_scan import BifurcationConfig

    cfg = BifurcationConfig()
    assert cfg.n_kappa == 401
    assert cfg.n_sigma_v == 31
    assert cfg.kappa_v == 2.0
    # G_x sign convention: negative in long-gamma regime
    assert cfg.G_x < 0.0


def test_bifurcation_scan_main_writes_phase_diagram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run main() on a tiny grid (n_kappa=11, n_sigma_v=2) → persists metrics.json
    and phase_diagram.npz. Verifies the runner's full happy path."""
    from reflexive_options.experiments import _common, bifurcation_scan

    monkeypatch.setattr(_common, "RUNS_DIR", tmp_path / "runs")

    argv = ["bifurcation_scan", "--n-kappa", "11", "--n-sigma-v", "2"]
    with patch.object(sys, "argv", argv):
        bifurcation_scan.main()

    runs = list((tmp_path / "runs" / "bifurcation_scan").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "metrics.json").exists()
    assert (runs[0] / "phase_diagram.npz").exists()
