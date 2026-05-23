"""Smoke test for the gamma-aware ablation runner stub.

The current implementation is a TODO placeholder gated on tasks #13/#14/#17;
exercising the entry point keeps lint + import-time errors visible and gives
the CLI runner basic coverage instrumentation until the real ablation lands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_ablation_config_defaults_round_trip() -> None:
    """Dataclass instantiates with the documented defaults."""
    from reflexive_options.experiments.ablation_gamma_aware import AblationConfig

    cfg = AblationConfig()
    assert cfg.n_seeds == 20
    assert cfg.n_eval_episodes == 100
    assert cfg.seed == 42


def test_ablation_main_writes_stub_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`main()` writes a stub metrics.json describing the gated-on-other-work
    status. We patch the runs directory to a temp tree so we don't pollute the
    repo on test runs and patch argv to set a tiny n_seeds.
    """
    from reflexive_options.experiments import _common, ablation_gamma_aware

    monkeypatch.setattr(_common, "RUNS_DIR", tmp_path / "runs")

    test_argv = ["ablation_gamma_aware", "--n-seeds", "2", "--seed", "7"]
    with patch.object(sys, "argv", test_argv):
        ablation_gamma_aware.main()

    runs = list((tmp_path / "runs" / "ablation_gamma_aware").iterdir())
    assert len(runs) == 1
    metrics_path = runs[0] / "metrics.json"
    assert metrics_path.exists()
    assert "stub" in metrics_path.read_text()
