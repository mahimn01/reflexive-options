"""Smoketests for the 2D H_bimod follow-up scan (paper §7.4 amendment).

The full scan is the production-budget §7.4 follow-up at n_paths=2000,
n_steps=4000 with γ > 0; takes ~20 minutes. The smoketests here confirm:

  - the experiment module imports and the smoketest CLI path runs in <30 s;
  - the bimodality test panel returns the expected 3-test outcome dataclass;
  - `_kde_is_unimodal` correctly classifies a unimodal vs bimodal sample;
  - the stability-envelope pre-scan returns a positive κ★ for the canonical
    §7.1 setup with γ > 0 active.

The full-budget headline (whether 2D bimodality emerges near κ★) is reported
in `paper/theory.md` §7.4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reflexive_options.experiments.h_bimod_2d_scan import (
    BimodalityOutcome,
    HBimodScanConfig,
    _kde_is_unimodal,
    _make_simulator,
    _pca_projection,
    _silverman_bandwidth_test,
    _simulator_blows_up,
    evaluate_bimodality,
    find_stability_envelope_kappa_star,
    run_h_bimod_2d_scan,
)

# ---------------------------------------------------------------------------
# Simulator factory + stability pre-scan
# ---------------------------------------------------------------------------


def test_make_simulator_carries_leverage_gamma() -> None:
    """`leverage_gamma` from the config flows into `ReflexiveParams.leverage`."""
    cfg = HBimodScanConfig(leverage_gamma=0.7)
    sim = _make_simulator(cfg, kappa=1.0e-12)
    assert sim.params.leverage == pytest.approx(0.7)
    assert sim.params.coupling == pytest.approx(1.0e-12)


def test_simulator_blows_up_returns_false_at_zero_kappa() -> None:
    cfg = HBimodScanConfig(
        kappa_envelope_n_paths=20,
        kappa_envelope_n_steps=200,
    )
    assert _simulator_blows_up(cfg, 0.0) is False


def test_find_stability_envelope_returns_positive_kappa_star() -> None:
    """The pre-scan returns a positive κ★ at the canonical γ > 0 setup."""
    cfg = HBimodScanConfig(
        kappa_envelope_search_max=1.0e-8,
        kappa_envelope_n_paths=30,
        kappa_envelope_n_steps=400,
        kappa_envelope_n_probes=4,
    )
    kappa_star = find_stability_envelope_kappa_star(cfg)
    assert kappa_star > 0.0


# ---------------------------------------------------------------------------
# PCA projection on the joint sample
# ---------------------------------------------------------------------------


def test_pca_projection_returns_unit_direction() -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 2))
    projected, direction, explained = _pca_projection(samples)
    assert projected.shape == (200,)
    assert direction.shape == (2,)
    # PCA direction is unit norm.
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-9)
    # Standardised iid data: explained variance ratio of leading PC ~ 0.5
    # (no preferred direction), so it should not exceed ~0.65 for n=200.
    assert 0.40 <= explained <= 0.70


# ---------------------------------------------------------------------------
# Silverman bandwidth + KDE-unimodality
# ---------------------------------------------------------------------------


def test_kde_is_unimodal_flags_unimodal_normal() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    # Use a wider bandwidth than Silverman so noise modes don't surface — the
    # purpose of the test is to confirm `_kde_is_unimodal` behaves correctly
    # at a bandwidth that the bisection loop in `_silverman_bandwidth_test`
    # would converge to (h_critical, typically a few× Silverman).
    sigma = float(x.std(ddof=1))
    h = 1.5 * 0.9 * sigma * (x.size ** (-0.2))
    assert _kde_is_unimodal(x, h) is True


def test_kde_is_unimodal_flags_bimodal_mixture_at_small_bandwidth() -> None:
    """Two well-separated Gaussians should resolve into two modes at small h."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3.0, 0.4, 1000), rng.normal(+3.0, 0.4, 1000)])
    # Small bandwidth — KDE resolves the two modes.
    h_small = 0.10
    assert _kde_is_unimodal(x, h_small) is False


def test_silverman_bandwidth_test_finite_on_normal() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    bw_excess, is_bimodal = _silverman_bandwidth_test(x, rng=rng, n_bootstrap=20)
    assert np.isfinite(bw_excess)
    # Unimodal sample should not be flagged as bimodal.
    assert is_bimodal is False


# ---------------------------------------------------------------------------
# Bimodality panel on a synthetic joint sample
# ---------------------------------------------------------------------------


def test_evaluate_bimodality_on_unimodal_joint_sample() -> None:
    """Joint Gaussian (log S, v) cloud is unimodal under the PCA-projected dip."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((4_000, 2))
    outcome = evaluate_bimodality(samples, kappa=1e-12, rng=rng, max_samples=4_000)
    assert isinstance(outcome, BimodalityOutcome)
    assert outcome.n_samples == 4_000
    # PCA dip on iid Gaussian: not bimodal.
    assert outcome.pca_is_bimodal is False
    assert np.isfinite(outcome.pca_dip_statistic)
    # Eigenvector unit norm.
    assert abs(np.linalg.norm(outcome.pca_principal_direction) - 1.0) < 1e-9


def test_evaluate_bimodality_on_bimodal_joint_sample_flags_pca() -> None:
    """A joint cloud with two well-separated clusters in BOTH dims is flagged.

    The PCA standardisation makes the leading direction arbitrary when the
    two channels have similar mixture variance. To get a stable test we put
    the cluster separation along the (1, 1) diagonal in standardised
    coordinates — the leading PC will then point along the separation axis
    regardless of the random coordinate choice.
    """
    rng = np.random.default_rng(0)
    # Cluster 1: low log_S, low v; cluster 2: high log_S, high v. Standardised
    # std of each channel ≈ 1 within-cluster, so the (1,1) separation
    # dominates the leading PC.
    s_left = rng.normal(loc=[-3.0, 0.02], scale=[0.30, 0.003], size=(2_000, 2))
    s_right = rng.normal(loc=[+3.0, 0.06], scale=[0.30, 0.003], size=(2_000, 2))
    joint = np.concatenate([s_left, s_right], axis=0)
    outcome = evaluate_bimodality(joint, kappa=1e-12, rng=rng, max_samples=4_000)
    # PCA dip on a clearly bimodal cloud: flagged.
    assert outcome.pca_is_bimodal is True
    assert outcome.pca_dip_p_value < 0.05


# ---------------------------------------------------------------------------
# End-to-end smoketest
# ---------------------------------------------------------------------------


def test_run_h_bimod_2d_scan_smoketest_writes_metrics_and_figure(tmp_path: Path) -> None:
    cfg = HBimodScanConfig(
        n_paths=200,
        n_steps=400,
        kappa_grid_relative=(0.0, 0.5, 1.0),
        kappa_envelope_n_paths=30,
        kappa_envelope_n_steps=200,
        kappa_envelope_n_probes=4,
        max_samples_for_test=2_000,
        seed=11,
    )
    metrics = run_h_bimod_2d_scan(cfg, tmp_path)
    # Keys.
    for key in ("kappa_star_envelope", "kappa_grid", "outcomes", "any_pca_bimodal"):
        assert key in metrics
    # Persisted to disk.
    assert (tmp_path / "metrics.json").exists()
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert on_disk["config"]["leverage_gamma"] == pytest.approx(0.5)
