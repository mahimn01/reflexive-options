"""Tests for the no-Hopf-wedge taxonomy (paper §3.5 extension, Theorem 6).

Cover three things:
    1. `is_in_no_hopf_wedge` correctly classifies known anchors.
    2. `scan_no_hopf_wedge_bifurcations` returns the Theorem 6(a) verdict
       on the canonical scan window: every wedge cell is globally
       asymptotically stable, no positive saddle-node, max Re λ < 0.
    3. The `NoHopfWedgeScanResult` dataclass round-trips its summary scalars
       consistently with the underlying grids.
"""

from __future__ import annotations

import numpy as np

from reflexive_options.theory.bifurcation import (
    NoHopfWedgeScanResult,
    bifurcations_in_no_hopf_wedge,
    is_in_no_hopf_wedge,
    scan_no_hopf_wedge_bifurcations,
)

_CANONICAL = dict(
    mu_q=float(np.log(100.0)),
    T_eff=0.25,
    kappa_v=2.0,
    alpha=0.05,
    beta=1.0,
    a_star=float(np.log(100.0)),
    v_star=0.04,
)


def test_is_in_no_hopf_wedge_classifies_known_anchors() -> None:
    """At the canonical specification, low-γ anchors fall inside the wedge
    (because H(0) > 0 requires γ < 2 α κ_v (α + κ_v) / β ≈ 0.41), while
    high-γ anchors admit a positive κ★ and so are outside."""
    in_wedge_anchor = is_in_no_hopf_wedge(sigma_q=0.10, gamma=0.1, **_CANONICAL)
    assert in_wedge_anchor is True, "(σ_q=0.10, γ=0.1) should be in the wedge"

    # γ=5 is well above the wedge boundary — Hopf is accessible.
    hopf_anchor = is_in_no_hopf_wedge(sigma_q=0.10, gamma=5.0, **_CANONICAL)
    assert hopf_anchor is False, "(σ_q=0.10, γ=5.0) should admit a positive κ★"


def test_bifurcations_in_wedge_reports_global_stability_at_canonical() -> None:
    """At a wedge anchor, all three closed-form indicators (Hopf, SN, c_2-flip)
    should be empty on κ ∈ (0, κ_max], and the numerical spectral abscissa
    should stay strictly negative — the Theorem 6(a) verdict."""
    r = bifurcations_in_no_hopf_wedge(
        sigma_q=0.10,
        gamma=0.1,
        kappa_max=100.0,
        n_kappa_samples=50,
        **_CANONICAL,
    )
    assert r.is_in_wedge is True
    assert r.is_globally_stable is True
    assert r.kappa_sn is None, f"expected no positive SN root, got κ_SN={r.kappa_sn}"
    assert r.kappa_c2_zero is None
    assert r.kappa_H_zero is None
    assert r.spectral_abscissa_max < 0.0, (
        f"expected max Re λ < 0 over κ-scan, got {r.spectral_abscissa_max}"
    )


def test_scan_no_hopf_wedge_bifurcations_verdict_theorem_6a() -> None:
    """The 11×11 canonical scan returns the Theorem 6(a) verdict:
    every wedge cell is globally stable, zero positive saddle-nodes, and the
    maximum spectral abscissa over the wedge is strictly negative."""
    sigma_q_grid = np.linspace(0.02, 0.40, 11)
    gamma_grid = np.linspace(0.05, 5.0, 11)
    result = scan_no_hopf_wedge_bifurcations(
        sigma_q_grid=sigma_q_grid,
        gamma_grid=gamma_grid,
        kappa_max=100.0,
        n_kappa_samples=30,
        **_CANONICAL,
    )
    assert isinstance(result, NoHopfWedgeScanResult)
    assert result.n_wedge_cells > 0, "wedge should be non-empty at canonical params"
    assert result.n_globally_stable_cells == result.n_wedge_cells, (
        "Theorem 6(a): every wedge cell should be globally stable"
    )
    assert result.n_positive_saddle_node_cells == 0, (
        "the §3.5 BT analysis already gives κ_SN < 0 — no positive SN should appear"
    )
    assert result.wedge_max_spectral_abscissa < 0.0, (
        f"max Re λ over wedge × κ-scan should be < 0, got {result.wedge_max_spectral_abscissa}"
    )
    # Grids carry through consistently.
    assert result.sigma_q_grid.shape == (11,)
    assert result.gamma_grid.shape == (11,)
    assert result.in_wedge_grid.shape == (11, 11)
    assert result.globally_stable_grid.shape == (11, 11)
    assert result.kappa_sn_grid.shape == (11, 11)
    assert result.spectral_abscissa_grid.shape == (11, 11)


def test_scan_no_hopf_wedge_bifurcations_smoke_quick_grid() -> None:
    """Smoke test: 5×5 quick grid + κ_max=10 returns a consistent dataclass
    without raising. This is what the CI runner uses in --quick mode."""
    result = scan_no_hopf_wedge_bifurcations(
        sigma_q_grid=np.linspace(0.05, 0.30, 5),
        gamma_grid=np.linspace(0.1, 2.0, 5),
        kappa_max=10.0,
        n_kappa_samples=10,
        **_CANONICAL,
    )
    assert isinstance(result, NoHopfWedgeScanResult)
    assert 0 <= result.n_wedge_cells <= 25
    assert result.n_globally_stable_cells <= result.n_wedge_cells
    assert result.kappa_max_scanned == 10.0
