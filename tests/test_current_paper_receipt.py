"""Selective CI recomputation of fields in the current paper's numerical receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflexive_options.experiments.centered_hopf_validation import (
    CenteredHopfValidationConfig,
    simulate,
)
from reflexive_options.experiments.centered_model_robustness import (
    RobustnessConfig,
    _independent_lyapunov_check,
    _lyapunov_classification_map,
    _mixture_metrics,
)
from reflexive_options.theory.centered_model import (
    canonical_centered_configuration,
    gaussian_book_hopf_point,
)

RECEIPT = Path(__file__).parent / "repro" / "centered_model_v0.4.1.json"


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_current_paper_receipt_locks_canonical_hopf_point() -> None:
    receipt = _receipt()
    canonical = receipt["canonical"]
    assert isinstance(canonical, dict)
    model, book = canonical_centered_configuration()
    point = gaussian_book_hopf_point(model, book)
    assert point.kappa == pytest.approx(canonical["kappa_star"], rel=1e-12)
    assert point.omega == pytest.approx(canonical["omega_star"], rel=1e-12)
    assert point.period_years == pytest.approx(canonical["period_years"], rel=1e-12)
    assert point.first_lyapunov == pytest.approx(canonical["first_lyapunov"], rel=1e-12)
    assert point.transversality == pytest.approx(canonical["transversality"], rel=1e-12)


def test_current_paper_receipt_recomputes_grid_and_mixture_claims() -> None:
    receipt = _receipt()
    robustness = receipt["robustness"]
    mixtures_expected = receipt["mixtures"]
    assert isinstance(robustness, dict)
    assert isinstance(mixtures_expected, dict)

    _, _, _, counts = _lyapunov_classification_map(RobustnessConfig())
    assert counts == robustness["classification_counts"]

    mixtures = _mixture_metrics()
    for name, expected in mixtures_expected.items():
        assert isinstance(expected, dict)
        actual = mixtures[name]
        assert actual["classification"] == expected["classification"]
        if "kappa_star" in expected:
            assert actual["kappa_star"] == pytest.approx(expected["kappa_star"], rel=1e-11)
            assert actual["first_lyapunov"] == pytest.approx(expected["first_lyapunov"], rel=1e-10)
            assert actual["G_x"] == pytest.approx(expected["G_x"], rel=1e-11)
            assert actual["G_v"] == pytest.approx(expected["G_v"], rel=1e-11)


def test_current_paper_receipt_recomputes_finite_difference_lyapunov_check() -> None:
    receipt = _receipt()
    robustness = receipt["robustness"]
    assert isinstance(robustness, dict)
    check = _independent_lyapunov_check()
    assert check["finite_difference"] == pytest.approx(
        robustness["finite_difference_first_lyapunov"], rel=2e-8
    )
    assert check["absolute_difference"] == pytest.approx(
        robustness["first_lyapunov_absolute_difference"], rel=2e-8
    )
    assert check["relative_difference"] == pytest.approx(
        robustness["first_lyapunov_relative_difference"], rel=2e-8
    )


def test_current_paper_receipt_recomputes_displayed_canonical_orbit() -> None:
    receipt = _receipt()
    canonical = receipt["canonical"]
    assert isinstance(canonical, dict)
    metrics, _ = simulate(CenteredHopfValidationConfig())
    assert metrics["x_min"] == pytest.approx(canonical["x_min_at_1.02"], abs=3e-12)
    assert metrics["x_max"] == pytest.approx(canonical["x_max_at_1.02"], abs=3e-12)
    assert metrics["variance_min"] == pytest.approx(canonical["variance_min_at_1.02"], abs=3e-12)
    assert metrics["variance_max"] == pytest.approx(canonical["variance_max_at_1.02"], abs=3e-12)
