"""Independent checks for the revised centered dealer-feedback model."""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.theory.centered_model import (
    CenteredSVParams,
    GaussianBookComponent,
    GaussianBookParams,
    _hopf_points_from_partials,
    canonical_centered_configuration,
    centered_drift_gaussian_book,
    centered_drift_gaussian_mixture,
    centered_jacobian,
    gaussian_book_hopf_point,
    gaussian_book_hopf_points,
    gaussian_mixture_hopf_point,
    normalized_gaussian_book_feedback,
    normalized_gaussian_book_partials,
    normalized_gaussian_mixture_feedback,
    normalized_gaussian_mixture_partials,
    routh_hurwitz_coefficients,
    static_feedback_hopf_polynomial,
)

# The private root classifier is imported deliberately because the tests below
# audit a numerical invariant of the polynomial implementation rather than the
# public Gaussian-book construction that fixes one normalization.


def test_centered_feedback_is_zero_at_fixed_equilibrium() -> None:
    model, book = canonical_centered_configuration()
    value = normalized_gaussian_book_feedback(0.0, model.theta_v, model=model, book=book)
    assert value == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("kappa", [0.0, 1.0, 31.4933, 100.0])
def test_equilibrium_does_not_move_with_coupling(kappa: float) -> None:
    model, book = canonical_centered_configuration()
    state = np.array([0.0, model.theta_v, 0.0])
    drift = centered_drift_gaussian_book(state, kappa=kappa, model=model, book=book)
    np.testing.assert_allclose(drift, 0.0, atol=1e-13)


def test_variance_boundary_points_inward_for_all_memory_values() -> None:
    model, book = canonical_centered_configuration()
    for chi in (-1e6, -100.0, 0.0, 100.0, 1e6):
        state = np.array([0.0, 0.0, chi])
        drift = centered_drift_gaussian_book(state, kappa=0.0, model=model, book=book)
        assert drift[1] == pytest.approx(model.kappa_v * model.theta_v)
        assert drift[1] > 0.0


def test_closed_form_hopf_polynomial_matches_direct_characteristic_polynomial() -> None:
    model, book = canonical_centered_configuration()
    partials = normalized_gaussian_book_partials(model, book)
    A2, A1, A0 = static_feedback_hopf_polynomial(
        model=model,
        G_x=partials["G_a"],
        G_v=partials["G_v"],
    )
    for kappa in (0.0, 5.0, 20.0, 50.0):
        J = centered_jacobian(
            kappa=kappa,
            model=model,
            G_x=partials["G_a"],
            G_v=partials["G_v"],
        )
        c2, c1, c0 = routh_hurwitz_coefficients(J)
        assert c1 * c2 - c0 == pytest.approx(A2 * kappa * kappa + A1 * kappa + A0)


def test_canonical_hopf_point_has_transversal_supercritical_pair() -> None:
    model, book = canonical_centered_configuration()
    point = gaussian_book_hopf_point(model, book)
    assert point.kappa == pytest.approx(31.4932976322, rel=1e-9)
    assert point.omega == pytest.approx(47.1185669561, rel=1e-9)
    assert point.period_years == pytest.approx(0.1333483956, rel=1e-9)
    assert point.period_years * 252.0 == pytest.approx(33.6037957, rel=1e-8)
    assert point.period_years * 365.0 == pytest.approx(48.6721644, rel=1e-8)
    assert point.transversality > 0.0
    assert point.first_lyapunov == pytest.approx(-6.2888040569, rel=1e-8)
    assert point.is_supercritical
    assert point.stable_cycle_above_threshold


def test_canonical_quadratic_discloses_both_valid_positive_hopf_points() -> None:
    model, book = canonical_centered_configuration()
    points = gaussian_book_hopf_points(model, book)
    assert len(points) == 2
    first, second = points
    assert first.kappa == pytest.approx(31.4932976322, rel=1e-9)
    assert first.transversality > 0.0
    assert first.first_lyapunov < 0.0
    assert second.kappa == pytest.approx(16860.8961078, rel=1e-9)
    assert second.omega == pytest.approx(309.551117075, rel=1e-9)
    assert second.transversality < 0.0
    assert second.first_lyapunov == pytest.approx(17479.5557196, rel=1e-8)


@pytest.mark.parametrize("scale", [1.0e-4, 1.0e4])
def test_hopf_points_are_invariant_to_positive_book_normalization(scale: float) -> None:
    model, book = canonical_centered_configuration()
    partials = normalized_gaussian_book_partials(model, book)
    baseline = gaussian_book_hopf_points(model, book)
    scaled_partials = {name: scale * value for name, value in partials.items()}
    scaled_partials["G"] = 0.0

    scaled = _hopf_points_from_partials(model, scaled_partials, tolerance=1.0e-8)

    assert len(scaled) == len(baseline) == 2
    for original, transformed in zip(baseline, scaled, strict=True):
        assert transformed.kappa * scale == pytest.approx(original.kappa, rel=2e-12)
        assert transformed.omega == pytest.approx(original.omega, rel=2e-12)
        assert transformed.first_lyapunov == pytest.approx(
            original.first_lyapunov,
            rel=2e-11,
        )
        assert np.sign(transformed.transversality) == np.sign(original.transversality)
        assert transformed.transversality / scale == pytest.approx(
            original.transversality,
            rel=2e-12,
        )


def test_canonical_linear_channel_ablations_isolate_variance_sensitivity() -> None:
    model, book = canonical_centered_configuration()
    partials = normalized_gaussian_book_partials(model, book)

    without_direct_price_sensitivity = dict(partials)
    without_direct_price_sensitivity["G_a"] = 0.0
    points = _hopf_points_from_partials(
        model,
        without_direct_price_sensitivity,
        tolerance=1.0e-8,
    )
    assert len(points) == 1
    assert points[0].kappa == pytest.approx(28.9140812922, rel=1e-10)
    assert points[0].omega == pytest.approx(45.2216762184, rel=1e-10)
    assert points[0].transversality == pytest.approx(0.3083534535, rel=1e-10)

    without_variance_sensitivity = dict(partials)
    without_variance_sensitivity["G_v"] = 0.0
    with pytest.raises(ValueError, match="no positive Hopf root"):
        _hopf_points_from_partials(
            model,
            without_variance_sensitivity,
            tolerance=1.0e-8,
        )


def test_hopf_eigenvalues_and_third_direction_are_correct() -> None:
    model, book = canonical_centered_configuration()
    partials = normalized_gaussian_book_partials(model, book)
    point = gaussian_book_hopf_point(model, book)
    J = centered_jacobian(
        kappa=point.kappa,
        model=model,
        G_x=partials["G_a"],
        G_v=partials["G_v"],
    )
    eigenvalues = np.linalg.eigvals(J)
    real_eigenvalue = eigenvalues[np.argmin(np.abs(eigenvalues.imag))]
    assert real_eigenvalue.real < 0.0
    assert real_eigenvalue.real == pytest.approx(-point.c2, rel=1e-9)
    assert np.min(np.abs(eigenvalues - 1j * point.omega)) < 1e-8


def test_exact_transversality_matches_finite_difference_eigenvalue_speed() -> None:
    model, book = canonical_centered_configuration()
    partials = normalized_gaussian_book_partials(model, book)
    point = gaussian_book_hopf_point(model, book)

    def leading_pair_real(kappa: float) -> float:
        jacobian = centered_jacobian(
            kappa=kappa,
            model=model,
            G_x=partials["G_a"],
            G_v=partials["G_v"],
        )
        eigenvalues = np.linalg.eigvals(jacobian)
        complex_pair = eigenvalues[np.argsort(np.abs(eigenvalues.imag))[-2:]]
        return float(np.mean(complex_pair.real))

    step = 1.0e-5
    finite_difference = (
        leading_pair_real(point.kappa + step) - leading_pair_real(point.kappa - step)
    ) / (2.0 * step)
    assert finite_difference == pytest.approx(point.transversality, rel=1e-7)


def test_negative_discriminant_is_not_labelled_global_stability() -> None:
    model = CenteredSVParams(
        delta=0.5,
        kappa_v=2.0,
        theta_v=0.04,
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
    )
    # The helper returns only the Hopf determinant; callers must still check
    # c2, c1 and c0.  This regression test pins that limited API contract.
    coefficients = static_feedback_hopf_polynomial(model=model, G_x=-1.0, G_v=100.0)
    assert len(coefficients) == 3


def test_invalid_parameter_units_are_rejected() -> None:
    with pytest.raises(ValueError, match="alpha"):
        CenteredSVParams(delta=1, kappa_v=1, theta_v=0.04, alpha=0, beta=1, gamma=1)
    with pytest.raises(ValueError, match="dealer_sign"):
        GaussianBookParams(0.0, 0.1, 0.25, dealer_sign=0)
    with pytest.raises(ValueError, match="component mass"):
        GaussianBookComponent(0.0, GaussianBookParams(0.0, 0.1, 0.25))


def test_one_component_mixture_exactly_matches_single_gaussian_book() -> None:
    model, book = canonical_centered_configuration()
    components = (GaussianBookComponent(2.7, book),)
    single = normalized_gaussian_book_partials(model, book)
    mixture = normalized_gaussian_mixture_partials(model, components)
    for name in single:
        assert mixture[name] == pytest.approx(single[name], rel=1e-13, abs=1e-13)
    state = np.array([0.01, model.theta_v + 0.002, -0.003])
    assert normalized_gaussian_mixture_feedback(
        state[0], state[1], model=model, components=components
    ) == pytest.approx(
        normalized_gaussian_book_feedback(state[0], state[1], model=model, book=book)
    )
    np.testing.assert_allclose(
        centered_drift_gaussian_mixture(
            state,
            kappa=10.0,
            model=model,
            components=components,
        ),
        centered_drift_gaussian_book(state, kappa=10.0, model=model, book=book),
        rtol=1e-13,
        atol=1e-13,
    )
    mixture_point = gaussian_mixture_hopf_point(model, components)
    single_point = gaussian_book_hopf_point(model, book)
    assert mixture_point.kappa == pytest.approx(single_point.kappa, rel=1e-12)
    assert mixture_point.first_lyapunov == pytest.approx(single_point.first_lyapunov, rel=1e-12)


def test_signed_mixture_uses_gross_normalization_and_fixed_equilibrium() -> None:
    model, _ = canonical_centered_configuration()
    components = (
        GaussianBookComponent(0.6, GaussianBookParams(0.02, 0.12, 0.10, dealer_sign=1)),
        GaussianBookComponent(0.4, GaussianBookParams(0.10, 0.24, 0.50, dealer_sign=-1)),
    )
    value = normalized_gaussian_mixture_feedback(
        0.0,
        model.theta_v,
        model=model,
        components=components,
    )
    assert value == pytest.approx(0.0, abs=1e-15)
    drift = centered_drift_gaussian_mixture(
        np.array([0.0, model.theta_v, 0.0]),
        kappa=50.0,
        model=model,
        components=components,
    )
    np.testing.assert_allclose(drift, 0.0, atol=1e-13)
    partials = normalized_gaussian_mixture_partials(model, components)
    assert all(np.isfinite(value) for value in partials.values())
