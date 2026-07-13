"""Centered, positive-variance dealer-feedback model used by the revised paper.

The original manuscript varied the spot drift together with the feedback
coupling in order to pin the equilibrium.  That construction does not identify
a one-parameter bifurcation in the coupling.  This module instead works with a
detrended log-price deviation ``x`` and a *centered* dealer-book functional
``g`` satisfying ``g(0, theta_v, 0) = 0``.  The equilibrium

    (x, v, chi) = (0, theta_v, 0)

is therefore fixed while ``kappa`` varies and every other structural parameter
is held constant.

The physical-measure local model is

    dx   = [-delta*x - 0.5*(v-theta_v) + kappa*g(x, v, chi)] dt
           + sqrt(v) dW_S,
    dv   = [kappa_v*(theta_v-v) + gamma*v*chi] dt
           + xi*sqrt(v) dW_v,
    dchi = alpha*(beta*x-chi) dt.

Multiplying the leverage-memory term by ``v`` keeps the drift at the variance
boundary equal to ``kappa_v*theta_v > 0``.  Thus the feedback is compatible
with the non-negative state space of the square-root variance process; no
global non-explosion or stochastic-invariance theorem is claimed here.
The Gaussian dealer-book specialization below is a *signed dealer-position*
model.  Public open interest does not identify that sign and is treated only as
a proxy in the empirical protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reflexive_options.theory.bifurcation import (
    G_lognormal_oi,
    G_lognormal_oi_partials,
    compute_lyapunov_coefficient,
)


@dataclass(frozen=True)
class CenteredSVParams:
    """Structural parameters, all expressed in years.

    ``delta``, ``kappa_v``, ``alpha``, ``gamma``, and the coupling ``kappa``
    have units yr^-1.  ``beta`` and ``chi`` are dimensionless.  ``theta_v`` is
    annualized instantaneous variance.
    """

    delta: float
    kappa_v: float
    theta_v: float
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        positive = {
            "delta": self.delta,
            "kappa_v": self.kappa_v,
            "theta_v": self.theta_v,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")


@dataclass(frozen=True)
class GaussianBookParams:
    """Gaussian signed-dealer-book approximation in fixed log-moneyness.

    ``mean_moneyness`` and ``sigma_moneyness`` parameterize the density of
    signed dealer positions over k = log(K/F_star), where the local reference
    forward is normalized to one.  ``dealer_sign`` is +1 or -1 and is a model
    input, not something inferable from public open interest.
    """

    mean_moneyness: float
    sigma_moneyness: float
    effective_maturity: float
    dealer_sign: int = 1
    rate: float = 0.0
    dividend: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_moneyness <= 0.0:
            raise ValueError("sigma_moneyness must be positive")
        if self.effective_maturity <= 0.0:
            raise ValueError("effective_maturity must be positive")
        if self.dealer_sign not in {-1, 1}:
            raise ValueError("dealer_sign must be -1 or +1")


@dataclass(frozen=True)
class GaussianBookComponent:
    """One mass-weighted component of a signed Gaussian dealer book.

    ``mass`` is non-negative.  The orientation of the component is carried by
    ``book.dealer_sign``.  Keeping mass and sign separate makes the
    normalization use gross rather than net equilibrium gamma, so an
    offsetting long/short book cannot create a spuriously large feedback by
    dividing through a nearly zero net position.
    """

    mass: float
    book: GaussianBookParams

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError(f"component mass must be finite and positive, got {self.mass}")


@dataclass(frozen=True)
class HopfPoint:
    """A Routh-Hurwitz-validated Hopf point."""

    kappa: float
    omega: float
    transversality: float
    c2: float
    c1: float
    c0: float
    first_lyapunov: float

    @property
    def period_years(self) -> float:
        return float(2.0 * np.pi / self.omega)

    @property
    def is_supercritical(self) -> bool:
        return self.first_lyapunov < 0.0

    @property
    def stable_cycle_above_threshold(self) -> bool:
        return self.first_lyapunov < 0.0 and self.transversality > 0.0


def normalized_gaussian_book_partials(
    model: CenteredSVParams,
    book: GaussianBookParams,
) -> dict[str, float]:
    """Analytic derivatives of the centered, dimensionless book functional.

    The raw Black-Scholes-gamma integral is divided by its positive magnitude
    at the fixed equilibrium.  The constant is then subtracted, so the returned
    functional has value zero at equilibrium while all derivatives are simply
    the normalized raw derivatives.
    """

    raw = G_lognormal_oi_partials(
        a_star=0.0,
        v_star=model.theta_v,
        mu_q=book.mean_moneyness,
        sigma_q=book.sigma_moneyness,
        T_eff=book.effective_maturity,
        coupling_units=1.0,
        rate=book.rate,
        dividend=book.dividend,
    )
    scale = float(raw["G"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid equilibrium gamma scale {scale}")

    sign = float(book.dealer_sign)
    out = {name: sign * float(value) / scale for name, value in raw.items()}
    out["G"] = 0.0
    return out


def normalized_gaussian_book_feedback(
    x: float,
    variance: float,
    *,
    model: CenteredSVParams,
    book: GaussianBookParams,
) -> float:
    """Centered dimensionless dealer-book pressure g(x, v)."""

    if variance < 0.0:
        raise ValueError(f"variance must be non-negative, got {variance}")
    kwargs = {
        "mu_q": book.mean_moneyness,
        "sigma_q": book.sigma_moneyness,
        "T_eff": book.effective_maturity,
        "coupling_units": 1.0,
        "rate": book.rate,
        "dividend": book.dividend,
    }
    raw_0 = G_lognormal_oi(0.0, model.theta_v, **kwargs)
    raw = G_lognormal_oi(x, variance, **kwargs)
    return float(book.dealer_sign * (raw - raw_0) / raw_0)


def normalized_gaussian_mixture_partials(
    model: CenteredSVParams,
    components: tuple[GaussianBookComponent, ...],
) -> dict[str, float]:
    """Derivatives through order three for a centered signed Gaussian mixture.

    The numerator is the signed net book.  The denominator is gross
    equilibrium gamma, ``sum_j mass_j * G_j(0, theta_v)``, which is positive
    and invariant to cancellations between component orientations.
    """

    if not components:
        raise ValueError("a Gaussian mixture requires at least one component")
    raw_components: list[tuple[GaussianBookComponent, dict[str, float]]] = []
    gross_scale = 0.0
    for component in components:
        book = component.book
        raw = G_lognormal_oi_partials(
            a_star=0.0,
            v_star=model.theta_v,
            mu_q=book.mean_moneyness,
            sigma_q=book.sigma_moneyness,
            T_eff=book.effective_maturity,
            coupling_units=1.0,
            rate=book.rate,
            dividend=book.dividend,
        )
        gross_scale += component.mass * float(raw["G"])
        raw_components.append((component, raw))
    if not np.isfinite(gross_scale) or gross_scale <= 0.0:
        raise ValueError(f"invalid gross equilibrium gamma scale {gross_scale}")

    output: dict[str, float] = {}
    for name in raw_components[0][1]:
        signed_value = sum(
            component.mass * component.book.dealer_sign * float(raw[name])
            for component, raw in raw_components
        )
        output[name] = float(signed_value / gross_scale)
    output["G"] = 0.0
    return output


def normalized_gaussian_mixture_feedback(
    x: float,
    variance: float,
    *,
    model: CenteredSVParams,
    components: tuple[GaussianBookComponent, ...],
) -> float:
    """Centered signed-book pressure for a gross-normalized Gaussian mixture."""

    if variance < 0.0:
        raise ValueError(f"variance must be non-negative, got {variance}")
    if not components:
        raise ValueError("a Gaussian mixture requires at least one component")
    signed_current = 0.0
    signed_equilibrium = 0.0
    gross_equilibrium = 0.0
    for component in components:
        book = component.book
        kwargs = {
            "mu_q": book.mean_moneyness,
            "sigma_q": book.sigma_moneyness,
            "T_eff": book.effective_maturity,
            "coupling_units": 1.0,
            "rate": book.rate,
            "dividend": book.dividend,
        }
        raw_equilibrium = G_lognormal_oi(0.0, model.theta_v, **kwargs)
        raw_current = G_lognormal_oi(x, variance, **kwargs)
        signed_current += component.mass * book.dealer_sign * raw_current
        signed_equilibrium += component.mass * book.dealer_sign * raw_equilibrium
        gross_equilibrium += component.mass * raw_equilibrium
    if not np.isfinite(gross_equilibrium) or gross_equilibrium <= 0.0:
        raise ValueError(f"invalid gross equilibrium gamma scale {gross_equilibrium}")
    return float((signed_current - signed_equilibrium) / gross_equilibrium)


def centered_drift_gaussian_book(
    state: NDArray[np.float64],
    *,
    kappa: float,
    model: CenteredSVParams,
    book: GaussianBookParams,
) -> NDArray[np.float64]:
    """Deterministic drift of the centered model.

    The function is intended for deterministic bifurcation validation.  It
    rejects negative variance instead of silently truncating it.
    """

    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    x, variance, chi = map(float, state)
    if variance < 0.0:
        raise ValueError(f"variance left the physical state space: {variance}")
    feedback = normalized_gaussian_book_feedback(x, variance, model=model, book=book)
    return np.array(
        [
            -model.delta * x - 0.5 * (variance - model.theta_v) + kappa * feedback,
            model.kappa_v * (model.theta_v - variance) + model.gamma * variance * chi,
            model.alpha * (model.beta * x - chi),
        ],
        dtype=np.float64,
    )


def centered_drift_gaussian_mixture(
    state: NDArray[np.float64],
    *,
    kappa: float,
    model: CenteredSVParams,
    components: tuple[GaussianBookComponent, ...],
) -> NDArray[np.float64]:
    """Deterministic drift under a gross-normalized signed Gaussian mixture."""

    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    x, variance, chi = map(float, state)
    if variance < 0.0:
        raise ValueError(f"variance left the physical state space: {variance}")
    feedback = normalized_gaussian_mixture_feedback(
        x,
        variance,
        model=model,
        components=components,
    )
    return np.array(
        [
            -model.delta * x - 0.5 * (variance - model.theta_v) + kappa * feedback,
            model.kappa_v * (model.theta_v - variance) + model.gamma * variance * chi,
            model.alpha * (model.beta * x - chi),
        ],
        dtype=np.float64,
    )


def centered_jacobian(
    *,
    kappa: float,
    model: CenteredSVParams,
    G_x: float,
    G_v: float,
    G_chi: float = 0.0,
) -> NDArray[np.float64]:
    """Jacobian at the fixed equilibrium (0, theta_v, 0)."""

    a = -model.delta + kappa * G_x
    b = -0.5 + kappa * G_v
    d = kappa * G_chi
    c = model.gamma * model.theta_v
    m = model.alpha * model.beta
    return np.array(
        [
            [a, b, d],
            [0.0, -model.kappa_v, c],
            [m, 0.0, -model.alpha],
        ],
        dtype=np.float64,
    )


def routh_hurwitz_coefficients(jacobian: NDArray[np.float64]) -> tuple[float, float, float]:
    """Return (c2, c1, c0) for det(lambda I - J)."""

    if jacobian.shape != (3, 3):
        raise ValueError(f"expected a 3x3 Jacobian, got {jacobian.shape}")
    polynomial = np.poly(jacobian)
    return tuple(float(value.real) for value in polynomial[1:])  # type: ignore[return-value]


def static_feedback_hopf_polynomial(
    *,
    model: CenteredSVParams,
    G_x: float,
    G_v: float,
) -> tuple[float, float, float]:
    """Coefficients of H(kappa)=c1*c2-c0 when G_chi=0.

    Writing A=alpha+kappa_v, M=alpha*kappa_v and
    L=(gamma*theta_v)*(alpha*beta), the coefficients are

        A2 = G_x^2 A,
        A1 = G_v L - G_x A(A+2 delta),
        A0 = A(M + delta A + delta^2) - L/2.

    A negative discriminant proves only that no local Hopf point exists; it
    does *not* imply global or even linear stability for all kappa.
    """

    A = model.alpha + model.kappa_v
    M = model.alpha * model.kappa_v
    L = model.gamma * model.theta_v * model.alpha * model.beta
    return (
        G_x * G_x * A,
        G_v * L - G_x * A * (A + 2.0 * model.delta),
        A * (M + model.delta * A + model.delta * model.delta) - 0.5 * L,
    )


def _analytic_tensors(
    *,
    kappa: float,
    model: CenteredSVParams,
    partials: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Taylor tensors at the fixed equilibrium for the Gaussian-book model."""

    B = np.zeros((3, 3, 3), dtype=np.float64)
    B[0, 0, 0] = kappa * partials["G_aa"]
    B[0, 0, 1] = B[0, 1, 0] = kappa * partials["G_av"]
    B[0, 1, 1] = kappa * partials["G_vv"]
    # d^2[gamma*v*chi]/dv dchi = gamma.
    B[1, 1, 2] = B[1, 2, 1] = model.gamma

    C = np.zeros((3, 3, 3, 3), dtype=np.float64)
    C[0, 0, 0, 0] = kappa * partials["G_aaa"]
    for indices in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        C[(0, *indices)] = kappa * partials["G_aav"]
    for indices in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        C[(0, *indices)] = kappa * partials["G_avv"]
    C[0, 1, 1, 1] = kappa * partials["G_vvv"]
    return B, C


def _hopf_points_from_partials(
    model: CenteredSVParams,
    partials: dict[str, float],
    *,
    tolerance: float,
) -> tuple[HopfPoint, ...]:
    """Return every positive Routh--Hurwitz-valid root in ascending order.

    A quadratic Hopf determinant can have two positive roots.  The first is
    the initial loss-of-stability threshold when the baseline is stable, but a
    second root can restore local stability at a much larger coupling.  Keeping
    all valid roots prevents the economically convenient first threshold from
    being mistaken for the complete algebraic result.
    """

    G_x = partials["G_a"]
    G_v = partials["G_v"]
    A2, A1, A0 = static_feedback_hopf_polynomial(model=model, G_x=G_x, G_v=G_v)

    # A positive rescaling g -> scale*g must be absorbed exactly by
    # kappa -> kappa/scale.  Absolute coefficient cutoffs violate that
    # invariance because A2 and A1 then scale as scale**2 and scale,
    # respectively.  Here A2 = G_x**2*A with A > 0, so the polynomial is
    # genuinely linear exactly when G_x (and hence A2) is exactly zero.
    if A2 == 0.0:
        if A1 == 0.0:
            raise ValueError("Hopf polynomial is degenerate")
        candidates = [-A0 / A1]
    else:
        discriminant = A1 * A1 - 4.0 * A2 * A0
        discriminant_scale = max(A1 * A1, abs(4.0 * A2 * A0), np.finfo(float).tiny)
        if discriminant < -tolerance * discriminant_scale:
            raise ValueError("no real root of the Hopf determinant")
        sqrt_discriminant = float(np.sqrt(max(discriminant, 0.0)))
        # Use the cancellation-resistant quadratic formula.  The canonical
        # witness has roots separated by more than two orders of magnitude,
        # making the stable form preferable for a reproducibility routine.
        q = -0.5 * (A1 + np.copysign(sqrt_discriminant, A1))
        q_scale = max(abs(A1), sqrt_discriminant, np.finfo(float).tiny)
        if abs(q) <= tolerance * q_scale:
            candidates = [
                (-A1 - sqrt_discriminant) / (2.0 * A2),
                (-A1 + sqrt_discriminant) / (2.0 * A2),
            ]
        else:
            candidates = [q / A2, A0 / q]

    valid_points: list[HopfPoint] = []
    positive_candidates: list[float] = []
    for candidate in sorted(value for value in candidates if value > 0.0):
        if positive_candidates and np.isclose(
            candidate,
            positive_candidates[-1],
            rtol=max(tolerance, 10.0 * np.finfo(float).eps),
            atol=0.0,
        ):
            continue
        positive_candidates.append(float(candidate))

    for kappa in positive_candidates:
        jacobian = centered_jacobian(kappa=kappa, model=model, G_x=G_x, G_v=G_v)
        c2, c1, c0 = routh_hurwitz_coefficients(jacobian)
        H = c1 * c2 - c0
        scale = max(1.0, abs(c0), abs(c1 * c2))
        if c2 <= tolerance or c1 <= tolerance or c0 <= tolerance:
            continue
        if abs(H) > 1e-7 * scale:
            continue
        H_prime = 2.0 * A2 * kappa + A1
        transversality = -H_prime / (2.0 * (c1 + c2 * c2))
        derivative_scale = max(abs(2.0 * A2 * kappa), abs(A1), np.finfo(float).tiny)
        if abs(H_prime) <= tolerance * derivative_scale:
            continue
        omega = float(np.sqrt(c1))
        eigenvalues = np.linalg.eigvals(jacobian)
        pair_distance = np.min(np.abs(eigenvalues - 1j * omega))
        if pair_distance > 1e-6 * max(1.0, omega):
            continue
        B, C = _analytic_tensors(kappa=kappa, model=model, partials=partials)
        ell1 = compute_lyapunov_coefficient(jacobian, B, C, omega=omega)
        valid_points.append(
            HopfPoint(
                kappa=float(kappa),
                omega=omega,
                transversality=float(transversality),
                c2=c2,
                c1=c1,
                c0=c0,
                first_lyapunov=float(ell1),
            )
        )

    if not valid_points:
        raise ValueError("no positive Hopf root satisfies the Routh-Hurwitz side conditions")
    return tuple(valid_points)


def gaussian_book_hopf_points(
    model: CenteredSVParams,
    book: GaussianBookParams,
    *,
    tolerance: float = 1e-8,
) -> tuple[HopfPoint, ...]:
    """Return all valid positive local Hopf points of one Gaussian book."""

    partials = normalized_gaussian_book_partials(model, book)
    return _hopf_points_from_partials(model, partials, tolerance=tolerance)


def gaussian_book_hopf_point(
    model: CenteredSVParams,
    book: GaussianBookParams,
    *,
    tolerance: float = 1e-8,
) -> HopfPoint:
    """Return the first positive, Routh-Hurwitz-valid single-book Hopf point.

    Both algebraic roots are enumerated.  A candidate is accepted only if
    c2, c1 and c0 are positive, the Routh determinant vanishes, the imaginary
    frequency is non-zero, and the eigenvalue crossing is transversal.
    """

    return gaussian_book_hopf_points(model, book, tolerance=tolerance)[0]


def gaussian_mixture_hopf_points(
    model: CenteredSVParams,
    components: tuple[GaussianBookComponent, ...],
    *,
    tolerance: float = 1e-8,
) -> tuple[HopfPoint, ...]:
    """Return all valid positive local Hopf points of a static mixture."""

    partials = normalized_gaussian_mixture_partials(model, components)
    return _hopf_points_from_partials(model, partials, tolerance=tolerance)


def gaussian_mixture_hopf_point(
    model: CenteredSVParams,
    components: tuple[GaussianBookComponent, ...],
    *,
    tolerance: float = 1e-8,
) -> HopfPoint:
    """Return the first valid local Hopf point of a static Gaussian mixture."""

    return gaussian_mixture_hopf_points(model, components, tolerance=tolerance)[0]


def canonical_centered_configuration() -> tuple[CenteredSVParams, GaussianBookParams]:
    """Transparent illustrative configuration used in the revised paper.

    The rates are chosen on daily-to-monthly scales and are not labelled an
    SPX calibration.  The purpose of the configuration is to validate the
    mechanism on the *actual nonlinear Gaussian-book drift* while variance
    remains strictly positive.
    """

    return (
        CenteredSVParams(
            delta=0.5,
            kappa_v=40.0,
            theta_v=0.0625,
            alpha=50.0,
            beta=2.0,
            gamma=500.0,
        ),
        GaussianBookParams(
            mean_moneyness=0.06,
            sigma_moneyness=0.20,
            effective_maturity=0.25,
            dealer_sign=1,
        ),
    )
