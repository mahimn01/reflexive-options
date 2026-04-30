"""Tests for the linearization-based coupling stability check.

Covers `simulator/coupling.py`. The function under test takes a Jacobian and
returns a `CouplingStability` snapshot — stability sign, oscillatory flag, and
the raw eigenvalue vector for diagnostic plots.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from reflexive_options.simulator.coupling import (
    CouplingStability,
    check_coupling_stable,
)


def test_stable_real_negative_eigenvalues() -> None:
    """Diagonal negative-real Jacobian is stable, non-oscillatory."""
    jac = np.diag([-1.0, -2.0])
    out = check_coupling_stable(jac)
    assert isinstance(out, CouplingStability)
    assert out.is_stable is True
    assert out.is_oscillatory is False
    assert out.largest_real_part == pytest.approx(-1.0)
    assert out.eigenvalues.shape == (2,)
    np.testing.assert_allclose(np.sort(out.eigenvalues.real), [-2.0, -1.0])


def test_unstable_positive_eigenvalue() -> None:
    """Any positive-real eigenvalue flips is_stable to False."""
    jac = np.diag([1.5, -1.0])
    out = check_coupling_stable(jac)
    assert out.is_stable is False
    assert out.largest_real_part == pytest.approx(1.5)
    assert out.is_oscillatory is False


def test_marginal_at_zero_is_unstable() -> None:
    """Tolerance is strict: max(Re λ) == 0 fails the < tolerance check."""
    jac = np.array([[0.0, 0.0], [0.0, -1.0]])
    out = check_coupling_stable(jac)
    assert out.largest_real_part == pytest.approx(0.0)
    assert out.is_stable is False


def test_oscillatory_complex_eigenvalues() -> None:
    """A rotation-style Jacobian has pure-imaginary eigenvalues — oscillatory, marginally unstable."""
    jac = np.array([[0.0, -1.0], [1.0, 0.0]])
    out = check_coupling_stable(jac)
    assert out.is_oscillatory is True
    # Pure imaginary ±i ⇒ Re=0, fails < tolerance.
    assert out.largest_real_part == pytest.approx(0.0, abs=1e-12)
    assert out.is_stable is False
    np.testing.assert_allclose(np.sort(out.eigenvalues.imag), [-1.0, 1.0], atol=1e-12)


def test_stable_spiral_negative_real_with_imag() -> None:
    """Damped spiral: Re(λ) < 0 AND Im(λ) ≠ 0 ⇒ stable AND oscillatory."""
    jac = np.array([[-0.5, -1.0], [1.0, -0.5]])
    out = check_coupling_stable(jac)
    assert out.is_stable is True
    assert out.is_oscillatory is True
    assert out.largest_real_part == pytest.approx(-0.5)


def test_custom_tolerance_loosened() -> None:
    """A loosened (less negative) tolerance accepts borderline-stable systems."""
    jac = np.diag([-1e-10, -1.0])
    # Default tol -1e-8 ⇒ -1e-10 is NOT < -1e-8, so unstable.
    assert check_coupling_stable(jac).is_stable is False
    # Loosened tol = 0.0 ⇒ -1e-10 < 0 is True, so stable.
    assert check_coupling_stable(jac, tolerance=0.0).is_stable is True


def test_three_by_three_jacobian_supported() -> None:
    """Function accepts n×n square inputs, not only 2×2."""
    jac = np.diag([-3.0, -1.0, -2.0])
    out = check_coupling_stable(jac)
    assert out.eigenvalues.shape == (3,)
    assert out.is_stable is True
    assert out.largest_real_part == pytest.approx(-1.0)


def test_rejects_non_square_jacobian() -> None:
    with pytest.raises(ValueError, match="square 2D"):
        check_coupling_stable(np.zeros((2, 3)))


def test_rejects_non_2d_jacobian() -> None:
    with pytest.raises(ValueError, match="square 2D"):
        check_coupling_stable(np.zeros((4,)))


def test_dataclass_is_frozen() -> None:
    """CouplingStability is frozen — protects callers caching the result."""
    out = check_coupling_stable(np.diag([-1.0, -1.0]))
    with pytest.raises(FrozenInstanceError):
        out.is_stable = False  # type: ignore[misc]
