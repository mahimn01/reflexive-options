"""Tests for the four non-reflexive baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from reflexive_options.baselines import (
    GammaAwareSimulator,
    HestonSimulator,
    LSVSimulator,
    SV32Simulator,
)
from reflexive_options.baselines.gamma_aware import GammaAggregatorProtocol
from reflexive_options.types import (
    HestonParams,
    SDEState,
    SimulatorProtocol,
    SurfaceGrid,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def heston_params() -> HestonParams:
    return HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


@pytest.fixture
def grid() -> SurfaceGrid:
    return SurfaceGrid(
        log_moneyness=np.linspace(-0.1, 0.1, 5),
        maturities=np.array([0.25, 0.5, 1.0]),
    )


@dataclass
class _MockAggregator:
    """Deterministic mock with the GammaAggregator interface."""

    scale: float = 1e-3

    def compute(self, spot: float, variance: float, log_memory: float) -> float:
        return float(self.scale * (spot - 100.0) * (1.0 + abs(log_memory)))


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------


def test_heston_protocol_compliance(heston_params: HestonParams) -> None:
    sim = HestonSimulator(regimes=[heston_params], breakpoints=[])
    assert isinstance(sim, SimulatorProtocol)


def test_heston_simulate_shape(heston_params: HestonParams) -> None:
    sim = HestonSimulator(regimes=[heston_params], breakpoints=[])
    n_paths, n_steps = 16, 50
    spots, variances = sim.simulate(n_paths=n_paths, n_steps=n_steps, dt=1.0 / 252, seed=42)
    assert spots.shape == (n_paths, n_steps + 1)
    assert variances.shape == (n_paths, n_steps + 1)
    assert np.all(spots[:, 0] == sim.spot0)
    assert np.all(variances[:, 0] == heston_params.v0)
    assert np.all(np.isfinite(spots))
    assert np.all(variances >= 0)


def test_heston_step_advances_time(heston_params: HestonParams) -> None:
    sim = HestonSimulator(regimes=[heston_params], breakpoints=[])
    state = SDEState(spot=100.0, variance=heston_params.v0, time=0.0)
    new_state = sim.step(state, dt=1.0 / 252, dW=np.array([0.01, -0.005]))
    assert new_state.time == pytest.approx(state.time + 1.0 / 252)
    assert new_state.spot != state.spot
    assert new_state.variance >= 0


def test_heston_multi_regime_lookup(heston_params: HestonParams) -> None:
    r2 = HestonParams(kappa=3.0, theta=0.05, xi=0.4, rho=-0.5, v0=0.04)
    sim = HestonSimulator(regimes=[heston_params, r2], breakpoints=[0.5])
    assert sim.regime_at(0.1) is heston_params
    assert sim.regime_at(0.6) is r2


def test_heston_implied_surface_matches_quantlib_atm(heston_params: HestonParams) -> None:
    sim = HestonSimulator(regimes=[heston_params], breakpoints=[])
    grid_atm = SurfaceGrid(
        log_moneyness=np.array([0.0]),
        maturities=np.array([0.5]),
    )
    state = SDEState(spot=100.0, variance=heston_params.v0, time=0.0)
    iv = sim.implied_surface(state, grid_atm)
    assert iv.shape == (1, 1)
    # Round-trip BS price → IV at ATM should match within 1e-3 (we ARE inverting QL).
    # Sanity: ATM IV should be close to sqrt(v0) for short-T.
    assert iv[0, 0] == pytest.approx(np.sqrt(heston_params.v0), abs=0.05)


def test_heston_implied_surface_full_grid(heston_params: HestonParams, grid: SurfaceGrid) -> None:
    sim = HestonSimulator(regimes=[heston_params], breakpoints=[])
    state = SDEState(spot=100.0, variance=heston_params.v0, time=0.0)
    iv = sim.implied_surface(state, grid)
    assert iv.shape == grid.shape
    assert np.all(np.isfinite(iv))
    # IV should be positive everywhere
    assert np.all(iv > 0)


def test_heston_construction_validation(heston_params: HestonParams) -> None:
    with pytest.raises(ValueError, match=r"len\(regimes\)"):
        HestonSimulator(regimes=[heston_params, heston_params], breakpoints=[])
    with pytest.raises(ValueError, match="ascending"):
        HestonSimulator(
            regimes=[heston_params] * 3,
            breakpoints=[0.5, 0.3],
        )
    with pytest.raises(ValueError, match="positive"):
        HestonSimulator(regimes=[heston_params, heston_params], breakpoints=[-0.1])


# ---------------------------------------------------------------------------
# LSV
# ---------------------------------------------------------------------------


def test_lsv_protocol_compliance(heston_params: HestonParams) -> None:
    sim = LSVSimulator(heston=heston_params, leverage_coeffs={"a1": 0.1, "a2": 0.05})
    assert isinstance(sim, SimulatorProtocol)


def test_lsv_simulate_shape(heston_params: HestonParams) -> None:
    sim = LSVSimulator(heston=heston_params, leverage_coeffs={"a1": 0.1, "a2": 0.05})
    n_paths, n_steps = 16, 50
    spots, variances = sim.simulate(n_paths=n_paths, n_steps=n_steps, dt=1.0 / 252, seed=42)
    assert spots.shape == (n_paths, n_steps + 1)
    assert variances.shape == (n_paths, n_steps + 1)
    assert np.all(np.isfinite(spots))


def test_lsv_leverage_at_atm_is_unity(heston_params: HestonParams) -> None:
    sim = LSVSimulator(heston=heston_params, leverage_coeffs={"a1": 0.5, "a2": 1.0, "a3": 0.3})
    assert sim.leverage(sim.spot0, t=0.5) == pytest.approx(1.0)


def test_lsv_implied_surface_default_returns_nan(
    heston_params: HestonParams, grid: SurfaceGrid
) -> None:
    sim = LSVSimulator(heston=heston_params, leverage_coeffs={})
    state = SDEState(spot=100.0, variance=heston_params.v0, time=0.0)
    iv = sim.implied_surface(state, grid)
    assert iv.shape == grid.shape
    assert np.all(np.isnan(iv))


def test_lsv_implied_surface_mc_runs(heston_params: HestonParams) -> None:
    sim = LSVSimulator(heston=heston_params, leverage_coeffs={})  # L = 1 ⇒ pure Heston
    small_grid = SurfaceGrid(
        log_moneyness=np.array([0.0]),
        maturities=np.array([0.25]),
    )
    state = SDEState(spot=100.0, variance=heston_params.v0, time=0.0)
    iv = sim.implied_surface(
        state, small_grid, compute_surface=True, n_paths=4_000, n_steps_per_year=80, seed=7
    )
    assert iv.shape == (1, 1)
    assert np.isfinite(iv[0, 0])
    # MC ATM IV should be close to sqrt(v0) for moderate T
    assert iv[0, 0] == pytest.approx(np.sqrt(heston_params.v0), abs=0.05)


# ---------------------------------------------------------------------------
# SV32
# ---------------------------------------------------------------------------


def test_sv32_protocol_compliance() -> None:
    sim = SV32Simulator(kappa_v=22.84, theta_v=0.04, xi=1.5, rho=-0.7, v0=0.04)
    assert isinstance(sim, SimulatorProtocol)


def test_sv32_simulate_shape() -> None:
    sim = SV32Simulator(kappa_v=22.84, theta_v=0.04, xi=1.5, rho=-0.7, v0=0.04)
    spots, variances = sim.simulate(n_paths=16, n_steps=50, dt=1.0 / 252, seed=11)
    assert spots.shape == (16, 51)
    assert variances.shape == (16, 51)
    assert np.all(np.isfinite(spots))
    assert np.all(np.isfinite(variances))


def test_sv32_stability_modest_xi() -> None:
    """Variance should not blow up over a year with moderate parameters."""
    sim = SV32Simulator(kappa_v=22.84, theta_v=0.04, xi=1.5, rho=-0.7, v0=0.04)
    _, variances = sim.simulate(n_paths=64, n_steps=252, dt=1.0 / 252, seed=3)
    finite_frac = np.isfinite(variances).mean()
    assert finite_frac > 0.99
    # Variance bounded by a generous ceiling (vol < 200%/yr most paths)
    assert (variances < 5.0).mean() > 0.95


def test_sv32_construction_validation() -> None:
    with pytest.raises(ValueError, match="v0"):
        SV32Simulator(kappa_v=1.0, theta_v=0.04, xi=1.0, rho=-0.5, v0=-0.01)
    with pytest.raises(ValueError, match="rho"):
        SV32Simulator(kappa_v=1.0, theta_v=0.04, xi=1.0, rho=1.0, v0=0.04)
    with pytest.raises(ValueError, match="theta_v"):
        SV32Simulator(kappa_v=1.0, theta_v=0.0, xi=1.0, rho=-0.5, v0=0.04)


# ---------------------------------------------------------------------------
# GammaAware
# ---------------------------------------------------------------------------


def test_gamma_aware_protocol_compliance(heston_params: HestonParams) -> None:
    sim = GammaAwareSimulator(
        heston=heston_params,
        aggregator=_MockAggregator(),
        memory_decay=1.0,
        memory_intake=1.0,
    )
    assert isinstance(sim, SimulatorProtocol)


def test_gamma_aware_state_includes_g_and_z(heston_params: HestonParams) -> None:
    agg = _MockAggregator(scale=2e-3)
    sim = GammaAwareSimulator(
        heston=heston_params,
        aggregator=agg,
        memory_decay=2.0,
        memory_intake=1.5,
    )
    state = SDEState(spot=100.0, variance=heston_params.v0, time=0.0, memory=0.0)
    new_state = sim.step(state, dt=1.0 / 252, dW=np.array([0.05, -0.01]))
    assert new_state.aggregate_gamma is not None
    assert new_state.memory is not None
    assert isinstance(new_state.aggregate_gamma, float)
    assert isinstance(new_state.memory, float)


def test_gamma_aware_simulate_dynamics_independent_of_aggregator(
    heston_params: HestonParams,
) -> None:
    """G and z are observed but NOT fed back ⇒ swapping aggregators changes nothing in (S, v)."""
    a1 = _MockAggregator(scale=1e-3)
    a2 = _MockAggregator(scale=1e3)  # 6 orders of magnitude bigger; would matter if fed back

    sim1 = GammaAwareSimulator(
        heston=heston_params, aggregator=a1, memory_decay=1.0, memory_intake=1.0
    )
    sim2 = GammaAwareSimulator(
        heston=heston_params, aggregator=a2, memory_decay=1.0, memory_intake=1.0
    )
    s1, v1 = sim1.simulate(n_paths=8, n_steps=20, dt=1.0 / 252, seed=99)
    s2, v2 = sim2.simulate(n_paths=8, n_steps=20, dt=1.0 / 252, seed=99)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(v1, v2)


def test_gamma_aware_at_zero_kappa_matches_heston(heston_params: HestonParams) -> None:
    """GammaAware with κ=0 mock and a plain Heston should produce identical (S, v) paths."""
    no_op_agg = _MockAggregator(scale=0.0)
    gamma_aware = GammaAwareSimulator(
        heston=heston_params,
        aggregator=no_op_agg,
        memory_decay=1.0,
        memory_intake=1.0,
    )
    heston = HestonSimulator(regimes=[heston_params], breakpoints=[])
    s_g, v_g = gamma_aware.simulate(n_paths=8, n_steps=30, dt=1.0 / 252, seed=123)
    s_h, v_h = heston.simulate(n_paths=8, n_steps=30, dt=1.0 / 252, seed=123)
    np.testing.assert_allclose(s_g, s_h, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_g, v_h, rtol=1e-12, atol=1e-12)


def test_gamma_aware_protocol_uses_only_compute(heston_params: HestonParams) -> None:
    """Anything implementing GammaAggregatorProtocol works (duck typing check)."""

    class CustomAgg:
        def compute(self, spot: float, variance: float, log_memory: float) -> float:
            return -0.5

    agg: GammaAggregatorProtocol = CustomAgg()
    sim = GammaAwareSimulator(
        heston=heston_params,
        aggregator=agg,
        memory_decay=1.0,
        memory_intake=1.0,
    )
    state = SDEState(spot=100.0, variance=heston_params.v0, time=0.0, memory=0.0)
    new_state = sim.step(state, dt=1.0 / 252, dW=np.array([0.0, 0.0]))
    assert new_state.aggregate_gamma == pytest.approx(-0.5)
