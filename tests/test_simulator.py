"""Tests for the reflexive 3D simulator and dealer-gamma aggregator."""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.simulator.gamma_aggregator import (
    GammaAggregator,
    GammaAggregatorConfig,
    GammaSignConvention,
)
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.simulator.stability import detect_blowup
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SDEState,
    SimulatorProtocol,
    SurfaceGrid,
)


def _make_grid(
    log_moneyness: np.ndarray | None = None,
    maturities: np.ndarray | None = None,
) -> SurfaceGrid:
    if log_moneyness is None:
        log_moneyness = np.array([-0.05, 0.0, 0.05])
    if maturities is None:
        maturities = np.array([30 / 365.25, 90 / 365.25])
    return SurfaceGrid(log_moneyness=log_moneyness, maturities=maturities)


def _make_oi(
    grid: SurfaceGrid, contracts: np.ndarray | None = None
) -> OpenInterestGrid:
    if contracts is None:
        contracts = np.zeros(grid.shape, dtype=np.float64)
    return OpenInterestGrid(grid=grid, contracts_open=contracts)


def _heston_params(
    *, kappa: float = 2.0, theta: float = 0.04, xi: float = 0.3, rho: float = -0.7, v0: float = 0.04
) -> HestonParams:
    return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


# ---------------------------------------------------------------------------
# GammaAggregator
# ---------------------------------------------------------------------------


def test_gamma_aggregator_zero_oi_returns_zero() -> None:
    grid = _make_grid()
    oi = _make_oi(grid)
    agg = GammaAggregator(oi_grid=oi, risk_free_rate=0.05)
    g = agg.compute(spot=100.0, variance=0.04, log_memory=0.0)
    assert g == 0.0


def test_gamma_aggregator_atm_call_positive() -> None:
    log_moneyness = np.array([0.0])
    maturities = np.array([30 / 365.25])
    grid = _make_grid(log_moneyness=log_moneyness, maturities=maturities)
    contracts = np.array([[10_000.0]])
    oi = _make_oi(grid, contracts=contracts)
    agg = GammaAggregator(
        oi_grid=oi,
        risk_free_rate=0.05,
        sign=GammaSignConvention(call_sign=1.0, put_sign=-1.0),
    )
    g = agg.compute(spot=100.0, variance=0.04, log_memory=0.0)
    assert g > 0.0
    # opposite sign convention flips it
    agg_flip = GammaAggregator(
        oi_grid=oi,
        risk_free_rate=0.05,
        sign=GammaSignConvention(call_sign=-1.0, put_sign=1.0),
    )
    assert agg_flip.compute(100.0, 0.04, 0.0) < 0.0


def test_gamma_aggregator_tau_floor_caps_singularity() -> None:
    log_moneyness = np.array([0.0])
    maturities = np.array([0.0])  # τ = 0 → would blow up without floor
    grid = _make_grid(log_moneyness=log_moneyness, maturities=maturities)
    contracts = np.array([[1_000.0]])
    oi = _make_oi(grid, contracts=contracts)
    agg = GammaAggregator(oi_grid=oi, risk_free_rate=0.05)
    g = agg.compute(spot=100.0, variance=0.04, log_memory=0.0)
    assert np.isfinite(g)
    assert g > 0.0


# ---------------------------------------------------------------------------
# ReflexiveSimulator — protocol & shape
# ---------------------------------------------------------------------------


def _make_simulator(
    *,
    coupling: float = 0.0,
    leverage: float = 0.0,
    drift: float = 0.0,
    initial_spot: float = 100.0,
    contracts_scale: float = 0.0,
    heston: HestonParams | None = None,
) -> ReflexiveSimulator:
    grid = _make_grid()
    contracts = np.full(grid.shape, contracts_scale, dtype=np.float64)
    oi = _make_oi(grid, contracts=contracts)
    aggregator = GammaAggregator(oi_grid=oi, risk_free_rate=0.05)
    params = ReflexiveParams(
        base=heston or _heston_params(),
        coupling=coupling,
        drift=drift,
        leverage=leverage,
    )
    return ReflexiveSimulator(
        params=params,
        gamma_aggregator=aggregator,
        initial_spot=initial_spot,
    )


def test_reflexive_simulator_protocol_compliance() -> None:
    sim = _make_simulator()
    assert isinstance(sim, SimulatorProtocol)


def test_simulate_shape() -> None:
    sim = _make_simulator()
    n_paths, n_steps = 8, 50
    spots, variances = sim.simulate(n_paths=n_paths, n_steps=n_steps, dt=1 / 252, seed=0)
    assert spots.shape == (n_paths, n_steps + 1)
    assert variances.shape == (n_paths, n_steps + 1)
    assert np.all(spots[:, 0] == 100.0)
    assert np.all(variances[:, 0] == sim.params.base.v0)


def test_simulate_no_nans() -> None:
    sim = _make_simulator(coupling=1e-12, contracts_scale=10_000.0)
    spots, variances = sim.simulate(n_paths=1000, n_steps=250, dt=1 / 252, seed=42)
    assert np.isfinite(spots).all()
    assert np.isfinite(variances).all()
    assert (variances > 0).all()


# ---------------------------------------------------------------------------
# κ = γ = 0 ⇒ Heston reduction
# ---------------------------------------------------------------------------


def _heston_reference(
    params: HestonParams,
    initial_spot: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    drift: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain full-truncation Heston (independent reference) for comparison."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    rho = params.rho
    dW_S = z1 * np.sqrt(dt)
    dW_v = (rho * z1 + np.sqrt(1.0 - rho * rho) * z2) * np.sqrt(dt)

    s = np.full(n_paths, initial_spot)
    v = np.full(n_paths, params.v0)
    spots = np.empty((n_paths, n_steps + 1))
    variances = np.empty((n_paths, n_steps + 1))
    spots[:, 0] = s
    variances[:, 0] = v

    for k in range(n_steps):
        sqrt_v = np.sqrt(np.maximum(v, 0.0))
        s = s + s * drift * dt + s * sqrt_v * dW_S[:, k]
        v = v + params.kappa * (params.theta - v) * dt + params.xi * sqrt_v * dW_v[:, k]
        v = np.maximum(v, 1e-8)
        spots[:, k + 1] = s
        variances[:, k + 1] = v

    return spots, variances


def test_reflexive_reduces_to_heston_at_zero_coupling() -> None:
    heston = _heston_params(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    sim = _make_simulator(coupling=0.0, leverage=0.0, heston=heston)
    # disable antithetic so the RNG draws line up with the reference
    sim.antithetic = False

    n_paths, n_steps, dt = 2000, 252, 1 / 252
    spots_sim, var_sim = sim.simulate(n_paths=n_paths, n_steps=n_steps, dt=dt, seed=123)
    spots_ref, var_ref = _heston_reference(
        heston, initial_spot=100.0, n_paths=n_paths, n_steps=n_steps, dt=dt, drift=0.0, seed=123
    )

    # Compare terminal moments (should match within MC noise on the same dW draws).
    assert np.isclose(spots_sim[:, -1].mean(), spots_ref[:, -1].mean(), rtol=5e-3)
    assert np.isclose(spots_sim[:, -1].std(), spots_ref[:, -1].std(), rtol=5e-2)
    assert np.isclose(var_sim[:, -1].mean(), var_ref[:, -1].mean(), rtol=5e-2)


# ---------------------------------------------------------------------------
# Blow-up detection on extreme κ
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_blowup_detection_on_extreme_kappa() -> None:
    # Use a long-call OI so G(S) > 0 → positive feedback → drift explodes when κ huge.
    grid = _make_grid()
    contracts = np.full(grid.shape, 100_000.0, dtype=np.float64)
    oi = _make_oi(grid, contracts=contracts)
    agg = GammaAggregator(
        oi_grid=oi,
        risk_free_rate=0.05,
        config=GammaAggregatorConfig(fixed_iv=0.2),
    )
    params = ReflexiveParams(
        base=_heston_params(),
        coupling=10.0,
        drift=0.0,
    )
    sim = ReflexiveSimulator(
        params=params,
        gamma_aggregator=agg,
        initial_spot=100.0,
    )
    spots, variances = sim.simulate(n_paths=32, n_steps=100, dt=1 / 252, seed=7)
    result = detect_blowup(spots, variances, initial_spot=100.0)
    assert result.n_rejected > 0


# ---------------------------------------------------------------------------
# step() interface
# ---------------------------------------------------------------------------


def test_step_interface_advances_state_correctly() -> None:
    sim = _make_simulator(coupling=0.0, leverage=0.0)
    s0, v0 = 100.0, 0.04
    state = SDEState(spot=s0, variance=v0, time=0.0, aggregate_gamma=0.0, memory=0.0)
    dt = 1 / 252
    dW = np.array([0.01, -0.005], dtype=np.float64)

    new_state = sim.step(state, dt=dt, dW=dW)

    # Manual one-step Euler — coupling 0, leverage 0, drift 0
    sqrt_v = np.sqrt(v0)
    expected_spot = s0 + s0 * sqrt_v * dW[0]
    drift_v = sim.params.base.kappa * (sim.params.base.theta - v0)
    expected_v = max(v0 + drift_v * dt + sim.params.base.xi * sqrt_v * dW[1], 1e-8)
    expected_z = (
        0.0
        + (-sim.params.memory_decay * 0.0 + sim.params.memory_intake * (np.log(s0) - np.log(s0))) * dt
    )

    assert new_state.spot == pytest.approx(expected_spot, rel=1e-12)
    assert new_state.variance == pytest.approx(expected_v, rel=1e-12)
    assert new_state.memory == pytest.approx(expected_z, abs=1e-12)
    assert new_state.time == pytest.approx(dt)
    assert new_state.aggregate_gamma == 0.0


def test_step_z_evolution_matches_log_price() -> None:
    sim = _make_simulator(coupling=0.0, leverage=0.0)
    state = SDEState(spot=110.0, variance=0.04, time=0.0, aggregate_gamma=0.0, memory=0.0)
    dt = 1 / 252
    dW = np.zeros(2, dtype=np.float64)
    new_state = sim.step(state, dt=dt, dW=dW)
    # z update with z_0 = 0, S = 110, S_0 = 100
    expected = sim.params.memory_intake * np.log(110.0 / 100.0) * dt
    assert new_state.memory == pytest.approx(expected, rel=1e-12)


def test_implied_surface_returns_constant_iv_at_sqrt_v() -> None:
    sim = _make_simulator()
    grid = _make_grid()
    state = SDEState(spot=100.0, variance=0.09, time=0.0)
    surf = sim.implied_surface(state, grid)
    assert surf.shape == grid.shape
    assert np.allclose(surf, np.sqrt(0.09))
