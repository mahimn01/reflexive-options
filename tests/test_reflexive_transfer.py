"""Tests for the κ-sensitivity / reflexive-transfer experiment.

The smoketest end-to-end (`test_full_experiment_smoketest_n_seeds_2`) is
budgeted to <60 s on a CPU laptop — keep BC episodes, eval episodes, kappa
grid, and episode length minimal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reflexive_options.experiments.reflexive_transfer import (
    TransferConfig,
    evaluate_at_kappa,
    make_reflexive_sim_factory,
    run_experiment,
    train_bc_anchor_agent,
)
from reflexive_options.rl.actions import ActionConfig
from reflexive_options.rl.env import OptionsHedgeEnv
from reflexive_options.rl.experts import (
    DeltaHedgedShortVolExpert,
    bs_delta_call,
    bs_vega_call,
    make_delta_hedged_short_vol_expert,
)
from reflexive_options.rl.rewards import RewardConfig
from reflexive_options.rl.state import StateConfig
from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SurfaceGrid,
)

# ---------------------------------------------------------------------------
# Fixtures — small env so the smoketest stays under 60 s.
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_grid() -> SurfaceGrid:
    return SurfaceGrid(
        log_moneyness=np.array([-0.05, 0.0, 0.05], dtype=np.float64),
        maturities=np.array([30.0 / 365.0, 90.0 / 365.0], dtype=np.float64),
    )


@pytest.fixture
def tiny_cfg(tiny_grid: SurfaceGrid) -> TransferConfig:
    """Minimum-viable config for the smoketest."""
    del tiny_grid  # the sim_factory inside the cfg uses the *default* grid
    return TransferConfig(
        kappa_anchor=5.0e-12,
        # 5 points = the spline minimum (k=4) — keeps the smoketest in spec.
        kappa_grid_n_points=5,
        kappa_grid_low_mult=0.0,
        kappa_grid_high_mult=2.0,
        n_seeds_per_kappa=2,
        n_eval_episodes_per_seed=1,
        n_bc_train_episodes=2,
        episode_length=8,
        bc_epochs=2,
        bc_batch_size=8,
        seed=0,
    )


def _make_tiny_sim(grid: SurfaceGrid, kappa: float) -> ReflexiveSimulator:
    base = HestonParams(kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, v0=0.04)
    params = ReflexiveParams(
        base=base,
        coupling=float(kappa),
        drift=0.0,
        memory_decay=252.0,
        memory_intake=1.0,
        leverage=1.0,
    )
    n_k, n_t = grid.shape
    contracts = np.zeros((n_k, n_t), dtype=np.float64)
    atm = int(np.argmin(np.abs(grid.log_moneyness)))
    contracts[atm, :] = 1000.0
    oi = OpenInterestGrid(grid=grid, contracts_open=contracts)
    agg = GammaAggregator(oi_grid=oi, risk_free_rate=0.0)
    return ReflexiveSimulator(
        params=params,
        gamma_aggregator=agg,
        initial_spot=100.0,
        surface_grid=grid,
    )


def _make_tiny_env(
    sim: ReflexiveSimulator,
    grid: SurfaceGrid,
    *,
    episode_length: int = 8,
    seed: int = 0,
) -> OptionsHedgeEnv:
    return OptionsHedgeEnv(
        sim=sim,
        state_cfg=StateConfig(
            surface_grid=grid,
            position_dim=grid.n_strikes * grid.n_maturities,
            include_gamma=True,
            include_memory=True,
            history_window=0,
        ),
        action_cfg=ActionConfig(
            grid=grid,
            max_position_per_strike=10.0,
            discrete=False,
        ),
        reward_cfg=RewardConfig(
            transaction_cost_bps=1.0,
            position_size_penalty_lambda=0.0,
            sharpe_shaping=False,
        ),
        episode_length=episode_length,
        dt=1.0 / 252.0,
        initial_spot=100.0,
        initial_variance=0.04,
        rho=-0.70,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. Expert sanity
# ---------------------------------------------------------------------------


def test_delta_hedge_expert_sane_actions(tiny_grid: SurfaceGrid) -> None:
    """On a flat IV surface the expert should produce a small, bounded action."""
    sim = _make_tiny_sim(tiny_grid, kappa=0.0)
    env = _make_tiny_env(sim, tiny_grid)
    env.reset(seed=0)
    expert = DeltaHedgedShortVolExpert(
        env=env,
        short_atm_contracts=1.0,
        front_month_idx=0,
        max_position_per_strike=10.0,
    )
    action = expert.act()

    assert action.shape == (env.action_cfg.action_dim,)
    assert np.all(np.isfinite(action))
    # Bounded by max_position_per_strike.
    assert np.all(np.abs(action) <= 10.0 + 1e-9)
    # Most cells should be exactly zero — only short ATM and one hedge cell.
    nonzero = int(np.count_nonzero(action))
    assert 1 <= nonzero <= 2

    # The ATM front-month cell must be the negative leg.
    atm_idx = int(np.argmin(np.abs(tiny_grid.log_moneyness)))
    flat_idx = int(np.ravel_multi_index((atm_idx, 0), tiny_grid.shape))
    assert action[flat_idx] == pytest.approx(-1.0)


def test_bs_delta_and_vega_basics() -> None:
    """ATM call delta ~ 0.5 + small drift correction; vega is positive."""
    spot, strike, ttm, sigma, rate = 100.0, 100.0, 0.25, 0.20, 0.0
    delta = bs_delta_call(spot, strike, ttm, sigma, rate)
    vega = bs_vega_call(spot, strike, ttm, sigma, rate)
    assert 0.45 < delta < 0.55
    assert vega > 0.0
    # Degenerate inputs.
    assert bs_delta_call(spot, strike, 0.0, sigma, rate) in (0.0, 1.0)
    assert bs_vega_call(spot, strike, 0.0, sigma, rate) == 0.0


def test_make_delta_hedged_short_vol_expert_returns_callable(tiny_grid: SurfaceGrid) -> None:
    sim = _make_tiny_sim(tiny_grid, kappa=0.0)
    env = _make_tiny_env(sim, tiny_grid)
    env.reset(seed=0)
    expert = make_delta_hedged_short_vol_expert(env)
    action = expert()
    assert action.shape == (env.action_cfg.action_dim,)


# ---------------------------------------------------------------------------
# 2. BC training
# ---------------------------------------------------------------------------


def test_bc_train_runs_n_episodes(tmp_path: Path, tiny_cfg: TransferConfig) -> None:
    """Smallest possible BC train returns a valid checkpoint."""
    cfg = TransferConfig(**{**tiny_cfg.__dict__, "n_bc_train_episodes": 2, "bc_epochs": 1})
    sim_factory = make_reflexive_sim_factory(
        initial_spot=cfg.initial_spot,
        initial_variance=cfg.initial_variance,
    )
    ckpt = train_bc_anchor_agent(
        sim_factory,
        cfg=cfg,
        checkpoint_dir=tmp_path / "ckpt",
        use_cache=False,
    )
    assert ckpt.exists()
    assert ckpt.stat().st_size > 0

    # Cache hit on second call must return the same path without retraining.
    ckpt2 = train_bc_anchor_agent(
        sim_factory,
        cfg=cfg,
        checkpoint_dir=tmp_path / "ckpt",
        use_cache=True,
    )
    assert ckpt2 == ckpt


# ---------------------------------------------------------------------------
# 3. Evaluate at κ
# ---------------------------------------------------------------------------


def test_evaluate_at_kappa_returns_finite(tmp_path: Path, tiny_cfg: TransferConfig) -> None:
    cfg = TransferConfig(**{**tiny_cfg.__dict__, "n_bc_train_episodes": 2, "bc_epochs": 1})
    sim_factory = make_reflexive_sim_factory(
        initial_spot=cfg.initial_spot,
        initial_variance=cfg.initial_variance,
    )
    ckpt = train_bc_anchor_agent(
        sim_factory,
        cfg=cfg,
        checkpoint_dir=tmp_path / "ckpt",
        use_cache=False,
    )
    pnl = evaluate_at_kappa(
        agent_ckpt=ckpt,
        sim_factory=sim_factory,
        kappa=cfg.kappa_anchor,
        seed=123,
        cfg=cfg,
        n_eval_episodes=1,
    )
    assert isinstance(pnl, float)
    assert np.isfinite(pnl)


# ---------------------------------------------------------------------------
# 4. End-to-end smoketest — must complete in < 60 s.
# ---------------------------------------------------------------------------


def test_full_experiment_smoketest_n_seeds_2(tmp_path: Path, tiny_cfg: TransferConfig) -> None:
    metrics = run_experiment(tiny_cfg, tmp_path)

    # metrics.json was written.
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    on_disk = json.loads(metrics_path.read_text())
    assert on_disk["kappa_anchor"] == pytest.approx(tiny_cfg.kappa_anchor)
    assert len(on_disk["kappa_grid"]) == tiny_cfg.kappa_grid_n_points
    assert len(on_disk["metric_means"]) == tiny_cfg.kappa_grid_n_points
    assert len(on_disk["metric_stds"]) == tiny_cfg.kappa_grid_n_points

    # Returned dict matches the on-disk payload for the curve fields.
    assert metrics["metric_means"] == on_disk["metric_means"]
    assert isinstance(metrics["slope_at_anchor"], float)
    assert isinstance(metrics["slope_ci_low"], float)
    assert isinstance(metrics["slope_ci_high"], float)


# ---------------------------------------------------------------------------
# 5. Hand-rolled "perfect" agent check — slope is measurable when the metric
#    truly depends on κ. Skipped if the synthetic test box is too tight.
# ---------------------------------------------------------------------------


def test_sensitivity_curve_is_positive_at_anchor_for_correctly_trained_agent() -> None:
    """Synthetic check: a metric that depends linearly on κ produces a non-zero slope.

    This bypasses the BC student entirely and feeds a hand-crafted metric_fn
    directly into `kappa_sensitivity_curve`. The point is to verify that the
    *curve fitting* part of the pipeline correctly recovers a known slope; if
    this passes and the BC-driven slope is zero, the failure is in the agent,
    not in the sensitivity machinery.
    """
    from reflexive_options.theory.sensitivity import kappa_sensitivity_curve

    kappa_anchor = 5.0e-12
    kappa_grid = np.linspace(0.0, 2.0 * kappa_anchor, 9)
    true_slope = 1.0 / kappa_anchor  # so that f(κ_anchor) - f(0) = 1.0

    def linear_metric(kappa: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return true_slope * kappa + 0.01 * float(rng.standard_normal())

    result = kappa_sensitivity_curve(
        metric_fn=linear_metric,
        kappa_grid=kappa_grid,
        kappa_anchor=kappa_anchor,
        n_seeds=20,
        n_bootstrap=200,
        rng_seed=0,
    )
    # Recovered slope should be within an order of magnitude of true_slope.
    assert abs(result.slope_at_anchor - true_slope) / abs(true_slope) < 0.5
    # CI should exclude zero (slope is clearly positive).
    assert result.slope_ci_low > 0 or result.slope_ci_high < 0
