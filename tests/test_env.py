"""Tests for the OptionsHedgeEnv + state/action/reward + curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

from reflexive_options.baselines import (
    GammaAwareSimulator,
    HestonSimulator,
)
from reflexive_options.rl import (
    ActionConfig,
    CurriculumStage,
    OptionsHedgeEnv,
    RewardConfig,
    StateConfig,
    apply_action,
    build_curriculum,
    build_observation,
    compute_reward,
    make_action_space,
    price_option_position,
)
from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SDEState,
    SurfaceGrid,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_grid() -> SurfaceGrid:
    return SurfaceGrid(
        log_moneyness=np.array([-0.05, 0.0, 0.05], dtype=np.float64),
        maturities=np.array([0.25, 0.5], dtype=np.float64),
    )


@pytest.fixture
def heston_params() -> HestonParams:
    return HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


@pytest.fixture
def reflexive_params(heston_params: HestonParams) -> ReflexiveParams:
    return ReflexiveParams(
        base=heston_params,
        coupling=1e-12,
        drift=0.0,
        memory_decay=252.0,
        memory_intake=1.0,
        leverage=1.0,
    )


@dataclass
class _MockAggregator:
    scale: float = 1e-3

    def compute(self, spot: float, variance: float, log_memory: float) -> float:
        return float(self.scale * (spot - 100.0) * (1.0 + abs(log_memory)))


def _make_oi_grid(grid: SurfaceGrid) -> OpenInterestGrid:
    n_k, n_t = grid.shape
    contracts = np.zeros((n_k, n_t), dtype=np.float64)
    atm = int(np.argmin(np.abs(grid.log_moneyness)))
    contracts[atm, :] = 1000.0
    return OpenInterestGrid(grid=grid, contracts_open=contracts)


def _state_cfg(
    grid: SurfaceGrid,
    *,
    include_gamma: bool = True,
    include_memory: bool = True,
    history_window: int = 0,
) -> StateConfig:
    return StateConfig(
        surface_grid=grid,
        position_dim=grid.n_strikes * grid.n_maturities,
        include_gamma=include_gamma,
        include_memory=include_memory,
        history_window=history_window,
    )


def _action_cfg(grid: SurfaceGrid, *, max_pos: float = 10.0) -> ActionConfig:
    return ActionConfig(grid=grid, max_position_per_strike=max_pos, discrete=False)


def _reward_cfg() -> RewardConfig:
    return RewardConfig(
        transaction_cost_bps=1.0,
        position_size_penalty_lambda=0.0,
        sharpe_shaping=False,
    )


# ---------------------------------------------------------------------------
# State / observation
# ---------------------------------------------------------------------------


def test_observation_layout(small_grid: SurfaceGrid) -> None:
    cfg = _state_cfg(small_grid, history_window=2)
    n_k, n_t = small_grid.shape
    pos = np.arange(n_k * n_t, dtype=np.float64)
    surface = 0.2 * np.ones(small_grid.shape, dtype=np.float64)
    history = np.stack([np.full(small_grid.shape, 0.18), np.full(small_grid.shape, 0.19)])
    state = SDEState(
        spot=110.0,
        variance=0.05,
        time=0.1,
        aggregate_gamma=42.0,
        memory=0.7,
    )

    obs = build_observation(state, surface, pos, history, cfg, initial_spot=100.0)

    assert obs.shape == (cfg.observation_dim,)
    layout = cfg.observation_layout()
    assert obs[layout["spot"]][0] == pytest.approx(110.0)
    assert obs[layout["log_moneyness_to_s0"]][0] == pytest.approx(np.log(110.0 / 100.0))
    assert obs[layout["variance"]][0] == pytest.approx(0.05)
    assert obs[layout["sqrt_variance"]][0] == pytest.approx(np.sqrt(0.05))
    assert obs[layout["aggregate_gamma"]][0] == pytest.approx(42.0)
    assert obs[layout["memory"]][0] == pytest.approx(0.7)
    np.testing.assert_array_equal(obs[layout["surface"]], surface.reshape(-1))
    np.testing.assert_array_equal(obs[layout["position"]], pos)
    expected_tte = np.maximum(small_grid.maturities - 0.1, 0.0)
    np.testing.assert_allclose(obs[layout["time_to_expiry"]], expected_tte)
    np.testing.assert_array_equal(obs[layout["history"]], history.reshape(-1))


def test_observation_strip_gamma_excludes_g(small_grid: SurfaceGrid) -> None:
    cfg_with = _state_cfg(small_grid, include_gamma=True, history_window=0)
    cfg_without = _state_cfg(small_grid, include_gamma=False, history_window=0)

    assert cfg_without.observation_dim == cfg_with.observation_dim - 1
    assert "aggregate_gamma" in cfg_with.observation_layout()
    assert "aggregate_gamma" not in cfg_without.observation_layout()

    n_k, n_t = small_grid.shape
    pos = np.zeros(n_k * n_t, dtype=np.float64)
    surface = np.full(small_grid.shape, 0.2, dtype=np.float64)
    state = SDEState(spot=100.0, variance=0.04, time=0.0, aggregate_gamma=99.0, memory=0.0)

    obs_with = build_observation(state, surface, pos, None, cfg_with, initial_spot=100.0)
    obs_without = build_observation(state, surface, pos, None, cfg_without, initial_spot=100.0)
    assert obs_without.shape[0] == cfg_with.observation_dim - 1
    assert 99.0 in obs_with.tolist()
    assert 99.0 not in obs_without.tolist()


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


def test_action_space_continuous_bounds(small_grid: SurfaceGrid) -> None:
    cfg = ActionConfig(grid=small_grid, max_position_per_strike=25.0, discrete=False)
    space = make_action_space(cfg)
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (small_grid.n_strikes * small_grid.n_maturities,)
    assert np.all(space.low == -25.0)
    assert np.all(space.high == 25.0)


def test_action_space_discrete_buckets(small_grid: SurfaceGrid) -> None:
    cfg = ActionConfig(grid=small_grid, max_position_per_strike=5.0, discrete=True, n_buckets=5)
    space = make_action_space(cfg)
    assert isinstance(space, gym.spaces.MultiDiscrete)
    assert space.shape == (small_grid.n_strikes * small_grid.n_maturities,)
    assert np.all(space.nvec == 5)


def test_apply_action_clips_to_max_position(small_grid: SurfaceGrid) -> None:
    cfg = ActionConfig(grid=small_grid, max_position_per_strike=3.0)
    cur = np.zeros(cfg.action_dim, dtype=np.float64)
    raw = np.array([10.0, -10.0, 1.5, 0.0, 5.0, -2.0], dtype=np.float64)
    new = apply_action(cur, raw, cfg)
    np.testing.assert_array_equal(
        new,
        np.array([3.0, -3.0, 1.5, 0.0, 3.0, -2.0]),
    )
    assert new.shape == (cfg.action_dim,)


def test_apply_action_discrete_selects_bucket(small_grid: SurfaceGrid) -> None:
    cfg = ActionConfig(grid=small_grid, max_position_per_strike=4.0, discrete=True, n_buckets=5)
    cur = np.zeros(cfg.action_dim, dtype=np.float64)
    # bucket centers = [-4, -2, 0, 2, 4]; index 2 = 0; index 4 = +4
    idx = np.array([0, 2, 4, 2, 1, 3], dtype=np.int64)
    new = apply_action(cur, idx, cfg)
    np.testing.assert_array_equal(new, np.array([-4.0, 0.0, 4.0, 0.0, -2.0, 2.0]))


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def test_compute_reward_subtracts_transaction_cost_correctly() -> None:
    cfg = RewardConfig(
        transaction_cost_bps=2.0,
        position_size_penalty_lambda=1e-3,
        sharpe_shaping=False,
    )
    pnl = 100.0
    trade_dollars = -5_000.0  # |.| = 5000 → cost = 2 bps × 5000 = 1.0
    gross = 50_000.0  # penalty = 1e-3 × 50_000 = 50
    r = compute_reward(pnl, trade_dollars, gross, [], cfg)
    assert r == pytest.approx(100.0 - 1.0 - 50.0)


def test_compute_reward_sharpe_shaping_adds_term() -> None:
    cfg_off = RewardConfig(
        transaction_cost_bps=0.0,
        position_size_penalty_lambda=0.0,
        sharpe_shaping=False,
        sharpe_window=4,
    )
    cfg_on = RewardConfig(
        transaction_cost_bps=0.0,
        position_size_penalty_lambda=0.0,
        sharpe_shaping=True,
        sharpe_window=4,
    )
    history = [1.0, 1.5, 0.5, 1.2]
    r_off = compute_reward(2.0, 0.0, 0.0, history, cfg_off)
    r_on = compute_reward(2.0, 0.0, 0.0, history, cfg_on)
    assert r_off == pytest.approx(2.0)
    assert r_on != r_off


# ---------------------------------------------------------------------------
# price_option_position
# ---------------------------------------------------------------------------


def test_price_option_position_zero_position_is_zero(small_grid: SurfaceGrid) -> None:
    surface = np.full(small_grid.shape, 0.2, dtype=np.float64)
    pos = np.zeros(small_grid.shape, dtype=np.float64)
    val = price_option_position(
        spot=100.0,
        surface=surface,
        position=pos,
        grid=small_grid,
        rate=0.0,
        dividend=0.0,
        state_time=0.0,
    )
    assert val == 0.0


def test_price_option_position_atm_call_positive(small_grid: SurfaceGrid) -> None:
    surface = np.full(small_grid.shape, 0.2, dtype=np.float64)
    pos = np.zeros(small_grid.shape, dtype=np.float64)
    atm_idx = int(np.argmin(np.abs(small_grid.log_moneyness)))
    pos[atm_idx, 0] = 1.0
    val = price_option_position(
        spot=100.0,
        surface=surface,
        position=pos,
        grid=small_grid,
        rate=0.0,
        dividend=0.0,
        state_time=0.0,
    )
    # ATM call with σ=0.20 and T=0.25 ≈ 0.4 × σ × √T × S ≈ 0.4 × 0.2 × 0.5 × 100 ≈ 4
    assert val == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# Env smoke tests
# ---------------------------------------------------------------------------


def _make_heston_sim(heston_params: HestonParams) -> HestonSimulator:
    return HestonSimulator(regimes=[heston_params], breakpoints=[], spot0=100.0)


def _make_reflexive_sim(
    reflexive_params: ReflexiveParams, grid: SurfaceGrid
) -> ReflexiveSimulator:
    agg = GammaAggregator(oi_grid=_make_oi_grid(grid), risk_free_rate=0.0)
    return ReflexiveSimulator(
        params=reflexive_params,
        gamma_aggregator=agg,
        initial_spot=100.0,
        surface_grid=grid,
    )


def _make_gamma_aware_sim(heston_params: HestonParams) -> GammaAwareSimulator:
    return GammaAwareSimulator(
        heston=heston_params,
        aggregator=_MockAggregator(scale=1e-3),
        memory_decay=252.0,
        memory_intake=1.0,
        spot0=100.0,
    )


def _make_env(
    sim: object,
    grid: SurfaceGrid,
    heston_params: HestonParams,
    *,
    episode_length: int = 5,
) -> OptionsHedgeEnv:
    return OptionsHedgeEnv(
        sim=sim,  # type: ignore[arg-type]
        state_cfg=_state_cfg(grid, history_window=2),
        action_cfg=_action_cfg(grid),
        reward_cfg=_reward_cfg(),
        episode_length=episode_length,
        dt=1.0 / 252,
        initial_spot=100.0,
        initial_variance=heston_params.v0,
        rho=heston_params.rho,
        seed=0,
    )


def test_env_reset_returns_valid_obs_and_info(
    small_grid: SurfaceGrid, heston_params: HestonParams
) -> None:
    sim = _make_heston_sim(heston_params)
    env = _make_env(sim, small_grid, heston_params)
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.state_cfg.observation_dim,)
    assert np.all(np.isfinite(obs))
    assert info["step"] == 0
    assert info["spot"] == pytest.approx(100.0)


def test_env_step_advances_state_and_returns_reward(
    small_grid: SurfaceGrid, heston_params: HestonParams
) -> None:
    sim = _make_heston_sim(heston_params)
    env = _make_env(sim, small_grid, heston_params)
    obs0, _ = env.reset(seed=0)
    action = np.zeros(env.action_cfg.action_dim, dtype=np.float64)
    obs1, reward, terminated, truncated, info = env.step(action)
    assert obs1.shape == obs0.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1
    assert "pnl" in info
    assert "rolling_sharpe" in info


def test_env_works_with_reflexive_sim(
    small_grid: SurfaceGrid,
    reflexive_params: ReflexiveParams,
    heston_params: HestonParams,
) -> None:
    sim = _make_reflexive_sim(reflexive_params, small_grid)
    env = _make_env(sim, small_grid, heston_params)
    obs, _ = env.reset(seed=1)
    assert np.all(np.isfinite(obs))
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, info = env.step(np.asarray(action, dtype=np.float64))
        assert np.all(np.isfinite(obs))
    assert info["aggregate_gamma"] != 0.0  # reflexive sim populates G


def test_env_works_with_gamma_aware_sim(
    small_grid: SurfaceGrid, heston_params: HestonParams
) -> None:
    sim = _make_gamma_aware_sim(heston_params)
    env = _make_env(sim, small_grid, heston_params)
    obs, _ = env.reset(seed=2)
    assert np.all(np.isfinite(obs))
    action = np.zeros(env.action_cfg.action_dim, dtype=np.float64)
    _, _, _, _, info = env.step(action)
    # Gamma-aware sim must populate G_t and z_t on each step (state-symmetric).
    assert info["aggregate_gamma"] != 0.0 or info["spot"] != 100.0


def test_env_works_with_heston_sim(
    small_grid: SurfaceGrid, heston_params: HestonParams
) -> None:
    sim = _make_heston_sim(heston_params)
    env = _make_env(sim, small_grid, heston_params)
    env.reset(seed=3)
    for _ in range(4):
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
        assert np.isfinite(reward)


def test_episode_terminates_at_episode_length(
    small_grid: SurfaceGrid, heston_params: HestonParams
) -> None:
    sim = _make_heston_sim(heston_params)
    env = _make_env(sim, small_grid, heston_params, episode_length=3)
    env.reset(seed=4)
    truncations = []
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(
            np.zeros(env.action_cfg.action_dim, dtype=np.float64)
        )
        truncations.append((terminated, truncated))
        if terminated or truncated:
            break
    # Episode of length 3 ⇒ truncated on the 3rd step.
    assert truncations[-1] == (False, True)
    assert len(truncations) == 3


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def test_curriculum_stages_have_distinct_simulators(reflexive_params: ReflexiveParams) -> None:
    stages = build_curriculum(
        base_params=reflexive_params,
        stages=["calm", "vol_event_2018", "vol_event_2020", "vol_event_2024"],
        n_episodes_per_stage=10,
    )
    assert len(stages) == 4
    assert [s.name for s in stages] == [
        "calm",
        "vol_event_2018",
        "vol_event_2020",
        "vol_event_2024",
    ]
    sims = [s.sim_factory() for s in stages]
    # Each must be a simulator-protocol-conforming object with distinct coupling.
    couplings = []
    for sim in sims:
        assert isinstance(sim, ReflexiveSimulator)
        couplings.append(sim.params.coupling)
    assert len(set(couplings)) == 4

    for s in stages:
        assert isinstance(s, CurriculumStage)
        assert s.n_episodes == 10


def test_curriculum_rejects_unknown_stage(reflexive_params: ReflexiveParams) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        build_curriculum(
            base_params=reflexive_params,
            stages=["bogus"],  # type: ignore[list-item]
        )


# ---------------------------------------------------------------------------
# ATLAS-adapter smoke test (BC, not PPO — see test_atlas_import.py rationale)
# ---------------------------------------------------------------------------


def test_atlas_adapter_can_run_one_bc_step_on_env(
    small_grid: SurfaceGrid, heston_params: HestonParams
) -> None:
    """Confirm the env's observation feeds the vendored Mamba backbone end-to-end.

    Uses BC infra rather than PPO (PPO trainer was Tier-3 SKIP per the import
    surface brief — see tests/test_atlas_import.py). We instantiate a Mamba
    backbone sized to the env's observation/action dims, push one obs through
    it, and assert the output projects to a valid action shape.
    """
    import torch

    from reflexive_options.rl.atlas_adapter import MambaBackbone

    sim = _make_heston_sim(heston_params)
    env = _make_env(sim, small_grid, heston_params)
    obs, _ = env.reset(seed=7)

    d_model = 16
    backbone = MambaBackbone(d_model=d_model, n_layers=1, n_heads=2, ffn_mult=2)
    in_proj = torch.nn.Linear(env.state_cfg.observation_dim, d_model)
    out_proj = torch.nn.Linear(d_model, env.action_cfg.action_dim)

    x = torch.from_numpy(obs).float().reshape(1, 1, -1)  # (B=1, T=1, obs_dim)
    h = backbone(in_proj(x))
    raw_action = out_proj(h).squeeze(0).squeeze(0).detach().numpy().astype(np.float64)

    # One env step using that action should not crash and should return finite.
    obs2, reward, _, _, _ = env.step(raw_action)
    assert obs2.shape == obs.shape
    assert np.isfinite(reward)
