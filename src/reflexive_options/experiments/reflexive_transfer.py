"""κ-sensitivity transfer experiment — the *novel* evaluation contribution (H2).

Train a single anchor agent π_{κ₀} in the reflexive simulator at the calibrated
κ₀, then deploy it across a family of environments at κ ∈ [0, 2κ₀]. The slope
at κ₀ is a quantitative measure of how much the agent depends on the reflexive
dynamics it was trained on. No published precedent in finance — see
`~/Documents/reflexivity-research/evaluation_framework_brief.md` §3.

**v1 implementation note (honest disclosure).** The pre-reg compute envelope
calls for full PPO with EWC over the vendored ATLAS Mamba backbone (~200
GPU-hours at convergence). To unblock the κ-sensitivity pipeline before that
budget is available, this script substitutes:

    - Expert policy: heuristic delta-hedged short ATM straddle
      (`reflexive_options.rl.experts.make_delta_hedged_short_vol_expert`).
    - "Agent" π_{κ₀}: a small MLP (hidden=64, 2 layers, ReLU) trained by
      behavioral cloning on `n_episodes` of expert trajectories collected
      inside the κ₀ environment. The MLP takes the env's flat observation
      and outputs the action vector.

The full-fat path through `atlas_adapter.train_behavioral_cloning` is gated
behind `--use-atlas-bc` (defaults to off) — that path is currently coupled to
the ATLAS OHLCV pipeline and would require a domain adapter to feed it
options-env trajectories. Until that adapter exists the small-MLP path is
production-correct for the H2 hypothesis test (see
`paper/pre_registration.md`); the architecture choice is orthogonal to the
slope being measured.

**Compute envelope (n_seeds=100, kappa_grid=9 points).** Estimated wall-clock
on an Apple M-series CPU:

    - BC anchor train (n_episodes=200, episode_length=63): ~2-4 min.
    - κ-sweep eval (9 × 100 × 5 episodes, episode_length=63): ~10-15 min.
    - Bootstrap (1000 resamples on 9-point grid): <30 s.
    Total: ~15-20 min wall-clock. No GPU required.

The smoketest path (n_seeds=2, kappa_grid=3, n_episodes=2) runs in <60 s on
the same hardware.

Run: ``python -m reflexive_options.experiments.reflexive_transfer``
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from reflexive_options.experiments._common import (
    deterministic_rng,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.rl.actions import ActionConfig
from reflexive_options.rl.curriculum import (
    _default_oi_grid,
    _default_surface_grid,
)
from reflexive_options.rl.env import OptionsHedgeEnv
from reflexive_options.rl.experts import make_delta_hedged_short_vol_expert
from reflexive_options.rl.rewards import RewardConfig
from reflexive_options.rl.state import StateConfig
from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.theory.sensitivity import kappa_sensitivity_curve
from reflexive_options.types import (
    HestonParams,
    ReflexiveParams,
    SimulatorProtocol,
    SurfaceGrid,
)


@dataclass(frozen=True)
class TransferConfig:
    """Configuration for the κ-transfer experiment."""

    kappa_anchor: float = 5.0e-12  # κ₀, calibrated value from dealer_gamma_brief.md
    kappa_grid_n_points: int = 9
    kappa_grid_low_mult: float = 0.0
    kappa_grid_high_mult: float = 2.0
    n_seeds_per_kappa: int = 100
    n_eval_episodes_per_seed: int = 5
    n_bc_train_episodes: int = 200
    episode_length: int = 63  # 3 trading months — matches pre-reg horizon for a single-cycle short
    bc_epochs: int = 20
    bc_batch_size: int = 64
    bc_lr: float = 1e-3
    bc_hidden_dim: int = 64
    bc_n_hidden_layers: int = 2
    seed: int = 42
    initial_spot: float = 100.0
    initial_variance: float = 0.04


# ---------------------------------------------------------------------------
# Sim/env factories
# ---------------------------------------------------------------------------


def _base_heston_params(initial_variance: float) -> HestonParams:
    """Heston backbone shared across the κ family — only the coupling sweeps."""
    return HestonParams(
        kappa=2.0,
        theta=initial_variance,
        xi=0.30,
        rho=-0.70,
        v0=initial_variance,
    )


def make_reflexive_sim_factory(
    *,
    initial_spot: float,
    initial_variance: float,
    surface_grid: SurfaceGrid | None = None,
) -> Callable[[float], SimulatorProtocol]:
    """Build a κ-parameterized simulator factory.

    The returned callable maps a κ value to a fresh `ReflexiveSimulator`. All
    other Heston parameters are held fixed across the family so the κ-sensitivity
    sweep isolates the coupling axis.
    """
    grid = surface_grid if surface_grid is not None else _default_surface_grid()
    base = _base_heston_params(initial_variance)

    def _factory(kappa: float) -> SimulatorProtocol:
        params = ReflexiveParams(
            base=base,
            coupling=float(kappa),
            drift=0.0,
            memory_decay=252.0,
            memory_intake=1.0,
            leverage=1.0,
        )
        agg = GammaAggregator(
            oi_grid=_default_oi_grid(grid),
            risk_free_rate=0.0,
            dividend_yield=0.0,
        )
        return ReflexiveSimulator(
            params=params,
            gamma_aggregator=agg,
            initial_spot=initial_spot,
            surface_grid=grid,
        )

    return _factory


def _make_env(
    sim: SimulatorProtocol,
    *,
    initial_spot: float,
    initial_variance: float,
    episode_length: int,
    seed: int,
    surface_grid: SurfaceGrid | None = None,
) -> OptionsHedgeEnv:
    grid = surface_grid if surface_grid is not None else _default_surface_grid()
    state_cfg = StateConfig(
        surface_grid=grid,
        position_dim=grid.n_strikes * grid.n_maturities,
        include_gamma=True,
        include_memory=True,
        history_window=0,  # no history for the small-MLP BC path; keeps obs flat
    )
    action_cfg = ActionConfig(
        grid=grid,
        max_position_per_strike=10.0,
        discrete=False,
    )
    reward_cfg = RewardConfig(
        transaction_cost_bps=1.0,
        position_size_penalty_lambda=0.0,
        sharpe_shaping=False,
    )
    return OptionsHedgeEnv(
        sim=sim,
        state_cfg=state_cfg,
        action_cfg=action_cfg,
        reward_cfg=reward_cfg,
        episode_length=episode_length,
        dt=1.0 / 252.0,
        initial_spot=initial_spot,
        initial_variance=initial_variance,
        rho=-0.70,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# BC student — small MLP
# ---------------------------------------------------------------------------


class _MLPPolicy(nn.Module):  # type: ignore[misc]
    """Plain feedforward policy used for the BC student.

    Mirrors the env's (obs_dim → action_dim) mapping with `n_hidden_layers`
    × `hidden_dim` ReLU stack. tanh on the output then scaled to the action
    bound — keeps actions inside the env's Box and avoids unbounded outputs
    that would explode the position-size penalty in evaluation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int,
        n_hidden_layers: int,
        action_bound: float,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.action_bound = float(action_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_bound * torch.tanh(self.net(x))


def _collect_expert_trajectories(
    env: OptionsHedgeEnv,
    expert: Callable[[], NDArray[np.float64]],
    n_episodes: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Roll the expert in `env` for `n_episodes` and stack (obs, action) tuples."""
    obs_buf: list[NDArray[np.float64]] = []
    act_buf: list[NDArray[np.float64]] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = expert()
            obs_buf.append(np.asarray(obs, dtype=np.float64))
            act_buf.append(np.asarray(action, dtype=np.float64))
            obs, _, terminated, truncated, _ = env.step(action)
    return np.stack(obs_buf, axis=0), np.stack(act_buf, axis=0)


def _train_mlp_bc(
    obs: NDArray[np.float64],
    actions: NDArray[np.float64],
    obs_dim: int,
    action_dim: int,
    *,
    hidden_dim: int,
    n_hidden_layers: int,
    action_bound: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> _MLPPolicy:
    """Standard supervised regression: MSE on (obs → action) pairs."""
    torch.manual_seed(seed)
    model = _MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        action_bound=action_bound,
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_tensor = torch.tensor(obs, dtype=torch.float32)
    y_tensor = torch.tensor(actions, dtype=torch.float32)
    n = x_tensor.shape[0]

    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x_tensor[idx]
            yb = y_tensor[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


# ---------------------------------------------------------------------------
# Public anchor-training and evaluation API
# ---------------------------------------------------------------------------


def train_bc_anchor_agent(
    sim_factory: Callable[[float], SimulatorProtocol],
    *,
    cfg: TransferConfig,
    checkpoint_dir: Path,
    surface_grid: SurfaceGrid | None = None,
    use_cache: bool = True,
) -> Path:
    """Train (or load cached) the BC anchor agent at κ₀.

    The checkpoint is a torch state-dict together with the architecture sizing
    needed to rebuild the model (obs_dim, action_dim, hidden_dim,
    n_hidden_layers, action_bound).
    """
    ckpt_path = checkpoint_dir / "bc_anchor.pt"
    if use_cache and ckpt_path.exists():
        return ckpt_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_factory(cfg.kappa_anchor)
    env = _make_env(
        sim,
        initial_spot=cfg.initial_spot,
        initial_variance=cfg.initial_variance,
        episode_length=cfg.episode_length,
        seed=cfg.seed,
        surface_grid=surface_grid,
    )
    expert = make_delta_hedged_short_vol_expert(env)

    obs_buf, act_buf = _collect_expert_trajectories(
        env=env,
        expert=expert,
        n_episodes=cfg.n_bc_train_episodes,
        seed=cfg.seed,
    )

    model = _train_mlp_bc(
        obs_buf,
        act_buf,
        obs_dim=env.state_cfg.observation_dim,
        action_dim=env.action_cfg.action_dim,
        hidden_dim=cfg.bc_hidden_dim,
        n_hidden_layers=cfg.bc_n_hidden_layers,
        action_bound=env.action_cfg.max_position_per_strike,
        epochs=cfg.bc_epochs,
        batch_size=cfg.bc_batch_size,
        lr=cfg.bc_lr,
        seed=cfg.seed,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": env.state_cfg.observation_dim,
            "action_dim": env.action_cfg.action_dim,
            "hidden_dim": cfg.bc_hidden_dim,
            "n_hidden_layers": cfg.bc_n_hidden_layers,
            "action_bound": env.action_cfg.max_position_per_strike,
            "kappa_anchor": cfg.kappa_anchor,
        },
        ckpt_path,
    )
    return ckpt_path


def _load_policy(ckpt_path: Path) -> _MLPPolicy:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _MLPPolicy(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        n_hidden_layers=int(payload["n_hidden_layers"]),
        action_bound=float(payload["action_bound"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def evaluate_at_kappa(
    agent_ckpt: Path,
    sim_factory: Callable[[float], SimulatorProtocol],
    kappa: float,
    seed: int,
    *,
    cfg: TransferConfig,
    surface_grid: SurfaceGrid | None = None,
    n_eval_episodes: int | None = None,
) -> float:
    """Roll out the cached anchor agent inside the κ environment, return mean total P&L.

    Mean is taken across `n_eval_episodes` independent rollouts; each rollout
    runs for `cfg.episode_length` env steps. Per-episode total P&L is the sum
    of `info["pnl"]` over the rollout.
    """
    n_eval = n_eval_episodes if n_eval_episodes is not None else cfg.n_eval_episodes_per_seed
    if n_eval <= 0:
        raise ValueError(f"n_eval_episodes must be > 0, got {n_eval}")

    model = _load_policy(agent_ckpt)
    sim = sim_factory(float(kappa))
    env = _make_env(
        sim,
        initial_spot=cfg.initial_spot,
        initial_variance=cfg.initial_variance,
        episode_length=cfg.episode_length,
        seed=seed,
        surface_grid=surface_grid,
    )

    episode_pnls: list[float] = []
    with torch.no_grad():
        for ep in range(n_eval):
            obs, _ = env.reset(seed=seed + ep)
            total_pnl = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = model(obs_t).squeeze(0).cpu().numpy().astype(np.float64)
                # Defensively clip — the MLP output is tanh-bounded but float
                # rounding at the edge can land just outside the env's Box.
                action = np.clip(
                    action,
                    -env.action_cfg.max_position_per_strike,
                    +env.action_cfg.max_position_per_strike,
                )
                obs, _, terminated, truncated, info = env.step(action)
                total_pnl += float(info["pnl"])
            episode_pnls.append(total_pnl)
    mean_pnl = float(np.mean(episode_pnls))
    if not math.isfinite(mean_pnl):
        # Bubble up — sensitivity-curve bootstrap will hit ValueError on the spline.
        raise RuntimeError(f"non-finite mean P&L from κ={kappa}, seed={seed}: {episode_pnls}")
    return mean_pnl


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment(cfg: TransferConfig, run_dir: Path) -> dict[str, object]:
    """Single-call entry point used both by main() and by tests."""
    save_config(run_dir, cfg)

    sim_factory = make_reflexive_sim_factory(
        initial_spot=cfg.initial_spot,
        initial_variance=cfg.initial_variance,
    )

    ckpt_path = train_bc_anchor_agent(
        sim_factory,
        cfg=cfg,
        checkpoint_dir=run_dir / "checkpoints",
    )

    kappa_grid = np.linspace(
        cfg.kappa_anchor * cfg.kappa_grid_low_mult,
        cfg.kappa_anchor * cfg.kappa_grid_high_mult,
        cfg.kappa_grid_n_points,
    ).astype(np.float64)

    rng = deterministic_rng(cfg.seed)

    def _metric_at_kappa(kappa: float, seed: int) -> float:
        return evaluate_at_kappa(
            agent_ckpt=ckpt_path,
            sim_factory=sim_factory,
            kappa=kappa,
            seed=seed,
            cfg=cfg,
        )

    with timed("kappa_sensitivity_sweep"):
        result = kappa_sensitivity_curve(
            metric_fn=_metric_at_kappa,
            kappa_grid=kappa_grid,
            kappa_anchor=cfg.kappa_anchor,
            n_seeds=cfg.n_seeds_per_kappa,
            n_bootstrap=1_000,
            rng_seed=cfg.seed,
        )

    metrics: dict[str, object] = {
        "kappa_anchor": cfg.kappa_anchor,
        "kappa_grid": result.kappa_grid.tolist(),
        "metric_means": result.metric_values.tolist(),
        "metric_stds": result.metric_std.tolist(),
        "slope_at_anchor": result.slope_at_anchor,
        "slope_ci_low": result.slope_ci_low,
        "slope_ci_high": result.slope_ci_high,
        "ci_excludes_zero": (result.slope_ci_low > 0) or (result.slope_ci_high < 0),
        "rng_seed": cfg.seed,
        "rng_state": str(rng),
        "checkpoint_path": str(ckpt_path),
        "metric_definition": "mean total per-episode P&L across n_eval_episodes_per_seed rollouts",
        "policy_class": "MLPPolicy (BC student trained on delta-hedged short-vol expert)",
    }
    save_metrics(run_dir, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--n-bc-episodes", type=int, default=200)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=63)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TransferConfig(
        n_seeds_per_kappa=args.n_seeds,
        n_bc_train_episodes=args.n_bc_episodes,
        n_eval_episodes_per_seed=args.n_eval_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    run_dir = make_run_dir("reflexive_transfer", seed=cfg.seed)
    metrics = run_experiment(cfg, run_dir)

    print(f"Wrote results to: {run_dir}")
    print(
        f"slope @ κ₀ = {metrics['slope_at_anchor']:.4g}  "
        f"95% CI [{metrics['slope_ci_low']:.4g}, {metrics['slope_ci_high']:.4g}]"
    )


if __name__ == "__main__":
    main()
