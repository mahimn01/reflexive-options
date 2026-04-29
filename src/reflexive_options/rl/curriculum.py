"""Regime curriculum bridge for the EWC-trained agent.

The vendored `train_ewc.py` (`reflexive_options.third_party.atlas.train_ewc`) holds
the elastic-weight-consolidation machinery; this module produces an ordered list
of `CurriculumStage`s that the EWC adapter can iterate over. Each stage is a
(name, simulator-factory, episode-budget) tuple. The factories defer simulator
construction so each stage gets a fresh stateful sim — important for
deterministic re-runs across seeds.

Regime parameters are *hard-coded* for v1 (calm / 2018 / 2020 / 2024). Phase 4
of `~/Documents/reflexivity-research/TODO.md` will replace these with
data-calibrated parameter sets once SPX data is acquired; the function
signature is stable across that swap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SimulatorProtocol,
    SurfaceGrid,
)

StageName = Literal["calm", "vol_event_2018", "vol_event_2020", "vol_event_2024"]


@dataclass(frozen=True)
class CurriculumStage:
    """One stage of the curriculum.

    Attributes:
        name: human-readable label, also used as the EWC task identifier.
        sim_factory: zero-arg callable returning a fresh `SimulatorProtocol`
            instance for this stage. Deferred construction keeps stages
            independent across seeds and across training restarts.
        n_episodes: episode budget for this stage.
    """

    name: str
    sim_factory: Callable[[], SimulatorProtocol]
    n_episodes: int


# ---------------------------------------------------------------------------
# Regime presets — hardcoded for v1, see module docstring.
# ---------------------------------------------------------------------------

# Calm: low vol-of-vol, low coupling. Heston-like behaviour.
_CALM_OVERRIDES: dict[str, float] = {
    "kappa": 2.0,
    "theta": 0.04,
    "xi": 0.20,
    "rho": -0.55,
    "v0": 0.04,
    "coupling": 1e-13,  # essentially off — well below the calibrated O(5e-12)
    "leverage": 0.0,
}

# Volmageddon (Feb 2018): elevated vol-of-vol, moderate coupling.
_VOL_2018_OVERRIDES: dict[str, float] = {
    "kappa": 2.5,
    "theta": 0.06,
    "xi": 0.55,
    "rho": -0.75,
    "v0": 0.06,
    "coupling": 5e-12,
    "leverage": 1.5,
}

# COVID crash (Mar 2020): high vol-of-vol, strong coupling.
_VOL_2020_OVERRIDES: dict[str, float] = {
    "kappa": 3.0,
    "theta": 0.10,
    "xi": 0.95,
    "rho": -0.85,
    "v0": 0.10,
    "coupling": 1e-11,
    "leverage": 3.0,
}

# Yen-carry unwind (Aug 2024): moderate-high vol, elevated coupling.
_VOL_2024_OVERRIDES: dict[str, float] = {
    "kappa": 2.7,
    "theta": 0.07,
    "xi": 0.65,
    "rho": -0.70,
    "v0": 0.07,
    "coupling": 7e-12,
    "leverage": 2.0,
}

_REGIME_OVERRIDES: dict[StageName, dict[str, float]] = {
    "calm": _CALM_OVERRIDES,
    "vol_event_2018": _VOL_2018_OVERRIDES,
    "vol_event_2020": _VOL_2020_OVERRIDES,
    "vol_event_2024": _VOL_2024_OVERRIDES,
}


def _params_for_stage(base_params: ReflexiveParams, stage: StageName) -> ReflexiveParams:
    """Apply per-stage overrides on top of `base_params`."""
    o = _REGIME_OVERRIDES[stage]
    new_base = HestonParams(
        kappa=o["kappa"],
        theta=o["theta"],
        xi=o["xi"],
        rho=o["rho"],
        v0=o["v0"],
    )
    return ReflexiveParams(
        base=new_base,
        coupling=o["coupling"],
        drift=base_params.drift,
        memory_decay=base_params.memory_decay,
        memory_intake=base_params.memory_intake,
        leverage=o["leverage"],
    )


def _default_oi_grid(grid: SurfaceGrid) -> OpenInterestGrid:
    """OI grid concentrated around ATM — placeholder until calibrated OI is available."""
    n_k, n_t = grid.shape
    contracts = np.zeros((n_k, n_t), dtype=np.float64)
    atm_idx = int(np.argmin(np.abs(grid.log_moneyness)))
    # Concentrate 10k contracts at ATM across maturities; the magnitudes are
    # placeholders — the agent's loss should scale these out, and Phase 4 will
    # replace with SPX OI snapshots.
    contracts[atm_idx, :] = 10_000.0
    return OpenInterestGrid(grid=grid, contracts_open=contracts)


def _default_surface_grid() -> SurfaceGrid:
    """7 maturities × 11 strikes, matching `paper/pre_registration.md` §4."""
    return SurfaceGrid(
        log_moneyness=np.linspace(-0.20, 0.20, 11, dtype=np.float64),
        maturities=np.array(
            [7, 14, 30, 60, 90, 180, 365], dtype=np.float64
        )
        / 365.0,
    )


def build_curriculum(
    base_params: ReflexiveParams,
    *,
    stages: list[StageName],
    n_episodes_per_stage: int = 100,
    initial_spot: float = 100.0,
    surface_grid: SurfaceGrid | None = None,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> list[CurriculumStage]:
    """Build the regime-curriculum stage list.

    Args:
        base_params: shared `ReflexiveParams` — only the fields *not* overridden
            by the regime preset (`drift`, `memory_decay`, `memory_intake`) are
            inherited; (κ, θ, ξ, ρ, v0, coupling, leverage) are replaced per stage.
        stages: ordered list of stage names; each must be one of the four
            presets keyed in `_REGIME_OVERRIDES`.
        n_episodes_per_stage: episode budget shared by every stage.
        initial_spot: S_0 for the simulators.
        surface_grid: grid for the dealer-gamma aggregator. Defaults to the
            7×11 grid from `pre_registration.md` §4.
        risk_free_rate: passed to the gamma aggregator.
        dividend_yield: passed to the gamma aggregator.

    Returns:
        List of `CurriculumStage` in the order requested.
    """
    if not stages:
        raise ValueError("stages must be non-empty")
    grid = surface_grid if surface_grid is not None else _default_surface_grid()

    out: list[CurriculumStage] = []
    for stage in stages:
        if stage not in _REGIME_OVERRIDES:
            raise ValueError(
                f"unknown stage {stage!r}; must be one of {sorted(_REGIME_OVERRIDES)}"
            )
        stage_params = _params_for_stage(base_params, stage)

        # Late-bound factory — capture stage_params/grid by default arg so each
        # closure doesn't share a mutable cell.
        def _factory(
            params: ReflexiveParams = stage_params,
            grid: SurfaceGrid = grid,
            spot0: float = initial_spot,
            r: float = risk_free_rate,
            q: float = dividend_yield,
        ) -> SimulatorProtocol:
            agg = GammaAggregator(
                oi_grid=_default_oi_grid(grid),
                risk_free_rate=r,
                dividend_yield=q,
            )
            return ReflexiveSimulator(
                params=params,
                gamma_aggregator=agg,
                initial_spot=spot0,
                surface_grid=grid,
            )

        out.append(
            CurriculumStage(
                name=stage,
                sim_factory=_factory,
                n_episodes=n_episodes_per_stage,
            )
        )
    return out


__all__ = ["CurriculumStage", "StageName", "build_curriculum"]
