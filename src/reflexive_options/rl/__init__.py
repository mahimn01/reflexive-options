"""RL environment + state/action/reward + curriculum for options-RL.

Wraps any `SimulatorProtocol` (reflexive or baseline) in a gymnasium.Env so the
same agent code trains across every simulator family — the architectural
commitment behind the κ-sensitivity experiment in `paper/pre_registration.md`.
"""

from reflexive_options.rl.actions import (
    ActionConfig,
    apply_action,
    make_action_space,
)
from reflexive_options.rl.curriculum import (
    CurriculumStage,
    StageName,
    build_curriculum,
)
from reflexive_options.rl.env import OptionsHedgeEnv, price_option_position
from reflexive_options.rl.rewards import RewardConfig, compute_reward, rolling_sharpe
from reflexive_options.rl.state import StateConfig, build_observation

__all__ = [
    "ActionConfig",
    "CurriculumStage",
    "OptionsHedgeEnv",
    "RewardConfig",
    "StageName",
    "StateConfig",
    "apply_action",
    "build_curriculum",
    "build_observation",
    "compute_reward",
    "make_action_space",
    "price_option_position",
    "rolling_sharpe",
]
