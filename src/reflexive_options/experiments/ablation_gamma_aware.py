"""The clean ablation: gamma-aware non-reflexive baseline vs reflexive simulator.

Trains an agent in `GammaAwareSimulator` (state-symmetric to reflexive,
agent observes G_t and z_t, but G_t and z_t do NOT feed into dynamics).
Compares performance to an agent trained in the reflexive simulator on
the same surface-distribution evaluation.

This isolates the contribution of the FEEDBACK CHANNEL itself from the
contribution of having a richer state representation.

Run: python -m reflexive_options.experiments.ablation_gamma_aware
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from reflexive_options.experiments._common import (
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)


@dataclass(frozen=True)
class AblationConfig:
    n_seeds: int = 20
    n_eval_episodes: int = 100
    seed: int = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = AblationConfig(n_seeds=args.n_seeds, seed=args.seed)
    run_dir = make_run_dir("ablation_gamma_aware", seed=cfg.seed)
    save_config(run_dir, cfg)

    with timed("ablation_run"):
        # TODO(post-implementation, blocked on tasks #13, #14, #17):
        #   1. Build ReflexiveSimulator + GammaAwareSimulator with matched params.
        #   2. For seed in 1..n_seeds: train PPO agent in each, save policies.
        #   3. Evaluate both on the SAME held-out surface-window eval set.
        #   4. Compute sliced-W2 to a reference SPX-like surface distribution,
        #      P&L Sharpe, and the κ-sensitivity slope at κ₀.
        #   5. Report ΔW2, ΔSharpe, Δslope between the two trained agents.
        results = {
            "ablation": "gamma_aware_vs_reflexive",
            "status": "stub — wire up after #13, #14, #17 land",
        }

    save_metrics(run_dir, results)
    print(f"Wrote results to: {run_dir}")


if __name__ == "__main__":
    main()
