"""Λ stochastic-Hopf shift at the §4.2 canonical regime — reproducibility runner.

Constructs a `ReflexiveSimulator` at the §4.2 dimensionless regime (memory
parameters only; the OI grid is held trivial so the price-channel feedback
vanishes from the Jacobian and we measure the bare Heston-with-memory
linearisation), plus an SPX-representative (ξ, ρ) variant, runs the
Khasminskii sphere-process Λ estimator (`compute_lambda_correction`), and
persists the result alongside config + metrics.

The output magnitude is $|\\Lambda| \\sim 10^{-3}$ at the locked seed and
path budget; signs are configuration-dependent and the paper text reports
only the magnitude. The earlier published value (+1.85×10⁻², appearing in
v0.2.x and earlier README / theory.md / abstract drafts) was a stale legacy
estimate from a different OI configuration; the v0.3.1 paper text is
amended to use the magnitude bound from this script.

Run:
    python -m reflexive_options.experiments.lambda_correction_canonical
    python -m reflexive_options.experiments.lambda_correction_canonical --quick   # CI

Outputs:
    runs/lambda_correction_canonical/<timestamp>/{config,metrics}.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np

from reflexive_options.experiments._common import (
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.theory.bifurcation import compute_lambda_correction
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SurfaceGrid,
)

# Two regimes from paper/theory.md §4.2:
#   "canonical-§4.2"  — the dimensionless Hopf-exhibiting regime where κ* ≈ 0.8964
#   "spx-representative" — same memory channel but ξ = 0.3, ρ = -0.7 (SPX-realistic)
# Both share the §4.2 dimensionless deterministic-skeleton parameters; they
# differ only in the stochastic-channel coefficients (ξ, ρ) that drive Λ.


@dataclass(frozen=True)
class LambdaCanonicalConfig:
    """Configuration for the Λ canonical-regime evaluation."""

    # §4.2 dimensionless parameters (deterministic skeleton)
    kappa_v: float = 2.0
    theta_v: float = 0.04
    alpha: float = 0.5  # memory decay (multi-day)
    beta: float = 1.0
    gamma: float = 0.5  # leverage
    coupling_at_kappa_star: float = 0.8964  # the §4.2 Hopf threshold

    # Khasminskii sphere-process estimator parameters
    epsilon_low: float = 0.05
    epsilon_high: float = 0.20
    n_paths: int = 2_000  # mid-fidelity (paper §4.2 used 200; we widen for stability)
    n_steps: int = 5_000
    dt: float = 1e-2
    renorm_every: int = 50
    seed: int = 20260422  # locked


@dataclass(frozen=True)
class RegimeSpec:
    """A single (xi, rho) noise-channel configuration for the Λ estimator."""

    name: str
    xi: float
    rho: float


_REGIMES: tuple[RegimeSpec, ...] = (
    RegimeSpec(name="canonical_section_4_2", xi=0.3, rho=0.0),
    RegimeSpec(name="spx_representative", xi=0.3, rho=-0.7),
)


def _build_simulator(cfg: LambdaCanonicalConfig, regime: RegimeSpec) -> ReflexiveSimulator:
    """Build a `ReflexiveSimulator` at the §4.2 regime with the given (ξ, ρ).

    OI grid is set to all-zeros so the gamma aggregator returns G ≡ 0 — the
    Λ correction depends only on the linearised Jacobian and noise structure
    at the equilibrium, not on the OI surface itself. The §4.2 numerical
    example fixes the partials (G_x, G_v, G_z) explicitly; the Λ extraction
    here uses the simulator's own drift Jacobian, which at the trivial
    equilibrium reduces to the bare Heston-with-memory linearisation.
    """
    grid = SurfaceGrid(
        log_moneyness=np.array([-0.05, 0.0, 0.05], dtype=np.float64),
        maturities=np.array([30 / 365.25, 90 / 365.25], dtype=np.float64),
    )
    contracts = np.zeros(grid.shape, dtype=np.float64)
    oi = OpenInterestGrid(grid=grid, contracts_open=contracts)
    aggregator = GammaAggregator(oi_grid=oi, risk_free_rate=0.0)
    params = ReflexiveParams(
        base=HestonParams(
            kappa=cfg.kappa_v,
            theta=cfg.theta_v,
            xi=regime.xi,
            rho=regime.rho,
            v0=cfg.theta_v,
        ),
        coupling=cfg.coupling_at_kappa_star,
        drift=0.0,
        memory_decay=cfg.alpha,
        memory_intake=cfg.beta,
        leverage=cfg.gamma,
    )
    return ReflexiveSimulator(params=params, gamma_aggregator=aggregator, initial_spot=100.0)


def run(cfg: LambdaCanonicalConfig) -> dict[str, float]:
    """Compute Λ at every regime in `_REGIMES`. Returns dict regime → Λ."""
    out: dict[str, float] = {}
    for regime in _REGIMES:
        sim = _build_simulator(cfg, regime)
        with timed(f"lambda_{regime.name}"):
            Lambda = compute_lambda_correction(
                sim,
                kappa=cfg.coupling_at_kappa_star,
                epsilon_low=cfg.epsilon_low,
                epsilon_high=cfg.epsilon_high,
                n_paths=cfg.n_paths,
                n_steps=cfg.n_steps,
                dt=cfg.dt,
                renorm_every=cfg.renorm_every,
                seed=cfg.seed,
            )
        out[regime.name] = float(Lambda)
        print(f"  {regime.name}: Λ = {Lambda:+.4e}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduce n_paths × n_steps for CI")
    args = parser.parse_args()

    if args.quick:
        cfg = LambdaCanonicalConfig(n_paths=200, n_steps=1_000)
    else:
        cfg = LambdaCanonicalConfig()

    run_dir = make_run_dir("lambda_correction_canonical")
    save_config(run_dir, cfg)

    lambdas = run(cfg)

    metrics = {
        "regimes": [
            {
                "name": r.name,
                "xi": r.xi,
                "rho": r.rho,
                "lambda": lambdas[r.name],
            }
            for r in _REGIMES
        ],
        **{f"lambda_{name}": value for name, value in lambdas.items()},
        "config": asdict(cfg),
    }
    save_metrics(run_dir, metrics)
    print(f"Wrote results to: {run_dir}")


if __name__ == "__main__":
    main()
