# reflexive-options

**Reflexive options market simulator with dealer-gamma feedback, RL-trained agents, and analytical bifurcation results.**

In-progress research codebase for the working paper *Reflexivity in Options Markets* (Patel, 2026).

## What's here

A full implementation of:

1. A **reflexive SDE simulator** for an underlying coupled to its own option market via a dealer-gamma feedback channel:
   $$\frac{dS}{S} = (\mu + \kappa \cdot G(S, t, v))\, dt + \sigma(S, t, v)\, dW_S$$
   where $G(S, t, v)$ aggregates net market-maker gamma exposure from the open-interest grid (Garleanu-Pedersen-Poteshman 2009 demand-pressure mapping) and $\kappa$ is the feedback coupling strength. When $\kappa = 0$, the system reduces to standard time-dependent Heston.

2. **Three non-reflexive baselines** for clean comparison:
   - Time-dependent Heston (5–10 piecewise-constant regimes)
   - Local-stochastic vol (LSV)
   - 3/2 stochastic vol (smile-shape robustness)
   - **Gamma-aware non-reflexive** baseline: state-symmetric to the reflexive simulator (agent observes $G_t$) but $G_t$ does not feed back into dynamics. Isolates state-richness from the feedback contribution.

3. **Hard arbitrage-free filter** on every simulated surface: convexity-in-strike, monotonicity-in-maturity, calendar-spread positivity, Lee's moment bounds.

4. **An RL training infrastructure** (Mamba state-space + cross-attention transformer, PPO + behavioral cloning + EWC + curriculum learning) vendored from the [`mahimn01/trading-algo`](https://github.com/mahimn01/trading-algo) ATLAS module.

5. **Two novel analytical results**:
   - A Hopf bifurcation theorem characterizing the critical coupling $\kappa^*$ at which endogenous limit cycles in volatility appear.
   - The reflexive simulator's stationary marginal density, contrasted analytically with Heston's known stationary distribution.

6. **A novel evaluation framework**:
   - Sliced Wasserstein-2 distance over arbitrage-free IV surface distributions.
   - $\kappa$-sensitivity curves: train an agent at $\kappa = \kappa_0$, deploy across $\kappa \in [0, 2\kappa_0]$, slope-of-degradation as a quantitative measure of reflexivity-importance.

7. **Pre-registered hypotheses** (commit-anchored) in `paper/pre_registration.md`.

## Quick start

```bash
# Reproducible install (recommended): uv resolves from uv.lock for bit-identical environments
uv sync --locked --all-extras --group dev

# Or, plain pip (no lockfile): editable install with dev extras
pip install -e ".[dev,calibration]"

# Run the test suite
pytest

# Reproduce Marketron paper figures from published parameters
python -m reflexive_options.experiments.synthetic_replication

# Generate the (κ, σ_v) phase diagram (data-free)
python -m reflexive_options.experiments.phase_diagram

# Run the κ-sensitivity transfer experiment (data-free, novel)
python -m reflexive_options.experiments.reflexive_transfer

# Numerical Hopf bifurcation scan (data-free, novel)
python -m reflexive_options.experiments.bifurcation_scan
```

## Project structure

```
src/reflexive_options/
├── simulator/         # The contribution: reflexive SDE, dealer-gamma aggregator, integrators, stability
├── baselines/         # Time-dep Heston, LSV, 3/2 SV, gamma-aware non-reflexive
├── surface/           # IV surface generator + arbitrage-free filter + parquet I/O
├── rl/                # Gymnasium env + state/action/reward + curriculum
├── theory/            # Hopf bifurcation, Fokker-Planck stationary density, sensitivity
├── experiments/       # Reproducible experiment scripts (one per paper figure)
└── third_party/       # Vendored ATLAS (Mamba+PPO+BC+EWC) and RAT (reflexivity meter, topology) from trading-algo

paper/
├── theory.md             # Analytical results writeup
├── pre_registration.md   # Commit-anchored hypothesis pre-registration
├── notation.md           # Canonical symbol table
└── figures/

tests/                    # pytest suite
notebooks/                # Tutorial walkthroughs
```

## Status

| Component | Status |
|-----------|--------|
| Reflexive 3D simulator (S, v, z) + dealer-gamma aggregator | **implemented + tested** |
| Time-dep Heston baseline (QuantLib analytic IV) | **implemented + tested** |
| LSV polynomial baseline | **implemented + tested** |
| 3/2 SV baseline | **implemented + tested** |
| Gamma-aware non-reflexive (clean ablation) | **implemented + tested** |
| Surface generator + arbitrage filter (~85k surfaces/sec) | **implemented + tested** |
| ATLAS vendored (Mamba + BC + EWC + RAT) | **vendored (~3,700 LOC, 8 smoke tests pass)** |
| Gymnasium RL env + state/action/reward + curriculum | **implemented + tested** |
| Hopf bifurcation analysis (Theorem 1 + numerical $\ell_1$ + stochastic shift $\Lambda$) | **derived + computed: $\kappa^* \approx 0.8964$, $\omega^* \approx 0.5724$, $\ell_1 \approx -0.025$ (supercritical), $\Lambda \approx +0.0185$** |
| Fokker-Planck stationary density vs Heston (analytical + MC) | **derived: H_tail ✓, H_skew ✓, H_bimod ✗ (honest finding documented)** |
| κ-sensitivity transfer experiment (BC-trained MLP, ~15-20 min/run) | **implemented + tested** |
| Marketron replication infrastructure | **implemented + tested (0 cells hit, mechanism mismatch — documented)** |
| Pre-registration document | **drafted (2,624 words, commit-anchored)** |
| Test suite | **116/116 passing** |
| CI (GitHub Actions) | **wired** |

**Known open items** (post-v1):
- Joint VIX/SPX simulation (Marketron explicitly fails this; we don't address it in v1).
- Vectorized BS-gamma over paths in `simulator.reflexive._g_vectorized` (~10× speedup possible).
- `train_ppo` from ATLAS not vendored — used `train_bc` for the smoketest experiments. Production PPO loop is the natural Phase 4 follow-up.
- Empirical calibration to real SPX surfaces (gated on WRDS / ALLSPX data acquisition — Phase 4 of the master TODO).

Phase 0 of [TODO.md](https://github.com/mahimn01/reflexivity-research/blob/main/TODO.md) (data acquisition) is the *only* part that requires real SPX data. Everything in this repo is data-free and reproducible from synthetic priors and published parameter sets.

## Quality

The repo's CI gauntlet (mirrored locally by `bash scripts/verify.sh`):

```text
ruff check src tests           # lint + flake8-bandit (S) security rules
ruff format --check src tests  # formatter conformance
mypy src                       # strict type check (Python 3.12 / 3.13 / 3.14 matrix in CI)
pytest --cov-fail-under=80     # branch coverage hard gate
```

Pinned tooling versions live in `pyproject.toml`'s `[dependency-groups].dev`
(PEP 735) and the bit-identical resolution is captured in `uv.lock`.
`uv sync --locked` reproduces the exact environment a paper reviewer would see.

Pre-commit (`.pre-commit-config.yaml`) runs ruff + format + the standard
hygiene hooks; mypy runs in CI only — see `docs/quality_research_brief.md`
§5 for the rationale.

## Reproducibility

Every figure in the paper is produced by a script in `experiments/` with deterministic seeds. The pre-registration in `paper/pre_registration.md` is committed before empirical evaluation; subsequent results either confirm or refute the pre-registered hypotheses.

## Citation

```bibtex
@unpublished{patel2026reflexivity,
  author = {Patel, Mahimn},
  title  = {Reflexivity in Options Markets: Dealer-Gamma Feedback and Reinforcement-Learned Hedging Policies},
  year   = {2026},
  note   = {Working paper, in preparation},
  url    = {https://github.com/mahimn01/reflexive-options}
}
```

## License

MIT — see [LICENSE](LICENSE).

Vendored ATLAS / RAT modules under `src/reflexive_options/third_party/` derive from [`mahimn01/trading-algo`](https://github.com/mahimn01/trading-algo) (also MIT). See [NOTICE](NOTICE) for attribution.
