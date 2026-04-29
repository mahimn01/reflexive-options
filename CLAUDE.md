# reflexive-options — agent context

This is Mahimn's research codebase for the in-progress paper *Reflexivity in Options Markets* (target: NeurIPS GenAI in Finance Workshop 2026, ICAIF 2026 stretch).

## Sibling repos

- `~/Documents/reflexivity-research/` — reading, paper drafts, correspondence with Halperin/Itkin, technical briefs from research agents (`marketron_technical_brief.md`, `hopf_bifurcation_brief.md`, `dealer_gamma_brief.md`, `arbitrage_filter_brief.md`, `evaluation_framework_brief.md`, `atlas_import_surface.md`).
- `~/Documents/Dev/randomThings/trading_algo/` — production trading codebase. Source of vendored ATLAS (Mamba + PPO + BC + EWC, in `quant_core/models/atlas/`) and RAT (reflexivity meter, topology detector, in `rat/`).

## Vendoring discipline

The `third_party/atlas/` and `third_party/rat/` directories are vendored from `trading-algo`. Do not modify them in place except to (a) patch import paths for standalone use, (b) strip trading-domain code (broker, IBKR, strategy adapters). Adapters that bridge ATLAS to options-domain state/action/reward live in `src/reflexive_options/rl/atlas_adapter.py` — that's where domain-specific glue belongs.

When upstream `trading-algo` changes the ATLAS internals, re-vendor: `scripts/revendor_atlas.sh` (TODO).

## Code style

- Python 3.12, strict mypy, ruff lint.
- Numpy-first; reach for JAX or torch only where performance demands.
- Dataclasses over `dict` for any structured config.
- One concept per file — `simulator/reflexive.py` does the reflexive SDE, `simulator/gamma_aggregator.py` does $G(S, t)$, etc.
- Tests required for every module under `src/`. Aim for ≥80% coverage.

## Mathematical conventions

- **State**: $(S_t, v_t, G_t)$ — underlying spot, instantaneous variance, aggregate dealer gamma.
- **Coupling**: $\kappa$ is the feedback strength. $\kappa = 0$ ⇒ standard time-dep Heston. Literature priors: $\kappa \in [10^{-4}, 10^{-2}]$ per \$bn dealer gamma (GPP 2009).
- **Surface**: $\sigma(K, T)$ on a 7 maturities × 11 strikes grid around ATM. Strikes in log-moneyness, maturities in years.
- **Time**: 1-minute step in the simulator; surfaces sampled end-of-day.

## Pre-registration

`paper/pre_registration.md` is committed *before* empirical evaluation against real SPX data. Once data is acquired (Phase 0 of `~/Documents/reflexivity-research/TODO.md`), the analysis pipeline is run exactly as specified. Any deviation must be flagged in the paper.

## Running things

```bash
pytest                                                          # full test suite
pytest tests/test_simulator.py                                  # one module
python -m reflexive_options.experiments.synthetic_replication  # reproduce Marketron figures
python -m reflexive_options.experiments.bifurcation_scan       # numerical Hopf
python -m reflexive_options.experiments.phase_diagram          # (κ, σ_v) phase scan
python -m reflexive_options.experiments.reflexive_transfer     # κ-sensitivity (novel)
```

## What NOT to do

- Don't add a real-data path until the data acquisition phase is funded (UofT WRDS or $805 ALLSPX). The whole point of this repo is to be data-free until Phase 4.
- Don't gate on Itkin/Halperin replies — the simulator + baselines + RL infra + theory + pre-registration can all proceed in parallel with academic correspondence.
- Don't import broker / IBKR / strategy code from trading-algo. The vendor is strictly the ATLAS+RAT model code, nothing else.
