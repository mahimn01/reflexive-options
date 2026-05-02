# reflexive-options — agent context

This is Mahimn's research codebase for the in-progress paper *Reflexivity in Options Markets* (target: NeurIPS GenAI in Finance Workshop 2026, ICAIF 2026 stretch).

## Sibling repos

- `~/Documents/reflexivity-research/` — reading, paper drafts, correspondence with Halperin/Itkin, technical briefs from research agents (`marketron_technical_brief.md`, `hopf_bifurcation_brief.md`, `dealer_gamma_brief.md`, `arbitrage_filter_brief.md`, `evaluation_framework_brief.md`, `atlas_import_surface.md`).
- `~/Documents/Dev/randomThings/trading_algo/` — production trading codebase. Source of vendored ATLAS (Mamba + PPO + BC + EWC, in `quant_core/models/atlas/`) and RAT (reflexivity meter, topology detector, in `rat/`).

## Vendoring discipline

The `third_party/atlas/` and `third_party/rat/` directories are vendored from `trading-algo`. Do not modify them in place except to (a) patch import paths for standalone use, (b) strip trading-domain code (broker, IBKR, strategy adapters). Adapters that bridge ATLAS to options-domain state/action/reward live in `src/reflexive_options/rl/atlas_adapter.py` — that's where domain-specific glue belongs.

When upstream `trading-algo` changes the ATLAS internals, re-vendor: `scripts/revendor_atlas.sh` (TODO).

## Code style

- Python 3.12, strict mypy, ruff lint + format.
- Numpy-first; reach for JAX or torch only where performance demands.
- Dataclasses over `dict` for any structured config.
- One concept per file — `simulator/reflexive.py` does the reflexive SDE, `simulator/gamma_aggregator.py` does $G(S, t)$, etc.
- Tests required for every module under `src/`. Aim for ≥85% coverage (hard CI gate).

## Tooling

Pinned to the versions in `docs/quality_research_brief.md` (April 2026):

| Tool | Version | Purpose |
| --- | --- | --- |
| mypy | 1.20.2 | Strict type checking. Runs in CI only (not pre-commit) — ML deps can't be reliably checked in an isolated venv. |
| ruff | 0.15.12 | Lint + format. Includes `S` security rules (bandit replacement). |
| pytest / pytest-cov | 8.4.2 / 7.0.0 | Test runner with branch coverage; 85% gate. |
| pre-commit | 4.3.0 | Local hook orchestration. |
| uv | ≥0.10 | Reproducible environment manager. `uv.lock` committed. |

Dev dependencies live under PEP 735 `[dependency-groups].dev` in `pyproject.toml`,
mirrored in `[project.optional-dependencies].dev` for plain-pip users. Lockfile
is `uv.lock` at the repo root — bit-identical resolution across machines and
CI per the Scientific Python reproducibility recommendation.

## Before pushing

Run the full CI gauntlet locally:

```bash
bash scripts/verify.sh
```

This runs `ruff check` → `ruff format --check` → `mypy src` → `pytest --cov-fail-under=85`
in the same order as `.github/workflows/ci.yml`. Uses `uv run` if uv is
installed, falls back to system Python otherwise. Fail-fast: stops at the
first error.

## Mathematical conventions

- **State**: $(S_t, v_t, G_t)$ — underlying spot, instantaneous variance, aggregate dealer gamma.
- **Coupling**: $\kappa$ is the feedback strength. $\kappa = 0$ ⇒ standard time-dep Heston. Literature priors: $\kappa \in [10^{-4}, 10^{-2}]$ per \$bn dealer gamma (GPP 2009).
- **Surface**: $\sigma(K, T)$ on a 7 maturities × 11 strikes grid around ATM. Strikes in log-moneyness, maturities in years.
- **Time**: 1-minute step in the simulator; surfaces sampled end-of-day.

## Pre-registration

`paper/pre_registration.md` is committed *before* empirical evaluation against real SPX data. Once data is acquired (Phase 0 of `~/Documents/reflexivity-research/TODO.md`), the analysis pipeline is run exactly as specified. Any deviation must be flagged in the paper.

The pre-reg's chain-of-custody anchor is `paper/pre_registration.md.ots` — an OpenTimestamps proof binding the file's SHA256 into the Bitcoin blockchain. Verify with:

```bash
uv run ots verify paper/pre_registration.md.ots
```

After any amendment to the locked pre-reg text, regenerate the proof (`uv run ots stamp paper/pre_registration.md`) and document the change in `paper/pre_registration_amendments.md`.

## Running things

```bash
pytest                                                          # full test suite
pytest tests/test_simulator.py                                  # one module
python -m reflexive_options.experiments.synthetic_replication  # reproduce Marketron figures
python -m reflexive_options.experiments.bifurcation_scan       # numerical Hopf
python -m reflexive_options.experiments.phase_diagram          # (κ, σ_v) phase scan
python -m reflexive_options.experiments.reflexive_transfer     # κ-sensitivity
```

## Reproducibility receipt

Every experiment's v0.1.0 outputs are recorded in `tests/repro/baseline_v0.1.0.json` with blake2b hashes. The regression test `test_reproducibility.py` re-runs each experiment and asserts metrics match. To intentionally update the baseline (after a science change):

```bash
bash scripts/generate_repro_baseline.sh
```

Commit the resulting `baseline_v0.1.0.json` change with a description of *why* the numbers moved.

The deterministic-bucket experiments (`bifurcation_scan`, `phase_diagram`, `synthetic_replication`) are locked at exact-equality (1e-12 absolute tolerance). The torch-trained `reflexive_transfer` is locked at distributional 5% relative tolerance across 5 seeds — flag any drift larger than that as a real regression.

The repro test is gated behind `@pytest.mark.slow` and runs in ~30 s. Skip locally with `CI_FAST=1 pytest`; in CI the test runs by default and there's a `SKIP_REPRO=1` escape hatch for genuine flakes.

## Releases

Versioned changes go in `CHANGELOG.md` (Keep-a-Changelog format, SemVer).
Tag the release once the entry lands on `main`:

```bash
git tag -a v0.1.0 -m "v0.1.0 — first tagged research release"
git push --tags
```

Convention: bump the patch version for doc-only or test-only changes,
the minor version for new simulator/baseline/theory artifacts, and the
major version only after empirical calibration lands (Phase 4). Each tag
must correspond to a `## [x.y.z] - YYYY-MM-DD` heading in `CHANGELOG.md`.

## What NOT to do

- Don't add a real-data path until the data acquisition phase is funded (UofT WRDS or $805 ALLSPX). The whole point of this repo is to be data-free until Phase 4.
- Don't gate on Itkin/Halperin replies — the simulator + baselines + RL infra + theory + pre-registration can all proceed in parallel with academic correspondence.
- Don't import broker / IBKR / strategy code from trading-algo. The vendor is strictly the ATLAS+RAT model code, nothing else.
