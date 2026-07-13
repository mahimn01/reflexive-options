# reflexive-options — agent context

This is Mahimn's research codebase for the in-progress paper *Gamma-Shaped
Dealer-Book Pressure and Endogenous Volatility Cycles* (v0.4.1 centered-model reconstruction).
`paper/main.tex`, `src/reflexive_options/theory/centered_model.py`, and
Amendments A13--A16 are authoritative. v0.3 theory/experiment files are historical
unless explicitly imported by the current paper.

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

- **Current state**: $(X_t,v_t,\chi_t)$, where $X_t=\log(S_t/F_t)$ is a local
  detrended deviation.
- **Measure**: physical measure $\mathbb P$.
- **Coupling**: $\kappa$ has yr$^{-1}$ units because the current book
  functional is centered and dimensionless. Never compare it directly with a
  per-dollar empirical GEX coefficient.
- **Variance drift**: $\kappa_v(\theta_v-v)+\gamma v\chi$; do not restore the
  superseded additive $\gamma\chi$ term.
- **Book**: theory uses a latent signed dealer-position density. Public OI is
  unsigned and cannot identify dealer sign.
- **Equilibrium**: $(0,\theta_v,0)$ for every $\kappa$.
- **Scope**: local deterministic Hopf only. No global stability, stochastic
  shift, Hawkes equivalence, mean-field threshold, or stationary-tail claim.

## Pre-registration

`paper/pre_registration.md` is the preserved historical registration.
`paper/pre_registration_amendments.md` Amendments A13--A14, supplemented by
`paper/pre_registration_amendment_a15.md` and
`paper/pre_registration_amendment_a16.md`, form the operative primary protocol and
supersede A9/A11 where inconsistent. They were added before WRDS access. Do not
rewrite a historical file; new changes require a disclosed pre-extraction amendment
and their own proof.

The A13-only bytes and receipt are preserved as
`paper/pre_registration_amendments.md.a13{,.ots}`. The preserved A13--A14 file is
paired with `paper/pre_registration_amendments.md.ots`; A15 is paired with
`paper/pre_registration_amendment_a15.md.ots`; A16 is paired with
`paper/pre_registration_amendment_a16.md.ots`. Treat all registration documents
and receipts as immutable provenance artifacts.

The pre-reg's chain-of-custody anchor is `paper/pre_registration.md.ots` — an OpenTimestamps proof binding the file's SHA256 into the Bitcoin blockchain. Verify with:

```bash
uv run ots verify paper/pre_registration.md.ots
```

Do not overwrite either existing proof. Stamp the amended record as a separate
artifact and document its exact hash.

## Running things

```bash
pytest                                                          # full test suite
pytest tests/test_simulator.py                                  # one module
python -m reflexive_options.experiments.centered_hopf_validation
pytest tests/test_centered_model.py tests/test_oi_proxy_protocol.py
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
