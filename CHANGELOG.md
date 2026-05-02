# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-22

### Added

**Manuscript**
- `paper/main.tex` — 9-page (body) LaTeX manuscript per
  `paper/MANUSCRIPT_SKELETON.md` 8-page workshop layout, with full proofs
  of Theorem 1 (Hopf bifurcation), the closed-form $\kappa^\star$ and
  $\ell_1$ for log-normal open-interest in moneyness, the
  $(\xi,\rho,\sigma_v)$ phase diagram, the pre-registered evaluation
  framework, and the mechanism decomposition vs. Marketron. Compiles
  cleanly under TeXLive 2024+ via `make pdf`.
- `paper/references.bib` — 62 BibTeX entries; 61 cited in the manuscript,
  one (`baxendale1994`) included for the Khasminskii-stochastic-Hopf
  citation chain.
- `paper/Makefile` — `make pdf` (pdflatex + bibtex + pdflatex × 2),
  `make clean`, `make watch` targets.
- `paper/arxiv_metadata.txt` — arXiv submission metadata: title, author,
  primary `q-fin.MF` + cross-list `q-fin.CP, q-fin.PR, q-fin.ST, math.DS`,
  MSC 2020 codes `91G80, 37G15, 60H10, 91G60, 93E20`, CC-BY-4.0 license.
- `paper/pre_registration_amendments.md` — pre-data amendments A1–A7
  (H4 Welch window adaptation, dual-signal H4 on $|r_t|$ and $\widehat{v}_t$,
  IAAFT surrogate null replacing iid permutation, GP-posterior slope CI
  for κ-sensitivity, and TOST equivalence on the dimensionless elasticity).
- `paper/threats_to_validity.md`, `paper/related_work.md`,
  `paper/mechanism_decomposition.md` (three-table mechanism analysis),
  `paper/notation.md`, `paper/abstract.md`, `paper/SUBMISSION_READINESS.md`,
  and `paper/MANUSCRIPT_SKELETON.md`.

**Mechanism decomposition + simulator hardening**
- Mechanism-decomposition reporter in `synthetic_replication.py`: every
  Marketron-vs-reflexive cell now carries a `mechanism_class` ∈
  {`shape_target`, `level_artifact`, `calibration_artifact`}, plus
  `sign_match`, `order_of_magnitude_match`, and the legacy `within_8pct`
  flag. The headline gate is the shape-match rate (≥30% sign-agreement
  on `shape_target` cells) — replaces the prior "0 cells hit" report.
- `marketron_tuning.py` — coarse 5D grid search over (κ, γ, T_eff, μ_q, σ_q)
  picking the best reflexive overrides per Marketron parameter set; writes
  `runs/marketron_tuning/<ts>/grid_results.parquet` and `best_overrides.json`,
  consumed by `synthetic_replication.load_tuned_overrides`.
- CLI exit code on `synthetic_replication.py`: 0 if shape-match rate ≥ 30%,
  1 otherwise — enforces the headline number on every CI run.

**Theory**
- Closed-form first Lyapunov coefficient $\ell_1$ for log-normal OI in
  moneyness (`theory/bifurcation.lyapunov_coefficient_lognormal_oi`),
  matching the FD-tensor pipeline to <0.6% relative; canonical regime
  $\kappa^\star = 17.81$, $\omega^\star = 1.18$, $\ell_1 = -0.48$
  (supercritical).
- Closed-form Hopf threshold $\kappa^\star$ as the smallest positive root
  of the §4.3.2 quadratic.
- $(\sigma_q, \gamma)$ closed-form phase boundary rendered to
  `paper/figures/ell1_phase_boundary.pdf`.
- 4D phase scan `experiments/hopf_phase_scan_4d.py` rendered to
  `paper/figures/hopf_phase_diagram.pdf`.
- H4 spectral-peak detector with adaptive Welch window, dual-signal
  ($|r_t|$ and realised-variance proxy $\widehat{v}_t$), IAAFT surrogate
  null per Schreiber & Schmitz (1996); rendered to
  `paper/figures/h4_detector_power.pdf`.
- GP-posterior slope CI for the κ-sensitivity protocol
  (`theory/sensitivity.py`), replacing the spline-derivative + iid
  bootstrap that under-covered out-of-span function classes.

### Changed

- `synthetic_replication.py` default `n_steps` raised from 252 → 756 to cover
  Marketron's 3-year horizon.
- Test suite expanded from 252 → **329 tests**; coverage 85.12% → **89.05%**
  (gate at 85% in CI).
- Pre-registration amendment set is now A1–A7 (was A1–A4 at v0.2.x).

## [0.1.0] - 2026-04-30

First tagged research release. Everything is data-free and reproducible from
synthetic priors and published parameter sets; empirical SPX calibration is
gated on Phase 4 of the master TODO.

### Added

**Simulator**
- Reflexive 3D SDE simulator over $(S_t, v_t, z_t)$ with a dealer-gamma
  feedback channel $G(S, t, v)$ following the Garleanu-Pedersen-Poteshman
  (2009) demand-pressure mapping; reduces to time-dependent Heston at
  $\kappa = 0$.
- Three non-reflexive baselines for clean comparison: time-dependent Heston
  (5-10 piecewise-constant regimes, QuantLib analytic IV), local-stochastic
  vol (LSV polynomial), 3/2 stochastic vol, plus a gamma-aware non-reflexive
  baseline that observes $G_t$ without feeding it back into dynamics.
- Hard arbitrage-free filter on every simulated surface: convexity-in-strike,
  monotonicity-in-maturity, calendar-spread positivity, Lee moment bounds.
- Surface generator + parquet I/O at roughly 85k surfaces/sec.

**Theory**
- Hopf bifurcation theorem (paper/theory.md, Theorem 1) with computed
  threshold $\kappa^* \approx 0.8964$, angular frequency
  $\omega^* \approx 0.5724$, first Lyapunov coefficient
  $\ell_1 \approx -0.025$ (supercritical), and stochastic shift
  $\Lambda \approx +0.0185$.
- Fokker-Planck stationary density for the reflexive simulator contrasted
  analytically and via Monte Carlo with Heston's known stationary
  distribution (H_tail confirmed, H_skew confirmed, H_bimod refuted and
  documented).

**RL infrastructure**
- Vendored ATLAS module from `mahimn01/trading-algo` (Mamba state-space +
  cross-attention transformer, BC + EWC + RAT reflexivity meter and topology
  detector), roughly 3,700 LOC with eight smoke tests.
- Gymnasium env, state/action/reward design, and curriculum schedule for
  reflexivity-aware policy training.
- $\kappa$-sensitivity transfer experiment: train a BC-trained MLP at
  $\kappa = \kappa_0$ and deploy across $\kappa \in [0, 2\kappa_0]$ to
  measure slope-of-degradation as a quantitative reflexivity-importance
  scalar.

**Evaluation**
- Sliced Wasserstein-2 distance over arbitrage-free IV surface
  distributions.
- Marketron replication infrastructure (mechanism mismatch documented:
  zero cells hit, honest finding preserved).

**Pre-registration**
- `paper/pre_registration.md` (2,624 words) committed and anchored to
  initial commit hash `268c061` before any empirical evaluation.

**Docs**
- `paper/notation.md` canonical symbol table.
- `docs/quality_research_brief.md` documenting tooling decisions.
- `CHANGELOG.md` (this file).

### Changed

**Quality stack**
- Pinned developer tooling under PEP 735 `[dependency-groups].dev` in
  `pyproject.toml`: ruff 0.15.12, mypy 1.20.2, pytest 8.4.2, pytest-cov
  7.0.0, pre-commit 4.3.0, ipykernel 6.30.1, jupyter 1.1.1.
- `uv.lock` committed for bit-identical environments via
  `uv sync --locked --all-extras --group dev`.
- Pre-commit (`.pre-commit-config.yaml`) wires ruff check, ruff format, and
  the standard hygiene hooks; mypy runs in CI only (rationale in
  `docs/quality_research_brief.md` §5).
- Multi-Python CI matrix: Ubuntu x [3.12, 3.13, 3.14] runs ruff check,
  ruff format --check, mypy, pytest with branch coverage and an 80%
  fail-under gate.
- Project-tuned strict mypy (`strict = true` with
  `disallow_any_unimported = false`, `warn_unused_ignores = false`,
  `disallow_subclassing_any` relaxed via overrides) per the conventions
  PyTorch and the wider scientific Python ecosystem use for ML/scientific
  codebases.
- `scripts/verify.sh` runs the full CI gauntlet locally in the same order,
  fail-fast.

### Fixed

- Lint surface upgraded to ruff 0.15.12 with rule families
  `E,F,W,I,N,UP,B,SIM,RUF,S` (S = flake8-bandit security checks; bandit
  itself dropped from the dep graph).
- All mypy errors under the project-tuned strict configuration; CI no
  longer carries `continue-on-error`.
- `astral-sh/setup-uv` pinned to `v8.1.0` (no major-version tag is
  published upstream).
- Pre-registration document anchored to the initial-commit SHA so the
  hypothesis set cannot be silently revised.

### Security

- ruff `S` rule group enabled (flake8-bandit successor at roughly 25x the
  speed); narrow ignores for `S101`, `S301`, `S311`, `S403` documented in
  `pyproject.toml` with rationale.
- Vendored third-party code under `src/reflexive_options/third_party/`
  excluded from coverage and mypy strict checks per the vendoring
  discipline in `CLAUDE.md`.

<!-- TODO: link the version once the v0.1.0 git tag is pushed. -->
[0.1.0]: https://github.com/mahimn01/reflexive-options/releases/tag/v0.1.0
