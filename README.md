# reflexive-options

[![release](https://img.shields.io/github/v/tag/mahimn01/reflexive-options?sort=semver&label=release)](https://github.com/mahimn01/reflexive-options/releases)
[![tests](https://img.shields.io/badge/tests-409%20passing-brightgreen)](#quality)
[![coverage](https://img.shields.io/badge/coverage-88.10%25-brightgreen)](#quality)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![paper](https://img.shields.io/badge/paper-PDF%20(30%20pp)-blue)](paper/main.pdf)

**Reflexive options market simulator with dealer-gamma feedback, four bifurcation theorems, RL-trained agents, and a pre-registered evaluation framework.**

In-progress research codebase for the working paper *Reflexivity in Options Markets* (Patel, 2026). Current release: **v0.3.3** (master 30 pp + ICAIF 6 pp + NeurIPS workshop 4-body variant).

## What's here

A full implementation of:

1. A **reflexive SDE simulator** for an underlying coupled to its own option market via a dealer-gamma feedback channel:
   $$\frac{dS}{S} = (\mu + \kappa \cdot G(S, t, v))\, dt + \sigma(S, t, v)\, dW_S$$
   where $G(S, t, v)$ aggregates net market-maker gamma exposure from the open-interest grid (Garleanu-Pedersen-Poteshman 2009 demand-pressure mapping) and $\kappa$ is the feedback coupling strength. When $\kappa = 0$, the system reduces to standard time-dependent Heston.

2. **Four non-reflexive baselines** for clean comparison:
   - Time-dependent Heston (5–10 piecewise-constant regimes)
   - Local-stochastic vol (LSV)
   - 3/2 stochastic vol (smile-shape robustness)
   - **Gamma-aware non-reflexive** baseline: state-symmetric to the reflexive simulator (agent observes $G_t$) but $G_t$ does not feed back into dynamics. Isolates state-richness from the feedback contribution.

3. **Hard arbitrage-free filter** on every simulated surface: convexity-in-strike, monotonicity-in-maturity, calendar-spread positivity, Lee's moment bounds.

4. **An RL training infrastructure** (Mamba state-space + cross-attention transformer, PPO + behavioral cloning + EWC + curriculum learning) vendored from the [`mahimn01/trading-algo`](https://github.com/mahimn01/trading-algo) ATLAS module.

5. **Four bifurcation-theoretic results** (synthesis of established machinery in a configuration not previously published in this combination — see `paper/related_work.md` §1 for the precedent comparison against Halperin–Itkin Marketron, Dai 2025, Brock–Hommes–Wagener, and He–Li–Zheng 2009):
   - **Theorem 1 (Hopf):** critical coupling $\kappa^\star$ at which endogenous limit cycles in volatility appear, with closed-form first Lyapunov coefficient $\ell_1$ for log-normal open-interest.
   - **Theorem 2 (BT-empty):** Bogdanov-Takens locus is empty in the canonical scan window; the model is structurally Hopf-only and cannot generate excitable spike-and-recovery dynamics from its autonomous skeleton alone.
   - **Theorem 3 (McKean-Vlasov correction):** $\kappa^\star_{\mathrm{MV}}/\kappa^\star_{\mathrm{single}} = \sqrt{1 + (\omega^\star \tau_G)^2}$, strictly multiplicative; single-dealer model is structurally biased toward over-predicting instability when dealer-hedging is slower than the Hopf period.
   - **Theorem 4 (Hawkes–SV criticality correspondence):** via the Bacry-Delattre-Hoffmann-Muzy + Jaisson-Rosenbaum diffusive near-critical limit, Hardiman 2013's critical branching ratio $n \approx 1$ is the literal analogue of the *real-eigenvalue (saddle-node)* stratum of the SV drift boundary; the model's *Hopf* threshold $\kappa^\star$ is a strictly stronger, oscillatory instability beyond any scalar branching ratio — the genuinely novel "unoccupied cell." A falsifiable spectral discriminator separates the two strata on synthetic ground truth (the earlier tautological $n_{\mathrm{SV}}$ "verification" was removed in pre-data amendment A10).
   - The reflexive simulator's stationary marginal density, contrasted analytically with Heston's known stationary distribution.

6. **An evaluation framework** (each ingredient borrowed; combination not previously published — see `paper/related_work.md` §§2–3 for the comparison against He–Li–Zheng 2025 NeurIPS, Ning et al. 2021/2024, VolGAN, FuNVol, Subbaswamy–Saria 2022, Packer 2018):
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

# Run the κ-sensitivity transfer experiment (data-free)
python -m reflexive_options.experiments.reflexive_transfer

# Numerical Hopf bifurcation scan (data-free)
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

## Manuscript

The full LaTeX paper lives in `paper/main.tex` with bibliography at `paper/references.bib`. To rebuild the PDF:

```bash
cd paper
make pdf
```

The Makefile runs `pdflatex` → `bibtex` → `pdflatex` → `pdflatex` (the standard four-pass cycle for cleveref + natbib cross-references). Output: `paper/main.pdf`. Other targets: `make clean`, `make watch` (latexmk live preview).

The arXiv submission metadata (subjects, MSC codes, license, comments) lives in `paper/arxiv_metadata.txt`.

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
| Theorem 1 (Hopf) + closed-form $\ell_1$ + symbolic ℓ_1 (Appendix A) | **derived + computed (closed-form-OI regime, §3.5): $\kappa^\star = 17.81$, $\omega^\star = 1.18$, $\ell_1 = -0.48$ (supercritical). Symbolic 7.8 KB rational verified against numerical to ~$10^{-13}$. Limit cycle past $\kappa^\star$ validated: $T = 10.561$ yr vs theory 10.977 yr (3.79%)** |
| Theorem 2 (BT-empty, §3.7) + Bautin curve with 6 anchors | **proved (closed-form $G_v < 0$ dominance argument); 71×97 scan confirms $\kappa_{\mathrm{SN}} \leq -1.31$ everywhere** |
| Theorem 3 (McKean-Vlasov correction, §3.8) | **derived + numerically validated: propagation-of-chaos slope -0.58 (theory -0.5); MV/single ratio 1.000277 at canonical $\theta_G = 50$/yr** |
| Theorem 4 (Hawkes–SV criticality correspondence, §3.9) | **repositioned (amendment A10): Hardiman $n\approx1$ = real-eigenvalue/saddle-node stratum; the model's Hopf is the strictly-stronger oscillatory "unoccupied cell". Falsifiable spectral discriminator (`hawkes_sv_bifurcation.py`) separates the strata with zero overlap on synthetic data. Tautological $n_{\mathrm{SV}}$ "1e-15 verification" removed** |
| Empirical $\|\Lambda\| \sim \|\rho\xi\|^B$ fit | **measured: $\hat B = 0.082$, 95% CI $[-0.010, 0.168]$ — refutes ELR $B = 2/3$ prediction at the trivial-equilibrium regime ($p \ll 0.01$, 13σ)** |
| Fokker-Planck stationary density vs Heston (analytical + MC) | **derived: H_tail confirmed, H_skew confirmed; H_bimod refuted on 1D marginal, *supported* on 2D PCA-projection at $\kappa = 1.05 \kappa^\star_{\mathrm{env}}$ ($p = 0.033$)** |
| $\kappa^\star$ robustness to OI misspecification (§3.6) | **elasticities $\eta_{\sigma_q} = -1.58$, $\eta_{\mu_q} = +703$; calibration tolerance for Phase 4: $\mu_q$ to $\pm 5$bp is binding** |
| H1 synthetic-pipeline validation (§5.4) | **demonstrated working: SW2(κ_0) < SW2(2κ_0) < SW2(Heston) with disjoint bootstrap CIs** |
| H4 detector power on Stuart-Landau positive control | **$\geq 80\%$ peak power at $T = 512$ for 8/9 $(\mu, \sigma)$ configurations; non-monotone in $T$** |
| Sliced-W2 sample-complexity | **$n_{\min} \approx 4{,}000$ windows for $\pm 10\%$ bootstrap CI half-width** |
| κ-sensitivity transfer experiment (BC-trained MLP, ~15-20 min/run) | **implemented + tested** |
| Marketron mechanism decomposition | **8/24 OOS shape-cell match (33.3%) at per-set tuned coupling; a priori long-horizon restricted subset 7/10 in-sample ($p = 0.172$)** |
| Pre-registration document | **locked + OpenTimestamps Bitcoin-anchored proof (`paper/pre_registration.md.ots`). A1–A7 amendments closed at commit `63078f5`; no further amendments permitted post-data-load** |
| Manuscript variants | **NeurIPS GenAI Finance Workshop (4-body) + ICAIF 2026 double-blind ACM sigconf (6 pp), both compiled clean** |
| Test suite | **409 passing, 88.10% branch coverage** [^cov] |
| CI (GitHub Actions) | **green on Python 3.12 / 3.13 / 3.14** |

[^cov]: Coverage measured by the most recent `bash scripts/verify.sh` run (88.10% as of v0.3.3); gated at ≥ 85% in CI via `[tool.coverage.report] fail_under = 85`.

**v0.3.3 shipped** (2026-05-14) — see [`CHANGELOG.md`](CHANGELOG.md) for the full release notes. Highlights:
- 4 theorems (Hopf + BT-empty + MV-Hopf + Hawkes-SV) with closed-form proofs / proof sketches.
- Closed-form symbolic $\ell_1$ as Appendix A (7.8 KB rational in 13 symbols, machine-verified).
- Empirical $|\Lambda| \sim |\rho\xi|^B$ scaling fit ($\hat B = 0.082$, refutes ELR).
- Limit cycle past $\kappa^\star$ numerically validated to 3.79%.
- H1 synthetic pipeline demonstrated end-to-end before empirical SPX target.
- 2D bimodality flip on $(\log S, v)$ joint PCA-projection.
- NeurIPS workshop + ICAIF manuscript variants ready.

## How to cite

If you use this software or paper, please cite:

```bibtex
@software{patel2026reflexive,
  author       = {Patel, Mahimn},
  title        = {Reflexivity in Options Markets:
                  A Stochastic-Volatility Model with Dealer-Gamma Feedback,
                  Hopf Bifurcation Calculus, and a Pre-Registered Evaluation Framework},
  year         = 2026,
  version      = {v0.3.3},
  url          = {https://github.com/mahimn01/reflexive-options},
  note         = {Pre-registered at commit 268c061 via OpenTimestamps proof}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is committed at the repo root and is recognised by GitHub's "Cite this repository" button.

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
mypy src                       # project-tuned strict mode (see pyproject.toml)
pytest --cov-fail-under=85     # branch coverage hard gate
```

- **Reproducibility receipt**: `tests/test_reproducibility.py` re-runs every experiment and asserts the v0.1.0 numbers reproduce within tolerance. Refresh with `bash scripts/generate_repro_baseline.sh`.

Pinned tooling versions live in `pyproject.toml`'s `[dependency-groups].dev`
(PEP 735) and the bit-identical resolution is captured in `uv.lock`.
`uv sync --locked` reproduces the exact environment a paper reviewer would see.

Pre-commit (`.pre-commit-config.yaml`) runs ruff + format + the standard
hygiene hooks; mypy runs in CI only — see `docs/quality_research_brief.md`
§5 for the rationale.

mypy is set to `strict = true` with a small set of project-level relaxations
(`disallow_any_unimported = false`, `warn_unused_ignores = false`, plus
per-module overrides for QuantLib / diptest / ripser / mystic / scipy and
the vendored `third_party.*` tree). The full justification — including the
PyTorch and scientific-Python precedents — is in
`docs/quality_research_brief.md` §3 and §7.

## Continuous Integration

CI (`.github/workflows/ci.yml`) runs on every push and pull request to
`main` against an Ubuntu x [3.12, 3.13, 3.14] matrix. Each job installs
the locked environment with `uv sync --locked --all-extras --group dev`,
then executes ruff check, ruff format --check, mypy, and pytest with
branch coverage. The pipeline blocks merge unless every job is green and
total coverage stays at or above 85% (`[tool.coverage.report] fail_under
= 85`). Coverage is uploaded to Codecov via the OIDC-based v5 action.

## Releases

Versioned change history lives in [CHANGELOG.md](CHANGELOG.md), formatted
per [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and tagged
under [SemVer](https://semver.org/spec/v2.0.0.html). The current release
is `0.3.0`.

## Reproducibility

Every figure in the paper is produced by a script in `src/reflexive_options/experiments/` with deterministic seeds. The pre-registration in `paper/pre_registration.md` is committed before empirical evaluation; subsequent results either confirm or refute the pre-registered hypotheses.

### Pre-registration timestamping (OpenTimestamps)

`paper/pre_registration.md.ots` is the OpenTimestamps proof binding the SHA256 of `paper/pre_registration.md` into the Bitcoin blockchain. To verify the timestamp on a fresh clone:

```bash
uv run ots verify paper/pre_registration.md.ots
```

To regenerate the proof after a substantive pre-reg amendment (each amendment must be flagged in `paper/pre_registration_amendments.md` first):

```bash
uv run ots stamp paper/pre_registration.md
```

If `opentimestamps-client` cannot be installed in a reviewer's environment, the chain-of-custody anchor falls back to the git tag `prereg-anchor-vX` whose commit hash is the immutable reference.

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
