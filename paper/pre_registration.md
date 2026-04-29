# Pre-registration — *Reflexivity in Options Markets*

**Author.** Mahimn Patel (Patel 2026)
**Target venues.** NeurIPS GenAI in Finance Workshop 2026 (primary); ICAIF 2026 (stretch).
**Status.** Locked prior to any empirical evaluation against real SPX data. Theory, simulator, baselines, RL stack, and evaluation pipeline are implemented; this document fixes the analysis plan before SPX data acquisition (UofT WRDS / OptionMetrics IvyDB US, or commercial fallback historicaloptiondata.com ALLSPX, $805).
**Template.** Adapted from Camerer et al. (2018, *Nature Human Behaviour*) for computational finance.

---

## 1. Background and motivation

The paper introduces a 3D reflexive stochastic-volatility SDE in $(S_t, v_t, z_t)$ in which dealer-gamma exposure $G(S, z, v)$ feeds back into the spot-price drift through a coupling $\boldsymbol{\kappa} \geq 0$ and a leverage channel $\gamma z$ feeds the variance drift. We prove (Theorem 1, `paper/theory.md` §4) that the deterministic skeleton undergoes a Hopf bifurcation at $\boldsymbol{\kappa}^\star$ given by Liu's criterion $H(\kappa) = c_1 c_2 - c_0 = 0$ with transversality and positivity conditions; the 2D reduction is upper-triangular and *cannot* Hopf, which is why the memory variable $z_t$ is structural rather than cosmetic. Empirically, the question is whether this reflexive structure — when wrapped in an RL training loop — produces IV-surface dynamics statistically closer to historical SPX than four standard non-reflexive baselines (time-dependent Heston, LSV, 3/2 SV, gamma-aware-but-non-reflexive). This document locks the analysis pipeline so the comparison is honest.

## 2. Primary hypothesis (H1)

> **H1.** A model-free RL agent $\pi_{\boldsymbol{\kappa}_0}$ trained inside the reflexive 3D simulator $\mathcal{S}_{\boldsymbol{\kappa}_0}$ at the calibrated coupling $\boldsymbol{\kappa}_0$ produces, in evaluation, an IV-surface distribution closer — in sliced Wasserstein-2 distance over arbitrage-free 21-trading-day rolling windows — to the empirical SPX surface distribution from the three pre-specified historical periods than any of the four non-reflexive baseline-trained agents $\{\pi^{\text{Heston-td}}, \pi^{\text{LSV}}, \pi^{\text{3/2}}, \pi^{\text{GammaAware-NR}}\}$.

**Term definitions.**

- *RL agent* — Mamba-encoder + PPO + EWC stack vendored from `third_party/atlas/`, with options-domain adapter at `src/reflexive_options/rl/atlas_adapter.py`. Same architecture, hyperparameters, optimizer, and per-seed training budget for all five agents; the *only* difference is the simulator they are trained in.
- *Calibrated coupling* $\boldsymbol{\kappa}_0$ — fitted on a held-out *pre-event* calibration window (60 trading days ending at $t_{\text{event}} - 60$) by minimising sliced-W2 between simulator-generated and empirical 21-day surface windows. Calibration window is non-overlapping with the evaluation window.
- *IV-surface distribution* — empirical distribution over $X \in \mathbb{R}^{21 \times M \times K}$ rolling windows; surfaces interpolated to the fixed pillar grid in §4.
- *Historical periods* (each defined as $t_{\text{event}} \pm 60$ trading days, $\approx 121$ trading days, $\approx 100$ rolling 21-day windows per event):

  | Event | $t_{\text{event}}$ | Window |
  |---|---|---|
  | Volmageddon | 2018-02-05 | 2017-11-08 → 2018-05-04 |
  | COVID crash | 2020-03-16 | 2019-12-19 → 2020-06-12 |
  | Yen carry unwind | 2024-08-05 | 2024-05-10 → 2024-10-30 |

- *Closer* — strictly smaller sliced-W2 with non-overlapping 95% block-bootstrap CIs (see §6).
- *Four baselines.*

  | Tag | Name | What it has | What it lacks |
  |---|---|---|---|
  | B1 | Time-dependent Heston | Piecewise-constant $(\kappa, \theta, \xi, \rho)$ per regime | No reflexive coupling, no $G$, no $z$ |
  | B2 | Local-stochastic vol (LSV) | Heston + Dupire leverage function $L(S,t)$ | No $G$, no $z$ |
  | B3 | 3/2 SV | Variance follows 3/2 dynamics | No $G$, no $z$ |
  | B4 | Gamma-aware (non-reflexive) | $G$ included as agent state input only | $G$ does *not* feed back into price drift; $\boldsymbol{\kappa} \equiv 0$ |

  B4 is the critical control — it isolates the *reflexive feedback channel* from mere *gamma awareness* in the agent's observation space.

## 3. Secondary hypotheses (H2, H3, H4)

- **H2 (κ-sensitivity, novel).** The slope of the κ-sensitivity curve at $\boldsymbol{\kappa}_0$ for the reflexive-trained agent is positive *and* statistically distinguishable from zero (block-bootstrap 95% CI excludes zero). The same slope, computed for each of the four baseline-trained agents on the same κ-grid $\{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0\}\cdot\boldsymbol{\kappa}_0$, is statistically indistinguishable from zero (TOST equivalence test at $\pm 0.1$, $\alpha = 0.05$). Headline metric for the slope is $\tilde{\rho}_{\text{Sharpe}}$; secondaries are $\tilde{\rho}_{\text{PnL}}, \tilde{\rho}_{\text{MDD}}, \tilde{\rho}_{\text{SW2}}$ as defined in `evaluation_framework_brief.md` §3.2.

- **H3 (smile dynamics, simulator-only).** Post-FOMC skew shifts — measured as (i) 25-delta risk-reversal (RR25) change over the FOMC ±5-day window and (ii) ATM term-structure slope change over the same window — are reproduced by the *reflexive simulator* (no agent involved; pure forward simulation from initial state) with strictly smaller mean absolute error against empirical SPX than by any of the four baseline simulators, across the union of FOMC dates falling inside the three event windows of §2.

- **H4 (Hopf signature).** Empirical SPX absolute returns $|r_t|$ exhibit autocorrelation with a peak at frequency near $\omega^\star = \sqrt{c_1(\boldsymbol{\kappa}^\star)}$ (the Hopf angular frequency from Theorem 1 evaluated at the calibrated $\boldsymbol{\kappa}_0$). No baseline simulator reproduces a peak at the same frequency. Operationally: the spectral density of $|r_t|$ is estimated via Welch's method (1024-day Hann window, 50% overlap), and we compute the height of the spectral peak in a $\pm 20\%$ band around $\omega^\star$. Under H4 this peak height is strictly larger for empirical SPX and the reflexive simulator than for all four baseline simulators.

## 4. Primary metric definition

**Sliced Wasserstein-2 over arbitrage-free 21-day surface windows.**

| Component | Specification |
|---|---|
| Window length | 21 trading days |
| Window stride | 1 trading day (rolling, overlapping; CIs use block bootstrap to handle dependence) |
| Maturity grid $\mathcal{T}$ | $\{7, 14, 30, 60, 90, 180, 365\}$ days (M = 7) |
| Strike grid (log-moneyness) | $k = \log(K/F) \in \{-0.20, -0.15, -0.10, -0.05, 0, +0.05, +0.10, +0.15, +0.20\}$ <!-- TBD: spec brief says 11 strikes, framework brief says 9; locking 11 strikes per the original task spec. Resolution: $\Delta k = 0.04$, $k \in \{-0.20, -0.16, ..., 0, ..., +0.16, +0.20\}$ (K = 11). --> |
| Per-day vector dim | $M \times K = 7 \times 11 = 77$ |
| Per-window vector dim | $21 \times 77 = 1617$ |
| Distance | Sliced-W2 with $N_{\text{slices}} = 1000$ projections, projection vectors $\theta \sim \text{Unif}(S^{d-1})$, $d=1617$. 1D W2 in closed form via sorted quantiles. |
| Interpolation | Cubic spline in $k$ at each $\tau$, then linear in $\tau$ (matches `surface/interpolation.py` once implemented). |
| Arbitrage filter | Window dropped if *any* daily surface fails any of the four checks below. Filter implementation deferred to `surface/arbitrage.py`; criteria locked here. |

**Arbitrage-filter passing criteria** (all four required, tolerance $1$ bp in implied total variance $w = \sigma^2 \tau$):

1. **Convexity in $K$** — $\partial^2 C / \partial K^2 \geq 0$ at every observed strike (Roper 2010, equivalent to butterfly arbitrage absence; checked via second forward difference of $w$ in $k$, Durrleman condition $g(k) \geq -10^{-4}$).
2. **Monotonicity in $T$** — total variance $w(k, \tau)$ non-decreasing in $\tau$ at every $k$.
3. **Calendar-spread positivity** — $C(K, T_2) \geq C(K, T_1)$ for $T_2 > T_1$ at every $K$.
4. **Lee bounds** — wing slopes satisfy $|\partial w / \partial k| \leq 2$ as $|k| \to \infty$ (Lee 2004 moment-formula bound), enforced as $|w(k_{\max}, \tau) - w(k_{\max-1}, \tau)| / \Delta k \leq 2$ at the wings.

For each model (reflexive simulator and the four baselines) we additionally log the *rejection rate* — proportion of generated windows dropped by the filter — as a secondary diagnostic. A model whose unfiltered output violates arbitrage at higher rates than another is not penalised in the primary metric (the filter equalises) but the rejection rate is reported.

## 5. Secondary metrics

| Metric | Definition | Used for |
|---|---|---|
| RR25 sign + magnitude | $\sigma_{25\Delta P}(\tau) - \sigma_{25\Delta C}(\tau)$ at $\tau = 30$ days, daily | H3 |
| ATM term-structure delta | $\sigma_{\text{ATM}}(\tau=30) - \sigma_{\text{ATM}}(\tau=90)$, daily | H3 |
| Tail percentiles of $r_t$ | 1%, 5%, 95%, 99% of daily log-returns | H4 supporting evidence |
| Hurst exponent of $|r_t|$ | DFA estimator on lags $\{4, ..., 256\}$ days; replicates Marketron §5.2.3 (target ~0.3 from Livieri et al. 2018) | H4 supporting evidence |
| Spectral peak height at $\omega^\star \pm 20\%$ | Welch PSD of $\vert r_t\vert$, 1024-day Hann window, 50% overlap | H4 primary indicator |
| Agent realised Sharpe | Per-episode P&L mean / std, annualised | H2 + agent-level reporting |
| Agent max drawdown | Per-episode peak-to-trough P&L drop | H2 + agent-level reporting |
| Agent vega-adjusted alpha | Per-episode P&L regressed on portfolio vega; intercept reported | H2 + agent-level reporting |
| κ-sensitivity slope $\tilde{\rho}_m$ | $\partial \hat{m}/\partial \kappa\big\vert_{\boldsymbol{\kappa}_0}$ from cubic-spline fit, normalised by metric scale | H2 |
| Conditional sliced-W2 | SW2 conditioned on (5-day prior return, VIX level, VIX 1m–3m slope), 8 quantile bins | H1 secondary diagnostic |

## 6. Statistical procedure

| Step | Choice |
|---|---|
| Bootstrap | Stationary block bootstrap (Politis–Romano 1994), $B = 1000$ resamples, geometric block length with mean $L = 21$ trading days. |
| Significance threshold | $\alpha = 0.05$. |
| H1 multiple-comparison correction | Single primary hypothesis $\Rightarrow$ Bonferroni with $m = 1$ (no correction). |
| H2/H3/H4 secondary correction | Benjamini–Hochberg FDR at $q = 0.05$ across the secondary-hypothesis grid (H2, H3, H4 plus their per-event variants when applicable). |
| H1 decision rule | **Accept H1 iff** the reflexive-trained agent strictly dominates *all four* baseline-trained agents in primary sliced-W2 with *non-overlapping* 95% block-bootstrap CIs at *every one of the three* historical event windows (so 4 × 3 = 12 pairwise dominance checks must all hold). Failure on any one of the 12 = reject H1. |
| H2 decision rule | Accept H2 iff $\tilde{\rho}_{\text{Sharpe}}^{\text{reflexive}} > 0$ with 95% CI excluding 0, *and* TOST equivalence at $\pm 0.1$ holds for all four baselines. |
| H3 decision rule | Accept H3 iff reflexive simulator MAE strictly less than every baseline MAE on both RR25 and ATM-term-structure shifts, with 95% CI separation, after BH-FDR. |
| H4 decision rule | Accept H4 iff spectral peak height in $\omega^\star \pm 20\%$ band for empirical SPX and reflexive simulator both exceed *every* baseline by a margin with non-overlapping 95% CIs, after BH-FDR. |
| Stopping rule | No optional stopping. Full evaluation pipeline runs end-to-end before any analysis begins. No interim peeking. |

## 7. Ablations (mandatory, pre-registered)

| Tag | Ablation | Question answered |
|---|---|---|
| A1 | Set $\boldsymbol{\kappa} = 0$ in the reflexive simulator (kill the dealer-gamma feedback channel) | Does the reflexive sim collapse to baseline behaviour? Confirms $\boldsymbol{\kappa}$ is the active mechanism, not incidental architecture. |
| A2 | Strip $G$ from the agent's state vector (Heston-style state only: $(S_t, \text{IV-surface}, v_t, \text{position}, \tau)$), retrain inside the reflexive simulator | Does the *reflexive-trained* agent still beat baselines if it cannot observe gamma? Tests whether the gain is from the environment, the observation, or both. |
| A3 | Single-regime calibration vs regime-switched (per-event $\boldsymbol{\kappa}_0$ vs single global $\boldsymbol{\kappa}_0$) | Brittleness to calibration window. Reports H1 outcome under both. |
| A4 | Transfer experiment — vanilla-Heston-trained agent deployed inside the reflexive simulator at $\boldsymbol{\kappa}_0$ | Quantify P&L degradation; per-event Sharpe gap and W2 gap reported. |

Each ablation is reported as a delta to the H1 result (point estimate + 95% block-bootstrap CI). No ablation can be added post-hoc without flagging under §9.

## 8. Software / data versions

| Item | Specification |
|---|---|
| Code repository | `github.com/mahimn01/reflexive-options` |
| Pinned commit | `<TO_BE_FILLED_AT_PRE_REG_COMMIT>` (the commit that adds this document; will be inserted before the public push) |
| Vendored sub-modules | `third_party/atlas/` (Mamba+PPO+BC+EWC, vendored from `trading-algo`), `third_party/rat/` (reflexivity meter + topology detector). Vendored commit hashes recorded in `third_party/VERSIONS.txt`. |
| Empirical data — primary | WRDS OptionMetrics IvyDB US (UofT institutional access). SPX index level from CRSP daily file. |
| Empirical data — fallback | historicaloptiondata.com ALLSPX bundle ($805, one-time). |
| FOMC date list | Federal Reserve Board release calendar, 2017–2025 (committed verbatim to `data/fomc_dates.csv` before evaluation). |
| Calibration window | 60 trading days ending at $t_{\text{event}} - 60$ for each of the three events, non-overlapping with the evaluation window. |
| Random seeds | Primary seed = 42. Ablation robustness seeds = $\{1, 2, ..., 99\}$ (99 additional). All RL training runs use the same seed list across agents to preserve paired-comparison validity. |
| Compute envelope | $\leq 200$ GPU-hours total across all training and evaluation runs. Hardware: single-node A100 (40 GB) or equivalent. |
| Software stack | Python 3.12, strict mypy, ruff lint. Numpy-first; PyTorch for the RL stack only. Dependency lock via `pyproject.toml` + `uv.lock`. |
| Reproducibility | `make reproduce` target executes the full pipeline from the pinned commit; expected total wall-clock $\leq 80$ hours on the spec'd hardware. |

## 9. Deviations clause

Any deviation from this pre-registered analysis — including but not limited to: changes to the H1/H2/H3/H4 decision rules; changes to the metric definition or arbitrage filter; addition or removal of baselines, ablations, or event windows; changes to bootstrap parameters or significance thresholds; addition of post-hoc tests — must be explicitly disclosed in the final paper as a *post-hoc* analysis. Each disclosed deviation must include (i) the original pre-registered specification, (ii) the deviation as actually executed, (iii) the reason for the deviation (data availability, bug, reviewer request, etc.), and (iv) both the pre-registered and post-hoc results side-by-side. No silent edits to this document; all amendments tracked through dated commits to `paper/pre_registration.md` with the original version preserved in git history.

## 10. Timestamping

Chain of custody for this pre-registration is established by (i) the git commit hash recorded in §8 (whose tree includes this file verbatim), (ii) a public push to `github.com/mahimn01/reflexive-options` immediately after that commit, and (iii) an OpenTimestamps proof of the commit hash submitted to the Bitcoin-anchored OTS calendar servers (`ots stamp paper/pre_registration.md` from the repo root, producing `pre_registration.md.ots`, then `ots upgrade` after Bitcoin confirmation). The OTS submission step is performed manually at pre-registration commit time and is not part of the automated CI; the resulting `.ots` proof is committed alongside the document. Verification by any third party at any later date: `ots verify pre_registration.md.ots` against the document at the pinned commit. This binding makes any post-hoc edit detectable.

## 11. Connections to prior pre-registrations in finance/ML

Pre-registration is well-established in psychology and biomedical research (Camerer et al. 2018, *Nature Human Behaviour*, replicating 21 social-science experiments with pre-registered protocols and finding markedly different effect-size distributions vs the original literature) and is increasingly normalised in algorithmic-fairness research at FAccT (Cooper et al. 2022 on reproducibility checklists; Pineau et al. 2021 NeurIPS reproducibility checklist). In quantitative finance and RL-for-finance, pre-registration is essentially absent — the 2024 systematic review of RL applications in finance (<!-- TBD: exact citation; the brief at evaluation_framework_brief.md §4 references a 2024 systematic review flagging the absence of a robustness-measurement framework — will need to find the exact arXiv ID before public push -->) explicitly identifies the lack of standardised robustness measurement and pre-committed analysis as a methodological deficit driving the credibility gap between published Sharpe ratios and live-trading outcomes. To our knowledge this is the first pre-registered evaluation protocol for an RL-trained options market simulator, and it is offered partly as a methodological contribution: the κ-sensitivity slope (H2) and the SW2-on-arbitrage-filtered-windows protocol (H1) are reusable templates for future reflexive-finance work regardless of whether our specific H1 is supported.

---

**Open TBDs in this document.**

- §4: strike-grid resolution. Spec brief used 9 strikes ($\Delta k = 0.05$); the task brief specified 11 strikes. Locked at 11 strikes with $\Delta k = 0.04$. Flagged so the discrepancy is on record.
- §11: exact citation for the 2024 systematic review of RL-for-finance that flags the robustness-measurement gap. To be resolved before the public push of the pre-registration commit.
