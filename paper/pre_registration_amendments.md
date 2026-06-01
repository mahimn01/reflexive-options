# Pre-registration amendments

Per `pre_registration.md` §9 (deviations clause), this document records every amendment to the locked analysis pipeline made *after* the v0.1.0 commit (`268c061`, 2026-04-29) but *before* empirical evaluation against real SPX data. Each amendment must be (a) discovered through pre-validation work that did not touch the empirical data, (b) documented here with the date, the discovery, and the new locked behaviour, and (c) committed before the empirical run begins.

The intent of these amendments is *not* to weaken the pre-registration but to fix operational specs that were under-determined or operationally infeasible at v0.1.0. Each amendment is the result of pre-validating the detectors / metrics on synthetic ground truth — exactly the work the deviations clause anticipates.

---

## A1 — H4 spectral-test window resolution (2026-05-02)

**Original locked spec** (pre_registration.md §3, §10): "Welch's method with 1024-day Hann window, 50% overlap" with the H4 statistic computed on $|r_t|$ in a $\pm 20\%$ band around $\omega^\star = \sqrt{c_1(\boldsymbol{\kappa}^\star)}$.

**Discovery** (during C3 H4 detector validation, `runs/h4_validation/2026-05-02T*`): the locked 1024-day Hann window has spectral resolution $\Delta f = f_s / N_{\text{window}} = 252 / 1024 \approx 0.246$ cycles/year. The dimensionless §4.2 example regime in `paper/theory.md` produces $\omega^\star \approx 0.10$ cycles/year — **smaller than one Welch bin**. The peak would land in bin 0 or bin 1 and could not be discriminated from low-frequency leakage.

**Amendment**: the spectral-window length is no longer fixed a priori. It is determined post-calibration by:

$$N_{\text{window}} = \min\!\Bigl(\text{closest power of 2 to } 10 \cdot f_s / \omega^\star,\ N_{\text{trajectory}} / 2\Bigr).$$

This guarantees at least 10 Welch bins in the $\pm 20\%$ band around $\omega^\star$ (resolving the peak at SNR-relevant scales) while not exceeding half the trajectory length (preserving the 50% overlap requirement for variance reduction).

The Hann window family and 50% overlap remain locked.

**Effect on H4**: the locked decision rule (peak height in band > all baselines') is unchanged in form. Only the operational $N_{\text{window}}$ is data-dependent, and it is determined entirely from the calibrated $\omega^\star$ before the empirical SPX is touched. No look-ahead bias.

---

## A2 — H4 spectral test computed on both $|r_t|$ and $v_t$ (2026-05-02)

**Original locked spec**: H4 statistic on $|r_t|$ alone.

**Discovery**: $|r_t| = \sqrt{v_t}\cdot|\varepsilon_t|\cdot\sqrt{dt}$ where $\varepsilon_t \sim \mathcal{N}(0, 1)$. The multiplicative chi-distributed $|\varepsilon_t|$ noise injects strong out-of-band harmonics into the PSD of $|r_t|$, masking the in-band peak under realistic SNRs. On the synthetic Stuart-Landau positive control at the canonical regime, $r_t^2$ requires roughly twice the trajectory length to achieve the same detector power as $v_t$ directly.

The reflexive simulator and all four baselines expose $v_t$ as part of `SimulatorProtocol.simulate(...)`; empirical $v_t$ is not directly observable but realised-variance proxies (5-minute squared-return aggregation) provide an unbiased estimator.

**Amendment**: H4 is now reported on **both** signals:
1. $|r_t|$ from daily SPX log-returns (per the original spec, defensive primary).
2. Daily realised-variance proxy $\widehat{v}_t = \sum_{i=1}^{N_{\text{intraday}}} r_{t, i}^2$ from 5-minute SPX returns (per this amendment, sensitivity-enhanced secondary).

H4 is *accepted* iff at least one of the two signals shows the in-band peak structure required by the original decision rule. This is *one-sided* (a single positive is sufficient) because both signals carry the same physical information and the noise structures are statistically independent enough that a finding on one without the other is interpretable.

A pre-registered Bonferroni correction across the two signals is applied to the per-signal $\alpha = 0.05$ (i.e., each signal must show $p \leq 0.025$ to count as a positive).

---

## A3 — Permutation-surrogate null wording (2026-05-02)

**Original locked spec** (pre_registration.md §10): "p-value via circular shuffle of the input."

**Discovery**: a literal circular shuffle (e.g., `numpy.roll`) preserves the magnitude spectrum of the input and only rotates the phase. Such a surrogate cannot serve as a null distribution for an in-band *peak height* statistic because the peak height is invariant under phase rotation.

**Amendment**: the surrogate is now an **independent random permutation** of the input time series (i.i.d. shuffle of indices). This destroys the temporal correlation while preserving the marginal — exactly the null required by the original spec's stated *intent* ("preserves marginal but breaks frequency content"). The wording in the original §10 was operationally inconsistent with the intent; this amendment locks the intent.

---

## A4 — `in_band` decision rule definition (2026-05-02)

**Original locked spec**: "the in-band peak exists" — operationally tautological because the peak frequency was searched within the band by construction.

**Discovery**: the original wording does not produce a meaningful pass/fail signal. Any signal will have *some* maximum within the band (the band is non-empty); the question is whether that maximum is the *globally dominant* feature of the spectrum.

**Amendment**: `in_band = True` iff the maximum PSD value within the $\pm 20\%$ band around $\omega^\star$ is ≥ the maximum PSD value over the *non-band* spectrum (excluding very-low frequencies $< \omega^\star / 3$ and the in-band region itself). This makes `in_band` a global-vs-local dominance check rather than a tautology.

---

## A5 — IAAFT surrogate null replaces iid permutation in §10's H4 statistic (2026-05-02)

**Original locked spec** (post-A3): empirical p-value via independent random permutation of the input time series.

**Discovery** (V3 statistical audit, `~/Documents/reflexivity-research/verification_v3_statistics.md` WARNING-1; reproducible at `/tmp/audit_v3_h4_heston.py`): the iid permutation surrogate destroys the autocorrelation that gives the H_0 spectrum its red-noise character. Under realistic non-Hopf nulls — Heston volatility clustering or AR(1) — the permuted-spectrum in-band peak ratio has lighter right tail than under H_0, leading to anti-conservative p-values. Empirically:

| H_0 generator | nominal α | observed FPR | median p |
|---|---|---|---|
| white noise (matches null) | 0.05 | 0.010 | 0.502 |
| AR(1) φ = 0.6 | 0.05 | **0.095** | 0.005 |
| Heston \|r_t\| | 0.05 | **0.120** | 0.005 |
| Heston r_t² | 0.05 | **0.135** | 0.005 |

The `in_band` rescue from A4 keeps the joint test (in_band ∧ p < α) from exploding past 13.5%, but the surrogate null remains operationally inappropriate for the H4 hypothesis the test is supposed to discriminate against.

**Amendment**: the H4 surrogate null in `theory.spectral.detect_psd_peak` is now the **IAAFT (Iterative Amplitude-Adjusted Fourier Transform) surrogate** of Schreiber & Schmitz (1996, *Physica D* 142, 346–382). IAAFT preserves both (a) the marginal distribution and (b) the linear autocorrelation of the input by alternating between (i) replacing the surrogate's magnitude spectrum with the original's and (ii) rank-mapping the surrogate to the original's order-statistics. The implementation lives in `theory.spectral.iaaft_surrogate` and is called from `_surrogate_p_value` when `null_method='iaaft'` (the new default). Pre-A5 behaviour is preserved as `null_method='permutation'` for back-compat with the v0.1.0 test fixtures.

**Effect on H4**: the locked decision rule (in_band ∧ p < α) is unchanged in form. Only the surrogate scheme is replaced, restoring nominal FPR under realistic Heston / AR(1) H_0 (≤ 7% empirically; cf. 13.5% for the iid permutation null).

---

## A6 — GP-posterior slope CI replaces UnivariateSpline-derivative + iid bootstrap (2026-05-02)

**Original locked spec** (pre_registration.md §6, §3 H2): "block-bootstrap 95% CI" on the spline-derivative slope at $\boldsymbol{\kappa}_0$, where the slope is computed via `scipy.interpolate.UnivariateSpline(k=4, s=0)` derivative through 9 (κ, mean) collocation points.

**Discovery** (V3 statistical audit, BLOCKER-1; reproducible at `/tmp/audit_v3_bootstrap_v2.py`, n_reps=200, n_seeds=100, n_bootstrap=1000): with `s=0` the spline interpolates the noisy means exactly. The bootstrap then captures *only* the seed-resampling MC variance — it has no way to see the spline-fitting bias when the underlying mean-vs-κ curve is not in the piecewise-degree-4 polynomial span. Coverage results:

| underlying curve | true slope | empirical coverage |
|---|---|---|
| linear | 2.0e+11 | 0.97 |
| quadratic (in span) | 6.0e+11 | 0.97 |
| quintic (out of span) | 3.0e+12 | **0.00** |
| sinusoidal (out of span) | 6.3e+12 | **0.00** |
| kink at anchor (right slope) | 2.0e+11 | **0.00** |
| kink at anchor (mid value) | 1.0e+11 | **0.00** |

The kink-at-anchor scenario is plausible for an RL agent trained at exactly $\boldsymbol{\kappa}_0$ — the policy may have learned a piecewise behaviour that is locally non-smooth in κ.

**Amendment**: the κ-sensitivity slope CI is now derived from a **Gaussian-process posterior over the function**, with closed-form posterior over the derivative at the anchor:

For a GP with RBF kernel $k(x, x') = \exp(-(x - x')^2 / 2\ell^2)$ and observation noise $\sigma^2 I$, the posterior over $f'(x_{\text{anchor}})$ is also Gaussian with:
- mean: $k_{x*}(x_{\text{anchor}}, X)\,(K + \sigma^2 I)^{-1}\,(y - \bar{y})$
- variance: $k_{xx*}(x_{\text{anchor}}, x_{\text{anchor}}) - k_{x*}\,(K + \sigma^2 I)^{-1}\,k_{x*}^\top$

where $k_{x*}$ and $k_{xx*}$ are the closed-form RBF-kernel partial derivatives. The 95% CI is the resulting posterior interval ($\pm 1.96$ posterior SE).

Hyperparameter optimisation (length-scale, noise variance) uses sklearn's `GaussianProcessRegressor` with RBF + WhiteKernel and 5 restarts. The derivative-posterior calculation itself is analytic. A local-quadratic-regression fallback (with closed-form OLS variance) handles the rare case where the GP optimiser diverges. The `SensitivityResult` dataclass exposes `method ∈ {"gp", "local_quadratic_fallback"}` and `slope_se` so downstream consumers (TOST under A7) get the right standard error.

**Effect on H2**: the locked decision rule (CI excludes 0 for reflexive; TOST equivalence within ±0.1 for baselines) is unchanged in form. The CI is now correctly calibrated under broader function classes — including non-smooth policy responses at $\boldsymbol{\kappa}_0$.

---

## A7 — TOST equivalence test in §3 H2 normalised to dimensionless elasticity (2026-05-02)

**Original locked spec** (pre_registration.md §3 H2): "TOST equivalence test at ±0.1, α = 0.05" applied to the κ-sensitivity slope at $\boldsymbol{\kappa}_0$.

**Discovery** (V3 statistical audit, BLOCKER-2): the implementation returns the slope in raw units (metric / κ). In the v0.1.0 baseline (`tests/repro/baseline_v0.1.0.json`), the recorded `slope_at_anchor` is order $10^{11}$–$10^{13}$. A TOST margin of ±0.1 applied to a raw slope of that order treats anything but slopes $> 10^{12}$ as "equivalent to zero" — a vacuous test. The pre-reg's intent (per `evaluation_framework_brief.md` §3.2 and the §5 secondary-metrics table) was the **dimensionless elasticity** $\tilde{\rho} = \rho \cdot \boldsymbol{\kappa}_0 / |m(\boldsymbol{\kappa}_0)|$, on which ±0.1 is a meaningful "10% effect" margin. The implementation–spec gap was undetected because the H2 evaluator was never wired.

**Amendment**: the TOST in §3 H2 is now applied to the **dimensionless elasticity**:

$$\tilde{\rho} = \hat{\rho} \cdot \frac{\boldsymbol{\kappa}_0}{|\hat{m}(\boldsymbol{\kappa}_0)|}$$

where $\hat{\rho}$ is the GP-posterior slope estimate (A6) and $\hat{m}(\boldsymbol{\kappa}_0)$ is the metric mean at the anchor. The TOST is computed via `theory.inference.tost_equivalence` on $(\tilde{\rho}, \mathrm{SE}(\tilde{\rho}))$ at margin = 0.1 (the intended ±10% bound) and α = 0.05. The reflexive-trained agent's H2 acceptance criterion remains "$\tilde{\rho} > 0$ with 95% CI excluding zero AND TOST equivalence at ±0.1 for all four baseline-trained agents."

**Effect on H2**: the locked margin (±0.1) and α (0.05) are unchanged. Only the units on which the test is computed are corrected. The test now has its intended power-vs-effect-size profile.

---

## Procedural commitment

This document is committed *before* the empirical SPX evaluation. The final empirical analysis pipeline is the union of `pre_registration.md` v1 and the seven amendments above (A1–A7), with **no further amendments permitted** after the SPX data is loaded into the codebase. Any post-data deviation must be reported in the paper as an *exploratory* analysis under the §9 deviations clause.

The amendments are commit-hash anchored to this file. The set of permissible amendments is closed at the commit landing A5–A7 (`63078f5`).

---

## Re-opening the amendment set for pre-data structural corrections (A8–A11, 2026-06-01)

The "Procedural commitment" above closed the amendment set at A1–A7 (commit `63078f5`) *before the SPX data is loaded*. We re-open it here, **still before any empirical data is loaded**, to correct two material specification errors and one over-claim discovered through pre-data analysis and independent expert review. The §9 deviations clause requires disclosing such deviations; it is more honest to correct a knowingly-broken protocol *before* data load than to run it. All four amendments below are validated on simulator-generated ground truth only (no real market data), exactly as A1–A7 were. After these land, `paper/pre_registration.md` is corrected (original preserved in git history per §9) and the OpenTimestamps proof `paper/pre_registration.md.ots` is regenerated. The A1–A7 set remains valid where untouched; A8–A11 supersede the original H1 and H4 specifications.

### A8 — H4 redesigned as a critical-slowing-down test (the spectral H4 is geometrically un-resolvable)

**Discovery.** The original H4 (and amendments A1, A2, A5 to it) sought a spectral peak in $|r_t|$ at the Hopf angular frequency $\omega^\star = \sqrt{c_1(\kappa^\star)}$ via a 1024-day Welch window. The Hopf limit-cycle period is $T = 2\pi/\omega^\star = 5.3$–$11.0$ years (`theory.md` L144: $\omega^\star = 0.5724$ rad/yr $\Rightarrow$ $11.0$ yr; L778: $\omega^\star = 1.18$ rad/yr $\Rightarrow$ $5.3$ yr). The target frequency therefore lies **below the lowest non-DC bin** of any window that fits the data, and each $\pm 60$-trading-day event window spans only $4$–$9\%$ of a single cycle. The test can never reject regardless of whether the phenomenon exists — a specification error, not a power problem. A1/A2/A5 (window resolution, dual signal, surrogate null) patched a detector that was geometrically incapable of detecting its target.

**Original spec.** §3 H4: spectral peak height in $|r_t|$ at $\omega^\star \pm 20\%$ via Welch PSD (1024-day Hann window, 50% overlap), per A1–A5.

**Amendment.** Replace with a critical-slowing-down (CSD) early-warning test (Scheffer et al. 2009; Dakos et al. 2012). As $\kappa \to \kappa^\star$ the leading eigenvalue's real part vanishes, so within a short pre-transition window the rolling lag-1 autocorrelation (gating statistic) and rolling variance (confirmatory) of $|r_t|$ rise; the trend is quantified by Kendall's $\tau$ against an AR(1)/phase-randomised stationary surrogate ensemble ($N = 1000$), BH-FDR at $0.10$ within the H4 family, nested in the study-wide BH. Detrend: Gaussian kernel, bandwidth $10\%$ of record. Rolling sub-window: absolute $w = 30$ trading days. Robustness proxies: $r_t^2$, $\log r_t^2$.

**Detectability (the thing the original lacked), validated on synthetic ground truth** (`scripts/csd_validation.py`, reproduced independently): on a stationary negative control the false-positive rate is $\le 0.083$ (nominal $\alpha = 0.05$); on a positive control with the AR(1) recovery rate ramped toward criticality, the **autocorrelation EWS attains power $0.82$ ($\kappa \ge 0.95\,\kappa^\star$) to $0.85$ ($\kappa \ge 0.99\,\kappa^\star$) on a 252-trading-day record**. **Honest limitation, pre-registered:** the bare $\pm 60$-day event window is underpowered ($\sim 0.50$); the confirmatory $\ge 80\%$ test therefore uses the $\pm 126$-day (252-trading-day) record surrounding each event (events are quarterly-spaced, so these records exist), and H4 at the 121-day horizon is reported as *exploratory* only.

**Effect.** H4 becomes a finite, falsifiable, pre-registered test where there was none. Implementation: `src/reflexive_options/theory/critical_slowing_down.py`, tests `tests/test_critical_slowing_down.py`.

### A9 — Primary hypothesis H1 replaced by a direct dealer-gamma (GEX) regression (the RL route is confounded)

**Discovery.** The original H1 routes the entire reflexivity claim through a Mamba+PPO+EWC RL agent trained inside the simulator at seed 42, then compares the IV-surface distribution the *agent's trading* induces against SPX in sliced-W2. The IV surface is an emergent artifact of agent behaviour, so "the reflexive sim matches markets" is statistically inseparable from "this RL architecture, at this seed, learned to trade toward SPX." A non-reflexive simulator with a sufficiently expressive agent could match equally well; the test conflates the **model mechanism** with the **estimator**.

**Original spec.** §2 H1: RL agent $\pi_{\kappa_0}$ trained in the reflexive sim produces surfaces closer (sliced-W2 over 21-day windows) to SPX than four baseline-trained agents; 12 pairwise dominance checks.

**Amendment.** New primary hypothesis **H1'**: a direct, no-RL, no-simulator regression testing the mechanism's core sign-conditional prediction. Estimate aggregate signed dealer gamma $\mathrm{GEX}(t) = \sum_{K,T} \mathrm{OI}_{K,T}\,\Gamma_{\mathrm{BS}}\,\mathrm{sign}_{K,T}\,S^2\,\cdot 0.01\,\cdot\mathrm{mult}$ from the EOD SPX OI grid (SqueezeMetrics dealer-sign primary; `all_long` and `naive` conventions reported as a sensitivity band). Then a strictly predictive pooled panel (event fixed effects, $\approx 3\times113 \approx 339$ daily rows): $y_{t+1} = b_0 + b_{\mathrm{GEX}} z(\mathrm{GEX}_t) + b_1 \mathrm{RV}_t + b_2 z(\mathrm{VIX}_t) + \sum_d \delta_d \mathrm{DOW}_{d,t} + \varepsilon_{t+1}$, primary outcome $y = \mathrm{RVV}_{t+1}$ (realized vol-of-vol), secondary $y = \mathrm{CSD}_{t+1}$. Inference: Newey–West HAC (Bartlett, $\approx 4$ lags) + moving-block bootstrap (block 10, 2000 resamples). **Decision rule:** reject in favour of H1' iff $b_{\mathrm{GEX}} < 0$ with both bootstrap and HAC one-sided $p < 0.05$, surviving BH-FDR $q=0.05$ across $\{\mathrm{RVV},\mathrm{CSD}\}\times\{\mathrm{pooled},\mathrm{per\text{-}event}\}$, AND the quiet-regime control window (A11) shows a weaker/absent effect. The RL surface tournament is **demoted to a secondary/exploratory result** under the secondary BH family, flagged confounded.

**Detectability, validated on synthetic ground truth** (`gex_validation`, reproduced from disk): at the exact $3\times121$-day footprint, pooled positive-control ($\kappa=0.8$) power $0.86$ with $0.98$ sign recovery; $\kappa=0$ negative-control FPR $0.02$; quiet ($\kappa=0.15$) reject-rate $0.12$. A partial correlation of only $\sim 0.14$ is detectable at $n\approx 339$. Single-window power is $0.50$, so the **pooled 3-event panel is the load-bearing estimator**.

**Effect.** The primary falsifiable claim is tested directly on data, free of the RL/simulator confound, with a quiet-window control. Implementation: `src/reflexive_options/empirical/gex_regression.py`, `gex_simulator.py`, `gex_validation.py`, tests `tests/test_gex_regression.py`.

### A10 — Theorem (Hawkes–SV) repositioned; the $n_{\mathrm{SV}}$ "verification" removed; $\kappa$ unit-chain made explicit

**Discovery (over-claim).** The Hawkes–SV section defined a dimensionless ratio $n_{\mathrm{SV}} := c_0/(c_1 c_2)$ (equivalently $1 + \lambda_{\max}/\beta_0$) engineered so that the identity $n_{\mathrm{SV}} = n$ held *by definition*, then reported it "verified to $\sim 10^{-15}$" — a tautology whose residual is floating-point noise. Three independent expert reviewers flagged it. Separately, the "empirical SPX market sits exactly at $\kappa^\star$" claim chained the dimensionless $\kappa^\star \approx 17.81$ (artificial $G_x \approx 0.5$ regime) to the per-USD empirical $\kappa_0 \sim 5\times 10^{-12}$ — quantities $\sim 13$ orders of magnitude apart with no rescaling map, and `theory.md` itself notes the empirically-tuned config does not Hopf in the literature-prior range.

**Amendment.** (i) Remove the $n_{\mathrm{SV}}$ definition, its "machine-precision" verification, and the obsolete figure. (ii) Reposition the theorem: under the BDHM (2013) + Jaisson–Rosenbaum (2015) diffusive near-critical limit, Hawkes branching-ratio criticality $n \to 1$ is the literal analogue of the **real-eigenvalue (saddle-node)** stratum (non-oscillatory); the model's **Hopf** stratum is a strictly stronger, oscillatory instability beyond any scalar branching ratio — the genuinely novel "unoccupied cell." (iii) Replace the deleted identity with a falsifiable spectral discriminator `classify_stratum` (`hawkes_sv_bifurcation.py`) separating Hopf-side (peak prominence $5.4$–$9.6$) from real-edge/stable side ($1.8$–$2.4$) with zero overlap on synthetic ground truth. (iv) Add an explicit $\kappa$ rescaling map (`theory/kappa_rescaling.py`): $\kappa_{\mathrm{dimensionless}} = \kappa_0 \cdot G_{\mathrm{char}}$ with $G_{\mathrm{char}} \sim S^1\cdot\mathrm{OI}$; at empirical magnitudes the dimensionless coupling spans $\sim 3.3$ orders of magnitude and *brackets* $\kappa^\star = 17.81$ — proximity to the Hopf is **indeterminate** and is deferred to the empirical GEX test (A9), not asserted.

**Effect.** The Hawkes connection is honest (a correspondence and a position, not a tautological identity), and the empirical-proximity claim is correctly scoped as open. Future-work note: whether the Hopf cell survives the rough-volatility ($H<1/2$) limit (cite, not prove; expected negative).

### A11 — Event-window dates, strike grid, control window, researcher-DOF locks

**Date reconciliation.** The §2 event-window dates were hand-edited and differ by $\sim 2$ trading days from the authoritative `paper/event_windows.txt` (computed via `pandas_market_calendars` NYSE calendar). The locked dates are now the `event_windows.txt` values: Volmageddon $2017\text{-}11\text{-}07 \to 2018\text{-}05\text{-}02$; COVID ($t_{\mathrm{event}} = 2020\text{-}03\text{-}16$) $2019\text{-}12\text{-}17 \to 2020\text{-}06\text{-}10$; Yen carry $2024\text{-}05\text{-}08 \to 2024\text{-}10\text{-}29$.

**Strike grid.** §4's displayed strike list (9 strikes) is corrected to the locked $K = 11$ grid, $k = \log(K/F) \in \{-0.20, -0.16, \dots, 0, \dots, +0.16, +0.20\}$ ($\Delta k = 0.04$), consistent with the per-day dimension $M \times K = 7 \times 11 = 77$ and per-window $21 \times 77 = 1617$.

**Quiet-regime control window (new, pre-specified).** A 121-trading-day calm window with no macro stress, locked at **2017-05-01 $\to$ 2017-10-20** (the low-VIX 2017 grind), used by both H1' (A9) and the CSD null. The mechanism predicts a weaker/absent GEX$\to$vol-of-vol effect there.

**Researcher-degree-of-freedom locks (pre-data).** CSD: gating statistic = lag-1 autocorrelation, $w = 30$ days, 252-day operative record. GEX: SqueezeMetrics dealer-sign convention primary, forward dating (regressors $t$, outcome $t+1$). $\kappa$ map: $G_{\mathrm{char}} = $ peak $|G|$ of the aggregator. Quiet-window dates as above.

**Residual correction.** A hallucinated DOI in §References (Faff/Brailsford pre-registration pathway) is corrected to Faff (2023), PBFJ vol 79 art. 101837, DOI `10.1016/j.pacfin.2022.101837` (the original `…101859` resolved to an unrelated paper).
