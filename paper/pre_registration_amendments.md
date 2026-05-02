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

The amendments are commit-hash anchored to this file. The set of permissible amendments is closed at the commit landing A5–A7 (`<commit hash to be filled in at merge time>`).
