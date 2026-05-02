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

## Procedural commitment

This document is committed *before* the empirical SPX evaluation. The final empirical analysis pipeline is the union of `pre_registration.md` v1 and the four amendments above, with **no further amendments permitted** after the SPX data is loaded into the codebase. Any post-data deviation must be reported in the paper as an *exploratory* analysis under the §9 deviations clause.

The amendments are commit-hash anchored to this file. The set of permissible amendments is closed at this commit.
