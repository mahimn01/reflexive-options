# H4 Redesign — Critical-Slowing-Down Early-Warning Signal

**Status:** PRE-DATA pre-registration amendment (amendment A5). Validated on
mechanistic synthetic ground truth with known coupling proximity
`kappa/kappa_star`; no real-market data used (deferred to September WRDS pull).

**Purpose:** Replace the original H4 ("autocorrelation/spectral peak in `|r_t|`
at `omega_star = sqrt(c1(kappa_star))` via a 1024-day Welch window"), which is
*geometrically impossible* and therefore unfalsifiable in the available data.

---

## A5.1 Why the original H4 must be retired (the fatal flaw)

The Hopf limit cycle has angular frequency `omega_star`, period
`T = 2*pi/omega_star`. From `theory.md`:

- L144: `omega_star = 0.5724 rad/yr  =>  T = 10.98 yr`
- L778: `omega_star = 1.18 rad/yr (canonical param)  =>  T = 5.33 yr`

So the oscillation period is **5.3–11.0 years**. Two independent geometric facts
make the spectral test impossible in the available data:

1. **Welch-bin resolution.** A 1024-trading-day window (~4.06 yr) has frequency
   resolution `Delta_f = 1/(4.06 yr) = 0.246 cyc/yr`, i.e. its lowest non-DC bin
   sits at `~1.55 rad/yr`. The target `omega_star ∈ [0.57, 1.18] rad/yr` lies
   **below the first resolvable non-DC bin** — it is aliased into the DC/leakage
   region and cannot be assigned a testable bin ordinate.
2. **Event-window coverage.** Each `±60`-day event window (121 trading days,
   0.48 yr) covers only `0.48/10.98 = 0.044` to `0.48/5.33 = 0.090` of one cycle
   — **4–9% of a single oscillation**. No periodicity is estimable from under a
   tenth of a cycle.

You cannot detect a multi-year oscillation in `<=4` years (let alone `<=0.5`
years) of data. The test as written can never reject, regardless of whether the
phenomenon exists. This is a specification error, not a power problem.

---

## A5.2 The replacement: critical slowing down (CSD)

Instead of detecting the *post-bifurcation oscillation* (impossible in short
records), we detect **nearness to the bifurcation** with the standard
early-warning-signal (EWS) methodology of Scheffer et al. (2009) and Dakos et
al. (2012) — the canonical tool for short *pre-transition* windows.

**Mechanism.** Near a supercritical Hopf (or any codimension-1) bifurcation the
real part of the leading Jacobian eigenvalue tends to zero: `Re(lambda) -> 0` as
`kappa -> kappa_star`. The system's recovery rate from perturbations slows
("critical slowing down"). In discrete time the dominant mode behaves like an
AR(1) with coefficient `phi = exp(Re(lambda) * dt) -> 1`. Before the transition
this produces, in a short rolling window:

- **(i) rising lag-1 autocorrelation** of the volatility proxy (`phi -> 1`),
- **(ii) rising variance** (perturbations decay more slowly, so accumulate),
- **(iii) (auxiliary) rising fitted AR(1) coefficient.**

These are leading indicators that appear in records far shorter than one cycle —
by construction CSD is designed for short pre-transition data.

**Target series.** Volatility proxy `x_t = |r_t|` (the original H4 target),
with `r_t^2` and `log(r_t^2)` as pre-registered robustness variants.

---

## A5.3 Hypothesis statement (LaTeX-ready)

> **H4 (redesigned).** As the reflexive coupling `kappa` approaches its critical
> value `kappa_star`, the realised-volatility proxy `|r_t|` exhibits critical
> slowing down: within a pre-transition event window the rolling lag-1
> autocorrelation and rolling variance of `|r_t|` rise monotonically in time.
> Formally, the Kendall rank correlation `tau` between each rolling EWS statistic
> and time is positive, and exceeds what a trend-free stationary process with the
> same linear autocorrelation structure produces.

**Null hypothesis `H4_0`:** the rolling EWS statistic has no positive time trend
beyond that of a stationary linear (AR(1) / phase-randomised) process;
`E[tau] = 0`.

---

## A5.4 Estimator (pre-registered, frozen)

Implemented in `src/reflexive_options/theory/critical_slowing_down.py`.

1. **Detrend.** Remove a slow trend from `x_t = |r_t|` with a Gaussian-kernel
   smoother (bandwidth = 10% of the record; Dakos et al. 2012 default) so that a
   deterministic mean drift cannot masquerade as rising variance.
2. **Rolling EWS.** Over a right-anchored rolling sub-window of **absolute**
   length `w = 30` trading days (~6 weeks; Dakos et al. 2012 use fixed-length
   windows), computed daily across the event window, take:
   - lag-1 autocorrelation `rho_1(t)`, and
   - rolling variance `s^2(t)` (ddof = 1).
   A short absolute window (rather than a fixed *fraction* of the record) is
   essential: it leaves enough Kendall-tau samples (`121 - 30 + 1 = 92` in the
   event window) for the trend test to have power; a 50%-of-record window
   over-smooths and roughly halves power.
3. **Trend statistic.** Kendall's `tau` between each rolling EWS series and time.

## A5.5 Null model and test statistic

- **Surrogate null (primary):** AR(1) surrogate matching the detrended series'
  mean, fitted `phi`, and innovation variance — preserves the linear stochastic
  structure (so EWS baselines are calibrated) but is **stationary** (no trend).
- **Surrogate null (robustness):** Fourier phase-randomised surrogate (preserves
  the full power spectrum, destroys monotone trend / nonlinearity).
- **Ensemble:** `N = 1000` surrogates per series; recompute the rolling EWS and
  Kendall `tau` on each.
- **One-sided surrogate p-value** (directional CSD prediction `tau > 0`):
  `p = (#{tau_surr >= tau_obs} + 1) / (N_valid + 1)`.

## A5.6 Decision rule (with BH-FDR)

Per ticker × event window, two primary EWS statistics are tested
(autocorrelation, variance). Across the redesigned-H4 family — the two EWS
statistics aggregated over the analysed event windows — control the false
discovery rate with **Benjamini–Hochberg at FDR = 0.10**, nested inside the
study-wide BH that also covers H1–H3. Reject `H4_0` for a statistic iff its
BH-adjusted surrogate p-value `<= 0.10` **and** `tau > 0` (sign-consistent with
the CSD prediction). The combined H4 verdict requires the **autocorrelation** EWS
to reject sign-consistently on the `252`-trading-day record (it is the
higher-powered statistic — see A5.7), with variance as the confirmatory
statistic.

---

## A5.7 Detectability — quantitative proof (the thing original H4 lacked)

**Why this is detectable where the spectral test was not.** CSD reads the
*rate of change* of second-order statistics within the window, not a frequency.
A `±60`-day (121-day) record with an absolute `w = 30`-day rolling window yields
`121 - 30 + 1 = 92` rolling-statistic samples — ample for a Kendall-tau trend
test. (Crucially, the rolling window must be **absolute** and short, ~6 weeks, as
in Dakos et al. 2012; a "50% of the record" window leaves too few Kendall samples
and over-smooths, roughly halving power.) The signal (a monotone rise in
`rho_1`/variance as `phi -> 1`) accrues over the *whole* record rather than
requiring a full oscillation, so detection scales with record length, not with
cycle count. This removes the geometric obstruction.

**Validation harness.** `scripts/csd_validation.py` runs `N_SEEDS = 60`,
`N_SURR = 500`, `alpha = 0.05`, rolling window `w = 30` days, over record lengths
{60, 121, 252} days × ramp endpoints `kappa_end ∈ {0.85, 0.95, 0.99}·kappa_star`
× {autocorr, variance}, plus stationary negative controls. Ground truth is an
AR(1) latent with **fixed innovation size** `sigma = 0.05` whose recovery
coefficient `phi` ramps from `phi_min = 0.20` (far below criticality) to
`phi = 1 - (1-phi_min)(1-kappa_end_frac)` (near criticality), the linearised
image of `Re(lambda) -> 0`. The volatility proxy is the folded transform `|z|` of
the slowing latent. Artifacts: `runs/csd_validation/20260531_224018/`
(`results.json`, `results.csv`, `summary.txt`).

**Negative controls (stationary, no critical slowing down):** stationary AR(1),
fixed `phi = 0.5`, fixed `sigma`. Empirical false-positive rate, all cells:

| statistic | d=60 | d=121 | d=252 |
|-----------|------|-------|-------|
| autocorr  | 0.083 | 0.050 | 0.033 |
| variance  | 0.083 | 0.067 | 0.033 |

Max FPR across all negative cells = **0.083** (nominal `alpha = 0.05`; within
Monte-Carlo slack at `N_SEEDS = 60`). The detector does not fire on stationarity.

**Positive controls (`phi` ramped toward 1, i.e. `kappa -> kappa_star`):**
detection power (fraction of 60 seeds firing at `alpha = 0.05`), verbatim from
`runs/csd_validation/20260531_224018/results.csv`:

| statistic | record d | kf=0.85 | kf=0.95 | kf=0.99 |
|-----------|----------|---------|---------|---------|
| autocorr  | 60  | 0.15 | 0.22 | 0.25 |
| autocorr  | 121 | 0.45 | 0.50 | 0.53 |
| autocorr  | 252 | 0.60 | **0.82** | **0.85** |
| variance  | 60  | 0.28 | 0.30 | 0.30 |
| variance  | 121 | 0.25 | 0.32 | 0.33 |
| variance  | 252 | 0.52 | 0.75 | 0.77 |

(`kf = kappa_end/kappa_star`; bold = power ≥ 0.80.)

**Minimum window/ramp for power ≥ 80% (the operative result).** The **autocorrelation
EWS at the 252-trading-day record reaches power 0.82 at `kappa >= 0.95 kappa_star`
and 0.85 at `kappa >= 0.99 kappa_star`** — this is the minimum (window, ramp)
configuration clearing the ≥80% target, and it is the **pre-registered primary
test**. Three honest consequences follow, and we register all three:

1. **Autocorrelation is the gating statistic, not variance.** On this synthetic
   ground truth the lag-1 autocorrelation is the higher-powered of the two EWS at
   every record length; variance trails it (0.77 vs 0.85 at 252 days). The
   pre-registered H4 verdict therefore gates on **autocorrelation**, with variance
   as the confirmatory statistic. (This reverses an earlier draft; the data
   dictate the choice.)

2. **The operative record is 252 trading days, not the ±60-day event window.** The
   `121`-day event window does **not** reach 80% (autocorrelation tops out at
   0.53, variance at 0.33). It nonetheless yields a *directional, significant-on-
   median* signal — at `121` days the median surrogate p-value is `~0.05` and the
   mean Kendall `tau` is `+0.41`–`+0.47` (positive, as the CSD prediction
   requires) — so the event window provides supporting evidence, but the **gating
   test uses the longer `±126`-day (252-trading-day) window centred on each event**.

3. **This is still a categorical improvement over the original H4.** The spectral
   test had *exactly zero* detectability at any window length that fits the data
   (the target frequency is unresolvable). The CSD test attains `0.85` power at a
   `252`-day record that *does* fit between events, with a calibrated `≤ 0.083`
   false-positive rate — a finite, pre-registered, falsifiable test where there
   was none.

**Honest limitation.** We do **not** claim ≥80% power inside the bare `±60`-day
window. If the analysis must be confined to `±60` days, the test is underpowered
(`~0.5`) and H4 should be reported as *exploratory* at that horizon; the
confirmatory 80%-powered test requires the `252`-day surrounding record.

---

## A5.8 Pre-registered choices, frozen before data

- Proxy: `|r_t|` primary; `r_t^2`, `log(r_t^2)` robustness.
- Rolling sub-window: **absolute `w = 30` trading days** (~6 weeks; NOT a
  fraction of the record).
- Detrend: Gaussian kernel, bandwidth 10% of record.
- Surrogate: AR(1) primary, phase-randomised robustness; `N = 1000`.
- One-sided (`tau > 0`); `alpha = 0.05` per test; BH-FDR = 0.10 within family,
  nested in study-wide BH with H1–H3.
- Ground-truth generator: fixed innovation `sigma = 0.05`; `phi_min = 0.20`.
- Gating statistic: **autocorrelation** EWS (higher power); variance confirmatory.
- Operative record: **252 trading days** (`±126` days centred on the event); the
  bare `±60`-day window is exploratory only.
- Power target stated and met: **≥80% at the 252-day record, autocorrelation EWS,
  `kappa >= 0.95 kappa_star`** (0.82 at `kf = 0.95`, 0.85 at `kf = 0.99`); max
  negative-control FPR = 0.083 (nominal `alpha = 0.05`).

---

## A5.9 main.tex paragraph (drop-in)

> The original H4 sought a spectral peak in $|r_t|$ at the Hopf frequency
> $\omega_\star=\sqrt{c_1(\kappa_\star)}$. Because the limit-cycle period is
> $5.3$–$11.0$ years, this frequency lies below the lowest resolvable bin of any
> window that fits our records, and each $\pm 60$-day event window spans under
> one-tenth of a cycle; the test is therefore unfalsifiable in our data. We
> replace it with a critical-slowing-down early-warning test
> \citep{scheffer2009ews,dakos2012methods}. As $\kappa\to\kappa_\star$ the leading
> eigenvalue's real part vanishes, so within a short pre-transition window the
> rolling lag-1 autocorrelation and rolling variance of $|r_t|$ rise; we quantify
> the trend with Kendall's $\tau$ and assess significance against an
> AR(1)/phase-randomised stationary surrogate ensemble, controlling the false
> discovery rate at $0.10$ (Benjamini--Hochberg). On simulator-derived ground
> truth in which the AR(1) recovery rate is ramped toward criticality, the test
> holds its false-positive rate at $\le 0.083$ under stationary nulls and attains
> $0.82$–$0.85$ power (rolling lag-1 autocorrelation, the gating statistic) for
> detecting the approach to $\kappa_\star$ on a $252$-trading-day record
> ($\kappa \ge 0.95\,\kappa_\star$) — where the spectral test had exactly zero
> detectability at any usable window length by construction. Within the bare
> $\pm 60$-day event window the test is directional but underpowered ($\sim 0.5$);
> the confirmatory $\ge 80\%$ test uses the $\pm 126$-day surrounding record.

---

## A5.10 Verified bibliography entries

```bibtex
@article{scheffer2009ews,
  author  = {Scheffer, Marten and Bascompte, Jordi and Brock, William A. and
             Brovkin, Victor and Carpenter, Stephen R. and Dakos, Vasilis and
             Held, Hermann and van Nes, Egbert H. and Rietkerk, Max and
             Sugihara, George},
  title   = {Early-warning signals for critical transitions},
  journal = {Nature},
  year    = {2009},
  volume  = {461},
  number  = {7260},
  pages   = {53--59},
  doi     = {10.1038/nature08227}
}

@article{dakos2012methods,
  author  = {Dakos, Vasilis and Carpenter, Stephen R. and Brock, William A. and
             Ellison, Aaron M. and Guttal, Vishwesha and Ives, Anthony R. and
             K{\'e}fi, Sonia and Livina, Valerie and Seekell, David A. and
             van Nes, Egbert H. and Scheffer, Marten},
  title   = {Methods for Detecting Early Warnings of Critical Transitions in
             Time Series Illustrated Using Simulated Ecological Data},
  journal = {PLoS ONE},
  year    = {2012},
  volume  = {7},
  number  = {7},
  pages   = {e41010},
  doi     = {10.1371/journal.pone.0041010}
}

@article{lenton2011ews,
  author  = {Lenton, Timothy M.},
  title   = {Early warning of climate tipping points},
  journal = {Nature Climate Change},
  year    = {2011},
  volume  = {1},
  number  = {4},
  pages   = {201--209},
  doi     = {10.1038/nclimate1143}
}
```

---

## A5.11 Files

- `src/reflexive_options/theory/critical_slowing_down.py` — detector.
- `tests/test_critical_slowing_down.py` — deterministic tests.
- `scripts/csd_validation.py` — power/FPR harness.
- `runs/csd_validation/20260531_224018/` — validation artifacts
  (`results.json`, `results.csv`, `summary.txt`).

## A5.12 Note for the integrating agent

The power/FPR tables in A5.7 are lifted verbatim from the harness run
`runs/csd_validation/20260531_224018/`. **`runs/` is git-ignored**, so re-run
`uv run python scripts/csd_validation.py` to regenerate the artifacts (a new
timestamped dir; the harness and tests are deterministic so the numbers
reproduce). After dropping A5 into `pre_registration_amendments.md`, A5.9 into
`main.tex`, and the A5.10 entries into `references.bib`, regenerate the timestamp
proof: `uv run ots stamp paper/pre_registration.md`.
