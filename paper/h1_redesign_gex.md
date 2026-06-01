# A5 — Primary-hypothesis redesign: direct dealer-gamma (GEX) regression (H1')

**Status.** Pre-data pre-registration amendment. LaTeX-ready spec for
`paper/pre_registration_amendments.md` (amendment A5) and a new primary-result
subsection of `paper/main.tex` (§ Evaluation). Anchored before any SPX data is
loaded. No real market data is used anywhere in this amendment; the validation
is on simulator-generated data with known ground-truth coupling, exactly as the
original H4 detector was validated.

Implementation: `src/reflexive_options/empirical/gex_regression.py` (estimator +
inference), `src/reflexive_options/empirical/gex_simulator.py` (data-free
ground-truth generator), `src/reflexive_options/empirical/gex_validation.py`
(positive/negative/quiet controls), tests in `tests/test_gex_regression.py`.

---

## A5.0 Why the original H1 is confounded, and what H1' fixes

The original primary hypothesis (pre-registration §2, H1) routes the entire
reflexivity claim through a **Mamba + PPO + EWC** reinforcement-learning agent
trained inside the reflexive simulator at seed 42, then compares the IV-surface
distribution the *agent's trading* induces against SPX in sliced Wasserstein-2.
The IV surface in that pipeline is an **emergent artifact of agent behaviour**,
so "the reflexive sim matches markets" is statistically inseparable from "this
particular RL architecture, at this seed, with this reward, learned to trade in
a way that happens to produce SPX-like surfaces." A *non-reflexive* simulator
wrapped in a *sufficiently expressive* agent could match equally well; an
agent-capacity or seed artifact is indistinguishable from a mechanism result.
The confound is structural, not a tuning issue: the test conflates the **model
mechanism** (dealer-gamma feedback) with the **estimator** (a deep RL policy).

**H1' removes the agent and the simulator from the inference path entirely.** It
tests the reflexive mechanism's *core falsifiable prediction* directly on the
data, using only (i) the SPX option open-interest grid and (ii) SPX index
returns. No policy is trained; no surface is simulated; no Wasserstein distance
to a generated distribution is computed. The RL surface tournament is **demoted
to a secondary/exploratory result** (it remains reported as old-H1, re-labelled,
under the secondary-hypothesis FDR family, with the explicit caveat above).

The mechanism (theory.md §1, predictions 3–4) says: when dealers are net **short**
gamma, delta-hedging is **destabilizing** — they buy into rallies and sell into
selloffs, amplifying realized variance and pushing the system toward the Hopf
edge; when net **long** gamma, hedging is **stabilizing** and damps variance.
This is a sign-conditional statement about *forward* volatility dynamics that can
be read straight off returns once the signed dealer-gamma state is estimated.

---

## A5.1 Hypothesis H1' (primary, replacing H1)

> **H1'.** Let `GEX(t)` be the aggregate signed dealer gamma exposure estimated
> from the end-of-day SPX option open-interest grid (A5.2). Then, controlling for
> the VIX level, lagged realized volatility, and day-of-week, **more negative
> `GEX(t)` predicts higher next-day realized vol-of-vol and a higher next-day
> short-window critical-slowing-down (CSD) signal**. Operationally, in the
> standardized predictive regression (A5.3) with dependent variable measured at
> `t+1` and all regressors dated `t`, the coefficient on standardized signed GEX
> is **strictly negative and statistically significant** (one-sided), and the
> effect is **markedly weaker or absent in a pre-specified quiet-regime control
> window** (A5.5).

The negative sign is the directional content: GEX is signed so that short-gamma
days are negative; the mechanism predicts those days carry elevated forward
vol-of-vol, hence `b_GEX < 0`.

---

## A5.2 GEX estimator (no RL, no simulator)

For each trading day `t`, from the EOD option chain (strikes `K`, maturities `T`,
per-contract open interest `OI`, per-contract implied vol `σ`, spot `S`):

```
GEX(t) = Σ_{K,T}  OI_{K,T} · Γ_BS(S, K, T, σ_{K,T}) · sign_{K,T}
                  · S² · 0.01 · multiplier
```

- `Γ_BS = e^{-qτ} φ(d1) / (S σ √τ)` is the Black–Scholes spot gamma (identical
  for calls and puts), `d1 = [ln(S/K) + (r − q + ½σ²)τ] / (σ√τ)`.
- `sign_{K,T}` is the dealer-net convention. Primary: **SqueezeMetrics / SpotGamma**
  — dealers long calls (`+1`), short puts (`−1`); this matches `G(S,t)` in
  theory.md eq. (G-aggregator) and the SqueezeMetrics SPX default cited there.
- `S²·0.01` converts BS gamma into **dollar gamma** (the $-change in aggregate
  delta per 1% spot move), the quantity that drives hedging flow; `multiplier`
  = 100 for SPX. A global `scale` (default `1e-9`, ≈ $bn-gamma units) is applied;
  it is immaterial because only **standardized** GEX enters the regression.
- Robustness conventions logged alongside the primary: `all_long` (sign ≡ +1) and
  `naive_put_call` (signed but no dollar-gamma scaling). The decision rule keys on
  the SqueezeMetrics primary; conventions are reported as a sensitivity band.

Estimator correctness is unit-tested on a hand-built one-contract grid (exact
closed-form value), on call/put sign symmetry, and on sign + strict monotonicity
versus a known latent dealer gamma (`tests/test_gex_regression.py`).

---

## A5.3 The predictive regression

Outcomes (the pre-registered family, BH-controlled in A5.6):

- **Primary outcome** `y = RVV_{t+1}`: realized **vol-of-vol** — the rolling std
  (vv_window = 5d) of the rolling realized-vol series (rv_window = 5d) of daily
  log returns. Defined identically in simulator and in data (no latent variance
  path is used), so the synthetic validation transfers cleanly.
- **Secondary outcome** `y = CSD_{t+1}`: short-window **critical-slowing-down**
  signal — rolling lag-1 autocorrelation of `|r_t|` (window = 20d). Rising lag-1
  autocorr of the vol proxy is the early-warning signature (pre-reg §5, H4
  supporting).

Specification (strictly predictive — dependent at `t+1`, regressors at `t`):

```
y_{t+1} = b0 + b_GEX · z(GEX_t)
              + b1 · RV_t                 (lagged realized vol)
              + b2 · z(VIX_t)             (VIX level control)
              + Σ_d δ_d · DOW_{d,t}       (day-of-week dummies, Mon baseline)
              + ε_{t+1}
```

`z(·)` standardizes **within window** (so cross-event scale differences in raw
GEX are absorbed). Dating all regressors at `t` removes the mechanical same-day
GEX↔vol correlation and isolates the **forward feedback** claim, which is what
the mechanism actually asserts.

**Pooled primary estimator (the headline).** Stack the three event windows
(A5.5) into one panel with **event fixed effects** (per-event intercept dummies);
the single pooled `b_GEX` is identified off **within-window** variation — exactly
the short-vs-long-gamma contrast — and pools `≈ 3 × 113 ≈ 339` usable rows
(363 calendar rows minus the per-window warm-up for the rolling outcomes).
`run_h1prime_pooled` implements this. Each window is also fit on its own
(`run_h1prime`) and reported for transparency.

---

## A5.4 Inference

- **HAC (Newey–West).** OLS point estimate with Newey–West HAC covariance,
  Bartlett kernel, lag = `floor(4 (n/100)^{2/9})` (the standard rule-of-thumb;
  ≈ 4 lags at n ≈ 340), to handle the overlap induced by the rolling outcome
  windows and any residual serial dependence. Hand-rolled (no statsmodels
  dependency); validated against a known-slope DGP and an AR(1)-error DGP (HAC SE
  exceeds the homoskedastic SE) in the tests.
- **Moving-block bootstrap.** One-sided (`b_GEX < 0`) p-value from a
  moving-block bootstrap (block length 10 trading days, 2000 resamples,
  seed 42), which preserves serial dependence non-parametrically and is the
  primary inferential object given the modest n. Reports point estimate,
  bootstrap SE, and a 90% CI.

---

## A5.5 Event windows and the quiet-regime control

Same three pre-specified historical episodes as the original pre-reg (each
`t_event ± 60` ≈ **121 trading days**), so the data footprint is unchanged:

| Event | `t_event` | window |
|---|---|---|
| Volmageddon | 2018-02-05 | 2017-11-08 → 2018-05-04 |
| COVID crash | 2020-03-16 | 2019-12-19 → 2020-06-12 |
| Yen-carry unwind | 2024-08-05 | 2024-05-10 → 2024-10-30 |

Total ≈ **363 trading days** of EOD SPX OI + returns across the three events,
plus the quiet-regime control window below.

**Quiet-regime control (pre-specified).** A 121-day window of calm, range-bound
SPX with no macro stress event (locked candidate: **2017-05-01 → 2017-10-20**,
the low-VIX 2017 grind; final dates fixed in the amendment commit before data
load). The mechanism predicts the GEX→vol-of-vol effect is **weaker or absent**
here: with VIX pinned low and dealers persistently long gamma, the destabilizing
short-gamma regime is rarely entered, so there is little sign variation to
exploit. A confirmed effect in the event windows that **also** appears at full
strength in the quiet window would indicate a spurious mechanical channel rather
than the reflexive mechanism, and counts against H1'.

---

## A5.6 Decision rule

Reject H0 in favour of **H1'** iff **all** hold:

1. **Sign + significance (primary outcome, pooled panel).** `b_GEX < 0` with
   moving-block-bootstrap one-sided `p < 0.05` AND Newey–West one-sided
   `p < 0.05`.
2. **Multiple-testing across the outcome family.** Benjamini–Hochberg at FDR
   `q = 0.05` across the family {RVV, CSD} × {pooled, per-event} keeps the
   primary (pooled-RVV) rejection.
3. **Control contrast.** The quiet-window `|b_GEX|` is smaller than the pooled
   event-window `|b_GEX|`, and the quiet-window rejection at the same rule does
   **not** hold (effect weaker/absent off-event).

Any single failure ⇒ do not reject. The RL surface tournament (old H1) is
reported separately, in the **secondary** BH family, flagged as confounded per
A5.0; it can support but cannot establish the reflexivity claim.

---

## A5.7 Data-free synthetic validation (positive / negative / quiet controls)

Validated exactly like the original H4: on **simulator-generated** data where the
ground-truth coupling `kappa` is **known**, never on real data.
`gex_simulator.GEXReflexiveSimulator` produces, per day, a latent signed dealer
gamma `G_t` (persistent AR(1), `φ = 0.95`) **and** a synthetic OI grid whose
SqueezeMetrics GEX recovers `sign(G_t)` with monotone magnitude (unit-tested).
The grid is rendered at a **fixed reference IV** (`√θ`), not the realized vol,
so the BS-gamma weights do not leak the contemporaneous variance level into GEX
— estimated GEX is then a clean monotone function of the dealer-positioning state
`G_t` alone, eliminating a spurious same-sign GEX↔vol-of-vol channel that would
otherwise fight the mechanism. The variance feedback is the bounded map

```
s_t      = G_t / σ_G ;   f_t = −tanh(kappa · s_t) ∈ (−1, 1)   (short gamma ⇒ f_t > 0)
vov_t    = base_vov · exp(vov_gain · f_t)                      (vov_gain = 4.0; conditional vol-of-vol)
log rv_t = log θ_rv + φ_rv·(log rv_{t−1} − log θ_rv) + vov_t · ε_t   (φ_rv = 0.85, θ_rv = 0.12, base_vov = 0.12)
r_t      = (rv_t / √252) · z_t                                 (return drawn from the realized-vol level)
```

GEX modulates the **conditional vol-of-vol of a log-realized-vol AR(1)**: short-
gamma regimes (`f_t > 0`) raise `vov_t` exponentially, so the latent realized-vol
state `rv_t` is more *dispersed* over a run of days. Because each day's return is
drawn from `rv_t`, the downstream rolling realized-vol series inherits that
dispersion — realized **vol-of-vol**, the H1' primary outcome, rises with negative
GEX (measured GEX→vol-of-vol correlation ≈ −0.30 at `kappa = 0.8`, partial
correlation comfortably above the ≈ 0.14 detectability floor of A5.8). The latent
gamma `f_t` is **persistent** (inherits the AR(1) memory of `G_t`, `φ = 0.95`).
The log-realized-vol AR(1) is unconditionally stationary (and `log rv` is clamped
to a wide band purely to prevent fat-tail overflow, far outside the operating
range), so realized vol can never go negative or blow up at any `kappa` (0/40 NaN
paths). At `kappa = 0`, `f_t ≡ 0` and `vov_t ≡ base_vov` (pure log-AR(1), GEX a
null predictor — the negative control). `G_t` is drawn independently of the
contemporaneous return shock, so it is a non-mechanical predictor of `y_{t+1}`.
The synthetic OI grid is rendered at a **fixed reference IV** (`√θ`), not the
realized vol, so the BS-gamma weights never leak the vol level into GEX; estimated
GEX is then a clean monotone function of the dealer-positioning state `G_t` alone.
(This fixed-IV step is a simulator-only construction to keep the ground-truth
recovery test clean; on real WRDS data GEX is computed from the observed
per-contract IVs.)

Controls run by `gex_validation.run_validation`:

- **Positive control** — strong feedback (`kappa = 0.8`): `b_GEX < 0`, powered on
  the pooled panel, sign-correct on essentially every trial.
- **Negative control** — `kappa = 0`: rejection rate ≤ nominal, `b_GEX` ≈ 0.
- **Quiet control** — weak feedback (`kappa = 0.15`): a markedly lower rejection
  rate than the positive control, mirroring the empirical quiet-window prediction.

`tests/test_gex_regression.py` asserts the determinate facts: estimator exactness
and monotonicity, HAC behaviour, sign recovery at `kappa = 0.8` (single window
and pooled) plus pooled rejection power, non-rejection at `kappa = 0` for the
reference seed, and bounded false-positive rates across independent Heston panels.

**Measured results (50 trials; `runs/gex_validation/<ts>/report.json`, read back
from disk).** The stationary log-realized-vol AR(1) backbone (0/40 NaN paths)
gives clean, well-separated controls at the exact empirical footprint:

| Control | regime | pooled 3×121 | single 121-day |
|---|---|---|---|
| Positive (power) | `kappa = 0.8` | **0.86** | 0.50 |
| Sign-correct | `kappa = 0.8` | 0.98 | — |
| Negative (FPR) | `kappa = 0` (Heston) | **0.02** | **0.08** |
| Quiet (reject rate) | `kappa = 0.15` | 0.12 | — |

Mean pooled GEX coefficient is `−2.0e-03` at `kappa = 0.8` (negative, as H1'
predicts) and `+2.7e-05` at `kappa = 0` (centered at zero). Under the conjunctive
decision rule (`b_GEX < 0` AND both the moving-block bootstrap and one-sided
Newey–West HAC `p < α`), the negative-control FPR sits at/under the nominal 5%;
the **pooled** positive-control power is 0.86 (the primary estimator) with 0.98
sign recovery, while a single 121-day window alone rejects on only 0.50 —
**pooling the three events is what delivers the headline power, and single-window
inference is underpowered at this effect size and is not the primary estimator.**
The quiet regime rejects materially less often (0.12) than the strong-feedback
regime (0.86), exactly the ordering the empirical quiet-window contrast (A5.6 rule
3) relies on. This is the data-free analogue of the original H4 power/FPR
validation, on the redesigned test, at the real `3 × 121`-day data size. All 15
unit tests pass.

---

## A5.8 Detectability on 3 × 121 daily observations (quantitative)

The original H1's deepest practical risk was *power*: 12 pairwise W2 dominance
checks, each on ≈ 100 overlapping 21-day windows, with agent-training noise on
top. H1' is far more favourable, for three compounding reasons.

1. **One pooled scalar coefficient, not 12 dominance checks.** H1' tests a single
   sign on one pooled slope. The original required *all twelve* `(baseline ×
   event)` W2 gaps to clear non-overlapping CIs simultaneously — a conjunction
   whose joint power is the product of twelve marginal powers. Collapsing to one
   coefficient multiplies effective power by orders of magnitude.

2. **n ≈ 340, daily, with within-window identification.** The pooled panel has
   `≈ 3 × 113 ≈ 339` usable daily rows after rolling-window warm-up. For a simple
   one-sided test of a standardized slope, the non-centrality is
   `√n · (b_GEX / σ_resid-scaled) = √n · ρ_partial / √(1 − ρ_partial²)` where
   `ρ_partial` is the partial correlation of `z(GEX_t)` with `y_{t+1}`. To reach
   power ≈ 0.80 at one-sided α = 0.05 needs non-centrality ≈ 2.49, i.e.
   `|ρ_partial| ≳ 2.49/√339 ≈ 0.135`. **A partial correlation of only ~0.14 is
   detectable.** Dealer-gamma → forward-vol effects reported in the practitioner
   and academic literature (the GEX/realized-vol relationship) are comfortably
   larger than this in stress regimes, where short-gamma episodes are frequent
   and the sign varies — precisely the event windows H1' is run on.

3. **The synthetic power study confirms it at honest effect sizes.** The
   positive control (`kappa = 0.8`, GEX→vol-of-vol correlation ≈ −0.30) achieves
   pooled-panel power **0.86** with **0.98** sign recovery, while the `kappa = 0`
   negative control holds the false-positive rate at **0.02** (pooled) under the
   conjunctive bootstrap-and-HAC decision rule — demonstrating the test is both
   powered and calibrated at the exact data footprint the empirical phase will
   have. Single-window power is only 0.50, so the pooled 3-event panel is the
   load-bearing estimator. (Numbers recorded under
   `runs/gex_validation/<ts>/report.json`.)

The block bootstrap and Newey–West both target the overlap-induced serial
dependence from the rolling outcome, so the n ≈ 340 is not overstated by the
windowing.

**Data requirement.** EOD SPX option OI grid (strike × maturity × {OI, IV}) plus
SPX index level, for the three event windows + one quiet control window
(≈ 484 trading days total incl. control, ≈ 363 across the three events).
Obtainable in September from **WRDS OptionMetrics IvyDB US** (UofT institutional
access) or the ALLSPX fallback already budgeted in pre-reg §8. No 0DTE,
intraday, or tick data is needed; the test is purely end-of-day, which is the
cleanest and most widely available slice of the OptionMetrics panel.

---

## A5.9 Threats and mitigations

- **Dealer-sign convention.** SqueezeMetrics long-call/short-put is an
  assumption, not ground truth. Mitigation: report all three conventions; the
  decision rule keys on the primary but the sign of `b_GEX` is convention-stable
  for the loaded-leg construction, and a sign flip across conventions is flagged.
- **GEX ↔ VIX collinearity.** GEX and VIX co-move. Mitigation: VIX enters as a
  control and the regressor is the **standardized residual** GEX variation after
  controls; the within-window event fixed effects further absorb level shifts.
- **Endogeneity / reverse causation.** Forward dating (regressors at `t`, outcome
  at `t+1`) and the independence of the latent gamma innovation from the return
  shock (in the DGP) target the predictive, not contemporaneous, channel. On real
  data, reverse causation (vol drives OI) is bounded by EOD timing: OI is the
  prior-session settled position, fixed before the `t+1` returns realize.
- **Quiet-window date sensitivity.** Locked before data load; reported as a fixed
  pre-registration choice, not tuned post hoc.
