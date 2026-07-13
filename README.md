# reflexive-options

[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![paper](https://img.shields.io/badge/paper-PDF-blue)](paper/main.pdf)

Research code for *Gamma-Shaped Dealer-Book Pressure and Endogenous Volatility Cycles: A
Reduced-Form Fixed-Equilibrium Hopf Model and a Pre-Extraction Predictive Protocol*
(Patel, 2026).

## Current claim

The paper gives a local physical-measure model for detrended log price $X$,
instantaneous variance $v$, and filtered price memory $\chi$:

$$
\begin{aligned}
dX_t&=[-\delta X_t-\tfrac12(v_t-\theta_v)+\kappa g(X_t,v_t,\chi_t)]dt
      +\sqrt{v_t}\,dW_t^S,\\
dv_t&=[\kappa_v(\theta_v-v_t)+\gamma v_t\chi_t]dt
      +\xi\sqrt{v_t}\,dW_t^v,\\
d\chi_t&=\alpha(\beta X_t-\chi_t)dt.
\end{aligned}
$$

The dealer-book functional is centered so $(0,\theta_v,0)$ is an equilibrium
for every coupling $\kappa$. The $v\chi$ variance feedback leaves the boundary
drift equal to $\kappa_v\theta_v>0$ at $v=0$.

For a Gaussian density of **signed dealer positions** in fixed log moneyness,
the Routh--Hurwitz Hopf determinant is quadratic in $\kappa$. The transparent,
non-calibrated example reproduces

- $\kappa^\star=31.4932976\,\mathrm{yr}^{-1}$,
- $\omega^\star=47.1185670\,\mathrm{rad}\,\mathrm{yr}^{-1}$,
- $\ell_1=-6.2888041$,
- a nonlinear attracting cycle at $1.02\kappa^\star$ with
  $v\in[0.04453,0.08187]$.

The same quadratic also has a remote second valid point at
$\kappa^{\star\star}=16860.8961\,\mathrm{yr}^{-1}$, where the pair crosses back
into the stable half-plane. The API returns both roots; plots and phase maps
labeled “threshold” use the first.

On the closest numerical grid, the actual nonlinear cycle amplitude scales as
$\{(\kappa-\kappa^\star)/\kappa^\star\}^{0.509}$, close to the local Hopf
exponent $1/2$. A gross-normalized mixture audit is deliberately less tidy:
nearby same-sign mixtures remain supercritical, but dispersed and
offsetting-sign books can be subcritical, and reversing the canonical sign
orientation removes the valid positive root. The existence and type of the
local bifurcation are therefore book-dependent.

This is a **local deterministic possibility result** for a postulated
gamma-shaped pressure term inside an independently assumed
price--memory--variance loop. It is neither an SPX calibration nor evidence
that markets are near a Hopf threshold. The canonical crossing is driven
mainly by the book kernel's variance sensitivity, not delta hedging alone.

## Measurement boundary

Public option open interest counts outstanding contracts. It does not identify
which side is held by a dealer. The theory's signed position density is
therefore latent; an OI-weighted signed GEX series is a convention-dependent
proxy, not observed dealer inventory.

Amendments A13--A16 replace the former event-selected directional GEX test with
a pre-extraction registered-horizon protocol. On eligible OptionMetrics dates from
2017-01-03 through 2024-10-29 it will study four observable summaries:

1. nonnegative OI-gamma mass;
2. call--put composition;
3. gamma-weighted mean log moneyness;
4. gamma-weighted log-moneyness dispersion.

The contract universe, parity/carry forward rule, rate/dividend tuple, spot and
return sources, OI availability rule, fractional settlement time, liquidity
filters, and duplicate/attrition policy are fixed before access. Leads, lags,
HAC offsets, and bootstrap blocks remain on the complete CRSP trading-session calendar, so
a missing option date cannot compress the next-session outcome. These summaries
predict next-session log squared CRSP returns with 22-session inference horizons,
official Cboe VIX, log spot, a linear session trend,
outcome-session weekday indicators, and one regressor-session monthly-expiration
control. HAC and moving-block-bootstrap p-values are BH-adjusted as separate
families. “Robustly associated” requires both adjusted p-values below 0.05 and
a 95% bootstrap interval excluding zero; a one-family rejection is explicitly
method-sensitive. Convention-signed GEX and stress-window interactions are
secondary. Even a robust association is not causal or dealer-sign
identification.

No registered WRDS, OptionMetrics, CRSP, or VIX dataset has been extracted or
analyzed in the project. The 2017--2024 market path and named stress episodes
were historically public when the plan was written; this is a retrospective
pre-analysis plan, not a blinded prospective experiment. Access is expected in
September 2026.

## Reproduce the current result

```bash
uv sync --locked --all-extras --group dev

# Core analytic and empirical-protocol checks
uv run pytest -q tests/test_centered_model.py tests/test_oi_proxy_protocol.py

# Rebuild the actual-nonlinearity validation figure
uv run python -m reflexive_options.experiments.centered_hopf_validation

# Rebuild amplitude, sensitivity, mixture, and derivative robustness checks
uv run python -m reflexive_options.experiments.centered_model_robustness

# Rebuild the paper
cd paper && make pdf

# Full repository verification
cd .. && bash scripts/verify.sh
```

The principal implementation is in:

- `src/reflexive_options/theory/centered_model.py`
- `src/reflexive_options/experiments/centered_hopf_validation.py`
- `src/reflexive_options/experiments/centered_model_robustness.py`
- `src/reflexive_options/empirical/oi_proxy_protocol.py`
- `paper/main.tex`
- `paper/pre_registration_amendments.md` (preserved A13--A14 record)
- `paper/pre_registration_amendment_a15.md` (calendar and disclosure correction)
- `paper/pre_registration_amendment_a16.md` (timing, settlement, inference, and measurement correction)
- `docs/wrds_day_one_validation_plan.md`

## What was withdrawn from v0.3

The repository retains older modules and artifacts for reproducibility, but the
current paper does **not** claim:

- a non-zero stochastic threshold correction from the affine additive
  surrogate; its tangent cocycle is $e^{Jt}$ and the correction is zero. The
  full state-dependent stochastic variational equation remains unanalyzed;
- a Hawkes--SV equivalence theorem;
- global stability from absence of a Hopf root;
- a McKean--Vlasov threshold theorem for the centered model;
- an information-theoretic critical-edge theorem;
- stationary-tail or bimodality consequences of the local Hopf result;
- that synthetic RL, CSD, or event-window exercises validate the market
  mechanism.

Affected modules are marked archived/exploratory. The obsolete 4D noise scan
is no longer an installed command.

## Repository map

```text
src/reflexive_options/
├── theory/          # centered current model plus archived legacy utilities
├── empirical/       # A13--A16 OI proxy utilities and legacy A9 reproducer
├── experiments/     # current validation plus archived experiments
├── simulator/       # legacy/full simulator infrastructure
├── baselines/       # comparison simulators
├── surface/         # IV-surface and arbitrage-filter utilities
├── rl/              # exploratory agent infrastructure
└── third_party/     # vendored ATLAS/RAT code

paper/
├── main.tex
├── main.pdf
├── references.bib
├── pre_registration.md
├── pre_registration_amendments.md
└── variants/
```

## Pre-registration provenance

The original document and earlier amendments retain their historical
OpenTimestamps proofs. A13--A16 are disclosed pre-extraction amendments made
before anticipated WRDS access. The preserved A13--A14 amendment-file SHA-256 is
`603e89366c0dbe49718e8c31f805d6f85d3c508e2e0ed4276a6310c80f5f9cd7`, with
receipt `paper/pre_registration_amendments.md.ots` pending Bitcoin
consolidation. The A13-only snapshot `paper/pre_registration_amendments.md.a13`
remains independently verifiable at hash
`83950ede7049cec9842246cb291307a144bd620923898e9793c2f1197f558e17` with
receipt `paper/pre_registration_amendments.md.a13.ots`. Do not rewrite either
historical state to imply that a later clarification existed earlier.
The separate A15 correction has SHA-256
`a5f694f99953d57563d4f17dc5646ef0b87452c45119ccbdc12fc90efd034a52`
and receipt `paper/pre_registration_amendment_a15.md.ots`, pending Bitcoin
consolidation at creation.
The separate A16 correction has SHA-256
`a5cbf9ef56c9a402ff05b61bb720d8487313b154f30393b4835b43fe5c33e61d`
and receipt `paper/pre_registration_amendment_a16.md.ots`, pending Bitcoin
consolidation at creation.

## Citation

```bibtex
@unpublished{patel2026dealergamma,
  author = {Patel, Mahimn},
  title  = {Gamma-Shaped Dealer-Book Pressure and Endogenous Volatility Cycles:
            A Reduced-Form Fixed-Equilibrium Hopf Model and a Pre-Extraction
            Predictive Protocol},
  year   = {2026},
  note   = {Working paper, v0.4.1},
  url    = {https://github.com/mahimn01/reflexive-options}
}
```

## License

Code is MIT-licensed. The manuscript and its original figures are licensed
under CC BY 4.0; see [paper/LICENSE.md](paper/LICENSE.md). Vendored ATLAS/RAT
modules derive from `mahimn01/trading-algo`; see [NOTICE](NOTICE).
