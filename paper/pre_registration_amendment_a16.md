# Pre-extraction amendment A16: timing-safe predictors and calendar-time inference

**Recorded:** 12 July 2026, before WRDS/OptionMetrics/CRSP extraction  
**Scope:** Prospective correction to Amendments A13--A15  
**Data status at recording:** No registered OptionMetrics option chain, CRSP return, or
Cboe VIX dataset has been queried, downloaded, loaded, or analyzed for this project.

This amendment changes the rules below because a clean-room audit found that vendor-date
open interest need not be observable at the regressor close, integer day-to-expiration can
misstate short-dated gamma, and inference on retained rows can compress gaps in the complete
CRSP session calendar. It also fixes persistent-level and option-surface confounding checks
before any registered outcome is opened. All A13--A15 rules not explicitly changed here
remain in force.

## A16.1 Information-availability timing

The primary regressor time is the SPX close on CRSP session \(t\). Let
\(\mathrm{OI}^{\mathrm{avail}}_{i,t}\) be the most recent contract-level open-interest record
documented as publicly/vendor-available by 16:00 America/New_York on session \(t\).

Before constructing returns, the extraction report must archive the vendor documentation,
query text, retrieval time, and SHA-256 hash that establish the date semantics and release
time of the OptionMetrics open-interest field. The following branch is determined only by
that documentation:

1. If the record dated \(t\) is documented as available by the close of \(t\) and contains
   no information generated after that close, use it.
2. Otherwise use the latest earlier record whose availability by the close of \(t\) is
   documented. If documentation is ambiguous, use the previous CRSP session's dated record.

Every option row must store quote session, OI source session, and OI availability lag in
sessions. Same-session quotes, IV, spot, rates, dividends, and forwards may be combined only
with \(\mathrm{OI}^{\mathrm{avail}}_{t}\). The primary outcome remains the close-to-close
CRSP return from \(t\) to the immediately following CRSP session. The availability branch
is frozen before any outcome column is extracted or joined.

## A16.2 Fractional time to settlement

Replace integer calendar DTE divided by 365 with an ACT/365 year fraction computed from
timestamps:

- valuation time: 16:00 America/New_York on the quote session;
- PM-settled expiration: the official expiration-session close;
- AM-settled expiration: the official settlement-morning timestamp;
- holiday and exceptional schedules: the official Cboe calendar archived with the raw
  manifest.

The pipeline must retain the AM/PM settlement flag and both timestamps. Contracts whose
settlement timestamp is not strictly after the valuation timestamp are excluded. The primary
range is \(0<\tau\leq1\) year after the existing date/settlement filters. Rates, dividends,
forwards, and Black--Scholes gamma use the identical fractional \(\tau\).

## A16.3 Quote-liquidity gate

The primary contract sample requires a strictly positive bid, ask greater than bid, and
relative spread \((ask-bid)/mid\leq0.50\), in addition to the A14 IV, moneyness, adjustment,
root, multiplier, and settlement rules. The report must show contract counts and
pre-filter gamma-mass shares for zero-bid, crossed/locked, and relative-spread buckets.

Two prespecified measurement sensitivities are reported: a stricter 0.25 relative-spread
cap and the former nonnegative-bid rule. Neither sensitivity can replace the primary sample.

## A16.4 Complete-calendar HAC and block bootstrap

The primary Newey--West covariance uses Bartlett weights and lag 22 measured in actual CRSP
session offsets. Missing option rows contribute no score product and do not turn nonadjacent
sessions into adjacent observations. The covariance uses no finite-sample sandwich
multiplier; two-sided \(p\)-values use Student \(t_{n-k}\) tails after a full-rank OLS fit.

The primary moving-pairs bootstrap uses length-22 blocks on the complete CRSP session index.
Starting sessions are sampled uniformly from all overlapping calendar blocks. Within each
selected block, only retained outcome-and-design rows enter the refit; blocks are concatenated
until \(n\) retained rows are obtained and the last block is truncated. The same sampled
starts are used for all four primary proxies. Rank-deficient draws are discarded and
resampled until 2,000 full-rank draws are obtained; attempted and discarded draw counts are
reported. The seed remains 42 and the interval remains the 95% percentile interval.

The former lag-5 HAC and length-10 retained-row bootstrap are reported only as sensitivities.
They cannot determine the primary evidence label.

## A16.5 Persistent levels and pricing-input decomposition

The four A13 summaries are renamed **OI-and-pricing-derived book summaries** because their
gamma weights also depend on spot, IV, rates, dividends, maturity, and forward moneyness.
They are not pure position measures.

All four primary regressions add \(\log S_t\) and a deterministic linear CRSP-session trend
to the A14 control set. The report also includes, without changing the primary label:

1. year-fixed-effect and first-difference specifications;
2. a scale-free mass \(\log\{U_t/\sum_i\mathrm{OI}^{\mathrm{avail}}_{i,t}\}\);
3. stability estimates for 2017--2019, 2020--2022, and 2023--2024;
4. OI-only call--put, mean-moneyness, and dispersion summaries;
5. gamma-weighted summaries recomputed with fixed volatility 20%, \(r=q=0\), and the
   registered spot, forward, maturity, and contract geometry.

These decompositions distinguish persistence and contemporaneous option-surface inputs from
changes in the open-interest cross-section. They are measurement diagnostics, not additional
primary hypothesis families.

## A16.6 Precision language

The minimum-detectable-effect table remains a scenario calculation. No realized effective
sample-size estimator is selected. The final report gives raw \(n\), \(R_Q^2\), residual
standard deviation, complete-calendar HAC uncertainty, and calendar-block-bootstrap
uncertainty directly.

## A16.7 Fail-closed implementation gate

Before registered outcomes are constructed, production code must:

- require explicit parity/flagged-carry forwards and the matching rate/dividend tuple;
- require multiplier 100 and validated boolean call/put mapping;
- retain root, option identifier, adjustment, settlement, quote, expiration, and timing
  fields needed for every registered filter;
- produce a sequential attrition table and unresolved-duplicate stop;
- freeze raw option manifests, query text, documentation, row counts, date coverage, and
  SHA-256 hashes.

The existing utility functions are not represented as a complete WRDS extraction pipeline.
Day-one validation and this implementation gate must pass before outcomes are joined.

## A16.8 Reporting and deviations

The final paper must reproduce this amendment's hash and timestamp status, report the
availability branch chosen from vendor documentation, and list every deviation from
A13--A16. No post-data result may be used to choose the OI lag, settlement clock, spread
filter, HAC lag, bootstrap block length, trend control, or measurement decomposition.
