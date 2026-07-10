# WRDS day-one validation plan (A13--A15)

**Timing.** University of Toronto access is expected in September 2026. Run this mapping
validation in the first access week and freeze the full option extraction promptly.

**Purpose.** Day one validates access, coverage, identifiers, units, and deterministic
field mappings. It does not construct a return outcome, join VIX to a proxy, estimate a
regression, inspect an event statistic, or test an economically desired sign. Amendments
A13--A14 in `paper/pre_registration_amendments.md`, supplemented by A15 in
`paper/pre_registration_amendment_a15.md`, are the operative specification.

## 1. Data firewall

- Allowed before the option freeze: schema descriptions, table coverage, row counts,
  identifier/root resolution, missingness, filter attrition, parity/carry/vendor-forward
  comparisons, and recomputed-versus-vendor gamma on the two dates below.
- Forbidden before the option freeze: `sprtrn`, constructed return outcomes, VIX joins,
  proxy/outcome correlations, regressions, CSD statistics, event comparisons, and claims
  that any date should have a particular signed-GEX value.
- Save every query, raw response, touched date, package/table vintage, and mapping decision
  in the append-only disclosure log.
- If coverage or mapping fails, stop before outcomes. Do not alter dates, filters, proxy
  definitions, or inference after viewing returns.

## 2. Access, schema, and coverage gate

Connect with the official `wrds` client and confirm access to `optionm` and `crsp`. Record
the library and table listings because IvyDB field availability can vary by WRDS vintage.
Price tables are normally partitioned as `optionm.opprcdYYYY`; verify rather than assume.

**Hard coverage pass:** OptionMetrics spans 2017--2024 and reaches 2024-10-29. If the
available vintage ends earlier, record the failure and stop. Do not shorten the registered
sample after viewing any outcome.

Before writing the full query, confirm that the 2017--2024 tables expose or can be joined to
all required fields: `secid`, `date`, `symbol`, `optionid`, `exdate`, `cp_flag`,
`strike_price`, `best_bid`, `best_offer`, `open_interest`, `impl_volatility`, vendor
`gamma`, `cfadj`, `contract_size`, `ss_flag`, and `am_settlement`. Also locate security
close, forward-price, zero-curve, and index-dividend tables. A missing required field is a
mapping issue to disclose and resolve before extraction, not a license to improvise later.

## 3. Identifier, root, and spot mapping

Resolve SPX through OptionMetrics security history. The expected `secid` is 108105, but
verify the active ticker/issuer record and index classification. On 2019-06-12:

1. confirm exactly one positive `optionm.secprd.close` for that `secid` and date;
2. compare it descriptively with `crsp.dsp500.spindx` and investigate any material unit or
   date mismatch;
3. parse the whitespace-normalized OSI root from `symbol` and demonstrate that the retained
   roots are exactly `SPX` and `SPXW`;
4. verify that `cfadj = 1`, `contract_size = 100`, and `ss_flag = '0'` mean unadjusted,
   standard-settlement contracts in the installed data dictionary.

The OptionMetrics close is the registered spot input for forward/gamma construction. The
CRSP level is a cross-check; CRSP later supplies only the registered index return.

## 4. Locked field and unit map

| Quantity | Primary field/source | Locked rule |
|---|---|---|
| Spot | `optionm.secprd.close` | one finite positive SPX value per date |
| Strike | `strike_price` | divide by 1,000 after verifying raw units |
| Type | `cp_flag` | exactly `C` or `P` |
| Expiry | `exdate` | 1--365 calendar DTE; same-day excluded |
| Quote | `best_bid`, `best_offer` | offer $>0$, bid $\geq0$, offer $\geq$ bid |
| Implied vol | `impl_volatility` | finite in [0.01, 5.00]; no imputation |
| Open interest | `open_interest` | finite, non-negative, exactly vendor-dated |
| Identity | `optionid` | unique within date after exact-duplicate collapse |
| Contract | `symbol`, `cfadj`, `contract_size`, `ss_flag` | SPX/SPXW, 1, 100, `0` |
| Settlement | `am_settlement` | forward groups split by AM/PM flag |
| Rate | OptionMetrics zero curve | continuous rate; linear in calendar days |
| Dividend | OptionMetrics index-dividend file | annualized continuous SPX yield |
| Forward | matched call/put mids | parity primary; flagged carry fallback |
| Gamma | recomputed Black--Scholes spot gamma | vendor gamma is a cross-check only |

Verify whether rate/dividend fields are stored as decimals or percentage points by matching
vendor documentation and a hand calculation. Do not choose a conversion because it makes
gamma look plausible.

## 5. Deterministic contract pipeline

For each date, record rows remaining and OI/gamma mass after every stage:

1. verified SPX `secid`;
2. OSI root exactly SPX or SPXW;
3. `cfadj = 1`, `contract_size = 100`, `ss_flag = '0'`;
4. finite positive spot/strike and 1--365 calendar DTE;
5. valid bid/offer, finite IV in [0.01, 5.00], finite non-negative OI;
6. unique date--`optionid` after collapsing and counting exact duplicate rows;
7. forward assigned by date--expiration--AM/PM group;
8. $|\log(K/F)|\leq0.50$;
9. finite recomputed gamma and positive aggregate gamma-weighted OI mass.

Non-identical duplicate date--option identifiers are a no-go. Zero-OI rows remain in
attrition counts, then may be dropped computationally. No daily value is winsorized,
clipped, or deleted because it is extreme.

## 6. Forward and gamma arithmetic

Within each date--expiration--`am_settlement` group, match valid call and put mids at the
same strike. Select the pair minimizing $|C_{mid}-P_{mid}|$; an exact tie uses the smaller
strike. Compute

$$F=K+e^{r\tau}(C_{mid}-P_{mid}).$$

If no pair survives, use $F=Se^{(r-q)\tau}$ and flag the group. Interpolate the continuous
zero rate linearly in calendar days; nearest-endpoint extension is allowed only outside the
available curve and is separately flagged. Use ACT/365 throughout. Preserve descriptive
differences from the vendor forward and carry forward; neither comparison is a fit-based
selection rule.

Recompute

$$d_1={\log(S/K)+(r-q+\sigma^2/2)\tau\over\sigma\sqrt\tau},\qquad
\Gamma^{BS}={e^{-q\tau}\phi(d_1)\over S\sigma\sqrt\tau}.$$

The exact same $(S,K,r,q,\tau,\sigma)$ tuple feeds gamma and the forward/moneyness pipeline.
Reconcile selected rows by hand and report vendor-gamma relative differences; investigate
systematic discrepancies before proceeding, but do not delete contracts merely because
vendor and recomputed gamma differ.

## 7. Two mapping dates

**2019-06-12:** exercise ordinary-chain mapping, roots, strike scale, settlement grouping,
zero-rate interpolation, dividend units, parity selection, OI dating, and gamma
recomputation. Report full-chain and near-money IV missingness.

**2020-03-16:** exercise the identical code on a stress date to detect overflow, unit, and
wide-market problems. Outputs must be finite and contract-level arithmetic must reconcile.
There is no expected sign or external magnitude target.

The old 7-by-11 IV surface and its arbitrage filter are not part of the primary A13--A15
pipeline and cannot be used to delete primary contracts or days.

## 8. Freeze options before outcomes

After both mapping dates pass, extract every eligible option date from 2017-01-03 through
2024-10-29 in monthly chunks. For each raw chunk and the combined manifest preserve:

- exact query text and parameters;
- table vintages and retrieval timestamps;
- first/last date, distinct dates, rows, and distinct option identifiers;
- byte size and SHA-256;
- missingness, duplicates, filter attrition, parity/carry/endpoint flags, and exclusions.

Store raw files in the gitignored data area and make the frozen manifest append-only. A
re-extraction gets a new manifest; never silently replace a frozen file.

## 9. Outcomes and controls, only after the option freeze

1. Pull `crsp.dsp500.sprtrn`; reject `sprtrn <= -1`; compute `log1p(sprtrn)` only after
   archiving the raw CRSP response and query.
2. Download the official Cboe 1990--present VIX daily-close CSV from the Cboe historical
   data page, preserving the unmodified file, resolved URL, retrieval timestamp, and hash.
   Do not interpolate, fill, or substitute another VIX source.
3. Build the complete ordered CRSP trading-session calendar. Construct returns and the
   one-, five-, and 22-session controls on that calendar before removing any session with
   a missing option proxy.
4. Generate Tuesday--Friday indicators for the immediately following outcome session, with
   Monday omitted, and exactly one standard-monthly expiration indicator for the regressor
   session (third Friday, or preceding open session if closed).
5. Join proxies and VIX without compressing the CRSP calendar, count listwise deletions,
   and reserve the first 22 CRSP sessions for lag initialization. Do not run a proxy
   correlation during this step.

Freeze the CRSP and Cboe manifests separately and after the option-manifest timestamp.

## 10. Analysis integrity gate

Before running the four regressions, assert in code and log:

- four separately standardized proxies in registered order;
- immediately following CRSP-session log squared return outcome, even if that outcome
  session has no option proxy;
- fixed controls and a single monthly-expiration indicator;
- HAC lag 5;
- moving-pairs blocks of length 10, 2,000 draws, NumPy seed 42 with common row indices
  across regressions, 95% percentile interval,
  and Monte Carlo-corrected two-sided sign p-value;
- separate BH adjustments over four HAC and four bootstrap p-values;
- labels restricted to `robustly_associated`, `method_sensitive`, or `not_detected` under
  A14's conjunctive rule.

## 11. Go/no-go and disclosure

**GO:** coverage reaches 2024-10-29; identifier/root/spot checks pass; required fields and
units are resolved; duplicate policy is satisfied; both mapping dates reconcile; and the
option and outcome/control manifests are frozen in that order.

**NO-GO:** missing access, short vintage, unresolved identifier/root, unavailable required
field, non-identical duplicate, unexplained unit mismatch, or failed arithmetic. Resolve
and disclose before analysis. Never relax the sample or invent a sign benchmark.

The append-only log must report every touched date and purpose, OI-timing documentation,
all mapping decisions and failures, every hash, daily exclusions, and any deviation with
its timing relative to outcome construction. Publish the log even if all results are null.
