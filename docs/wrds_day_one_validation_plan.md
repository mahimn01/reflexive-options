# WRDS day-one validation plan

**Purpose.** The first WRDS session validates *access, coverage, and field mapping* against the
locked pre-registration — it does **not** run any pre-registered test. Everything here is
designed so that the confirmatory H1′/H4 pipeline runs exactly once, later, on a fully
specified panel. Execute top to bottom; every step has a pass criterion. Estimated time: one
sitting.

**Pre-registration hygiene (read first).**

- Day one touches *regressor-side* raw data (option chains, index levels) on **two dates only**
  (steps 5–6). It computes **no outcome variable** (no RVV, no CSD statistic, no regression).
- One of the two dates is inside the COVID event window (unavoidable: the GEX sign check is
  only informative on a stress date). This is a data-quality validation, not an analysis;
  record it in the disclosure log (step 9) per pre-registration §9.
- If any pass criterion fails, stop and fix the mapping — do not improvise around the locked
  spec. Deviations, if any become necessary, go through the §9 disclosure route.

**Locked constants used throughout** (source: `paper/pre_registration.md` §2/§4 + amendments
A8/A9/A11; do not re-derive, do not change):

| Item | Locked value |
|---|---|
| Maturity pillars 𝒯 | {7, 14, 30, 60, 90, 180, 365} calendar days (M = 7) |
| Strike grid | k = log(K/F) ∈ {−0.20, −0.16, …, +0.16, +0.20}, Δk = 0.04 (K = 11) |
| Interpolation | cubic spline in k at each τ, then linear in τ |
| Arbitrage filter | 4 checks (butterfly/Durrleman g ≥ −1e−4; w non-decreasing in τ; calendar C(K,T₂) ≥ C(K,T₁); Lee wing slope ≤ 2), tolerance 1 bp in total variance |
| Event windows (±60 td) | Volmageddon 2017-11-07 → 2018-05-02 (event 2018-02-05); COVID 2019-12-17 → 2020-06-10 (event 2020-03-16); Yen carry 2024-05-08 → 2024-10-29 (event 2024-08-05) |
| CSD confirmatory records | ±126 td (252-td record) around each event ⇒ Yen record extends to **2025-02-05** (NYSE closed 2025-01-09, Carter mourning day). Note: the CSD statistic is computed on **CRSP index returns**, not option data — this date gates the CRSP pull (step 7), not the OptionMetrics vintage. |
| Quiet-regime control | 2017-05-01 → 2017-10-20 |
| GEX (locked, A9) | GEX(t) = Σ_{K,T} OI_{K,T} · Γ_BS · sign_{K,T} · S² · 0.01 · mult; SqueezeMetrics dealer-sign primary (calls +, puts −); `all_long` and `naive` reported as sensitivity band; mult = 100 |
| Dating (locked DOF) | forward dating: regressors dated t, outcome t+1 |
| Data sources (pre-reg §8) | WRDS OptionMetrics IvyDB US (options); SPX index level from CRSP daily |

---

## 1. Access setup

```bash
pip install wrds
python -c "import wrds; db = wrds.Connection(wrds_username='mahimn'); print(db.list_libraries()[:20])"
```

First connection interactively creates `~/.pgpass`. **Pass:** connection succeeds and both
`optionm` (OptionMetrics) and `crsp` appear in `list_libraries()`.

If `optionm` is missing: UofT's WRDS subscription does not include IvyDB US → **stop**;
fallback is the commercial ALLSPX bundle (pre-reg §8 fallback, $805).

## 2. Coverage gate (run before anything else)

IvyDB option prices on WRDS are **year-partitioned**: `optionm.opprcd1996` … `optionm.opprcdYYYY`.

```python
tables = db.list_tables(library="optionm")
years = sorted(int(t[6:]) for t in tables if t.startswith("opprcd") and t[6:].isdigit())
print(years[-3:])
latest = db.raw_sql(f"select max(date) as maxd from optionm.opprcd{years[-1]}")
```

**Pass (hard gate):** partitions exist for 2017, 2018, 2019, 2020, and 2024, **and**
`max(date) ≥ 2024-10-29` — the Yen window end, which is the option-data requirement of the
locked pooled panel. The 252-td CSD records (to 2025-02-05) are computed on CRSP returns
(step 7), not option data, so they impose no `optionm` requirement; a 2025 partition is a
bonus, not a gate.

**Fail mode:** if `optionm` data ends before 2024-10-29, the Yen window is unusable → the
pooled panel drops to 2 events and the pre-registered power analysis no longer applies →
disclose and escalate before proceeding. (A vintage ending between 2024-10-29 and 2025-02-05
is fine for every locked test — only CRSP must reach 2025-02-05.)

## 3. Identifier resolution

```python
db.raw_sql("select secid, ticker, issuer, issue_type from optionm.securd1 where ticker = 'SPX'")
```

**Expected:** SPX (S&P 500 index) at `secid = 108105` — verify from the lookup, do not
hardcode blind. Then dual-source the index level on 2019-06-12:

```python
db.raw_sql("select date, close from optionm.secprd where secid = 108105 and date = '2019-06-12'")
db.raw_sql("select caldt, spindx, sprtrn from crsp.dsp500 where caldt = '2019-06-12'")
```

**Pass:** `optionm.secprd.close` and `crsp.dsp500.spindx` agree to rounding (< 0.1 index pt).
The pre-registered index-level source is **CRSP** (`dsp500.spindx` / returns `sprtrn`);
`secprd` is the cross-check.

## 4. Field mapping (locked-spec ↔ IvyDB)

| Locked-spec quantity | IvyDB column (opprcdYYYY) | Transform / note |
|---|---|---|
| Strike K | `strike_price` | **divide by 1000** |
| Option type | `cp_flag` | 'C' / 'P' |
| Expiry | `exdate` | τ = (exdate − date) in calendar days |
| Mid price | `best_bid`, `best_offer` | mid = (bid+offer)/2; drop if bid ≤ 0 |
| Implied vol | `impl_volatility` | NULL when vendor could not compute — count NULL rate, drop NULLs from surface, log the rate |
| Open interest | `open_interest` | **IvyDB convention: OI reported for date t is as of the close of t−1** (post-Nov-2000 convention). Under the locked *forward dating* rule (regressors at t, outcome t+1) this is information available at t ⇒ no lookahead. Verify empirically in step 5; document the convention in the disclosure log. |
| Greeks | `gamma` (also `delta`,`vega`,`theta`) | cross-check only; production GEX uses our own Γ_BS from impl_volatility (A9 formula) |
| Special settlement | `ss_flag` | drop `ss_flag != '0'` |
| Root / settlement | `symbol`, `am_settlement` | SPX (AM-settled monthlies) and SPXW (PM weeklies) both live under secid 108105. **GEX aggregates the full chain (SPX + SPXW)**; the 7×11 surface uses all listed expiries then interpolates to pillars. |
| Multiplier | — | 100 (locked `mult`) |
| Risk-free r | `optionm.zerocd` | linear interpolation in days to τ |
| Dividend yield q | `optionm.idxdvd` | index dividend yield at date |
| Forward F for k = log(K/F) | derived | **primary: put-call parity** at the nearest-ATM strike pair per expiry, F = K + e^{rτ}(C_mid − P_mid); cross-check: F = S·e^{(r−q)τ} from zerocd+idxdvd. The pre-reg does not fix the F construction; adopting parity-F is an implementation note recorded pre-analysis in the disclosure log (not a deviation). |

## 5. Validation date A — calm, outside all windows: **2019-06-12**

Pull the full SPX chain for one day:

```python
q = """select date, symbol, exdate, cp_flag, strike_price, best_bid, best_offer,
              volume, open_interest, impl_volatility, delta, gamma, ss_flag, am_settlement
       from optionm.opprcd2019
       where secid = 108105 and date = '2019-06-12'"""
chain = db.raw_sql(q)
```

Checks (all must pass):

1. **Row count sanity:** full SPX+SPXW chain in 2019 ≈ 8k–20k rows. Zero or tiny ⇒ mapping wrong.
2. **NULL IV rate** < 25% overall, and near-money (|k| ≤ 0.05) NULL rate < 5%.
3. **Parity-F vs carry-F** agree within 0.2% at the front pillars.
4. **Greeks recompute:** our Black–Scholes Γ (from impl_volatility, S, K, r, q, τ) vs IvyDB
   `gamma`, near-money 30–90d contracts: median relative difference < 2%. This validates the
   exact Γ_BS entering the locked GEX formula.
5. **Grid build:** construct the 7×11 surface via the locked interpolation (cubic in k, linear
   in τ). All 77 nodes populated (wings at |k| = 0.20 may extrapolate at the 7-day pillar —
   log how many nodes needed extrapolation).
6. **Arbitrage filter:** run all four locked checks on the built surface. A calm 2019 day
   should pass all four; investigate any failure as a pipeline bug, not a data event.
7. **OI convention probe:** pull 2019-06-11 + 2019-06-13 too; for the 20 highest-volume
   contracts confirm OI changes lag volume by one day (i.e., OI(t+1) − OI(t) tracks net
   activity on t, not t+1).

## 6. Validation date B — stress, GEX sanity only: **2020-03-16**

Pull the chain from `optionm.opprcd2020`, same query. Compute GEX(t) under all three locked
sign conventions (SqueezeMetrics primary, `all_long`, `naive`) using the A9 formula with our
own Γ_BS. Checks:

1. **Sign:** SqueezeMetrics-convention GEX on 2020-03-16 should be **deeply negative**
   (dealers short gamma in the crash) — the mechanism's own precondition.
2. **Magnitude order:** |GEX| per 1% move in the $10⁹–$10¹¹ range (order-of-magnitude only;
   this is a units check on S²·0.01·mult, not a benchmark).
3. **Convention band:** the three conventions differ (they must — that is the sensitivity
   band); record all three values in the validation report.

Explicitly **not** computed: RVV, CSD statistics, any regression, any other date in any event
window.

## 7. Returns leg (CRSP)

```python
r = db.raw_sql("""select caldt, spindx, sprtrn from crsp.dsp500
                  where caldt between '2016-10-01' and '2025-03-31'""")
```

**Pass:** no gaps against the NYSE calendar (`pandas_market_calendars`, the same calendar that
generated `paper/event_windows.txt`); coverage spans 2016-10-01 → 2025-03-31 (all three
252-td CSD records + quiet window + burn-in).

## 8. Go / no-go

**GO** = all of: coverage gate passed (≥ 2025-02-05); secid resolved; index dual-source agrees;
all 7 checks on date A pass; GEX sign/magnitude/band sane on date B; CRSP leg complete.

GO unlocks the full extraction (a separate, second session): per event window, query
`opprcdYYYY` for secid 108105 over [window_start − 60 td, window_end]. The extra 60 td ahead
of each window covers the locked calibration window (60 td ending at t_event − 60, pre-reg
§2/§8 — it feeds κ₀ for the retained secondary RL tournament, H2, H3, and the ablations) and
subsumes the 30-td burn-in. That is ~181 trading days × ~10–25k rows/day ≈ 2–4.5M rows per
window — chunk by month, write to `data/optionmetrics/` (gitignored), record SHA256 + row
counts per chunk in the extraction log. Then and only then:
the locked pipeline, run once.

**NO-GO** paths: missing `optionm` → ALLSPX fallback ($805). Vintage short of 2024-10-29 →
2-event panel, disclose + re-assess power before running anything. Field-mapping failures →
fix mapping, re-run day one; the locked spec does not move.

## 9. Disclosure log (start it day one, append-only)

Record in `data/optionmetrics/DISCLOSURE_LOG.md`: (i) dates touched and why (2019-06-12
mapping validation; 2020-03-16 GEX sanity — regressor-side only, no outcomes computed);
(ii) the OI dating convention as verified in step 5.7; (iii) the parity-F implementation note;
(iv) any deviation candidates with their §9 justification. This log is quoted verbatim in the
paper's Phase-4 methods disclosure.
