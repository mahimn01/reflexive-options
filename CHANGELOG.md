# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.12] - 2026-07-06

### Fixed — arXiv pre-flight verification (pre-submission)

A final multi-agent pre-flight on the actual submission tarball caught one
referee-falsifiable blocker on the paper's core credibility claim, plus minor items:

- **Pre-registration OTS anchor commit corrected (BLOCKER).** `main.tex` cited commit
  `268c061` as carrying the OpenTimestamps proof at four sites. That commit's
  `pre_registration.md` hashes to `79957f08…` and is stamped by no proof. The real anchor
  is **`63078f5`** (`f968b22e…` = `pre_registration.md.v1.ots`, the original pre-data lock
  where A1–A7 close); the amended document (`ccae3ba2…`) is stamped by
  `pre_registration.md.ots` at `764c2d3`. A referee reconciling `git show … | shasum`
  against `ots info` would have found `268c061` unproven — falsifying the pre-registration.
  The three anchor sites now cite `63078f5`; the reproducibility line points to release tag
  `v0.3.12`. Documented as pre-data amendment A12 (annotation-only); `pre_registration.md`
  left unedited to preserve its timestamped hash.
- **Appendix-A wide-equation overflow fixed.** The expanded $\mathcal{T}_1$ rational ran
  ~27 in past the right margin (2003 pt overfull hbox); wrapped in `\resizebox` to text width.
- **arXiv metadata abstract re-synced to the PDF abstract** (the empirical-scale caveat
  sentence was missing from the listing-page abstract).
- arXiv build bumped to v0.3.12; tarball rebuilt and re-verified.

## [0.3.11] - 2026-07-03

### Changed — referee-panel response (submission freeze candidate)

A simulated 3-referee panel (math-finance academic, ICAIF reviewer, stochastic-analysis
skeptic) + meta-review rated the paper substantive and correct (all three re-derived the
core math), embarrassment-risk low once seven concrete, permanent, adversarially-checkable
items were fixed. All applied and adversarially re-verified:

- **Baxendale B=2/3 "refutation" retracted.** The Lyapunov-shift scan claimed to
  "empirically refute (p<<0.01)" Baxendale's B=2/3. That is his *large-shear* (twist b->inf)
  asymptotic; the scan runs at small |rho*xi| and at the trivial G=0 equilibrium where the
  shear term is identically zero — outside the hypotheses, not a refutation. Reframed in
  main.tex, both variants, and the README (which still said "refutes ELR ... 13 sigma").
- **Hallucination comment removed from references.bib** (it shipped "was a hallucination" in
  the arXiv source). The engellambrasmussen2024 -> baxendale2025 lineage is otherwise clean.
- **README synced to the corrected McKean-Vlasov theorem.** It stated the *superseded*
  pre-v0.3.6 result (ratio = sqrt(1+(omega* tau_G)^2) > 1, opposite economic sign); now the
  rational closed form with the regime-dependent sign (ratio<1 at G_y>0, >1 at G_y<0).
- **Hawkes-SV dimensional slip fixed.** The Jaisson-Rosenbaum near-critical limit is 1D
  (CIR), so its drift boundary has no Hopf stratum; the Hopf lives only in the
  higher-dimensional dealer-gamma model. Statement corrected; sharpens the "unoccupied cell".
- **Theorem right-sizing.** BT-empty (grid-scoped) and Hawkes-SV (a position, not an
  identity) downgraded Theorem -> Proposition. Now 4 theorems + 2 propositions.
- **excess-entropy "mean-field critical exponent beta=1"** relabelled the generic linear
  (exponent-1) approach (mean-field beta is 1/2; the linear rate is just analyticity).
- **Honesty caveats up front:** abstract now states the model does not Hopf at
  empirically-scaled dealer gamma; a timescale paragraph confronts the ~11 yr canonical cycle
  vs intraday reflexivity.
- SUBMISSION_READINESS certification banner (v0.3.11, 40 pp, 579 tests, 89.47% coverage,
  verify.sh green — the stale "84.04% dip" is resolved). arXiv build -> v0.3.11 / 40 pp.

Builds clean: main 40 pp, ICAIF 7 pp, workshop 7 pp, zero undefined references; arXiv tarball
verified via simulated build. Paper body and both variants numerically identical pre/post
except the intended reframes.

## [0.3.10] - 2026-07-03

### Changed — reviewer-response corrections + full readability/comprehension overhaul

The arc from Andrey Itkin's review (2026-06-02) through the pre-submission
polish. The paper is now the ICAIF/arXiv submission candidate.

- **Marketron correction + memory-variable rename (Itkin review).** The memory
  variable z renamed z → χ throughout paper, theory.md, and both variants
  (their z is a market signal, their memory is y with its own Brownian motion);
  the "structurally analogous / closest model-structure precedent" claims
  replaced with an honest contrast (their memory = latent stochastic factor via
  the potential V_M; ours = deterministic price filter). §7 now states χ's full
  correlation with spot as a design property and that κ★ lives in the drift.
- **Three prose passes** (abstract de-LLM rewrite; 10-finding audit round;
  full-paper sentence-level pass, 19 spans): plain register, em-dashes 66 → 38,
  no CAPS emphasis, no filler. Verified by an adversarial 3-lens fidelity check;
  4 self-introduced claim-strength drifts caught and reverted; a stray
  uncommitted h4_detector_power.pdf with divergent plotted data reverted to the
  repro-pinned version.
- **Fresh-eyes comprehension pass** (5 zero-context readers, 79 findings; all
  sections judged substantive): fixed two foundational gaps — the dealer-gamma
  functional is now G(S, χ, v) with trend-dependent open interest q(χ), and the
  equilibrium gains the missing variance stationarity condition + S₀ = S★
  normalisation. MV particle SDE corrected to relax toward g(S, χ, v) (sympy-
  verified load-bearing). Hawkes–SV theorem gains a Perron–Frobenius proof
  sketch. Hartman–Grobman misuse → centre manifold + normal form. Defined:
  κ_NS, κ_u, τ_G, σ_v := ∂_vσ², m̂, r/q_div; expanded IAAFT/TOST/RR25/CSD/DOW;
  "Phase 4" and the amendment ledger (A1–A11) introduced for outside readers;
  "canonical log-normal-OI specification" disambiguated from the dimensionless
  regime; κ-scale reconciliation note added. Foundational fixes ported to both
  variants.
- **Author email** → mahimn.patel@mail.utoronto.ca; date → July 2026 (39 pp).
- **arXiv build** bumped to v0.3.10 / 39 pp; tarball verified via simulated
  arXiv build (sha256 4ef6187c…).
- **New:** `docs/wrds_day_one_validation_plan.md` — the locked-spec WRDS/
  OptionMetrics day-one validation protocol (coverage gate through 2025-02-05
  for the Yen CSD record, field mapping, two-date validation, disclosure log).

## [0.3.9] - 2026-06-01

### Changed — empirical-leg redesign (pre-data amendments A8–A11) + Theorem 4 reposition

Corrects two material specification errors and one over-claim in the empirical leg
and the Hawkes–SV theorem, found through pre-data analysis + independent expert
review — all **before any SPX data is loaded**. Formalised as amendments A8–A11 in
`paper/pre_registration_amendments.md`; the original pre-registration is preserved
(git history + `pre_registration.md.v1.ots`) and the amended pre-reg re-stamped via
OpenTimestamps.

- **A8 — H4 redesigned (critical slowing down).** The original spectral H4 sought a
  peak at the Hopf frequency ω⋆, but the limit-cycle period is 5.3–11 yr — below the
  lowest resolvable bin of any window fitting the data, so the test was geometrically
  unfalsifiable. Replaced by a critical-slowing-down early-warning test (rolling lag-1
  autocorrelation of |r_t| vs AR(1)/phase-randomised surrogate; Scheffer 2009, Dakos
  2012), validated to 0.82–0.85 power on a 252-day record at FPR ≤ 0.083.
  New `theory/critical_slowing_down.py` (+14 tests).
- **A9 — primary H1 replaced by a direct dealer-gamma (GEX) regression.** The original
  H1 routed the reflexivity claim through a Mamba+PPO+EWC RL agent, confounding the
  mechanism with the estimator. H1′ removes agent and simulator: it regresses next-day
  realised vol-of-vol on signed dealer GEX from the OI grid, pooled over three event
  windows with a quiet-regime control. Synthetic pooled power 0.86 / sign 0.98 /
  FPR 0.02. RL tournament demoted to secondary/exploratory. New `empirical/` package
  (gex_regression, gex_simulator, gex_validation; +15 tests).
- **A10 — Theorem 4 (Hawkes–SV) repositioned; tautological n_SV "verification" removed.**
  The dimensionless n_SV := c0/(c1 c2) "verified to 1e-15" was a definitional tautology.
  Removed. Repositioned: Hardiman's n≈1 = the real-eigenvalue (saddle-node) stratum;
  the model's Hopf is the strictly-stronger oscillatory "unoccupied cell" beyond any
  scalar branching ratio. Falsifiable spectral discriminator
  `theory/hawkes_sv_bifurcation.py` (zero-overlap strata separation on synthetic data).
  κ unit-chain made explicit (`theory/kappa_rescaling.py`): empirical proximity to κ⋆
  is indeterminate, deferred to the GEX test, not asserted.
- **A11 — corrections.** Event-window dates reconciled to `event_windows.txt`
  (pandas_market_calendars); §4 strike grid corrected to 11 (7×11=77); quiet-regime
  control window 2017-05-01 → 2017-10-20 added; researcher-DOF locks; Faff/Brailsford
  reference DOI corrected (.101837, Faff 2023).

### Added
- 8 verified bib entries: Scheffer 2009, Dakos 2012, Lenton 2011 (early-warning);
  Jaisson–Rosenbaum 2016, Gatheral–Jaisson–Rosenbaum 2018, El Euch–Rosenbaum 2018/2019,
  Abi Jaber 2019 (rough volatility, cited as future-work precedent).

### Notes
- 567 tests pass; 37-page master PDF compiles clean (0 undefined refs). Variants
  (ICAIF/workshop) still carry v0.3.6 Hawkes-SV/H1/H4 content — a known follow-up re-sync.
- Intervening v0.3.3–v0.3.8 are recorded in git tags; this CHANGELOG had fallen behind
  at v0.3.2-in-flight and resumes here.

## [Unreleased] — v0.3.2 in-flight (2026-05-14)

### Added — substantive theoretical extensions
- **Theorem 2 (Hawkes-SV equivalence at the Hopf boundary)** — main.tex §3.7. Formal
  identification of the Hardiman 2013 critical branching ratio $n \approx 1$ with
  our continuous-time Hopf threshold $\kappa^\star$, via the Bacry-Delattre-Hoffmann-Muzy
  diffusive-limit identity and the kernel-universal stability boundary. Numerical
  anchor at canonical regime: $n_{\mathrm{SV}}(\kappa^\star) = 1$ exactly (definitional
  identity). Implementation: `src/reflexive_options/theory/hawkes_equivalence.py`,
  runner `experiments/hawkes_sv_equivalence.py`. New tests:
  `tests/test_hawkes_equivalence.py` (3 tests).
- **Theorem 3 (BT locus empty in canonical scan window)** — main.tex §3.6. Closed-form
  argument that $G_v < 0$ uniformly dominates $G_y \alpha \kappa_v / (\beta\gamma)$
  on the scanned $(\sigma_q, \gamma) \in [0.05, 0.40] \times [0.20, 5.00]$ window,
  forcing $\kappa_{\mathrm{SN}} < 0$ and excluding Bogdanov-Takens bifurcations there.
  Falsifiable economic prediction: no homoclinic burst-relax dynamics from this model
  at fixed parameters. Implementation: `experiments/codim2_analysis.py`,
  `theory/bifurcation.py` extensions. New tests: `tests/test_codim2_bifurcation.py`
  (8 tests).
- **Bautin curve** with 6 anchors at the canonical specification, characterising the
  supercritical → sub-critical transition in $(\sigma_q, \gamma)$ space (main.tex §3.6,
  Table tab:bautin-anchors).
- **Closed-form ℓ_1 (Eq. 19 + Appendix A)** — symbolic Kuznetsov-formula expansion in
  the 13-symbol parameter space, verified to $\sim 10^{-13}$ relative against the
  numerical pipeline at the canonical regime. New `notebooks/closed_form_ell1_derivation.py`
  step 8 + auto-generated `paper/figures/ell1_closed_form.tex` (10.6 KB).
- **Empirical $|\Lambda| \sim |\rho\xi|^B$ scaling fit** — main.tex §3.4. OLS on a
  $6 \times 6$ $(\xi, \rho)$ grid yields $\hat B = 0.082$ (95% CI $[-0.010, 0.168]$),
  empirically refuting the Engel-Lamb-Rasmussen prediction $B = 2/3$ at this regime
  ($p \ll 0.01$); structural reason is the trivial $G \equiv 0$ equilibrium where the
  shear-stretching term $\partial_a v$ vanishes. New experiment
  `experiments/lambda_scaling.py`; figure `figures/lambda_scaling_loglog.pdf`.
- **Limit-cycle numerical validation past $\kappa^\star$** — main.tex §3.4. Deterministic
  skeleton at $\kappa = 1.05\kappa^\star$ converges to a closed orbit; measured period
  $T = 10.561$ yr matches Hopf prediction $T_\kappa = 10.977$ yr to 3.79% (within the
  leading-order normal-form correction). New experiment
  `experiments/limit_cycle_supercritical.py`; figure `figures/limit_cycle_supercritical.pdf`.
- **H1 synthetic-pipeline end-to-end validation** — main.tex §5.4. SW2 ordering on
  simulator-vs-simulator data: SW2(κ_0) = 0.005 < SW2(2κ_0) = 0.034 < SW2(Heston) = 0.054
  with disjoint bootstrap CIs. The H1 protocol is now demonstrated working on synthetic
  ground truth before the empirical SPX target arrives. New experiment
  `experiments/h1_synthetic_validation.py`.
- **2D bimodality on (log S, v) joint density** — main.tex §7. H_bimod was refuted on
  the 1D log-S marginal at $\gamma = 0$; the 2D PCA-projected dip statistic at
  $\kappa = 1.05\kappa^\star_{\mathrm{env}}$ flips to *supported* ($p = 0.033$) on a
  $\sim 79\%$-survival sample. Result is preliminary and selection-conditioned. New
  experiment `experiments/h_bimod_2d_scan.py`.
- **A priori mechanism-relevant cell restriction** — main.tex §6.1. Defensible
  restriction to long-horizon shape moments + within-envelope cells; in-sample 7/10
  matches ($p = 0.172$), OOS 4/8 ($p = 0.637$). Does not clear $p < 0.05$ at the present
  budget; the restricted result is reported as a transparent secondary statistic and the
  original 8/24 is retained as headline.
- **H4 detector power on Stuart-Landau positive control** — main.tex §5. Achieves
  $\geq 80\%$ peak power at $T = 512$ for 8/9 $(\mu, \sigma)$ configurations under the
  locked IAAFT-surrogate $\alpha = 0.05$ rule; non-monotone in $T$ (degrades at
  $T \in \{1024, 2048\}$). New script `scripts/h4_power_realistic.py`; figure
  `figures/h4_detector_power_v2.pdf`.
- **Sliced-W2 sample-complexity table** — main.tex §5. $n_{\min} \approx 4{,}000$
  windows for $\pm 10\%$ bootstrap CI half-width; the implication for H1 is that each
  event contributes $\sim 280$ windows giving $\sim 23\%$ ratio, requiring inter-baseline
  SW2 gap $\geq 0.46 \cdot \mathrm{SW2}_{\mathrm{true}}$ for discriminability. New script
  `scripts/sw2_sample_complexity.py`; figure `figures/sw2_sample_complexity.pdf`.
- **GP-CI coverage audit** — `scripts/gp_ci_coverage_audit.py`. Confirms RBF nominal-95\%
  coverage at 91.5–100\% in the σ=0.10 single-seed regime, dropping to 67–94\% under
  σ=0.05 multi-seed conditions. Documented in main.tex §5 H2 paragraph as a known
  finite-sample limitation under the A6-locked RBF kernel.
- **Manuscript variants** — `paper/variants/neurips_workshop/main.tex` (4-page body for
  NeurIPS GenAI Finance Workshop) and `paper/variants/icaif/main.tex` (5-page body,
  double-blind ACM sigconf for ICAIF 2026). Both reuse master figures + bib via symlinks.

### Fixed — adversarial-grill findings (L2-G + L3-G)
- Stripped fictional "Matérn-3/2 kernel per pre-reg amendment A8" claim from main.tex
  and both variants — A1–A7 are locked at commit `63078f5` and A6 explicitly fixes
  the H2 kernel as RBF + WhiteKernel; the Matérn experiment showed vacuous CIs
  (non-shrinking finite-difference variance from a once-differentiable kernel).
- Corrected the $|\Lambda(\kappa^\star)|$ scan band from $[5\!\times\!10^{-2},
  9\!\times\!10^{-2}]$ to the actual $[4.8\!\times\!10^{-2}, 1.21\!\times\!10^{-1}]$
  (median $\approx 7.3\!\times\!10^{-2}$).
- Reframed H4 power claim from "≥80% at $T \geq 512$" to "≥80% peak power at $T = 512$
  specifically; non-monotone in $T$" (6/9 at $T = 1024$, 5/9 at $T = 2048$).
- Resolved the $\kappa^\star$ vs $\kappa^\star_{\mathrm{env}}$ notation collision in
  the 2D bimodality discussion (canonical Hopf threshold $\approx 0.896$ vs the
  $\sim 3.9 \times 10^{-9}$ empirical-magnitude stability envelope).
- Corrected H_bimod 2D survival fraction from "~1.6%" to actual ~79%
  ($n = 15{,}769$ surviving cells out of $20{,}000$).
- Reframed Theorem 2 numerical anchor: the $|n_{\mathrm{SV}}(\kappa^\star_4) - 1| =
  3.85 \times 10^{-5}$ residual is the truncation error in the published 4-decimal
  $\kappa^\star_4 = 0.8964$, not eigenvalue-solver noise; at machine-precision $\kappa^\star$
  the identity is exact ($< 10^{-15}$).
- Added Stuart-Landau citation `kuznetsov2004` in the H4 paragraph.
- Fixed `runs/h_bimod_2d_scan/` → `runs/h_bimod_2d/` directory reference in 2D KDE caption.
- Deleted redundant `paper/abstract.md` (main.tex `\begin{abstract}` is canonical).
- Coverage config: experiment-runner CLI scripts added to `[tool.coverage.run].omit`
  (smoke-tested through integration tests rather than unit-coverage instrumentation).
  Added `N815` to ruff ignore list (paper-faithful dataclass attribute names like
  `log_A`, `B_ci_low` follow the OLS-fit notation of paper §3.4).

### Tests added
- `test_n_sv_at_brent_root_is_machine_epsilon` (`tests/test_hawkes_equivalence.py`) —
  validates the §3.9 claim that the n_SV residual at machine-precision κ★ is < 1e-12
  (i.e., truncation in the 4-decimal published value, not solver noise). At
  κ★_brent = 0.8964305216085703 the measured residual is 1.998e-15 (machine-ε on a
  3×3 matrix); at the published κ★_4 = 0.8964 the residual is 3.851e-05, both
  consistent with the paper's framing.
- `experiments/hawkes_sv_equivalence.py` now also records `kappa_star_brent`,
  `n_sv_at_kappa_star_brent`, and `criticality_residual_brent` in `metrics.json`,
  so future audits can verify the §3.9 precision claim from the run artefacts
  directly. (The pre-existing `kappa_star_grid = 0.8981928` is the first
  sign-change index of the 1001-pt κ-grid, bounded by grid resolution ≈ 1.79e-3 —
  not a bug, but uninformative for the precision claim; the new Brent field
  resolves the audit gap flagged by L5-G P2-2.)

### Notes
- **Pre-registration lock preserved.** All amendments file (A1–A7) is closed at commit
  `63078f5`; the A8 claim that briefly appeared in main.tex was an unrealised proposal,
  not a registered amendment.
- **PDF**: 29 pages from 14 baseline. 9 figures. New Appendix A (closed-form $\ell_1$).

## [0.3.1] - 2026-04-22

### Fixed

**Bibliography hygiene (paper/references.bib)**
- 5 hallucinated bibliographic entries corrected after author-by-author
  WebFetch verification against arXiv abstracts and journal landing pages:
  - `bai2024rlfinance`: authors corrected to **Yahui Bai, Yuhe Gao, Runzhe
    Wan, Sheng Zhang, Rui Song** (was: "Bai, Junkun and Gao, Xinyu and Wan,
    Cheng and Zhang, Yuan and Song, Le").
  - `hou2025robust` → renamed to **`ma2025robust`** (Shaocong Ma, Heng
    Huang); was wrongly attributed to "Hou, Yifan and others".
  - `jin2025diffusion`: first names corrected to **Chen Jin, Ankush
    Agarwal** (was: "Yufeng Jin, Ankur Agarwal").
  - `chevallier2025ot`: first names + title corrected to **Marius
    Chevallier, Stefano De Marco, Pierre-Emmanuel Lévy-dit-Vehel**, "An
    Optimal Transport Approach to Arbitrage Correction: Application to
    Volatility Stress-Tests" (was: "Julien", "Antoine", and "Optimal
    Transport-Based Approach to Arbitrage Correction of Implied Volatility
    Surfaces").
  - `hosseini2024conditional` → renamed to **`hosseini2023conditional`**
    (Hosseini, Hsu, Taghvaei); arXiv ID corrected to **2311.05672** and
    year to **2023** (was incorrectly attributed to 2403.18705 / Hosseini,
    Bunne, Cuturi 2024 — that arXiv ID is a different paper by Chemseddine
    et al.).
  - `stadnytska2025regimes` → renamed to **`luanhamp2025regimes`** (Qinmeng
    Luan, James Hamp).
  - `brailsford2022prereg` → renamed to **`faff2022prereg`** (Robert Faff
    is the actual editorial author).
- 3 metadata-level corrections to existing entries:
  - `heli2009`: added missing 4th author **Junjie Wei** + correct DOI
    suffix (`07.016` not `07.011`) and issue number (6).
  - `faragohjalmarsson2023`: journal corrected to **Review of Finance**
    (vol 27, issue 2, pp 495–538, 2023), title to **"Long-Horizon Stock
    Returns Are Positively Skewed"** — the working paper title "Compound
    Returns" appearing in the bib was the SSRN/working title, not the
    published title.
  - `murray2022multi`: title expanded to the full published version, added
    middle initial `S.` to Pakkanen.
- All `\cite{}` keys updated in `paper/main.tex`,
  `paper/related_work.md`, `paper/threats_to_validity.md`, and
  `paper/MANUSCRIPT_SKELETON.md` to match the renamed keys.

**Λ stochastic-Hopf shift (paper/theory.md §4.2 + paper/main.tex §3.4)**
- The published value $\Lambda(\kappa^\star) \approx +1.85 \times 10^{-2}$
  was **not reproducible** from any committed runner. The current
  `experiments/lambda_correction_canonical` produces $|\Lambda| \sim 10^{-3}$
  with sign that depends on the OI configuration (canonical-§4.2 trivial
  $G \equiv 0$ equilibrium gives $\Lambda \approx -6.9 \times 10^{-3}$; the
  paper's stale +1.85e-2 number was from a different and undocumented OI
  setup). Paper text rewritten to honestly report the reproducible
  magnitude only and defer the sign characterisation to the empirical
  phase. The $(\rho\xi)^{2/3}$ Engel–Lamb–Rasmussen scaling discussion in
  theory.md §5 is correspondingly softened to "predicted but not validated
  in sign by the present finite-budget estimator."

**Figure 3 (paper/figures/h4_detector_power.pdf) — pulled**
- The figure is removed from the §5 paper text. Re-rendering at a
  moderate budget (n_seeds=30, n_perm=50, full T and SNR grids) confirmed
  what the v0.3.0 stale rendering had hinted: at the locked IAAFT
  surrogate null (amendment A5), the H4 detector's power on a pure
  sinusoid-in-noise positive control is essentially zero across the entire
  $(T, \text{SNR})$ grid. This is *not* a bug in the detector — IAAFT is
  designed to preserve the linear autocorrelation of the input, and a
  pure sinusoid has a degenerate ACF that the IAAFT surrogate reproduces
  almost exactly, so the in-band peak ratio is similar across surrogates
  and the p-value stays high. The realistic positive controls — limit-
  cycle Stuart-Landau, supercritical reflexive simulator at long path
  budgets — have a richer spectral signature that IAAFT can plausibly
  discriminate against, but those characterisations need the realised
  path budget that only becomes available in the empirical SPX phase.
  The §5 paragraph is reframed accordingly: H4 is "validated synthetically
  in the IAAFT-FPR sense (FPR $\leq 7\%$ on Heston / AR(1) $H_0$);
  detection-power characterisation deferred to empirical SPX evaluation."

**Engineering fixes (G3 audit)**
- `theory/sensitivity.py::kappa_sensitivity_curve`: GP-posterior 95% CI
  was under-covering at ~70% empirical vs the 95% nominal because the
  WhiteKernel noise MLE was collapsing to its 1e-10 lower bound on n=9
  grid points. Fix: pin the noise variance to the seed-mean MC variance
  (averaged across the κ grid) and pass it via
  `noise_level_bounds="fixed"`. Coverage on smooth-truth synthetic cases
  (quadratic / linear / sin) recovers to ≥ 80% at n_seeds = 100. New test
  `tests/test_sensitivity.py::test_gp_slope_ci_coverage_with_pinned_noise`
  (parametrised over the three truths) enforces the bound.
- `theory/spectral.py::adaptive_welch_nperseg`: when the
  `n_trajectory // 2` cap was not itself a power of 2, the function
  returned the raw cap (e.g. 50 on a `n_trajectory=100` input). Fixed by
  snapping *down* to the largest power of 2 ≤ the capped target. New
  parametrised test
  `tests/test_spectral_amendments.py::test_adaptive_welch_nperseg_always_returns_power_of_two`
  enforces this contract on a wider input grid.

**Theorem 1 / consistency**
- `paper/main.tex` Theorem 1 A2: re-added the dropped sentence "Multiple
  equilibria may exist globally; our analysis is local." to match the
  `paper/theory.md` version verbatim.
- `tests/test_bifurcation.py::test_compute_lambda_correction_runs_and_returns_finite`
  was tautologically asserting only `np.isfinite(Lambda)`. Replaced with
  `test_compute_lambda_correction_in_canonical_section_4_2_regime` which
  asserts the magnitude bound $-1 \leq \Lambda \leq +1$ at the §4.2
  canonical regime — a real (not vacuous) check on the runner's behaviour.

**Notation hygiene**
- `paper/main.tex` §3.5 (closed-form section) now carries a one-line
  footnote at first use of $G_y$ explaining that $G_y := \partial_a G$ is
  identical to the §3.2 $G_x$ up to the choice of deviation symbol.
  `paper/theory.md` §4.3.2 carries the same parenthetical clarification.
- `paper/notation.md` relaxed the "reflexive coupling must be
  $\boldsymbol{\kappa}$" rule: plain $\kappa$ is the reflexive coupling
  throughout main.tex / theory.md, $\kappa_v$ is Heston mean-reversion;
  the bold-symbol convention is preserved only in
  `paper/pre_registration.md` for chain-of-custody reasons.

**Metadata + housekeeping**
- `paper/arxiv_metadata.txt`: page count corrected to **14 pages, 3
  figures** (was 8); pre-registration commit hash placeholder
  `<FILL_IN_AT_SUBMISSION>` replaced with the actual v0.1.0 anchor commit
  `268c061`.
- `paper/pre_registration_amendments.md`: commit-hash placeholder for the
  amendments-set anchor replaced with the v0.3.0 release commit `63078f5`.
- `README.md`: stale "current release is `0.1.0`" line updated to
  `0.3.0`.

## [0.3.0] - 2026-04-22

### Added

**Manuscript**
- `paper/main.tex` — 9-page (body) LaTeX manuscript per
  `paper/MANUSCRIPT_SKELETON.md` 8-page workshop layout, with full proofs
  of Theorem 1 (Hopf bifurcation), the closed-form $\kappa^\star$ and
  $\ell_1$ for log-normal open-interest in moneyness, the
  $(\xi,\rho,\sigma_v)$ phase diagram, the pre-registered evaluation
  framework, and the mechanism decomposition vs. Marketron. Compiles
  cleanly under TeXLive 2024+ via `make pdf`.
- `paper/references.bib` — 62 BibTeX entries; 61 cited in the manuscript,
  one (`baxendale1994`) included for the Khasminskii-stochastic-Hopf
  citation chain.
- `paper/Makefile` — `make pdf` (pdflatex + bibtex + pdflatex × 2),
  `make clean`, `make watch` targets.
- `paper/arxiv_metadata.txt` — arXiv submission metadata: title, author,
  primary `q-fin.MF` + cross-list `q-fin.CP, q-fin.PR, q-fin.ST, math.DS`,
  MSC 2020 codes `91G80, 37G15, 60H10, 91G60, 93E20`, CC-BY-4.0 license.
- `paper/pre_registration_amendments.md` — pre-data amendments A1–A7
  (H4 Welch window adaptation, dual-signal H4 on $|r_t|$ and $\widehat{v}_t$,
  IAAFT surrogate null replacing iid permutation, GP-posterior slope CI
  for κ-sensitivity, and TOST equivalence on the dimensionless elasticity).
- `paper/threats_to_validity.md`, `paper/related_work.md`,
  `paper/mechanism_decomposition.md` (three-table mechanism analysis),
  `paper/notation.md`, `paper/abstract.md`, `paper/SUBMISSION_READINESS.md`,
  and `paper/MANUSCRIPT_SKELETON.md`.

**Mechanism decomposition + simulator hardening**
- Mechanism-decomposition reporter in `synthetic_replication.py`: every
  Marketron-vs-reflexive cell now carries a `mechanism_class` ∈
  {`shape_target`, `level_artifact`, `calibration_artifact`}, plus
  `sign_match`, `order_of_magnitude_match`, and the legacy `within_8pct`
  flag. The headline gate is the shape-match rate (≥30% sign-agreement
  on `shape_target` cells) — replaces the prior "0 cells hit" report.
- `marketron_tuning.py` — coarse 5D grid search over (κ, γ, T_eff, μ_q, σ_q)
  picking the best reflexive overrides per Marketron parameter set; writes
  `runs/marketron_tuning/<ts>/grid_results.parquet` and `best_overrides.json`,
  consumed by `synthetic_replication.load_tuned_overrides`.
- CLI exit code on `synthetic_replication.py`: 0 if shape-match rate ≥ 30%,
  1 otherwise — enforces the headline number on every CI run.

**Theory**
- Closed-form first Lyapunov coefficient $\ell_1$ for log-normal OI in
  moneyness (`theory/bifurcation.lyapunov_coefficient_lognormal_oi`),
  matching the FD-tensor pipeline to <0.6% relative; canonical regime
  $\kappa^\star = 17.81$, $\omega^\star = 1.18$, $\ell_1 = -0.48$
  (supercritical).
- Closed-form Hopf threshold $\kappa^\star$ as the smallest positive root
  of the §4.3.2 quadratic.
- $(\sigma_q, \gamma)$ closed-form phase boundary rendered to
  `paper/figures/ell1_phase_boundary.pdf`.
- 4D phase scan `experiments/hopf_phase_scan_4d.py` rendered to
  `paper/figures/hopf_phase_diagram.pdf`.
- H4 spectral-peak detector with adaptive Welch window, dual-signal
  ($|r_t|$ and realised-variance proxy $\widehat{v}_t$), IAAFT surrogate
  null per Schreiber & Schmitz (1996); rendered to
  `paper/figures/h4_detector_power.pdf`.
- GP-posterior slope CI for the κ-sensitivity protocol
  (`theory/sensitivity.py`), replacing the spline-derivative + iid
  bootstrap that under-covered out-of-span function classes.

### Changed

- `synthetic_replication.py` default `n_steps` raised from 252 → 756 to cover
  Marketron's 3-year horizon.
- Test suite expanded from 252 → **329 tests**; coverage 85.12% → **89.05%**
  (gate at 85% in CI).
- Pre-registration amendment set is now A1–A7 (was A1–A4 at v0.2.x).

## [0.1.0] - 2026-04-30

First tagged research release. Everything is data-free and reproducible from
synthetic priors and published parameter sets; empirical SPX calibration is
gated on Phase 4 of the master TODO.

### Added

**Simulator**
- Reflexive 3D SDE simulator over $(S_t, v_t, z_t)$ with a dealer-gamma
  feedback channel $G(S, t, v)$ following the Garleanu-Pedersen-Poteshman
  (2009) demand-pressure mapping; reduces to time-dependent Heston at
  $\kappa = 0$.
- Three non-reflexive baselines for clean comparison: time-dependent Heston
  (5-10 piecewise-constant regimes, QuantLib analytic IV), local-stochastic
  vol (LSV polynomial), 3/2 stochastic vol, plus a gamma-aware non-reflexive
  baseline that observes $G_t$ without feeding it back into dynamics.
- Hard arbitrage-free filter on every simulated surface: convexity-in-strike,
  monotonicity-in-maturity, calendar-spread positivity, Lee moment bounds.
- Surface generator + parquet I/O at roughly 85k surfaces/sec.

**Theory**
- Hopf bifurcation theorem (paper/theory.md, Theorem 1) with computed
  threshold $\kappa^* \approx 0.8964$, angular frequency
  $\omega^* \approx 0.5724$, first Lyapunov coefficient
  $\ell_1 \approx -0.025$ (supercritical), and stochastic shift
  $\Lambda \approx +0.0185$.
- Fokker-Planck stationary density for the reflexive simulator contrasted
  analytically and via Monte Carlo with Heston's known stationary
  distribution (H_tail confirmed, H_skew confirmed, H_bimod refuted and
  documented).

**RL infrastructure**
- Vendored ATLAS module from `mahimn01/trading-algo` (Mamba state-space +
  cross-attention transformer, BC + EWC + RAT reflexivity meter and topology
  detector), roughly 3,700 LOC with eight smoke tests.
- Gymnasium env, state/action/reward design, and curriculum schedule for
  reflexivity-aware policy training.
- $\kappa$-sensitivity transfer experiment: train a BC-trained MLP at
  $\kappa = \kappa_0$ and deploy across $\kappa \in [0, 2\kappa_0]$ to
  measure slope-of-degradation as a quantitative reflexivity-importance
  scalar.

**Evaluation**
- Sliced Wasserstein-2 distance over arbitrage-free IV surface
  distributions.
- Marketron replication infrastructure (mechanism mismatch documented:
  zero cells hit, honest finding preserved).

**Pre-registration**
- `paper/pre_registration.md` (2,624 words) committed and anchored to
  initial commit hash `268c061` before any empirical evaluation.

**Docs**
- `paper/notation.md` canonical symbol table.
- `docs/quality_research_brief.md` documenting tooling decisions.
- `CHANGELOG.md` (this file).

### Changed

**Quality stack**
- Pinned developer tooling under PEP 735 `[dependency-groups].dev` in
  `pyproject.toml`: ruff 0.15.12, mypy 1.20.2, pytest 8.4.2, pytest-cov
  7.0.0, pre-commit 4.3.0, ipykernel 6.30.1, jupyter 1.1.1.
- `uv.lock` committed for bit-identical environments via
  `uv sync --locked --all-extras --group dev`.
- Pre-commit (`.pre-commit-config.yaml`) wires ruff check, ruff format, and
  the standard hygiene hooks; mypy runs in CI only (rationale in
  `docs/quality_research_brief.md` §5).
- Multi-Python CI matrix: Ubuntu x [3.12, 3.13, 3.14] runs ruff check,
  ruff format --check, mypy, pytest with branch coverage and an 80%
  fail-under gate.
- Project-tuned strict mypy (`strict = true` with
  `disallow_any_unimported = false`, `warn_unused_ignores = false`,
  `disallow_subclassing_any` relaxed via overrides) per the conventions
  PyTorch and the wider scientific Python ecosystem use for ML/scientific
  codebases.
- `scripts/verify.sh` runs the full CI gauntlet locally in the same order,
  fail-fast.

### Fixed

- Lint surface upgraded to ruff 0.15.12 with rule families
  `E,F,W,I,N,UP,B,SIM,RUF,S` (S = flake8-bandit security checks; bandit
  itself dropped from the dep graph).
- All mypy errors under the project-tuned strict configuration; CI no
  longer carries `continue-on-error`.
- `astral-sh/setup-uv` pinned to `v8.1.0` (no major-version tag is
  published upstream).
- Pre-registration document anchored to the initial-commit SHA so the
  hypothesis set cannot be silently revised.

### Security

- ruff `S` rule group enabled (flake8-bandit successor at roughly 25x the
  speed); narrow ignores for `S101`, `S301`, `S311`, `S403` documented in
  `pyproject.toml` with rationale.
- Vendored third-party code under `src/reflexive_options/third_party/`
  excluded from coverage and mypy strict checks per the vendoring
  discipline in `CLAUDE.md`.

<!-- TODO: link the version once the v0.1.0 git tag is pushed. -->
[0.1.0]: https://github.com/mahimn01/reflexive-options/releases/tag/v0.1.0
