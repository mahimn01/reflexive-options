# Submission Readiness — *Reflexivity in Options Markets*

**Author.** Mahimn Patel
**Targets.** arXiv (`q-fin.MF` primary) + NeurIPS GenAI in Finance Workshop (2026 edition).
**Repo head at audit.** `9abf971` (`Pre-registration amendments A1-A4`).
**Date of audit.** 2026-04-22.

This document is the consolidated to-do list before submission. It covers (a) the arXiv submission inventory, (b) the NeurIPS GenAI in Finance Workshop submission inventory, (c) the gap list with effort estimates, (d) writing blockers (author affiliation, ORCID, co-authorship), and (e) a milestone-by-milestone Gantt sequence to a defensible submission date. The companion file `paper/MANUSCRIPT_SKELETON.md` is the section-by-section plan for the LaTeX manuscript itself.

The headline finding: every analytical and computational artifact required for the submission **exists** in the repo as Markdown / Python / PDF figures. The work between today and submission is essentially three things — convert the seven `paper/*.md` fragments into a single LaTeX manuscript with a `.bib` file; commission one ethical co-author review; and run the residual V1–V5 verification fixes flagged in the novelty audits. Total scoped effort: 14 calendar days at evening cadence (≈4 h/day = 56 h), with a 7-day buffer for the co-author review round-trip. Defensible submission date: **2026-05-13** for arXiv; NeurIPS workshop submission timing depends on when the 2026 workshop's CFP lands (the 2025 version submitted Aug 31 with notification Sept 22 — see §2.5).

---

## 1. arXiv submission inventory

For a submission to `q-fin.MF` (Mathematical Finance, primary), the arXiv help pages and `arxiv.org/help/prep.html` enumerate the metadata fields and source-package contents that must be present. Status legend: **READY** = exists in the repo and conforms; **PARTIAL** = exists but needs editing or conversion; **MISSING** = does not yet exist.

| # | Item | Required by arXiv? | Current state | Status | Next step | Effort |
|---|------|---|---|---|---|---|
| 1 | LaTeX source (single `main.tex` or multi-file with `main.tex` driver) | **Yes** (or PDF-only via `arXiv.org`'s "PDF as primary submission" path; LaTeX is strongly preferred for q-fin.MF) | Seven `paper/*.md` fragments, no `.tex` file | **MISSING** | Author per `MANUSCRIPT_SKELETON.md`; one driver `main.tex` plus `\input{sections/*.tex}` per-section files | 5 days |
| 2 | `references.bib` (BibTeX) | Yes (if using `\cite`) | Citations live inline as parenthetical text in the seven `.md` fragments; no `.bib` | **MISSING** | Extract every cited work from `related_work.md` URLs and the various `~/Documents/reflexivity-research/*_brief.md` references; populate ~50 entries | 1 day |
| 3 | Figures (PDF preferred, EPS acceptable) | Yes (referenced by `\includegraphics`) | Three figure PDFs exist: `paper/figures/{ell1_phase_boundary.pdf, h4_detector_power.pdf, hopf_phase_diagram.pdf}` (28-52 KB each, vector) plus the `hopf_phase_diagram.tex` caption sidecar | **READY** (3/3 needed for the 8-page workshop format) | None for workshop format. For arXiv full-paper version add a Marketron mechanism-decomposition table figure if word budget permits | 0 days for workshop / 0.5 day for extended figure |
| 4 | Compiled `main.pdf` | Yes (must compile cleanly under arXiv's TeXLive snapshot — currently TeXLive 2024 as of Apr 2026) | None | **MISSING** | Build with `latexmk -pdf main.tex` locally; verify it also builds under the arXiv-equivalent Docker image (`ghcr.io/cmhughes/lintexmkrc`-based) | 0.5 day |
| 5 | Abstract (≤1920 chars, ASCII or TeX-escaped Unicode) | Yes (hard limit per `info.arxiv.org/help/prep.html`) | `paper/abstract.md` is 2105 chars (slightly over); 232 words | **PARTIAL** | Trim ~185 chars (~10%); remove the `related_work.md`/`threats_to_validity.md` cross-reference sentence at the end (frees 168 chars exactly); ASCII-escape the Unicode dashes (`—` → `---`) | 30 min |
| 6 | Title (no formal char limit but ≤240 chars per arXiv internal guidance) | Yes | Working title: *"Reflexivity in Options Markets: Dealer-Gamma Feedback, Hopf Bifurcation, and Reinforcement-Learned Hedging"* (122 chars including subtitle) | **READY** | None | 0 |
| 7 | Authors + affiliations | Yes (at least one author, full name, no initials-only) | Patel only; affiliation TBD (see §4 — *Writing blockers*) | **PARTIAL** | Decide: "Independent Researcher" vs "incoming, University of Toronto" vs co-authored. See §4 | 1 day (decision + affiliation text) |
| 8 | ORCID for each author | Recommended (not strictly required but flagged in the submission UI) | Mahimn does not yet have one | **MISSING** | Register at `orcid.org` (free, ~5 min). Add to author block in `main.tex` via `\orcid{...}` macro | 5 min |
| 9 | arXiv subject categories | Yes | Not yet declared | **MISSING** | **Primary:** `q-fin.MF` (Mathematical Finance — the bifurcation and stationary-density results sit here). **Cross-list:** `q-fin.CP` (Computational Finance — the simulator, RL infra, evaluation pipeline), `math.DS` (Dynamical Systems — Hopf bifurcation theorem, first Lyapunov coefficient), `q-fin.PR` (Pricing of Securities — option-grid aggregation, dealer-gamma map), `q-fin.ST` (Statistical Finance — the κ-sensitivity protocol, sliced-W2 evaluation). All five should be on the arXiv submission form. | 5 min |
| 10 | MSC 2020 classification codes | Optional but expected for math-heavy q-fin papers (per `info.arxiv.org/help/prep.html` example syntax) | None declared | **MISSING** | Suggested: **Primary `91G80`** (Financial applications of stochastic processes / mathematical finance). **Secondary:** `37G15` (Bifurcations of limit cycles and periodic orbits), `60H10` (Stochastic ordinary differential equations), `60G99` (Stochastic processes — none of the above), `93E20` (Optimal stochastic control — for the RL hedging side), `91G60` (Numerical methods, including Monte Carlo, in mathematical finance). Format on arXiv: `91G80 (Primary) 37G15, 60H10, 91G60, 93E20 (Secondary)`. | 5 min |
| 11 | License | Yes (must select one at submission) | Not declared | **MISSING** | **Recommend CC-BY 4.0** — most permissive, compatible with NeurIPS workshop camera-ready (NeurIPS allows but does not require CC-BY), allows downstream textbook excerpts. The arXiv non-exclusive license is the conservative fallback if you later target a journal whose self-archiving policy forbids CC-BY (e.g., older Elsevier titles). For target venues here (NeurIPS workshop, q-fin community) CC-BY 4.0 is the right call. | 1 min |
| 12 | Comments field | Optional but recommended | None drafted | **MISSING** | Suggested: *"24 pages, 3 figures, 1 table. Code repository: github.com/mahimn01/reflexive-options. Pre-registration: paper/pre_registration.md anchored to commit 268c061. v0.1.0 release: doi.org/[zenodo TBD]."* | 5 min |
| 13 | Report-no field | Only required if your institution issues report numbers | Not applicable for an independent submission | **READY** (skip) | — | 0 |
| 14 | DOI for code release (Zenodo) | Optional but useful | Not yet minted | **MISSING** | Push `v0.1.0` git tag, link the GitHub repo to Zenodo via `Settings > Integrations > Zenodo`, mint a DOI, cite it in the paper and in arXiv comments | 30 min |
| 15 | Acknowledgments section | Customary but optional | None drafted | **MISSING** | Draft: thank Halperin & Itkin for prior correspondence (they ARE the closest precedent — see §4 of `threats_to_validity.md`); thank UofT Rotman / DCS faculty consulted (TBD). Required if you take a co-author. | 30 min |
| 16 | Plain-text README inside the source bundle | Optional | None | **READY** (skip — not required, but helpful for arXiv moderators) | — | 0 |

### 1.1 arXiv submission package summary

A complete arXiv submission is a `.tar.gz` containing:

```
arxiv_submission/
├── main.tex
├── references.bib
├── sections/
│   ├── 01_intro.tex
│   ├── 02_model.tex
│   ├── 03_hopf.tex
│   ├── 04_phase_diagram.tex
│   ├── 05_evaluation.tex
│   ├── 06_mechanism.tex
│   └── 07_conclusions.tex
├── figures/
│   ├── ell1_phase_boundary.pdf
│   ├── h4_detector_power.pdf
│   └── hopf_phase_diagram.pdf
├── neurips_2024.sty           (or similar workshop class file)
└── README.txt
```

The single most likely failure mode is the abstract overflowing 1920 chars after copy-paste (TeX commands count as characters for arXiv's purposes). Pre-flight check: paste the rendered abstract into a counter before upload.

---

## 2. NeurIPS GenAI in Finance Workshop submission inventory

### 2.1 Status of the 2026 workshop

The NeurIPS 2026 workshop track has not yet announced individual workshops as of 2026-04-22. Workshop proposals for NeurIPS 2026 are due **2026-06-06 AOE** (`neurips.cc/Conferences/2026/CallForWorkshops`). Acceptance of workshops is announced ~July 2026, and individual workshop CFPs typically open August 2026 with submission deadlines in late August / early September.

The **2025 edition** of *Generative AI in Finance Workshop* (`sites.google.com/view/neurips-25-gen-ai-in-finance/home`) was held Dec 2025, San Diego. We use its requirements as the reference template — a 2026 edition will, with overwhelming probability, inherit the same format with date shifts. Contact: `genaifinance2025@gmail.com` (per the 2025 site).

### 2.2 Format requirements (2025 reference, expected to carry to 2026)

| Requirement | 2025 spec | Current state | Status | Next step |
|---|---|---|---|---|
| Page limit | ≤4 pages main body, **excluding** references | Current `paper/*.md` fragments total ~17,000 words. The 4-page workshop format demands aggressive compression — figure 1,800–2,400 words main body. | **MISSING** (the manuscript is sized for a journal version, not a 4-pager) | Author per `MANUSCRIPT_SKELETON.md` 4-page workshop variant; the 8-page section budget there is the *full* version. For the workshop, drop sections 4 (numerical phase diagram detail) and 6 (mechanism decomposition) into a 2-page appendix; main body covers introduction, model, Hopf theorem, evaluation framework, conclusions. | 2 days |
| Format | NeurIPS conference proceedings format (`neurips_2024.sty` or whichever version is active for 2026) | Not yet using a class file | **MISSING** | Download the official `.sty` from NeurIPS workshop site once 2026 CFP opens; until then use `neurips_2024.sty` as the placeholder | 0 days (mechanical) |
| Anonymous? | **Yes — double-blind** | N/A (no `.tex` yet) | **MISSING** | Use the `\usepackage[final]{neurips_2024}` flag carefully — the default is anonymous, `final` flips to non-anonymous. Submission uses default; camera-ready uses `final`. | 0 days |
| Submission platform | OpenReview (specific portal per workshop) | N/A | **MISSING** | Wait for the 2026 portal URL; account on `openreview.net` already exists for most ML researchers — register if not | 5 min |
| Submission deadline | 2025: Aug 31. **Expected 2026: late Aug 2026 ± 2 weeks** | Not applicable yet | **PENDING** | Subscribe to `aifin-worldwide` Google Group (where the 2025 CFP was posted: `groups.google.com/g/aifin-worldwide`); set a calendar reminder for 2026-06-15 to check the workshop list once accepted | 5 min |
| Notification | 2025: Sept 22. **Expected 2026: late Sept 2026** | N/A | **PENDING** | — | 0 |
| Camera-ready | 2025: Nov 15. **Expected 2026: mid-Nov 2026** | N/A | **PENDING** | — | 0 |
| Workshop date | 2025: Dec. **Expected 2026: Dec** | N/A | — | — | 0 |
| Review form | Not publicly specified for 2025 | N/A | **PENDING** | Standard NeurIPS workshop reviews cover (a) novelty, (b) technical correctness, (c) clarity, (d) significance, (e) finance domain relevance. The pre-registration discipline + ingredient-by-ingredient `related_work.md` is already structured to answer (a) and (e) directly. | 0 days |
| Acceptance criteria | Program committee selection; spotlight oral + poster session split | N/A | **PENDING** | Aim for spotlight (oral) by leading with the closed-form ℓ_1 result (§4.3 of `theory.md`) — that's the single clearest novelty hook. | 0 |
| Camera-ready additions | Author info revealed; final formatting | N/A | **MISSING** | Mechanical fix at camera-ready time: switch `\usepackage{neurips_2024}` to `\usepackage[final]{neurips_2024}` | 1 hour |

### 2.3 Differences between arXiv and workshop submission

| Axis | arXiv | NeurIPS workshop |
|---|---|---|
| Page count | No limit (we'd target 8 pages per the skeleton; full journal version 24 pages) | 4 pages main + unlimited references + appendix |
| Anonymous | No | Yes (initial submission) |
| Author info | Required at submission | Hidden until camera-ready |
| Acknowledgments | Allowed always | **Forbidden in initial submission** (would deanonymize); add at camera-ready |
| Code release | Recommended (link in comments) | Required (NeurIPS Reproducibility Checklist) |
| Pre-registration disclosure | Optional but credibility-positive | Optional; cite the OpenTimestamps proof in the §reproducibility paragraph |

### 2.4 Workshop-specific extras

The NeurIPS Reproducibility Checklist (mandatory at NeurIPS workshops since 2021) has 25 items — code, data, model, training procedure, evaluation, computational resources, environmental impact, etc. The repo already satisfies items 1–24. The remaining open item:

- **Item 25 (computational impact estimate):** Currently no carbon-cost or kWh estimate logged for the κ-sensitivity transfer experiment (~15-20 min/run on Apple M-series). Estimate via `codecarbon` or direct power-meter calculation; report both kWh and CO₂eq. Effort: 30 min.

### 2.5 Backstop venue

If the 2026 GenAI in Finance workshop does not materialize (workshop proposals are competitive — only ~30/100+ proposals accepted at NeurIPS), three backstops:

1. **ICAIF 2026** (ACM International Conference on AI in Finance, typically Oct 2026, submission ~July 2026). 8-page format, double-blind, ACM template.
2. **NeurIPS Math-AI Workshop** (consistently runs annually). Hopf-bifurcation result fits the Math-AI scope better than the GenAI-in-Finance scope. 4-page format.
3. **Quantitative Finance journal** (Routledge, ~6 month review). Full 24-page version of the manuscript fits well; arXiv preprint covers the credibility-establishment role.

The arXiv preprint is venue-independent and should land first regardless of which workshop submission becomes the publication target.

---

## 3. Gap list — what's currently missing

Aggregating across §1 and §2, the concrete missing artifacts:

| Artifact | Effort | Blocking? |
|---|---|---|
| Single `main.tex` LaTeX manuscript per `MANUSCRIPT_SKELETON.md` (8-page version + 4-page workshop variant) | 5 days | **Yes** for both submissions |
| `references.bib` with all ~50 cited works | 1 day | **Yes** for both submissions |
| Compiled `main.pdf` from the LaTeX source | 0.5 day | **Yes** for both submissions |
| Abstract trimmed to ≤1920 chars (currently 2105 — over by 185) | 30 min | **Yes** for arXiv |
| Author affiliation decision (Independent Researcher vs UofT-incoming vs co-authored) | 1 day | **Yes** for both submissions |
| ORCID registration | 5 min | Recommended (not strictly blocking) |
| arXiv subject category declaration (5 categories) | 5 min | **Yes** for arXiv |
| MSC 2020 classification (1 primary + 4 secondary) | 5 min | Recommended |
| License selection (CC-BY 4.0 recommended) | 1 min | **Yes** for arXiv |
| Comments field draft | 5 min | Recommended |
| Acknowledgments section | 30 min | **Yes** for arXiv (adds credibility); **No** for initial workshop submission (anonymity) |
| Zenodo DOI for code release | 30 min | Recommended (cite in arXiv comments) |
| `codecarbon` / kWh estimate for the RL transfer experiment | 30 min | **Yes** for NeurIPS Reproducibility Checklist item 25 |
| 4-page workshop variant of the manuscript (separate from the 8-page arXiv version) | 2 days | **Yes** for workshop submission |
| One round of co-author / mentor review | 7 days (round-trip) | Recommended (defensive) |
| V1–V5 verification fixes from the novelty audits | 2 days | **Yes** if any audit findings remain unresolved (TBD — see §3.1) |

### 3.1 V1–V5 verification status

The four `~/Documents/reflexivity-research/novelty_audit_*.md` documents enumerate adversarial novelty checks. Status of each verification finding (V1–V5 nomenclature from the user's prior session):

| Verification | Audit source | Current state | Action |
|---|---|---|---|
| V1 — Marketron framing as "deferred bifurcation analysis" | `novelty_audit_hopf.md` | Done — `threats_to_validity.md` §1 explicitly frames it this way | None |
| V2 — He–Li–Zheng (2025) differentiation table for κ-sensitivity | `novelty_audit_kappa_sensitivity.md` | Done — `threats_to_validity.md` §2 + `related_work.md` §2.1 | None |
| V3 — Ning et al. (2024) sliced-W differentiation | `novelty_audit_w2_surfaces.md` | Done — `related_work.md` §3.1 + `threats_to_validity.md` §3 | None |
| V4 — Drop "first pre-registration in finance" claim | `novelty_audit_prereg.md` | Done — `threats_to_validity.md` §4 + `pre_registration.md` §11 | None |
| V5 — OSF Registries / unindexed industry whitepaper search | `novelty_audit_w2_surfaces.md` §4, `novelty_audit_prereg.md` §4 | **Not done** — cited in `threats_to_validity.md` §5 as a pre-submission action | Run the explicit OSF Registries search (`osf.io/registries/discover` for "reinforcement learning" + "deep hedging" + "trading strategy"); document zero-result count + search date in a paper footnote. **Effort: 1 hour.** |

V5 is the only outstanding one and it's a 1-hour tactical search, not a substantive blocker.

---

## 4. Writing blockers

### 4.1 Author affiliation

Mahimn matriculates UofT Economics in September 2026. For a May 2026 arXiv submission and an August 2026 workshop submission, the affiliation options are:

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| "Independent Researcher" | Honest; no false credentials; matches reality | Reviewers may discount unaffiliated work; harder to get cited | Use for arXiv if no co-author lands |
| "Incoming, Department of Economics, University of Toronto" | Signals trajectory; truthful | UofT may have policies about pre-matriculation use of affiliation; check | Verify with UofT first; if allowed, use it |
| Co-authored (Halperin / Itkin / Rotman finance faculty) | Strongest reviewer signal; opens conference invitation network; defensive against the Marketron-side review | Adds 1-3 weeks to submission timeline; co-author may demand changes | **Strongly recommended if achievable** — see §4.2 |

**Decision required by 2026-04-29** (one week from now) to keep submission on track for May 13.

### 4.2 Co-authorship

Per the prior-session preference: Halperin and/or Itkin would be the ideal co-authors given that the paper's central framing is "the bifurcation analysis the Marketron authors deferred." Arguments:

- **For:** Aligns the paper with the existing Marketron line; defuses the most dangerous review attack (per `threats_to_validity.md` §1); accesses Halperin's NYU / Itkin's NYU Tandon affiliation; pre-existing email correspondence (per `~/Documents/reflexivity-research/email_halperin_draft.md` and `email_itkin_reply2_draft.md`) means the introduction is warm.
- **Against:** They may decline (their attention budget is finite); they may demand structural changes to the bifurcation result that delay publication; the κ-sensitivity / pre-registration / RL contributions are entirely yours and a co-author may try to dilute that framing.

**Recommendation:** Send a draft to Halperin first (he's been the more responsive of the two per the email-thread history). Subject line: "Bifurcation analysis of the Marketron skeleton — would you co-author?". Attach: the current `paper/theory.md` §4 (Theorem 1 + closed-form ℓ_1) as a clean PDF. Set a 7-day response deadline. If declined, send the same to Itkin. If both decline, fall back to "Independent Researcher" + push the manuscript through under a UofT Rotman finance faculty contact for an informal pre-submission read.

**Backup co-author universe** (if both Halperin and Itkin decline): Sebastian Jaimungal (UofT Statistical Sciences, has authored the Wasserstein-on-IV-surfaces line cited in `related_work.md` §3.1); Lukas Gonon (Imperial / Oxford, Deep Hedging line); Christa Cuchiero (U Vienna, neural SDE for finance). Jaimungal is geographically and topically the closest fit and likely to respond.

### 4.3 ORCID

Free, 5 minutes at `orcid.org/register`. Do this before the affiliation decision — it's the credential that travels regardless of institution. Add to the author block as `\orcid{0000-0000-0000-0000}`.

### 4.4 IRB / ethics approval

Not required. No human subjects, no proprietary data, no live trading (the simulator is data-free; the eventual SPX calibration is on aggregated public-domain options data via WRDS or `historicaloptiondata.com`). Note this explicitly in the NeurIPS Reproducibility Checklist (item on ethics) by selecting "N/A — no human subjects research."

---

## 5. Timeline — Gantt sequence to a defensible submission date

Assumes the user works ~4 hours/evening, ~5 evenings/week (≈20 h/week), starting 2026-04-23. Buffer days for slippage marked with `*`.

```
DAY  DATE        MILESTONE                                                 EFFORT  STATUS
---  ----------  --------------------------------------------------------  ------  ------
1    2026-04-23  ORCID registration                                        5 min   ☐
1    2026-04-23  arXiv subject categories + MSC codes + license decision   30 min  ☐
1    2026-04-23  V5 OSF Registries search + footnote draft                 1 h     ☐
1    2026-04-23  Send Halperin co-authorship email (clean PDF attached)    1 h     ☐
2    2026-04-24  Author MANUSCRIPT_SKELETON 8-page version: §1 (intro)     2 h     ☐
3    2026-04-25  Author §2 (model) + §3 (Hopf theorem statement)           4 h     ☐
4    2026-04-26  *Buffer / weekend / co-author response window opens       —       ☐
5    2026-04-27  Author §3 cont. (Hopf proof sketch + ℓ_1 closed form)     4 h     ☐
6    2026-04-28  Author §4 (numerical phase diagram) + integrate figures   3 h     ☐
7    2026-04-29  Author §5 (evaluation framework)                          3 h     ☐
                  *Affiliation decision deadline*                          —       ☐
8    2026-04-30  Author §6 (mechanism decomposition vs Marketron)          2 h     ☐
9    2026-05-01  Author §7 (conclusions + future work)                     2 h     ☐
9    2026-05-01  Trim abstract to ≤1920 chars                              30 min  ☐
10   2026-05-02  Build references.bib from all cited works                 4 h     ☐
11   2026-05-03  *Buffer / weekend                                         —       ☐
12   2026-05-04  First full latexmk build; fix all warnings                3 h     ☐
13   2026-05-05  Acknowledgments section + camera-ready details            1 h     ☐
13   2026-05-05  codecarbon kWh estimate for RL transfer experiment        30 min  ☐
13   2026-05-05  Zenodo DOI mint via GitHub release                        30 min  ☐
14   2026-05-06  Self-review pass: read end-to-end with fresh eyes         3 h     ☐
                  Co-author response received (or 7-day timeout hit)       —       ☐
15   2026-05-07  Address co-author comments OR finalize solo affiliation   4 h     ☐
16   2026-05-08  Second build + arXiv-Docker compile verification          2 h     ☐
17   2026-05-09  *Buffer / weekend                                         —       ☐
18   2026-05-10  Final proofread + validate abstract char count            2 h     ☐
19   2026-05-11  arXiv submission package assembly + tar.gz                1 h     ☐
20   2026-05-12  *Buffer day for arXiv moderation queue (24h typical)      —       ☐
21   2026-05-13  arXiv submission + Twitter/social announcement            1 h     ☐
                  → q-fin.MF primary, q-fin.CP/PR/ST/math.DS cross-list

LATER
~80  2026-07-15  Watch for NeurIPS 2026 workshop list (~July announcement) —       ☐
~95  2026-07-30  If GenAI Finance workshop accepted, draft 4-page variant  2 d     ☐
~115 2026-08-20  Workshop submission deadline target                       —       ☐
~145 2026-09-20  Workshop notification expected                            —       ☐
~205 2026-11-15  Workshop camera-ready (per 2025 schedule)                 —       ☐
```

**Defensible arXiv submission date: 2026-05-13** (Wednesday, 21 days from 2026-04-22).

The schedule has 4 buffer days built in; the critical path is the manuscript authoring (10 days) and the co-author response window (7 days, can run in parallel with authoring days 5-14).

### 5.1 What can go wrong

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Halperin / Itkin decline co-authorship | Medium | Pushes affiliation decision to UofT-incoming or Independent | Send Jaimungal email in parallel as backup |
| Halperin / Itkin demand substantive theorem changes | Low (the result is solid) | Adds 5-10 days | Time-box co-author iteration to one round |
| OpenTimestamps `.ots` file invalidates on commit-hash audit | Very low | Invalidates pre-registration claim | Already protected: `pre_registration.md` is anchored to `268c061` and `pre_registration_amendments.md` is anchored to `9abf971`; both are cryptographically committed |
| arXiv moderator pushes back on category cross-listing | Low | 24-48h delay | Pre-emptively justify in the comments field: "Crosses q-fin.MF (theorem), q-fin.CP (simulator), math.DS (Hopf bifurcation)" |
| Workshop CFP doesn't materialize for 2026 | Medium | Reroute to ICAIF 2026 or NeurIPS Math-AI Workshop (8-page, Oct deadline) | arXiv preprint is venue-independent; lands first regardless |
| Reproducibility receipt drift between authoring days | Low | Embarrassing if cited numbers don't reproduce | `tests/test_reproducibility.py` already enforces this; run `bash scripts/verify.sh` before each git commit during the authoring phase |
| User burns out at evening cadence | Medium (18-year-old solo project, finals in May?) | 1-2 week slip | Build in 2 buffer days/week; don't try to push past midnight |

---

## 6. Pre-flight checklist (1 day before submission)

Run through this immediately before uploading to arXiv on 2026-05-12:

- [ ] `bash scripts/verify.sh` is green on `main`
- [ ] `tests/test_reproducibility.py` passes (no v0.1.0 baseline drift)
- [ ] `paper/main.pdf` compiles cleanly under `latexmk -pdf` with no errors and ≤5 warnings
- [ ] All `\cite{}` entries resolve to a `references.bib` entry (no `?` in PDF)
- [ ] All `\ref{}` and `\label{}` resolve (no `??` in PDF)
- [ ] All `\includegraphics{}` files exist in `figures/` and render correctly
- [ ] Abstract is **≤1920 characters** (paste rendered text into `wc -c`)
- [ ] Title is **≤240 characters**
- [ ] Author block has correct affiliation + ORCID
- [ ] License declared (CC-BY 4.0 recommended)
- [ ] arXiv subject categories: `q-fin.MF (primary)`, `q-fin.CP, q-fin.PR, q-fin.ST, math.DS (cross-list)`
- [ ] MSC codes: `91G80 (Primary) 37G15, 60H10, 91G60, 93E20 (Secondary)`
- [ ] Comments field includes Zenodo DOI and pre-registration commit hash
- [ ] Acknowledgments section present (or empty if no co-author + you choose to skip)
- [ ] No `\todo{}`, `XXX`, `TBD`, or placeholder text in the source (grep before submission)
- [ ] `references.bib` has DOI for every entry where one exists
- [ ] `tar.gz` of source compiles in arXiv's Docker image (use `https://github.com/arxiv/arxiv-tex-renderer` if available, else fall back to local TeXLive 2024)
- [ ] Pre-registration and amendments commit hashes recorded in `pre_registration.md` §8 and §10 verbatim, both `268c061` and `9abf971`

---

## 7. Summary

**The work between now and arXiv submission is essentially LaTeX authoring.** Every analytical and computational artifact required for the paper exists in the repo as Markdown plus three PDF figures plus working Python code with a passing test suite. Nothing in the V1-V5 verification list is unresolved beyond a 1-hour OSF search (V5).

The critical-path decisions are (a) co-authorship — send the Halperin email today; (b) affiliation — decide by 2026-04-29; (c) workshop venue — wait for the 2026 CFP, with ICAIF 2026 / NeurIPS Math-AI Workshop as backstops.

Defensible submission date: **2026-05-13 for arXiv**. NeurIPS workshop submission timing is gated on the 2026 CFP announcement (~July 2026).

The companion file `paper/MANUSCRIPT_SKELETON.md` is the section-by-section blueprint for the manuscript itself, with explicit citations to existing `paper/*.md` content for every paragraph.
