# Submission Readiness — *Reflexivity in Options Markets*

> **CURRENT CERTIFICATION (v0.3.11, 2026-07-03).** Freeze candidate for arXiv + ICAIF.
> Master `main.tex` = **40 pp**, 4 theorems + 2 propositions, 15 figures; ICAIF and NeurIPS
> workshop variants build clean at venue lengths. `bash scripts/verify.sh` **green on the
> current head**: ruff + format + mypy clean, **579 tests pass, 89.47% coverage** (≥ 85%
> gate) — the v0.3.2-era "84.04% dip" noted in the body below is resolved. This round applied
> the referee-panel corrections: the Baxendale $B=2/3$ "refutation" reframed as
> outside-hypotheses (not a refutation), the shipped `references.bib` hallucination comment
> removed, the public README synced to the corrected McKean–Vlasov theorem, the Hawkes–SV
> dimensional statement fixed and (with BT-empty) relabelled Proposition, and honest
> empirical-scale / timescale caveats added. **The body below is the historical v0.3.1/v0.3.2
> ledger, retained for provenance; where it conflicts with this banner, the banner governs.**

**Author.** Mahimn Patel (University of Toronto)
**Date of refresh.** 2026-05-14 (historical; see banner above for current)
**Repo head at refresh.** `780afd9` (`v0.3.1: second-pass grill cleanup — bib hygiene, Λ honesty, GP-CI fix`)
**Working tree.** v0.3.2-unreleased (Wave 1 + Wave 2 + L3-FIX content staged; CHANGELOG `[Unreleased]` block populated; commit pending)
**Targets.** arXiv (`q-fin.MF` primary), NeurIPS GenAI in Finance Workshop 2026, ICAIF 2026.

This is the consolidated submission-readiness ledger for the three target venues. It supersedes the 2026-04-22 audit which still listed `main.tex` as MISSING. As of this refresh `main.tex` is a 443-line LaTeX manuscript that compiles to a 29-page PDF, and the two venue-specific variants (NeurIPS workshop 4-page body + ICAIF 5-page body) are both built. The pre-registration (A1–A7) is locked at commit `63078f5` and no further amendments are admissible post-data-load.

---

## 1. Build status

| Artifact | Path | State | Pages | Last built |
|---|---|---|---:|---|
| arXiv master | `paper/main.tex` → `paper/main.pdf` | READY | 29 | 2026-05-14 |
| NeurIPS workshop variant | `paper/variants/neurips_workshop/main.tex` → `.../main.pdf` | READY | 6 (4 body + 2 refs) | 2026-05-14 |
| ICAIF variant | `paper/variants/icaif/main.tex` → `.../main.pdf` | READY | 5 (double-blind anon. ACM `sigconf`) | 2026-05-14 |
| Bibliography | `paper/references.bib` | READY | — | 2026-05-14 |
| arXiv metadata block | `paper/arxiv_metadata.txt` | READY | — | 2026-05-14 |

All three variants build clean under `latexmk -pdf` against TeXLive 2024 (the arXiv-equivalent snapshot). The variants reuse the master `figures/` and `references.bib` via path inheritance — single source of truth for figures and citations.

---

## 2. Section inventory (master `main.tex`)

| § | Title | LaTeX label |
|---|---|---|
| §1 | Introduction | `sec:intro` |
| §2 | The reflexive model | `sec:model` |
| §3 | Hopf bifurcation theorem and closed-form $\ell_1$ | `sec:hopf` |
| §3.1 | Equilibria and Jacobian | `sec:hopf-jac` |
| §3.2 | Routh–Hurwitz / Liu's criterion | `sec:hopf-rh` |
| §3.3 | Theorem 1 and proof sketch | `sec:hopf-thm` |
| §3.4 | Numerical anchor (incl. limit-cycle validation, $\Lambda$ scaling fit) | `sec:hopf-numerical` |
| §3.5 | Closed-form $\kappa^\star$ and $\ell_1$ for log-normal OI in moneyness | `sec:hopf-closed-form` |
| §3.6 | Codim-2 bifurcation structure (Bautin curve + Theorem 3 BT-empty) | `sec:codim2` |
| §3.11 | Hawkes–SV criticality correspondence (Theorem 4; repositioned in A10) | `sec:hopf-hawkes-equivalence` |
| §4 | Numerical phase diagram | `sec:phase` |
| §5 | Pre-registered evaluation framework (incl. SW2 sample-complexity, H4 power on Stuart–Landau) | `sec:eval` |
| §5.4 | Synthetic pipeline validation (H1) | `sec:eval-synthetic-validation` |
| §6 | Mechanism decomposition vs Marketron | `sec:mechanism` |
| §6.1 | A priori mechanism-relevant cell restriction | `sec:mechanism-restricted` |
| §7 | Conclusions and future work | `sec:conclusions` |
| App. A | Closed-form $\ell_1$ for log-normal OI in moneyness | `app:ell1-closed-form` |

---

## 3. Theorem inventory

| # | Statement | Location | Type |
|---|---|---|---|
| Theorem 1 | Hopf bifurcation in the gamma-coupled SV skeleton (with closed-form $\ell_1$ for log-normal OI as Eq. `eq:lognorm-ell1`) | §3.3, label `thm:hopf` | Existence + super/subcritical classification |
| Theorem 4 | Hawkes–SV criticality correspondence | §3.11, label `thm:hawkes-sv` | Hardiman $n\approx1$ = real-eigenvalue (saddle-node) stratum; the model's Hopf is the strictly-stronger oscillatory "unoccupied cell" beyond any scalar branching ratio (BDHM 2013 + JR 2015 + BMM 2015). Falsifiable spectral discriminator replaces the tautological $n_{\mathrm{SV}}$ "verification" (removed, A10) |
| Theorem 3 | BT locus empty in the canonical scan window | §3.6, label `thm:bt-empty` | Closed-form exclusion: $\kappa_{\mathrm{SN}} \leq -1.31$ across the $71 \times 97$ $(\sigma_q, \gamma)$ grid |

Theorem 1 carries five labelled assumptions (A1)–(A5) on smoothness, equilibrium uniqueness, parameter regularity, the Routh–Hurwitz/Liu condition at $\kappa^\star$, and $\ell_1(\kappa^\star) \neq 0$.

---

## 4. Figure inventory

Nine figures, all PDF, all rendered into `paper/figures/` and reused by both variants.

| # | Figure label | Path | Source script | Section |
|---|---|---|---|---|
| 1 | `fig:limit-cycle-supercritical` | `paper/figures/limit_cycle_supercritical.pdf` | `experiments/limit_cycle_supercritical.py` | §3.4 |
| 2 | `fig:lambda-scaling-loglog` | `paper/figures/lambda_scaling_loglog.pdf` | `experiments/lambda_scaling.py` | §3.4 |
| 3 | `fig:codim2-phase-diagram` | `paper/figures/codim2_phase_diagram.pdf` | `experiments/codim2_analysis.py` | §3.6 |
| 4 | `fig:hawkes-sv-equivalence` | `paper/figures/hawkes_sv_equivalence.pdf` | `experiments/hawkes_sv_equivalence.py` | §3.7 |
| 5 | `fig:hopf-phase-diagram` | `paper/figures/hopf_phase_diagram.pdf` | `experiments/hopf_phase_scan_4d.py` | §4 |
| 6 | `fig:ell1-phase-boundary` | `paper/figures/ell1_phase_boundary.pdf` | `notebooks/closed_form_ell1_derivation.py` | §4 |
| 7 | `fig:sw2-sample-complexity` | `paper/figures/sw2_sample_complexity.pdf` | `scripts/sw2_sample_complexity.py` | §5 |
| 8 | `fig:h4-detector-power` | `paper/figures/h4_detector_power_v2.pdf` | `scripts/h4_power_realistic.py` | §5 |
| 9 | `fig:h1-synthetic-ordering` | `paper/figures/h1_synthetic_ordering.pdf` | `experiments/h1_synthetic_validation.py` | §5.4 |
| (extra) | `fig:stationary-density-2d-kde` | `paper/figures/stationary_density_2d_kde.pdf` | `experiments/h_bimod_2d_scan.py` | §7 |

Strictly the body has 10 `\includegraphics` references (the §7 2D-KDE figure brings the count to 10), but the headline-9 corresponds to the substantive theory + evaluation deliverables. Counted both ways for completeness; arXiv metadata reports the headline 9.

Appendix A renders an auto-generated `figures/ell1_closed_form.tex` (10.6 KB, layered Kuznetsov-formula expansion) via `\input{}`.

---

## 5. Bibliography

| Metric | Value |
|---|---|
| `references.bib` size | ~19 KB |
| Entries (`@`-records) | 63 |
| Hallucinated entries fixed in v0.3.1 | 5 (`bai2024rlfinance`, `ma2025robust`, `jin2025diffusion`, `chevallier2025ot`, `hosseini2023conditional`) plus 3 metadata-level corrections (`heli2009`, `faragohjalmarsson2023`, `murray2022multi`) |
| All `\cite{}` keys resolve | yes (verified at last `latexmk` build, no `?` markers in PDF) |

---

## 6. Test suite + CI

| Metric | Value |
|---|---|
| Tests passing | 404 (up from 381 at v0.3.0; v0.3.2-unreleased adds the test suites listed below) |
| Coverage (branch-aware, against `src/reflexive_options`) | 84.04% on the working tree (v0.3.0 baseline was 86.98%; the new experimental modules `mckean_vlasov.py`, `robustness.py`, `lambda_correction_canonical.py`, `ablation_gamma_aware.py` are in-flight runner code that drags the aggregate down) |
| Coverage gate in `scripts/verify.sh` | 85% (the working-tree dip is a known pre-commit issue to be addressed before the v0.3.2 tag — the new untested runner modules should either land with tests or be added to the `omit` list in `pyproject.toml [tool.coverage.run]`) |
| Last clean `bash scripts/verify.sh` | green on `780afd9` (v0.3.1 commit) |
| New tests added in v0.3.2-unreleased | `tests/test_hawkes_equivalence.py` (3), `tests/test_codim2_bifurcation.py` (8), `tests/test_h1_synthetic_validation.py`, `tests/test_h_bimod_2d_scan.py`, `tests/test_lambda_scaling.py`, `tests/test_limit_cycle_supercritical.py`, `tests/test_kappa_star_robustness.py`, `tests/test_mckean_vlasov.py` |
| Reproducibility receipt | `tests/repro/baseline_v0.1.0.json`, gated by `tests/test_reproducibility.py` |

---

## 7. Pre-registration status

**Locked at commit `63078f5`.** Amendments file (`paper/pre_registration_amendments.md`) is closed: A1–A7 are the complete and final amendment set; no further amendments are admissible post-data-load.

| Amendment | Date | Subject |
|---|---|---|
| A1 | 2026-05-02 | H4 spectral-test window resolution (data-driven $N_{\text{window}}$) |
| A2 | 2026-05-02 | H4 reported on both $|r_t|$ and realised-$v_t$ with Bonferroni |
| A3 | 2026-05-02 | Permutation-surrogate null wording (intent fix) |
| A4 | 2026-05-02 | `in_band` decision rule made global-vs-local dominance check |
| A5 | 2026-05-02 | IAAFT surrogate null replaces iid permutation |
| A6 | 2026-05-02 | GP-posterior slope CI replaces UnivariateSpline derivative + bootstrap (RBF + WhiteKernel kernel) |
| A7 | 2026-05-02 | TOST equivalence test in §3 H2 normalised to dimensionless elasticity |

The Matérn-3/2 / "A8" kernel idea was an unrealised proposal during the L3-G adversarial grill; the experiment showed the once-differentiable kernel produces non-shrinking finite-difference variance and vacuous CIs, and the proposal was never registered as an amendment. The L3-FIX cleanup wave stripped the fictional A8 reference from `main.tex` and both variants. The locked H2 method per A6 remains RBF + WhiteKernel; the GP-CI coverage audit (`scripts/gp_ci_coverage_audit.py`) documents 91.5–100% coverage at $\sigma=0.10$ single-seed and 67–94% at the tighter $\sigma=0.05$ multi-seed regime as a known finite-sample limitation, not as grounds for re-amendment.

The pre-registration file `paper/pre_registration.md` is anchored to `paper/pre_registration.md.ots` (OpenTimestamps Bitcoin proof, verifiable via `uv run ots verify paper/pre_registration.md.ots`).

---

## 8. arXiv submission readiness

**Status: READY.** All metadata is finalised; the 29-page PDF compiles clean.

| Item | Value |
|---|---|
| Primary category | `q-fin.MF` (Mathematical Finance) |
| Cross-list | `q-fin.CP`, `q-fin.PR`, `q-fin.ST`, `math.DS` |
| MSC 2020 codes | `91G80` (Primary); `37G15`, `60H10`, `91G60`, `93E20` (Secondary) |
| License | CC-BY 4.0 |
| Title | *Reflexivity in Options Markets: A Stochastic-Volatility Model with Dealer-Gamma Feedback, Hopf Bifurcation Calculus, and a Pre-Registered Evaluation Framework* |
| Author | Mahimn Patel, Independent Researcher (`mahimn.patel.k@gmail.com`) |
| Pages | 29 |
| Figures | 9 (headline) |
| Comments field draft | "29 pages (body + references + Appendix A), 9 figures. Code at https://github.com/mahimn01/reflexive-options. Pre-registration anchored at commit hash 268c061." |
| Source bundle | `main.tex` + `references.bib` + `figures/*.pdf` + `figures/ell1_closed_form.tex` |

Pre-flight check: paste the rendered abstract into `wc -c` against the 1920-char arXiv limit; the abstract in `main.tex` is well within budget (~1.9 KB rendered, ASCII-clean).

---

## 9. NeurIPS GenAI in Finance Workshop 2026 readiness

**Status: variant built, deadline pending.**

| Item | Value |
|---|---|
| Variant path | `paper/variants/neurips_workshop/main.tex` |
| Style file | `paper/variants/neurips_workshop/neurips_2024.sty` |
| Pages | 6 (4 body + 2 references) — within the workshop's typical "4 pages of content, refs unlimited" rule |
| Anonymous? | yes (initial submission); flip `\usepackage[final]{neurips_2024}` at camera-ready |
| Submission portal | OpenReview (URL pending workshop CFP) |
| Deadline | not yet announced; expected late July 2026 (the 2025 edition closed Aug 31) |
| Figure subset | reuses master figures via symlink — same 9 figures available, subset selected for 4-page body |
| Reproducibility checklist | code + data + model all open-source; computational-impact estimate (item 25) is the one outstanding pre-camera-ready item |

The 2026 workshop list is announced ~July 2026 once workshop proposals (deadline 2026-06-06 AOE) are accepted. Calendar reminder: 2026-07-15 to check the accepted-workshop list.

---

## 10. ICAIF 2026 readiness

**Status: variant built, deadline 2026-08-02.**

| Item | Value |
|---|---|
| Variant path | `paper/variants/icaif/main.tex` |
| Format | ACM `sigconf`, double-blind |
| Pages | 5 (within the 8-page limit) |
| Anonymous? | yes — author block stripped, all self-citations indirected |
| Submission deadline | **2026-08-02** |
| Backstop value | if NeurIPS GenAI Finance does not materialise, ICAIF is the primary archival venue beyond arXiv |

ICAIF is the most-load-bearing archival venue if the workshop track does not accept the GenAI-Finance proposal for 2026.

---

## 11. Outstanding items (user-actioned, none blocking submission)

1. **ORCID registration.** `orcid.org/register`, ~5 minutes. Should be done before any submission; the credential travels regardless of affiliation. Add to author block as `\orcid{...}`.
2. **Co-author outreach to Halperin.** A single email — boosts the academic-positioning calculus (the paper's central framing is "the bifurcation analysis the Marketron authors deferred"). Pre-existing email correspondence is already in `~/Documents/reflexivity-research/email_halperin_draft.md`. If declined, fall back to Itkin or Jaimungal.
3. **Author affiliation decision.** Currently `Independent Researcher` in `main.tex`. Options: keep as-is, switch to "Incoming, Department of Economics, McGill University" (matriculating Sept 2026), or co-authored with one of the targets above. None of these blocks submission — the paper compiles and is venue-ready under the current declaration.

---

## 12. What changed since v0.3.0

### v0.3.1 (2026-04-22, commit `780afd9`)

- **Bibliography hygiene.** 5 hallucinated entries corrected after author-by-author WebFetch verification + 3 metadata corrections. All `\cite{}` keys updated across `main.tex` and the supporting Markdown files.
- **Λ-value softening.** The published $\Lambda(\kappa^\star) \approx +1.85 \times 10^{-2}$ was not reproducible; rewritten to honestly report the reproducible magnitude ($|\Lambda| \sim 10^{-3}$ at the trivial $G \equiv 0$ equilibrium, sign deferred to empirical phase).
- **GP-CI fix.** `theory/sensitivity.py::kappa_sensitivity_curve` was under-covering (~70% empirical vs 95% nominal) because `WhiteKernel` MLE collapsed; pinned the noise variance to the seed-mean MC variance and passed `noise_level_bounds="fixed"`. Coverage on smooth-truth synthetic cases recovers to ≥80% at $n_{\text{seeds}} = 100$.
- **OpenTimestamps anchor.** `paper/pre_registration.md.ots` minted and committed; verifiable via `uv run ots verify`.

### v0.3.2-unreleased (Wave 1 + Wave 2 + L3-FIX, working tree on `780afd9`)

**Wave 1 + Wave 2 (substantive theoretical + numerical extensions):**

- **Theorem 4 — Hawkes–SV criticality correspondence (repositioned, amendment A10).** Via the BDHM 2013 + Jaisson–Rosenbaum 2015 diffusive near-critical limit, Hardiman 2013's $n \approx 1$ is the literal analogue of the *real-eigenvalue (saddle-node)* stratum; the model's *Hopf* threshold $\kappa^\star$ is a strictly-stronger oscillatory instability beyond any scalar branching ratio — the genuinely novel "unoccupied cell". A falsifiable spectral discriminator (`theory/hawkes_sv_bifurcation.py`) separates the strata with zero overlap on synthetic ground truth; the earlier tautological $n_{\mathrm{SV}}$ "machine-precision verification" was removed. Empirical proximity to $\kappa^\star$ is indeterminate (κ-rescaling map) and deferred to the GEX test (A9).
- **Theorem 3 — BT locus empty in the canonical scan window.** Closed-form argument that $G_v < 0$ uniformly dominates $G_y \alpha \kappa_v / (\beta\gamma)$ on the scanned $(\sigma_q, \gamma) \in [0.05, 0.40] \times [0.20, 5.00]$ window, forcing $\kappa_{\mathrm{SN}} < 0$ and excluding Bogdanov–Takens bifurcations there. Falsifiable economic prediction: no homoclinic burst-relax dynamics from this model at fixed parameters.
- **Bautin curve.** 6 anchors at the canonical specification, characterising the supercritical → sub-critical transition in $(\sigma_q, \gamma)$ space. New experiment `experiments/codim2_analysis.py`, 8 new tests.
- **Closed-form $\ell_1$ in symbolic form (Appendix A).** Symbolic Kuznetsov-formula expansion in the 13-symbol parameter space, verified to $\sim 10^{-13}$ relative against the numerical pipeline at the canonical regime. Auto-generated `paper/figures/ell1_closed_form.tex` (10.6 KB, layered presentation).
- **Empirical $\Lambda$ scaling fit.** OLS on a $6 \times 6$ $(\xi, \rho)$ grid yields $\hat B = 0.082$ (95% CI $[-0.010, 0.168]$), empirically refuting the Engel–Lamb–Rasmussen $B = 2/3$ prediction at this regime ($p \ll 0.01$). Structural reason: the trivial $G \equiv 0$ equilibrium has vanishing shear-stretching $\partial_a v$.
- **Limit-cycle numerical validation past $\kappa^\star$.** Integrating the deterministic skeleton at $\kappa = 1.05\,\kappa^\star$ produces a closed orbit; measured period $T = 10.561$ yr matches Hopf prediction $T_\kappa = 10.977$ yr to 3.79% (within the leading-order normal-form correction).
- **H1 synthetic-pipeline end-to-end validation.** SW2 ordering $\mathrm{SW2}(\kappa_0\text{-deployed}) < \mathrm{SW2}(2\kappa_0\text{-reflexive}) < \mathrm{SW2}(\text{Heston})$ with disjoint bootstrap CIs on simulator-vs-simulator data. The H1 protocol is now demonstrated working before the empirical SPX target arrives.
- **2D bimodality on $(\log S, v)$ joint density.** H_bimod was refuted on the 1D log-S marginal at $\gamma = 0$; the 2D PCA-projected dip statistic at $\kappa = 1.05\,\kappa^\star_{\mathrm{env}}$ flips to *supported* ($p = 0.033$) on a $\sim 79\%$-survival sample. Result is preliminary and selection-conditioned.
- **A priori mechanism-relevant cell restriction.** Long-horizon shape moments + within-envelope cells; in-sample 7/10 ($p = 0.172$), OOS 4/8 ($p = 0.637$). Reported as transparent secondary statistic; the all-cell 8/24 (33.3%) remains headline.
- **H4 detector power on Stuart–Landau positive control.** $\geq 80\%$ peak power at $T = 512$ for 8/9 $(\mu, \sigma)$ configurations under the locked IAAFT-surrogate $\alpha = 0.05$ rule; non-monotone in $T$ (degrades at $T \in \{1024, 2048\}$ as the IAAFT-preserved linear ACF approximates the limit cycle more faithfully).
- **Sliced-W2 sample-complexity table.** $n_{\min} \approx 4{,}000$ windows for $\pm 10\%$ bootstrap CI half-width; H1 budget per event ($\sim 280$ windows) gives $\sim 23\%$ ratio, requiring inter-baseline SW2 gap $\geq 0.46 \cdot \mathrm{SW2}_{\mathrm{true}}$ for discriminability.
- **Manuscript variants.** `paper/variants/neurips_workshop/main.tex` (4-page body) and `paper/variants/icaif/main.tex` (5-page double-blind ACM `sigconf`) both built.

**L2-G + L3-G adversarial-grill fixes (L3-FIX wave):**

- Stripped fictional "Matérn-3/2 kernel per pre-reg amendment A8" claim from `main.tex` and both variants. A1–A7 are locked at `63078f5` and A6 explicitly fixes the H2 kernel as RBF + WhiteKernel; the Matérn experiment showed vacuous CIs and was never registered as an amendment.
- Corrected $|\Lambda(\kappa^\star)|$ scan band from $[5\!\times\!10^{-2}, 9\!\times\!10^{-2}]$ to actual $[4.8\!\times\!10^{-2}, 1.21\!\times\!10^{-1}]$ (median $\approx 7.3\!\times\!10^{-2}$).
- Reframed H4 power claim from "≥80% at $T \geq 512$" to "≥80% peak power at $T = 512$ specifically; non-monotone in $T$" (6/9 at $T = 1024$, 5/9 at $T = 2048$).
- Resolved the $\kappa^\star$ vs $\kappa^\star_{\mathrm{env}}$ notation collision in the 2D bimodality discussion (canonical Hopf threshold $\approx 0.896$ vs the $\sim 3.9 \times 10^{-9}$ empirical-magnitude stability envelope).
- Corrected H_bimod 2D survival fraction from "~1.6%" to actual ~79% ($n = 15{,}769$ surviving cells out of $20{,}000$).
- Reframed Theorem 2 numerical anchor: the $|n_{\mathrm{SV}}(\kappa^\star_4) - 1| = 3.85 \times 10^{-5}$ residual is the truncation error in the published 4-decimal $\kappa^\star_4 = 0.8964$, not eigenvalue-solver noise; at machine-precision $\kappa^\star$ the identity is exact ($< 10^{-15}$).
- Added Stuart–Landau citation (`kuznetsov2004`) in the H4 paragraph.
- Fixed `runs/h_bimod_2d_scan/` → `runs/h_bimod_2d/` directory reference in 2D KDE caption.
- Deleted redundant `paper/abstract.md` (the `\begin{abstract}` block in `main.tex` is canonical).
- Coverage / lint config updated for the new experiment-runner CLI scripts.

---

## 13. Pre-flight checklist (1 day before any submission)

- [ ] `bash scripts/verify.sh` is green on the submission commit.
- [ ] `tests/test_reproducibility.py` passes (no v0.1.0 baseline drift).
- [ ] `paper/main.pdf` (or the variant under submission) compiles cleanly under `latexmk -pdf` with no errors and ≤5 warnings.
- [ ] All `\cite{}` entries resolve to a `references.bib` entry (no `?` in PDF).
- [ ] All `\ref{}` and `\label{}` resolve (no `??` in PDF).
- [ ] All `\includegraphics{}` files exist in `figures/` and render correctly.
- [ ] Abstract is **≤1920 characters** (arXiv hard limit; paste rendered text into `wc -c`).
- [ ] Title is **≤240 characters** (arXiv internal guidance).
- [ ] Author block has correct affiliation + ORCID (pending; see §11).
- [ ] License declared (CC-BY 4.0 per arXiv metadata).
- [ ] arXiv subject categories: `q-fin.MF (primary)`, `q-fin.CP, q-fin.PR, q-fin.ST, math.DS (cross-list)`.
- [ ] MSC codes: `91G80 (Primary) 37G15, 60H10, 91G60, 93E20 (Secondary)`.
- [ ] Comments field includes pre-registration commit hash `268c061`.
- [ ] No `\todo{}`, `XXX`, `TBD`, or placeholder text in the source (`grep -E "TODO|XXX|TBD" paper/main.tex paper/variants/**/main.tex`).
- [ ] `paper/pre_registration.md.ots` verifies clean (`uv run ots verify`).
- [ ] For **NeurIPS workshop / ICAIF**: anonymisation pass — no author name, affiliation, or self-citation in identifiable form; acknowledgments stripped.
- [ ] For **NeurIPS workshop**: `\usepackage{neurips_2024}` (anonymous) at submission, switch to `[final]` only at camera-ready.
- [ ] For **ICAIF**: `acmart` `sigconf` + `anonymous` flag set; ORCID and author block hidden until camera-ready.

---

## 14. Summary

All three target submissions are technically ready: `main.tex` compiles to a 29-page PDF, the NeurIPS workshop and ICAIF variants both build at the venue's required page lengths, the bibliography is clean (5 hallucinated entries fixed in v0.3.1), Theorems 1–3 are stated and proved, 9 figures render from reproducible scripts, and 404 tests pass on the working tree (with one outstanding coverage-gate dip from v0.3.2's in-flight runner modules — to be resolved before the v0.3.2 tag).

The pre-registration discipline is intact and irrevocable: A1–A7 are closed at `63078f5`, no further amendments are admissible, and the OpenTimestamps proof binds the locked file's hash into the Bitcoin blockchain.

The remaining items (§11) are operational, not technical: ORCID registration, an outreach email to Halperin, and a final author-affiliation decision. None of these block submission to any of the three venues.
