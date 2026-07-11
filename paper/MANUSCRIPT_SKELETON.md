# Manuscript Skeleton — *Reflexivity in Options Markets*

> **ARCHIVED / WITHDRAWN.** This predates the v0.4 centered-model
> reconstruction. Its model, anchors, stochastic shift, and empirical
> hierarchy are not current. Use `paper/main.tex` and Amendments A13--A15.

> **⚠ Superseded planning artifact (pre-v0.3.9).** This skeleton predates the v0.3.9 empirical-leg redesign. Wherever it describes **H1** as a "primary realism" RL-agent / sliced-W2 test or **H4** as a "Hopf-frequency spectral peak via Welch's method", those are retracted: the primary empirical test is now a direct, model-free **dealer-gamma (GEX) regression (H1′)** and H4 is a **critical-slowing-down** early-warning test. The Hawkes–SV theorem was repositioned (no `n_SV` numerology) and theorem numbering shifted (Hawkes–SV is now Theorem 5). The authoritative state is `paper/main.tex` + amendments **A8–A11** in `paper/pre_registration_amendments.md`. Kept as the historical authoring blueprint, not a current spec.

**Target.** 8-page workshop format (NeurIPS GenAI in Finance Workshop 2026, expected double-blind). The same skeleton scales to 24 pages for a journal version (e.g., *Quantitative Finance*) by expanding the proofs, the empirical roadmap, and the threats-to-validity section.

**How to use this file.** Each section below is a self-contained authoring brief. For each section: (a) the title and word-count target; (b) the equations to include and where they live in `paper/theory.md`; (c) the figures to include from `paper/figures/`; (d) the tables to write; (e) the citations to lift from `paper/related_work.md`; (f) the *status* tag, telling you whether prose is ready to lift, needs new prose, or borrows from a `~/Documents/reflexivity-research/*_brief.md` source.

**Status tags.**
- `WRITTEN_IN_FRAGMENT`: the prose exists in `paper/<file>.md` and can be paraphrased / lifted directly.
- `NEEDS_NEW_PROSE`: synthesizing across fragments; new authoring required.
- `BORROWS_FROM_BRIEF`: technical detail lives in `~/Documents/reflexivity-research/<brief>.md` and needs distillation.

**Word budget.** ~3,200 words main body for the 8-page workshop format (assuming ~400 words/page after equations and figures). Distribution shown below; pad / trim per section as the LaTeX renders.

---

## Section 1 — Introduction (1 page, ~400 words)

**Status.** `NEEDS_NEW_PROSE` — synthesizes across `paper/abstract.md`, `paper/related_work.md` §1, `paper/theory.md` §1.

### Outline

1. **Hook (50 words).** Real options markets exhibit endogenous volatility cycles that pure stochastic-volatility (SV) models cannot reproduce. The Hardiman-Bercot-Bouchaud (2013) finding that the Hawkes branching ratio sits at $n \approx 1$ on E-mini futures (1998-2011) is the empirical anchor; the question is what continuous-time mechanism produces this critical reflexivity.
2. **Contribution statement (150 words, four bullets).**
   - We introduce a 3D reflexive SV simulator $(S_t, v_t, z_t)$ with explicit dealer-gamma feedback derived from the option open-interest grid via the Gârleanu-Pedersen-Poteshman (2009) demand-pressure mapping.
   - We prove a Hopf bifurcation theorem (Theorem 1) for the deterministic skeleton at $\kappa^\star$ given by Liu's criterion, with closed-form first Lyapunov coefficient $\ell_1$ for log-normal open-interest in moneyness.
   - We compute the stochastic-Hopf shift $\Lambda$ via Khasminskii's sphere process and document the parametric phase boundary in $(\sigma_q, \gamma)$.
   - We pre-register the empirical evaluation pipeline (sliced-Wasserstein-2 over 21-day surface windows, κ-sensitivity slope, Hopf-frequency spectral test) at git commit `268c061` with OpenTimestamps Bitcoin-anchored proof.
3. **Comparison snapshot (100 words).** Closest precedents: Halperin & Itkin (2025) Marketron — same memory-channel structure, no bifurcation analysis; Dai (2025) — same dealer-gamma mechanism, saddle-node onset (not Hopf, since the 2D reduction has an upper-triangular Jacobian); He–Li–Zheng (2025 NeurIPS) — same RL-with-environmental-sweep concept, separate agents per grid point rather than slope-at-anchor.
4. **Roadmap (50 words).** §2 the model; §3 the Hopf theorem with closed-form $\ell_1$; §4 the numerical phase diagram; §5 the pre-registered evaluation framework; §6 the mechanism-decomposition vs. Marketron; §7 conclusions and the empirical-calibration roadmap (gated on Phase 4 of the project TODO).
5. **Pointer to repository (50 words).** Code at `github.com/mahimn01/reflexive-options`, Zenodo DOI [TBD]; pre-registration anchored to commit `268c061` (initial public commit, 2026-04-29) per Camerer et al. (2018) template adapted for computational finance.

### Citations needed (lift from `related_work.md`)
- Hardiman, Bercot & Bouchaud (2013) — `related_work.md` §1.5
- Gârleanu, Pedersen & Poteshman (2009) — `related_work.md` §1.6 + URL block
- Halperin & Itkin (2025) — `related_work.md` §1.1
- Dai (2025) — `related_work.md` §1.2
- He, Li & Zheng (2025 NeurIPS) — `related_work.md` §2.1
- Camerer et al. (2018) — `related_work.md` §4.4
- Kuznetsov (2004) — cited in `theory.md` §4

### Equations
None in the introduction — keep the prose clean.

### Figures / tables
None in the introduction.

---

## Section 2 — The reflexive model (1 page, ~400 words)

**Status.** `WRITTEN_IN_FRAGMENT` — directly lift from `paper/theory.md` §1 and `paper/notation.md`.

### Outline

1. **System (200 words + equations 1a-d).** State the 3D SDE $(S_t, v_t, z_t)$ with explicit Heston backbone (eq. 1a, 1b), the memory equation (1c) that closes the leverage feedback, and the correlation structure (1d). Define $G(S, z, v)$ as the dealer-gamma aggregator from the open-interest grid.
2. **Why three states (150 words).** The 2D reduction has an upper-triangular Jacobian $J_{2D}$ with eigenvalues $\{a(\kappa), -\kappa_v\}$ — only saddle-node / transcritical bifurcations possible. To Hopf, you need cyclic bidirectional feedback: price → variance and variance → price. The minimal modification is the leverage channel $\gamma z$ with the auxiliary memory equation (1c). This argument is structural, not aesthetic — Dai (2025) derived precisely the saddle-node condition our 2D reduction would produce.
3. **Reduction to Heston (50 words).** When $\kappa = \gamma = 0$, the system collapses to standard Heston (1993). Both `synthetic_replication.py` and `tests/test_simulator.py` verify the collapse numerically.

### Equations to include (from `theory.md` §1)
- Eq. (1a) — spot SDE with dealer-gamma drift
- Eq. (1b) — Heston variance with leverage feedback $\gamma z$
- Eq. (1c) — memory equation for $z_t$
- Eq. (1d) — Brownian correlation
- Definition of $G(S, t) = \sum_{K, T} q_{K, T} \cdot \Gamma_{K, T}(S, t) \cdot \text{sign}(K, T)$ (from `notation.md` §"Open-interest grid")

### Figures / tables
None — model statement should fit on a single page.

### Citations needed
- Heston (1993) — `related_work.md` URL block
- Gârleanu, Pedersen & Poteshman (2009) — already in §1
- Dai (2025) — already in §1, here cite for the 2D-saddle-node parallel
- Halperin & Itkin (2025) — already in §1, cite the memory-variable parallel

---

## Section 3 — Hopf bifurcation theorem and closed-form ℓ₁ (2 pages, ~800 words)

**Status.** `WRITTEN_IN_FRAGMENT` — directly lift from `paper/theory.md` §§2-4. This is the technical heart of the paper.

### Outline

1. **Equilibria and Jacobian (200 words + Eq. 2-3).** State eq. (2) — the equilibrium equation defining $(S^\star, \theta_v, z^\star)$. State Eq. (3) — the linearised Jacobian $J(\kappa)$ in deviation variables. Define the partials $G_x, G_v, G_z$ at the equilibrium. (Lift verbatim from `theory.md` §2.)
2. **Routh-Hurwitz / Liu's criterion (150 words + Eq. 4-5).** State the characteristic polynomial $P(\lambda; \kappa) = \lambda^3 + c_2\lambda^2 + c_1\lambda + c_0$ with explicit $c_2, c_1, c_0$. State the Hopf condition $H(\kappa) := c_1 c_2 - c_0 = 0$ (Eq. 4) and the threshold definition $\kappa^\star$ (Eq. 5). Frequency $\omega^\star = \sqrt{c_1(\kappa^\star)}$. (Lift verbatim from `theory.md` §3.)
3. **Theorem 1 statement + proof sketch (250 words).** State Theorem 1 verbatim (assumptions A1-A5, conclusions 1-3 about the limit-cycle family $\Gamma_\kappa$). Two-paragraph proof sketch via implicit function theorem on $P(\lambda; \kappa) = 0$, centre manifold reduction (Kuznetsov 2004 Thm 5.4), Poincaré normal form (Kuznetsov 2004 Thm 3.3). (Lift from `theory.md` §4.)
4. **Numerical anchor (100 words + Table).** Report the canonical regime values: $\kappa^\star = 0.8964$, $\omega^\star = 0.5724$ rad/yr, $\ell_1 = -2.53 \times 10^{-2}$ (supercritical), $|\Lambda| \sim 10^{-3}$ at the §4.2 trivial-equilibrium evaluation; sign configuration-dependent and deferred to the empirical phase. (Lift Table from `theory.md` §4.2.)
5. **Closed-form ℓ₁ for log-normal OI (100 words + Eq. 17-18).** State the log-normal OI aggregator (Eq. 14-15a-c), state the closed-form $\kappa^\star$ (Eq. 17) and the closed-form $\ell_1$ via Kuznetsov 2004 eq. 3.20 (Eq. 18). Cite the verification: closed form vs FD-tensor pipeline agree to <0.6% relative on every test parameter set (`tests/test_lognormal_lyapunov.py`). (Lift from `theory.md` §4.3.)

### Equations to include (all from `theory.md` §§2-4)
- Eq. (2) — equilibrium equation
- Eq. (3) — Jacobian $J(\kappa)$
- Eq. (4) — Hopf condition (Liu's criterion)
- Eq. (5) — boxed threshold $\kappa^\star$
- Theorem 1 statement (boxed display block)
- Eq. (14) — log-normal OI aggregator integral
- Eq. (15a-c) — closed-form $G(a, v)$
- Eq. (17) — boxed closed-form $\kappa^\star$
- Eq. (18) — Kuznetsov ℓ₁ formula

### Tables to include
- Table from `theory.md` §4.2: $\{\kappa^\star, \omega^\star, \lambda_3, \ell_1, \text{bifurcation type}\}$ at the canonical regime — single column, four rows.

### Figures
None in §3 (the phase diagram lives in §4).

### Citations
- Kuznetsov (2004) — for centre manifold theorem and ℓ₁ formula
- Liu (1994) for the equivalent Routh-Hurwitz formulation (cite via Kuznetsov)
- Khasminskii (1980) — for the stochastic shift $\Lambda$ method (cite when introducing $\Lambda$ at the end of the section)
- Engel, Lamb & Rasmussen (2024) — for the shear-induced corrections to $\Lambda$

### Notes for authoring
- Theorem 1 should be in a `\begin{theorem}` block with proof sketch in `\begin{proof}[Proof sketch]`. The full proof goes to an appendix in the journal version; for the workshop format, the proof sketch is sufficient.
- The closed-form $\ell_1$ result is the single strongest novelty hook for the workshop — emphasize that no precedent (Marketron, Dai, He–Li–Zheng, Brock–Hommes-Wagener) provides it.

---

## Section 4 — Numerical phase diagram (1 page, ~400 words)

**Status.** `WRITTEN_IN_FRAGMENT` — lift from `paper/theory.md` §4.4 and the existing `paper/figures/hopf_phase_diagram.tex` caption.

### Outline

1. **Setup (100 words).** The full $(\kappa, \xi, \rho, \sigma_v)$ phase diagram is computed by `python -m reflexive_options.experiments.hopf_phase_scan_4d`. Sweep $\kappa \in [0, 2]$ on a 401-point grid for each of $31 \times 21 \times 4 = 2{,}604$ cells over $(\xi, \rho, \sigma_v)$ at the canonical Hopf-exhibiting regime. Reports $\kappa^\star$ as the contour where $\mathrm{Re}\,\lambda_\pm$ crosses zero. Wall-clock: ~16 s on a single M-series core.
2. **Figure 1 + caption (200 words).** Embed `figures/hopf_phase_diagram.pdf` (already exists, 28KB vector). Use the caption verbatim from `figures/hopf_phase_diagram.tex` — Panel A is the heatmap of $\kappa^\star(\xi, \rho)$; Panel B is four line cuts of $\kappa^\star(\sigma_v)$ at representative $(\xi, \rho)$ probes.
3. **Substantive interpretation (100 words).** The no-Hopf wedge in the upper-left (high $\xi$, strong negative $\rho$) is consistent with the Engel-Lamb-Rasmussen shear-induced destabilization. The four line cuts substantiate the §3 claim that real options markets sit *near but not across* $\kappa^\star$ in the SPX-relevant $(\xi, \rho) \approx (0.30, -0.70)$ corner. Pre-registered hypothesis H4 (`pre_registration.md` §3) tests for the $\omega^\star$-band spectral peak in absolute returns.

### Figure
- `paper/figures/hopf_phase_diagram.pdf` (already exists). Caption: lift from `paper/figures/hopf_phase_diagram.tex` (already written).

### Bonus — closed-form ℓ₁ phase boundary (if space permits)
- `paper/figures/ell1_phase_boundary.pdf` (already exists, 51KB vector) — supercritical / sub-critical / no-Hopf wedges in $(\sigma_q, \gamma)$ space at the canonical specification. Cite §4.3.4 of `theory.md`. **For the 4-page workshop variant, drop this figure to the appendix; for the 8-page version, include in §4.**

### Citations
- Engel, Lamb & Rasmussen (2024) — already cited in §3
- No new citations needed

---

## Section 5 — Pre-registered evaluation framework (1 page, ~400 words)

**Status.** `WRITTEN_IN_FRAGMENT` (lift from `paper/pre_registration.md` §§2-6 + `paper/related_work.md` §§2-3) + `BORROWS_FROM_BRIEF` (`~/Documents/reflexivity-research/evaluation_framework_brief.md` for the W2 motivation).

### Outline

1. **Two evaluation primitives (100 words).** (i) Sliced-Wasserstein-2 over arbitrage-free 21-day surface windows — the realism metric. (ii) κ-sensitivity slope at the calibrated coupling $\kappa_0$ — the reflexivity-importance scalar. Both committed to the analysis pipeline at commit `268c061` before any empirical SPX data is touched.
2. **H1 — primary realism hypothesis (100 words).** Reflexive-trained agent's IV-surface distribution is closer (in sliced-W2) to empirical SPX than four baseline-trained agents (time-dep Heston, LSV, 3/2 SV, gamma-aware-non-reflexive). Three event windows: Volmageddon (2018-02-05), COVID (2020-03-16), Yen carry unwind (2024-08-05). Decision rule: 12 pairwise dominance checks (4 baselines × 3 windows) with non-overlapping 95% block-bootstrap CIs.
3. **H2/H3/H4 — secondary hypotheses (100 words).** H2 is the κ-sensitivity slope $\tilde{\rho} = \partial \hat{m}/\partial \kappa\rvert_{\kappa_0}$ — positive and statistically distinguishable from zero for the reflexive-trained agent, statistically indistinguishable for baselines (TOST equivalence). H3 is the post-FOMC RR25 + ATM-term-structure shifts. H4 is the Hopf-frequency spectral peak in $|r_t|$ at $\omega^\star \pm 20\%$ via Welch's method (post-amendment A2: dual signal on $|r_t|$ and realised-variance proxy $\widehat{v}_t$).
4. **Differentiation from precedents (100 words).** Cite Ning et al. (2021/2024) for $W_1$ on single-day marginals (we use sliced-$W_2$ on 21-day windows); cite He, Li & Zheng (2025 NeurIPS) for the per-grid-point Wasserstein-ball Table 1 (we report the slope-at-anchor scalar from a single agent); cite Subbaswamy, Adams & Saria (2022 NeurIPS) for the local-2nd-order-approximation framing of robustness.

### Tables
- Table 1: H1-H4 hypothesis grid (4 rows × {hypothesis, primary metric, decision rule}). Lift from `pre_registration.md` §6.

### Figures
- For the 8-page version: `paper/figures/h4_detector_power.pdf` (already exists, 22KB vector) — H4 detector power curves at canonical settings. **Drop to appendix in 4-page workshop variant.**

### Citations
- Ning, Jaimungal, Zhang & Bergeron (2021/2024)
- He, Li & Zheng (2025 NeurIPS)
- Subbaswamy, Adams & Saria (2022)
- VolGAN (Vuletić & Cont 2025)
- Bonneel, Rabin, Peyré & Pfister (2015) — sliced-Wasserstein definition
- Manole, Balakrishnan & Wasserman (2022) — minimax CIs for sliced-W
- Politis & Romano (1994) — block bootstrap
- Roper (2010) + Gatheral & Jacquier (2014) — arbitrage filter
- Faff (2022 PBFJ) — pre-registration in finance precedent
- Pérignon et al. (2024 RFS) — reproducibility crisis in finance

### Notes
- For workshop submission anonymity: do **not** mention the OpenTimestamps proof or the personal commit hash in the paper body — it's a deanonymizing detail. Move to camera-ready version.

---

## Section 6 — Mechanism decomposition vs. Marketron (0.5 page, ~200 words)

**Status.** `WRITTEN_IN_FRAGMENT` — lift the headline numbers from `paper/mechanism_decomposition.md`.

### Outline

1. **Setup (75 words).** Explain that the two SDEs (reflexive vs. Marketron quasi-particle) are *not* the same and a 1:1 reproduction is mathematically impossible. Instead we report mechanism decomposition: 5D coarse-grid tuning over $(\kappa, \gamma, T_{\text{eff}}, \mu_q, \sigma_q)$, then per-cell classification into `shape_target` / `level_artifact` / `calibration_artifact`. The headline gate is the shape-match rate on `shape_target` cells.
2. **Headline number + Table 2 (75 words).** **8/24 shape-feature cells match (33.3%) at the per-set tuned coupling, validated at 10k paths.** Long-horizon excess-kurt sign agrees on 5/6 horizons in `table_5_calibrated_2017` (Marketron Table 8). Long-horizon skew sign disagrees predictably under risk-neutral drift — Marketron's positive skew comes from Bessembinder compounding under calibrated drift; ours comes from the dealer-gamma + leverage feedback under $\mu = 0$.
3. **Implication (50 words).** The two mechanisms are independent in parameterization but agree on the long-horizon excess-kurt sign — the most robust agreement. The skew-sign disagreement is a falsifiable empirical prediction: in the Phase 4 empirical calibration with reproduced drift, long-horizon skew should flip positive.

### Tables
- Table 2 (compressed from `mechanism_decomposition.md` Tables 1-3): three rows showing aggregate shape-match rate, predictable divergences, level artifacts not chased. Single-page format; reduce from the full mechanism-decomposition document.

### Figures
None — the mechanism table carries the section.

### Citations
- Halperin & Itkin (2025) — already cited multiple times
- Bessembinder (2018), Farago & Hjalmarsson (2023) — for the multiplicative-compounding-positive-skew framing
- Gârleanu, Pedersen & Poteshman (2009) — already cited

### Notes
- This section is the **defensive shield against the Marketron-side review**. The framing is: "we are not Marketron, we explain the predictable disagreements, the agreement we do find is structurally meaningful." This matches `threats_to_validity.md` §1 verbatim.

---

## Section 7 — Conclusions and future work (0.5 page, ~200 words)

**Status.** `NEEDS_NEW_PROSE` — synthesizing across everything.

### Outline

1. **Recap (50 words).** Three contributions: (a) the 3D reflexive SV simulator with closed-form Hopf threshold and $\ell_1$ for log-normal OI; (b) the pre-registered evaluation framework for RL-hedged surface dynamics; (c) the Marketron mechanism-decomposition that isolates dealer-gamma feedback from potential-well memory.
2. **Empirical extension (75 words).** The empirical calibration to real SPX surfaces is the natural Phase 4 follow-up, gated on data acquisition (UofT WRDS or `historicaloptiondata.com` ALLSPX bundle). The pre-registration locks the analysis pipeline before that calibration touches the data; results — confirming or refuting H1-H4 — are pre-committed to publication regardless of direction.
3. **Open theoretical items (50 words).** Closed-form $\Lambda(\kappa)$ via Engel-Lamb-Rasmussen-style asymptotics adapted to Heston multiplicative noise (currently numerical only); formal Hawkes-$n$ ↔ SV-Jacobian-eigenvalue reduction relating $\kappa^\star$ to Hardiman-Bercot-Bouchaud's empirical $n \approx 1$ (most ambitious; would itself be a separate paper).
4. **Reproducibility statement (25 words).** All experiments reproducible via `bash scripts/verify.sh` from commit `268c061`; pre-registration anchored via OpenTimestamps; Zenodo DOI [TBD]. (For double-blind workshop submission: **omit the commit hash and the GitHub URL until camera-ready**.)

### Citations
- Hardiman, Bercot & Bouchaud (2013) — already cited in §1
- Engel, Lamb & Rasmussen (2024) — already cited
- Pérignon et al. (2024 RFS) — already cited

---

## Section R — References (off-page, ~50 entries)

**Status.** `NEEDS_NEW_PROSE` — needs `references.bib` file built from `paper/related_work.md` URL blocks.

### Source for each entry

The URLs are consolidated at the bottom of `paper/related_work.md`. Convert each to a BibTeX entry:

| Author / Year | BibTeX type | Source |
|---|---|---|
| Halperin & Itkin (2025) | `@article` (arXiv) | URL block |
| Dai (2025) | `@article` (arXiv) | URL block |
| Heston (1993) | `@article` (RFS) | URL block |
| Gârleanu, Pedersen & Poteshman (2009) | `@article` (RFS) | URL block |
| Brock & Hommes (1998) | `@article` (Econometrica) | `hopf_bifurcation_brief.md` §2.1 |
| Brock, Hommes & Wagener (2009) | `@article` (JEDC) | URL block |
| Hardiman, Bercot & Bouchaud (2013) | `@article` (EPJ B) | URL block |
| Filimonov & Sornette (2012) | `@article` (PRE) | `hopf_bifurcation_brief.md` §2.1 |
| He, Li & Zheng (2009) | `@article` (Economic Modelling) | URL block |
| Chiarella, He & Hommes (2006) | `@techreport` (UTS QFRC) | URL block |
| Frey & Stremme (1997) | `@article` (Math Finance) | URL block |
| Sircar & Papanicolaou (1998) | `@article` (App Math Finance) | URL block |
| Platen & Schweizer (1998) | `@article` (Math Finance) | URL block |
| Engel, Lamb & Rasmussen (2024) | `@article` (PTRF) | URL block |
| Kuznetsov (2004) | `@book` (Springer) | `theory.md` §4 |
| Arnold (1998) | `@book` (Springer) | `hopf_bifurcation_brief.md` §2.2 |
| Baxendale (1994) | `@article` (PTRF) | `hopf_bifurcation_brief.md` §2.2 |
| Khasminskii (1980) | `@book` (Sijthoff) | for $\Lambda$ method |
| He, Li & Zheng (2025 NeurIPS) | `@inproceedings` (NeurIPS) | URL block |
| Hou et al. (2025) | `@article` (arXiv / NeurIPS poster) | URL block |
| Murray, Wood, Buehler et al. (2022) | `@article` (arXiv) | URL block |
| Subbaswamy, Adams & Saria (2022 NeurIPS) | `@inproceedings` (NeurIPS) | URL block |
| Packer et al. (2018) | `@article` (arXiv) | URL block |
| Buehler, Gonon, Teichmann & Wood (2019) | `@article` (Quantitative Finance) | URL block |
| Cao, Chen, Hull & Poulos (2021) | `@article` (arXiv) | URL block |
| Ning, Jaimungal, Zhang & Bergeron (2024) | `@article` (SIAM SIFIN) | URL block |
| Vuletić & Cont (2025 VolGAN) | `@article` (App Math Finance) | URL block |
| Choudhary, Jaimungal & Bergeron (2024 FuNVol) | `@article` (Quantitative Finance) | URL block |
| Issa, Horvath, Lemercier & Salvi (2023 NeurIPS) | `@inproceedings` (NeurIPS) | URL block |
| Jin & Agarwal (2025) | `@article` (arXiv) | URL block |
| Chevallier, De Marco & Lévy-dit-Vehel (2025) | `@article` (arXiv) | URL block |
| Luan & Hamp (2025) | `@article` (DSFE) | URL block |
| Bonneel, Rabin, Peyré & Pfister (2015) | `@article` (J Math Imaging Vis) | URL block |
| Manole, Balakrishnan & Wasserman (2022) | `@article` (EJS) | URL block |
| Politis & Romano (1994) | `@article` (JASA) | URL block |
| Roper (2010) | `@techreport` (Sydney) | URL block |
| Gatheral & Jacquier (2014) | `@article` (Quantitative Finance) | URL block |
| Hosseini, Hsu & Taghvaei (2023) | `@article` (arXiv) | URL block |
| Faff (2022 PBFJ) | `@article` (PBFJ editorial) | URL block |
| Pérignon et al. (2024 RFS) | `@article` (RFS) | URL block |
| Camerer et al. (2018) | `@article` (Nature Human Behaviour) | URL block |
| Huang et al. (2024 Open RL Benchmark) | `@article` (arXiv) | URL block |
| Sun et al. (2024) | `@article` (arXiv) | URL block |
| Bessembinder (2018) | `@article` (JFE) | `mechanism_decomposition.md` references |
| Farago & Hjalmarsson (2023) | `@article` (Review of Finance) | `mechanism_decomposition.md` references |
| SqueezeMetrics (2017 GEX) | `@misc` (whitepaper) | `dealer_gamma_brief.md` §2 |
| Barbon & Buraschi (2021) | `@article` (working paper) | `dealer_gamma_brief.md` §2 |
| Lee (2004) | `@article` (Math Finance) | `pre_registration.md` §4 |
| Soros (2009) | `@misc` (OSF) | `theory.md` §1 |
| Hartigan & Hartigan (1985) | `@article` (Annals of Stats) | `theory.md` §7.4 |
| Feller (1951) | `@article` (Annals of Math) | `theory.md` §7 |
| Lord, Koekkoek & van Dijk (2010) | `@article` (Quant Finance) | `theory.md` §7.2 |

Total: ~50 entries. Effort: 1 day to assemble; use `arxiv2bib`, `doi2bib`, and Google Scholar's BibTeX export to bootstrap each.

---

## Optional appendix sections (8-page version + journal version)

Drop these to appendix in the workshop format:

- **Appendix A — Stationary density (200 words).** From `theory.md` §7. The Fokker-Planck stationary density vs. Heston comparator; H_tail / H_skew / H_bimod findings table.
- **Appendix B — Closed-form ℓ₁ derivation details (300 words + Eq. 14-18).** From `theory.md` §4.3. The full derivation of $G(a, v)$ via Gaussian-product identity; the third-order partials; verification table at the canonical regime.
- **Appendix C — Mechanism decomposition full tables (300 words).** From `mechanism_decomposition.md` Tables 1-3.
- **Appendix D — Pre-registration full document (1 page).** Reference to `pre_registration.md` and OpenTimestamps proof; for camera-ready only.

---

## 4-page workshop variant compression

For the 4-page workshop format, the cuts are:

| Section | 8-page word target | 4-page word target | What gets cut |
|---|---|---|---|
| §1 Introduction | 400 | 250 | Drop the comparison snapshot; keep only the contribution bullets and roadmap |
| §2 Model | 400 | 250 | Drop the "why three states" structural argument (move to appendix); keep the SDE and Heston-collapse note |
| §3 Hopf theorem + ℓ₁ | 800 | 600 | Drop the proof sketch (cite Kuznetsov 2004 and the appendix); keep Theorem 1 statement, eq. 17, eq. 18 closed forms, numerical anchor table |
| §4 Phase diagram | 400 | 200 | Keep figure + 3-sentence caption; drop the substantive interpretation paragraph |
| §5 Evaluation | 400 | 250 | Drop H3/H4 detail; keep H1 + H2 + the differentiation-from-precedents paragraph |
| §6 Mechanism | 200 | 150 | Drop the implication paragraph; keep the headline number + Table 2 |
| §7 Conclusions | 200 | 150 | Drop the open theoretical items; keep recap + empirical extension + reproducibility |
| **Total main body** | **2,800** | **1,850** | — |

The 4-page variant is essentially a "spotlight talk" version — it makes the narrative airtight at the cost of the technical depth. Reviewers will appreciate the depth pointers to the arXiv preprint.

---

## Section ordering rationale

The ordering above is **theorem-first** (Hopf in §3 before evaluation in §5). The alternative is **evaluation-first** (lead with the pre-registration discipline as the headline novelty). Choice depends on the venue:

- **NeurIPS GenAI in Finance Workshop:** evaluation-first. The pre-registered + RL + sliced-W2 protocol is what the workshop's program committee will care about. Reorder to §5 → §3 → §2 → §6 → §1.
- **arXiv q-fin.MF / journal version:** theorem-first (the ordering above). The closed-form $\ell_1$ and the Hopf threshold are the mathematical contribution; the workshop / RL piece is the application.

**Recommendation:** author the manuscript theorem-first (this skeleton); for the workshop submission, swap §3 and §5 in the 4-page variant.

---

## Cross-reference index

For each section, the in-fragment source it draws from:

| Section | Primary source | Secondary source |
|---|---|---|
| §1 Introduction | NEW PROSE | `abstract.md`, `related_work.md` §§1-4 (one-liner each) |
| §2 Model | `theory.md` §1 | `notation.md` |
| §3 Hopf + ℓ₁ | `theory.md` §§2-4 | `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §3-4 |
| §4 Phase diagram | `theory.md` §4.4 + `figures/hopf_phase_diagram.tex` | — |
| §5 Evaluation | `pre_registration.md` §§2-6 | `~/Documents/reflexivity-research/evaluation_framework_brief.md`; `related_work.md` §§2-3 |
| §6 Mechanism | `mechanism_decomposition.md` (compressed) | — |
| §7 Conclusions | NEW PROSE | `theory.md` §8, `pre_registration.md` §11 |
| References | NEW PROSE | `related_work.md` URL block (consolidated) |

---

## Authoring order recommendation

Don't write top-to-bottom. The recommended order maximizes the % of fragment-lift early (which is faster than new prose):

1. **Day 1: §3 (Hopf + ℓ₁)** — the most fragment-lift, longest section, pure copy-paste from `theory.md`. Get the LaTeX-math habits warmed up.
2. **Day 2: §2 (model) + §6 (mechanism)** — both are short fragment-lift sections.
3. **Day 3: §4 (phase diagram) + §5 (evaluation)** — fragment-lift with light synthesis.
4. **Day 4: §1 (intro) + §7 (conclusions)** — pure new prose, but informed by what you've already written.
5. **Day 5: references.bib** — mechanical, paste-and-edit.
6. **Day 6: full latexmk build + first proofread.**
7. **Day 7: buffer / external-feedback response window / abstract trim.**

---

## Final notes

- **Equations should be numbered consistently** with `theory.md` so reviewers can cross-check against the public Markdown source. Use the same `(1a)`, `(1b)`, `(2)`, `(3)`, etc. labels.
- **Figure captions should be self-contained** — a reviewer skimming should be able to read just the caption and grasp the result. The existing `figures/hopf_phase_diagram.tex` caption is the model.
- **Anonymize for workshop submission.** Strip: GitHub URL, commit hash, ORCID,
  OpenTimestamps proof reference, and the author name. Add back only accurate
  author metadata at camera-ready; do not introduce an institutional affiliation.
- **Acknowledgments only at camera-ready** for the workshop; can include from arXiv submission.
- **Ditch every footnote** in the 4-page workshop version. They burn vertical space and disrupt the reader's flow at this density.
