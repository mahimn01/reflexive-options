# Related Work

This section documents, for each of the four contributions of *Reflexivity in
Options Markets* (Patel 2026), the closest published precedents and an
ingredient-by-ingredient comparison. The intent is adversarial: we identify
the paper most likely to be confused with each contribution and enumerate
exactly which load-bearing ingredients we share with it and which we do not.

The table format throughout is `Yes` / `No` / `Partial`, with a one-clause
justification per cell. Where an audit brief in
`~/Documents/reflexivity-research/novelty_audit_*.md` already produced the
analysis, we use its language verbatim or with light editing.

---

## 1. Continuous-time SV with feedback channels (Hopf bifurcation result)

### 1.1 Closest model-structure precedent — Halperin & Itkin (2025)

Igor Halperin and Andrey Itkin, *Marketron Through the Looking Glass*,
arXiv:2508.09863, August 2025. Companion SSRN: *Marketron Games*,
abstract_id=5107305.

The paper closest to ours on the *structural-model* axis. Continuous-time
SDE for log-price with a memory-variable channel encoding "money flows",
utility-based pricing of options for an incomplete market, three-regime
taxonomy ("Good / Bad / Ugly"). Drift contains an explicit feedback term;
the authors note that the model produces a rich variety of dynamic
scenarios. Halperin and Itkin are also the most likely reviewers of the
present paper.

| Ingredient | Halperin–Itkin (2025) | Patel (2026, this paper) |
| --- | --- | --- |
| Continuous-time SDE for the underlying | Yes | Yes |
| Memory channel in the drift | Yes (quasi-particle $y$, $\theta$) | Yes ($z = $ LP-filtered log-spot) |
| Stochastic-vol backbone (separate variance SDE) | No (SV is the squared diffusion of the log-spot SDE modulated by memory) | Yes (Heston mean-reverting variance) |
| Dealer-gamma drift from option open-interest grid | No (memory is a generic money-flow channel, not GPP-derived from OI) | Yes (Gârleanu–Pedersen–Poteshman 2009 demand-pressure mapping) |
| Formal Hopf bifurcation theorem | No (regime changes via shape of the potential $V(x,y)$, not eigenvalue crossings) | Yes (Theorem 1 — Liu's criterion / Routh–Hurwitz) |
| First Lyapunov coefficient $\ell_1$ | No | Yes (analytical when log-normal OI; numerical otherwise via Kuznetsov 2004 eq. 3.20) |
| Stochastic Hopf shift $\Lambda$ via Khasminskii | No | Yes (Khasminskii sphere process; Engel–Lamb–Rasmussen 2024 toolkit) |

We position our contribution as *the bifurcation analysis the Marketron
authors do not perform*. The structural similarity (continuous-time, memory
channel, options-aware) is high and the differentiation is mathematical
rather than philosophical. See `threats_to_validity.md` §1 for the framing
strategy.

### 1.2 Dealer-gamma axis precedent — Dai (2025)

Haoying Dai, *Beta-Dependent Gamma Feedback and Endogenous Volatility
Amplification in Option Markets*, arXiv:2511.22766, November 2025.

Closest precedent on the *dealer-gamma axis*. Explicit market-maker
hedging closes the loop via $\Delta H_t = G \Delta S_t \phi(x)$; GPP-style
demand pressure; beta-dependent normalisation; clean stability condition
$\lambda G_0 \phi(x_0) < 1$ characterising onset of "gamma-squeeze."

| Ingredient | Dai (2025) | Patel (2026) |
| --- | --- | --- |
| Continuous-time SDE | No (discrete-time recursion $\Delta S_{t+1} = \mu_t S_t + \mathcal{I}(\lambda_t N_t \Gamma_t \phi(x_t)) \Delta S_t$ with AR(1) option arrivals) | Yes |
| Stochastic-vol backbone | No (single price recursion; no separate variance state) | Yes |
| Dealer-gamma feedback in drift | Yes (GPP-style, beta-modulated) | Yes (GPP-style, OI-grid-aggregated) |
| Bifurcation type | Saddle-node / transcritical (single real eigenvalue crossing $F \to 1$) | Hopf (complex-conjugate pair crossing imaginary axis) |
| Output regime | Unbounded amplification on one side of threshold | Bounded limit cycle (supercritical $\ell_1 < 0$) |
| First Lyapunov coefficient | Not applicable (saddle-node, not Hopf) | Yes |
| Stochastic shift $\Lambda$ | No | Yes |

The two onsets are *structurally distinct*: Dai's instability is
unbounded amplification past a real-eigenvalue crossing, ours is a stable
limit cycle past a complex-pair crossing. Both are correct results; they
characterise different phenomena. Dai's recursion is also the static
instability surface our 2D upper-triangular reduction would produce —
which is precisely why we need the 3D extension with memory variable
$z_t$ to get the Hopf.

### 1.3 Continuous-time HAM bifurcation precedents — He–Li–Zheng (2009) and Chiarella–He–Hommes (2006)

He, Li & Zheng, *Market stability switches in a continuous-time financial
market with heterogeneous beliefs*, *Economic Modelling* 26 (2009),
1432–1442. Chiarella, He & Hommes, *Dynamics of moving average rules in a
continuous-time financial market model*, UTS QFRC RP 268 (2006), later
*JEDC*.

Closest published precedents for *continuous-time financial-market Hopf*.
Continuous-time delay-differential-equation models with fundamentalists
and trend-followers; Hopf bifurcation as the time delay (memory length of
the moving average rule) crosses a critical value; periodic solutions
analysed.

| Ingredient | He–Li–Zheng (2009) / Chiarella–He–Hommes (2006) | Patel (2026) |
| --- | --- | --- |
| Continuous-time dynamical system | Yes (DDE) | Yes (3D SDE) |
| Stochastic-vol backbone (separate variance SDE) | No (single price equation with delay) | Yes |
| Options market / dealer gamma | No (heterogeneous-belief weighting; no derivatives feedback) | Yes |
| Hopf via complex-pair eigenvalue crossing | Yes (delay-induced) | Yes (coupling-induced) |
| First Lyapunov coefficient via Kuznetsov's formula | No (qualitative bifurcation diagrams only) | Yes |
| Stochastic Lyapunov shift via Khasminskii | No (deterministic skeleton only) | Yes |

The *mathematical machinery* (Routh–Hurwitz / Liu's criterion / centre
manifold) is the same; the *physical mechanism* (dealer hedging from an
explicit options grid with a parallel Heston variance state) is different.
We cite as the methodological template for HAM-Hopf in continuous time.

### 1.4 Discrete-time-Hopf-with-derivatives precedent — Brock–Hommes–Wagener (2009)

Brock, Hommes & Wagener, *More hedging instruments may destabilize markets*,
*Journal of Economic Dynamics and Control* 33 (2009), 1912–1928.

Discrete-time Neimark–Sacker (Hopf-for-maps) bifurcation analysis in three
settings (mean-variance Arrow-securities, OLG, noisy REE). As bifurcation
parameter increases, steady state loses stability via Hopf, then becomes
quasi-periodic, then chaotic.

| Ingredient | Brock–Hommes–Wagener (2009) | Patel (2026) |
| --- | --- | --- |
| Continuous-time SDE | No (discrete-time map) | Yes |
| Stochastic-vol backbone | No (mean-variance utility maximisation) | Yes |
| Options / derivatives in model | Yes (Arrow securities) | Yes (full vanilla option grid) |
| Dealer-gamma destabilisation channel | No (heterogeneous-belief switching is the mechanism) | Yes |
| Bifurcation type | Neimark–Sacker (Hopf-for-maps) | Hopf (for flows) |
| First Lyapunov coefficient computed | No | Yes |
| Stochastic Hopf shift | No | Yes |

Different mechanism (belief switching vs. dealer hedging), different state
space (discrete map vs. continuous SDE), different bifurcation formalism
(Neimark–Sacker vs. Hopf). Cite as the methodological template for
"derivatives can destabilise"; the Arrow-securities abstraction is too
removed from a real options market to be confused with our model.

### 1.5 Hawkes-process critical-reflexivity precedent — Hardiman–Bercot–Bouchaud (2013)

Hardiman, Bercot & Bouchaud, *Critical reflexivity in financial markets: a
Hawkes process analysis*, arXiv:1302.1405; *European Physical Journal B*,
2013.

Empirical anchor for "markets sit near criticality": Hawkes branching
ratio $n \approx 1$ on E-mini mid-price changes over 1998–2011.

| Ingredient | Hardiman–Bercot–Bouchaud (2013) | Patel (2026) |
| --- | --- | --- |
| Continuous-time formalism | Yes (event-time point process) | Yes (calendar-time SDE) |
| Stochastic-vol backbone | No (Hawkes intensity, not Heston variance) | Yes |
| Dealer-gamma feedback | No | Yes |
| Hopf bifurcation analysis | No (criticality is branching-ratio = 1) | Yes |
| Empirical-criticality framing | Yes (the source of it) | Conjectural connection only (Theory §6) |

A *methodological cousin*, not a competitor. We conjecture that our
$\kappa^\star$ is the structural mechanism for HBB's empirical $n \approx
1$, but a formal reduction Hawkes-$n$ ↔ SV-Jacobian-eigenvalues does not
exist in the published literature and would itself be a separate paper
(Theory §6, Open Items §8 item 3).

### 1.6 Hedger-flow feedback precedents — Frey–Stremme (1997), Sircar–Papanicolaou (1998), Platen–Schweizer (1998), Schönbucher–Wilmott (2000)

Foundational hedger-flow-feedback literature. Modified local-volatility
PDEs derived from delta-hedger demand. Forerunners of GPP and of our $G$
functional.

| Ingredient | Frey–Stremme / Sircar–Papanicolaou / Platen–Schweizer | Patel (2026) |
| --- | --- | --- |
| Hedger-flow feedback in price dynamics | Yes (in spirit; aggregate hedger demand) | Yes (GPP demand-pressure) |
| Continuous-time SDE | Partial (local-vol PDE feedback; limited stochastic structure) | Yes |
| Separate Heston variance state | No | Yes |
| Bifurcation analysis | No | Yes |
| First Lyapunov coefficient / stochastic shift | No | Yes |

Cite as conceptual ancestors; no novelty conflict.

---

## 2. RL for derivatives + sim-to-sim robustness ($\kappa$-sensitivity curve)

### 2.1 Closest precedent — He, Sutter & Gonon (2025, NeurIPS)

Guangyi He, Tobias Sutter & Lukas Gonon, *Distributional Adversarial Attacks and
Training in Deep Hedging*, arXiv:2508.14757, NeurIPS 2025; OpenReview
forum vBtfIafffU.

Single most dangerous paper for our $\kappa$-sensitivity novelty. Deep-
hedging agents under a Wasserstein-ball adversarial-training framework;
Table 1 sweeps a Wasserstein-ball radius $\delta \in \{0, 0.01, 0.03,
0.05, 0.1, 0.3, 0.5\}$ with hedging-loss reported per grid point;
Theorem 3.3 mentions an asymptotic sensitivity expansion in $\delta$.

| Ingredient | He, Sutter & Gonon (2025, NeurIPS) | Patel (2026) |
| --- | --- | --- |
| RL agent (sequential decision) | Yes (deep hedging) | Yes (Mamba+PPO+EWC, ATLAS-vendored) |
| Finance / market-simulator domain | Yes (option hedging) | Yes (option hedging in reflexive simulator) |
| Parametric environment family with sweep | Yes (Wasserstein-ball radius $\delta$) | Yes (reflexive coupling $\kappa$) |
| Single agent deployed across the grid | No (separate robust agent trained at each $\delta$) | Yes (one $\pi_{\kappa_0}$ deployed across $\kappa \in [0, 2\kappa_0]$) |
| Slope-at-anchor as scalar summary statistic | No (table of per-point losses; Thm 3.3 expansion never operationalised) | Yes ($\tilde{\rho} = \partial \hat{m}/\partial \kappa\rvert_{\kappa_0}$) |
| Bootstrap CI on the slope | No | Yes (paired-episode block bootstrap on the spline derivative) |
| Coupling axis is physically meaningful | Partial (Wasserstein-ball radius is a perturbation budget, not a physical coupling) | Yes ($\kappa$ is dealer-gamma feedback strength with literature priors) |

The mandatory differentiation in our related-work treatment: (a) single
agent vs many, (b) slope-at-anchor scalar vs grid table, (c) physical
reflexivity coupling vs Wasserstein perturbation budget.

### 2.2 Wasserstein-ball robust-RL-with-impact precedent — Hou et al. (2025, NeurIPS poster)

*Robust Reinforcement Learning in Finance: Modeling Market Impact with
Elliptic Uncertainty Sets*, arXiv:2510.19950, NeurIPS 2025 poster (virtual
session 118821).

Market-impact RL with parameterised elliptic uncertainty set; closed-form
worst-case is the headline.

| Ingredient | Hou et al. (2025) | Patel (2026) |
| --- | --- | --- |
| RL agent | Yes | Yes |
| Finance domain | Yes (market impact / execution) | Yes (option hedging) |
| Parametric env family | Yes (elliptic ball over impact parameters) | Yes (reflexive coupling $\kappa$) |
| Worst-case minimax framework | Yes | No (smooth sensitivity curve, not adversarial worst-case) |
| Slope-at-anchor scalar with CIs | No | Yes |

Different *form* (worst-case vs sensitivity curve), similar *spirit* (both
quantify how policy cost grows with mis-specified environment). Cite as
the robust-RL-with-impact analogue.

### 2.3 Multi-risk-aversion deep-hedging precedent — Murray, Wood, Buehler et al. (2022)

Phillip Murray, Ben Wood, Hans Buehler et al., *Deep Hedging: Continuous
Reinforcement Learning Across Multiple Risk Aversions*, arXiv:2207.07467,
2022.

Actor-critic RL hedging with a continuous risk-aversion $\lambda$ axis the
agent is trained jointly across — closer to domain randomisation than a
$\kappa$-sweep.

| Ingredient | Murray et al. (2022) | Patel (2026) |
| --- | --- | --- |
| RL agent | Yes (actor-critic) | Yes |
| Finance domain | Yes | Yes |
| Parametric axis | Partial ($\lambda$ is a *reward-weighting* hyperparameter, not an environment coupling) | Yes ($\kappa$ is environment coupling) |
| Single anchor + sweep deployment | No (agent trained jointly across the family) | Yes |
| Slope-at-anchor scalar | No | Yes |

Cite as the multi-axis deep-hedging analogue. Distance from claim: large.

### 2.4 Subbaswamy–Adams–Saria (NeurIPS 2022) — closest theoretical precursor for slope-at-anchor

Adarsh Subbaswamy, Roy Adams & Suchi Saria, *Evaluating Robustness to
Dataset Shift via Parametric Robustness Sets*, NeurIPS 2022. Proceedings
PDF at proceedings.neurips.cc/paper_files/paper/2022/file/6b7f9d9c1217a748391800871ff7d17d-Paper-Conference.pdf.

The closest published object to a "slope-at-anchor" robustness summary in
the entire literature surveyed.

| Ingredient | Subbaswamy et al. (2022) | Patel (2026) |
| --- | --- | --- |
| Parametric shift in causal mechanism | Yes | Yes ($\kappa$ axis) |
| Local 2nd-order approximation to loss-under-shift | Yes (theoretical) | Yes (empirical, smoothed via spline) |
| Reinforcement learning | No (supervised) | Yes |
| Finance / market simulator | No | Yes |
| Bootstrap CI on the derivative | No (worst-case quadratic optimisation over the parametric ball) | Yes |
| Anchor is a calibrated physical parameter | No (arbitrary distribution shift) | Yes ($\kappa_0$ from sliced-W2 calibration) |

This is the right intellectual citation for the *concept* of using a
local sensitivity at an anchor as a robustness summary. The
$\kappa$-sensitivity curve is the RL-finance specialisation.

### 2.5 RL-generalisation precedent — Packer et al. (2018)

Charles Packer, Katelyn Gao, Jernej Kos, Philipp Krähenbühl, Vladlen
Koltun & Dawn Song, *Assessing Generalization in Deep Reinforcement
Learning*, arXiv:1810.12282, 2018.

Standard RL benchmark generalisation with parametric env families
(gravity, mass, friction).

| Ingredient | Packer et al. (2018) | Patel (2026) |
| --- | --- | --- |
| RL agent | Yes | Yes |
| Parametric env family (gravity, mass, friction) | Yes | Yes |
| Per-grid-point scoring | Yes | Reported as the curve |
| Slope-at-training-point scalar | No (per-point scores and geometric-mean extrapolation) | Yes |
| Finance domain | No | Yes |

Methodological ancestor — the slope-at-anchor extension of their
sweep design *is* the novelty in this lineage, not the sweep itself.

### 2.6 Foundational deep-hedging precedent — Buehler, Gonon, Teichmann & Wood (2019)

Hans Buehler, Lukas Gonon, Josef Teichmann & Ben Wood, *Deep Hedging*,
*Quantitative Finance* 19(8), 1271–1291; arXiv:1802.03042.

The seminal paper that brought neural-network policy training to derivatives
hedging. Static deep network, supervised on a chosen risk metric.

| Ingredient | Buehler et al. (2019) | Patel (2026) |
| --- | --- | --- |
| Sequential decision policy | Partial (static feed-forward; not full RL with credit assignment) | Yes |
| Finance / option hedging | Yes | Yes |
| Reflexive feedback simulator | No | Yes |
| Slope-at-anchor robustness scalar | No | Yes |

Cite as the foundation of the deep-hedging line.

### 2.7 Cao–Chen–Hull–Poulos (2021) — RL hedging across SV environments

Jay Cao, Jacky Chen, John Hull & Zissis Poulos, *Deep Hedging of
Derivatives Using Reinforcement Learning*, arXiv:2103.16409, 2021.

| Ingredient | Cao et al. (2021) | Patel (2026) |
| --- | --- | --- |
| RL agent | Yes | Yes |
| Finance domain | Yes (vanilla options under GBM and SV) | Yes |
| Robustness check | Partial ("robust to doubled transaction costs" — point comparison) | Yes (full sensitivity curve, slope-at-anchor) |
| Slope-at-anchor scalar | No | Yes |

Cite as the closest RL-hedging-with-robustness paper without the slope.

---

## 3. Market-generator evaluation metrics (sliced-W2 on IV-surface windows)

### 3.1 Closest precedent — Ning, Jaimungal, Zhang & Bergeron (2021, SIAM SIFIN 2024)

Brian Ning, Sebastian Jaimungal, Xiaorong Zhang & Maxime Bergeron,
*Arbitrage-Free Implied Volatility Surface Generation with Variational
Autoencoders*, arXiv:2108.04941, 2021; *SIAM Journal on Financial
Mathematics* 15(1), 2024, also DOI 10.1137/21M1443546.

Strongest precedent. Uses a Wasserstein metric between the multivariate
distribution of IVs at grid points on test data versus generator output as
a post-hoc evaluation metric, and reports impact of training-window length
on average Wasserstein.

| Ingredient | Ning et al. (2021/2024) | Patel (2026) |
| --- | --- | --- |
| Wasserstein distance between IV distributions | Yes ($W_1$) | Yes (sliced $W_2$) |
| IV surface as the state object | Yes | Yes |
| 21-day rolling-window path-distribution object | No (single-day surface marginals) | Yes (1617-dim windows) |
| Arbitrage filter on metric input | Partial (handled inside the model, not as a pre-filter on the metric) | Yes (hard filter; window dropped if any daily surface fails) |
| Post-hoc external evaluation (vs training loss) | Yes | Yes |
| Block-bootstrap CIs reported per regime | No | Yes |

We report the Ning-style $W_1$ on single-day marginals as a *baseline
secondary metric* alongside our SW2 path metric, which both gives credit
to the precedent and forces direct comparison.

### 3.2 VolGAN — Vuletić & Cont (2025)

Milena Vuletić & Rama Cont, *VolGAN: A Generative Model for Arbitrage-Free
Implied Volatility Surfaces*, *Applied Mathematical Finance* (2025),
DOI 10.1080/1350486X.2025.2471317; SSRN abstract_id=4617536; code at
github.com/milenavuletic/VolGAN.

WGAN-GP with Wasserstein as the *training loss* (Kantorovich–Rubinstein
dual via critic); evaluation by VIX dynamics, FPCA components, arbitrage
rejection rates.

| Ingredient | VolGAN (2025) | Patel (2026) |
| --- | --- | --- |
| Wasserstein in the protocol | Yes (training loss only) | Yes (post-hoc evaluation only) |
| IV surface object | Yes | Yes |
| Arbitrage handling | Yes (constraint inside model + rejection rate diagnostic) | Yes (hard pre-filter on metric input) |
| 21-day rolling-window object | No (single-day surfaces) | Yes |
| Post-hoc external evaluation by Wasserstein | No | Yes |
| Block-bootstrap CIs per regime | No | Yes |

The training-loss vs external-evaluation distinction is the load-bearing
difference. Pre-empt by stating it explicitly in the related-work
paragraph.

### 3.3 FuNVol — Choudhary, Jaimungal & Bergeron (2024)

Vedant Choudhary, Sebastian Jaimungal & Maxime Bergeron, *FuNVol:
Multi-Asset Implied Volatility Market Simulator using Functional Principal
Components and Neural SDEs*, arXiv:2303.00859, *Quantitative Finance*
(2024), DOI 10.1080/14697688.2024.2396977.

Multi-asset IV simulator with functional PCA + neural SDE; arbitrage-aware.

| Ingredient | FuNVol (2024) | Patel (2026) |
| --- | --- | --- |
| IV surface object | Yes | Yes |
| Arbitrage handling | Yes | Yes |
| Wasserstein evaluation metric | No (FPC matching + delta-hedging P&L distribution) | Yes |
| 21-day path-window object | No | Yes |
| Block-bootstrap CIs per regime | No | Yes |

Cite as multi-asset IV simulator baseline.

### 3.4 Signature-kernel scoring rule — Issa, Horvath, Lemercier & Salvi (NeurIPS 2023)

Zacharia Issa, Blanka Horvath, Maud Lemercier & Cristopher Salvi,
*Non-adversarial Training of Neural SDEs with Signature Kernel Scores*,
arXiv:2305.16274, NeurIPS 2023.

| Ingredient | Issa et al. (2023) | Patel (2026) |
| --- | --- | --- |
| Path-space distance (strictly proper scoring rule) | Yes (signature-MMD) | Yes (sliced-W2 over windows) |
| Post-hoc-eligible | Yes | Yes |
| Applied to IV surfaces | No (price paths / rough volatility) | Yes |
| Wasserstein metric | No (MMD with signature kernel) | Yes |
| 21-day surface windows | No | Yes |

We additionally report Sig-MMD on the same surface windows as a secondary
evaluation metric, which directly inherits this machinery and forces the
comparison.

### 3.5 Diffusion-based IV-surface forecasting — Jin & Agarwal (2025)

Chen Jin & Ankush Agarwal, *Forecasting Implied Volatility Surface with
Generative Diffusion Models*, arXiv:2511.07571, November 2025.

Most temporally adjacent competitor. Verified via WebFetch of the HTML
that evaluation uses MAPE, calibration of 90% CIs, time-series slice
analysis, distributional moment analysis, and arbitrage penalty in
training; explicitly does *not* use Wasserstein, sliced-Wasserstein, MMD,
signature kernel, or any optimal-transport metric on the surface
distribution.

| Ingredient | Jin & Agarwal (2025) | Patel (2026) |
| --- | --- | --- |
| IV surface generative model | Yes (DDPM) | Comparator (we evaluate generators, not propose one) |
| Wasserstein / sliced-W / MMD evaluation | No | Yes (sliced-W2) |
| 21-day path-window object | No | Yes |
| Post-hoc external benchmark across generators | Partial (benchmarks against VolGAN) | Yes |
| Block-bootstrap CIs per regime | No | Yes |

### 3.6 OT-for-arbitrage-correction — Chevallier, De Marco & Lévy-dit-Vehel (2025)

M. Chevallier, S. De Marco & P.-E. Lévy-dit-Vehel, *An Optimal Transport
Approach to Arbitrage Correction: Application to Volatility Stress-Tests*,
arXiv:2501.12195, January 2025.

| Ingredient | Chevallier et al. (2025) | Patel (2026) |
| --- | --- | --- |
| Wasserstein on signed measures over option-price space | Yes | Yes (on IV-surface vectors) |
| Role of OT | Arbitrage-correction operator (projection onto martingale measures) | Generator-evaluation metric |
| 21-day rolling-window object | No | Yes |
| Generator comparison | No | Yes |
| Block-bootstrap CIs | No | Yes |

Same building block (OT on option-price space), entirely different role.

### 3.7 Sliced-W in finance — Luan & Hamp (2025)

Qinmeng Luan & James Hamp, *Automated regime classification in
multidimensional time series data using sliced Wasserstein k-means
clustering*, *Data Science in Finance and Economics*, 2025,
DOI 10.3934/DSFE.2025016.

| Ingredient | Luan & Hamp (2025) | Patel (2026) |
| --- | --- | --- |
| Sliced-Wasserstein in finance | Yes (FX time series) | Yes (IV-surface windows) |
| IV surface object | No | Yes |
| Generator-evaluation role | No (k-means clustering of regimes) | Yes |
| 21-day path-window object | No | Yes |
| Block-bootstrap CIs per regime | No | Yes |

The metric SW2 has appeared in finance for clustering, not as an external
evaluation rubric for IV-surface generators.

### 3.8 Methodological building blocks (cited, no novelty conflict)

| Source | Role |
| --- | --- |
| Bonneel, Rabin, Peyré & Pfister (2015), *J. Math. Imaging Vis.* | Sliced-Wasserstein definition and Monte-Carlo estimator |
| Manole, Balakrishnan & Wasserman (arXiv:1909.07862, *EJS* 2022) | Minimax CIs for sliced-Wasserstein; $n^{-1/2}$ convergence in any dimension |
| Politis & Romano (1994), *JASA* | Stationary block bootstrap |
| Roper (2010), U. Sydney preprint; Gatheral & Jacquier (2014), arXiv:1204.0646 | Static-arbitrage filter on IV surfaces |
| Hosseini, Hsu & Taghvaei (arXiv:2311.05672, 2023) | Conditional optimal transport on function spaces; state-conditional comparisons |

---

## 4. Pre-registration in computational finance

### 4.1 Pacific-Basin Finance Journal pre-registration pathway (Faff 2022; PBFJ 2025 update)

Robert Faff, *PBFJ Editorial … Engaging with responsible science. "OPEN
FOR BUSINESS" — Launching the PBFJ pre-registration publication
initiative*, *Pacific-Basin Finance Journal* (2022), DOI
10.1016/j.pacfin.2022.101859; 2025 update DOI 10.1016/j.pacfin.2025.102697.

First finance journal with a formal pre-registration pathway. Four-phase
process. Editor commits to publication regardless of result direction.

| Ingredient | PBFJ pre-reg (2022–) | Patel (2026) |
| --- | --- | --- |
| Pre-registration in finance | Yes (institutional) | Yes (project-level) |
| Aimed at empirical / behavioural finance | Yes | No (computational / RL) |
| Aimed at RL-for-finance | No | Yes |
| Git commit-hash anchoring | No (methods locked via narrative) | Yes |
| Block-bootstrap + BH-FDR statistical pre-spec | No | Yes |
| OpenTimestamps Bitcoin-anchored proof | No | Yes |

We do *not* claim "first pre-registration in finance" — PBFJ owns that.
The contribution is the synthesis applied to a different subfield with
different verification artifacts.

### 4.2 AEA RCT Registry (2018–)

American Economic Association Randomized Controlled Trial Registry,
aeaweb.org/journals/policies/rct-registry. Mandatory for AEA-journal field
experiments since 2018; hosts trading and market-microstructure RCTs.

| Ingredient | AEA RCT Registry (2018–) | Patel (2026) |
| --- | --- | --- |
| Pre-registration of analysis plan | Yes | Yes |
| Field-experiment scope | Yes | No (observational data + simulator-based RL) |
| Computational / RL methodology | No | Yes |
| Code anchoring required | No | Yes (commit hash + OTS) |
| Block-bootstrap + BH-FDR pre-spec | No (analysis plan format only) | Yes |

Cite as the canonical finance pre-registration registry; our use case is
distinct.

### 4.3 Pérignon, Akmansoy, Hurlin, Dreber, Holzmeister, Huber, Johannesson, Kirchler, Menkveld, Razen & Weitzel (2024, RFS) — Computational reproducibility in finance

Christophe Pérignon et al., *Computational Reproducibility in Finance:
Evidence from 1,000 Tests*, *Review of Financial Studies* 37(11), 3558,
2024; SSRN abstract_id=4064172.

Found 52% reproducibility rate. Recommends reproducibility policies.

| Ingredient | Pérignon et al. (2024) | Patel (2026) |
| --- | --- | --- |
| Diagnoses reproducibility crisis in finance | Yes | Cited |
| Recommends code-sharing policies | Yes | Yes (we comply) |
| Mandates / models commit-hash pre-registration | No | Yes |
| RL-for-finance-specific | No | Yes |

We position our pre-registration discipline as a *response* to the
documented crisis rather than a discovery — per `novelty_audit_prereg.md`
§5 framing.

### 4.4 Camerer et al. (2018, Nature Human Behaviour) — pre-registration template

Colin Camerer, Anna Dreber, Felix Holzmeister et al., *Evaluating the
replicability of social science experiments in Nature and Science between
2010 and 2015*, *Nature Human Behaviour* 2, 637–644, 2018.

Replication study of 21 social-science experiments using pre-registered
analysis plans approved by original authors.

| Ingredient | Camerer et al. (2018) | Patel (2026) |
| --- | --- | --- |
| Pre-registered analysis plan | Yes | Yes |
| Domain | Social science (lab experiments) | Computational finance / RL |
| Commit-hash code anchor | No | Yes |
| Block-bootstrap + BH-FDR | No (per-experiment $p$-values) | Yes |

Acknowledged "gold standard" template we adapt from.

### 4.5 NeurIPS Paper Checklist & 2024 Checklist Assistant pilot

NeurIPS Paper Checklist (mandatory submission-time, since 2021); 2024
Checklist Assistant pilot (LLM-assisted enrolment).

| Ingredient | NeurIPS Checklist | Patel (2026) |
| --- | --- | --- |
| Reproducibility / ethics / code-release checklist | Yes (mandatory) | Yes (we satisfy items 1-25) |
| Hypothesis pre-registration | No (post-hoc checklist, not prospective lock) | Yes |
| Commit-hash anchor | Recommended (PaperswithCode 2020) | Yes (mandatory in our protocol) |
| Block-bootstrap statistical pre-spec | No | Yes |

Cite as the closest ML reproducibility precedent; the pre-registration
gap is what we fill.

### 4.6 Open RL Benchmark — Huang et al. (2024)

Shengyi Huang et al., *Open RL Benchmark: Comprehensive Tracked
Experiments for Reinforcement Learning*, arXiv:2402.03046, 2024;
github.com/openrlbenchmark/openrlbenchmark.

| Ingredient | Open RL Benchmark | Patel (2026) |
| --- | --- | --- |
| Library / dependency-version pinning | Yes (every tracked run reproducible) | Yes (uv.lock + commit hash) |
| Hypothesis pre-spec | No (records runs, doesn't pre-spec hypotheses) | Yes |
| Finance domain | No | Yes |

Cite as the version-pinning precedent; the pre-registration extension is
what we add.

### 4.7 RL-for-finance systematic reviews flagging the gap

| Source | Identifies pre-registration gap? | Proposes commit-hash anchoring? |
| --- | --- | --- |
| Bai, Gao, Wan, Zhang & Song, *A Review of Reinforcement Learning in Financial Applications*, arXiv:2411.12746, 2024 | Yes (reproducibility flagged) | No |
| *Reinforcement Learning in Financial Decision Making: A Systematic Review*, arXiv:2512.10913, December 2025 | Yes (fragmented metrics, limited reproducibility) | No |
| Hambly, Xu & Yang (2023), *Mathematical Finance*, "Recent Advances in RL in Finance" | Yes (calls for benchmarking) | No |

The fact that the field's *own* recent self-diagnoses flag the
reproducibility gap without proposing the commit-hash + statistical pre-
spec response is itself supportive evidence that our specific synthesis
has not been adopted in this subfield.

### 4.8 Honest framing per novelty_audit_prereg.md §5

We frame the contribution as the *first synthesis of [pre-registration +
git commit-hash anchoring + block-bootstrap/BH-FDR statistical pre-spec +
applied to RL-for-finance]*, not as "first pre-registration in finance"
(false; PBFJ owns that since 2022) and not as "first commit-hash in ML
reproducibility" (false; PaperswithCode 2020 owns that). The novelty is
the specific four-way intersection in this subfield.

---

## URLs (consolidated)

**Continuous-time SV with feedback:**
- Halperin & Itkin (2025): https://arxiv.org/abs/2508.09863 ; companion SSRN 5107305
- Dai (2025): https://arxiv.org/abs/2511.22766
- He, Li & Zheng (2009): https://www.sciencedirect.com/science/article/abs/pii/S026499930900128X
- Chiarella, He & Hommes (2006): https://www.uts.edu.au/globalassets/sites/default/files/qfr-archive-03/QFR-rp268.pdf
- Brock, Hommes & Wagener (2009): https://wrap.warwick.ac.uk/id/eprint/1757/
- Hardiman, Bercot & Bouchaud (2013): https://arxiv.org/abs/1302.1405
- Frey & Stremme (1997): https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-9965.00036
- Sircar & Papanicolaou (1998): http://math.stanford.edu/~papanico/pubftp/feedback.pdf
- Platen & Schweizer (1998): https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-9965.00038
- Gârleanu, Pedersen & Poteshman (2009): https://academic.oup.com/rfs/article-abstract/22/10/4259/1590158
- Heston (1993): https://academic.oup.com/rfs/article-abstract/6/2/327/1574747
- Engel, Lamb & Rasmussen (2024): https://link.springer.com/article/10.1007/s00440-024-01301-4

**RL for derivatives + sim-to-sim robustness:**
- He, Sutter & Gonon (2025): https://arxiv.org/abs/2508.14757 ; OpenReview vBtfIafffU
- Hou et al. (2025) elliptic uncertainty: https://arxiv.org/abs/2510.19950
- Murray, Wood, Buehler et al. (2022): https://arxiv.org/abs/2207.07467
- Subbaswamy, Adams & Saria (2022): https://proceedings.neurips.cc/paper_files/paper/2022/file/6b7f9d9c1217a748391800871ff7d17d-Paper-Conference.pdf
- Packer et al. (2018): https://arxiv.org/abs/1810.12282
- Buehler, Gonon, Teichmann & Wood (2019): https://arxiv.org/abs/1802.03042
- Cao, Chen, Hull & Poulos (2021): https://arxiv.org/abs/2103.16409

**Market-generator evaluation:**
- Ning, Jaimungal, Zhang & Bergeron (2021/2024): https://arxiv.org/abs/2108.04941 ; SIAM 10.1137/21M1443546
- VolGAN (Vuletić & Cont 2025): https://www.tandfonline.com/doi/full/10.1080/1350486X.2025.2471317
- FuNVol (Choudhary, Jaimungal & Bergeron 2024): https://arxiv.org/abs/2303.00859
- Issa, Horvath, Lemercier & Salvi (2023): https://arxiv.org/abs/2305.16274
- Jin & Agarwal (2025): https://arxiv.org/abs/2511.07571
- Chevallier, De Marco & Lévy-dit-Vehel (2025): https://arxiv.org/abs/2501.12195
- Luan & Hamp (2025): http://www.aimspress.com/article/doi/10.3934/DSFE.2025016
- Manole, Balakrishnan & Wasserman (2022): https://arxiv.org/abs/1909.07862

**Pre-registration:**
- Faff (PBFJ 2022): https://www.sciencedirect.com/science/article/abs/pii/S0927538X22001329
- AEA RCT Registry: https://www.aeaweb.org/journals/policies/rct-registry
- Pérignon et al. (RFS 2024): https://academic.oup.com/rfs/article-abstract/37/11/3558/7697104
- Camerer et al. (2018): https://www.nature.com/articles/s41562-018-0399-z
- NeurIPS Paper Checklist: https://neurips.cc/public/guides/PaperChecklist
- Open RL Benchmark (Huang et al. 2024): https://arxiv.org/abs/2402.03046
- Bai et al. (2024) RL-in-finance review: https://arxiv.org/abs/2411.12746
- RL-in-finance systematic review (December 2025): https://arxiv.org/abs/2512.10913
