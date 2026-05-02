# Threats to Validity

This document enumerates the most likely review-cycle attacks against the
contributions of *Reflexivity in Options Markets* (Patel 2026) and the
defensive framing we adopt for each. The taxonomy follows the four
adversarial novelty audits at
`~/Documents/reflexivity-research/novelty_audit_*.md`.

---

## 1. Most dangerous review attack — Halperin & Itkin (2025) Marketron

Igor Halperin and Andrey Itkin published *Marketron Through the Looking
Glass* (arXiv:2508.09863, August 2025) and the companion *Marketron
Games* (SSRN 5107305). Their continuous-time SDE for log-price with a
memory-variable channel is the structurally closest published model.
Halperin and Itkin are also potential reviewers given prior correspondence
(see `~/Documents/reflexivity-research/`).

**Threat.** A reviewer says: "this is just the bifurcation calculation
Halperin–Itkin didn't do."

**Defence.** That is the correct framing of our contribution and we adopt
it explicitly. We position our paper as *the bifurcation analysis the
Marketron authors deferred*, with the additional ingredients (Heston
variance backbone, GPP-style dealer-gamma drift from the option open-
interest grid, first Lyapunov coefficient $\ell_1$, Khasminskii stochastic
shift $\Lambda$) clearly laid out in the comparison table at
`related_work.md` §1.1. Pre-submission action: send a draft to
Halperin/Itkin so the framing is not a surprise. A friendly preface in
related work and a cleanly-cited extension is the cheapest insurance
against an adversarial Marketron-side review.

---

## 2. He, Li & Zheng (2025, NeurIPS) — adversarial deep-hedging

He, Li & Zheng's *Distributional Adversarial Attacks and Training in Deep
Hedging* (arXiv:2508.14757) sweeps a Wasserstein-ball radius
$\delta \in \{0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5\}$ in their Table 1; the
visual similarity to a $\kappa$-sweep is high.

**Threat.** A reviewer says: "He et al. 2025 already does this."

**Defence.** They do not. Three load-bearing differences, enumerated in
`related_work.md` §2.1 and recapitulated for clarity:

| Axis | He, Li & Zheng (2025) | Patel (2026) |
| --- | --- | --- |
| Number of agents | One per grid point ($\delta$); separate adversarial training at each | One agent ($\pi_{\kappa_0}$) deployed across $\kappa \in [0, 2\kappa_0]$ |
| Headline summary | Table of per-point hedging losses | Slope-at-anchor $\tilde{\rho} = \partial \hat{m}/\partial \kappa\rvert_{\kappa_0}$ with bootstrap CI |
| Coupling axis interpretation | Wasserstein-ball perturbation budget (purely statistical) | Reflexive coupling with literature priors and physical units (per USD-of-dollar-gamma) |

Theorem 3.3 of He et al. *mentions* an asymptotic sensitivity expansion in
$\delta$ but never operationalises it as a reported empirical scalar with
CIs. This is the critical gap our $\kappa$-sensitivity protocol fills.

We cite He et al. as the closest precedent at the top of the related-work
paragraph for $\kappa$-sensitivity (§2 of `related_work.md`) and include
the differentiation table in the body of the paper.

---

## 3. Ning, Jaimungal, Zhang & Bergeron (2021, 2024) — Wasserstein on IV surfaces

Ning et al.'s *Arbitrage-Free Implied Volatility Surface Generation with
Variational Autoencoders* (arXiv:2108.04941; SIAM SIFIN 2024,
DOI 10.1137/21M1443546) uses a $W_1$ metric between the multivariate IV
distribution on test data and generator output as a post-hoc evaluation.

**Threat.** A reviewer says: "Ning et al. already does $W$ on IV
surfaces."

**Defence.** True at 2/6 ingredients per the `novelty_audit_w2_surfaces.md`
analysis. The combinatorial differences are:

| Axis | Ning et al. (2021/2024) | Patel (2026) |
| --- | --- | --- |
| Distance | $W_1$ | Sliced-$W_2$ |
| Object | Single-day surface marginals | 21-business-day rolling-window vectors (1323-dim path-distribution objects) |
| Arbitrage handling | Constraint inside the model | Hard pre-filter on metric input (window dropped if any daily surface fails butterfly / calendar / Lee-bound checks) |
| CI structure | Single point estimate | Stationary block-bootstrap (Politis–Romano 1994) per regime, $B = 1000$ resamples, mean block length 21 days |

We additionally report Ning's $W_1$ on single-day marginals as a *baseline
secondary metric* alongside our SW2 path metric. This both gives credit to
the precedent and forces direct empirical comparison.

---

## 4. Pre-registration framing — drop "first in finance"

The Pacific-Basin Finance Journal has had a formal pre-registration
pathway since Brailsford et al.'s 2022 editorial (DOI
10.1016/j.pacfin.2022.101859), with a 2025 update. The AEA RCT Registry
has hosted finance / market-microstructure RCTs since 2018.

**Threat.** A reviewer says: "PBFJ has been doing this for three years."

**Defence.** We never claim "first pre-registration in finance" (it would
be false). Per `novelty_audit_prereg.md` §5, the conservative claim is:

> Pre-registration is established in experimental and behavioural finance
> (PBFJ since 2022; AEA RCT Registry since 2018) and reproducibility-by-
> version-control is established in machine-learning workflows
> (PaperswithCode 2020 code-release recommendations; Open RL Benchmark
> 2024). To our knowledge, the present work is the first to combine
> these into a single discipline — analysis plan locking H1–H4, primary
> and secondary metrics, ablations, and statistical procedure (block-
> bootstrap CIs, BH-FDR control) committed to a public git repository at
> a verifiable commit hash, with an OpenTimestamps Bitcoin-anchored proof
> of the document hash, *prior to any contact with the empirical
> evaluation data*, applied to a reinforcement-learning question in
> derivatives markets.

The novelty hook is the *combination* (commit-hash + RL-for-finance +
locked statistical procedure), not any single element. We frame the
discipline as a *response* to the documented reproducibility crisis
(Pérignon et al. 2024 RFS) rather than a discovery.

---

## 5. Industry / unindexed work and OSF preregistrations

The four novelty audits cover indexed venues (arXiv, journals, conference
proceedings, SSRN, OpenReview, ResearchGate). They do not exhaustively
cover:

- Industry whitepapers (JPM AI Research, Goldman QIS, Two Sigma, Citadel,
  Renaissance Technologies, Man Group quant-research) where partial
  precedents may exist behind paywalls.
- PhD / Master's theses (Imperial, Oxford, NYU, CMU) on market generators
  that may contain overlapping protocols.
- OSF Registries preregistrations (https://osf.io/registries/discover)
  which are not well-indexed by Google Scholar.

**Defence.** Per `novelty_audit_w2_surfaces.md` §4 and `novelty_audit_prereg.md`
§4, we phrase claims as **"we are unaware of any *published* work that…"**
rather than "no work exists." Pre-submission action: run an explicit OSF
Registries search for "reinforcement learning" / "deep hedging" / "trading
strategy" before final submission and document the search date and zero-
result count in a footnote. Subscribe to SSRN q-fin alerts on "dealer
gamma" and "endogenous volatility" through to the submission date so a
late-breaking paper is caught before it surprises a reviewer.

---

## 6. Other less-likely but worth-noting threats

### 6.1 Egebjerg & Kokholm (2024 SSRN), 0DTE empirical literature

Egebjerg & Kokholm SSRN 4936978; Adams, Dim, Eraker, Fontaine,
Ornthanalai & Vilkov SSRN 5641974 (2025); O'Donovan, Yu & Zhang SSRN
4567604 (2024). All empirical work on dealer-hedging effects in SPX /
0DTE; complementary to our theoretical contribution. Cite for
empirical motivation; no novelty conflict.

### 6.2 Bergault et al. (2025) — option market making with hedging-induced impact

Bergault et al., arXiv:2511.02518, November 2025. Continuous-time control
problem; abstract does not mention bifurcation but full PDF should be
re-skimmed before submission. Pre-submission action: confirm no Hopf
appendix.

### 6.3 SABR-MTGP and recent (Q1 2026) preprints

Audit window is April 2026; Q1 2026 preprints are partially indexed.
Pre-submission action: re-run a final ego search across q-fin.CP /
q-fin.MF / q-fin.ST / stat.ML arXiv listings for January–April 2026
immediately before submission.

### 6.4 Sample-complexity concern — $W_2$ in 1323 dimensions

Classical $W_2$ has $n^{-1/d}$ convergence rate. Our reliance on the
sliced-$W_2$ curse-of-dimension escape ($n^{-1/2}$ in any dimension per
Manole, Balakrishnan & Wasserman 2022) is honest but reviewers may push
on whether 21-day windows over a few thousand SPX trading days give
enough effective sample. Pre-submission action: report a power simulation
against a known-different generator (e.g. pure GBM with constant vol)
showing the protocol detects the difference at the configured sample
budget.

---

## 7. Summary

The four most dangerous attack vectors are explicitly addressed by:
(i) framing relative to Marketron as "the bifurcation analysis they
deferred" with a clean comparison table; (ii) leading the
$\kappa$-sensitivity related-work paragraph with He et al. (2025) and
enumerating the three differentiations; (iii) reporting Ning's $W_1$ as a
baseline secondary metric alongside our SW2; (iv) dropping "first in
finance" framing for pre-registration in favour of the four-way
intersection synthesis. Combined, these defences leave reviewers a clean
question to answer ("is the synthesis novel and useful?") rather than a
muddied one ("is any single ingredient new?").
