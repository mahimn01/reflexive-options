# Submission readiness — v0.4.0

This checklist applies to the centered-model paper. It does not certify the
economic truth of the reduced-form mechanism; it records what has and has not
been established before submission.

## Theory

- [x] Physical measure and local detrended state stated.
- [x] Equilibrium $(0,\theta_v,0)$ is fixed for every $\kappa$.
- [x] Variance feedback is $\gamma v\chi$ and points inward at $v=0$.
- [x] Normalized theoretical coupling is not compared with per-dollar GEX.
- [x] Cubic coefficients are checked against the direct characteristic polynomial.
- [x] Every positive root passes all Routh--Hurwitz, frequency, and transversality checks;
  the remote second, re-stabilizing crossing is disclosed rather than hidden by a first-root API.
- [x] First-Lyapunov tensors include the nonlinear variance term.
- [x] The figure integrates the actual Gaussian book and keeps variance positive.
- [x] Local amplitude and period approach the Hopf normal form; tolerances and initial
  conditions converge.
- [x] Gross-normalized signed mixtures disclose that supercriticality is book-dependent.
- [x] Gaussian convolution and every derivative through order three are shown and checked
  independently by finite differences.
- [x] Global-stability, hysteresis, stochastic-shift, Hawkes, mean-field,
  information, and stationary-density overclaims are removed.

## Measurement and empirical design

- [x] Signed theoretical dealer book is separated from public OI.
- [x] No registered market dataset extracted or analysed as of A15.
- [x] Registered historical horizon fixed: 2017-01-03--2024-10-29; it is not described as
  the eventual full sample.
- [x] Four primary summaries are distinct and observable.
- [x] Contract universe, spot/return/VIX sources, forwards, rates, OI timing,
  controls, and inference are fixed.
- [x] Outcome is strictly after regressors; all four primary tests are two-sided.
- [x] HAC and bootstrap families receive separate BH adjustment and a locked
  conjunctive evidence label.
- [x] Event windows and convention-signed GEX are secondary.
- [x] Decision language prevents association becoming causal/dealer-sign confirmation.
- [x] Day-one plan has no expected sign or magnitude gate.
- [x] A15 preserves the complete CRSP session calendar, forbids missing-option-day time
  compression, and defines weekday controls on the outcome session.
- [x] Preserve the A13-only snapshot `pre_registration_amendments.md.a13`, hash
  `83950ede…58e17`, and proof `pre_registration_amendments.md.a13.ots`.
- [x] Stamp A13--A14; SHA-256 `603e8936…f9cd7`, proof
  `pre_registration_amendments.md.ots` (pending Bitcoin consolidation).
- [x] Stamp the separate retrospective pre-extraction A15 correction; SHA-256
  `a5f694f9…34a52`, proof `pre_registration_amendment_a15.md.ots` (pending Bitcoin
  consolidation).
- [ ] On access day, freeze/hash options before constructing outcomes.

## Artifacts

- [x] Master and both variants compile.
- [x] Master references only the current validation and robustness figures.
- [x] Metadata, README, citation, version, and changelog updated.
- [x] Sole-author metadata consistently identifies Mahimn Patel and asserts no
  institutional affiliation.
- [x] Legacy invalid modules/documents are marked archived or withdrawn.
- [x] Full verification passes: 607 tests, 88.48% coverage; strict mypy and ruff clean.
  The 80 tests touching the final warning fixes also pass with warnings promoted to errors.
- [x] Render and inspect all 22 master pages; final LaTeX log has no warning, undefined
  reference, underfull box, or overfull box.
- [x] arXiv tarball extraction/build test passes at 22 pages; SHA-256
  `fa4813d33568e5059022f75f486286056e7eedc8ebb8df562bb6d2b71ac27264`.
- [x] Tar members are exactly `main.tex`, `main.bbl`, `references.bib`,
  `figures/centered_hopf_validation.pdf`, and
  `figures/centered_hopf_robustness.pdf`.

## Remaining scientific risk

Reviewers can reasonably reject the economic microfoundation, memory channel, or relevance
of the illustrative regime. The paper is a local possibility result and pre-extraction protocol
because those points remain open. The mixture audit now makes the single-Gaussian risk
quantitative rather than leaving it implicit.
The ICAIF variant meets mechanical 2026 format/length rules but has weak venue
fit without a genuine AI/ML contribution. The NeurIPS workshop file remains a
layout placeholder until a specific 2026 workshop publishes its own call.
