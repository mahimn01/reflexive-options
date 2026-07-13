# Submission readiness — v0.4.1

This checklist applies to the centered-model paper. It does not certify the
economic truth of the reduced-form mechanism; it records what has and has not
been established before submission.

## Theory

- [x] Physical measure and local detrended state stated.
- [x] Equilibrium $(0,\theta_v,0)$ is fixed for every $\kappa$.
- [x] Variance feedback is $\gamma v\chi$ and points inward at $v=0$.
- [x] The title and abstract describe $g$ as gamma-shaped reduced-form pressure rather than
  derived hedge flow.
- [x] The canonical $G_v$ dependence, positive price-to-variance sign, and coordinate
  dependence of the drift skeleton are disclosed.
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
- [x] No registered market dataset extracted or analyzed as of A16.
- [x] Registered historical horizon fixed: 2017-01-03--2024-10-29; it is not described as
  the eventual full sample.
- [x] Four primary summaries are distinct and observable.
- [x] Contract universe, spot/return/VIX sources, forwards, rates, OI timing,
  controls, and inference are fixed through A16.
- [x] Outcome is strictly after regressors; all four primary tests are two-sided.
- [x] HAC and bootstrap families receive separate BH adjustment and a locked
  conjunctive evidence label.
- [x] Event windows and convention-signed GEX are secondary.
- [x] Decision language prevents association becoming causal/dealer-sign confirmation.
- [x] Day-one plan has no expected sign or magnitude gate.
- [x] A15 preserves the complete CRSP session calendar, forbids missing-option-day time
  compression, and defines weekday controls on the outcome session.
- [x] A16 fixes OI availability timing, fractional settlement time, liquidity filters,
  22-session complete-calendar HAC/bootstrap, persistence controls, and pricing-input
  decompositions.
- [x] Preserve the A13-only snapshot `pre_registration_amendments.md.a13`, hash
  `83950ede…58e17`, and proof `pre_registration_amendments.md.a13.ots`.
- [x] Stamp A13--A14; SHA-256 `603e8936…f9cd7`, proof
  `pre_registration_amendments.md.ots` (pending Bitcoin consolidation).
- [x] Stamp the separate retrospective pre-extraction A15 correction; SHA-256
  `a5f694f9…34a52`, proof `pre_registration_amendment_a15.md.ots` (pending Bitcoin
  consolidation).
- [x] Stamp A16; SHA-256 `a5cbf9ef…3e61d`, proof
  `pre_registration_amendment_a16.md.ots` (pending Bitcoin consolidation).
- [ ] Before outcome extraction, complete the fail-closed WRDS adapter, timing gate,
  attrition ledger, and raw option manifest required by A16.
- [ ] On access day, freeze/hash options before constructing outcomes.

## Artifacts

- [x] Master and both variants compile.
- [x] Master references only the current validation and robustness figures.
- [x] Metadata, README, citation, version, and changelog updated.
- [x] Sole-author metadata consistently identifies Mahimn Patel and asserts no
  institutional affiliation.
- [x] Legacy invalid modules/documents are marked archived or withdrawn.
- [x] Full verification passes: 618 tests, 88.43% coverage; strict mypy and ruff clean.
- [x] Render and inspect all 24 master pages; final LaTeX log has no warning, undefined
  reference, underfull box, or overfull box.
- [x] arXiv tarball extraction/build test passes at 24 pages; SHA-256
  `81141e285320f9d011f702d4265e29280bb391b087cd17a72bd7f733cda9655e`.
- [x] Tar members are exactly `main.tex`, `main.bbl`, `references.bib`,
  `figures/centered_hopf_validation.pdf`, and
  `figures/centered_hopf_robustness.pdf`.

## Remaining scientific risk

Reviewers can reasonably reject the reduced-form pressure map, the positive
price-to-variance memory channel, or the relevance of the illustrative frozen-book regime.
The paper is a local dynamical example and pre-extraction protocol because those points
remain open. The ablation and mixture analyses make those dependencies explicit.
The ICAIF variant is a layout draft that still requires the latest ACM template and has weak
venue fit without a genuine AI/ML contribution. The NeurIPS workshop file remains a
layout placeholder until a specific 2026 workshop publishes its own call.
