# arXiv submission checklist — v0.4.0

This is the operating procedure for the centered-model paper. The upload file is
`dist/arxiv_v0.4.0.tar.gz`; metadata is in `paper/arxiv_metadata.txt`.

## Local pre-flight

- [x] `bash scripts/arxiv_build.sh` exits with `arxiv_build.sh DONE`.
- [x] Tarball SHA-256 is
  `fb3b753d49c8a371e2180a13b871206fabe724eab9db90d5ee64a9e2014b6127`.
- [x] Simulated clean extraction compiles to 22 pages, within 13 bytes of the master PDF.
- [x] Tar members are exactly:
  - `main.tex`
  - `main.bbl`
  - `references.bib`
  - `figures/centered_hopf_validation.pdf`
  - `figures/centered_hopf_robustness.pdf`
- [x] Master PDF has 22 pages, one theorem, two figures, six tables, and no unresolved
  citation/reference, underfull box, overfull box, or final-pass LaTeX warning.
- [x] Full verification passes: 607 tests, 88.48% coverage, strict mypy, and ruff; the 80
  tests touching the final warning fixes pass with warnings treated as errors.
- [ ] Confirm that the author field contains only `Mahimn Patel`; no institutional
  affiliation is asserted.
- [ ] Re-run the build immediately before upload and compare the reported checksum. If any
  source changed, use the new checksum rather than the value above.

## Submission form

1. Sign in at <https://arxiv.org/user/login> and start a new submission. Complete any
   category endorsement requested by the arXiv dashboard before the deadline.
2. Select **CC BY 4.0**, matching `paper/arxiv_metadata.txt`.
3. Select primary category **q-fin.MF**. Request cross-lists **q-fin.TR** and **math.DS**, as
   recorded in `paper/arxiv_metadata.txt`, only if the interface permits them.
4. Upload `dist/arxiv_v0.4.0.tar.gz`. Do not add `paper/main.pdf`; the source archive already
   contains everything AutoTeX needs.
5. Select `main.tex` as the top-level file if AutoTeX does not infer it.
6. Paste title, author, abstract, comments, MSC codes, and categories from
   `paper/arxiv_metadata.txt`. Leave journal reference, report number, and DOI blank unless
   one has actually been assigned.
7. Confirm the form says **no market data are used** and does not resurrect the withdrawn
   stochastic-shift, Hawkes-equivalence, mean-field, stationary-tail, or dealer-sign claims.

## AutoTeX PDF inspection

Compare the rendered file with `paper/main.pdf`:

- [ ] exactly 22 pages;
- [ ] title is *Dealer-Gamma Feedback and Local Volatility Cycles: A Fixed-Equilibrium
  Bifurcation Model and a Pre-Extraction Identification Protocol*;
- [ ] author name, email, and ORCID are correct, and no institutional affiliation appears;
- [ ] Figures 1--2 appear once each, with all panels and legends legible;
- [ ] equations (1)--(27), theorem, six tables, and appendices A--D are present;
- [ ] no `??`, missing-asset box, blank page, clipped equation, or overfull line;
- [ ] bibliography contains the 28 references cited by this manuscript;
- [ ] the empirical section states the A13--A15 contract/source/inference/alignment locks and the
  appendix gives the option-before-outcome audit order;
- [ ] the limitations say the full state-dependent stochastic variational equation is not
  analysed and public OI does not identify dealer sign.

Do not submit until the AutoTeX PDF passes every item. A successful compilation alone is
not sufficient.

## Provenance files that are not part of the source upload

Keep these in the public repository and link the repository in comments; do not place them
inside the TeX tarball:

- `paper/pre_registration.md` and its historical proofs;
- `paper/pre_registration_amendments.md`, SHA-256
  `603e89366c0dbe49718e8c31f805d6f85d3c508e2e0ed4276a6310c80f5f9cd7`;
- `paper/pre_registration_amendments.md.ots` (A13--A14 receipt, pending Bitcoin
  consolidation at creation);
- `paper/pre_registration_amendments.md.a13` and `.a13.ots` (matching preserved A13-only
  snapshot and receipt; SHA-256 `83950ede…58e17`).
- `paper/pre_registration_amendment_a15.md`, SHA-256
  `a5f694f99953d57563d4f17dc5646ef0b87452c45119ccbdc12fc90efd034a52`, and its
  `.ots` receipt (pending Bitcoin consolidation at creation).

## If AutoTeX fails

1. Save the complete arXiv log before replacing files.
2. Compare it with `dist/arxiv_build.log` and the extracted source tree.
3. Reproduce the error from a fresh extraction; do not edit the frozen amendment record.
4. Fix only the source-level cause, rebuild the tarball, rerun its extraction test, inspect
   the new PDF, and upload the rebuilt archive.

## After announcement

- [ ] Record the arXiv identifier and v1 announcement date in README/CITATION metadata.
- [ ] Link the arXiv record from the repository description.
- [ ] Preserve the exact v1 tarball checksum and source tag/release.
- [ ] For any v2 containing market data, disclose the frozen raw manifests and every
  deviation before changing the manuscript's “no market data” statement.
