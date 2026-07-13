# Zenodo release/DOI checklist — v0.4.1

Zenodo is optional. It should archive an immutable tagged software release that matches the
paper, not a moving working tree. Consult Zenodo's current GitHub-integration documentation
before using the UI because OAuth and release-ingestion steps can change.

## Metadata now in the repository

`.zenodo.json` and `CITATION.cff` use:

- the centered-model title and v0.4.1 description;
- Mahimn Patel's ORCID `0009-0002-2422-9005`;
- MIT for source code and CC BY 4.0 for the manuscript/current figures;
- the A13--A16 sign-agnostic, no-registered-market-dataset scope;
- no fake DOI or arXiv identifier.

The legacy `.zenodo.json` format has no single accurate license value for this mixed-rights
deposit. Record both rights in the Zenodo draft if the interface supports file-level or multiple
licenses; otherwise create separate software and paper deposits. Do not apply MIT to the paper
or CC BY to vendored/code files by implication.

## Before creating a release

- [ ] Commit only the intended v0.4.1 reconstruction after reviewing `git diff`.
- [ ] Re-run `scripts/verify.sh` and all historical receipt replays.
- [ ] Re-run `scripts/arxiv_build.sh` and preserve the source-tar checksum.
- [ ] Confirm `paper/main.pdf` is 24 pages and matches the tagged source.
- [ ] Validate `.zenodo.json` as JSON and `CITATION.cff` with a current CFF validator.
- [ ] Confirm the draft records both MIT (code) and CC BY 4.0 (paper/current figures), or
  split the deposits before publication.
- [ ] Confirm the sole creator is Mahimn Patel, the ORCID is correct, and no
  institutional affiliation is asserted.
- [ ] Do not add an arXiv placeholder; add the real identifier only after assignment.

## First DOI-bearing release

1. Enable Zenodo's GitHub integration for `mahimn01/reflexive-options` if it is not already
   enabled.
2. Create and publish the GitHub tag/release `v0.4.1` from the exact verified commit.
3. Open the resulting Zenodo draft and verify title, creator, ORCID, mixed rights, version,
   description, keywords, and files before publication.
4. Ensure the archived repository contains the paper and timestamp receipts, but does not
   contain proprietary WRDS data.
5. Publish the deposit only after the draft metadata matches the tagged repository. Record
   both the version DOI and concept DOI.

## After a DOI or arXiv ID exists

- Add the version DOI for the exact v0.4.1 snapshot to `CITATION.cff` and, if useful, the
  manuscript's reproducibility paragraph.
- Put the concept DOI in the README badge so it resolves to the newest release.
- Add the real arXiv identifier to `.zenodo.json` as a related identifier and to the paper
  metadata. Never publish `XXXX.XXXXX`.
- If a future release includes market-data results, use a new version and describe the data
  manifests, access restrictions, and deviations. Do not modify the v0.4.1 deposit.
