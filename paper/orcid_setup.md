# ORCID registration walkthrough

## What ORCID is

A unique persistent identifier for researchers (16 digits in four groups of four), required by most journals and by arXiv for new author submissions in 2026 and beyond — disambiguates "Mahimn Patel" from any other researcher with the same name across the literature.

## Why now

The arXiv, NeurIPS GenAI in Finance Workshop, and ICAIF 2026 submissions all benefit from an ORCID iD attached to the author block; arXiv specifically links the iD into its author-claim infrastructure, which propagates to Google Scholar and Semantic Scholar.

## Step-by-step registration (~5 minutes)

1. Visit https://orcid.org/register
2. **Email:** enter `mahimn.patel.k@gmail.com`. (Re-enter to confirm.)
3. **Name:** First name `Mahimn`, last name `Patel`.
4. **Date of birth:** enter your real DOB (privacy is configurable; default is private to ORCID).
5. **Password:** generate via password manager; ORCID requires ≥8 characters with one number and one letter or symbol.
6. **Notification preferences:** opt out of marketing emails; keep transactional notifications on so you receive submission/affiliation confirmations.
7. **Visibility:** set the iD itself to *Everyone* (the default and the entire point of having one).
8. **CAPTCHA** + accept the terms.
9. ORCID emails a verification link to `mahimn.patel.k@gmail.com` — click it within 7 days or the registration expires.
10. Note the resulting iD on the dashboard. Format: `0000-XXXX-XXXX-XXXX`. Copy it to a password-manager note titled "ORCID iD".

## Where to add the iD in the repo

After verification, edit each of the following:

1. **`paper/main.tex` line 21** — uncomment the placeholder and substitute the iD:
   ```tex
   % \orcid{XXXX-XXXX-XXXX-XXXX}  % register at orcid.org/register, then uncomment
   ```
   becomes
   ```tex
   \orcid{0000-XXXX-XXXX-XXXX}
   ```
   Note: the bare `\orcid{}` command is not in the default `article` class. Either (a) add `\usepackage{orcidlink}` and use `\orcidlink{0000-XXXX-XXXX-XXXX}` after the email in the `\author{}` block, or (b) define a no-op `\providecommand{\orcid}[1]{\href{https://orcid.org/#1}{#1}}` near the preamble. Pick (a) for proper ORCID branding (renders the green ORCID logo).

2. **`paper/arxiv_metadata.txt` line 4** — replace `<FILL_IN_AT_SUBMISSION via orcid.org/register, see paper/orcid_setup.md>` with the bare iD `0000-XXXX-XXXX-XXXX`.

3. **`paper/variants/icaif/main.tex`** — ICAIF is double-blind on initial submission, so do **not** add the ORCID until camera-ready. At camera-ready, add it via `\orcidlink{}` in the author block alongside the de-anonymised name and affiliation.

4. **`paper/variants/neurips_workshop/main.tex`** — same anonymisation rule as ICAIF. Add ORCID at camera-ready only, after flipping `\usepackage{neurips_2024}` to `\usepackage[final]{neurips_2024}`.

## One-time profile setup recommendations

After registration, on https://orcid.org logged-in dashboard:

- **Affiliation → Add Education:** University of Toronto, Department of Economics, start date 2025-09 (current undergraduate). Visibility: *Everyone*.
- **Works:** add this paper as a manual "Work" entry once it has a DOI (post-arXiv submission yields a DOI of the form `10.48550/arXiv.XXXX.XXXXX`). For ICAIF / NeurIPS, the venue's publisher (ACM / OpenReview) auto-pushes accepted works once the iD is included in the submission metadata.
- **Visibility defaults:** set the *Biography*, *Country*, and *Keywords* fields to *Everyone*; keep DOB and email *Trusted parties* or *Only me*.
- **Trusted organisations:** authorise arXiv when prompted at first submission; this enables auto-population of the works list on future submissions.

## Verification

After editing `paper/main.tex` to add the ORCID command, recompile and verify the iD renders in the PDF:

```bash
cd paper && make
```

Open `paper/main.pdf` and confirm the iD (or the green ORCID logo if `orcidlink` is used) appears in the author block on page 1. Re-run `bash scripts/verify.sh` from the repo root to confirm no LaTeX warnings from the new package.

If the variants have been de-anonymised for camera-ready, recompile each:

```bash
cd paper/variants/icaif && latexmk -pdf main.tex
cd paper/variants/neurips_workshop && latexmk -pdf main.tex
```

and confirm the iD renders in both variant PDFs.
