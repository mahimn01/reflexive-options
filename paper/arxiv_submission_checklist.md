# arXiv submission checklist — Reflexivity in Options Markets v0.3.3

One-page operating procedure. Run `scripts/arxiv_build.sh` (or `make -C
paper arxiv`) first to refresh `dist/arxiv_v0.3.3.tar.gz`. The simulated
extract+rebuild already passed, so the only remaining risk is the live
arXiv UI.

## Pre-flight (local)

- [ ] `bash scripts/arxiv_build.sh` exits cleanly (last line: `arxiv_build.sh DONE`)
- [ ] `dist/arxiv_v0.3.3.tar.gz` exists, size ≈ 425 KB
- [ ] Tarball SHA-256 matches the value reported by the script (re-run; bit-identical)
- [ ] `dist/arxiv_build.log` shows `page count = 30` and `arXiv-sim built PDF: 30 pages`
- [ ] `paper/main.tex` and `paper/references.bib` unmodified (no accidental drift)

## arXiv submission flow

1. **Log in** at <https://arxiv.org/user/login> (or register at
   <https://arxiv.org/user/register> if first submission). First-time
   submitters need endorsement for `q-fin.MF` — see
   <https://arxiv.org/help/endorsement>. Plan ahead: endorsement requests
   can take 1–7 days.

2. **Start a new submission**: <https://arxiv.org/submit> → "Start new
   submission".

3. **License**: choose **`CC BY 4.0`** (matches `arxiv_metadata.txt`).

4. **Primary archive**: `q-fin` → primary class `q-fin.MF` (Mathematical
   Finance).

5. **Upload source**: drag-and-drop `dist/arxiv_v0.3.3.tar.gz` into the
   file-upload area. AutoTeX will untar and identify `main.tex` as the
   top file. **Do NOT** upload the pre-built PDF; arXiv must compile
   from source.

6. **Wait for AutoTeX**: typically 30–120 s. Successful build shows
   "Processing complete" and offers a "View PDF" link.

7. **Verify the rendered PDF** against `paper/main.pdf`:
   - Page count: **30**
   - First page title: *Reflexivity in Options Markets: A
     Stochastic-Volatility Model with Dealer-Gamma Feedback, Hopf
     Bifurcation Calculus, and a Pre-Registered Evaluation Framework*
   - Abstract present, no missing-figure boxes
   - All 13 figure references resolve (no `??` markers)
   - Bibliography renders 63 entries (matches `\begin{thebibliography}{63}`
     in `main.bbl`)
   - Appendix A typesets the closed-form ℓ₁ expression

8. **Metadata page** (paste from `paper/arxiv_metadata.txt`):
   - **Title**: Reflexivity in Options Markets: A Stochastic-Volatility
     Model with Dealer-Gamma Feedback, Hopf Bifurcation Calculus, and a
     Pre-Registered Evaluation Framework
   - **Authors**: `Mahimn Patel`
   - **Author affiliation**: Department of Economics, University of Toronto
   - **Abstract**: copy the full `\begin{abstract}...\end{abstract}` block
     from `main.tex` (lines 32–34). Strip LaTeX commands when pasting
     (arXiv auto-converts most `\citep{}` but \\cref{} renders literally —
     replace with prose).
   - **Comments**: `29 pages (body + references + Appendix A), 9 figures.
     Code at https://github.com/mahimn01/reflexive-options. Pre-registration
     anchored at commit hash 268c061.`
   - **MSC codes**: `91G80, 37G15, 60H10, 91G60, 93E20`
   - **Report-no / journal-ref / DOI**: leave blank for v1
   - **ACM-class**: leave blank

9. **Cross-list**: q-fin.CP, q-fin.PR, q-fin.ST, math.DS (in that order).
   Cross-list requires endorsement in each archive if the primary is in a
   different archive. `q-fin.*` is the same archive; `math.DS` is
   different and may need a separate endorsement.

10. **Final preview**: re-check title, authors, abstract, categories.
    Submit.

11. **Wait for moderation**: ~24–48 h typical; up to a week if flagged.
    arXiv will email when announced. You'll receive an arXiv ID of the
    form `2605.NNNNN` (with current year 2026 → 26 prefix, current
    month → 05). The canonical URL is `https://arxiv.org/abs/2605.NNNNN`.

## After acceptance

- [ ] Add the arXiv ID to `README.md` and `paper/main.tex` if/when a v2 is
      prepared.
- [ ] Add a `## [v0.3.3] - 2026-05-23` entry to `CHANGELOG.md` noting the
      arXiv submission.
- [ ] Add the arXiv URL to the GitHub repo description.
- [ ] Tweet/mailing-list announce (optional; consider after correspondence
      with Halperin/Itkin lands).

## If AutoTeX fails

1. Click "View Log" on the arXiv submission page — the AutoTeX log
   pinpoints the missing file / package / error.
2. Compare against the local `pdflatex main.tex` log under
   `dist/arxiv_extract_test/arxiv_sanity.*/main.log`. The two should be
   nearly identical except for arXiv banner lines.
3. **Do not** edit `paper/main.tex` reactively. The local
   `arxiv_build.sh` already passed a full extract+rebuild simulation; any
   AutoTeX failure points to a TeXLive 2023/2024 vs 2025 drift, not a
   manuscript bug. Note the drift and report back — the main agent owns
   `main.tex` edits.
4. arXiv lets you "Replace files" without losing the metadata draft.
   Rebuild via `scripts/arxiv_build.sh`, re-upload, re-trigger the
   sanity-check.

## Reference

- `dist/arxiv_v0.3.3.tar.gz` — the source tarball to upload
- `dist/arxiv_build.log` — full build trace (extract + rebuild)
- `paper/arxiv_compatibility_check.md` — package-by-package TeXLive audit
- `paper/arxiv_metadata.txt` — metadata to paste into the arXiv form
- `paper/orcid_setup.md` — ORCID enrollment notes (if you want
  `\orcid{}` lit on a future v2)
