# arXiv compatibility check — v0.4.0

The source archive `dist/arxiv_v0.4.0.tar.gz` was rebuilt and tested from a clean extraction
on 2026-07-11. Local TeX Live 2025 produced the same 22-page document as the master within
12 bytes of timestamp-level PDF metadata drift.

## Source closure

The tarball contains only:

1. `main.tex`;
2. `main.bbl`;
3. `references.bib`;
4. `figures/centered_hopf_validation.pdf`;
5. `figures/centered_hopf_robustness.pdf`.

There is no precompiled `main.pdf`, auxiliary file, absolute path, shell-escape dependency,
custom class/style, hidden macOS file, or unreferenced legacy figure. `main.bbl` is included
so arXiv does not need to resolve bibliography generation to compile the paper.

## TeX dependencies

The document uses the standard `article` class and packages `geometry`, `amsmath`,
`amssymb`, `amsthm`, `mathtools`, `bm`, `booktabs`, `array`, `graphicx`, `microtype`, `hyperref`,
`cleveref`, `natbib`, and `placeins`, with bundled `plainnat`. All are normal TeX Live
components. The
ORCID macro is a local one-line `\providecommand` built from `hyperref` and requires no
external package or asset.

## Final local result

- page count: 22;
- theorem count: 1;
- figure count: 2;
- table count: 6;
- cited bibliography entries: 28;
- unresolved references/citations: 0;
- overfull boxes: 0;
- arXiv source-tar SHA-256:
  `fa4813d33568e5059022f75f486286056e7eedc8ebb8df562bb6d2b71ac27264`.

The operative upload procedure is `paper/arxiv_submission_checklist.md`. If any shipped
source changes, rerun `scripts/arxiv_build.sh`; this checksum and compatibility result then
cease to certify the new archive until the clean-extraction test passes again.
