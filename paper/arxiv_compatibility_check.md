# arXiv TeXLive compatibility check — main.tex (v0.3.3)

arXiv's AutoTeX pipeline runs **TeX Live 2023** (default) with TeX Live 2024
opt-in via a top-of-file pragma. Every package in `paper/main.tex` is on
the standard TeXLive distribution; no contributed/experimental packages,
no `texmf-local` customisations, no `\usepackage{minted}`-style shell-escape
gotchas.

Local TeXLive used to verify the build: **TeX Live 2025** (pdfTeX
3.141592653-2.6-1.40.27). All listed versions are arXiv-side equivalents
unless noted.

## Package-by-package verdict

| `\usepackage{...}`        | TeXLive component | Local version          | arXiv 2023/2024 | Verdict |
|---------------------------|-------------------|------------------------|-----------------|---------|
| `geometry`                | `geometry`        | v5.9 (2020-01-02)      | included        | OK      |
| `amsmath,amssymb,amsthm`  | `amsmath`/`amsfonts`/`amscls` | v2.17t (amsmath), v3.01 (amssymb), v2.20.6 (amsthm) | included | OK |
| `mathtools`               | `mathtools`       | v1.31 (2024-10-04)     | included        | OK      |
| `bm`                      | `tools`           | v1.2f (2023-12-19)     | included        | OK      |
| `microtype`               | `microtype`       | (no header version, current) | included  | OK      |
| `booktabs`                | `booktabs`        | v1.61803398 (2020-01-12) | included      | OK      |
| `graphicx`                | `graphics`        | (LaTeX kernel-bundle)  | included        | OK      |
| `hyperref` (option `hidelinks`) | `hyperref`  | v7.01l (2024-11-05)    | included        | OK      |
| `cleveref`                | `cleveref`        | current                | included        | OK      |
| `natbib` (option `round`) | `natbib`          | v8.31b (2010-09-13)    | included        | OK      |
| `plainnat` bibstyle (via `\bibliographystyle{plainnat}`) | `natbib` | bundled | included | OK |

All eleven `\usepackage` lines resolve to the **base/recommended TeXLive
scheme**. No `scheme-full`-only packages. No deprecated packages.

## `\providecommand{\orcid}{...}` shim — portability check

```latex
% line 12-14 of main.tex
% ORCID command shim — if you have the `orcidlink` package, replace this line with
% `\usepackage{orcidlink}` and use `\orcidlink{XXXX-XXXX-XXXX-XXXX}` for the icon-link.
\providecommand{\orcid}[1]{\\\texttt{ORCID: #1}}
```

- `\providecommand` is a LaTeX2e primitive that no-ops if `\orcid` is
  already defined elsewhere. Defining `\orcid` here is harmless on every
  TeX engine (pdflatex / xelatex / lualatex).
- The body `\\\texttt{ORCID: #1}` evaluates as `\\` (forced linebreak) +
  `\texttt{...}` (typewriter span), so the macro safely no-ops when not
  invoked and produces predictable output when invoked.
- The `orcidlink` package itself **is** on TeXLive (verified via
  `kpsewhich orcidlink.sty` → present in `/texmf-dist/tex/latex/orcidlink/`
  on TeXLive 2025 and earlier). If you later switch to the icon-link
  variant, `\usepackage{orcidlink}` will work on arXiv without further
  changes.
- The current shim is invoked **only** if you uncomment line 25
  (`% \orcid{XXXX-XXXX-XXXX-XXXX}`). In the shipped v0.3.3 the line is
  commented, so the shim is dormant and the manuscript renders without
  any ORCID side-effects.

**Verdict: OK — portable, no shim replacement required for arXiv.**

## Document-level features cross-check

| Feature                                      | Status            |
|----------------------------------------------|-------------------|
| `\documentclass[11pt]{article}`              | standard          |
| UTF-8 source encoding                        | OK (no inputenc needed in modern LaTeX) |
| Unicode in math (`χ`, `κ`, etc. NOT used — Greek via `\kappa` etc.) | OK |
| `\input{figures/ell1_closed_form.tex}`       | shipped inside tarball |
| `\bibliography{references}` + `.bbl` shipped | OK (arXiv prefers bbl) |
| `\bibliographystyle{plainnat}`               | bundled with `natbib` |
| `\includegraphics{...}` paths                | all under `figures/` (relative, no `..`, no absolute) |
| Figure formats                               | PDF only (no EPS, no PNG) — preferred by AutoTeX |
| Theorem environments via `\newtheorem`       | OK (amsthm provides) |
| Custom macros                                | only `\orcid` shim — see above |
| Microtype protrusion/expansion               | OK with pdfTeX (skipped silently under xetex/luatex) |

## Empty-glue / encoding spot-check

`file paper/main.tex` reports `LaTeX 2e document text, Unicode text, UTF-8 text,
with very long lines`. No BOM, no Windows line endings (verified via the
file(1) classifier). Long-line wrapping is intentional (paragraphs as
single physical lines); AutoTeX is line-length-agnostic.

## Things that would have been blockers (and aren't)

| Common arXiv rejection cause | Present in v0.3.3? |
|------------------------------|--------------------|
| Pre-compiled `main.pdf` in source tarball  | **No** (stripped by `arxiv_build.sh` step 6) |
| Missing `.bbl` (with no `.bib` route)      | **No** (`.bbl` shipped + `.bib` shipped) |
| `\includegraphics{/abs/path}` outside source | **No** (all paths under `figures/`) |
| `\input{}` referencing files outside source  | **No** (only `figures/ell1_closed_form.tex`, shipped) |
| `.DS_Store` / macOS metadata                 | **No** (`find -delete` in step 6) |
| Auxiliary build files (`.aux`, `.log`, `.fls`, `.fdb_latexmk`) | **No** (stripped) |
| Custom .sty / .cls / .bst not on TeXLive     | **No** (zero custom files) |
| Shell-escape requirement (`minted`, `pgfplots` with external) | **No** |
| Outdated package version requirement         | **No** (all bundled in TL2023+) |

## Summary

**Zero blockers. Zero shims required beyond the existing dormant
`\providecommand{\orcid}`.** The manuscript is built with the
recommended-scheme TeXLive packages only, all bundled in TL2023 and
later. Submission should pass arXiv's `AutoTeX` sanity check on first
upload.
