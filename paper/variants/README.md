# Manuscript variants

Compressed-target variants of `paper/main.tex` (the 23-page master) for
specific submission venues. Both variants share `paper/figures/` and
`paper/references.bib` via symlinks; do not edit them in place.

## Variants

| Directory | Target venue | Page limit (excl. refs) | Body pages | Style |
| --- | --- | --- | --- | --- |
| `neurips_workshop/` | NeurIPS GenAI-in-Finance Workshop 2026 | 4 | 4 | `neurips_2024.sty` (single-column) |
| `icaif/` | ICAIF 2026 (double-blind) | 8 | 4 | `acmart` `[sigconf,review,anonymous]` (two-column) |

Both variants compile cleanly with `pdflatex` + `bibtex` + `pdflatex`
+ `pdflatex` (no errors, no overfull boxes, no missing references).

## Build

```bash
# NeurIPS workshop variant
cd neurips_workshop && make pdf
# -> main.pdf (6 pages total: 4 body + 2 refs)

# ICAIF variant
cd icaif && make pdf
# -> main.pdf (5 pages total: 4 body + ~1.5 refs in two columns)
```

`make clean` removes all build artefacts (`*.aux`, `*.bbl`, `*.blg`,
`*.log`, `*.out`, `*.pdf`).

## Style files

### NeurIPS workshop

`style/neurips_2024.sty` — official NeurIPS 2024 submission/camera-ready
LaTeX style. The official source is the NeurIPS 2024 LaTeX template
distributed by neurips.cc; we use the file mirrored at
<https://raw.githubusercontent.com/official-Auralin/Multimodal-World-Simulation-Architecture/main/neurips_2024.sty>
because the canonical neurips.cc URL returns 403 to non-browser clients.
A symlink at the variant root (`neurips_2024.sty`) makes the file
discoverable to `pdflatex` from the working directory.

The file's `\ProvidesPackage` line attests to its provenance:
```
\ProvidesPackage{neurips_2024}[2024/03/31 NeurIPS 2024 submission/camera-ready style file]
```

### ICAIF

Uses the system `acmart` (TeX Live `acmart.cls`, version 2024/12/28
v2.12). No vendored style file required.

## Compression strategy

Both variants reuse the master figures unchanged
(`figures/hopf_phase_diagram.pdf`, `figures/h4_detector_power_v2.pdf`)
and the master `references.bib`.

### NeurIPS workshop (4 pages)

- **Drop**: closed-form proof details; appendix A; SW2 sample-complexity
  table; Marketron mechanism details; numerical-anchor figure;
  $\ell_1$ phase-boundary figure; stationary-density figure.
- **Keep**: Theorem 1 statement; closed-form $\kappa^\star$ (Eq. 5);
  Hopf phase-diagram figure (1 panel); H4 detector-power figure;
  H1 synthetic-validation result (text + numbers, no separate table);
  pre-registration four-way novelty paragraph.
- **Audience emphasis**: pre-registered evaluation framework as the
  RL/generative-model evaluation methodology contribution; Bayesian /
  frequentist rigor (block-bootstrap, BH-FDR, TOST, IAAFT); reflexivity
  as the high-stakes corner of GenAI-in-finance.

### ICAIF (8 pages)

- **Drop**: appendix A (closed-form $\ell_1$ in 13 symbols, deferred to
  arXiv per ICAIF cross-reference policy); $\ell_1$ closed-form-tex
  inclusion; numerical-anchor table on its own; $\ell_1$ phase-boundary
  figure; lambda-scaling figure; stationary-density figure;
  H1-synthetic-ordering figure (table only is kept).
- **Keep**: full Theorem 1 with assumptions (A1)--(A5) and proof sketch;
  closed-form $\kappa^\star$ + Kuznetsov $\ell_1$ formula; Hopf phase
  diagram; SW2 sample-complexity table; H1 synthetic-validation table;
  mechanism decomposition with the $8/24$ headline + restricted-subset
  result.
- **Anonymization**: `\author{Anonymous Author(s)}` /
  `\affiliation{Anonymous Institution}`; ORCID commented out; git commit
  hash replaced with `\texttt{[anonymized]}`; repository URL replaced
  with `[anonymous repository link]`.
