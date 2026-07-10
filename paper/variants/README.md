# Manuscript variants

Venue-length versions of the v0.4 centered-model manuscript. Both variants
share `paper/figures/` and `paper/references.bib` through symlinks.

| Directory | Format | Status |
|---|---|---|
| `neurips_workshop/` | NeurIPS workshop placeholder style | author-visible draft |
| `icaif/` | ACM SIGCONF review/anonymous | double-blind draft; Milan, 14--17 Nov. 2026 |

Both contain only the fixed-equilibrium, positive-variance model, the actual
Gaussian-book nonlinear validation, and the sign-agnostic A13--A15 data protocol.
The withdrawn stochastic-shift, Hawkes-equivalence, mean-field, information,
stationary-tail, and RL-primary claims do not appear.

Build with `make pdf` inside either directory. The expected page count should
be checked against the venue's current call before submission; conference
names, dates, style versions, and anonymity requirements can change.

The ICAIF 2026 call allows at most eight pages including references and uses
anonymous ACM `sigconf`; the current file satisfies those mechanical rules.
Its substantive venue fit is weak unless the submission makes a genuine AI/ML
contribution rather than relabelling nonlinear dynamics as AI. The NeurIPS file
still carries the 2024 style only as a layout placeholder: 2026 contribution
format is workshop-specific and must be replaced after an actual workshop call
is available. Neither caveat affects the master arXiv manuscript.
