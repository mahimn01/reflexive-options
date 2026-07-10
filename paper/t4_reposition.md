# Theorem 4 Repositioning: Hawkes–SV Criticality Correspondence

> **ARCHIVED / WITHDRAWN.** There is no Hawkes--SV equivalence theorem in the
> current centered-model paper.

**Status:** PRE-DATA pre-registration amendment. Replaces the §3.11 block in `paper/main.tex`
and the §6 block in `paper/theory.md`. Implements the locked decision (a)–(d):

- (a) State the Hopf bifurcation as a **strictly stronger, oscillatory** instability sitting
  **beyond** the real-eigenvalue branching-ratio edge.
- (b) Acknowledge the `c0 = 0` (saddle-node / real-eigenvalue) locus as the **literal
  Hardiman n=1 analogue** — Hardiman's criticality is non-oscillatory.
- (c) **DROP** the `n_SV := c0/(c1 c2)` "machine-precision 1e-15 verification" framing entirely
  (three referees: definitional tautology / numerology).
- (d) Add rough-volatility precedent citations; frame the rough-vol bridge as explicit
  **FUTURE WORK** (cite, do not prove).

---

## 0. Summary of the change for the main agent

The original Theorem 4 claimed an exact equivalence "Hawkes criticality ⟺ Hopf bifurcation of
the SV drift," and *verified* it by defining a dimensionless quantity
`n_SV := c0 / (c1 c2)` from the SV coefficients, evaluating it numerically, and reporting that
it equals the Hawkes branching ratio `n` "to machine precision (≈1e-15)." That verification is
a tautology: `n_SV` is *defined* so that the identity holds, so the 1e-15 residual measures
floating-point arithmetic, not an empirical or structural correspondence. We remove it.

What survives is the genuine, defensible content:

1. The **diffusive (large-baseline) limit** of a nearly-unstable Hawkes process is a
   continuous-time stochastic-volatility process whose integrated intensity solves a
   Cox–Ingersoll–Ross-type SDE (Bacry–Delattre–Hoffmann–Muzy 2013, "BDHM"; Jaisson–Rosenbaum
   2015 for the *near-critical* rescaling). This is a theorem, not a definition.

2. Under this limit, the Hawkes **stability boundary** (branching ratio `n → 1`, equivalently
   the spectral radius of the kernel norm reaching 1) maps to a **degeneracy of the SV mean-
   reversion drift** — the linearized drift loses a stable eigenvalue.

3. The endpoint of the SV bifurcation diagram has two distinct codimension-1 loci, and we are
   careful about which one corresponds to Hardiman:
   - a **real-eigenvalue (saddle-node) locus**, the literal `n = 1` Hardiman analogue, where a
     real eigenvalue of the drift Jacobian crosses zero — **non-oscillatory** criticality; and
   - a **Hopf locus**, where a *complex-conjugate pair* crosses the imaginary axis — a
     **strictly stronger** condition that additionally requires nonzero imaginary part, and
     which produces sustained **oscillations / limit cycles** (the reflexive feedback loop the
     paper is actually about).

4. The novelty we keep — the "unoccupied cell" — is that the **oscillatory** (Hopf) endpoint of
   the SV picture has *no* counterpart in the scalar branching-ratio criterion of the Hawkes
   literature, because a scalar branching ratio can only encode a real-eigenvalue crossing. The
   Hopf cell is genuinely new; we claim it as a *position*, not a verified identity.

5. The rough-volatility extension (does the Hopf cell survive when the Hawkes kernel is heavy-
   tailed, giving rough vol with Hurst `H < 1/2`?) is framed as **future work that is probably
   negative**: fractional Brownian motion with `H < 1/2` is nowhere differentiable, so no smooth
   vector field and hence no smooth limit cycle / Hopf normal form survives the rough limit. We
   cite the rough-vol literature for the limit construction and state the obstruction; we do not
   prove anything.

---

## 1. LaTeX-ready replacement for `paper/main.tex` §3.11

> **Find-and-replace instructions are in §4 below.** Drop the following block in verbatim.

```latex
\subsection{Hawkes criticality and the stochastic-volatility bifurcation endpoint}
\label{subsec:hawkes-sv}

The reflexive feedback mechanism of this paper has a self-exciting point-process
representation: order flow that begets order flow is, in the linear regime, a
Hawkes process, and the onset of endogeneity is its approach to criticality. We
record here the precise sense in which the Hawkes criticality boundary
corresponds to a degeneracy of the drift of a stochastic-volatility (SV)
diffusion, and we are careful to separate two distinct codimension-one events at
that boundary: a non-oscillatory \emph{real-eigenvalue} crossing, which is the
literal analogue of the scalar branching-ratio criticality studied by
\citet{hardiman2013} and \citet{bacrymastromatteomuzy2015}, and an
\emph{oscillatory} Hopf crossing, which is strictly stronger and which we claim
as the novel object.

\paragraph{The diffusive limit (what is a theorem).}
Let $\lambda_t$ be the intensity of a (possibly multivariate) linear Hawkes
process with baseline $\mu$, kernel $\phi$, and kernel $L^1$-norm
$n := \lVert \phi \rVert_1$ (the \emph{branching ratio}). When the baseline is
sent to infinity and the process is observed on the matching macroscopic
timescale, the rescaled intensity converges to a continuous diffusion
\citep{bacrydelattrehoffmannmuzy2013}. In the \emph{near-critical} regime
$n \uparrow 1$, taken jointly with the long-time/large-baseline rescaling, the
limit is a stochastic-volatility process: the integrated intensity solves a
Cox--Ingersoll--Ross-type stochastic differential equation
\citep{jaissonrosenbaum2015}. Write the limiting SV dynamics in the generic
mean-reverting form
\begin{equation}
  \mathrm{d} v_t \;=\; b(v_t)\,\mathrm{d} t \;+\; \sigma(v_t)\,\mathrm{d} W_t ,
  \qquad b(v_\ast) = 0 ,
  \label{eq:sv-drift}
\end{equation}
with a fixed point $v_\ast$ and drift Jacobian $J := Db(v_\ast)$. This passage
from Hawkes to SV is a limit theorem; nothing below depends on any choice of
parametrisation of $b$.

\paragraph{The criticality endpoint (the correspondence we keep).}
Criticality of the underlying Hawkes process, $n \to 1$, corresponds to the loss
of a stable direction of the SV drift in \eqref{eq:sv-drift}: an eigenvalue of
$J$ reaches the imaginary axis. The boundary of the stable region therefore
carries two distinct codimension-one strata.
\begin{enumerate}
  \item \textbf{Real-eigenvalue (saddle-node) locus.} A single real eigenvalue
  of $J$ crosses zero, $\det J = 0$ with the crossing eigenvalue real. This is
  the \emph{literal} stochastic-volatility image of scalar branching-ratio
  criticality $n = 1$: the scalar Hardiman criterion can only ever encode a real
  crossing, because a scalar branching ratio is a single nonnegative number. The
  resulting criticality is \emph{non-oscillatory}; there is no intrinsic
  frequency.
  \item \textbf{Hopf locus.} A complex-conjugate pair of eigenvalues of $J$
  crosses the imaginary axis, $\operatorname{Re}\lambda = 0$ with
  $\operatorname{Im}\lambda \neq 0$. This is a \emph{strictly stronger}
  condition than the real-eigenvalue crossing: it requires, in addition to a
  vanishing real part, a nonzero imaginary part. It produces sustained
  oscillations and, generically, a limit cycle.
\end{enumerate}

\begin{theorem}[Hawkes--SV criticality correspondence]
\label{thm:hawkes-sv}
Under the diffusive near-critical limit of
\citet{bacrydelattrehoffmannmuzy2013} and \citet{jaissonrosenbaum2015}, the
Hawkes branching-ratio criticality $n \to 1$ coincides with the real-eigenvalue
(saddle-node) stratum of the SV drift boundary \eqref{eq:sv-drift}, the literal
analogue of \citet{hardiman2013}. The Hopf stratum of the same boundary is a
strictly stronger, oscillatory instability that lies beyond any scalar
branching-ratio criterion: it has no representation as a single real branching
ratio and is the genuinely new ``unoccupied cell'' of the
endogeneity--bifurcation correspondence.
\end{theorem}

\begin{remark}[Why the Hopf cell is the novelty, and why we claim it as a
position rather than an identity]
The Hawkes literature measures endogeneity by a scalar branching ratio
\citep{hardiman2013,bacrymastromatteomuzy2015}; the kernel-universality results
of \citet{bacrymastromatteomuzy2015} show this scalar summary is robust across
estimated kernels. A scalar can only detect a real-eigenvalue crossing. The
oscillatory Hopf endpoint is therefore invisible to the standard branching-ratio
diagnostic by construction. We do \emph{not} assert a verified numerical identity
between a Hawkes quantity and an SV quantity; an earlier draft of this section
introduced a dimensionless ratio engineered so that such an identity held by
definition, which is a tautology and which we have removed. The content of
Theorem~\ref{thm:hawkes-sv} is structural: it places the oscillatory instability
strictly beyond the real branching-ratio edge and names the cell.
\end{remark}

\paragraph{Rough-volatility extension (future work; likely negative).}
The diffusive limit above assumes a short-memory (integrable, light-tailed)
Hawkes kernel. When the kernel is heavy-tailed, the same near-critical scaling
produces \emph{rough} volatility with Hurst exponent $H < 1/2$
\citep{jaissonrosenbaum2016,euchrosenbaum2019,gatheraljaissonrosenbaum2018,abijaber2019}.
It is natural to ask whether the oscillatory Hopf cell of
Theorem~\ref{thm:hawkes-sv} survives this rough limit. We flag this as future
work and we expect the answer for the headline oscillatory object to be
\emph{negative}: a rough volatility path is driven by fractional Brownian motion
with $H < 1/2$, whose sample paths are nowhere differentiable, so there is no
smooth vector field on the state space and hence no smooth Hopf normal form or
classical limit cycle to inherit. A surviving analogue would have to be a
stochastic or rough-path notion of recurrent oscillation rather than a smooth
limit cycle; the Markovian lifts of \citet{abijaber2019} and the
characteristic-function machinery of \citet{euchrosenbaum2019} are the natural
tools for such a study. We make no claim here beyond stating the obstruction.
```

---

## 2. LaTeX-ready replacement for `paper/theory.md` §6

`theory.md` is the prose companion; it uses Markdown with inline math. Drop in
the following block (it carries the same content, lighter on machinery, citations
as `[Author Year]` keys matching the bib so the main agent can convert to
`\citep` if `theory.md` is also compiled, or leave as plain text).

```markdown
## 6. Hawkes criticality and the stochastic-volatility bifurcation endpoint

The reflexive loop has a self-exciting point-process face: flow that triggers
flow is a Hawkes process in its linear regime, and the onset of endogeneity is
its approach to criticality. This section states, carefully, how that Hawkes
criticality boundary corresponds to a degeneracy in the drift of a stochastic-
volatility (SV) diffusion — and, crucially, distinguishes two different events at
that boundary.

**What is a theorem (the diffusive limit).** Take a linear Hawkes intensity with
kernel norm (branching ratio) `n = ‖φ‖₁`. Sending the baseline to infinity and
rescaling time, the intensity converges to a continuous diffusion [Bacry,
Delattre, Hoffmann & Muzy 2013]. In the near-critical regime `n ↑ 1`, taken
together with that rescaling, the limit is a stochastic-volatility process whose
integrated intensity is CIR-type [Jaisson & Rosenbaum 2015]. Write the limiting
SV dynamics generically as `dv = b(v) dt + σ(v) dW` with fixed point `v*` and
drift Jacobian `J = Db(v*)`. The Hawkes→SV passage is a limit theorem; none of
what follows depends on a particular parametrisation of `b`.

**The correspondence we keep.** Hawkes criticality `n → 1` is the loss of a
stable direction of the SV drift: an eigenvalue of `J` reaches the imaginary
axis. The boundary has two distinct codimension-1 strata.

- **Real-eigenvalue (saddle-node) locus** — a single real eigenvalue of `J`
  crosses zero. This is the *literal* SV image of scalar branching-ratio
  criticality `n = 1`: a scalar branching ratio is one nonnegative number, so it
  can only encode a real crossing. This is exactly the (non-oscillatory)
  criticality of [Hardiman, Bercot & Bouchaud 2013]. No intrinsic frequency.
- **Hopf locus** — a complex-conjugate pair of eigenvalues of `J` crosses the
  imaginary axis (`Re λ = 0`, `Im λ ≠ 0`). This is *strictly stronger* than the
  real crossing: on top of a vanishing real part it demands a nonzero imaginary
  part. It produces sustained oscillations and, generically, a limit cycle — the
  reflexive feedback loop this paper is about.

**Theorem 4 (Hawkes–SV criticality correspondence).** Under the diffusive near-
critical limit [Bacry–Delattre–Hoffmann–Muzy 2013; Jaisson–Rosenbaum 2015],
Hawkes branching-ratio criticality `n → 1` coincides with the **real-eigenvalue
(saddle-node)** stratum of the SV drift boundary — the literal analogue of
[Hardiman et al. 2013]. The **Hopf** stratum of the same boundary is a *strictly
stronger, oscillatory* instability lying beyond any scalar branching-ratio
criterion: it has no representation as a single real branching ratio and is the
genuinely new "unoccupied cell" of the endogeneity–bifurcation correspondence.

**Why the Hopf cell is the novelty (and why it is a position, not an identity).**
The Hawkes literature summarises endogeneity by a scalar branching ratio
[Hardiman et al. 2013], and kernel-universality [Bacry, Mastromatteo & Muzy 2015]
shows that scalar summary is robust across estimated kernels. A scalar can only
see a real-eigenvalue crossing, so the oscillatory Hopf endpoint is invisible to
the standard diagnostic *by construction*. We deliberately do **not** assert a
verified numerical identity between a Hawkes number and an SV number. An earlier
draft introduced a dimensionless ratio `n_SV := c0/(c1 c2)` engineered so the
identity `n_SV = n` held by definition and then "verified" it to ~1e-15 — that is
a tautology (the residual is floating-point noise, not evidence), and it has been
removed. The claim of Theorem 4 is structural: it positions the oscillatory
instability strictly beyond the real branching-ratio edge and names it.

**Rough-volatility extension (future work, expected negative).** The diffusive
limit assumes a short-memory kernel. A heavy-tailed kernel under the same scaling
gives *rough* volatility, `H < 1/2` [Jaisson & Rosenbaum 2016; Gatheral, Jaisson
& Rosenbaum 2018; El Euch & Rosenbaum 2019; Abi Jaber 2019]. Does the Hopf cell
survive? We flag this as future work and expect the headline oscillatory object
to **not** survive: rough volatility is driven by fractional Brownian motion with
`H < 1/2`, whose paths are nowhere differentiable, so there is no smooth vector
field and hence no smooth Hopf normal form / classical limit cycle to inherit.
Any surviving analogue would be a rough-path / stochastic notion of recurrent
oscillation, not a smooth limit cycle; the Markovian lifts of [Abi Jaber 2019]
and the characteristic-function tools of [El Euch & Rosenbaum 2019] are the
natural machinery for such a study. We make no claim beyond stating the
obstruction.
```

---

## 3. Verified bib entries to ADD to `paper/references.bib`

Bibliographic details verified against the publishers / arXiv. `jaissonrosenbaum2015`
is already in the bib (the *Limit theorems for nearly unstable Hawkes processes*,
Ann. Appl. Probab. 2015) — do **not** duplicate it. The entries below are the ones to add;
each citation key used in §1/§2 above matches one of these (or the already-present key).

```bibtex
@article{bacrydelattrehoffmannmuzy2013,
  author  = {Bacry, Emmanuel and Delattre, Sylvain and Hoffmann, Marc and Muzy, Jean-Fran\c{c}ois},
  title   = {Some limit theorems for {H}awkes processes and application to financial statistics},
  journal = {Stochastic Processes and their Applications},
  volume  = {123},
  number  = {7},
  pages   = {2475--2499},
  year    = {2013},
  doi     = {10.1016/j.spa.2013.04.007}
}

@article{bacrymastromatteomuzy2015,
  author  = {Bacry, Emmanuel and Mastromatteo, Iacopo and Muzy, Jean-Fran\c{c}ois},
  title   = {{H}awkes processes in finance},
  journal = {Market Microstructure and Liquidity},
  volume  = {1},
  number  = {1},
  pages   = {1550005},
  year    = {2015},
  doi     = {10.1142/S2382626615500057}
}

@article{hardiman2013,
  author  = {Hardiman, Stephen J. and Bercot, Nicolas and Bouchaud, Jean-Philippe},
  title   = {Critical reflexivity in financial markets: a {H}awkes process analysis},
  journal = {The European Physical Journal B},
  volume  = {86},
  number  = {10},
  pages   = {442},
  year    = {2013},
  doi     = {10.1140/epjb/e2013-40107-3}
}

@article{jaissonrosenbaum2016,
  author  = {Jaisson, Thibault and Rosenbaum, Mathieu},
  title   = {Rough fractional diffusions as scaling limits of nearly unstable heavy tailed {H}awkes processes},
  journal = {The Annals of Applied Probability},
  volume  = {26},
  number  = {5},
  pages   = {2860--2882},
  year    = {2016},
  doi     = {10.1214/15-AAP1164}
}

@article{gatheraljaissonrosenbaum2018,
  author  = {Gatheral, Jim and Jaisson, Thibault and Rosenbaum, Mathieu},
  title   = {Volatility is rough},
  journal = {Quantitative Finance},
  volume  = {18},
  number  = {6},
  pages   = {933--949},
  year    = {2018},
  doi     = {10.1080/14697688.2017.1393551}
}

@article{euchrosenbaum2019,
  author  = {{El Euch}, Omar and Rosenbaum, Mathieu},
  title   = {The characteristic function of rough {H}eston models},
  journal = {Mathematical Finance},
  volume  = {29},
  number  = {1},
  pages   = {3--38},
  year    = {2019},
  doi     = {10.1111/mafi.12173}
}

@article{euchrosenbaum2018,
  author  = {{El Euch}, Omar and Rosenbaum, Mathieu},
  title   = {Perfect hedging in rough {H}eston models},
  journal = {The Annals of Applied Probability},
  volume  = {28},
  number  = {6},
  pages   = {3813--3856},
  year    = {2018},
  doi     = {10.1214/18-AAP1408}
}

@article{abijaber2019,
  author  = {{Abi Jaber}, Eduardo},
  title   = {Lifting the {H}eston model},
  journal = {Quantitative Finance},
  volume  = {19},
  number  = {12},
  pages   = {1995--2013},
  year    = {2019},
  doi     = {10.1080/14697688.2019.1615113}
}
```

**Citation-key cross-reference** (so the main agent can wire `\citep` keys):

| Key used in §1/§2 | Paper | In bib already? |
|---|---|---|
| `bacrydelattrehoffmannmuzy2013` | BDHM, limit theorems for Hawkes (SPA 2013) | ADD |
| `jaissonrosenbaum2015` | Limit theorems for nearly unstable Hawkes (AAP 2015) | ALREADY PRESENT — reuse |
| `jaissonrosenbaum2016` | Rough fractional diffusions / heavy-tailed Hawkes (AAP 2016) | ADD |
| `hardiman2013` | Critical reflexivity, Hawkes (EPJ B 2013) | verify/ADD if absent |
| `bacrymastromatteomuzy2015` | Hawkes processes in finance (MML 2015) | verify/ADD if absent |
| `gatheraljaissonrosenbaum2018` | Volatility is rough (QF 2018) | ADD |
| `euchrosenbaum2019` | Characteristic fn of rough Heston (Math Finance 2019) | ADD |
| `euchrosenbaum2018` | Perfect hedging in rough Heston (AAP 2018) | ADD (optional, cited as alt) |
| `abijaber2019` | Lifting the Heston model (QF 2019) | ADD |

If `hardiman2013` or `bacrymastromatteomuzy2015` already exist under a different key in the bib,
keep the existing key and rename the `\citep{...}` calls in §1 accordingly (do not duplicate).

---

## 4. Exact find-and-replace targets

### 4.1 `paper/main.tex` §3.11

- **Locate the subsection** by its `\subsection{...}` line and `\label{...}` for the Hawkes/SV
  block (distinctive opening words to search for, in priority order):
  - any `\subsection{...Hawkes...}` heading in §3.11;
  - the sentence introducing the dimensionless ratio, beginning roughly
    *"We define the dimensionless ratio $n_{SV} := c_0/(c_1 c_2)$ ..."*;
  - the verification sentence containing the phrase *"to machine precision"* and/or
    *"$\approx 10^{-15}$"* / *"1e-15"*;
  - the original Theorem 4 statement environment, beginning *"\begin{theorem}"* immediately
    following the Hawkes/Hopf discussion (the original claimed an *equivalence* / *iff* between
    Hawkes criticality and the Hopf bifurcation).
- **Remove** from the start of that `\subsection` through the end of the original Theorem 4 /
  its proof / the `n_{SV}` verification paragraph (i.e., the entire Hawkes–SV block).
- **Replace** with the LaTeX block in §1 above.
- The single load-bearing strings to delete (so nothing tautological survives): every occurrence
  of `n_{SV}`, `n_SV`, `c_0/(c_1 c_2)`, `machine precision`, `10^{-15}`, and any sentence
  asserting an *"iff"/"equivalence"/"if and only if"* between Hawkes criticality and the Hopf
  bifurcation. The corrected statement asserts coincidence of `n→1` with the **real-eigenvalue**
  stratum, and positions Hopf as **strictly stronger**, not equivalent.

### 4.2 `paper/theory.md` §6

- **Locate** the `## 6` heading for the Hawkes/SV section (distinctive opening words:
  any heading containing *"Hawkes"* and/or the prose introducing *"branching ratio"* /
  *"self-exciting"* / the `n_SV` ratio).
- **Remove** from that `## 6` heading through the end of the section (up to the next `## 7`
  heading, or EOF if §6 is last).
- **Replace** with the Markdown block in §2 above.
- Same load-bearing deletions: `n_SV := c0/(c1 c2)`, any "machine precision" / "1e-15"
  language, and any "iff/equivalence" phrasing between Hawkes criticality and the Hopf
  bifurcation.

---

## 5. Detectability / pre-registration note (for the structured spec)

This component is a **theory repositioning**, not a new statistical test, so there is no new
estimator to power-analyze on the event windows. The relevant detectability question is the one
the original `n_SV` framing failed: *is there an observable that distinguishes the two strata?*

- The **real-eigenvalue (Hardiman) edge** is detectable from the standard branching-ratio
  estimator `n̂` on the |r_t| / count history (Hardiman's own estimator), and is the empirically
  established quantity — `n̂ ≈ 0.8–1` for equity index futures is the headline Hardiman result.
- The **Hopf edge** would be detectable, if present, as a **spectral peak at nonzero frequency**
  in the conditional-intensity / volatility autocovariance — i.e., a complex eigenvalue pair
  shows up as a damped oscillation (quasi-period) in the autocorrelation of `|r_t|`, whereas the
  real-eigenvalue critical slowing-down shows up as a pure power-law / monotone decay with no
  intrinsic frequency. The discriminating statistic is therefore the **presence of a finite-
  frequency peak in the spectrum of the realized-volatility / signed-flow series**, versus a
  monotone (zero-frequency) critical-slowing-down spectrum.
- This discriminator is, in principle, detectable in the available windows for the *real* edge
  (the Hardiman branching ratio is estimable on ~121-trading-day event windows and far better on
  the longer |r_t| history). The *Hopf* (oscillatory) discriminator requires resolving a
  nonzero quasi-frequency in the volatility spectrum, which needs the **longer |r_t| history**,
  not the 121-day event window — and we say so. We do **not** claim the Hopf edge is realized in
  any dataset; we only specify the statistic that would detect it. This is what replaces the
  tautological 1e-15 "verification": a falsifiable spectral discriminator with a clear null.

### Synthetic validation specified (to be run by the simulator harness, ground-truth κ known)

The existing simulator already generates paths with a known coupling `κ` (the same harness that
validated the original H4). The pre-registered synthetic check for *this* repositioning is:

- **Positive control (Hopf side):** simulate the SV limit in a parameter region with a
  complex-conjugate drift-eigenvalue pair crossing the imaginary axis (κ tuned so `Im λ ≠ 0`).
  Expected: a finite-frequency peak in the `|r_t|` spectrum, and the branching-ratio estimator
  `n̂ → 1`. Confirms the Hopf edge produces an oscillation the scalar `n̂` cannot see.
- **Negative control (saddle-node / Hardiman side):** simulate with a real eigenvalue crossing
  zero (κ tuned so `Im λ = 0`). Expected: `n̂ → 1` **and no** finite-frequency spectral peak
  (monotone critical slowing-down). Confirms the real edge is the literal Hardiman analogue and
  is non-oscillatory.
- **Discriminating power:** the test statistic is the spectral peak frequency `ω̂*` (and its
  significance against a monotone-AR null via a surrogate / phase-randomization test). Power is
  the probability that `ω̂* > 0` is flagged significant on the Hopf-side sims and not flagged on
  the saddle-node-side sims, as a function of `Im λ` and window length. The accompanying new
  module (`src/reflexive_options/theory/hawkes_sv_bifurcation.py`, written separately) implements
  `classify_stratum(path) -> {"real_edge","hopf_edge","stable"}` from the spectral discriminator
  so the synthetic controls can be scored automatically.

> The two synthetic controls above are the honest replacement for the deleted "1e-15
> verification": instead of confirming a definition, they confirm that a real, observable
> statistic (the spectral discriminator) separates the two strata on simulator data with known
> ground-truth κ — and they would also flag, on the rough-kernel sims, that the oscillation does
> **not** survive as a clean limit cycle (consistent with the future-work caveat).
