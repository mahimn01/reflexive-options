# The $\kappa$ rescaling map: per-USD coupling vs. the dimensionless Hopf threshold

**Status.** Pre-data design note resolving the unit-chain incoherence (blocking
issue #2). LaTeX-ready; intended to *replace* any text that compares the
dimensionless $\kappa^\star = 17.81$ directly against the per-USD $\kappa_0 \sim
5\times10^{-12}$. Implementation: `src/reflexive_options/theory/kappa_rescaling.py`;
tests: `tests/test_kappa_rescaling.py`.

---

## 1. The incoherence

Two distinct quantities in the paper are both written "$\kappa$":

1. **$\kappa_0$ (per-USD)** — the empirically-tuned reflexive coupling, with
   literature prior $\kappa_0 \in [10^{-12}, 10^{-11}]$ **per USD of dealer
   dollar-gamma** (theory.md 7.1, 6.5; GPP 2009 calibration). Anchor value
   $\kappa_0 \approx 5\times10^{-12}\ \mathrm{USD}^{-1}\,\mathrm{yr}^{-1}$.

2. **$\kappa^\star$ (dimensionless)** — the closed-form Hopf threshold
   $\kappa^\star = 17.8065068$ (theory.md 4.3.5), derived in a regime where the
   dealer-gamma functional $G$ is normalised to be $O(1)$ (the $\{G_x,G_v,G_z\} =
   \{0.5,-0.5,-0.5\}$ / $\sigma_q=0.10$ canonical specification).

The drift term in the SDE (theory.md Eq. 1a) is

$$
\text{drift}_S \;=\; \kappa\, G(S,z,v), \qquad [\text{drift}] = \mathrm{yr}^{-1}.
$$

For this to be dimensionally consistent, $\kappa$ and $G$ must be *conjugate*:
if $G$ carries USD-gamma units (the aggregator
`GammaAggregator.compute` returns $G$ in USD-of-dollar-gamma-per-unit-return,
magnitude $\sim 10^{9}$–$10^{13}$), then $\kappa$ must be per-USD; if $G$ is
nondimensionalised to $O(1)$, then $\kappa$ is dimensionless. **The original H4
silently evaluated $\omega^\star = \sqrt{c_1(\kappa^\star)}$ "at the calibrated
$\kappa_0$", chaining a per-USD number into a dimensionless formula** — a
category error spanning $\sim$12 orders of magnitude
($\log_{10}(\kappa^\star/\kappa_0) \approx 12.55$). That naive distance is
meaningless; it is an artifact of mixing unit systems, not a physical statement
about how far the market is from the Hopf.

## 2. Dimensional analysis of the drift term

Write $G$ in USD-gamma units as it comes out of the aggregator. From
`gamma_aggregator.py`, for an open-interest grid $\{q_{KT}\}$,

$$
G(S,v) \;=\; m \cdot S^2 \sum_{K,T} s_{KT}\, q_{KT}\,
\Gamma_{\mathrm{BS}}(S,K,T,\sqrt{v}),
\tag{1}
$$

where $m = 100$ is the contract multiplier, $s_{KT}\in\{+1,-1\}$ the dealer sign,
and $\Gamma_{\mathrm{BS}}$ the per-share Black–Scholes gamma (units
$\mathrm{shares}\,\mathrm{USD}^{-1}$, i.e. $1/S$ scaling). The leading $S^2$ comes
from the per-share-gamma $\to$ dollar-gamma conversion
(`g_shares_per_dollar * spot * spot` in `gamma_aggregator.py`). Hence

$$
[G] = \mathrm{USD}, \qquad [\kappa_0] = \mathrm{USD}^{-1}\mathrm{yr}^{-1},
\qquad [\kappa_0\, G] = \mathrm{yr}^{-1}. \checkmark
$$

The magnitude of $G$ is governed by

$$
G \;\sim\; m\, S^2 \cdot (\text{total OI}) \cdot \overline{\Gamma}_{\mathrm{BS}},
\qquad \overline{\Gamma}_{\mathrm{BS}} \sim \frac{1}{S\,\sigma\sqrt{2\pi T_{\mathrm{eff}}}},
\tag{2}
$$

so **$G$ scales as $S^{1}\cdot \text{OI}$ on the level**: the explicit $S^2$ from the
dollar-gamma conversion is partly cancelled by the $1/S$ inside
$\overline{\Gamma}_{\mathrm{BS}}$, leaving one net power of $S$ (the log-moneyness
*shape* of the OI grid is scale-invariant in log-spot, so only this overall factor
survives). This is verified numerically — a decade in $S_0$ produces a $\times 10$
change in $G_{\mathrm{char}}$, not $\times 100$
(`test_characteristic_scale_grows_linearly_in_spot`). The *characteristic scale*
$G_{\mathrm{char}}$ therefore still swings by orders of magnitude with the assumed
$S_0$ and total open interest.

## 3. The change of variables

Define the dimensionless aggregator and dimensionless coupling

$$
\widetilde{G} \;:=\; \frac{G}{G_{\mathrm{char}}}, \qquad
\boxed{\;\kappa_{\mathrm{dimless}} \;:=\; \kappa_0 \cdot G_{\mathrm{char}}.\;}
\tag{3}
$$

The drift is invariant by construction:

$$
\kappa_0\, G \;=\; (\kappa_0 G_{\mathrm{char}})\,(G/G_{\mathrm{char}})
\;=\; \kappa_{\mathrm{dimless}}\,\widetilde{G}.
$$

(Verified to machine precision, worst relative residual $7.3\times10^{-16}$,
`test_change_of_variables_preserves_drift_exactly`.) The inverse map is
$\kappa_0 = \kappa_{\mathrm{dimless}}/G_{\mathrm{char}}$.

**Identifying $G_{\mathrm{char}}$ from the simulator.** $G_{\mathrm{char}}$ is read
straight off the same `GammaAggregator` the simulator uses, as the peak gamma
magnitude in a $\pm15\%$ log-spot neighbourhood of the equilibrium,

$$
G_{\mathrm{char}} \;=\; \max_{|x|\le 0.15} \bigl| G(S^\star e^x, v^\star) \bigr|,
\tag{4}
$$

(`characteristic_gamma_scale`). The peak — rather than the ATM point value — is
used because $G_y$ is small and sign-changing near ATM (theory.md 4.3.6), so the
ATM level is a poor scale; the peak is stable.

## 4. The empirical-vs-dimensionless comparison (and its indeterminacy)

Evaluating (3) requires $G_{\mathrm{char}}$, which depends on the *unmeasured*
SPX open interest and spot. The table below sweeps plausible assumptions on the
flat OI fixture of theory.md 7.1 ($7\times3$ grid, $\mathrm{IV}=0.20$,
$m=100$), with $\kappa_0 = 5\times10^{-12}$:

$G_{\mathrm{char}}$ is the peak-over-scan value (Eq. 4), exactly as returned by
`flat_oi_characteristic_scale`:

| $S_0$ | OI / cell | $G_{\mathrm{char}}$ (USD) | $\kappa_{\mathrm{dimless}} = \kappa_0 G_{\mathrm{char}}$ | $\kappa_{\mathrm{dimless}}/\kappa^\star$ |
|---:|---:|---:|---:|---:|
| 100   | $5\times10^4$ | $3.14\times10^{10}$ | $1.57\times10^{-1}$ | $8.8\times10^{-3}$ |
| 1000  | $5\times10^4$ | $3.14\times10^{11}$ | $1.57\times10^{0}$  | $8.8\times10^{-2}$ |
| 1000  | $2\times10^6$ | $1.26\times10^{13}$ | $6.29\times10^{1}$  | $3.53\times10^{0}$ |
| 5000  | $5\times10^4$ | $1.57\times10^{12}$ | $7.86\times10^{0}$  | $4.41\times10^{-1}$ |
| 5000  | $2\times10^6$ | $6.29\times10^{13}$ | $3.14\times10^{2}$  | $1.77\times10^{1}$ |

(Reproduced exactly by `theory.kappa_rescaling.flat_oi_characteristic_scale`;
$\kappa^\star = 17.81$.)

The dimensionless image of $\kappa_0$ spans **more than three orders of
magnitude** ($\sim 1.6\times10^{-1}$ to $\sim 3.1\times10^{2}$) under modelling
choices that are unconstrained until Phase-4 data lands. Crucially,
$\kappa^\star = 17.81$ sits *inside* this band. A representative "realistic SPX"
point ($S_0=5000$, total chain OI a few $\times10^{6}$ per cell-equivalent
$\Rightarrow$ ATM dollar-gamma $\sim$\$ tens of bn / 1\% move) gives
$\kappa_{\mathrm{dimless}}$ ranging from $\sim 8$ (sparse OI) to $\sim 314$ (dense
OI), i.e. anywhere from a factor of $\sim 2$ *below* to an order of magnitude
*above* $\kappa^\star$ — while a paper-fixture point ($S_0=100$) gives
$\kappa_{\mathrm{dimless}} \approx 0.16$, two decades *below* $\kappa^\star$. The
position relative to $\kappa^\star$ flips the inequality across the plausible
range; it is genuinely undetermined.

## 5. Honest conclusion (this replaces the "market sits near the Hopf" claim)

> **The dimensionless threshold $\kappa^\star = 17.81$ does NOT establish that the
> SPX options market sits near a Hopf bifurcation.** The position of the empirical
> per-USD coupling relative to $\kappa^\star$ is *indeterminate* at the pre-data
> stage: it depends on the characteristic dealer-gamma scale $G_{\mathrm{char}}$,
> which scales as $\sim S_0 \times (\text{total SPX OI})$ and is unknown until the
> Phase-4 WRDS/OptionMetrics chain is calibrated. The plausible range of the
> dimensionless image of $\kappa_0$ brackets $\kappa^\star$ across roughly three
> orders of magnitude. We therefore make **no** claim of proximity to the Hopf on
> the basis of $\kappa^\star$ alone.**

What *would* place the market is the **empirical GEX regression (H1$'$)**:
calibrate $G(\cdot)$ from the empirical SPX OI grid (fixing $G_{\mathrm{char}}$,
hence $\kappa_{\mathrm{dimless}}$ via Eq. 3), then independently estimate the
realised per-USD coupling $\widehat{\kappa}_0$ from the GEX$\to$realised-vol
feedback regression on event windows. Proximity to the Hopf is the *jointly
data-determined* statement

$$
\widehat{\kappa}_0 \cdot G_{\mathrm{char}}^{\,\mathrm{emp}} \;\approx\; \kappa^\star,
$$

with both factors estimated from data and propagated error bars. Until then, the
operative threshold for any empirical statement is the **closed-form $\kappa^\star$
from theory.md Eq. 17 evaluated on the empirical OI** (which is itself in
dimensionless units conjugate to a calibrated $G_{\mathrm{char}}$), *not* the
canonical $\kappa^\star = 17.81$, and *not* a direct comparison to
$\kappa_0 = 5\times10^{-12}$.

This is consistent with theory.md 4.2/6.5, which already flag that the
empirically-tuned config "does not Hopf within the literature $\kappa$ range" and
defer the sign/position questions to the empirical phase; the present note makes
the *unit reconciliation* explicit and quantifies the indeterminacy rather than
asserting "near but not across $\kappa^\star$".

## 6. What the pre-registration must commit to

1. **Report $\kappa$ in two explicitly-labelled unit systems.** Every $\kappa$ in
   the paper is tagged either "per-USD" ($\mathrm{USD}^{-1}\mathrm{yr}^{-1}$) or
   "dimensionless", with the map (3) stated. No cross-system comparison without
   passing through $G_{\mathrm{char}}$.
2. **Pre-commit $G_{\mathrm{char}}$ as a *measured* Phase-4 quantity**, read off
   the empirical OI grid via `characteristic_gamma_scale` at the calibrated
   $(S^\star, v^\star)$. The dimensionless image of $\widehat\kappa_0$ is reported
   as a band reflecting the OI calibration uncertainty
   (`dimensionless_kappa_band`).
3. **Drop the dimensionless $\kappa^\star=17.81$ as evidence of market
   proximity.** It remains the headline *theoretical* threshold for the canonical
   parameter set; it is not evidence about real SPX.
4. **The proximity claim is decided by H1$'$**, not by the bifurcation theory:
   the empirical test is $\widehat\kappa_0\,G_{\mathrm{char}}^{\mathrm{emp}}$ vs.
   the empirical-OI closed-form $\kappa^\star$, on event-window-matched data.

## 7. Detectability

The rescaling map itself is a deterministic, exact algebraic identity — there is
nothing statistical to detect, and it is validated to machine precision on
synthetic OI grids (positive control: drift invariance to $7\times10^{-16}$;
negative control: the naive no-rescale comparison is shown to be $\sim$12 decades
spurious, `test_naive_no_rescale_comparison_is_meaningless`). The *empirical*
quantity that becomes detectable once data lands is $\widehat\kappa_0$ from the
GEX regression; its detectability is the subject of the H1$'$ power analysis on
the $\sim$121-trading-day event windows, not of this unit-chain note.
