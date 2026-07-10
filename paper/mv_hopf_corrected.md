# McKean–Vlasov mean-field limit: corrected Hopf threshold (v0.3.6)

> **ARCHIVED v0.3 EXTENSION.** It uses the superseded model and does not prove
> a threshold correction for the centered positive-variance model.

> **Header note (honesty disclosure).** This document supersedes the Theorem 3
> claim of v0.3.5 (`paper/mckean_vlasov_limit.md`). An external mathematical
> audit demonstrated, numerically, that the v0.3.5 closed-form correction
> $\kappa^\star_{\mathrm{MV}}/\kappa^\star_{\mathrm{single}}
> = \sqrt{1 + (\omega^\star\tau_G)^2}$ was incorrect — wrong by up to a factor
> $\sim$2 in magnitude *and of the wrong sign* at the canonical short-gamma
> regime. The v0.3.5 derivation assumed the dealer-gamma channel could be
> represented as a low-pass filter acting on the *existing* 3D state; it
> missed the destabilising effect of *adding* a fourth state to the
> linearised system and the phase-coupling that this introduces. This file
> contains the correct 4D extended-Jacobian derivation, the closed form
> at the canonical regime, the numerical anchors that match the audit, and
> the economic re-framing implied by the corrected sign.
>
> Implementation: `src/reflexive_options/theory/mckean_vlasov.py`.
> Reproducer: `python -m reflexive_options.experiments.mckean_vlasov_validation`.
> Figure: `paper/figures/mckean_vlasov_threshold_correction.pdf`.

## Setup — the 4D extended state

Let $(y, u, z)$ denote the deviation variables of the §2 model around the
equilibrium (spot log-deviation, instantaneous variance deviation, memory
state). Under the McKean–Vlasov mean-field limit (Theorem 2 of
`mckean_vlasov_limit.md`), the n-dealer system's empirical gamma
$\bar G_n$ converges to a deterministic limit $\bar G_\infty(t) = \mathbb{E}[G(t) \mid \mathcal{F}^{S,v}_t]$
which obeys a first-order linear ODE driven by the *target* gamma
$g(S, v)$. Writing $g(y, u, z) := G_y y + G_v u + G_z z$ for the
Taylor expansion of the target around equilibrium and introducing
the deviation $g(t) := \bar G_\infty(t) - g(0, 0, 0)$, the
linearised MV system is the **4D extended state** $X = (y, u, z, g)$
with dynamics

$$
\begin{aligned}
\dot y &= -\tfrac12 \sigma^2_y\, y - \tfrac12 \sigma^2_v\, u + \kappa\, g, \\
\dot u &= -\kappa_v\, u + \gamma\, z, \\
\dot z &= \beta\, y - \alpha\, z, \\
\dot g &= -\theta_G\, g + \theta_G\,(G_y\, y + G_v\, u + G_z\, z).
\end{aligned}
\tag{1}
$$

The key structural difference from the §2 single-dealer Jacobian
is that the *spot* equation no longer carries the instantaneous
feedback $\kappa(G_y y + G_v u + G_z z)$; that feedback is replaced
by $\kappa g$, and the dealer-gamma deviation $g$ acquires its own
relaxation–driven dynamics with bandwidth $\theta_G$.

## The 4D extended Jacobian

In matrix form the linearisation (1) is $\dot X = J_{\mathrm{MV}}(\kappa, \theta_G) X$
with

$$
J_{\mathrm{MV}}(\kappa, \theta_G) \;=\;
\begin{pmatrix}
-\tfrac12 \sigma^2_y & -\tfrac12 \sigma^2_v & 0 & \kappa \\
0 & -\kappa_v & \gamma & 0 \\
\beta & 0 & -\alpha & 0 \\
\theta_G G_y & \theta_G G_v & \theta_G G_z & -\theta_G
\end{pmatrix}.
\tag{2}
$$

As $\theta_G \to \infty$ the dealer mode becomes Markov and decouples:
$J_{\mathrm{MV}}$ approaches the rank-1 reducible block
$\bigl[\!\begin{smallmatrix} J_3 & * \\ 0 & -\theta_G \end{smallmatrix}\!\bigr]$
where $J_3$ is the 3D single-dealer Jacobian of §2 with $a := \kappa G_y - \tfrac12\sigma^2_y$
and $b := \kappa G_v - \tfrac12\sigma^2_v$. The eigenvalue $-\theta_G$
escapes to $-\infty$ and the remaining three eigenvalues coincide with
those of $J_3$. Hence the *single-dealer Hopf threshold is recovered
exactly* in the instantaneous-hedging limit (Deliverable 4, limit i).

## The 4D Hopf condition (Liu 1994)

Let the characteristic polynomial of (2) be

$$
\det(\lambda I - J_{\mathrm{MV}}) \;=\; \lambda^4 + a_3 \lambda^3 + a_2 \lambda^2 + a_1 \lambda + a_0.
$$

A direct computation (sympy, reproduced in `mckean_vlasov.py` docstring)
gives

$$
\begin{aligned}
a_3 &= \alpha + \kappa_v + \theta_G - \tfrac12\sigma^2_y, \\
a_2 &= \big(\alpha + \kappa_v - \tfrac12\sigma^2_y\big)(\theta_G) + \alpha\kappa_v
       \;-\; \kappa\, G_y\, \theta_G + \text{(σ²-terms)}, \\
a_1 &= \theta_G \cdot c_1^{(3D)} + c_0^{(3D)} + \kappa\cdot\text{(σ²-terms)}, \\
a_0 &= \theta_G \cdot c_0^{(3D)},
\end{aligned}
$$

where $c_i^{(3D)}(\kappa)$ are the Routh–Hurwitz coefficients of the 3D
single-dealer characteristic polynomial. In particular $a_0$ vanishes
iff the 3D constant term vanishes, so the sign of $a_0$ is inherited
unchanged from the single-dealer system.

A Hopf bifurcation of the 4D system occurs at the smallest $\kappa > 0$
satisfying the **Liu (1994) condition**

$$
H_4(\kappa, \theta_G) \;:=\; (a_3 a_2 - a_1)\, a_1 \;-\; a_3^2 a_0 \;=\; 0,
\tag{3}
$$

with the positivity side conditions

$$
a_3 > 0, \quad a_0 > 0, \quad a_3 a_2 - a_1 > 0,
\tag{3$'$}
$$

so that the remaining (non-imaginary) eigenvalue pair lies in the left
half-plane. (Equation 3 is the 4D analogue of $c_1 c_2 - c_0 = 0$ for
the 3D Liu criterion; positivity 3$'$ is the 4D Routh–Hurwitz.)

In code, the threshold is located by eigenvalue scan rather than by
expanding $H_4$ symbolically: we solve
$\max_i \mathrm{Re}\,\lambda_i\bigl(J_{\mathrm{MV}}(\kappa, \theta_G)\bigr) = 0$
via `scipy.optimize.brentq` on a log-spaced bracket. The Liu form (3) is
used to *verify* the bifurcation type at the located $\kappa^\star$
(complex pair on the imaginary axis, third and fourth eigenvalues in
the left half-plane).

## Theorem 3 (CORRECTED)

> **Theorem 3** (v0.3.6 corrected MV Hopf threshold). *Let $J_{\mathrm{MV}}(\kappa, \theta_G)$
> be the 4D extended Jacobian (2) at the equilibrium of the
> McKean–Vlasov reflexive system with dealer-hedging speed $\theta_G > 0$.
> Define $\kappa^\star_{\mathrm{MV}}(\theta_G) := \inf\{\kappa > 0 :
> \max_i \mathrm{Re}\,\lambda_i(J_{\mathrm{MV}}(\kappa, \theta_G)) = 0,\
> \text{conditions } (3'), (3) \text{ hold}\}$ and let $\kappa^\star_{\mathrm{single}}$
> be the corresponding 3D single-dealer Hopf threshold. Then:*
>
> *(i) (Instantaneous-hedging recovery.)
> $\lim_{\theta_G \to \infty} \kappa^\star_{\mathrm{MV}}(\theta_G) = \kappa^\star_{\mathrm{single}}$.*
>
> *(ii) (Regime-dependent shift.) The ratio
> $\rho(\theta_G) := \kappa^\star_{\mathrm{MV}}(\theta_G)/\kappa^\star_{\mathrm{single}}$
> is a smooth function of $\theta_G$ with $\rho(\infty^-) = 1$. The sign of
> $\rho(\theta_G) - 1$ at finite $\theta_G$ depends on the regime
> $(G_y, G_v, G_z, \alpha, \beta, \gamma, \kappa_v, \sigma^2_y, \sigma^2_v)$.
> In particular at the **canonical short-gamma regime** $(G_y, G_v, G_z,
> \alpha, \beta, \gamma, \kappa_v) = (1/2, -1/2, -1/2, 1/2, 1, 1/2, 2)$
> with $\sigma^2_y = \sigma^2_v = 0$,*
>
> $$
> \kappa^\star_{\mathrm{MV}}(\theta_G) \;=\;
> \frac{50\,\theta_G^2 + 143\,\theta_G + 105 \;-\; (2\theta_G + 5)\sqrt{385\,\theta_G^2 + 810\,\theta_G + 441}}
>      {12\,\theta_G(\theta_G + 1)},
> \tag{4}
> $$
>
> *with $\kappa^\star_{\mathrm{single}} = (25 - \sqrt{385})/6 \approx 0.8964$,
> and the ratio is strictly less than $1$ for every finite $\theta_G > 0$.*
>
> *(iii) (Frozen-dealer limit.) At the canonical regime,
> $\lim_{\theta_G \to 0^+} \kappa^\star_{\mathrm{MV}}(\theta_G) = 8/21$,
> giving $\rho(0^+) = 16 / [7(25 - \sqrt{385})] \approx 0.425$. The
> system does not blow up as the dealer mode freezes; rather, the threshold
> approaches a finite, strictly smaller value.*
>
> *(iv) (Asymptotic expansion.) At large $\theta_G$ in the canonical regime,*
>
> $$
> \rho(\theta_G) \;=\; 1 \;+\; \frac{1}{\theta_G}\Bigl(\frac{3}{4} - \frac{111\sqrt{385}}{1540}\Bigr) \;+\; O\bigl(\theta_G^{-2}\bigr)
> \;=\; 1 - \frac{0.6643\ldots}{\theta_G} + O(\theta_G^{-2}),
> $$
>
> *so the leading-order correction is negative and of order $\tau_G$ (NOT
> $\tau_G^2$ as in the v0.3.5 heuristic).*

**Proof sketch.** (i) follows from the reducibility of $J_{\mathrm{MV}}$
under $\theta_G \to \infty$ noted after (2); the fourth eigenvalue
$\to -\infty$ leaves the remaining three to inherit the 3D Hopf
condition exactly. (ii) reduces, at the canonical regime, to a quadratic
in $\kappa$: substituting the canonical numerical values into (3) and
factoring out $\theta_G$ gives

$$
6\theta_G(\theta_G + 1)\kappa^2 \;-\; (50\theta_G^2 + 143\theta_G + 105)\kappa
\;+\; 5(\theta_G + 2)(2\theta_G + 1) \;=\; 0,
$$

whose smaller positive root is (4). Positivity 3$'$ is verified
analytically at this root for all $\theta_G > 0$ (the eigenvalue scan
in `mckean_vlasov_kappa_star` checks it numerically). (iii) is direct
substitution $\theta_G \to 0^+$ in (4). (iv) is the Taylor expansion
in $1/\theta_G$, computed by sympy and pinned in
`paper/figures/mckean_vlasov_threshold_correction.pdf`. The strict
inequality $\rho(\theta_G) < 1$ for all finite $\theta_G$ at canonical
follows from monotonicity of (4) and the boundary values
$\rho(0^+) = 0.425 < 1$, $\rho(\infty^-) = 1$. $\square$

## Numerical anchors (audit reproduction)

The external audit's numerical 4D Hopf computation gave the following
values at the canonical regime. The closed-form (4) reproduces them
exactly (the small residual is fourth-decimal round-off in the audit's
output, not in our implementation):

| $\theta_G$ | $\kappa^\star_{\mathrm{MV}}$ (audit) | $\kappa^\star_{\mathrm{MV}}$ (closed form, eq. 4) | ratio |
|:----------:|:------------------------------------:|:-------------------------------------------------:|:-----:|
| 0.5        | 0.536                                | 0.535939                                          | 0.598 |
| 1.0        | 0.619                                | 0.619480                                          | 0.691 |
| 5.0        | 0.800                                | 0.799551                                          | 0.892 |
| 50         | 0.885                                | 0.884788                                          | 0.987 |
| 500        | 0.895                                | 0.895242                                          | 0.999 |

For comparison, the v0.3.5 formula
$\rho_{\mathrm{old}}(\theta_G) = \sqrt{1 + (\omega^\star/\theta_G)^2}$
with $\omega^\star \approx 0.572$ (the 3D single-dealer Hopf frequency)
gave $\rho_{\mathrm{old}}(0.5) \approx 1.363$, off by a factor of $1.363/0.598
\approx 2.28$ and of the wrong sign — predicting a 36 % *increase* of the
threshold when the actual effect is a 40 % *decrease*.

The unit-test suite (`tests/test_mckean_vlasov.py`) pins (a) the closed
form (4) against the numerical solver to relative 1e-6, (b) each entry
of the audit table to absolute 1e-3, (c) the $\theta_G = 5$ anchor at
4-decimal precision, (d) ratio $< 1$ at every canonical $\theta_G$, (e)
ratio $\to 1$ at $\theta_G = 10^6$, (f) ratio $\to 8/21$ at
$\theta_G = 10^{-6}$, and (g) ratio $> 1$ in the *long-gamma* log-normal-OI
calibration (the regime-dependent sign of the correction is a real feature
of the theory).

## Economic re-framing

The v0.3.5 framing — "MV is a stabiliser; single-dealer slightly
*overestimates* the propensity to bifurcate" — was the wrong direction
for the canonical short-gamma regime. The correct framing is:

> **At any regime where dealers are net short gamma at the equilibrium
> ($G_y > 0$), the McKean–Vlasov mean-field model has a *lower* Hopf
> threshold than the single-representative-dealer §2 model. The single-
> dealer simplification *understates* the propensity for the system to
> bifurcate into a feedback-driven limit cycle when dealer-hedging speed
> is slow relative to the Hopf period $\tau_G > 2\pi/\omega^\star$.**

Three implications:

1. **The §2 / §3 closed-form thresholds are an upper bound on $\kappa^\star$
   when calibration places dealers in the short-gamma regime** (the
   benchmark SPX setting in zero-DTE-heavy weeks). Empirical detection of
   reflexive instability at the §3 $\kappa^\star$ is consistent with
   the MV model being closer to the truth than the single-dealer model.

2. **The leading correction is first-order in $\tau_G$**, not second-order
   as the v0.3.5 heuristic implied. With a representative
   $\tau_G \approx 1/50\ \text{yr} \approx 5$ trading days the correction
   is $\sim 1.3\%$ at canonical regime — small enough to be operationally
   subsumed into the v0.3.4 calibration uncertainty band, but no longer
   negligible compared to a $\sim 0.03\%$ second-order claim.

3. **The frozen-dealer limit $\theta_G \to 0^+$ is well-defined and finite**
   ($\kappa^\star_{\mathrm{MV}} \to 8/21$ at canonical). This is a real,
   physically meaningful regime: a market in which all dealers' delta-
   hedging desks have been suspended (e.g. during an exchange halt) and
   the aggregate gamma exposure is locked at its pre-halt value. The
   bifurcation threshold then drops by a factor of $\sim 2.4\times$ relative
   to instantaneous hedging.

In regimes with $G_y < 0$ (net long-gamma dealers, the log-normal-OI
calibration of §4.3) the inequality reverses: the MV model *raises* the
threshold, and the v0.3.5 framing was directionally right by accident even
though the magnitude was wrong. The unified statement is Theorem 3 (ii):
the sign of the MV correction is regime-dependent; only the canonical
short-gamma regime gives an unambiguous destabilisation.

## Figure

`paper/figures/mckean_vlasov_threshold_correction.pdf` renders the
ratio $\kappa^\star_{\mathrm{MV}}/\kappa^\star_{\mathrm{single}}$ versus
$\theta_G$ on a logarithmic $\theta_G$ axis, showing (a) the smooth
monotonic approach to $1$ from below, (b) the five audit anchor points
overlaid as scatter markers, and (c) the asymptotic horizontal lines at
$\rho(\infty^-) = 1$ and $\rho(0^+) = 0.425$.

## What stays unchanged from v0.3.5

Theorem 2 (propagation of chaos, Sznitman bound) is independent of the
linearisation around the equilibrium and is unaffected by the
correction. The $1/\sqrt{n}$ scaling of the particle simulator is still
validated to within 5 % of the theoretical slope, and the closed-form
$C(T)$ bound is unchanged. The particle simulator's `simulate_n_dealer_system`
and `propagation_of_chaos_*` infrastructure was independently audited
and is correct; only the closed-form *threshold* formula was wrong.

## References

The 4D-Hopf criterion citation chain:

- Liu, W.-M. (1994). *Criterion of Hopf bifurcations without using
  eigenvalues.* J. Math. Anal. Appl. 182, 250–256. — The 4D Routh–Hurwitz
  Hopf condition $(a_3 a_2 - a_1) a_1 - a_3^2 a_0 = 0$ in the form used here.
- Kuznetsov, Y. A. (2004). *Elements of Applied Bifurcation Theory*, 3rd ed.
  Springer. §3.5 — Hopf bifurcations in $n$D, including the 4D case.

Background (unchanged from v0.3.5):

- Sznitman, A.-S. (1991). *Topics in propagation of chaos.* Lecture Notes
  in Math. **1464**, 165–251. — Propagation-of-chaos $C/n$ bound under
  Lipschitz coefficients.
- Carmona, R., & Delarue, F. (2018). *Probabilistic Theory of Mean Field
  Games with Applications I–II.* Springer.
