# Mixture-of-K-lognormals generalisation of the closed-form Hopf threshold

*Extension to paper §3.5 / §4.3. Suitable for integration into `main.tex`
either as a new subsection (3.5.X) or as an appendix; the boxed equations
below are LaTeX-clean.*

## Motivation

The §3.5 closed form
$$
\kappa^\star = \frac{G_y A^2 - G_v L - \sqrt{(G_v L - G_y A^2)^2 - 4 G_y^2 A (MA - L/2)}}{2 G_y^2 A}
\quad (\text{Eq.~18})
$$
assumes a single log-normal OI density $q(\log K) = \mathcal{N}(\mu_q, \sigma_q^2)$.
The §3.6 robustness analysis (`paper/kappa_star_robustness.md`) shows that
empirical OI multi-modality breaks this assumption hard: a symmetric bimodal
density with components at $\mu_q \pm \Delta/2$ produces $|\Delta\kappa^\star /
\kappa^\star_{\mathrm{true}}|$ of $0.23\%$ at $\Delta = 0.05$,
$4.83\%$ at $\Delta = 0.10$, and $119\%$ at $\Delta = 0.20$.

Empirical SPX OI grids are *structurally* multi-modal:

- ATM-strike concentration from gamma-scalping flow;
- Round-strike concentration ($K \in \{4500, 4600, 4700, \dots\}$);
- Calendar-near-expiry concentration from option-rolling flow;
- Dealer-portfolio-driven OTM-tail concentration from hedging.

A $\Delta \in [0.05, 0.20]$ separation between two such concentrations is the
empirical regime that breaks the single-lognormal closed form. The mixture
generalisation below closes that gap to $< 0.05\%$ relative error at every
tested $\Delta$, including $\Delta = 0.30$ where the single-lognormal closed
form is off by a factor of three.

## The mixture aggregator

Replace the single-lognormal $q$ by a $K$-component mixture
$$
q(\log K) = \sum_{k=1}^{K} w_k \, \mathcal{N}(\log K; \mu_k, \sigma_k^2),
\qquad w_k \geq 0,\ \sum_k w_k = 1.
$$

The dealer-gamma aggregator (Eq.~14) is a linear functional of $q$:
$$
G(a, v) = \kappa_u \int q(\log K) \, \Gamma_{\mathrm{BS}}(S, K, T_{\mathrm{eff}}, \sqrt{v}) \, d\log K.
$$
Linearity gives
$$
\boxed{\;G^{\mathrm{mix}}(a, v) = \sum_{k=1}^{K} w_k \, G_k(a, v),\;}
$$
where each $G_k$ is the single-lognormal closed form (Eq.~15a) with its own
$(\mu_k, \sigma_k)$. By multilinearity of differentiation, the same identity
holds for every partial of $G$:
$$
\partial^{|\alpha|} G^{\mathrm{mix}} / \partial x^\alpha
= \sum_{k=1}^{K} w_k \, \partial^{|\alpha|} G_k / \partial x^\alpha
\qquad \forall \alpha.
$$
In particular $(G^{\mathrm{mix}}_y, G^{\mathrm{mix}}_v) = \sum_k w_k (G_{y,k}, G_{v,k})$.

## Closed-form $\kappa^\star$ for the mixture

The Routh-Hurwitz coefficients (Eq.~16) are *linear* in $(G_y, G_v)$, so
the Hopf-threshold quadratic
$$
H(\kappa) = G_y^2 A \, \kappa^2 + (G_v L - G_y A^2) \, \kappa + (MA - L/2) = 0
$$
goes through unchanged when $(G_y, G_v) \mapsto (G^{\mathrm{mix}}_y, G^{\mathrm{mix}}_v)$.
The mixture Hopf threshold is therefore
$$
\boxed{\;\kappa^{\star,\mathrm{mix}} = \frac{G^{\mathrm{mix}}_y A^2 - G^{\mathrm{mix}}_v L
        - \sqrt{(G^{\mathrm{mix}}_v L - G^{\mathrm{mix}}_y A^2)^2
        - 4 (G^{\mathrm{mix}}_y)^2 A \, (MA - L/2)}}
       {2 (G^{\mathrm{mix}}_y)^2 A},\;}
$$
which is Eq.~18 evaluated at the mixture-aggregated partials.
The frequency $\omega^{\star,\mathrm{mix}} = \sqrt{-\kappa^{\star,\mathrm{mix}}
G^{\mathrm{mix}}_y A + M}$ follows from $\omega^{\star 2} = c_1$ exactly as in
the single-lognormal case.

## First Lyapunov coefficient $\ell_1^{\mathrm{mix}}$

The Kuznetsov formula (Eq.~19) involves
$\langle p, B(\bullet, \bullet)\rangle$ and $\langle p, C(\bullet, \bullet,
\bullet)\rangle$ where $B$, $C$ are the bilinear / trilinear Taylor tensors of
the drift around the equilibrium. Only the $f_1$ row contributes, and that row
is $\kappa^\star \cdot G$. By multilinearity of $G$, the mixture $B$, $C$ are
linear combinations of the single-component $B_k$, $C_k$:
$$
B^{\mathrm{mix}} = \sum_k w_k B_k,
\qquad C^{\mathrm{mix}} = \sum_k w_k C_k.
$$
The eigenvectors $p, q$ at $\pm i\omega^{\star,\mathrm{mix}}$, however, are
computed at the mixture Jacobian $J(\kappa^{\star,\mathrm{mix}}; G^{\mathrm{mix}}_y,
G^{\mathrm{mix}}_v)$, which itself depends nonlinearly on the $\{w_k\}$ through
the Hopf-threshold quadratic. The Kuznetsov contractions are then quadratic
forms in the $w$-mixture tensors against $p, q$-vectors that are themselves
$w$-dependent; the resulting $\ell_1^{\mathrm{mix}}$ is a rational function
of $(\{w_k, \mu_k, \sigma_k\}_{k=1}^K, \kappa_v, \alpha, \beta, \gamma)$
whose degree is bounded by the number of Kuznetsov terms ($O(K^2)$ in the
mixture-tensor contractions, $O(K^4)$ from the eigenvector polynomials).

For $K = 2$ the symbolic expression is roughly 3–5× longer than the
single-lognormal $\ell_1$; for $K \geq 3$ the symbolic form gets unwieldy but
numerical evaluation remains O(K) per call via
`G_mixture_lognormal_oi_partials` plus a single
`compute_lyapunov_coefficient` invocation. See
`notebooks/closed_form_ell1_derivation.py` step 9 for the symbolic
construction and the K=1 limit verification.

## Numerical robustness comparison

At the canonical regime $(\kappa_v = 2, \alpha = 0.05, \beta = 1, \gamma = 1,
T_{\mathrm{eff}} = 0.25, v^\star = 0.04, \mu_q = \log 100, a^\star = \log 100)$,
with a symmetric bimodal mixture at $\mu_q \pm \Delta/2$ with equal weights
and component spread $\sigma_{\mathrm{comp}} = 0.07$:

| $\Delta$ | $\kappa^\star_{\mathrm{true}}$ (FD) | $\kappa^\star$ K=1 (single) | $\kappa^\star$ K=2 (mixture) | K=1 rel.err | K=2 rel.err |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 25.142 | 25.201 | 25.139 | 0.232%   | 0.013% |
| 0.10 | 20.864 | 21.871 | 20.862 | 4.827%   | 0.010% |
| 0.20 |  5.677 | 12.460 |  5.677 | 119.482% | 0.008% |
| 0.30 |  2.458 |  7.809 |  2.458 | 217.733% | 0.007% |

The K=2 mixture closed form is essentially exact across the entire range —
the residual $\sim 10^{-4}$ relative error comes from the numerical FD step
in the *reference* pipeline, not from the closed form itself. Numerical
output: `runs/mixture_oi_robustness/<ts>/metrics.json`. Figure:
`paper/figures/mixture_oi_robustness_curve.pdf`.

## §3.6 robustness gap closure

The §3.6 calibration tolerance under the single-lognormal assumption was
*"$\pm 10\%$ on $\kappa^\star$ relaxes to $\sigma_q$ to $\pm 4.47\%$,
$\mu_q$ to $\pm 5 \times 10^{-4}$ log-strike"*. The $\mu_q$ tolerance was the
binding constraint, corresponding to centring the empirical OI mean on ATM
to within $\sim 5$bp.

Under the mixture closed form, the relevant calibration parameter is no longer
$\mu_q$ alone but the full $\{(\mu_k, \sigma_k, w_k)\}_{k=1}^K$ tuple. The
binding constraint becomes the *component weights*: a misallocation of mass
between the K=2 wings produces $O(\Delta) \cdot \delta w$ shift in
$G^{\mathrm{mix}}_y$ (vs the $O(1) \cdot \delta\mu_q$ shift in the single-mode
case). At $\Delta = 0.10$ this is a $\sim 20\times$ relaxation; the
empirical-OI weight estimation is the practical bottleneck and is achievable
with $\pm 5\%$ accuracy on a $5$-strike-wide ladder per cluster.

## When K=2 is enough

Empirical SPX OI typically exhibits 2–3 dominant concentrations within any
single expiry: ATM + next-round + (optionally) calendar-roll. K=2 handles the
first two; K=3 captures all three with negligible additional cost (the
$G_{\mathrm{mix}}$ evaluator is O(K) per call). The closed form remains
numerically stable at K up to $\sim 10$ in our tests. Beyond that — full
non-parametric OI — the brute-force `kappa_star_brute_force_from_G` pipeline
is the operational fallback, with negligible relative-error penalty.

## Implementation

- `src/reflexive_options/theory/bifurcation.py`: `MixtureOIComponent`,
  `G_mixture_lognormal_oi`, `G_mixture_lognormal_oi_partials`,
  `kappa_star_mixture_lognormal_oi`,
  `lyapunov_coefficient_mixture_lognormal_oi`.
- `src/reflexive_options/experiments/mixture_oi_robustness.py`: the
  $\Delta$-sweep robustness curve runner.
- `notebooks/closed_form_ell1_derivation.py` steps 8–9: symbolic K=2
  construction, multilinearity verification, K=1-limit equivalence test.
- `tests/test_mixture_oi.py`: K=1 equivalence (machine precision),
  weight-normalisation invariance, component-order invariance, K=2 vs FD
  tolerance, bimodal robustness regression, K=3 stability.
- `paper/figures/mixture_oi_robustness_curve.pdf`: K=1 vs K=2 vs K=3 relative
  error vs $\Delta$.

## Conclusion

The mixture-of-K-lognormals generalisation is the structurally correct fix
to the §3.6 fragility. It preserves the closed-form $\kappa^\star$ structure
of Eq.~18 verbatim (only the inputs $G_y$, $G_v$ change), gives a near-exact
Hopf threshold at empirical-magnitude bimodality, and reduces to the §3.5
single-lognormal result to machine precision in the $K = 1$ limit. The
operational consequence is that Phase 4 can calibrate against an empirical OI
grid with $\leq K = 3$ clusters without needing to invoke the brute-force
numerical-Jacobian pipeline, retaining the closed-form $\sim 50\,\mu\mathrm{s}$
per-cell evaluation cost.
