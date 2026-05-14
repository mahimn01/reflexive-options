# McKean–Vlasov mean-field limit of the dealer-gamma channel

> Self-contained writeup for integration into `paper/main.tex` as **§4 (after the codim-2 §3.6)** or **Appendix B**.
> All numerics produced from `runs/mckean_vlasov_validation/` and `paper/figures/mckean_vlasov_propagation_chaos.pdf`.
> Implementation: `src/reflexive_options/theory/mckean_vlasov.py`. Reproducer: `python -m reflexive_options.experiments.mckean_vlasov_validation`.

## Setup — the $n$-dealer system

§§2–3 treated the aggregate dealer gamma $G(S, t)$ as if it came from a single representative market-maker. Real markets have $n \sim 10^2$–$10^3$ dealers, each holding their own portfolio and hedging on their own clock. Dealer $i \in \{1, \ldots, n\}$ holds a deviation gamma $G_i$ that obeys an OU-style relaxation toward a common target $g(S, v)$ (e.g.\ the closed-form log-normal-OI aggregator of §4.3 evaluated at the current spot/variance) plus idiosyncratic noise:

$$
dG_i \;=\; -\theta_G\,(G_i - g(S, v))\,dt \;+\; \sigma_G\,dW^i_G,\qquad i = 1, \ldots, n, \tag{1}
$$

with $\{W^i_G\}_{i=1}^n$ independent standard Brownian motions, $\theta_G > 0$ the dealer-hedging speed, and $\sigma_G \geq 0$ the idiosyncratic noise scale. We write $\tau_G := 1/\theta_G$ for the autocorrelation time of the gamma deviation. The aggregate fed back into spot is the empirical mean $\bar G_n(t) := (1/n)\sum_{i=1}^n G_i(t)$:

$$
\frac{dS_t}{S_t} \;=\; \bigl(\mu + \kappa\,\bar G_n(t)\bigr)\,dt \;+\; \sigma(S_t, v_t)\,dW^S_t. \tag{2}
$$

The variance and memory equations (1b)–(1c) are unchanged.

## Theorem (propagation of chaos)

In the limit $n \to \infty$ the empirical measure $\bar\mu_n^t = (1/n)\sum_i \delta_{G_i^t}$ converges weakly to the deterministic measure $\mu^t = \mathrm{Law}(G^t)$ where $G$ solves the McKean–Vlasov SDE

$$
dG \;=\; -\theta_G\,(G - g(S, v))\,dt \;+\; \sigma_G\,dW_G,\qquad \bar G_\infty(t) \;:=\; \int g\,d\mu^t(g) \;=\; \mathbb{E}[G(t) \mid \mathcal{F}_t^{S,v}],\tag{3}
$$

coupled with $dS/S = (\mu + \kappa\,\bar G_\infty)\,dt + \sigma\,dW^S$.

> **Theorem 3** (Sznitman 1991 Théorème I.1.4 / Méléard 1996 Prop 2.5 / Carmona–Delarue 2018 Vol I Thm 2.12). *Assume $g(\cdot)$ is Lipschitz in $(S, v)$, the initial conditions $G_i^0$ are i.i.d.\ with $\mathrm{Var}(G_0) < \infty$, and the spot/variance path is fixed (i.e.\ the analysis is conditional on $\mathcal{F}_t^{S,v}$). Then for every $T > 0$,*
>
> $$
> \sup_{t \leq T} \mathbb{E}\bigl[(\bar G_n(t) - \bar G_\infty(t))^2\bigr] \;\leq\; \frac{C(T)}{n},\tag{4}
> $$
>
> *with*
>
> $$
> C(T) \;=\; \max\!\Bigl(\mathrm{Var}(G^0),\; \frac{\sigma_G^2}{2\theta_G}(1 - e^{-2\theta_G T}) + \mathrm{Var}(G^0)\,e^{-2\theta_G T}\Bigr) \;\leq\; \max\!\Bigl(\mathrm{Var}(G^0),\, \frac{\sigma_G^2}{2\theta_G}\Bigr).\tag{5}
> $$

**Proof (linear-target case).** Let $\delta_i(t) := G_i(t) - g(S(t), v(t))$. With the spot/variance path fixed, $\delta_i$ obeys $d\delta_i = -\theta_G\,\delta_i\,dt + \sigma_G\,dW^i_G - dg$. Independence of the noise gives $\mathrm{Cov}(\delta_i, \delta_j) = 0$ for $i \neq j$, so

$$
\mathrm{Var}(\bar G_n - \bar G_\infty) \;=\; \frac{1}{n}\,\mathrm{Var}(G_i - g) \;=\; \frac{1}{n}\Bigl[\mathrm{Var}(G^0)\,e^{-2\theta_G t} + \frac{\sigma_G^2}{2\theta_G}(1 - e^{-2\theta_G t})\Bigr].
$$

The supremum over $t \in [0, T]$ gives (4)–(5). The bound is *tight* for OU; the standard Sznitman Grönwall argument gives the same $C/n$ rate but with a possibly looser constant. $\square$

## Effect on the Hopf threshold

The MV system inserts a first-order low-pass filter (bandwidth $\theta_G$) between the spot/variance state and the aggregate gamma fed back into spot. At the deterministic Hopf frequency $\omega^\star$ from §3, the linearised transfer function from the target perturbation $\delta g$ to the aggregate $\bar G_\infty$ is

$$
\frac{\widehat{\bar G}_\infty(\omega)}{\widehat{\delta g}(\omega)} \;=\; \frac{\theta_G}{\theta_G + i\omega}\;\;\Rightarrow\;\;\Bigl|\frac{\widehat{\bar G}_\infty}{\widehat{\delta g}}\Bigr|(\omega^\star) \;=\; \frac{\theta_G}{\sqrt{\theta_G^2 + \omega^{\star 2}}}.\tag{6}
$$

The effective coupling at the Hopf frequency is therefore $\kappa_{\mathrm{eff}}(\omega^\star) = \kappa\cdot\theta_G/\sqrt{\theta_G^2 + \omega^{\star 2}}$, and the MV Hopf threshold expands by the reciprocal of the gain:

$$
\boxed{\;\frac{\kappa^\star_{\mathrm{MV}}}{\kappa^\star_{\mathrm{single}}} \;=\; \frac{\sqrt{\theta_G^2 + \omega^{\star 2}}}{\theta_G} \;=\; \sqrt{1 + (\omega^\star \tau_G)^2} \;=\; 1 \;+\; \tfrac{1}{2}(\omega^\star\tau_G)^2 \;+\; O\bigl((\omega^\star\tau_G)^4\bigr).\;} \tag{7}
$$

Two regimes are immediate:
- **Instantaneous hedging $\theta_G \to \infty$ (i.e.\ $\tau_G \to 0$):** ratio $\to 1$. MV recovers the single-dealer model — this is the implicit assumption of §§2–3.
- **Slow hedging $\theta_G < \omega^\star$:** ratio strictly $> 1$. The dealer ensemble damps the feedback channel, requiring stronger coupling to destabilise. The leading correction is $O((\omega^\star\tau_G)^2)$, so for $\omega^\star\tau_G = 0.1$ the MV correction is $\sim 0.5\%$; for $\omega^\star\tau_G = 1$ the threshold is $\sqrt{2}\times$ the single-dealer value.

**Numerical anchor at the canonical regime.** At the §4.3 canonical specification ($\sigma_q = 0.10$, $T_{\mathrm{eff}} = 0.25$, $\kappa_v = 2$, $\alpha = 0.05$, $\beta = 1$, $\gamma = 1$, $\mu_q = \log 100$, $v^\star = 0.04$) the closed-form Hopf threshold and frequency are $\kappa^\star_{\mathrm{single}} = 17.81$ and $\omega^\star = 1.18$ rad/yr. With a representative dealer-hedging speed $\theta_G = 50$/yr ($\tau_G \approx 5$ trading days), the MV threshold ratio is $\sqrt{1 + (1.18/50)^2} = 1.000277$, i.e.\ $\kappa^\star_{\mathrm{MV}} = 17.811$ — a $2.8\times 10^{-4}$ relative shift, operationally negligible. For a slower hedging cycle ($\theta_G = 5$/yr, $\tau_G \approx 50$ trading days) the ratio jumps to $\sqrt{1 + (1.18/5)^2} = 1.0273$, a $2.7\%$ shift at the edge of empirical detectability.

## Numerical validation

Implementation: `src/reflexive_options/theory/mckean_vlasov.py`. We sweep $n \in \{10, 32, 100, 316, 1000\}$ at the canonical regime ($\sigma_G = 0.05$, $\theta_G = 50$/yr, $T = 0.25$ yr, 250 Euler steps, 64 replicates per $n$, locked seed $20260514$) and measure $\sup_t \sqrt{\mathbb{E}[(\bar G_n(t) - \bar G_\infty(t))^2]}$. The fitted log-log slope of RMSE vs $1/\sqrt n$ is **$0.951$** (theoretical $1.0$), confirming the Sznitman $1/\sqrt n$ rate within finite-sample noise. The empirical RMSE sits *below* the closed-form bound $\sqrt{C(T)/n}$ across all $n$ — consistent with the sharp OU constant being an upper bound on the worst-case realisation.

See `paper/figures/mckean_vlasov_propagation_chaos.pdf` (RMSE vs $n$ on log-log axes with the $\sqrt{C(T)/n}$ reference line and the LS fit). Run dir: `runs/mckean_vlasov_validation/`.

## Conditions and scope

Theorem 3 requires (i) Lipschitz $g(\cdot)$ in $(S, v)$ — satisfied by the closed-form log-normal-OI aggregator (15a) on any compact $(S, v)$ neighbourhood of the equilibrium; (ii) finite second moment of $G_i^0$; and (iii) homogeneous dealer-hedging-speed $\theta_G$. Heterogeneous $\theta_G^{(i)}$ across dealers requires propagation-of-chaos over the joint $(G_i, \theta_G^{(i)})$ measure (the same Sznitman framework, doubled state space). Common dealer-noise channels (Carmona–Delarue 2018 Vol II Ch. 1) likewise extend the framework but introduce a residual non-vanishing variance that breaks the $1/n$ rate at leading order. Both are openly deferred.

The threshold-shift formula (7) is exact at the linearisation of the MV system around the equilibrium and matches the single-dealer linearisation in the $\tau_G \to 0$ limit. Higher-order corrections to $\kappa^\star_{\mathrm{MV}}$ from nonlinearities in $g(S, v)$ are $O(\sigma_G^2)$ and contribute to the stochastic-Hopf shift $\Lambda$ rather than the deterministic threshold; they are absorbed into the §5 stochastic-lift framework.

## References

- Sznitman, A.-S. (1991). *Topics in propagation of chaos*. Lecture Notes in Math. **1464**, 165–251. — Théorème I.1.4 is the canonical $C/n$ propagation-of-chaos bound under Lipschitz coefficients.
- Méléard, S. (1996). *Asymptotic behaviour of some interacting particle systems; McKean–Vlasov and Boltzmann models*. Lecture Notes in Math. **1627**, 42–95. — Prop 2.5 gives the explicit $L^2$ rate for the MV-McKean SDE.
- Carmona, R., & Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications I–II*. Springer. — Vol I Thm 2.12 is the modern textbook propagation-of-chaos statement; Vol II Ch. 1 covers common-noise extensions.
- Lacker, D. (2018). Mean field games via controlled martingale problems. *Stochastic Processes and Their Applications* **128**, 1099–1130. — Closes the existence–uniqueness gap for non-Lipschitz coefficients.
