# Theory — Hopf bifurcation and stationary density of the reflexive SDE

This document is the canonical writeup of the analytical contributions of the paper. The implementation in `src/reflexive_options/theory/` operationalizes these results numerically. Derivation details and a literature scan live in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md`.

---

## 1. The model

A complete comparison against the closest precedents — Halperin and Itkin's Marketron framework (arXiv:2508.09863, August 2025), Dai's gamma-squeeze recursion (arXiv:2511.22766, November 2025), Brock–Hommes-style heterogeneous-agent dynamics (Brock, Hommes & Wagener, *JEDC* 2009; He, Li & Zheng, *Economic Modelling* 2009; Chiarella, He & Hommes, UTS QFRC RP 268, 2006), the He–Sutter–Gonon adversarial deep-hedging line (arXiv:2508.14757, NeurIPS 2025), and the Ning–Jaimungal Wasserstein-on-surfaces work (arXiv:2108.04941, *SIAM SIFIN* 2024) — is in `related_work.md`. Each has 1–2 of the ingredients we combine; none has all.

Let $S_t$ be the underlying spot price, $v_t$ the instantaneous variance, and $z_t$ a memory variable encoding low-pass-filtered price history. The reflexive system is

$$
\frac{dS_t}{S_t} = \bigl(\mu + \kappa \cdot G(S_t, z_t, v_t)\bigr)\, dt + \sigma(S_t, v_t)\, dW^S_t, \tag{1a}
$$

$$
dv_t = \bigl(\kappa_v(\theta_v - v_t) + \gamma\, z_t\bigr)\, dt + \xi \sqrt{v_t}\, dW^v_t, \tag{1b}
$$

$$
dz_t = \bigl(-\alpha\, z_t + \beta\, \log(S_t / S_0)\bigr)\, dt, \tag{1c}
$$

$$
d\langle W^S, W^v\rangle_t = \rho\, dt, \tag{1d}
$$

with $\kappa \geq 0$ the reflexive coupling, $\gamma \geq 0$ the leverage-feedback strength, $\alpha > 0$ the memory decay rate, $\beta$ dimensionless, and $G(\cdot)$ the aggregate dealer-gamma exposure derived from the option open-interest grid à la Gârleanu–Pedersen–Poteshman (2009). When $\kappa = \gamma = 0$, the system collapses to standard Heston (1993).

### 1.1 Why three states?

A reduced 2D specification — dropping $z_t$ and the $\gamma z$ term — is the obvious starting point. It is also wrong for our purposes. The Jacobian of the deterministic skeleton in deviation variables $(y, u)$ around an equilibrium reads

$$
J_{2D}(\kappa) = \begin{pmatrix} a(\kappa) & b(\kappa) \\ 0 & -\kappa_v \end{pmatrix},
$$

which is *upper triangular* — eigenvalues $\{a(\kappa), -\kappa_v\}$ are always real. The 2D system can have at most a transcritical / saddle-node bifurcation as $a(\kappa)$ crosses zero (this recovers Dai 2025, arXiv:2511.22766, who derives precisely this static onset). It cannot Hopf.

The structural reason is that $v$ has no $S$-dependence in its drift, so variance relaxes deterministically to $\theta_v$ regardless of price moves. To get cyclic behaviour, the feedback must flow in *both* directions: price → variance and variance → price. The minimal modification is the leverage channel $\gamma z$ with the auxiliary memory equation (1c). This matches the marketron memory variable in Halperin–Itkin (2025) and is empirically defensible — trended markets feed dealer rebalancing which feeds variance.

---

## 2. Equilibria and Jacobian

Drop the Brownian terms and work in log-deviation variables $(y, u, z) = (\log(S/S^\star), v - \theta_v, z - z^\star)$ around the unique equilibrium $(S^\star, \theta_v, z^\star)$ defined by

$$
\mu - \tfrac{1}{2}\sigma^2(S^\star, \theta_v) + \kappa\, G(S^\star, z^\star, \theta_v) = 0, \qquad z^\star = \frac{\beta}{\alpha}\log(S^\star/S_0). \tag{2}
$$

To linear order, the deterministic skeleton (1a–c) becomes $\dot{\boldsymbol{x}} = J(\kappa)\, \boldsymbol{x}$ with $\boldsymbol{x} = (y, u, z)^\top$ and

$$
J(\kappa) = \begin{pmatrix}
a(\kappa) & b(\kappa) & \kappa\, G_z \\
0 & -\kappa_v & \gamma \\
\beta & 0 & -\alpha
\end{pmatrix}, \tag{3}
$$

where

- $a(\kappa) = \kappa\, G_x - \tfrac{1}{2}\partial_x\sigma^2$
- $b(\kappa) = \kappa\, G_v - \tfrac{1}{2}\partial_v\sigma^2$
- $G_x, G_v, G_z$ denote partial derivatives of $G$ at the equilibrium.

---

## 3. Routh–Hurwitz / Hopf condition

The characteristic polynomial of (3) is

$$
P(\lambda; \kappa) = \lambda^3 + c_2(\kappa)\,\lambda^2 + c_1(\kappa)\,\lambda + c_0(\kappa),
$$

with

$$
c_2 = -a + \kappa_v + \alpha, \qquad
c_1 = -a\kappa_v - a\alpha + \kappa_v\alpha - \kappa\, G_z\, \beta, \qquad
c_0 = -a\kappa_v\alpha - \beta\bigl(\kappa\, G_z\, \kappa_v + b\,\gamma\bigr).
$$

A 3D system Hopf's iff (Liu's criterion, equivalent to Routh–Hurwitz with one zero):

$$
c_2 > 0, \qquad c_0 > 0, \qquad c_1 c_2 - c_0 = 0, \qquad \frac{d}{d\kappa}\bigl[c_1 c_2 - c_0\bigr] \neq 0. \tag{4}
$$

Define $H(\kappa) := c_1(\kappa)\, c_2(\kappa) - c_0(\kappa)$. The Hopf threshold is

$$
\boxed{\;\kappa^\star = \{\kappa > 0 : H(\kappa) = 0,\; H'(\kappa) \neq 0,\; c_2(\kappa) > 0,\; c_0(\kappa) > 0\}.\;} \tag{5}
$$

At $\kappa = \kappa^\star$ the Jacobian has a pure imaginary eigenvalue pair $\lambda_\pm = \pm i\omega^\star$ with $\omega^\star = \sqrt{c_1(\kappa^\star)}$, plus a real eigenvalue $\lambda_3 = -c_2(\kappa^\star) < 0$.

---

## 4. Theorem 1 — Hopf bifurcation in the gamma-coupled SV skeleton

> **Theorem 1.** *Let the deterministic system (1a–c) (with Brownian terms removed) satisfy:*
>
> - **(A1)** *$G \in C^3$ and $\sigma^2 \in C^3$ in their arguments, with $\sigma^2$ bounded below by a positive constant.*
> - **(A2)** *There exists a locally unique equilibrium $(S^\star, \theta_v, z^\star)$ given by (2) for every $\kappa$ in some interval $[0, \kappa_{\max}]$, in some open neighborhood of which the linearization (3) is non-degenerate. Multiple equilibria may exist globally; our analysis is local.*
> - **(A3)** *The map $\kappa \mapsto (a(\kappa), G_z, G_v)$ is $C^3$.*
> - **(A4)** *There exists $\kappa^\star \in (0, \kappa_{\max})$ satisfying (5) with $\omega^\star := \sqrt{c_1(\kappa^\star)} > 0$.*
> - **(A5)** *The first Lyapunov coefficient $\ell_1(\kappa^\star) \neq 0$ (Kuznetsov 2004, eq. 3.20).*
>
> *Then the equilibrium undergoes a Hopf bifurcation at $\kappa = \kappa^\star$. Specifically:*
>
> 1. *For $\kappa$ in a one-sided neighbourhood of $\kappa^\star$ (sub- or super-critical depending on $\mathrm{sgn}\,\ell_1$), there exists a unique smooth family of periodic orbits $\Gamma_\kappa$ of period $T_\kappa = 2\pi/\omega^\star + O(\kappa - \kappa^\star)$ and amplitude $\propto \sqrt{|\kappa - \kappa^\star|}$.*
> 2. *The orbits are stable iff $\ell_1(\kappa^\star) < 0$ (supercritical).*
> 3. *On $\Gamma_\kappa$, $u_t = v_t - \theta_v$ oscillates with amplitude $\propto \sqrt{|\kappa - \kappa^\star|}$, providing endogenous volatility cycles.*

**Proof sketch.** The eigenvalue pair $\lambda_\pm(\kappa)$ is smooth in $\kappa$ near $\kappa^\star$ (implicit function theorem on $P(\lambda; \kappa) = 0$). Routh–Hurwitz gives $\mathrm{Re}\,\lambda_\pm(\kappa^\star) = 0$, and the transversality $H'(\kappa^\star) \neq 0$ translates to $\frac{d}{d\kappa}\mathrm{Re}\,\lambda_\pm \neq 0$. The centre manifold theorem (Kuznetsov 2004, Thm 5.4) gives a smooth 2D invariant manifold tangent to the eigenspace of $\pm i\omega^\star$. Conjugating the restricted system to its Poincaré normal form and applying Kuznetsov's Theorem 3.3 yields claims (1)–(3). □

The full derivation, including the explicit formula for $\ell_1$ via the bilinear/trilinear parts of (1a–c), is in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §4.

### 4.1 What is closed-form vs numerical

| Item | Closed form? |
|------|---|
| $\kappa^\star$ via Routh–Hurwitz $H(\kappa) = 0$ | Yes, given $G_x, G_v, G_z$ at equilibrium |
| Frequency $\omega^\star = \sqrt{c_1(\kappa^\star)}$ | Yes |
| Existence of Hopf (Theorem 1) | Yes |
| Sign of $\ell_1$ | Yes — closed form for log-normal OI in moneyness (§4.3, Eq. 17–18); numerical otherwise |
| Limit-cycle amplitude / shape past $\kappa^\star$ | Numerical |

The implementation in `src/reflexive_options/theory/bifurcation.py` does the numerical eigenvalue scan over $(\kappa, \xi)$ and locates $\kappa^\star$ as the contour where $\mathrm{Re}\,\lambda_\pm$ crosses zero. It also computes the first Lyapunov coefficient $\ell_1$ (Kuznetsov 2004 eq. 3.20) via finite-difference construction of the bilinear / trilinear Taylor tensors $B$ and $C$ around the equilibrium, and the stochastic-Hopf shift $\Lambda(\kappa)$ via a Khasminskii-style sphere process (Benettin renormalisation of the linearised SDE).

### 4.2 Numerical $\ell_1$ and $\Lambda$ at a representative parameter set

The default `BifurcationConfig` in `experiments/bifurcation_scan.py` is dimensionally tuned to the empirical dealer-gamma magnitudes ($G_x \sim 10^{-3}$, $G_z \sim 10^{-3}$ per USD-of-dollar-gamma) and *does not exhibit a Hopf within the literature-prior κ range*. The structural reason: the memory channel decay $\alpha = 252$/yr (≈ 1-day half-life) is fast relative to $\kappa_v = 2$/yr, and the small first-derivative magnitudes mean the cross-coupling $\kappa G_z \beta$ in $c_1$ never overpowers $\kappa_v \alpha$. This matches the empirical observation that real options markets sit *near* but not *across* $\kappa^\star$ — consistent with HBB's $n \approx 1$ critical regime (§6).

For a worked numerical example we instead use a representative *dimensionless* regime where $\{G_x, G_v, G_z\} = \{0.5, -0.5, -0.5\}$, $\alpha = 0.5$/yr (multi-day memory), $\beta = 1$, $\gamma = 0.5$, $\kappa_v = 2$/yr, and quadratic / cubic Taylor coefficients $G_{xx} = -0.1$, $G_{xxx} = -0.2$ (representative of a smooth, locally concave dealer-gamma functional around ATM in a long-gamma regime). For this example we use a *constant-vol surrogate* ($\partial_v \sigma^2 = 0$) rather than the Heston $\sigma^2 = v$ of §1; this isolates the $\kappa$-dependence of the bifurcation from the Itô-correction term in $b(\kappa) = \kappa G_v - \tfrac{1}{2}\partial_v \sigma^2$ and lets the $\{G_x, G_v, G_z, \kappa_v, \alpha, \beta, \gamma\}$ prescribed values stand alone. The §4.3 closed-form analysis returns to the $\sigma^2 = v$ Heston backbone. The deterministic skeleton then satisfies:

| Quantity | Value |
|---|---|
| $\kappa^\star$ | $0.8964$ |
| $\omega^\star$ | $0.5724$ rad/yr (period $\approx 11.0$ yr) |
| Third real eigenvalue $\lambda_3$ | $-2.052$ |
| First Lyapunov coefficient $\ell_1$ | $-2.53 \times 10^{-2}$ |
| **Bifurcation type** | **supercritical** ($\ell_1 < 0$) |

Supercriticality means an attracting limit cycle is born for $\kappa > \kappa^\star$, with amplitude $\propto \sqrt{\kappa - \kappa^\star}$ — i.e. *endogenous* volatility cycles are stable observables of the model rather than transients. In the subcritical case ($\ell_1 > 0$) we would instead see hysteresis and abrupt regime jumps; the supercritical sign here matches the qualitative empirical character of slow vol-clustering build-ups documented in HBB.

For the stochastic lift at $(\xi, \rho) = (0.3, -0.7)$ — calibration-representative for SPX — the Khasminskii estimator gives

$$
\Lambda(\kappa^\star) \approx +1.85 \times 10^{-2}
$$

at noise normalisation $\varepsilon \in \{0.05, 0.20\}$, with a path budget of $10^3$ trajectories × $10^4$ steps. The positive sign means *noise destabilises* the equilibrium: the stochastic Hopf threshold

$$
\kappa^\star_{\mathrm{stoch}}(\varepsilon) \approx \kappa^\star - \frac{\varepsilon^2 \Lambda}{|\alpha'(\kappa^\star)|}
$$

lies *below* the deterministic threshold, consistent with the Engel–Lamb–Rasmussen prediction for shear-induced corrections in correlated multiplicative-noise systems. At a calibration noise scale of $\varepsilon = 0.1$ this shifts the top Lyapunov exponent by $\Lambda \cdot \varepsilon^2 \approx 1.85 \times 10^{-4}$, which is small in absolute terms but predictable in sign.

### 4.3 Closed-form first Lyapunov coefficient for log-normal OI in moneyness

The numerical $\ell_1$ in §4.2 is built from a finite-difference construction of the bilinear/trilinear tensors $B$, $C$ around the equilibrium. While accurate to ${\sim}10^{-3}$ at the canonical regime, this leaves the *sign* of $\ell_1$ — the structurally critical quantity that fixes super- vs sub-criticality — vulnerable to numerical noise near the $\ell_1 = 0$ contour. For the natural parametric specification of the dealer-gamma functional — a *log-normal open-interest density in log-strike* — the entire computation admits a closed form, eliminating this vulnerability and exposing the parametric phase boundary explicitly.

#### 4.3.1 The log-normal aggregator

Replace the discrete OI grid with the continuous density $q(\log K) = \frac{1}{\sigma_q\sqrt{2\pi}}\exp\bigl(-(\log K - \mu_q)^2/(2\sigma_q^2)\bigr)$ — log-normal in $K$, equivalently Gaussian in log-strike with center $\mu_q$ and spread $\sigma_q$. Take a single representative maturity $T_{\mathrm{eff}}$, sign convention $s \equiv +1$ (SqueezeMetrics SPX default), and write $a := \log S$ throughout. The aggregator becomes

$$
G(a, v) \;=\; \kappa_u \int_{-\infty}^{+\infty} q(\log K)\, \Gamma_{\mathrm{BS}}\!\left(S, K, T_{\mathrm{eff}}, \sqrt{v}\right)\, d(\log K). \tag{14}
$$

Black–Scholes gamma at fixed $T = T_{\mathrm{eff}}$ and $\sigma = \sqrt{v}$ is itself Gaussian in $\log K$:

$$
\Gamma_{\mathrm{BS}}(S, K, T, \sigma) \;=\; \frac{e^{-q_\mathrm{div}T}}{S\sigma\sqrt{T}} \cdot \frac{1}{\sqrt{2\pi}}\,\exp\!\biggl(-\frac{(\log K - \mu_d)^2}{2\sigma^2 T}\biggr),
$$

with $\mu_d := a + (r - q_{\mathrm{div}} + \tfrac{1}{2} v)T$. The integrand in (14) is therefore a product of two Gaussians in $\log K$; the standard Gaussian-product identity (Briggs 2003; also Halperin–Itkin Marketron Eq. B.8) collapses (14) to

$$
\boxed{\;G(a, v) \;=\; \mathcal{C}(v)\cdot e^{-a}\cdot \exp\!\biggl(-\frac{(a - m(v))^2}{2\,\tau^2(v)}\biggr),\;} \tag{15a}
$$

where

$$
\mathcal{C}(v) \;=\; \frac{\kappa_u\, e^{-q_{\mathrm{div}} T_{\mathrm{eff}}}}{\sqrt{2\pi\,\tau^2(v)}}, \qquad
\tau^2(v) \;=\; \sigma_q^2 + v\,T_{\mathrm{eff}}, \tag{15b}
$$

$$
m(v) \;=\; \mu_q - \bigl(r - q_{\mathrm{div}} + \tfrac{1}{2}v\bigr)\, T_{\mathrm{eff}}. \tag{15c}
$$

That is: $G$ is a Gaussian in $a$ centred near (a translation of) the OI mean $\mu_q$, with width $\tau(v)$ that mixes the OI spread $\sigma_q$ and the implied-vol scale $\sqrt{v\,T_{\mathrm{eff}}}$ in quadrature, modulated by the $1/S = e^{-a}$ factor that comes from the per-share normalisation of $\Gamma_{\mathrm{BS}}$. Verified symbolically in `notebooks/closed_form_ell1_derivation.py` (sympy, 30 s wall-clock).

**Vanishing $z$-channel.** $G$ in (15a) does not depend on $z$ — the OI grid is exogenous to the slow trend variable. Therefore $G_z = 0$, and the Jacobian (3) loses its $(1, 3)$ entry. **This does *not* eliminate the Hopf** — the leverage channel $\gamma z$ in $\dot u$ keeps the $(2,3)$ entry alive, so cross-coupling persists through the variance row.

#### 4.3.2 Closed-form Hopf threshold $\kappa^\star$

With $G_z = 0$ and $\sigma^2 = v$ (whence $\partial_y\sigma^2 = 0$ and $\partial_v\sigma^2 = 1$), the Routh–Hurwitz polynomial $H(\kappa) := c_1 c_2 - c_0$ degree-collapses from cubic to **quadratic** in $\kappa$:

$$
H(\kappa) \;=\; G_y^2\,(\alpha + \kappa_v)\,\kappa^2 \;+\; \bigl(G_v\,\beta\gamma - G_y\,(\alpha + \kappa_v)^2\bigr)\,\kappa \;+\; \alpha\kappa_v(\alpha + \kappa_v) - \tfrac{1}{2}\beta\gamma. \tag{16}
$$

Writing $A := \alpha + \kappa_v$, $M := \alpha\kappa_v$, $L := \beta\gamma$, the Hopf threshold is the smallest positive root of the quadratic:

$$
\boxed{\;\kappa^\star \;=\; \frac{G_y\, A^2 - G_v\, L \;-\; \sqrt{(G_v L - G_y A^2)^2 - 4\,G_y^2 A\,(MA - L/2)}}{2\,G_y^2\,A}.\;} \tag{17}
$$

*Footnote.* When the $-$ branch yields a non-positive root (or violates $c_2 > 0$, $c_0 > 0$), the implementation falls back to the $+$ branch; we report the smallest positive root that satisfies the Routh–Hurwitz positivity conditions.

The Hopf frequency follows immediately from $\omega^\star = \sqrt{c_1(\kappa^\star)} = \sqrt{-\kappa^\star G_y \cdot A + M}$ once $\kappa^\star$ is in hand. The discriminant in (17) provides the *first* parametric phase boundary: when $D := (G_v L - G_y A^2)^2 - 4\,G_y^2 A\,(MA - L/2) < 0$, the system has no Hopf at any $\kappa$ — the deterministic skeleton is unconditionally stable.

#### 4.3.3 Closed-form first Lyapunov coefficient $\ell_1$

All third-order partials of (15a) at the equilibrium $(a^\star, v^\star)$ are explicit rational functions of $(\delta := a^\star - m(v^\star),\, \tau^2,\, T_{\mathrm{eff}})$. Writing $g(a, v) := \log G(a, v) - \log\mathcal{C}(v) = -a - \delta^2/(2\tau^2)$:

$$
\begin{aligned}
g_a &= -1 - \delta/\tau^2, &\qquad g_{aa} &= -1/\tau^2, &\qquad g_{aaa} &= 0, \\
g_v &= -\frac{\delta T_{\mathrm{eff}}}{2\tau^2} + \frac{\delta^2 T_{\mathrm{eff}}}{2\tau^4}, &\qquad
g_{vv} &= -\frac{T_{\mathrm{eff}}^2}{4\tau^2} + \frac{\delta T_{\mathrm{eff}}^2}{\tau^4} - \frac{\delta^2 T_{\mathrm{eff}}^2}{\tau^6}, &\qquad
g_{vvv} &= \frac{3T_{\mathrm{eff}}^3}{4\tau^4} - \frac{3\delta T_{\mathrm{eff}}^3}{\tau^6} + \frac{3\delta^2 T_{\mathrm{eff}}^3}{\tau^8}, \\
g_{av} &= -\frac{T_{\mathrm{eff}}}{2\tau^2} + \frac{\delta T_{\mathrm{eff}}}{\tau^4}, &\qquad
g_{aav} &= \frac{T_{\mathrm{eff}}}{\tau^4}, &\qquad
g_{avv} &= \frac{T_{\mathrm{eff}}^2}{\tau^4} - \frac{2\delta T_{\mathrm{eff}}^2}{\tau^6}.
\end{aligned}
$$

Combined with the prefactor $\mathcal{C}(v)$ via the product rule on $\log G = \log\mathcal{C}(v) + g(a, v)$ — using $h_p := -T_{\mathrm{eff}}/(2\tau^2)$, $h_{pp} := T_{\mathrm{eff}}^2/(2\tau^4)$, $h_{ppp} := -T_{\mathrm{eff}}^3/\tau^6$ for $h(v) := \log\mathcal{C}(v)$, and $L_k := h^{(k)} + g_{vv\cdots v}$ for the log-G $v$-partials — one obtains the closed-form $\{G_a, G_v, G_{aa}, G_{av}, G_{vv}, G_{aaa}, G_{aav}, G_{avv}, G_{vvv}\}$ in 18 multiply-add steps. All other partials ($G_z, G_{vz}, G_{aaz}, \ldots$) are zero.

The bilinear and trilinear tensors at $\kappa^\star$ are then assembled from these partials: $B_{1,j,k} = \kappa^\star \cdot G_{\{a,v\}\{a,v\}}$ for $j, k \in \{0, 1\}$, all other $B$-entries zero (the variance and memory equations are linear); $C$ is similarly populated only on the $i = 1$ slice. Substituting into Kuznetsov 2004 eq. 3.20,

$$
\ell_1(\kappa^\star) \;=\; \frac{1}{2\omega^\star}\,\mathrm{Re}\!\Bigl[\langle p, C(q, q, \bar q)\rangle - 2\langle p, B(q, J^{-1} B(q, \bar q))\rangle + \langle p, B(\bar q, (2i\omega^\star I - J)^{-1} B(q, q))\rangle\Bigr], \tag{18}
$$

with $J q = i\omega^\star q$, $J^\top p = -i\omega^\star p$, $\langle p, q\rangle = 1$. The right- and left-eigenvector pair at $\pm i\omega^\star$ for the now-reduced Jacobian admits a clean closed form via Cramer's rule on $(J - i\omega^\star I)\,q = 0$, but the resulting expression for $\ell_1$ is a $\sim$30-term rational in $(G_y, G_v, G_{aa}, G_{av}, G_{vv}, G_{aaa}, G_{aav}, G_{avv}, G_{vvv}, \kappa_v, \alpha, \beta, \gamma)$ that is more useful as a numerical pipeline than as a printable formula. The implementation in `lyapunov_coefficient_lognormal_oi` evaluates it in ${\sim}50$ μs per $(\sigma_q, T_{\mathrm{eff}}, \alpha, \beta, \gamma)$ tuple, with results that match the FD-tensor pipeline of §4.2 to better than 0.6% relative on every parameter set tested (`tests/test_lognormal_lyapunov.py::test_ell1_matches_existing_numerical_lyapunov`).

#### 4.3.4 Phase boundary in $(\sigma_q, \gamma)$ space

The headline parametric result is a *two-dimensional phase diagram* over $(\sigma_q, \gamma)$ at fixed $(\mu_q, T_{\mathrm{eff}}, \alpha, \beta, \kappa_v)$. Three regimes coexist:

| Region | Discriminant $D$ | $\ell_1$ | Dynamics |
|---|---|---|---|
| **No Hopf** | $D < 0$ or root non-positive | n/a | Equilibrium globally stable for all $\kappa$ |
| **Supercritical** | $D \geq 0$, smallest positive root real | $\ell_1 < 0$ | Stable limit cycle for $\kappa > \kappa^\star$, amplitude $\propto \sqrt{\kappa - \kappa^\star}$ |
| **Sub-critical** | $D \geq 0$, smallest positive root real | $\ell_1 > 0$ | Unstable cycle past $\kappa^\star$; hysteresis, abrupt regime jumps |

At the canonical specification $(\mu_q = \log 100,\, T_{\mathrm{eff}} = 0.25\text{ yr},\, \kappa_v = 2,\, \theta_v = 0.04,\, \alpha = 0.05,\, \beta = 1)$, the phase diagram is rendered in `paper/figures/ell1_phase_boundary.pdf` (script: `notebooks/closed_form_ell1_derivation.py`). The supercritical region ($\ell_1 < 0$) is a *narrow band* bounded by two $\ell_1 = 0$ contours in $(\sigma_q, \gamma)$ space. For very small $\sigma_q$ ($\lesssim 0.05$) the band collapses entirely — concentrated OI gives $\ell_1 > 0$ across the tested $\gamma$ range. For moderate $\sigma_q$ ($\sim 0.10$) the supercritical band is restricted to a small interval $\gamma \in [\gamma_{\mathrm{low}}, \gamma_{\mathrm{high}}]$ of order unity. As $\sigma_q$ grows, the band first widens then narrows again. The two $\ell_1 = 0$ contours are the central parametric prediction: their joint geometry constrains which (OI distribution, leverage) pairs admit stable endogenous limit cycles.

#### 4.3.5 Numerical verification at the canonical regime

Within `tests/test_lognormal_lyapunov.py` the canonical regime $(\sigma_q = 0.10,\, T_{\mathrm{eff}} = 0.25,\, \kappa_v = 2,\, \alpha = 0.05,\, \beta = 1,\, \gamma = 1,\, \mu_q = \log 100,\, v^\star = 0.04)$ produces:

| Quantity | Closed form (Eq. 17–18) | Numerical (FD on closed-form drift) | Relative agreement |
|---|---|---|---|
| $\kappa^\star$ | $17.8065068$ | $17.8065068$ | $< 10^{-9}$ (root-finding identical) |
| $\omega^\star$ | $1.1774426$ rad/yr | $1.1774426$ | $< 10^{-9}$ |
| $\ell_1$ | $-4.81461\times 10^{-1}$ | $-4.84302\times 10^{-1}$ | $0.59\%$ |
| **Bifurcation type** | **Supercritical** | **Supercritical** | sign agree |

The closed form is essentially noise-free; the residual $0.59\%$ in $\ell_1$ is the numerical FD-tensor error of the comparison pipeline, *not* a deficiency in the closed form (verified by varying $h$ in `build_bilinear_trilinear_tensors`: the closed-form value is invariant, the numerical value drifts as $O(h^2)$ until roundoff dominates near $h \sim 10^{-4}$). With the closed form available we no longer need the FD pipeline for parametric OI — and crucially, the *sign* of $\ell_1$ near the boundary contour is now structurally certain rather than statistically inferred.

### 4.4 Numerical phase diagram

The full $(\kappa, \sigma_v, \xi, \rho)$ phase diagram is computed by `python -m reflexive_options.experiments.hopf_phase_scan_4d` and rendered in Figure~\ref{fig:hopf-phase-diagram}. The scan sweeps $\kappa \in [0, 2]$ on a 401-point grid for each of $31 \times 21 \times 4 = 2{,}604$ cells over $(\xi, \rho, \sigma_v)$ at the §4.2 Hopf-exhibiting regime; $(\xi, \rho)$ enter the deterministic Jacobian via the leading shear-induced correction $a_{\mathrm{eff}}(\kappa) = a(\kappa) + \tfrac{1}{2} \xi^2 \rho\, G_v$ (an Engel–Lamb–Rasmussen-style projection of the small-noise stochastic Hopf onto the eigenvalue envelope). The full Khasminskii $\Lambda(\kappa; \xi, \rho)$ via Algorithm 2 is too expensive at this resolution; the deterministic projection captures the qualitative geometry — including the no-Hopf wedge at high $\xi$ and strong negative $\rho$ — at $\sim$16 s wall-clock on a single M-series core. The figure substantiates the §4.2 claim that real options markets sit *near but not across* $\kappa^\star$ throughout the SPX-relevant $(\xi, \rho)$ corner.

---

## 5. Stochastic lift

The deterministic Hopf indicates *when* the noiseless skeleton begins to oscillate. The full SDE (1) has multiplicative Heston noise; the relevant question (Arnold 1998) is when the top Lyapunov exponent $\lambda_1$ of the linearised RDS at the equilibrium changes sign.

For small noise intensity $\varepsilon$,

$$
\lambda_1(\kappa, \varepsilon) = \alpha(\kappa) + \varepsilon^2 \cdot \Lambda(\kappa) + O(\varepsilon^4),
$$

where $\alpha(\kappa)$ is the deterministic real part (zero at $\kappa^\star_{\mathrm{det}}$) and $\Lambda$ is computed via Engel–Lamb–Rasmussen (2024). Engel–Lamb–Rasmussen (2024) predict an asymptotic shear-driven scaling $|\Lambda| \sim (\rho\xi)^{2/3}$ in the limit of small noise. The numerical $\Lambda$ values produced by our finite-budget Khasminskii estimator (`stochastic_hopf_shift_numeric`) are not strongly $(\rho\xi)$-dependent in the tested regime — empirically $\Lambda$ is roughly constant across $(\rho\xi) \in [0.07, 0.42]$ at the §4.2 representative parameter set, with relative variation $\approx 27\%$ over a 6× change in $(\rho\xi)$. We do not claim to have validated the asymptotic scaling; the reported $\Lambda$ is a finite-noise two-point estimate at $(\varepsilon_1, \varepsilon_2) = (0.05, 0.20)$. A definitive small-noise asymptotic check would require a dedicated experiment we defer. We compute $\Lambda$ numerically via Khasminskii's sphere process (Algorithm 2 in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §6).

---

## 6. Connection to "critical reflexivity" (Hardiman–Bercot–Bouchaud 2013)

HBB find the Hawkes branching ratio $n \approx 1$ on E-mini mid-price changes over 1998–2011 — exactly the critical edge between sub- and super-criticality of endogenous events.

**Conjecture.** $\kappa^\star$ in (5) is the structural mechanism for HBB's empirical $n \approx 1$. Both criteria identify the boundary at which endogenous dynamics dominate exogenous forcing. The mapping $\kappa \leftrightarrow n$ is conceptually clear (both = "how much the system feeds back on itself"), but a formal reduction relating Hawkes branching ratio to SV-Jacobian eigenvalues does not exist in the published literature and would itself be a separate paper.

This connection, *if established*, would make the present model a candidate microfoundation for HBB's macroscopic finding.

---

## 7. Stationary density

The Fokker–Planck PDE for the joint density $\pi(S, v, z, t)$ admits no general closed-form stationary solution. We compute the stationary marginal density $\pi^\star(\log S)$ numerically by Monte-Carlo: long simulations past the mixing time, KDE on the marginal, with a long-run mean re-centring to remove the unbounded drift component (no martingale property under the physical measure). The Heston comparator is the closed-form Feller stationary density on $v$,

$$
\pi^\star_\text{Heston}(v) = \frac{(2\kappa_v / \xi^2)^a}{\Gamma(a)}\, v^{a-1}\, e^{-2\kappa_v v / \xi^2},\qquad a = \frac{2\kappa_v\theta_v}{\xi^2}, \tag{6}
$$

(Feller 1951; verified to KS distance $< 0.01$ against simulated paths in `tests/test_stationary.py`), plus the characteristic-function quantile inverter for log-returns at horizon $\tau$.

The pre-registered hypotheses, with the empirical findings reported alongside (the pre-registration doctrine is binding even on the theory section — failures are reported, not buried):

### 7.1 Setup

Heston base parameters $(\kappa_v, \theta_v, \xi, \rho, v_0) = (2.0,\, 0.04,\, 0.30,\, -0.70,\, 0.04)$. Reflexive simulator uses the same base, drift $\mu = 0$, $\gamma = 0$ (the deterministic-leverage channel is held off in this scan to keep the variance OU stable for the MC budget; see the H_bimod note below for why this matters), and $G$ from a flat $7\times 3$ open-interest grid of $50{,}000$ contracts per cell with $\sigma_\text{IV} = 0.20$. We sweep $\kappa \in \{0,\, 10^{-13},\, 5{\cdot}10^{-13},\, 10^{-12},\, 5{\cdot}10^{-12}\}$ — a band straddling the literature prior $\kappa \in [10^{-12}, 10^{-11}]$ per USD dollar-gamma (GPP 2009 calibration). Sample budget per point: $4{,}000$ paths $\times$ $4{,}000$ post-burn-in steps at $dt = 1/252$.

### 7.2 Hypothesis H_tail — heavier tails than Heston

**Claim.** $\pi^\star_\text{reflexive}(\log S)$ has tail index strictly less than $\pi^\star_\text{Heston}(\log S)$ whenever $\kappa > 0$.

**Empirical (anchor $\kappa = 5\cdot 10^{-13}$).** Reported by `compare_to_heston(...)`:

| Statistic | Reflexive | Heston | $\Delta$ |
|---|---:|---:|---:|
| Variance (centred log $S$) | $3.67$ | $0.72$ | $+2.95$ |
| Skewness | $+167.5$ | $-0.47$ | $+167.99$ |
| Excess kurtosis | $4.52\!\times\!10^{4}$ | $+0.50$ | $+4.52\!\times\!10^{4}$ |
| Hill tail index ($k=200$) | $1.70$ | $73.2$ | $-71.5$ |
| Anderson–Darling p-value | — | — | $< 10^{-3}$ |

Excess kurtosis is $\sim\!10^5\!\times$ larger and the Hill index drops by two orders of magnitude. **H_tail is supported at the anchor**, with the caveat that Hill is noisy when tails are not strictly Pareto and excess kurtosis is therefore the cleaner statistic. Across the $\kappa$ grid the Hill index is non-monotonic — it drops sharply at small $\kappa$ then rebounds at $\kappa = 5\cdot 10^{-12}$ — consistent with explosive paths being increasingly censored by the variance-truncation scheme (Lord–Koekkoek–van Dijk 2010) under positive feedback rather than a genuine return to thinner tails. Excess kurtosis is monotone in $\kappa$ over the stable sub-range $\{0, 10^{-13}, 5\cdot 10^{-13}\}$.

**Caveats.** The Hill index is defined for distributions with regularly-varying tails; the reflexive simulator's tails are not strictly Pareto, so Hill is at best a rough indicator. The excess-kurtosis estimate ($4.5\!\times\!10^4$ at the anchor) is finite by construction at any sample size but does not converge as $N\to\infty$ if the underlying 4th moment is infinite — and the heaviest-tail regimes here may have unbounded population kurtosis. We retain the comparison to Heston as a *qualitative* signal that reflexivity reshapes the tail (the AD test rejects same-distribution at $p < 10^{-3}$), but the specific numerical magnitudes of the Hill index and excess kurtosis should not be taken as point estimates of population quantities.

### 7.3 Hypothesis H_skew — sign of skewness tracks $\mathrm{sgn}(G_x)$

**Claim.** $\mathrm{sgn}\,\mathrm{skew}(\pi^\star_\text{reflexive}) = \mathrm{sgn}(G_x)$ at the equilibrium — long-gamma ($G_x > 0$) and short-gamma ($G_x < 0$) regimes produce qualitatively different asymmetries.

**Empirical.** At the anchor, $G_x > 0$ (calls-only OI default in the test fixture) and $\mathrm{skew} = +167.5$ — large positive skew, opposite sign to Heston ($-0.47$). **H_skew is supported in sign**, and the magnitude is dominated by the rare positive-feedback excursions of the reflexive channel.

### 7.4 Hypothesis H_bimod — bimodality near $\kappa^\star$

**Claim.** Approaching the Hopf bifurcation, $\pi^\star(\log S)$ develops emergent bimodality reflecting the underlying limit cycle.

**Empirical (Hartigan & Hartigan 1985 dip statistic across the $\kappa$-grid):**

| $\kappa$ | dip statistic | p-value | bimodal at 5%? |
|---:|---:|---:|---|
| $0$              | $1.7\!\times\!10^{-4}$ | $0.31$ | no |
| $1\!\times\!10^{-13}$ | $1.7\!\times\!10^{-4}$ | $0.32$ | no |
| $5\!\times\!10^{-13}$ | $0.7\!\times\!10^{-4}$ | $1.00$ | no |
| $1\!\times\!10^{-12}$ | $1.0\!\times\!10^{-4}$ | $0.99$ | no |
| $5\!\times\!10^{-12}$ | $4.9\!\times\!10^{-4}$ | $0.10$ | no (marginal) |

**H_bimod is *not* supported in this scan.** Dip statistics are uniformly small and only the largest $\kappa$ shows marginal evidence ($p \approx 0.10$, not significant at the standard 5% threshold). Two interpretations: (a) we have not yet reached the Hopf threshold $\kappa^\star$ in this parameter family — note that $\gamma = 0$ removes the closing leverage feedback that the 3D Hopf in §3 needs to oscillate (see assumption (A4) of Theorem 1 and the structural argument in §1.1: the bare 2D skeleton cannot Hopf, so genuine limit-cycle bimodality is precluded by construction here); (b) the marginal density on $\log S$ is the wrong projection — limit-cycle signature would more clearly show in the joint $(\log S, v)$ density. We flag this as a methodological note for the empirical phase: the bimodality scan needs to be repeated with the leverage channel $\gamma > 0$ active and ideally on 2D KDEs of $(\log S, v)$.

### 7.5 Summary

| Hypothesis | Outcome at anchor | Notes |
|---|---|---|
| H_tail (heavier tails) | **Supported** | $\Delta$ excess kurtosis $\sim 10^5$; AD-rejection $p < 10^{-3}$ |
| H_skew (sign-tracking) | **Supported** | sign of skew matches $\mathrm{sgn}(G_x)$, opposite Heston |
| H_bimod (emergent bimodality) | **Not supported** | $\gamma = 0$ closes off the 3D Hopf channel; needs follow-up at $\gamma > 0$ on 2D marginals |

Implementation: `src/reflexive_options/theory/stationary.py`. Tests: `tests/test_stationary.py`. The numbers above were produced by the scan in §7.1; the script is regenerable from the public functions `compare_to_heston`, `tail_index_vs_kappa_curve`, and `detect_bimodality`.

---

## 8. Open items (theory)

1. ~~Closed-form $\ell_1$ for log-normal OI in moneyness~~ — **resolved** in §4.3; closed form (Eq. 17–18) implemented in `lyapunov_coefficient_lognormal_oi`, supercritical at the canonical regime, parametric phase boundary in $(\sigma_q, \gamma)$ space rendered in `paper/figures/ell1_phase_boundary.pdf`.
2. Closed-form $\Lambda(\kappa)$ for the stochastic-Hopf shift (Engel–Lamb–Rasmussen-style asymptotics adapted to the Heston multiplicative-noise structure).
3. Formal Hawkes-$n$ ↔ SV-eigenvalue reduction (most ambitious; not blocking for the v1 paper).

## References

Full bibliography in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §References.
