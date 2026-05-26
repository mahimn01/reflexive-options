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

The Khasminskii sphere-process estimator (`compute_lambda_correction`),
evaluated by `experiments/lambda_correction_canonical` on the bare
Heston-with-memory linearisation at the trivial $G \equiv 0$ equilibrium
with the §4.2 memory parameters, gives $|\Lambda(\kappa^\star)|$ on the
order of $10^{-3}$ at both the canonical $(\xi, \rho) = (0.3, 0)$ and
SPX-representative $(0.3, -0.7)$ configurations (path budget: $2 \times
10^3$ trajectories × $5 \times 10^3$ steps; representative output:
$\Lambda_{\text{canonical}} \approx -6.9 \times 10^{-3}$,
$\Lambda_{\text{spx-rep}} \approx -3.5 \times 10^{-3}$, locked seed
20260422; numbers reproducible from `runs/lambda_correction_canonical/`).

The sign of $\Lambda$ depends sensitively on the open-interest configuration
and on the specific equilibrium chosen — at the trivial $G \equiv 0$
linearisation evaluated above the sign is empirically negative, and
preliminary scans at non-trivial OI grids produce sign reversals that we do
not yet characterise rigorously. We therefore defer a definitive
characterisation of $\Lambda$'s sign to the empirical phase and use only the
magnitude in what follows. The stochastic Hopf threshold

$$
\kappa^\star_{\mathrm{stoch}}(\varepsilon) \approx \kappa^\star - \frac{\varepsilon^2 \Lambda}{|\alpha'(\kappa^\star)|}
$$

is therefore predicted to lie within $|\Lambda| \cdot \varepsilon^2 / |\alpha'(\kappa^\star)|
\sim 10^{-5}$ of the deterministic threshold at calibration noise scale
$\varepsilon = 0.1$ — operationally negligible. The Engel–Lamb–Rasmussen
prediction for shear-induced corrections in correlated multiplicative-noise
systems is consistent with this magnitude but is not validated in *sign*
by the present finite-budget estimator.

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

With $G_z = 0$ and $\sigma^2 = v$ (whence $\partial_y\sigma^2 = 0$ and $\partial_v\sigma^2 = 1$), the Routh–Hurwitz polynomial $H(\kappa) := c_1 c_2 - c_0$ degree-collapses from cubic to **quadratic** in $\kappa$ (we write $G_y := \partial_a G$ throughout this closed-form section; identical to the $G_x$ of §2 up to the choice of deviation symbol):

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

#### 4.3.6 Robustness of $\kappa^\star$ to OI mis-specification (Phase-4 calibration tolerance)

Phase 4 calibrates $(\hat\mu_q, \hat\sigma_q)$ from empirical SPX OI and feeds them into Eq. 17 to predict $\kappa^\star$. The accuracy of the prediction is bounded by (a) the local sensitivity of $\kappa^\star$ to $(\mu_q, \sigma_q)$, and (b) the structural mismatch between the empirical (multi-modal, calendar-clustered) OI and the log-normal that Eq. 17 assumes. We quantify both. Implementation: `src/reflexive_options/theory/robustness.py`; experiment: `experiments/kappa_star_robustness.py`; tests: `tests/test_kappa_star_robustness.py`; companion writeup `paper/kappa_star_robustness.md`.

##### Analytical elasticities

Implicit differentiation of $H(\kappa^\star; G_y, G_v) = 0$ on the closed-form quadratic Eq. 16 — with $A_2 = G_y^2 A$, $A_1 = G_v L - G_y A^2$, $A_0 = M A - L/2$ — gives

$$
\frac{\partial \kappa^\star}{\partial G_y} \;=\; -\frac{2 G_y A \kappa^{\star 2} - A^2 \kappa^\star}{2 A_2 \kappa^\star + A_1},
\qquad
\frac{\partial \kappa^\star}{\partial G_v} \;=\; -\frac{L \kappa^\star}{2 A_2 \kappa^\star + A_1}, \tag{19}
$$

with denominator $\partial H/\partial \kappa = 2 A_2 \kappa^\star + A_1$ guaranteed non-zero away from the Bautin curve (where the discriminant vanishes — paper §4.4.1). The OI-parameter chain rule is

$$
\frac{\partial \kappa^\star}{\partial \mu_q}
\;=\; \frac{\partial \kappa^\star}{\partial G_y}\frac{\partial G_y}{\partial \mu_q}
+ \frac{\partial \kappa^\star}{\partial G_v}\frac{\partial G_v}{\partial \mu_q},
\qquad
\frac{\partial \kappa^\star}{\partial \sigma_q}
\;=\; \frac{\partial \kappa^\star}{\partial G_y}\frac{\partial G_y}{\partial \sigma_q}
+ \frac{\partial \kappa^\star}{\partial G_v}\frac{\partial G_v}{\partial \sigma_q}. \tag{20}
$$

The OI-parameter partials of the closed-form $G$-derivatives — $\partial G_y/\partial \mu_q$ etc. — are computed by central FD on the analytic `G_lognormal_oi_partials` machinery (the FD step operates on a smooth analytic function, so 2nd-order central differences with $h = 10^{-5}$ achieve $\le 10^{-3}$ relative error against a 4th-order Richardson reference; verified in the test suite).

At the canonical specification $(\mu_q = \log 100,\, \sigma_q = 0.10,\, T_{\mathrm{eff}} = 0.25,\, \kappa_v = 2,\, \alpha = 0.05,\, \beta = 1,\, \gamma = 1,\, a^\star = \log 100,\, v^\star = 0.04)$ — the same regime as §4.3.5 with $\kappa^\star = 17.806$, $\omega^\star = 1.177$ — the elasticities are

$$
\eta_{\sigma_q} := \frac{\sigma_q}{\kappa^\star}\frac{\partial \kappa^\star}{\partial \sigma_q} \;=\; -1.583,
\qquad
\frac{1}{\kappa^\star}\frac{\partial \kappa^\star}{\partial \mu_q} \;=\; +152.6\;\text{per log-strike unit}. \tag{21}
$$

$|\eta_{\sigma_q}| \approx 1.6$ is moderate — a $\pm X\%$ misspec in $\sigma_q$ produces a $\pm 1.6 X\%$ first-order misspec in $\kappa^\star$. The $\mu_q$ direction is **two orders of magnitude steeper**: a misalignment of the OI mean by $\Delta\mu_q = 0.001$ log-strike units produces $\Delta\kappa^\star/\kappa^\star \approx 15\%$.

The intuition: the closed-form aggregator (Eq. 15a) is a Gaussian in $a$ centred on $m(v) = \mu_q - (r - q + v/2) T_{\mathrm{eff}}$ with width $\tau(v)$. At the ATM equilibrium $a^\star \approx \mu_q$ (the dominant term in $\delta := a^\star - m(v^\star)$ is the $-v T_{\mathrm{eff}}/2 = -0.005$ shift), $G_y$ is small and changes sign nearby — small shifts in $\mu_q$ ratchet the equilibrium across the maximum of $G$ (where $G_y = 0$), drastically changing the linearisation that drives the Hopf condition. By contrast $G_v$ depends on $\sigma_q^2 + v T_{\mathrm{eff}}$ in quadrature, so $\sigma_q$ enters more gently.

##### Misspecification: when true OI is not log-normal

Empirical SPX OI exhibits structural multi-modality (calendar-effect clustering at the next quarterly + ATM). To bound the closed-form error against such a true OI we model the worst case via a symmetric mixture of two log-normals at $\mu_q \pm \Delta/2$ with equal weight and component spread $\sigma_{\mathrm{comp}} = 0.07$. The moment-matched single log-normal has $\hat\mu = \mu_q$, $\hat\sigma = \sqrt{\sigma_{\mathrm{comp}}^2 + (\Delta/2)^2}$. We compute (i) $\kappa^\star_{\mathrm{cf}}$ via the closed-form Eq. 17 at $(\hat\mu, \hat\sigma)$, and (ii) $\kappa^\star_{\mathrm{true}}$ via numerical Jacobian on the deterministic skeleton with $G$ built by direct quadrature against the actual mixture density. The two coincide at $\Delta = 0$ to numerical-quadrature precision ($< 10^{-4}$ relative); they diverge as

| $\Delta$ | $\hat\sigma$ | $\kappa^\star_{\mathrm{cf}}$ | $\kappa^\star_{\mathrm{true}}$ | $|\Delta\kappa^\star|/\kappa^\star$ |
|---:|---:|---:|---:|---:|
| $0.05$ | $0.074$ | $25.20$ | $25.14$ | $0.23\%$ |
| $0.10$ | $0.086$ | $21.87$ | $20.86$ | $4.83\%$ |
| $0.15$ | $0.102$ | — | — | — |
| $0.20$ | $0.122$ | $12.46$ | $5.68$ | $119\%$ |

(Full curve in `paper/figures/kappa_star_misspecification_curve.pdf`.) **Headline.** For bimodal separation $\Delta \le 0.10$ log-strike units (relative strike-range $\sim 10\%$) the closed form is robust to within $\sim 5\%$. Beyond $\Delta \approx 0.15$ the moment-matched single log-normal under-spreads the bimodal mass and the closed-form $\kappa^\star_{\mathrm{cf}}$ over-estimates by a factor of $2$–$3$. Empirical SPX bimodality is typically $\Delta \lesssim 0.05$–$0.10$, placing the closed form safely in the robust regime. The Phase-4 protocol therefore includes a Hartigan dip-test pre-screen on the empirical OI grid, with a $\Delta > 0.15$ flag that triggers a switch from Eq. 17 to the brute-force `kappa_star_brute_force_from_G` pipeline (numerical Jacobian on the empirical density without log-normal assumption).

##### Phase-4 calibration tolerance

Combining (21) with an independent-error budget allocation $\tfrac{1}{\sqrt 2}$ to each of $\sigma_q$ and $\mu_q$, the calibration tolerance for a target $|\Delta\kappa^\star/\kappa^\star|$ budget is

$$
\bigl|\Delta\sigma_q/\sigma_q\bigr| \;\le\; \frac{|\Delta\kappa^\star/\kappa^\star|/\sqrt{2}}{|\eta_{\sigma_q}|}, \qquad
\bigl|\Delta\mu_q\bigr| \;\le\; \frac{|\Delta\kappa^\star/\kappa^\star|/\sqrt{2}}{(1/\kappa^\star)|\partial \kappa^\star/\partial \mu_q|}. \tag{22}
$$

Concretely:

| Target $|\Delta\kappa^\star/\kappa^\star|$ | $\sigma_q$ tolerance | $\mu_q$ tolerance (log-strike) |
|---:|---:|---:|
| $5\%$ | $\pm 2.23\%$ relative | $\pm 0.00023$ |
| $10\%$ | $\pm 4.47\%$ relative | $\pm 0.00046$ |
| $25\%$ | $\pm 11.17\%$ relative | $\pm 0.00116$ |

In words: **predicting $\kappa^\star$ within $\pm 10\%$ at Phase 4 requires estimating $\sigma_q$ to $\pm 4.5\%$ relative and $\mu_q$ to $\pm 5 \times 10^{-4}$ log-strike units (i.e.\ $\pm 0.05\%$ relative strike error)**. The $\mu_q$ tolerance is the binding constraint and corresponds to centering the empirical OI mean on ATM to within $\sim\!5$bp — well within the precision of an SPX OI grid quoted on a $5$-strike-wide ladder. The $\sigma_q$ tolerance is the harder calibration target because moment matching against bimodal OI carries a structural bias (Section §4.3.6 paragraph above); when the empirical OI shows clear bimodality at $\Delta > 0.05$, the $\pm 10\%$ budget should be reduced or the brute-force pipeline used instead.

#### 4.3.7 Mixture-of-$K$-lognormals generalisation

The §4.3.6 fragility analysis identifies bimodal OI ($\Delta > 0.10$) as the regime where the single-lognormal closed form breaks structurally — not merely numerically. The mathematically correct fix is to lift the OI density from a single Gaussian-in-log-strike to a $K$-component mixture. Every step of §4.3 carries through verbatim by multilinearity, with the result that the closed-form $\kappa^\star$ becomes near-exact ($< 0.05\%$ relative error) at *all* empirically-relevant bimodality. Implementation: `src/reflexive_options/theory/bifurcation.py` (`MixtureOIComponent`, `G_mixture_lognormal_oi`, `G_mixture_lognormal_oi_partials`, `kappa_star_mixture_lognormal_oi`, `lyapunov_coefficient_mixture_lognormal_oi`); experiment runner: `experiments/mixture_oi_robustness.py`; tests: `tests/test_mixture_oi.py`; full writeup: `paper/mixture_oi_lyapunov.md`.

**Mixture aggregator.** Take a $K$-component mixture
$$
q(\log K) = \sum_{k=1}^K w_k \, \mathcal{N}(\log K; \mu_k, \sigma_k^2), \qquad w_k \ge 0, \;\; \sum_k w_k = 1. \tag{23}
$$
Eq.~14 is a linear functional of $q$, so the aggregator is itself a mixture:
$$
\boxed{\;G^{\mathrm{mix}}(a, v) = \sum_{k=1}^K w_k \, G_k(a, v),\;} \tag{24}
$$
where each $G_k$ is the single-lognormal closed form (Eq.~15a) with its own $(\mu_k, \sigma_k)$. By multilinearity of differentiation, every partial of $G^{\mathrm{mix}}$ inherits the weighted-sum form:
$$
\partial^{|\alpha|} G^{\mathrm{mix}} / \partial x^\alpha = \sum_k w_k \, \partial^{|\alpha|} G_k / \partial x^\alpha, \qquad \forall \alpha. \tag{25}
$$
In particular $(G^{\mathrm{mix}}_y, G^{\mathrm{mix}}_v) = \sum_k w_k (G_{y,k}, G_{v,k})$.

**Closed-form $\kappa^\star$ for the mixture.** The Routh–Hurwitz coefficients of §4.3.2 are *linear* in $(G_y, G_v)$. Substituting the mixture partials gives
$$
\boxed{\;\kappa^{\star,\mathrm{mix}} = \frac{G^{\mathrm{mix}}_y A^2 - G^{\mathrm{mix}}_v L - \sqrt{(G^{\mathrm{mix}}_v L - G^{\mathrm{mix}}_y A^2)^2 - 4 (G^{\mathrm{mix}}_y)^2 A (MA - L/2)}}{2 (G^{\mathrm{mix}}_y)^2 A},\;} \tag{26}
$$
which is Eq.~17 evaluated at the mixture-aggregated partials. The Hopf frequency is $\omega^{\star,\mathrm{mix}} = \sqrt{-\kappa^{\star,\mathrm{mix}} G^{\mathrm{mix}}_y A + M}$. K=1 collapses Eq.~26 to Eq.~17 to machine precision (verified in `tests/test_mixture_oi.py::test_K1_mixture_kappa_star_equals_single`).

**$\ell_1$ for the mixture.** The Kuznetsov contraction (Eq.~18) uses bilinear / trilinear tensors $B$, $C$ of the drift. Only the $f_1$ row contributes, and $f_1 = \mu - v/2 + \kappa G$. Multilinearity of $G$ in $\{w_k\}$ transfers to $B$, $C$:
$$
B^{\mathrm{mix}} = \sum_k w_k B_k, \qquad C^{\mathrm{mix}} = \sum_k w_k C_k. \tag{27}
$$
The eigenvectors $p, q$ at $\pm i\omega^{\star,\mathrm{mix}}$ are computed at the mixture Jacobian $J(\kappa^{\star,\mathrm{mix}}; G^{\mathrm{mix}}_y, G^{\mathrm{mix}}_v)$, which itself depends on $\{w_k\}$ through Eq.~26. The full $\ell_1^{\mathrm{mix}}$ is therefore a rational function of $(\{w_k, \mu_k, \sigma_k\}_{k=1}^K, \kappa_v, \alpha, \beta, \gamma)$ of degree bounded by the Kuznetsov contractions ($O(K^2)$ in the tensor terms, $O(K^4)$ from the eigenvector polynomials). For $K = 2$ the symbolic expression is roughly 3–5× longer than the single-lognormal $\ell_1$; for $K \geq 3$ it is more practical numerically than symbolically. The numerical pipeline `lyapunov_coefficient_mixture_lognormal_oi` evaluates $\ell_1^{\mathrm{mix}}$ in $O(K)$ time per call (one `G_lognormal_oi_partials` per component plus a single `compute_lyapunov_coefficient`).

**Multilinearity proof.** Let $T[q]$ denote any of the Kuznetsov contractions $\langle p, C(q, q, \bar q)\rangle$, $\langle p, B(q, J^{-1} B(q, \bar q))\rangle$, $\langle p, B(\bar q, (2 i \omega^\star I - J)^{-1} B(q, q))\rangle$ viewed as functionals of the OI density $q(\cdot)$ (with $\omega^\star$ and the eigenvectors regarded as $q$-dependent through $J(q)$). On the *constrained submanifold where $J$ is fixed* (i.e. holding $G_y$, $G_v$ constant), each term is bilinear in $C$ (one term) or trilinear in $B$ (two terms), and $B$, $C$ are linear in $q$ via Eqs.~14–15a. Hence the on-manifold $\ell_1$ is a polynomial of degree $\leq 3$ in $\{w_k\}$ at fixed $J$. Lifting back to the full mixture $\{w_k\}$-space introduces the nonlinear $J$-dependence through Eq.~26 but preserves rationality. ∎

**Numerical headline.** At the canonical regime $(\kappa_v = 2, \alpha = 0.05, \beta = 1, \gamma = 1, T_{\mathrm{eff}} = 0.25, v^\star = 0.04, \mu_q = a^\star = \log 100)$, with a symmetric bimodal mixture $w_1 = w_2 = 0.5$, $(\mu_1, \mu_2) = (\mu_q \mp \Delta/2)$, $\sigma_1 = \sigma_2 = 0.07$:

| $\Delta$ | $\kappa^\star_{\mathrm{true}}$ (FD) | $\kappa^\star$ K=1 single | $\kappa^\star$ K=2 mixture | K=1 rel.err | K=2 rel.err |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 25.142 | 25.201 | 25.139 | 0.232\%   | 0.013\% |
| 0.10 | 20.864 | 21.871 | 20.862 | 4.827\%   | 0.010\% |
| 0.20 |  5.677 | 12.460 |  5.677 | 119.482\% | 0.008\% |
| 0.30 |  2.458 |  7.809 |  2.458 | 217.733\% | 0.007\% |

The K=2 mixture closed form is essentially exact across the entire range — the residual $\sim 10^{-4}$ relative error is the *reference*-pipeline FD step, not the closed form itself. K=3 (the same K=2 bimodal plus a $w_3 = 0.10$ OTM-wing component at $\mu_q + 0.15$) sustains $< 0.05\%$ relative error across the same $\Delta$ range. See `paper/figures/mixture_oi_robustness_curve.pdf`.

**§4.3.6 robustness gap closure.** Under the single-lognormal assumption the binding $\mu_q$ tolerance was $\pm 5 \times 10^{-4}$ log-strike units for a $\pm 10\%$ $\kappa^\star$ budget. Under the mixture assumption the relevant calibration parameter is the full $\{(\mu_k, \sigma_k, w_k)\}_{k=1}^K$ tuple. The dominant new sensitivity is to the *component weights*: at $\Delta = 0.10$ a $\delta w_k$ misallocation produces an $O(\Delta) \cdot \delta w_k$ shift in $G^{\mathrm{mix}}_y$, a $\sim 20\times$ relaxation versus the $O(1) \cdot \delta \mu_q$ shift in the single-mode case. The empirical-OI weight estimation (which cluster carries what mass) is the operational bottleneck and is achievable with $\pm 5\%$ accuracy on a $5$-strike-wide ladder per cluster. The Phase-4 protocol therefore extends to: (i) a $K$-component Bayesian-information-criterion fit on the empirical OI grid (`make_mixture_lognormal_density` + a BIC/AIC selector — not yet implemented), (ii) closed-form $\kappa^\star$ via Eq.~26 evaluated at the fitted mixture, (iii) brute-force fallback only if the BIC-optimal $K$ exceeds the K=10 numerical-stability ceiling.

### 4.4 Codim-2 bifurcation structure

The codim-1 Hopf result (Theorem 1, §4) treats $\kappa^\star$ at fixed $(\sigma_q, \gamma)$. At the boundary of the Hopf region in the $(\sigma_q, \gamma)$ plane two codim-2 phenomena are admissible (Kuznetsov 2004 Ch. 8): the **Bautin** (degenerate Hopf, $\ell_1 = 0$) and the **Bogdanov–Takens** (BT, where the saddle-node curve $\{c_0 = 0\}$ coalesces with the Hopf curve $\{H = 0\}$). This subsection characterises both for the closed-form parameterization of §4.3 and reports the empirical scan; the matching presentation in `main.tex` lives at §3.6, Figure `fig:codim2-phase-diagram`, and Table `tab:bautin-anchors`.

#### 4.4.1 Bautin (degenerate Hopf)

When $\ell_1(\sigma_q, \gamma) = 0$, the supercritical / sub-critical sign flips. After centre-manifold reduction the local 2D normal form is (Kuznetsov 2004, §8.3)

$$
\dot \zeta = (\beta_1 + i\,\omega^\star)\zeta + a_1\,|\zeta|^2 \zeta + b_1\,|\zeta|^4 \zeta + O(|\zeta|^7), \qquad \zeta \in \mathbb{C},
$$

with $a_1/\omega^\star = \ell_1$ and $b_1/\omega^\star = \ell_2$ the second Lyapunov coefficient. The cusp in the $(\beta_1, a_1)$ plane organises a fold of cycles emanating from the Bautin point. Operationally the dynamics flip from a stable limit cycle past $\kappa^\star$ to hysteresis with abrupt jumps. Economically, a market parameterised by $(\sigma_q, \gamma)$ near the Bautin locus is *structurally fragile*: a small perturbation of OI dispersion or leverage flips the sign of $\ell_1$ and replaces a smooth cyclic regime with a discrete jump regime — catastrophic sensitivity to parameter drift. We do not compute $\ell_2$; the normal form above is reported as the locus structure prediction, not for quantitative use.

**Bautin curve at the canonical specification.** For the closed-form parameterization of §4.3, the Bautin condition is the implicit equation $\ell_1(\sigma_q, \gamma) = 0$ where $\ell_1$ is the rational expression of Eq. 18. We solve it by row-wise sign-change interpolation on the closed-form pipeline (`lyapunov_coefficient_lognormal_oi`, $\sim 50\,\mu$s per cell) at the canonical $(\mu_q, T_{\mathrm{eff}}, \kappa_v, \alpha, \beta) = (\log 100,\, 0.25,\, 2,\, 0.05,\, 1)$ specification on a $71 \times 97$ grid over $(\sigma_q, \gamma) \in [0.05, 0.40] \times [0.20, 5.00]$. Six anchor points along the curve:

| # | $\sigma_q$ | $\gamma$ | $\kappa^\star$ at the crossing |
|---|---:|---:|---:|
| 1 | 0.0598 | 0.55 | 2.15 |
| 2 | 0.1427 | 1.45 | 29.45 |
| 3 | 0.1995 | 2.35 | 52.59 |
| 4 | 0.2356 | 3.20 | 75.54 |
| 5 | 0.2638 | 4.10 | 100.74 |
| 6 | 0.2852 | 5.00 | 126.73 |

Along the curve $\kappa^\star$ grows monotonically with $\gamma$: stronger leverage feedback requires proportionally stronger reflexive coupling and OI dispersion to bifurcate. The supercritical region ($\ell_1 < 0$) is a narrow band bounded by the Bautin locus on one side and the Hopf-existence boundary $D = 0$ from Eq. 17 on the other; the band collapses entirely for $\sigma_q \lesssim 0.05$ and broadens monotonically with $\gamma$ in the moderate-$\sigma_q$ range. Within the scan window, the sub-critical region dominates the supercritical region by approximately $5{:}1$ in cell count — markets with mild dealer-gamma coupling are statistically biased toward hysteresis-and-jump rather than smooth cyclic behaviour.

#### 4.4.2 Bogdanov–Takens (BT) — Theorem 3

For the closed-form parameterization, the constant Routh–Hurwitz coefficient

$$
c_0(\kappa) = -\kappa\bigl(G_y\,\alpha\,\kappa_v + G_v\,\beta\,\gamma\bigr) + \tfrac{1}{2}\beta\,\gamma
$$

is linear in $\kappa$, so the saddle-node coupling admits the closed form

$$
\kappa_{\mathrm{SN}}(\sigma_q, \gamma) \;=\; \frac{\tfrac{1}{2}\,\beta\,\gamma}{G_y\,\alpha\,\kappa_v + G_v\,\beta\,\gamma}.
$$

The BT condition is $\kappa_{\mathrm{SN}} = \kappa^\star$ **and** $\kappa_{\mathrm{SN}} > 0$. The local 2D normal form is (Kuznetsov 2004 §8.4): $\dot x = y$, $\dot y = \beta_1 + \beta_2 x + x^2 + s\,xy$, $s = \pm 1$. BT generates homoclinic orbits and excitable spike-and-recovery dynamics.

> **Theorem 3** (BT locus empty in the canonical scan window).
> *On the $71 \times 97$ grid over $(\sigma_q, \gamma) \in [0.05, 0.40] \times [0.20, 5.00]$ at the canonical specification, $\kappa_{\mathrm{SN}}(\sigma_q, \gamma) \leq -1.31$ at every cell (range $[-68.05, -1.31]$), so the Bogdanov–Takens locus is empty in this scan window.*

**Proof sketch.** At the canonical specification, $G_v(\sigma_q) < 0$ throughout the scanned $\sigma_q$ range and $|G_v\,\beta\,\gamma| \gg |G_y\,\alpha\,\kappa_v|$ (ratio $\geq 13$ at every grid cell), so the denominator of $\kappa_{\mathrm{SN}}$ is uniformly negative while the numerator $\tfrac{1}{2}\beta\gamma$ is positive — giving $\kappa_{\mathrm{SN}} < 0$. The BT-empty conclusion is therefore a closed-form consequence of $G_v < 0$ dominating $G_y\,\alpha\,\kappa_v / (\beta\gamma)$ on the scanned window. Outside this window (extreme $\sigma_q$ or vanishing $\gamma$) the dominance argument can fail; we therefore restrict the claim to the scanned range and flag the unbounded-domain question as open. □

#### 4.4.3 Economic interpretation

The contrast between Bautin and BT is sharp: $\ell_1$ changes sign generically inside the physical range (Bautin curve non-trivial, §4.4.1), but $c_0$ never vanishes there ($\mathcal{B}_{\mathrm{BT}} = \emptyset$). The economic content of Theorem 3 is that the dealer-gamma + leverage parameter regime is structurally **Hopf-only within the scanned window**: there is no codim-2 BT point at which homoclinic orbits emanate, so the model does not generate the canonical excitable spike-and-recovery dynamics within its native parameter range. Burst-relax phenomena observed in real volatility surfaces near macro events must therefore originate from non-stationary parameter drift, regime-switching of the OI distribution, or higher-order (codim-3+) degeneracies — all outside the present model's autonomous skeleton. This is a falsifiable prediction: any empirical detection of a homoclinic-style spike-and-recovery mode at fixed dealer-gamma parameters would require revising the model.

**Implementation.** `src/reflexive_options/experiments/codim2_analysis.py` (full pipeline), `src/reflexive_options/theory/bifurcation.py` extensions for the Bautin sign-change extractor and BT residual. Tests: `tests/test_codim2_bifurcation.py` (8 tests). Figure: `paper/figures/codim2_phase_diagram.pdf` (left panel: four-region phase diagram with Bautin curve overlay; right panel: BT residual map).

#### 4.4.4 Bifurcations in the no-Hopf wedge — Theorem 6

Theorem 3 (BT-empty) addresses only one of the codim-1 instability routes inside the no-Hopf wedge $\mathcal{W}_{\mathrm{NH}}$: the saddle-node $c_0(\kappa) = 0$. Two further routes remain potentially open — the Hopf $H(\kappa) = 0$ (excluded by the wedge definition by construction) and the trace-flip $c_2(\kappa) = -\kappa G_y + (\alpha + \kappa_v) = 0$ (a real-eigenvalue crossing from the LHP into the RHP whenever $G_y > 0$). This subsection closes the codim-1 taxonomy of the model by showing that, at the canonical specification, every wedge cell is in fact globally asymptotically stable on the entire physical κ-half-line: no bifurcation of *any* codim-1 kind is accessible. The matching presentation in `main.tex` is the standalone subsection `saddle_node_no_hopf.md` (figure `fig:saddle-node-wedge`).

**Wedge definition (operational).** Let $H(\kappa) = A_2 \kappa^2 + A_1 \kappa + A_0$ with $(A_2, A_1, A_0)$ as in §4.3.2, and define
$$
\mathcal{W}_{\mathrm{NH}} \;:=\; \bigl\{(\sigma_q, \gamma) : H(\kappa) = 0 \text{ has no positive real root in } \kappa \in [0, \infty)\bigr\}. \tag{28}
$$
$\mathcal{W}_{\mathrm{NH}}$ is generated by two sub-cases: (i) the strict §3.5 wedge $D < 0$ where $D := A_1^2 - 4 A_2 A_0$, or (ii) $D \geq 0$ with both real roots of $H$ non-positive. At the canonical specification the strict case (i) is *empty* on the §3.7 scan window $(\sigma_q, \gamma) \in [0.05, 0.40] \times [0.20, 5.00]$; $\mathcal{W}_{\mathrm{NH}}$ is populated only via (ii), driven by $G_y < 0$ at the ATM-anchored equilibrium and the small-$\gamma$ baseline-stability condition
$$
H(0) = A_0 > 0 \;\Longleftrightarrow\; \gamma < \frac{2\,\alpha\,\kappa_v\,(\alpha + \kappa_v)}{\beta} \;\approx\; 0.41.
$$
Both classifier branches are implemented in `is_in_no_hopf_wedge`.

> **Theorem 6** (No-Hopf-wedge taxonomy).
> *Let $(\sigma_q, \gamma) \in \mathcal{W}_{\mathrm{NH}}$ at the canonical specification, and let $(G_y, G_v)$ be the log-normal-OI partials at $(a^\star, v^\star) = (\mu_q, \theta_v)$. Assume*
> $$
> \text{(S1)} \;\; G_y \leq 0, \quad
> \text{(S2)} \;\; G_y\,\alpha\,\kappa_v + G_v\,\beta\,\gamma \leq 0, \quad
> \text{(S3)} \;\; (\sigma_q, \gamma) \in \mathcal{W}_{\mathrm{NH}} \text{ and } A_2 > 0.
> $$
> *Then $c_2(\kappa) > 0$, $c_0(\kappa) > 0$, and $H(\kappa) > 0$ strictly for every $\kappa \geq 0$ — the equilibrium is asymptotically stable on the entire physical half-line and no codim-1 bifurcation occurs.*

**Proof.**
(S1) gives $c_2(\kappa) = -\kappa G_y + (\alpha + \kappa_v) \geq \alpha + \kappa_v > 0$ for $\kappa \geq 0$. (S2) gives $c_0(\kappa) = -\kappa\,(G_y\alpha\kappa_v + G_v\beta\gamma) + \tfrac{1}{2}\beta\gamma \geq \tfrac{1}{2}\beta\gamma > 0$ (the model assumes $\beta\gamma > 0$). For $H$: with $A_2 > 0$ the parabola opens upward, so the only failure mode would be a positive real root, ruled out by (S3); $H(0) = A_0$ is positive by the wedge classifier construction (cells with $A_0 \leq 0$ admit a positive root and are not in $\mathcal{W}_{\mathrm{NH}}$). All three strict inequalities together yield Liu's full Routh–Hurwitz criterion (Liu 1994); the spectral abscissa $\max_i \mathrm{Re}\,\lambda_i(J(\kappa))$ stays strictly negative for every $\kappa \geq 0$. ∎

**Numerical verification.** On the $41 \times 41$ grid over $(\sigma_q, \gamma) \in [0.02, 0.40] \times [0.05, 5.0]$ at the canonical specification, $123$ of $1{,}681$ cells fall inside $\mathcal{W}_{\mathrm{NH}}$ (all in the small-γ baseline-stable corner). All $123$ satisfy (S1)+(S2)+(S3) — Theorem 6 verdict (a) holds throughout. Sweeping $\kappa \in [0, 100]$ on $80$ points per cell and computing $\max_\kappa \max_i \mathrm{Re}\,\lambda_i(J(\kappa))$ as an independent numerical sanity-check: the maximum over all $123 \times 80$ samples is $-6.6 \times 10^{-3}$ — strictly negative, consistent with the closed-form verdict to working precision. Zero positive saddle-node $\kappa_{\mathrm{SN}}$ is detected at any wedge cell, sharpening Theorem 3 from "BT-empty on the §3.7 scan window" to "no SN-of-any-kind on the wedge subregion of the §4.4.4 scan window".

**Economic interpretation.** Theorem 6 closes the codim-1 bifurcation taxonomy: combined with Theorem 1 (Hopf at $\kappa^\star$ where the wedge does not apply) and Theorem 3 (saddle-node curve unphysical), the picture is complete. The wedge is the parameter region where dealer-gamma at the ATM-anchored equilibrium is locally decreasing in log-spot ($G_y \leq 0$, generic to the right tail of an ATM-centred OI distribution) and where the leverage flux $\beta\gamma$ is small enough to keep the bare Heston-with-memory triangle sub-critical at $\kappa = 0$. Inside this region, ramping the reflexive coupling $\kappa$ never destabilises the equilibrium — dealer-gamma acts as a *stabiliser* of the variance dynamics. The economic content: a market parameterised inside $\mathcal{W}_{\mathrm{NH}}$ is *structurally cycle-free* — no endogenous Hopf, no saddle-node jump, no trace-flip leak, no BT spike-and-recovery. Endogenous volatility cycles at a wedge-calibrated market can only originate from non-stationary parameter drift (the OI centre $\mu_q$ moving slowly toward the Hopf region), exogenous shocks, or a non-ATM equilibrium where $G_y > 0$ at $a^\star \neq \mu_q$. This is a sharp falsifiable Phase-4 prediction: an SPX-calibrated $(\sigma_q^{\mathrm{SPX}}, \gamma^{\mathrm{SPX}})$ inside $\mathcal{W}_{\mathrm{NH}}$ rules out endogenous reflexive cycles as the generator of empirically observed volatility clustering at that calibration.

**Implementation.** `is_in_no_hopf_wedge`, `bifurcations_in_no_hopf_wedge`, `NoHopfBifurcationResult`, `scan_no_hopf_wedge`, and the task-spec wrappers `NoHopfWedgeScanResult` + `scan_no_hopf_wedge_bifurcations` in `src/reflexive_options/theory/bifurcation.py`. Runner: `src/reflexive_options/experiments/saddle_node_wedge.py`. Tests: `tests/test_saddle_node_wedge.py` (4 tests). Figure: `paper/figures/saddle_node_wedge.pdf`. Standalone writeup: `paper/saddle_node_no_hopf.md`.

### 4.5 Numerical phase diagram

The full $(\kappa, \sigma_v, \xi, \rho)$ phase diagram is computed by `python -m reflexive_options.experiments.hopf_phase_scan_4d` and rendered in Figure~\ref{fig:hopf-phase-diagram}. The scan sweeps $\kappa \in [0, 2]$ on a 401-point grid for each of $31 \times 21 \times 4 = 2{,}604$ cells over $(\xi, \rho, \sigma_v)$ at the §4.2 Hopf-exhibiting regime; $(\xi, \rho)$ enter the deterministic Jacobian via the leading shear-induced correction $a_{\mathrm{eff}}(\kappa) = a(\kappa) + \tfrac{1}{2} \xi^2 \rho\, G_v$ (an Engel–Lamb–Rasmussen-style projection of the small-noise stochastic Hopf onto the eigenvalue envelope). The full Khasminskii $\Lambda(\kappa; \xi, \rho)$ via Algorithm 2 is too expensive at this resolution; the deterministic projection captures the qualitative geometry — including the no-Hopf wedge at high $\xi$ and strong negative $\rho$ — at $14.82$ s wall-clock on a single M-series core (`runs/hopf_phase_scan_4d/<ts>/metrics.json::elapsed_seconds`). The figure substantiates the §4.2 claim that real options markets sit *near but not across* $\kappa^\star$ throughout the SPX-relevant $(\xi, \rho)$ corner.

---

## 5. Stochastic lift

The deterministic Hopf indicates *when* the noiseless skeleton begins to oscillate. The full SDE (1) has multiplicative Heston noise; the relevant question (Arnold 1998) is when the top Lyapunov exponent $\lambda_1$ of the linearised RDS at the equilibrium changes sign.

For small noise intensity $\varepsilon$,

$$
\lambda_1(\kappa, \varepsilon) = \alpha(\kappa) + \varepsilon^2 \cdot \Lambda(\kappa) + O(\varepsilon^4),
$$

where $\alpha(\kappa)$ is the deterministic real part (zero at $\kappa^\star_{\mathrm{det}}$) and $\Lambda$ is computed numerically via the Khasminskii sphere process (Algorithm 2 in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §6). The reported $\Lambda$ at §4.2 is a finite-budget two-point estimate at $(\varepsilon_1, \varepsilon_2) = (0.05, 0.20)$ on the bare Heston-with-memory linearisation; magnitudes are $|\Lambda| \sim 10^{-3}$, signs are configuration-dependent. A definitive small-noise asymptotic check — including any test of the Engel–Lamb–Rasmussen (2024) shear-driven scaling — requires a dedicated experiment that we defer to the empirical phase, where the open-interest grid is the actual SPX surface rather than a synthetic prior.

---

## 6. Connection to "critical reflexivity" (Hardiman–Bercot–Bouchaud 2013) — Theorem 2

HBB find the Hawkes branching ratio $n \approx 1$ on E-mini mid-price changes over 1998–2011 — exactly the critical edge between sub- and super-criticality of endogenous events. We now formalise the connection to our $\kappa^\star$ via the **Bacry–Delattre–Hoffmann–Muzy (2013) diffusive limit** of an exponential-kernel Hawkes process.

### 6.1 Diffusive-limit derivation

**Setup.** Let $N(t)$ be a 1D self-exciting Hawkes process with intensity

$$
\lambda(t) \;=\; \mu \;+\; \int_{(-\infty, t)} \phi(t - s)\, dN(s) \;=\; \mu \;+\; \sum_{i\,:\,t_i < t} \phi(t - t_i),
$$

where $\mu > 0$ is the exogenous base rate, $\phi : \mathbb{R}_+ \to \mathbb{R}_+$ is the (causal, integrable) kernel, and the **branching ratio** is

$$
n \;:=\; \int_0^\infty \phi(s)\, ds.
$$

Stability of the population requires $n < 1$; the stationary mean intensity is then $\bar\lambda = \mu / (1 - n)$. For the canonical exponential kernel $\phi(s) = \alpha\,e^{-\beta s}$ with $\alpha, \beta > 0$, the branching ratio collapses to $n = \alpha/\beta$ and the centred intensity $\tilde\lambda(t) := \lambda(t) - \bar\lambda$ obeys an SDE-like recursion in the Markovian state representation $(\lambda(t))$.

**The BDHM theorem.** Bacry–Delattre–Hoffmann–Muzy (2013, *Annals of Applied Probability* 23(4), Theorem 2) prove the following diffusive-limit result. Consider the rescaling

$$
t \;\mapsto\; t / T, \qquad \lambda \;\mapsto\; T\,\bar\lambda(t/T),
$$

in the joint regime $T \to \infty$ and $n = n_T \to 1$ such that $T(1 - n_T) \to c \in (0, \infty)$. Then the rescaled centred intensity $\tilde\lambda_T(t)$ converges in law (in the Skorokhod topology) to the unique Ornstein–Uhlenbeck process

$$
d\bar\lambda(t) \;=\; -\,\beta\,(1 - n)\,\bar\lambda(t)\, dt \;+\; \sigma_\lambda\, dW(t), \qquad \sigma_\lambda^2 \;=\; \beta\,\bar\lambda_\infty.
$$

The deterministic relaxation rate of the rescaled centred intensity is $\beta(1 - n)$. Equivalently, the leading eigenvalue of the linearised intensity dynamics around the stationary mean is

$$
\lambda_{\max}^{(\mathrm{Hawkes})} \;=\; -\,\beta\,(1 - n),
$$

so that

$$
n \;=\; 1 \;+\; \frac{\lambda_{\max}^{(\mathrm{Hawkes})}}{\beta}.
$$

Criticality $n = 1$ corresponds exactly to $\lambda_{\max} = 0$. **The diffusive limit therefore furnishes a one-to-one map between the discrete-time Hawkes branching ratio and the continuous-time leading eigenvalue of the intensity dynamics.**

**Multivariate / kernel-universal generalisation.** Bacry–Mastromatteo–Muzy (2015, *Market Microstructure and Liquidity* 1(1), §2.4) extend the result to multivariate Hawkes processes with general (not necessarily exponential) kernels. The branching ratio $n$ is replaced by the **spectral radius** $\rho(\Phi)$ of the integrated kernel matrix $\Phi := \int_0^\infty \phi(s)\, ds \in \mathbb{R}^{d \times d}$. The criticality endpoint

$$
\rho(\Phi) \;=\; 1 \;\Longleftrightarrow\; \lambda_{\max} \;=\; 0
$$

is **universal across kernel shapes**: it depends only on the spectral radius of the integrated kernel, not on its temporal structure. This universality is the load-bearing fact for the equivalence with our Jacobian-eigenvalue formulation below — the SV-side $\lambda_{\max}(\kappa)$ and the Hawkes-side spectral radius coincide at criticality regardless of whether the empirical Hawkes kernel is exponential, power-law, or otherwise.

### 6.2 SV-equivalent branching ratio $n_{\mathrm{SV}}(\kappa)$

For our 3D reflexive skeleton with Jacobian $J(\kappa)$ from (3), let $\lambda_{\max}(\kappa) := \max_i \mathrm{Re}\,\lambda_i(J(\kappa))$. Define

$$
\boxed{\;n_{\mathrm{SV}}(\kappa) \;:=\; 1 \;+\; \frac{\lambda_{\max}(\kappa)}{\beta_0},\;} \qquad \beta_0 := \max_{\kappa \in [0, \kappa^\star]} \bigl(-\lambda_{\max}(\kappa)\bigr).
$$

The choice $\beta_0 = \max(-\lambda_{\max})$ is the SV analogue of the Hawkes baseline rate $\beta$ at $n = 0$. The **gauge-fixing** is necessary because the spot equation in deviation variables has a structural zero eigenvalue at $\kappa = 0$ under the constant-vol surrogate of §4.2 (the spot is a frozen mode in the noiseless skeleton at zero coupling), so the naive normalisation $\lambda_{\max}(0)$ is identically zero. Using the maximum-decay-rate as the baseline replaces the gauge zero with the gauge-invariant slowest-relaxation rate of the slow mode.

### 6.3 Theorem 2 — Hawkes-SV equivalence at the Hopf boundary

> **Theorem 2.** *Let $J(\kappa)$ be the Jacobian (3) of the 3D reflexive skeleton, satisfying the conditions of Theorem 1 so that $\kappa^\star \in (0, \kappa_{\max})$ is a Hopf threshold and $\lambda_{\max}(\kappa)$ is continuous in $\kappa$ with $\lambda_{\max}(\kappa) < 0$ for $\kappa \in (0, \kappa^\star)$, $\lambda_{\max}(\kappa^\star) = 0$, and $\partial\lambda_{\max}/\partial\kappa\rvert_{\kappa^\star} > 0$ (Hopf transversality). Then:*
>
> 1. *(critical-endpoint identity) $n_{\mathrm{SV}}(\kappa^\star) = 1$, exactly.*
> 2. *(local monotonicity at the Hopf boundary) There exists $\delta > 0$ such that $n_{\mathrm{SV}}$ is strictly increasing on $(\kappa^\star - \delta, \kappa^\star]$. Let $\kappa_{\mathrm{NS}}$ denote the smallest $\kappa \in [0, \kappa^\star)$ at which the leading eigenvalue is a complex pair; global monotonicity of $n_{\mathrm{SV}}$ on the entire interval $[\kappa_{\mathrm{NS}}, \kappa^\star]$ is verified numerically at the canonical regime (§6.4) but is not asserted as part of Theorem 2.*
> 3. *(Hawkes correspondence) For an exponential-kernel univariate Hawkes process the empirical $n_{\mathrm{Hawkes}} = 1 - |\lambda_{\max}^{(\mathrm{Hawkes})}|/\beta$, identical in form to (n_SV). The two coincide at criticality independently of the kernel shape (universality of the stability boundary, BMM 2015 §2.4).*

**Proof.** (1) is immediate from the definition: $\lambda_{\max}(\kappa^\star) = 0$ gives $n_{\mathrm{SV}}(\kappa^\star) = 1 + 0/\beta_0 = 1$. (2) follows by applying the implicit function theorem to the characteristic polynomial $P(\lambda; \kappa) = 0$ at the slow-mode complex pair, which is non-defective on a neighbourhood of $\kappa^\star$; $\mathrm{Re}\,\lambda_{\mathrm{pair}}(\kappa)$ is then smooth in $\kappa$ there and Hopf transversality $\partial\,\mathrm{Re}\,\lambda_{\mathrm{pair}}/\partial\kappa\rvert_{\kappa^\star} > 0$ persists on a one-sided open neighbourhood by continuity of the derivative, giving strict monotonicity of $n_{\mathrm{SV}} = 1 + \lambda_{\max}/\beta_0$ on $(\kappa^\star - \delta, \kappa^\star]$. The global statement on $[\kappa_{\mathrm{NS}}, \kappa^\star]$ does not follow from continuity alone — a continuous derivative can change sign without producing a second zero crossing of $\lambda_{\max}$ — and is reported as numerically verified (§6.4) rather than proven. (3) is the BDHM (2013) diffusive-limit identity, restated for the universal stability boundary via BMM (2015) §2.4. □

**Honest scope.** Theorem 2 is exact at the criticality endpoint and exact for the 1D exponential-kernel Hawkes globally. It is *approximate as a global Hawkes equivalence*: HBB's empirical $n$ is the $L^1$ norm of a fitted multivariate Hawkes kernel on order-flow events, not a direct mapping from the continuous SV state. The identification "Hardiman $n \approx 1 \Leftrightarrow$ market sits near $\kappa^\star$" rests on the universality of the $n = 1 \Leftrightarrow \lambda_{\max} = 0$ boundary across continuous-time reflexive systems, *not* on a path-by-path identity between event-counting Hawkes processes and our diffusion. Nevertheless, the criticality-endpoint identity is rigorous and gives the testable Phase-4 prediction below.

### 6.4 Numerical anchor (§4.2 canonical regime)

Implementation: `src/reflexive_options/theory/hawkes_equivalence.py`. Reproducer: `python -m reflexive_options.experiments.hawkes_sv_equivalence`. Evaluated on a 1001-point grid over $\kappa \in [0, 2\kappa^\star]$ at the §4.2 regime ($G_x = 0.5$, $G_v = -0.5$, $G_z = -0.5$, $\alpha = 0.5$, $\beta = 1$, $\gamma = 0.5$, $\kappa_v = 2$):

| Quantity | Value |
|---|---|
| $\beta_0$ | $0.2142$ |
| $\kappa_{\mathrm{ref}}$ at $\beta_0$ | $0.124$ (node–spiral transition; matches Theorem 2 claim 2) |
| $n_{\mathrm{SV}}(\kappa^\star_4)$ at published 4-decimal $\kappa^\star_4 = 0.8964$ | $0.99996$ |
| $\lvert n_{\mathrm{SV}}(\kappa^\star_4) - 1\rvert$ | $3.85 \times 10^{-5}$ (truncation error in 4-decimal $\kappa^\star_4$, *not* eigenvalue-solver noise) |
| $n_{\mathrm{SV}}(\kappa^\star)$ at higher-precision Brent root $\kappa^\star = 0.8964305216$ | $1$ ($\lvert n - 1\rvert < 10^{-15}$, the machine-$\varepsilon$ floor on a $3 \times 3$ matrix) |
| $n_{\mathrm{SV}}(2\kappa^\star)$ | $2.11$ |

The published 4-decimal anchor is retained for cross-section consistency with the §4.2 canonical-regime table; at machine-precision $\kappa^\star$ the identity $n_{\mathrm{SV}}(\kappa^\star) = 1$ is exact by construction (claim (1) of Theorem 2 is a definitional identity, not a statistical estimate).

Figure: `paper/figures/hawkes_sv_equivalence.pdf`. Top panel: $\mathrm{Re}\,\lambda_{\max}(J(\kappa))$ with the gauge zero at $\kappa = 0$, the most-stable point at $\kappa_{\mathrm{ref}} = 0.124$, and the Hopf threshold $\kappa^\star = 0.8964$ where $\lambda_{\max} = 0$. Bottom panel: $n_{\mathrm{SV}}(\kappa)$ with the empirical Hardiman $n \approx 1$ marked as a dashed reference line; past $\kappa^\star$ the curve enters the Hawkes-non-stationary regime $n_{\mathrm{SV}} > 1$.

### 6.5 Empirical SPX position in $\kappa$-space

Inverting the $n_{\mathrm{SV}}$ definition at the empirical Hardiman $n_{\mathrm{Hawkes}} = 1.0$ maps directly to $\lambda_{\max} = 0$, i.e., **the empirical SPX market sits exactly at $\kappa^\star$ under any reflexive SV calibration whose deterministic skeleton matches the empirical event-rate ACF in the diffusive limit**. This is a strong testable prediction for Phase 4: an SPX-calibrated $\kappa_0$ should satisfy $n_{\mathrm{SV}}(\kappa_0) \approx 1$.

Two caveats. First, the Hardiman $n = 1$ is a $\sim 5\%$-CI band, not a point estimate; the empirical anchor is the band $n_{\mathrm{Hawkes}} \in [0.95, 1.00]$. Second, in our *empirical-magnitude* regime ($G_x \sim 10^{-3}$, $\alpha = 252$/yr, σ²=v Heston backbone) there is no Hopf in the literature-prior range — a finding consistent with §4.2 — and a naive evaluation of $n_{\mathrm{SV}}$ at $\kappa_{\mathrm{market}} = 5 \times 10^{-12}$ gives $\approx 0.9998$ but this number is dominated by the $\lambda_{\max}(0) = 0$ gauge zero (the −1/2 from $\partial_v \sigma^2 = 1$ in $b(\kappa)$ shifts the spot eigenvalue only marginally), not by self-excitation. The correct empirical-phase test is therefore: (i) calibrate $G(\cdot)$ from the empirical SPX OI grid (so the closed-form $\kappa^\star$ from §4.3 Eq. 17 is the operative threshold); (ii) compute $n_{\mathrm{SV}}(\kappa_0)$ at the calibrated $\kappa_0$; (iii) compare against the Hardiman band on event-window-matched windows. We pre-commit this protocol to Phase 4.

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

#### 7.4.1 Follow-up: 2D bimodality scan with $\gamma > 0$

We executed both methodological revisions flagged above (re-run with $\gamma > 0$ active, test bimodality on the 2D joint density rather than the 1D log-spot marginal). Implementation: `src/reflexive_options/experiments/h_bimod_2d_scan.py` plus tests at `tests/test_h_bimod_2d_scan.py`.

**Setup.** §7.1 Heston backbone unchanged. Reflexive simulator with $\gamma = 0.5$ (active leverage feedback channel — closes the 3D Hopf channel per §1.1). Dealer-gamma aggregator over the §7.1 5×3 OI grid. 1000 paths × 2000 minute steps at $dt = 1/(252 \cdot 390)$, drop the first half as burn-in. The relative κ-grid is $\{0, 0.5, 0.9, 1.0, 1.05\}\cdot\kappa^\star_{\mathrm{env}}$ where $\kappa^\star_{\mathrm{env}} \approx 3.9 \times 10^{-9}$ is the simulator's *stability-envelope* upper bound determined by a pre-scan (`find_stability_envelope_kappa_star`). **Critical notation note:** $\kappa^\star_{\mathrm{env}}$ is *not* the deterministic Hopf threshold $\kappa^\star \approx 0.896$ of §4.2; the two are distinct objects defined at different parameter scales. The empirical-prior literature scale ($\kappa \sim 10^{-12}$) sits well inside this envelope; the §7.4 stability envelope is set by the high-OI dealer-gamma aggregator + γ > 0 + minute-bar discretisation.

**Test panel** (per κ): (i) Hartigan dip on the standardised-PCA leading direction of the joint $(\log S, v)$ sample cloud (1D test on the most informative direction); (ii) Silverman bandwidth test on each of the two channels separately; (iii) 2D KDE contour rendered for visual inspection.

**Result** (one representative seed = 42, n_paths = 1000):

| $\kappa$ | $\kappa / \kappa^\star_{\mathrm{env}}$ | n_finite (of $2 \times 10^4$ post-burn cells, kept-step $\times$ path) | PCA dip statistic | dip p-value | bimodal at 5%? |
|---:|---:|---:|---:|---:|---|
| $0$              | $0.00$ | 20{,}000 | $1.8\!\times\!10^{-3}$ | $0.98$ | no |
| $1.95\!\times\!10^{-9}$ | $0.50$ | 20{,}000 | $1.2\!\times\!10^{-3}$ | $1.00$ | no |
| $3.52\!\times\!10^{-9}$ | $0.90$ | 19{,}830 | $1.5\!\times\!10^{-3}$ | $0.99$ | no |
| $3.91\!\times\!10^{-9}$ | $1.00$ | 17{,}248 | $2.4\!\times\!10^{-3}$ | $0.68$ | no |
| $4.10\!\times\!10^{-9}$ | $1.05$ | 15{,}769 | $4.5\!\times\!10^{-3}$ | **$0.033$** | **yes** |

**H_bimod is now SUPPORTED in 2D PCA-projection at $\kappa = 1.05\,\kappa^\star_{\mathrm{env}}$, with the caveat that the supporting κ value sits just past the stability envelope.** The dip statistic is computed on $n = 15{,}769$ surviving cells out of $20{,}000$ post-burn-in (path × kept-step) entries — $\sim 79\%$ survival at this $\kappa$ in the canonical regime. The PCA principal-direction at $1.05\,\kappa^\star_{\mathrm{env}}$ has explained-variance ratio 0.564 — the bimodal axis is not aligned with either log-spot or variance individually but lives in the joint $(\log S, v)$ phase space, exactly the limit-cycle signature §7.4(b) flagged. The 2D KDE figure is `paper/figures/stationary_density_2d_kde.pdf`.

**Updated H_bimod outcome.** Was ✗ on the 1D log-spot marginal at $\gamma = 0$. Is **✓ (with caveats)** on the 2D $(\log S, v)$ joint at $\gamma > 0$ and $\kappa$ just past the stability envelope:

| Hypothesis | 1D scan ($\gamma = 0$) | 2D scan ($\gamma > 0$, this section) |
|---|---|---|
| H_bimod | ✗ (dip $p \geq 0.10$ across κ-grid) | ✓ at $\kappa = 1.05\,\kappa^\star_{\mathrm{env}}$ (PCA-projected dip $p = 0.033$); ✗ at $\kappa < \kappa^\star_{\mathrm{env}}$ |

The supporting evidence is at the *edge* of the simulator's stability envelope — the PCA-projected dip $p = 0.033$ at $\kappa = 1.05\,\kappa^\star_{\mathrm{env}}$ should be read as preliminary rather than definitive (the surviving sample is structurally selected: it is conditioned on path non-blowup, which itself shapes the joint distribution). The §7.5 summary table is updated to reflect this dual outcome. Run dir: `runs/h_bimod_2d/`.

### 7.5 Summary

| Hypothesis | Outcome at anchor | Notes |
|---|---|---|
| H_tail (heavier tails) | **Supported** | $\Delta$ excess kurtosis $\sim 10^5$; AD-rejection $p < 10^{-3}$ |
| H_skew (sign-tracking) | **Supported** | sign of skew matches $\mathrm{sgn}(G_x)$, opposite Heston |
| H_bimod (emergent bimodality) | **Not supported in 1D**; **supported in 2D PCA-projection** at $\kappa = 1.05\,\kappa^\star_{\mathrm{env}}$ with $\gamma > 0$ (§7.4.1) | 1D log-spot scan with $\gamma = 0$ negative across κ-grid; the §7.4(b) follow-up with $\gamma > 0$ on the 2D $(\log S, v)$ joint flips the outcome at $\kappa = 1.05\,\kappa^\star_{\mathrm{env}}$ on $\sim 79\%$-survival sample — preliminary, sample selection-conditioned |

Implementation: `src/reflexive_options/theory/stationary.py`. Tests: `tests/test_stationary.py`. The numbers above were produced by the scan in §7.1; the script is regenerable from the public functions `compare_to_heston`, `tail_index_vs_kappa_curve`, and `detect_bimodality`.

---

## 9. McKean–Vlasov mean-field limit of the dealer-gamma channel

§§2–7 treated the aggregate dealer gamma $G(S, t) = \sum_{K, T} q_{K,T}\, \Gamma_{K,T}(S, t)\, \mathrm{sgn}(K, T)$ as if it came from a single representative market-maker. Real markets have $n \sim 10^2$–$10^3$ dealers, each holding their own portfolio and hedging on their own clock. The §2 model implicitly assumes either (a) perfect coordination (all dealers act as one) or (b) the law-of-large-numbers limit where idiosyncratic dealer noise washes out. We now formalise (b) as a McKean–Vlasov SDE coupled to the *law* of the dealer-gamma process and quantify the threshold shift relative to the single-dealer model.

### 9.1 The $n$-dealer system

Each dealer $i \in \{1, \ldots, n\}$ holds a deviation gamma $G_i$ that follows an OU-style relaxation toward a common target $g(S, v)$ (e.g. the closed-form log-normal-OI aggregator of §4.3 evaluated at the current spot/variance), plus idiosyncratic noise:

$$
\boxed{\;dG_i \;=\; -\theta_G\,(G_i - g(S, v))\,dt \;+\; \sigma_G\, dW^i_G,\qquad i = 1, \ldots, n,\;} \tag{19a}
$$

with $\{W^i_G\}_{i=1}^n$ independent standard Brownian motions, $\theta_G > 0$ the dealer-hedging speed (autocorrelation time $\tau_G := 1/\theta_G$), and $\sigma_G \geq 0$ the idiosyncratic noise scale. The aggregate gamma feeding back into spot is the empirical mean

$$
\bar G_n(t) \;:=\; \frac{1}{n}\sum_{i=1}^n G_i(t),\qquad \frac{dS_t}{S_t} \;=\; \bigl(\mu + \kappa\,\bar G_n(t)\bigr)\,dt \;+\; \sigma(S_t, v_t)\,dW^S_t. \tag{19b}
$$

The variance and memory equations (1b)–(1c) are unchanged.

### 9.2 Theorem 3 — Propagation of chaos

In the limit $n \to \infty$, the empirical measure $\bar\mu_n^t = (1/n)\sum_i \delta_{G_i^t}$ converges weakly to the deterministic measure $\mu^t = \mathrm{Law}(G^t)$ where $G$ solves the McKean–Vlasov SDE

$$
dG \;=\; -\theta_G\,(G - g(S, v))\,dt \;+\; \sigma_G\,dW_G,\qquad \bar G_\infty(t) \;:=\; \int g\, d\mu^t(g) \;=\; \mathbb{E}[G(t) \mid \mathcal{F}_t^{S,v}], \tag{20}
$$

coupled with the spot equation $dS/S = (\mu + \kappa\,\bar G_\infty)\,dt + \sigma(S, v)\,dW^S$.

> **Theorem 3** (propagation-of-chaos $L^2$ rate; Sznitman 1991 Théorème I.1.4 / Méléard 1996 Prop 2.5 / Carmona–Delarue 2018 Vol I Thm 2.12).
> *Assume $g(\cdot)$ is Lipschitz in $(S, v)$, the initial conditions $G_i^0$ are i.i.d. with finite second moment $\mathrm{Var}(G_0) < \infty$, and the spot/variance path is fixed (i.e. condition the analysis on $\mathcal{F}_t^{S,v}$). Then for every $T > 0$,*
>
> $$
> \sup_{t \leq T} \mathbb{E}\bigl[\bigl(\bar G_n(t) - \bar G_\infty(t)\bigr)^2\bigr] \;\leq\; \frac{C(T)}{n}, \tag{21}
> $$
>
> *with*
>
> $$
> C(T) \;=\; \max\!\Bigl(\mathrm{Var}(G^0),\; \frac{\sigma_G^2}{2\theta_G}\bigl(1 - e^{-2\theta_G T}\bigr) + \mathrm{Var}(G^0)\,e^{-2\theta_G T}\Bigr) \;\leq\; \max\!\Bigl(\mathrm{Var}(G^0),\, \frac{\sigma_G^2}{2\theta_G}\Bigr). \tag{22}
> $$

**Proof (linear-target case).** Let $\delta_i(t) := G_i(t) - g(S(t), v(t))$. With $g$ deterministic on the conditioned path, $\delta_i$ is itself an OU process with $d\delta_i = -\theta_G\,\delta_i\,dt + \sigma_G\,dW^i_G - dg$. Independence of $\{W^i_G\}_i$ gives $\mathrm{Cov}(\delta_i(t), \delta_j(t)) = 0$ for $i \neq j$, so

$$
\mathrm{Var}(\bar G_n(t)) \;=\; \frac{1}{n}\,\mathrm{Var}(G_i(t) - g(S(t), v(t))) \;=\; \frac{1}{n}\Bigl[\mathrm{Var}(G^0)\,e^{-2\theta_G t} + \frac{\sigma_G^2}{2\theta_G}(1 - e^{-2\theta_G t})\Bigr]. \tag{23}
$$

Taking the supremum over $t \in [0, T]$ gives (21)–(22). The bound is *tight* for this OU structure: the standard Sznitman argument (compactness of empirical measures + uniqueness of MV SDE solutions under Lipschitz coefficients) gives the same $C/n$ rate but with a possibly looser constant from the Grönwall closure; the explicit OU calculation here is sharper. $\square$

### 9.3 Effect on the Hopf threshold

The MV system inserts a low-pass filter between the spot/variance state and the aggregate gamma fed back into spot. At the Hopf frequency $\omega^\star$ from §3, the linearised transfer function from the target perturbation $\delta g$ to the aggregate $\bar G_\infty$ is

$$
\widehat{\bar G}_\infty(\omega) / \widehat{\delta g}(\omega) \;=\; \frac{\theta_G}{\theta_G + i\omega} \;\Rightarrow\; \bigl|\widehat{\bar G}_\infty / \widehat{\delta g}\bigr|(\omega^\star) \;=\; \frac{\theta_G}{\sqrt{\theta_G^2 + \omega^{\star 2}}}. \tag{24}
$$

The effective coupling at the Hopf frequency is therefore $\kappa_{\mathrm{eff}}(\omega^\star) = \kappa \cdot \theta_G / \sqrt{\theta_G^2 + \omega^{\star 2}}$, and the MV Hopf threshold expands by the reciprocal of the gain:

$$
\boxed{\;\frac{\kappa^\star_{\mathrm{MV}}}{\kappa^\star_{\mathrm{single}}} \;=\; \frac{\sqrt{\theta_G^2 + \omega^{\star 2}}}{\theta_G} \;=\; \sqrt{1 + (\omega^\star \tau_G)^2} \;=\; 1 \;+\; \tfrac{1}{2}(\omega^\star \tau_G)^2 \;+\; O\bigl((\omega^\star \tau_G)^4\bigr).\;} \tag{25}
$$

Two regimes are immediate:
- **Instantaneous hedging $\theta_G \to \infty$ (i.e. $\tau_G \to 0$):** the ratio tends to 1, so MV agrees with the single-dealer model. This is the implicit assumption of §§2–7.
- **Slow hedging $\theta_G < \omega^\star$:** the ratio is strictly $> 1$, so the MV threshold is *higher* than the single-dealer threshold. The dealer ensemble damps the feedback channel, requiring stronger coupling to destabilise. The leading correction is $O((\omega^\star \tau_G)^2)$, so for $\omega^\star \tau_G = 0.1$ the MV correction is $\sim 0.5\%$; for $\omega^\star \tau_G = 1$ the threshold is $\sqrt{2} \times$ the single-dealer value.

**Numerical anchor at the canonical regime.** At the §4.3 canonical specification ($\sigma_q = 0.10$, $T_{\mathrm{eff}} = 0.25$, $\kappa_v = 2$, $\alpha = 0.05$, $\beta = 1$, $\gamma = 1$, $\mu_q = \log 100$, $v^\star = 0.04$) the closed-form Hopf threshold and frequency are $\kappa^\star_{\mathrm{single}} = 17.81$ and $\omega^\star = 1.18$ rad/yr. With a representative dealer-hedging speed of $\theta_G = 50$/yr (autocorrelation time $\tau_G \approx 5$ trading days), the MV threshold ratio is $\sqrt{1 + (1.18/50)^2} = 1.000277$, i.e. $\kappa^\star_{\mathrm{MV}} = 17.811$ — a $2.8 \times 10^{-4}$ relative shift, operationally negligible. For a slower hedging cycle ($\theta_G = 5$/yr, $\tau_G \approx 50$ trading days) the ratio jumps to $\sqrt{1 + (1.18/5)^2} = 1.0273$, a $2.7\%$ shift that is just at the edge of empirical detectability.

### 9.4 Conditions and scope

Theorem 3 requires (i) Lipschitz $g(\cdot)$ in $(S, v)$ — satisfied by the closed-form log-normal-OI aggregator (15a) on any compact $(S, v)$ neighbourhood of the equilibrium; (ii) finite second moment of $G_i^0$ — a standard initial-condition assumption; and (iii) bounded dealer-hedging-speed dispersion across the population (we have taken $\theta_G$ uniform across dealers; heterogeneous $\theta_G^{(i)}$ requires a propagation-of-chaos argument over the joint $(G_i, \theta_G^{(i)})$ measure, which the same Sznitman framework handles modulo a doubled state space).

The threshold-shift formula (25) is exact at the *linearisation* of the MV system around the equilibrium and matches the single-dealer linearisation in the $\tau_G \to 0$ limit. Higher-order corrections to $\kappa^\star_{\mathrm{MV}}$ from nonlinearities in $g(S, v)$ are $O(\sigma_G^2)$ and contribute to the stochastic-Hopf shift $\Lambda$ rather than the deterministic threshold; they are absorbed into the §5 stochastic-lift framework.

### 9.5 Numerical validation

Implementation: `src/reflexive_options/theory/mckean_vlasov.py` (the closed-form $C(T)$, the Hopf-threshold-shift formula, and the n-particle Euler–Maruyama simulator). Reproducer: `python -m reflexive_options.experiments.mckean_vlasov_validation`.

We sweep $n \in \{10, 32, 100, 316, 1000\}$ at the canonical regime ($\sigma_G = 0.05$, $\theta_G = 50$/yr, $T = 0.25$ yr, 250 Euler steps, 64 replicates per $n$, locked seed $20260514$) and measure $\sup_t \sqrt{\mathbb{E}[(\bar G_n(t) - \bar G_\infty(t))^2]}$. The fitted log-log slope of RMSE vs $1/\sqrt n$ is **$0.951$** (theoretical $1.0$), confirming the Sznitman $1/\sqrt n$ scaling within finite-sample noise. The empirical RMSE sits *below* the closed-form Sznitman bound $\sqrt{C(T)/n}$ across all $n$ in the sweep — consistent with the sharp OU constant being an *upper* bound on the worst-case path realisation.

Figure: `paper/figures/mckean_vlasov_propagation_chaos.pdf` (RMSE vs $n$ on log-log axes with the $\sqrt{C(T)/n}$ reference line and the LS fit). Run dir: `runs/mckean_vlasov_validation/`.

---

## 10. Information-theoretic reflexivity — Theorem 5

§§3 and 6 characterise the critical edge $\kappa^\star$ from two angles: bifurcation-theoretic (Routh–Hurwitz: $H(\kappa^\star) = 0$ with a transversal complex pair) and Hawkes-equivalent (BDHM-diffusive: $n_{\mathrm{SV}}(\kappa^\star) = 1$). Both are *spectral* statements. This section adds an *information-theoretic* characterisation: how much does the dealer-gamma channel contribute, in Shannon-entropy units, to the predictability of future returns from past spot prices? The result complements Theorem 4 from the same critical-edge question, attacking it now from the angle of statistical information rather than dynamical stability.

### 10.1 Setup — excess entropy of the dealer-gamma channel

Let $R_\tau := \int_0^\tau dS_s/S_s = y_\tau - y_0$ in log-deviation coordinates. The *excess entropy* of the spot process at horizon $\tau$ is

$$
E_\tau(\kappa) \;:=\; I\bigl(\mathcal{F}_{(-\infty, 0]}^y\,;\, R_\tau \,\big|\, v_0, z_0\bigr), \tag{26}
$$

the conditional mutual information between the past-spot history and the integrated future log-return, conditioned on the present non-spot state $(v_0, z_0)$. This is the natural measure of "how much past-spot information is needed to predict the next $\tau$-worth of returns, beyond what the present variance and memory state already tell us".

The conditioning kills the contribution of past spot that has been absorbed into the present $(v_0, z_0)$ — it isolates the *direct* feedback channel from past spot to future returns mediated by the dealer-gamma drift $\kappa G(\cdot)$. At $\kappa = 0$, the SDE (1a) reduces to standard Heston: future returns are a deterministic function of $v_0$ plus integrated noise, completely independent of past spot. So $E_\tau(0) = 0$ exactly — the Markov closure. For $\kappa > 0$, $G$ depends on the current spot $S$, so past spot enters the future drift, and $E_\tau(\kappa) > 0$.

For the linearised 3D OU $dx = J(\kappa) x\,dt + \Sigma\,dW$ around the equilibrium (constant-vol surrogate of §4.2, with $\Sigma = \mathrm{diag}(\sqrt{\theta_v},\,\xi\sqrt{\theta_v},\,0)$), the conditional MI admits a closed form. Let $P$ solve the Lyapunov equation $J P + P J^\top + \Sigma\Sigma^\top = 0$ (well-defined whenever $J(\kappa)$ is Hurwitz, i.e. for $\kappa \in (0, \kappa^\star)$). Then by Gaussian conditioning + the Markov property of the full 3D state:

$$
\boxed{\;E_\tau(\kappa) \;=\; \tfrac{1}{2}\,\log\!\left(1 \;+\; \frac{v_1^2 \cdot \sigma^2_{y \mid u, z}}{m_y(\tau)}\right),\;} \tag{27}
$$

where

$$
v_1 := \bigl[e^{J(\kappa)\tau} - I\bigr]_{11}, \quad
\sigma^2_{y \mid u, z} := P_{11} - P_{1,(2,3)}\, P_{(2,3),(2,3)}^{-1}\, P_{(2,3),1}, \quad
m_y(\tau) := \bigl[P - e^{J\tau} P\, e^{J^\top \tau}\bigr]_{11}.
$$

$v_1$ is the (1,1) entry of the transition operator minus identity (how much the linearised future spot depends on present spot, scalar); $\sigma^2_{y|u,z}$ is the Schur complement of $P$ — the residual variance of $y_0$ after conditioning on $(u_0, z_0)$; and $m_y(\tau)$ is the conditional variance of $R_\tau$ given the full present state, i.e. the *noise floor*. Both numerator factors vanish at $\kappa = 0$ ($v_1 = 0$ because the first row of $J(0)$ is identically zero in the constant-vol surrogate, so $e^{J(0)\tau}$ has first row $(1, 0, 0)$), recovering Markov closure.

### 10.2 Theorem 5 — Critical excess entropy at the Hopf boundary

> **Theorem 5.** *Assume the conditions of Theorem 1 hold, so that $J(\kappa)$ is Hurwitz for $\kappa \in (\kappa_{\mathrm{NS}}, \kappa^\star)$ with a complex pair $\lambda_\pm(\kappa) = \alpha(\kappa) \pm i\omega(\kappa)$ crossing the imaginary axis at $\kappa^\star$ with $\partial\alpha/\partial\kappa\rvert_{\kappa^\star} > 0$ (Hopf transversality). Let $E_\tau(\kappa)$ be defined by (26) and computed via (27). Then for any fixed $\tau > 0$:*
>
> 1. *(Markov closure)* $\lim_{\kappa \to 0^+} E_\tau(\kappa) = 0$.
> 2. *(Finite saturation at criticality)* $E_\tau(\kappa^\star) := \lim_{\kappa \uparrow \kappa^\star} E_\tau(\kappa)$ exists and is finite, with $E_\tau(\kappa^\star) > 0$.
> 3. *(Mean-field critical exponent $\beta = 1$)* The approach to the saturation is linear,
>    $$E_\tau(\kappa^\star) - E_\tau(\kappa) \;=\; C_\tau \cdot (\kappa^\star - \kappa) \;+\; O\bigl((\kappa^\star - \kappa)^2\bigr)$$
>    with $C_\tau > 0$.
> 4. *(Local monotonicity)* There exists $\delta > 0$ such that $\partial E_\tau / \partial \kappa > 0$ on $(\kappa^\star - \delta, \kappa^\star)$.

The *global* monotonicity claim — $\partial E_\tau / \partial \kappa \geq 0$ on the entire stable interval $(0, \kappa^\star)$ — is verified **numerically** at the canonical regime (§10.4) but is not asserted as part of Theorem 5. The strongest defensible claim is local monotonicity on a one-sided neighbourhood of $\kappa^\star$, which is what Theorem 5(4) states.

### 10.3 Proof

**(1) Markov closure.** At $\kappa = 0$ the constant-vol surrogate has Jacobian $J(0)$ with first row identically zero. Hence $J(0)^k$ has first row zero for all $k \geq 1$, so $e^{J(0)\tau}$ has first row $(1, 0, 0)$, giving $v_1 = 0$. The numerator of (27) vanishes and $E_\tau(0) = \tfrac{1}{2}\log 1 = 0$. By continuity of $v_1$ in $\kappa$ (the matrix exponential is analytic in $J$ which is analytic in $\kappa$), $\lim_{\kappa \to 0^+} E_\tau(\kappa) = 0$. $\square$

**(2) Finite saturation.** As $\kappa \uparrow \kappa^\star$ the leading complex pair $\lambda_\pm(\kappa)$ has real part $\alpha(\kappa) \to 0$. The stationary covariance $P(\kappa)$ that solves the Lyapunov equation diverges in the direction of the slow-mode eigenvector (i.e., $\lVert P(\kappa) \rVert \to \infty$), with the leading divergence $\propto 1/|\alpha(\kappa)|$ in the rank-1 projection onto the eigenvector. *But this divergence is coherent across the three state components* — it lives in a 1D subspace spanned by the eigenvector $q = (q_y, q_u, q_z)$ of $\lambda_+(\kappa^\star)$. The conditional variance $\sigma^2_{y|u,z}$ — the Schur complement of the rank-1-plus-bounded $P$ — extracts the component of the diverging mode that is orthogonal to $\mathrm{span}\{e_2, e_3\}$ in the metric induced by the bounded-part of $P$. After algebra (Carlson 1986 §3 on Schur complements of low-rank perturbations of positive-definite matrices), the Schur complement remains *finite* in the limit, with explicit form $\sigma^2_{y|u,z}(\kappa^\star) = q_y^2 / (q_u^2 \beta_u + q_z^2 \beta_z)$ for some positive constants $\beta_u, \beta_z$ determined by the bounded part of $P$. The factor $v_1$ in (27) is bounded — $\lVert e^{J\tau} \rVert$ stays bounded for fixed $\tau$ when one eigenvalue's real part touches zero — and $m_y(\tau)$ likewise stays bounded. Hence $E_\tau(\kappa^\star)$ is finite. Positivity at $\kappa^\star$ is immediate from positivity at any interior point + monotonicity (proved next). $\square$

**(3) Mean-field linear exponent.** Plug into (27) and expand $\sigma^2_{y|u,z}(\kappa^\star - \delta) = \sigma^2_{y|u,z}(\kappa^\star) - C_1 \delta + O(\delta^2)$ and similarly for $v_1, m_y(\tau)$ — each is real-analytic in $\kappa$ on the punctured neighbourhood $(\kappa^\star - \delta_0, \kappa^\star) \cup (\kappa^\star, \kappa^\star + \delta_0)$ because the Lyapunov-equation solution is analytic in the Hurwitz matrix (Bhatia 1997 §VII). The logarithm in (27) is then a smooth function of $(v_1^2 \sigma^2 / m)$, and a first-order Taylor expansion gives the linear scaling claimed. The constant $C_\tau = \tfrac{1}{2} \cdot d/d\kappa[\log(1 + v_1^2 \sigma^2 / m)]\rvert_{\kappa^\star}$ collects all derivative terms. $\square$

**(4) Local monotonicity.** $C_\tau$ in (3) is the leading-order coefficient of the linear approach. We claim $C_\tau > 0$. The fastest-growing contribution near $\kappa^\star$ is from $\sigma^2_{y|u,z}$: the Schur complement's bounded-part denominator $q_u^2 \beta_u + q_z^2 \beta_z$ *decreases* with $\delta = \kappa^\star - \kappa$ because the bounded part of $P$ has a sub-leading $\delta$-contribution from the off-diagonal Lyapunov terms — explicitly, both $\beta_u$ and $\beta_z$ are increasing in $\alpha(\kappa) = O(\delta)$. So $\sigma^2_{y|u,z}$ *increases* as $\delta \to 0$, $v_1$ stays bounded above 0, $m_y(\tau)$ has only a sub-leading $O(\delta)$ correction, and the ratio $v_1^2 \sigma^2 / m$ is monotonically increasing in $\kappa$ on $(\kappa^\star - \delta, \kappa^\star)$. The log is monotone, so $E_\tau$ is too. The sign of the leading coefficient $C_\tau$ is positive. $\square$

**Honest scope.** The proof of (4) gives *local* monotonicity on a one-sided neighbourhood of $\kappa^\star$; the sign argument relies on the explicit Schur-complement asymptotics near criticality and does not propagate to the whole $(0, \kappa^\star)$ interval. Global monotonicity is checked numerically on the canonical 101-grid in §10.4 — and is *empirically true* in our regime — but a closed-form proof on the entire interval would require Routh-Hurwitz-style sign tracking of $\partial E_\tau/\partial \kappa$ as the Jacobian moves through the node-spiral transition at $\kappa_{\mathrm{NS}} \approx 0.124$, which we leave open.

### 10.4 Numerical anchor (§4.2 canonical regime)

Implementation: `src/reflexive_options/theory/info_theoretic.py`. Reproducer: `python -m reflexive_options.experiments.info_theoretic_excess_entropy`. Evaluated on a 101-point κ-grid over $\kappa \in (10^{-4}, \kappa^\star)$ at the §4.2 regime with the canonical Heston diffusion ($\theta_v = 0.04$, $\xi = 0.3$):

| Quantity | $\tau = 0.1$ yr | $\tau = 1$ yr | $\tau = 5$ yr |
|---|---:|---:|---:|
| $E_\tau(10^{-4})$ (Markov-limit anchor) | $2.5 \times 10^{-10}$ | $2.5 \times 10^{-9}$ | $1.3 \times 10^{-8}$ |
| $E_\tau(\kappa^\star^-)$ (saturation) | $0.0168$ | $0.0530$ | $0.4135$ |
| Critical-edge enhancement ratio | $\sim 2 \times 10^8$ | $\sim 2 \times 10^8$ | $\sim 3 \times 10^8$ |
| Monotone on $(10^{-4}, \kappa^\star)$? | yes | yes | yes |
| Fitted $\hat\beta$ at boundary | $0.998$ | $0.998$ | $1.000$ |

The fitted $\hat\beta \approx 1$ to 3 decimals matches the Theorem 5(3) mean-field prediction. The critical-edge enhancement ratio of $\sim 10^8$ is dominated by the small-$\kappa$ anchor; the more interpretable headline is the *saturation value itself* — $E_1(\kappa^\star) \approx 0.05$ nats per year of horizon, or roughly $7\%$ of a uniform-bin's worth of predictability beyond what $(v_0, z_0)$ encode. This is operationally meaningful: a reflexive market at the critical edge leaks $\sim 0.05$ nats of return-direction information per year through the dealer-gamma channel alone.

Figure: `paper/figures/excess_entropy_curve.pdf`. Top panel: $E_\tau(\kappa)$ vs $\kappa$ at the three τ values with the $\kappa^\star$ anchor marked; bottom panel: log-log inset showing the $(\kappa^\star - \kappa)^1$ linear approach to saturation (slope = $\beta = 1$).

### 10.5 Structural insight — why the excess entropy does NOT diverge

The naive Crutchfield-Feldman intuition for second-order phase transitions ("statistical complexity diverges at criticality") *fails* in our setting, and the reason is itself structurally informative. The slow-mode collapse at the Hopf bifurcation is *coherent across $(y, u, z)$* — it lives in a 1D eigenvector subspace spanned by $q = (q_y, q_u, q_z)$ rather than in the $y$-direction alone. Conditioning $E_\tau$ on the present $(u_0, z_0)$ removes the component of past spot that flows through that *coherent* slow mode (which is the divergent one); what remains is the *orthogonal* component, which the bounded part of $P$ governs and which stays finite.

This is a positive finding rather than a defect of the formulation. It says: the dealer-gamma channel at criticality is informative, but its information content is *capped* by the Schur complement of the bounded part of $P$ — explicitly $\sigma^2_{y|u,z}(\kappa^\star) = q_y^2 / (q_u^2 \beta_u + q_z^2 \beta_z)$. The ratio $q_y^2 / (q_u^2 + q_z^2)$ — the eigenvector's "spot-purity" at criticality — is the structural object that sets the saturation. A market whose Hopf eigenvector is mostly $(u, z)$-loaded ($q_y$ small) saturates at a small $E_\tau(\kappa^\star)$ even at criticality, whereas a market whose Hopf eigenvector is mostly $y$-loaded saturates at a large $E_\tau(\kappa^\star)$. This is a parametric prediction that ties qualitative critical behaviour to spot-vs-vol mode partition at the boundary — testable in Phase 4 once the eigenvector decomposition is calibrated from SPX data.

### 10.6 Phase-4 testable prediction — Corollary to Theorem 5

> **Corollary.** *For an SPX market window with calibrated dealer-gamma series $\hat G_t$ and observed log-returns $\hat r_{t+1}$, the empirical Schreiber-2000 transfer entropy*
> $$\hat T_{G \to r} \;:=\; \widehat{H(r_{t+1} \mid r_t)} \;-\; \widehat{H(r_{t+1} \mid r_t, G_t)}$$
> *should be statistically significant ($p < 0.05$) under an IAAFT-surrogate null on the source series, AND should be larger on event windows where the system is conjectured to sit closer to $\kappa^\star$ (Volmageddon Feb 2018, COVID Mar 2020, Yen carry Aug 2024) than on quiescent windows.*

The prediction has two pieces — (i) directional significance under IAAFT, (ii) event-window dependence — and is logically independent of Theorem 4's $n_{\mathrm{SV}}(\kappa_0) \approx 1$ prediction. The IAAFT null preserves $G$'s marginal and linear ACF while randomising its nonlinear cross-coupling to $r$, so the test asks whether the *nonlinear feedback channel* is informative beyond what $G$'s own autocorrelation explains. This is the empirical analogue of $E_\tau(\kappa) > 0$ in the model — and the event-window dependence is the empirical analogue of $E_\tau$ growing with $\kappa$ along the critical-approach trajectory. Implementation: `transfer_entropy_iaaft_pvalue` in `info_theoretic.py`.

### 10.7 Relation to Theorem 4 and the Crutchfield literature

Theorem 4 (§6) characterises $\kappa^\star$ via the Bacry-Delattre-Hoffmann-Muzy diffusive-limit identity $n_{\mathrm{SV}}(\kappa^\star) = 1$. Theorem 5 characterises the same boundary via the saturation $E_\tau(\kappa^\star) > 0$ of the conditional excess entropy. The two are *independent* characterisations of the critical edge, not consequences of each other:

- $n_{\mathrm{SV}}$ is a *spectral* quantity (leading eigenvalue real part, normalised), insensitive to the off-spectrum structure of $J(\kappa)$.
- $E_\tau$ is an *information-theoretic* quantity that depends on the full Lyapunov-equation solution $P(\kappa)$ — both the eigenstructure AND the noise covariance $\Sigma\Sigma^\top$.

Two markets with the same $\lambda_{\max}(\kappa)$ trajectory can have very different $E_\tau$ curves if their noise structures differ. The two theorems are therefore complementary diagnostics: $n_{\mathrm{SV}}$ measures *how close to the Hopf boundary the system is*, while $E_\tau$ measures *how much past-spot information the system actually leaks through the dealer-gamma channel*.

Relation to the Crutchfield "complexity at criticality" literature (Crutchfield-Feldman 2003 *Chaos*, Crutchfield 2012 *Nat Phys*): for many lattice models the statistical complexity $C_\mu$ — the entropy of causal states — *diverges* at second-order critical points. Our finding $E_\tau(\kappa^\star) < \infty$ is *not* a contradiction: $E_\tau$ is a *conditional* mutual information that subtracts off the contribution of the present non-spot state, which is precisely the projection onto the coherent slow mode that would have produced the divergence. An unconditional excess-entropy proxy on the spot path alone *would* diverge at $\kappa^\star$ (it would inherit the $1/|\alpha(\kappa)|$ blowup of the stationary variance of the slow mode). The conditional formulation is the natural information-theoretic anchor for the dealer-gamma channel because it isolates the feedback signal from the non-feedback common-mode noise. This is in line with the more recent Lizier-Prokopenko-Zomaya (2012) "active information storage" framework, where the relevant question is the predictive information beyond a fixed-history baseline — exactly the role $(v_0, z_0)$ plays here.

---

## 8. Open items (theory)

1. ~~Closed-form $\ell_1$ for log-normal OI in moneyness~~ — **resolved** in §4.3; closed form (Eq. 17–18) implemented in `lyapunov_coefficient_lognormal_oi`, supercritical at the canonical regime, parametric phase boundary in $(\sigma_q, \gamma)$ space rendered in `paper/figures/ell1_phase_boundary.pdf`.
2. Closed-form $\Lambda(\kappa)$ for the stochastic-Hopf shift (Engel–Lamb–Rasmussen-style asymptotics adapted to the Heston multiplicative-noise structure).
3. ~~Formal Hawkes-$n$ ↔ SV-eigenvalue reduction~~ — **resolved at the criticality endpoint** in §6 via Theorem 2 (BDHM 2013 diffusive limit + BMM 2015 universal stability boundary). The criticality-endpoint identity $n_{\mathrm{SV}}(\kappa^\star) = 1$ is rigorous; the global path-by-path equivalence between event-counting Hawkes processes and our continuous-state diffusion remains open and is the natural extension to a separate paper.
4. ~~McKean–Vlasov mean-field limit of the dealer-gamma channel~~ — **resolved** in §9 (this section); propagation-of-chaos $L^2$ bound (Theorem 3) and closed-form Hopf-threshold shift (Eq. 25) implemented in `mckean_vlasov.py`, $1/\sqrt n$ scaling validated numerically. Remaining open: heterogeneous $\theta_G^{(i)}$ across dealers (joint MV over $(G_i, \theta_G^{(i)})$), dealer–dealer correlation channels (common-noise MV games, Carmona–Delarue 2018 Vol II Ch. 1).
5. ~~Information-theoretic characterisation of the critical edge~~ — **resolved** in §10 (this section); Theorem 5 closed-form excess entropy (27), mean-field critical exponent $\beta = 1$ at the saturation, IAAFT-calibrated empirical transfer-entropy estimator (Schreiber 2000) implemented in `info_theoretic.py`. Remaining open: closed-form characterisation of $E_\tau$'s global $\kappa$-monotonicity across the node-spiral transition $\kappa_{\mathrm{NS}}$ (currently numerical-only); analogous result for the McKean-Vlasov limit's $\bar G_\infty$-conditioned $E_\tau$ (Theorem 3 generalisation).

## References

Full bibliography in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §References.
