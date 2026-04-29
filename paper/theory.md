# Theory — Hopf bifurcation and stationary density of the reflexive SDE

This document is the canonical writeup of the analytical contributions of the paper. The implementation in `src/reflexive_options/theory/` operationalizes these results numerically. Derivation details and a literature scan live in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md`.

---

## 1. The model

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
c_0 = -a\kappa_v\alpha + \beta\bigl(\kappa\, G_z\, \kappa_v - b\,\gamma\bigr).
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
> - **(A2)** *There exists a unique equilibrium $(S^\star, \theta_v, z^\star)$ given by (2) for every $\kappa$ in some interval $[0, \kappa_{\max}]$.*
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
| Sign of $\ell_1$ | Yes for parametric $G$ (e.g. log-normal OI in moneyness); numerical otherwise |
| Limit-cycle amplitude / shape past $\kappa^\star$ | Numerical |

The implementation in `src/reflexive_options/theory/bifurcation.py` does the numerical eigenvalue scan over $(\kappa, \xi)$ and locates $\kappa^\star$ as the contour where $\mathrm{Re}\,\lambda_\pm$ crosses zero. It also computes the first Lyapunov coefficient $\ell_1$ (Kuznetsov 2004 eq. 3.20) via finite-difference construction of the bilinear / trilinear Taylor tensors $B$ and $C$ around the equilibrium, and the stochastic-Hopf shift $\Lambda(\kappa)$ via a Khasminskii-style sphere process (Benettin renormalisation of the linearised SDE).

### 4.2 Numerical $\ell_1$ and $\Lambda$ at a representative parameter set

The default `BifurcationConfig` in `experiments/bifurcation_scan.py` is dimensionally tuned to the empirical dealer-gamma magnitudes ($G_x \sim 10^{-3}$, $G_z \sim 10^{-3}$ per USD-of-dollar-gamma) and *does not exhibit a Hopf within the literature-prior κ range*. The structural reason: the memory channel decay $\alpha = 252$/yr (≈ 1-day half-life) is fast relative to $\kappa_v = 2$/yr, and the small first-derivative magnitudes mean the cross-coupling $\kappa G_z \beta$ in $c_1$ never overpowers $\kappa_v \alpha$. This matches the empirical observation that real options markets sit *near* but not *across* $\kappa^\star$ — consistent with HBB's $n \approx 1$ critical regime (§6).

For a worked numerical example we instead use a representative *dimensionless* regime where $\{G_x, G_v, G_z\} = \{0.5, -0.5, -0.5\}$, $\alpha = 0.5$/yr (multi-day memory), $\beta = 1$, $\gamma = 0.5$, $\kappa_v = 2$/yr, and quadratic / cubic Taylor coefficients $G_{xx} = -0.1$, $G_{xxx} = -0.2$ (representative of a smooth, locally concave dealer-gamma functional around ATM in a long-gamma regime). The deterministic skeleton then satisfies:

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

---

## 5. Stochastic lift

The deterministic Hopf indicates *when* the noiseless skeleton begins to oscillate. The full SDE (1) has multiplicative Heston noise; the relevant question (Arnold 1998) is when the top Lyapunov exponent $\lambda_1$ of the linearised RDS at the equilibrium changes sign.

For small noise intensity $\varepsilon$,

$$
\lambda_1(\kappa, \varepsilon) = \alpha(\kappa) + \varepsilon^2 \cdot \Lambda(\kappa) + O(\varepsilon^4),
$$

where $\alpha(\kappa)$ is the deterministic real part (zero at $\kappa^\star_{\mathrm{det}}$) and $\Lambda$ is computed via Engel–Lamb–Rasmussen (2024). Their shear-induced corrections give $|\Lambda| \sim (\rho \xi)^{2/3}$ — high vol-of-vol and strong correlation push the bifurcation around significantly. We compute $\Lambda$ numerically via Khasminskii's sphere process (Algorithm 2 in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §6).

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

1. Closed-form $\ell_1$ for log-normal OI in moneyness (parametric calibration target).
2. Closed-form $\Lambda(\kappa)$ for the stochastic-Hopf shift (Engel–Lamb–Rasmussen-style asymptotics adapted to the Heston multiplicative-noise structure).
3. Formal Hawkes-$n$ ↔ SV-eigenvalue reduction (most ambitious; not blocking for the v1 paper).

## References

Full bibliography in `~/Documents/reflexivity-research/hopf_bifurcation_brief.md` §References.
