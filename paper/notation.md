# Notation — canonical symbol table

This is the single source of truth for symbols used across the codebase, theory writeup, and paper. Any module or proof that uses a symbol must conform to this table or extend it via PR.

## State variables

The reflexive simulator is **3-dimensional**: $(S_t, v_t, z_t)$. The memory variable $z_t$ is required for the Hopf bifurcation theorem (the 2D skeleton has an upper-triangular Jacobian and cannot Hopf — see `paper/theory.md` §1.1).

| Symbol | Meaning | Range / units |
|--------|---------|---------------|
| $S_t$ | Underlying spot price at time $t$ | $\mathbb{R}_{>0}$, USD |
| $v_t$ | Instantaneous variance | $\mathbb{R}_{\geq 0}$, (decimal/yr)² |
| $\sigma_t = \sqrt{v_t}$ | Instantaneous volatility | $\mathbb{R}_{\geq 0}$, decimal/yr |
| $z_t$ | Memory variable: low-pass-filtered log-price | $\mathbb{R}$, dimensionless |
| $G(S, z, v)$ | Aggregate dealer-gamma exposure | scalar, USD per unit return |
| $t$ | Time | $[0, T]$, years |
| $T$ | Time horizon (terminal) | years |

## Heston parameters (per regime in time-dep variant)

| Symbol | Meaning |
|--------|---------|
| $\kappa$ (Heston) | Mean-reversion speed of variance |
| $\theta$ (Heston) | Long-run variance |
| $\xi$ | Vol-of-vol |
| $\rho$ | Correlation between $dW_S$ and $dW_v$ |
| $v_0$ | Initial variance |

> **Naming clash warning.** Heston's mean-reversion parameter is conventionally $\kappa$. The reflexive coupling parameter is also conventionally $\kappa$. Throughout the code and paper we use **`kappa`** (Heston) and **`coupling`** (reflexive) to disambiguate. In LaTeX use $\kappa$ for Heston, $\boldsymbol{\kappa}$ (bold) for the coupling, with explicit definitions on first use.

## Reflexive simulator parameters

| Symbol | Meaning | Literature prior |
|--------|---------|------------------|
| $\boldsymbol{\kappa}$ | Reflexive coupling strength | $\sim 5 \times 10^{-12}$ per USD-of-dollar-gamma per year (triangulated from GPP 2009 + Barbon–Buraschi + SqueezeMetrics; see `dealer_gamma_brief.md`) |
| $\mu$ | Drift (often risk-neutral or fitted) | scalar, $\mathbb{R}$ |
| $\alpha$ | Memory-variable decay rate | $\sim 252$/yr ⇒ ~1-day half-life |
| $\beta$ | Memory-variable intake from log-price | dimensionless, $O(1)$ |
| $\gamma$ | Leverage feedback: memory $z$ → variance drift | $\geq 0$, units 1/yr |
| $\boldsymbol{\kappa}^*$ | Hopf bifurcation threshold (Theorem 1, paper/theory.md) | solves $H(\kappa)=c_1 c_2 - c_0 = 0$ |
| $\omega^* = \sqrt{c_1(\kappa^*)}$ | Hopf angular frequency at bifurcation | rad/yr |
| $\ell_1$ | First Lyapunov coefficient (sub vs super-critical) | sign determines bifurcation type |

## Surface grid

| Symbol | Meaning |
|--------|---------|
| $K$ | Strike price |
| $k = \log(K / S_t)$ | Log-moneyness, centered at 0 = ATM |
| $T$ (in surface context) | Maturity |
| $\sigma(K, T)$ or $\sigma(k, T)$ | Implied vol surface |
| $w(k, T) = \sigma^2(k, T) \cdot T$ | Total implied variance |

## Open-interest grid

| Symbol | Meaning |
|--------|---------|
| $q_{K, T}$ | Open interest in contracts at strike $K$, maturity $T$ |
| $\Gamma_{K, T}(S, t)$ | Per-contract Black-Scholes gamma |
| $\text{sign}(K, T)$ | Dealer position sign convention |

The aggregator: $G(S, t) = \sum_{K, T} q_{K, T} \cdot \Gamma_{K, T}(S, t) \cdot \text{sign}(K, T)$.

## RL agent

| Symbol | Meaning |
|--------|---------|
| $\pi$ | Policy |
| $\pi_{\boldsymbol{\kappa}_0}$ | Policy trained at coupling $\boldsymbol{\kappa}_0$ |
| $a_t$ | Action — vector of option position deltas on the strike-expiry grid |
| $r_t$ | Reward — P&L − transaction cost − position-size penalty |
| $s_t$ | State — $(S_t, \text{IV-surface tensor}, v_t, G_t, \text{position}, \text{time-to-expiry})$ |

## Evaluation metrics

| Symbol | Meaning |
|--------|---------|
| $W_2$ | Wasserstein-2 distance between surface distributions |
| $\text{slope}_{\boldsymbol{\kappa}_0}$ | $\partial(\text{metric}) / \partial \boldsymbol{\kappa}$ at the training point — the κ-sensitivity scalar |
| $\text{IS}, \text{OOS}$ | In-sample, out-of-sample (per Bailey/Lopez de Prado deflated Sharpe convention) |
