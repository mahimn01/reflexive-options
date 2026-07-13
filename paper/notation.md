# Notation for the v0.4 centered model

This table follows `paper/main.tex`. Legacy simulator and pre-registration
symbols may differ and are not silently mapped into the current theorem.

## State and measure

| Symbol | Meaning | Units/domain |
|---|---|---|
| $\mathbb P$ | physical measure | -- |
| $F_t$ | predictable local reference level | price |
| $X_t=\log(S_t/F_t)$ | detrended log-price deviation | dimensionless, local |
| $v_t$ | instantaneous annualized variance | $\mathbb R_{\ge0}$ |
| $\chi_t$ | filtered price-memory signal | dimensionless |

## Structural parameters

| Symbol | Meaning | Units |
|---|---|---|
| $\delta$ | local reference pull | yr$^{-1}$ |
| $\kappa$ | coupling to normalized book pressure | yr$^{-1}$ |
| $\kappa_v$ | variance mean-reversion speed | yr$^{-1}$ |
| $\theta_v$ | long-run variance | annualized variance |
| $\alpha$ | memory adjustment speed | yr$^{-1}$ |
| $\beta$ | memory target loading | dimensionless |
| $\gamma$ | strength of $v\chi$ variance feedback; unrelated to option gamma | yr$^{-1}$ |
| $\xi$ | square-root variance diffusion scale | yr$^{-1}$ when $v$ is yr$^{-1}$ |
| $\rho$ | Brownian correlation | $[-1,1]$ |

The current dynamics are

$$dX=[-\delta X-\tfrac12(v-\theta_v)+\kappa g(X,v,\chi)]dt+\sqrt v\,dW^S,$$
$$dv=[\kappa_v(\theta_v-v)+\gamma v\chi]dt+\xi\sqrt v\,dW^v,$$
$$d\chi=\alpha(\beta X-\chi)dt.$$

## Dealer book

| Symbol | Meaning |
|---|---|
| $q(k)$ | latent **signed dealer-position** density in fixed log moneyness |
| $\mathcal G(X,v)$ | positive Gaussian book mass integrated against BS gamma |
| $s\in\{-1,+1\}$ | latent dealer orientation; not observed from public OI |
| $g$ | centered, normalized book pressure with $g(0,\theta_v,0)=0$ |
| $(\mu_q,\sigma_q,T)$ | Gaussian mean, dispersion, effective maturity |

Public OI is a count of outstanding contracts. A convention-signed OI-gamma
series is a proxy and must not be denoted as observed dealer inventory.

## Linearization and Hopf quantities

| Symbol | Meaning |
|---|---|
| $z_\star=(0,\theta_v,0)$ | fixed equilibrium for every $\kappa$ |
| $J(\kappa)$ | Jacobian at $z_\star$ |
| $c_2,c_1,c_0$ | coefficients of $\det(\lambda I-J)$ |
| $H=c_1c_2-c_0$ | cubic Routh--Hurwitz determinant |
| $\kappa^\star$ | positive root satisfying all side conditions |
| $\omega^\star=\sqrt{c_1(\kappa^\star)}$ | Hopf angular frequency |
| $\ell_1$ | first Lyapunov coefficient in Kuznetsov convention |

Absence of a valid root means only absence of this local Hopf mechanism. It
does not mean global stability.
