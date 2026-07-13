# Threats to validity — v0.4 centered model

## Construct validity

The theoretical $q(k)$ is a signed dealer-position density. Public open
interest is not that object. A13--A16 measure public positioning mass and shape.
Convention-signed GEX cannot validate dealer sign because its orientation is
imposed by construction.

The feedback drift is reduced form. Gârleanu--Pedersen--Poteshman supports a
demand-pressure role in option prices, not the exact spot drift here. Neither
$\kappa$ nor $\gamma$ is derived from dealer optimization or market clearing.

## Internal mathematical validity

The theorem is local and deterministic. The displayed cycle does not prove a
global attractor, stochastic periodic orbit, stationary law, tail result, or
non-explosion of the full SDE. Inward boundary drift is not a global existence
proof. A one-orientation, one-maturity Gaussian book suppresses actual book
heterogeneity and evolution.

The zero stochastic correction applies only to the frozen affine additive
surrogate. The square-root diffusion has non-zero state derivatives, so its
stochastic variational equation and any random-dynamical bifurcation remain
open.

The first Lyapunov coefficient uses analytic derivatives but depends on the
stated coordinate normalization. Its sign and the local stability
classification are the invariant conclusions of interest.

## Statistical validity

Daily log squared return is a noisy volatility proxy. Calendar-aware HAC and moving-block
bootstrap handle serial dependence imperfectly. A16 separately BH-adjusts both
families and requires their agreement plus an interval excluding zero; this
prevents ex-post method selection but does not make either approximation exact.
BH control covers the registered family, not unregistered model search.

Strict forward dating prevents same-day mechanical lookahead but does not make
OI exogenous. Risk appetite, macro news, and volatility demand can drive both
the book and subsequent returns.

## External validity

SPX has distinctive settlement, dealer, and 0DTE structure. Results need not
generalize to single names or other asset classes. End-of-day OI cannot test an
intraday hedging mechanism directly. A null daily result does not rule out an
intraday effect; a positive daily result does not identify one.

## Reproducibility validity

Theory and figures are data-free and deterministic. The future study depends
on proprietary database vintages. Queries, row counts, coverage, hashes,
filter attrition, and vendor conventions must be frozen for reproduction by a
researcher with equivalent access.
