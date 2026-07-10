r"""Archived exploratory mean-field extension; not a result of the paper.

The v0.3 claims used the superseded moving-equilibrium/additive-variance
model and did not establish an empirical dealer population. The numerical
particle routines remain reproducibility utilities, not a proof that the
centered model has this mean-field limit or threshold correction.

The single-representative-dealer SDE in the §2 model implicitly assumes
either (a) perfect coordination across dealers, or (b) the law-of-large-
numbers limit where idiosyncratic dealer noise washes out.  This module
formalises (b) as a McKean-Vlasov (MV) SDE coupled to the *law* of the
dealer-gamma process, provides a finite-particle simulator that
validates Sznitman (1991) / Méléard (1996)'s propagation-of-chaos $1/n$
rate, and provides the EXACT 4D Hopf-threshold correction.

Setup (n-dealer system).  Dealer $i \in \{1, \ldots, n\}$ holds gamma
exposure $G_i$ that obeys

    dG_i = -theta_G (G_i - g(S, v)) dt + sigma_G dW^i_G,

with $\{W^i_G\}_i$ independent Brownian motions and $g(S, v)$ the
"target" dealer-gamma map (e.g. the closed-form log-normal-OI aggregator
of paper §4.3, evaluated at the current spot/variance).  $\theta_G > 0$
is the dealer-hedging speed (equivalently $\tau_G := 1/\theta_G$ is the
autocorrelation time of the gamma deviation).  Aggregating into the
spot dynamics:

    dS / S = (mu + kappa * G_bar_n) dt + sigma(S, v) dW^S,
    G_bar_n := (1/n) sum_i G_i.

In the limit $n \to \infty$, propagation of chaos gives weak convergence
of the empirical measure $\bar\mu_n^t = (1/n)\sum_i \delta_{G_i^t}$ to
the deterministic measure $\mu^t = \mathrm{Law}(G^t)$ where $G$ solves
the MV SDE

    dG = -theta_G (G - g(S, v)) dt + sigma_G dW_G,
    dS / S = (mu + kappa * G_bar_inf) dt + sigma(S, v) dW^S,
    G_bar_inf(t) = E[G^t | F_t^{S, v}] = \int g\, d\mu^t(g).

For the OU-target dynamics here, $G_bar_inf(t) = E[G(t)]$ which obeys
the mean ODE $\dot{\bar G}_\infty = -\theta_G (\bar G_\infty - g(S, v))$ —
the conditional expectation matches the deterministic relaxation of
$G$ towards its target.

Key quantitative outputs:

1. Propagation-of-chaos $L^2$ error bound (Sznitman 1991, Théorème I.1.4;
   Méléard 1996 Prop 2.5; Carmona-Delarue 2018 Vol I Thm 2.12).  Under
   Lipschitz $b, sigma_G$ and finite second moment of $G_i^0$,

       sup_{t <= T} E[(G_bar_n(t) - G_bar_inf(t))^2] <= C(T) / n.

   For our linear-in-$G$ drift this constant is explicit:

       C(T) = sigma_G^2 / (2 theta_G) * (1 - exp(-2 theta_G T))
              + Var(G^0) * exp(-2 theta_G T)
            <= max(sigma_G^2 / (2 theta_G), Var(G^0)).

2. Hopf-threshold shift (CORRECTED in v0.3.6).  The mean-field limit
   adds a fourth state to the linearisation — the aggregate dealer
   gamma $g$ obeying $\dot g = -\theta_G g + \theta_G (G_y y + G_v u
   + G_z z)$ — and the spot equation now feeds back $\kappa g$ rather
   than the instantaneous map $\kappa (G_y y + G_v u + G_z z)$.  The
   extended 4D Jacobian is

       J_MV(kappa, theta_G) =
         [ -0.5 sig2_y   -0.5 sig2_v    0           kappa        ]
         [ 0             -kappa_v       gamma       0            ]
         [ beta          0              -alpha      0            ]
         [ theta_G G_y   theta_G G_v    theta_G G_z -theta_G     ]

   (with sig2_y = ∂_y σ², sig2_v = ∂_v σ² being the Ito terms from the
   spot SDE).  The Hopf threshold is the smallest κ > 0 at which a
   complex-conjugate eigenvalue pair has zero real part, equivalently
   the 4D Liu / Routh-Hurwitz condition

       a_3 a_2 a_1 - a_1^2 - a_3^2 a_0 = 0,   (Hopf line)
       a_3 > 0,  a_0 > 0,  a_3 a_2 - a_1 > 0  (positivity).

   The previous v0.3.5 "low-pass filter, reciprocate the gain" heuristic
   was incorrect: it ignored the destabilising effect of adding a new
   state to the linearised system (the rank of the unstable manifold
   can grow at the bifurcation) and missed the phase-coupling between
   $g$ and the price/variance channels.  An external audit verified
   numerically that the previous formula was wrong by up to a factor
   of $\sim$2 in magnitude AND of opposite sign at the canonical
   short-gamma regime (G_y > 0).  The exact 4D computation is
   implemented here.

   At the canonical short-gamma regime
   $(G_y, G_v, G_z, \alpha, \beta, \gamma, \kappa_v) = (0.5, -0.5, -0.5,
   0.5, 1, 0.5, 2)$ with $\sigma^2_y = \sigma^2_v = 0$, the threshold
   admits the closed form

       kappa_star_MV(theta_G) =
         [50 t^2 + 143 t + 105 - (2t + 5) sqrt(385 t^2 + 810 t + 441)]
         / [12 t (t + 1)],     t := theta_G,

   with single-dealer limit $\kappa^\star_\mathrm{single} = (25 -
   \sqrt{385}) / 6 \approx 0.8964$ as $\theta_G \to \infty$ (the dealer
   mode decouples) and $\kappa^\star_\mathrm{MV} \to 8/21 \approx 0.381$
   as $\theta_G \to 0^+$.  Across this regime the MV correction strictly
   *lowers* the threshold: dealer-hedging latency is destabilising.

   For general regimes the ratio is regime-dependent: in particular
   long-gamma regimes (G_y < 0, e.g.\ the log-normal-OI calibration of
   §4.3) the ratio is > 1 and diverges as $\theta_G \to 0$.

   See `paper/mv_hopf_corrected.md` for the full derivation.

This module implements:

* `mckean_vlasov_jacobian_4d` — the extended 4D Jacobian above.
* `mckean_vlasov_kappa_star` — solves the 4D Hopf condition numerically.
* `mckean_vlasov_kappa_star_shift` — returns the corrected ratio
   $\kappa^\star_\mathrm{MV} / \kappa^\star_\mathrm{single}$ together
   with both thresholds and the Hopf frequency.
* `simulate_n_dealer_system` — Euler-Maruyama on the n-particle SDE.
* `propagation_of_chaos_error` — measures $|G_bar_n - G_bar_inf|$ and
  the $L^2$ supremum over a path.
* `propagation_of_chaos_scaling` — sweeps $n$ and reports the fitted
  log-log slope (should be $\approx 0.5$ for $\sqrt{1/n}$ RMSE).

References:
    Sznitman, A.-S. (1991) "Topics in propagation of chaos." Lecture
        Notes in Math. 1464.  Théorème I.1.4 (the canonical $C/n$ bound).
    Méléard, S. (1996) "Asymptotic behaviour of some interacting particle
        systems; McKean-Vlasov and Boltzmann models." Lecture Notes in
        Math. 1627, Prop 2.5.
    Carmona, R. & Delarue, F. (2018) Probabilistic Theory of Mean Field
        Games with Applications I-II. Springer.  Vol I, Thm 2.12.
    Lacker, D. (2018) "Mean field games via controlled martingale problems."
        SPA 128.
    Liu, W.-M. (1994) "Criterion of Hopf bifurcations without using
        eigenvalues." J. Math. Anal. Appl. 182, 250–256.  The 4D
        Hopf criterion in Routh-Hurwitz form.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# 4D extended Jacobian + corrected Hopf threshold (Theorem 3, v0.3.6).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MVHopfResult:
    """Output of `mckean_vlasov_kappa_star` / `mckean_vlasov_kappa_star_shift`.

    `kappa_star_single`: the 3D single-dealer Hopf threshold (theta_G -> infty
        limit), computed from the same parameters by removing the dealer-OU
        state and feeding $g(S,v)$ back instantaneously.
    `kappa_star_mv`: the 4D MV Hopf threshold at finite theta_G.
    `ratio`: kappa_star_mv / kappa_star_single.  < 1 means MV destabilises
        (lower threshold than single-dealer); > 1 means MV stabilises.
    `omega_star_mv`: the Hopf frequency of the imaginary eigenvalue pair at
        kappa_star_mv.
    `theta_G`: the dealer-hedging speed used.
    """

    kappa_star_single: float
    kappa_star_mv: float
    ratio: float
    omega_star_mv: float
    theta_G: float


def mckean_vlasov_jacobian_4d(
    *,
    kappa: float,
    theta_G: float,
    G_y: float,
    G_v: float,
    G_z: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    sigma2_y: float = 0.0,
    sigma2_v: float = 0.0,
) -> NDArray[np.float64]:
    r"""Build the 4D extended Jacobian $J_\mathrm{MV}(\kappa, \theta_G)$.

    The extended state is $(y, u, z, g)$ where $g$ is the aggregate
    dealer-gamma deviation.  Linearising the spot/variance/memory/dealer
    SDEs at the equilibrium with $g(S, v)$ Taylor-expanded as
    $g(y, u, z) = G_y y + G_v u + G_z z$,

        \dot y = -\tfrac12 \sigma^2_y y - \tfrac12 \sigma^2_v u + \kappa g,
        \dot u = -\kappa_v u + \gamma z,
        \dot z = \beta y - \alpha z,
        \dot g = -\theta_G g + \theta_G (G_y y + G_v u + G_z z).

    Args:
        kappa: feedback strength.
        theta_G: dealer-hedging speed > 0.
        G_y, G_v, G_z: partials of the dealer-gamma target $g(y, u, z)$.
        kappa_v: variance mean-reversion speed > 0.
        alpha: memory-channel decay > 0.
        beta: memory-channel intake.
        gamma: leverage feedback.
        sigma2_y, sigma2_v: partials of $\sigma^2(y, u)$ at equilibrium
            (the Ito terms in the spot SDE; both zero for the pure 4D
            short-gamma analysis, sigma2_v = 1 for a Heston backbone
            $\sigma^2 = v$ at $v^\star = 1$).
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if kappa_v <= 0.0:
        raise ValueError(f"kappa_v must be > 0, got {kappa_v}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    return np.array(
        [
            [-0.5 * sigma2_y, -0.5 * sigma2_v, 0.0, kappa],
            [0.0, -kappa_v, gamma, 0.0],
            [beta, 0.0, -alpha, 0.0],
            [theta_G * G_y, theta_G * G_v, theta_G * G_z, -theta_G],
        ],
        dtype=np.float64,
    )


def _jacobian_3d_single_dealer(
    *,
    kappa: float,
    G_y: float,
    G_v: float,
    G_z: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    sigma2_y: float = 0.0,
    sigma2_v: float = 0.0,
) -> NDArray[np.float64]:
    """3D single-dealer Jacobian (theta_G -> infty limit).

    Recovers the §2 model: $g(S, v)$ is fed back instantaneously, so
    row 1 of the Jacobian gets $\\kappa G_y, \\kappa G_v, \\kappa G_z$.
    """
    a = kappa * G_y - 0.5 * sigma2_y
    b = kappa * G_v - 0.5 * sigma2_v
    return np.array(
        [
            [a, b, kappa * G_z],
            [0.0, -kappa_v, gamma],
            [beta, 0.0, -alpha],
        ],
        dtype=np.float64,
    )


def _max_real_eig(J: NDArray[np.float64]) -> float:
    """Maximum real part of the eigenvalues of J."""
    eig = np.linalg.eigvals(J)
    return float(np.asarray(eig, dtype=np.complex128).real.max())


def _imag_at_max_real(J: NDArray[np.float64]) -> float:
    """Imaginary part of the eigenvalue with the smallest |Re| (the Hopf pair)."""
    eig = np.asarray(np.linalg.eigvals(J), dtype=np.complex128)
    order = np.argsort(np.abs(eig.real))
    return float(abs(eig[order[0]].imag))


def _find_first_hopf(
    f: Callable[[float], float],
    kappa_min: float,
    kappa_max: float,
    n_grid: int,
) -> float | None:
    """Scan f(kappa) on a log grid, return brentq root at the smallest sign change.

    Returns None if no sign change is detected in [kappa_min, kappa_max].
    Detects both +→− and −→+ crossings; the smallest-kappa crossing is the
    Hopf threshold.
    """
    if kappa_min <= 0.0 or kappa_max <= kappa_min:
        raise ValueError("require 0 < kappa_min < kappa_max")
    ks = np.geomspace(kappa_min, kappa_max, n_grid)
    fs = np.array([f(float(k)) for k in ks], dtype=np.float64)
    sign = np.sign(fs)
    # Mask out exact zeros that aren't true sign changes (e.g. degenerate).
    changes = np.where(np.diff(sign) != 0)[0]
    if len(changes) == 0:
        return None
    i = int(changes[0])
    a, b = float(ks[i]), float(ks[i + 1])
    if math.copysign(1.0, fs[i]) == math.copysign(1.0, fs[i + 1]):
        return None
    return float(brentq(f, a, b, xtol=1e-10))


def mckean_vlasov_kappa_star(
    *,
    theta_G: float,
    G_y: float,
    G_v: float,
    G_z: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    sigma2_y: float = 0.0,
    sigma2_v: float = 0.0,
    kappa_min: float = 1e-3,
    kappa_max: float = 1e4,
    n_grid: int = 2001,
) -> tuple[float, float]:
    """Solve the 4D MV Hopf condition for $\\kappa^\\star_\\mathrm{MV}$.

    Finds the smallest $\\kappa > 0$ at which $\\max_i \\mathrm{Re}\\,
    \\lambda_i(J_\\mathrm{MV}(\\kappa, \\theta_G)) = 0$, scanning a
    log-spaced grid on $[\\kappa_\\min, \\kappa_\\max]$.  This is the
    Liu / 4D Routh-Hurwitz Hopf line; we use the eigenvalue formulation
    rather than the explicit polynomial discriminant because the latter
    is messy in general parameters and harder to verify the positivity
    conditions on.

    Returns:
        (kappa_star_mv, omega_star_mv) where omega_star_mv is the
        imaginary part of the complex pair at the bifurcation.

    Raises:
        RuntimeError if no sign change of max(Re λ) is found on the
        grid — either the regime has no Hopf in [kappa_min, kappa_max]
        or the bracket needs widening.
    """

    def f(kappa: float) -> float:
        J = mckean_vlasov_jacobian_4d(
            kappa=kappa,
            theta_G=theta_G,
            G_y=G_y,
            G_v=G_v,
            G_z=G_z,
            kappa_v=kappa_v,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sigma2_y=sigma2_y,
            sigma2_v=sigma2_v,
        )
        return _max_real_eig(J)

    kstar = _find_first_hopf(f, kappa_min, kappa_max, n_grid)
    if kstar is None:
        raise RuntimeError(
            f"no 4D MV Hopf found on [{kappa_min}, {kappa_max}] at theta_G={theta_G}; "
            "widen the bracket or check the regime parameters."
        )
    J_star = mckean_vlasov_jacobian_4d(
        kappa=kstar,
        theta_G=theta_G,
        G_y=G_y,
        G_v=G_v,
        G_z=G_z,
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        sigma2_y=sigma2_y,
        sigma2_v=sigma2_v,
    )
    omega_star = _imag_at_max_real(J_star)
    return float(kstar), float(omega_star)


def _kappa_star_single_3d(
    *,
    G_y: float,
    G_v: float,
    G_z: float,
    kappa_v: float,
    alpha: float,
    beta: float,
    gamma: float,
    sigma2_y: float = 0.0,
    sigma2_v: float = 0.0,
    kappa_min: float = 1e-3,
    kappa_max: float = 1e4,
    n_grid: int = 2001,
) -> tuple[float, float]:
    """Solve the 3D single-dealer Hopf condition; theta_G -> infty limit of MV."""

    def f(kappa: float) -> float:
        J = _jacobian_3d_single_dealer(
            kappa=kappa,
            G_y=G_y,
            G_v=G_v,
            G_z=G_z,
            kappa_v=kappa_v,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sigma2_y=sigma2_y,
            sigma2_v=sigma2_v,
        )
        return _max_real_eig(J)

    kstar = _find_first_hopf(f, kappa_min, kappa_max, n_grid)
    if kstar is None:
        raise RuntimeError(
            f"no 3D single-dealer Hopf found on [{kappa_min}, {kappa_max}]; "
            "widen the bracket or check the regime parameters."
        )
    J_star = _jacobian_3d_single_dealer(
        kappa=kstar,
        G_y=G_y,
        G_v=G_v,
        G_z=G_z,
        kappa_v=kappa_v,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        sigma2_y=sigma2_y,
        sigma2_v=sigma2_v,
    )
    omega_star = _imag_at_max_real(J_star)
    return float(kstar), float(omega_star)


def mckean_vlasov_kappa_star_shift(
    *,
    theta_G: float,
    G_y: float | None = None,
    G_v: float | None = None,
    G_z: float | None = None,
    kappa_v: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    sigma2_y: float = 0.0,
    sigma2_v: float = 0.0,
    omega_star: float | None = None,
    kappa_min: float = 1e-3,
    kappa_max: float = 1e4,
    n_grid: int = 2001,
) -> MVHopfResult:
    r"""Corrected MV Hopf-threshold shift via the 4D extended Jacobian.

    Returns the structured result containing the single-dealer threshold,
    the MV threshold, their ratio, and the Hopf frequency $\omega^\star_\mathrm{MV}$.
    The MV threshold is the smallest $\kappa > 0$ for which the 4D Jacobian
    $J_\mathrm{MV}(\kappa, \theta_G)$ has a pair of complex-conjugate
    eigenvalues with zero real part.  The single-dealer threshold is the
    $\theta_G \to \infty$ limit, recovered by removing the dealer state.

    Args:
        theta_G: dealer-hedging speed > 0 (autocorrelation time
            $\tau_G = 1/\theta_G$).
        G_y, G_v, G_z: partials of the dealer-gamma target.
        kappa_v: variance mean-reversion speed > 0.
        alpha: memory-channel decay > 0.
        beta: memory-channel intake.
        gamma: leverage feedback.
        sigma2_y, sigma2_v: partials of $\sigma^2$ at equilibrium
            (0 for the canonical 4D regime, sigma2_v = 1 for a Heston
            backbone $\sigma^2 = v$ at $v^\star = 1$).
        omega_star: DEPRECATED — was the single-dealer Hopf frequency used
            by the v0.3.5 heuristic formula.  Now ignored; the corrected
            computation does not need it as an input (it is reported as an
            output).  Passing a non-None value emits a DeprecationWarning.
        kappa_min, kappa_max, n_grid: bracket and resolution for the
            log-grid eigenvalue scan.

    BACKWARD COMPATIBILITY:
        The v0.3.5 signature was ``mckean_vlasov_kappa_star_shift(*, theta_G,
        omega_star)`` returning a scalar ratio.  That signature is no longer
        supported because the v0.3.5 formula was numerically incorrect by up
        to a factor of ~2 (and of wrong sign in canonical short-gamma
        regimes) per an external audit.  Callers must supply the full
        4D Jacobian structure (the same parameters used by the bifurcation
        module's `kappa_star_lognormal_oi` and friends).

    Returns:
        `MVHopfResult` with .kappa_star_single, .kappa_star_mv, .ratio,
        .omega_star_mv, .theta_G.
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if omega_star is not None:
        warnings.warn(
            "omega_star is deprecated and ignored: the corrected v0.3.6 4D Hopf "
            "computation does not require it as an input. See "
            "paper/mv_hopf_corrected.md for the corrected derivation.",
            DeprecationWarning,
            stacklevel=2,
        )
    required = {
        "G_y": G_y,
        "G_v": G_v,
        "G_z": G_z,
        "kappa_v": kappa_v,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise TypeError(
            "mckean_vlasov_kappa_star_shift requires the full 4D Jacobian structure; "
            f"missing keyword arguments: {missing}. "
            "The v0.3.5 (theta_G, omega_star)-only signature was retired because the "
            "heuristic formula was incorrect per external audit (see paper/mv_hopf_corrected.md)."
        )

    # Type-narrow after the None check.
    Gy_v = float(G_y)  # type: ignore[arg-type]
    Gv_v = float(G_v)  # type: ignore[arg-type]
    Gz_v = float(G_z)  # type: ignore[arg-type]
    kv_v = float(kappa_v)  # type: ignore[arg-type]
    al_v = float(alpha)  # type: ignore[arg-type]
    be_v = float(beta)  # type: ignore[arg-type]
    ga_v = float(gamma)  # type: ignore[arg-type]

    kstar_mv, omega_mv = mckean_vlasov_kappa_star(
        theta_G=theta_G,
        G_y=Gy_v,
        G_v=Gv_v,
        G_z=Gz_v,
        kappa_v=kv_v,
        alpha=al_v,
        beta=be_v,
        gamma=ga_v,
        sigma2_y=sigma2_y,
        sigma2_v=sigma2_v,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        n_grid=n_grid,
    )
    kstar_single, _ = _kappa_star_single_3d(
        G_y=Gy_v,
        G_v=Gv_v,
        G_z=Gz_v,
        kappa_v=kv_v,
        alpha=al_v,
        beta=be_v,
        gamma=ga_v,
        sigma2_y=sigma2_y,
        sigma2_v=sigma2_v,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        n_grid=n_grid,
    )
    return MVHopfResult(
        kappa_star_single=float(kstar_single),
        kappa_star_mv=float(kstar_mv),
        ratio=float(kstar_mv / kstar_single),
        omega_star_mv=float(omega_mv),
        theta_G=float(theta_G),
    )


# ---------------------------------------------------------------------------
# Canonical-regime closed form (audit anchor; matches mckean_vlasov_kappa_star
# numerically to floating-point precision).
# ---------------------------------------------------------------------------


def mckean_vlasov_kappa_star_canonical_closed_form(theta_G: float) -> float:
    r"""Closed-form $\kappa^\star_\mathrm{MV}(\theta_G)$ at the canonical regime.

    Derived in `paper/mv_hopf_corrected.md` for the regime
    $(G_y, G_v, G_z, \alpha, \beta, \gamma, \kappa_v) = (1/2, -1/2, -1/2,
    1/2, 1, 1/2, 2)$ with $\sigma^2_y = \sigma^2_v = 0$.  The 4D Liu
    Hopf condition $a_3 a_2 a_1 - a_1^2 - a_3^2 a_0 = 0$ is quadratic in
    $\kappa$ at this regime; taking the smaller positive root gives

        kappa_star_MV(t) =
          [50 t^2 + 143 t + 105 - (2t + 5) sqrt(385 t^2 + 810 t + 441)]
          / [12 t (t + 1)],   t := theta_G.

    Limits:
        theta_G -> infty:  (25 - sqrt(385)) / 6 \approx 0.8964  (single-dealer)
        theta_G -> 0+:     8 / 21 \approx 0.3810   (finite — frozen-dealer limit)

    This function is the audit anchor used by the regression tests.
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    t = float(theta_G)
    disc = 385.0 * t * t + 810.0 * t + 441.0
    if disc < 0.0:  # never, but defensively
        raise RuntimeError(f"closed-form discriminant negative at theta_G={t}")
    num = 50.0 * t * t + 143.0 * t + 105.0 - (2.0 * t + 5.0) * math.sqrt(disc)
    den = 12.0 * t * (t + 1.0)
    return num / den


KAPPA_STAR_SINGLE_CANONICAL: float = (25.0 - math.sqrt(385.0)) / 6.0
"""Closed-form single-dealer $\\kappa^\\star_\\mathrm{single}$ at the canonical regime
(theta_G -> infty limit of `mckean_vlasov_kappa_star_canonical_closed_form`)."""


def propagation_of_chaos_constant(
    *,
    theta_G: float,
    sigma_G: float,
    var_G0: float,
    T: float,
) -> float:
    """Closed-form $C(T)$ in $\\sup_t E[(G_bar_n - G_bar_inf)^2] \\le C(T)/n$.

    For OU dynamics $dG_i = -\\theta_G(G_i - g) dt + \\sigma_G dW^i_G$ with
    a *common* (non-particle-dependent) target $g = g(S, v)$, the
    deviations $\\delta_i := G_i - g(S, v)$ inherit the same OU and the
    cross-correlations $E[\\delta_i \\delta_j]$ for $i \\ne j$ vanish under
    independent Brownian motions.  The variance of the empirical mean is

        Var(G_bar_n - G_bar_inf) = (1/n) Var(G_i - g)
                                 = (1/n) [Var(G^0 - g^0) e^{-2 theta_G t}
                                          + (sigma_G^2 / 2 theta_G)
                                            * (1 - e^{-2 theta_G t})].

    The supremum over $t \\in [0, T]$ is bounded by
    max(Var(G^0), sigma_G^2 / (2 theta_G)) at the worst-case $t$.

    Returns:
        C(T) such that sup_{t<=T} E[(G_bar_n - G_bar_inf)^2] <= C(T) / n.
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if sigma_G < 0.0:
        raise ValueError(f"sigma_G must be >= 0, got {sigma_G}")
    if var_G0 < 0.0:
        raise ValueError(f"var_G0 must be >= 0, got {var_G0}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    decay = float(np.exp(-2.0 * theta_G * T))
    stationary = sigma_G * sigma_G / (2.0 * theta_G)
    # The two extreme regimes give the supremum over t in [0, T]:
    #   - If Var(G^0) > stationary : C(T) = Var(G^0) (decay-dominated, sup at t=0).
    #   - If Var(G^0) < stationary : C(T) = stationary * (1 - exp(-2 theta_G T))
    #     plus the residual decay term — sup at t=T.
    sup_t = max(var_G0, stationary * (1.0 - decay) + var_G0 * decay)
    return float(sup_t)


# ---------------------------------------------------------------------------
# Particle-system simulator (Deliverable 4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosErrorResult:
    """Output of `propagation_of_chaos_error` at a single particle count.

    `n_particles`: number of dealers in the simulation.
    `t_grid`: time points used for the Euler discretisation.
    `mean_traj_n`: $\\bar G_n(t)$ across the simulation horizon.
    `mean_traj_inf`: $\\bar G_\\infty(t)$ — the deterministic OU mean.
    `l2_error_sup`: $\\sup_t \\sqrt{E[(G_n - G_\\infty)^2]}$ over `n_replicates`
        independent draws (RMSE supremum).
    `n_replicates`: number of independent particle-system replicates used
        to estimate the expectation.
    """

    n_particles: int
    t_grid: NDArray[np.float64]
    mean_traj_n: NDArray[np.float64]  # one representative replicate, for plotting
    mean_traj_inf: NDArray[np.float64]
    l2_error_sup: float
    n_replicates: int


def simulate_n_dealer_system(
    *,
    n_particles: int,
    theta_G: float,
    sigma_G: float,
    g_target: Callable[[float], float],
    G0_distribution: Callable[[np.random.Generator, int], NDArray[np.float64]],
    T: float,
    n_steps: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Euler-Maruyama on the n-dealer system at a frozen $g(S, v)$ trajectory.

    For the propagation-of-chaos validation we hold the spot/variance path
    deterministic — the question is purely about how well $\\bar G_n$
    tracks $\\bar G_\\infty$ at finite $n$, irrespective of the spot
    feedback (which is identical between the two systems by construction).

    Args:
        n_particles: number of dealers ($n$).
        theta_G: hedging speed > 0.
        sigma_G: idiosyncratic noise scale >= 0.
        g_target: callable $t \\mapsto g(S(t), v(t))$ — the common target.
        G0_distribution: callable (rng, n) -> initial G values, length n.
        T: horizon (years).
        n_steps: Euler-Maruyama steps over [0, T].
        seed: optional RNG seed.

    Returns:
        (t_grid, G_paths) where t_grid is shape (n_steps+1,) and G_paths
        is shape (n_steps+1, n_particles).
    """
    if n_particles <= 0:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if sigma_G < 0.0:
        raise ValueError(f"sigma_G must be >= 0, got {sigma_G}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = float(np.sqrt(dt))
    t_grid = np.linspace(0.0, T, n_steps + 1, dtype=np.float64)

    G = np.asarray(G0_distribution(rng, n_particles), dtype=np.float64).reshape(n_particles)
    paths = np.zeros((n_steps + 1, n_particles), dtype=np.float64)
    paths[0] = G

    for k in range(1, n_steps + 1):
        t = float(t_grid[k - 1])
        g_t = float(g_target(t))
        dW = rng.standard_normal(n_particles) * sqrt_dt
        # OU drift towards the common target; independent noise per particle.
        G = G + (-theta_G * (G - g_t)) * dt + sigma_G * dW
        paths[k] = G
    return t_grid, paths


def mean_field_limit_trajectory(
    *,
    theta_G: float,
    g_target: Callable[[float], float],
    G_bar_inf_0: float,
    T: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Deterministic ODE for $\\bar G_\\infty(t)$ in the MV limit.

    For the OU dynamics $dG = -\\theta_G(G - g)dt + \\sigma_G dW$, taking
    expectation gives $\\dot E[G] = -\\theta_G(E[G] - g(S, v))$ — a first-
    order linear ODE with closed form

        E[G(t)] = E[G(0)] e^{-theta_G t}
                + theta_G \\int_0^t e^{-theta_G (t - s)} g(S(s), v(s)) ds.

    We integrate with the explicit Euler scheme matching the particle
    simulator's discretisation so the comparison is clean.
    """
    if theta_G <= 0.0:
        raise ValueError(f"theta_G must be > 0, got {theta_G}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1, dtype=np.float64)
    traj = np.zeros(n_steps + 1, dtype=np.float64)
    traj[0] = G_bar_inf_0
    for k in range(1, n_steps + 1):
        t = float(t_grid[k - 1])
        traj[k] = traj[k - 1] + (-theta_G * (traj[k - 1] - float(g_target(t)))) * dt
    return t_grid, traj


def propagation_of_chaos_error(
    *,
    n_particles: int,
    theta_G: float,
    sigma_G: float,
    g_target: Callable[[float], float],
    G0_mean: float,
    G0_std: float,
    T: float,
    n_steps: int,
    n_replicates: int = 32,
    seed: int | None = None,
) -> ChaosErrorResult:
    """Estimate $\\sup_t \\sqrt{E[(G_bar_n - G_bar_inf)^2]}$ over `n_replicates`.

    The expectation is over the joint Brownian noise + initial-condition
    randomness of the n-particle system; $G_bar_inf$ is the deterministic
    OU mean (closed form via `mean_field_limit_trajectory`).

    Args:
        n_particles: dealers per replicate.
        theta_G, sigma_G, T, n_steps: SDE config.
        g_target: deterministic target trajectory.
        G0_mean, G0_std: initial Gaussian distribution moments for $G_i^0$.
        n_replicates: independent simulations for the expectation estimate.
        seed: optional RNG seed (each replicate uses seed + replicate_idx).

    Returns:
        `ChaosErrorResult` with the path-supremum RMSE.
    """
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")

    # Mean-field limit: starts at G0_mean (the expectation of G_i^0).
    t_grid, traj_inf = mean_field_limit_trajectory(
        theta_G=theta_G,
        g_target=g_target,
        G_bar_inf_0=G0_mean,
        T=T,
        n_steps=n_steps,
    )

    def G0_dist(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return G0_mean + G0_std * rng.standard_normal(n)

    # Track squared error at every time step across replicates.
    squared_errors = np.zeros((n_replicates, n_steps + 1), dtype=np.float64)
    representative_traj_n = None
    for r in range(n_replicates):
        replicate_seed = None if seed is None else seed + r
        _, paths = simulate_n_dealer_system(
            n_particles=n_particles,
            theta_G=theta_G,
            sigma_G=sigma_G,
            g_target=g_target,
            G0_distribution=G0_dist,
            T=T,
            n_steps=n_steps,
            seed=replicate_seed,
        )
        mean_traj_n = paths.mean(axis=1)  # (n_steps+1,)
        squared_errors[r] = (mean_traj_n - traj_inf) ** 2
        if r == 0:
            representative_traj_n = mean_traj_n

    # Per-time MSE then sup over time, then sqrt — i.e. sup_t RMSE_t.
    per_time_mse = squared_errors.mean(axis=0)
    rmse_sup = float(np.sqrt(np.max(per_time_mse)))

    assert representative_traj_n is not None
    return ChaosErrorResult(
        n_particles=n_particles,
        t_grid=t_grid,
        mean_traj_n=representative_traj_n,
        mean_traj_inf=traj_inf,
        l2_error_sup=rmse_sup,
        n_replicates=n_replicates,
    )


@dataclass(frozen=True)
class ChaosScalingResult:
    """Output of `propagation_of_chaos_scaling` over a sweep of $n$.

    `n_grid`: array of particle counts swept (ascending).
    `rmse_sup`: $\\sup_t \\sqrt{E[(G_n - G_\\infty)^2]}$ at each $n$.
    `fitted_slope`: log-log slope of `rmse_sup` vs `1/sqrt(n)`; should be
        $\\approx 1$ when plotted against $1/\\sqrt n$ (i.e. $\\approx -1/2$
        when plotted vs $n$).
    `fitted_intercept`: log-log intercept; intercept $\\approx \\log\\sqrt{C(T)}$
        with $C(T)$ from `propagation_of_chaos_constant`.
    `theoretical_constant`: $C(T)$ from the closed form.
    """

    n_grid: NDArray[np.int64]
    rmse_sup: NDArray[np.float64]
    fitted_slope: float
    fitted_intercept: float
    theoretical_constant: float


def propagation_of_chaos_scaling(
    *,
    n_grid: NDArray[np.int64],
    theta_G: float,
    sigma_G: float,
    g_target: Callable[[float], float],
    G0_mean: float,
    G0_std: float,
    T: float,
    n_steps: int,
    n_replicates: int = 32,
    seed: int | None = None,
) -> ChaosScalingResult:
    """Sweep $n \\in n_grid$ and verify the $1/\\sqrt n$ RMSE scaling.

    Returns the empirical RMSEs plus a least-squares fit of
    $\\log(\\text{RMSE}) = a \\cdot \\log(1/\\sqrt n) + b$.  The Sznitman
    bound predicts $a \\approx 1$ (i.e. RMSE $\\propto 1/\\sqrt n$).
    """
    if n_grid.ndim != 1 or len(n_grid) < 2:
        raise ValueError("n_grid must be 1D with >= 2 entries")
    if not np.all(np.diff(n_grid) > 0):
        raise ValueError("n_grid must be strictly ascending")

    rmses = np.zeros(len(n_grid), dtype=np.float64)
    for i, n in enumerate(n_grid):
        # Use a different seed per n so noise patterns aren't correlated
        # across particle-count buckets.
        n_seed = None if seed is None else int(seed + 1000 * i)
        result = propagation_of_chaos_error(
            n_particles=int(n),
            theta_G=theta_G,
            sigma_G=sigma_G,
            g_target=g_target,
            G0_mean=G0_mean,
            G0_std=G0_std,
            T=T,
            n_steps=n_steps,
            n_replicates=n_replicates,
            seed=n_seed,
        )
        rmses[i] = result.l2_error_sup

    # Least-squares fit on log(RMSE) vs log(1/sqrt(n)).
    x = np.log(1.0 / np.sqrt(n_grid.astype(np.float64)))
    y = np.log(rmses)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    C_T = propagation_of_chaos_constant(
        theta_G=theta_G,
        sigma_G=sigma_G,
        var_G0=G0_std * G0_std,
        T=T,
    )

    return ChaosScalingResult(
        n_grid=n_grid.astype(np.int64),
        rmse_sup=rmses,
        fitted_slope=float(slope),
        fitted_intercept=float(intercept),
        theoretical_constant=float(C_T),
    )
