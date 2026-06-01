"""GEX-conditioned reflexive simulator for data-free validation of H1'.

This is a NEW, self-contained generator (it does not modify the frozen
``simulator/reflexive.py``). Its purpose is to produce synthetic data with a
KNOWN ground-truth coupling ``kappa`` together with a synthetic end-of-day
option open-interest (OI) grid, so that the entire empirical H1' pipeline --
estimate GEX from the OI grid, then regress next-day vol-of-vol on it -- can be
validated exactly the way the original H4 was, but on the redesigned test.

Mechanism (the falsifiable core of theory.md predictions 3-4):

    G_t : signed dealer gamma exposure, a slow-moving, persistent latent state
          (AR(1)). It is reconstructable from the OI grid via Black-Scholes
          gamma, so it is observable to the econometrician.
    sign(G_t) conditions the variance feedback. Define the standardized signed
    gamma s_t = G_t / sigma_G and a bounded "feedback intensity"

        f_t = -tanh(kappa * s_t)            in (-1, 1)

    f_t > 0 when dealers are net SHORT gamma (G_t < 0): hedging is
    destabilizing. f_t < 0 when net LONG gamma: hedging damps. kappa scales the
    strength; kappa = 0 gives f_t == 0 for all t -> pure Heston, GEX irrelevant.

    The instantaneous (log-)variance follows a stable discrete AR(1) driven by
    TWO bounded, persistent reflexive channels:

        sigma_eff(t) = sigma_lv * (1 + volvol_gain * f_t)      (>0, bounded)
        m_t          = level_gain * f_t                        (level shift)
        log v_t = log theta + phi_lv * (log v_{t-1} - log theta)
                  + (1 - phi_lv) * m_t + sigma_eff(t) * eps_v
        r_t = (mu - 0.5 v_{t-1}) dt + sqrt(v_{t-1} dt) z_t     (return uses level)

    Because f_t is PERSISTENT (it inherits the AR(1) persistence of G_t), a
    short-gamma REGIME both raises the variance LEVEL (via m_t) and inflates its
    daily innovation (via sigma_eff) for a run of consecutive days. Since returns
    scale by sqrt(v_{t-1}), the rolling realized-vol series tracks that regime and
    its dispersion -- realized VOL-OF-VOL, the H1' primary outcome -- rises with
    (negative) GEX. At kappa=0 both channels vanish identically (clean null). The
    log form makes the variance unconditionally stationary; nothing blows up at
    any kappa.

The synthetic OI grid is constructed each day so that estimate_gex() recovers a
quantity proportional to G_t (up to convention sign and scale), closing the loop
between the latent driver and the observable used in the regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reflexive_options.empirical.gex_regression import OIGrid


@dataclass
class GEXSimParams:
    """Parameters for the GEX-conditioned reflexive simulator."""

    kappa: float  # reflexive coupling strength (ground truth); 0 => Heston
    theta: float = 0.04  # long-run variance
    rho: float = -0.70  # leverage correlation
    v0: float = 0.04  # initial variance
    s0: float = 100.0  # initial spot
    dt: float = 1.0 / 252.0  # daily step
    mu: float = 0.0  # drift
    # Latent dealer-gamma (GEX) state process: persistent, mean-reverting at 0.
    gex_phi: float = 0.95  # AR(1) persistence (long-lived positioning regimes)
    gex_sigma: float = 1.0  # innovation std (sets the scale of G_t)
    gex_mean: float = 0.0  # long-run mean (0 => balanced positioning)
    # Realized-vol AR(1) backbone in log space (numerically bulletproof: log rv
    # can never go negative or blow up). theta_rv = long-run realized vol;
    # rv_phi = daily persistence of the log-realized-vol deviation.
    theta_rv: float = 0.12
    rv_phi: float = 0.85
    base_vov: float = 0.12  # baseline conditional vol-of-vol of the log-rv state
    # Reflexive gain: GEX modulates the CONDITIONAL VOL-OF-VOL of the log-rv
    # state via vov = base_vov * exp(vov_gain * f), f in (-1, 1). Short-gamma
    # regimes (f>0) raise the vol-of-vol, so the realized-vol series is more
    # dispersed -> elevated rolling realized vol-of-vol the next day, aligned
    # with -GEX. Vanishes identically at kappa=0 (f==0). Bounded f keeps vov
    # finite at any kappa.
    vov_gain: float = 4.00


@dataclass
class GEXSimOutput:
    """Container for one simulated path with its OI grids."""

    spot: np.ndarray  # (n_days,)
    variance: np.ndarray  # (n_days,)
    returns: np.ndarray  # (n_days,)
    gex_true: np.ndarray  # (n_days,) latent dealer gamma state
    grids: list[OIGrid]  # (n_days,) reconstructable OI grids
    vix: np.ndarray  # (n_days,) annualized vol proxy (control regressor)


class GEXReflexiveSimulator:
    """Simulate one GEX-conditioned reflexive path plus its daily OI grids.

    One path is the right unit: the empirical test sees ONE SPX history, so we
    validate that the H1' coefficient is recovered on a single path of realistic
    length (121-363 trading days), not pooled across thousands.
    """

    def __init__(self, params: GEXSimParams, seed: int | None = None):
        self.p = params
        self.rng = np.random.default_rng(seed)
        # Synthetic option ladder used to render the OI grid each day.
        self._k_offsets = np.array([-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10])
        self._taus = np.array([21.0 / 252.0, 63.0 / 252.0])

    def _build_grid(self, spot: float, sigma: float, g_true: float) -> OIGrid:
        """Render an OI grid whose squeeze_metrics GEX is proportional to g_true.

        Dealers are long calls (+) / short puts (-) (squeeze_metrics). A positive
        g_true loads net gamma onto calls; a negative g_true loads it onto puts.
        Near-ATM concentration makes the loaded leg dominate the BS-gamma-weighted
        aggregate, so sign(estimate_gex(grid)) == sign(g_true) with a monotone
        magnitude. Absolute scale is irrelevant (only standardized GEX enters the
        regression).
        """
        base_oi = 5000.0
        load = 4000.0 * g_true  # net directional gamma loading
        strikes, taus, sigmas, is_call, oi = [], [], [], [], []
        for off in self._k_offsets:
            k = spot * (1.0 + off)
            atm_weight = np.exp(-((off / 0.04) ** 2))  # concentrate near ATM
            for tau in self._taus:
                strikes.append(k)
                taus.append(tau)
                sigmas.append(sigma)
                is_call.append(True)
                oi.append(base_oi + max(load, 0.0) * atm_weight)
                strikes.append(k)
                taus.append(tau)
                sigmas.append(sigma)
                is_call.append(False)
                oi.append(base_oi + max(-load, 0.0) * atm_weight)
        return OIGrid(
            spot=spot,
            strike=np.array(strikes),
            tau=np.array(taus),
            sigma=np.array(sigmas),
            oi=np.array(oi),
            is_call=np.array(is_call, dtype=bool),
            contract_multiplier=100.0,
        )

    def simulate(self, n_days: int) -> GEXSimOutput:
        p = self.p

        spot = np.zeros(n_days)
        variance = np.zeros(n_days)
        returns = np.zeros(n_days)
        gex_true = np.zeros(n_days)
        vix = np.zeros(n_days)

        spot[0] = p.s0
        gex_true[0] = p.gex_mean + p.gex_sigma * self.rng.standard_normal()

        log_theta_rv = np.log(p.theta_rv)
        log_rv = log_theta_rv  # log realized vol, a stationary AR(1)
        variance[0] = float(np.exp(2.0 * log_rv))
        vix[0] = float(np.exp(log_rv))

        for t in range(1, n_days):
            # Latent dealer gamma: exogenous AR(1) positioning regime, drawn
            # independently of the contemporaneous return shock so it is a
            # legitimate (non-mechanical) predictor of y_{t+1}.
            eps_g = self.rng.standard_normal()
            gex_true[t] = (
                p.gex_mean
                + p.gex_phi * (gex_true[t - 1] - p.gex_mean)
                + p.gex_sigma * np.sqrt(1.0 - p.gex_phi**2) * eps_g
            )

            # Bounded feedback intensity conditioned on YESTERDAY's gamma sign.
            # Persistent (inherits G_t's AR(1) memory): short-gamma REGIMES
            # (f>0) raise the CONDITIONAL VOL-OF-VOL of the realized-vol state for
            # a run of days, so the rolling realized-vol series is more dispersed
            # -> elevated realized vol-of-vol the next day, aligned with -GEX.
            s = (gex_true[t - 1] - p.gex_mean) / max(p.gex_sigma, 1e-8)
            f = -np.tanh(p.kappa * s)
            vov = p.base_vov * np.exp(p.vov_gain * f)

            # Log realized-vol AR(1): unconditionally stationary, so realized vol
            # can never go negative regardless of kappa. At kappa=0, f==0 and
            # vov==base_vov (GEX irrelevant -> null). log_rv is clamped to a wide
            # band (rv in ~[0.6%, 240%] annualized) purely to prevent fat-tail
            # overflow of exp(ret); the clamp is far outside the operating range
            # so it does not affect the GEX->vol-of-vol relationship.
            log_rv = (
                log_theta_rv + p.rv_phi * (log_rv - log_theta_rv) + vov * self.rng.standard_normal()
            )
            log_rv = float(np.clip(log_rv, np.log(0.006), np.log(2.4)))
            rv = float(np.exp(log_rv))
            variance[t] = rv**2

            # Daily return is drawn from the current realized-vol level (per-day
            # std = rv / sqrt(252)); this is what makes the realized-vol estimator
            # downstream track the GEX-driven dispersion.
            ret = (rv / np.sqrt(252.0)) * self.rng.standard_normal()
            returns[t] = ret
            spot[t] = spot[t - 1] * np.exp(ret)
            vix[t] = rv

        # Build each OI grid at a FIXED reference IV (sqrt(theta)), NOT the
        # realized vol. Using realized vol would let the contemporaneous
        # variance leak into the BS-gamma weights and contaminate GEX with the
        # vol level (which itself correlates with vol-of-vol), injecting a
        # spurious same-sign GEX<->vol-of-vol relation. Pinning the IV makes
        # estimate_gex a clean monotone function of the latent dealer-gamma
        # state g_true alone -- exactly the observable the regression should see.
        ref_sigma = float(np.sqrt(p.theta))
        grids = [self._build_grid(spot[t], ref_sigma, gex_true[t]) for t in range(n_days)]
        return GEXSimOutput(
            spot=spot,
            variance=variance,
            returns=returns,
            gex_true=gex_true,
            grids=grids,
            vix=vix,
        )
