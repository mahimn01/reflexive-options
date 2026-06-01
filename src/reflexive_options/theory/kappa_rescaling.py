"""Change-of-variables between per-USD coupling and the dimensionless Hopf coupling.

Resolves the unit-chain incoherence flagged as blocking issue #2: the dimensionless
Hopf threshold ``kappa* = 17.81`` (theory.md 4.3.5) is derived in a regime where the
dealer-gamma functional ``G`` is O(1), whereas the empirically-tuned coupling
``kappa_0 ~ 5e-12`` is *per USD of dollar-gamma* and multiplies a ``G`` carrying
USD-gamma units of magnitude ~1e9..1e13. The two ``kappa`` quantities live in
different unit systems separated by the characteristic ``G`` scale and may NOT be
compared directly.

The SDE drift term (theory.md Eq. 1a) is

    drift = kappa * G(S, z, v)         [units of 1/yr]

This is invariant under the change of variables

    G_tilde := G / G_char              (dimensionless)
    kappa_dimensionless := kappa * G_char

so that ``kappa * G == kappa_dimensionless * G_tilde`` exactly. ``G_char`` is the
*characteristic dealer-gamma scale*: the magnitude of ``G`` at the equilibrium spot
under the relevant open-interest grid, in the same USD-gamma units the aggregator
(`simulator.gamma_aggregator.GammaAggregator.compute`) returns.

This module supplies (a) the exact map ``kappa_dimensionless = kappa_per_usd * G_char``,
(b) its inverse, and (c) ``characteristic_gamma_scale``, which reads ``G_char`` straight
off the same aggregator used by the simulator so the map is anchored to the actual
code path rather than a hand-typed constant.

Honest caveat (see ``paper/kappa_rescaling_map.md``): ``G_char`` for real SPX is
*not known* until the Phase-4 data lands. It scales as ``S0^2 * (total OI) *
(per-share BS gamma)`` and therefore swings several orders of magnitude across
plausible (S0, OI) assumptions. Consequently the dimensionless image of
``kappa_0 ~ 5e-12`` is INDETERMINATE relative to ``kappa* = 17.81`` at the pre-data
stage. This module exposes that indeterminacy quantitatively via
``dimensionless_kappa_band`` rather than papering over it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reflexive_options.simulator.gamma_aggregator import (
    GammaAggregator,
    GammaAggregatorConfig,
)
from reflexive_options.types import OpenInterestGrid


def characteristic_gamma_scale(
    aggregator: GammaAggregator,
    spot: float,
    variance: float,
    *,
    log_moneyness_halfwidth: float = 0.15,
    n_scan: int = 61,
) -> float:
    """Characteristic ``|G|`` scale (USD-gamma units) for a given aggregator.

    Defined as ``max_x |G(S e^x, v)|`` over ``x in [-h, h]`` log-spot, i.e. the peak
    dealer-gamma magnitude the aggregator produces in a neighbourhood of ``spot``.
    This is the ``G_char`` that nondimensionalises the coupling. Using the peak
    (rather than the ATM point value) makes ``G_char`` robust to ``G`` having a sign
    change or near-zero at the equilibrium, which is exactly the regime theory.md
    4.3.6 warns about (``G_y`` small and sign-changing near ATM).

    Args:
        aggregator: the same GammaAggregator the simulator uses.
        spot: equilibrium spot ``S^star``.
        variance: equilibrium variance ``v^star = theta_v``.
        log_moneyness_halfwidth: half-width of the log-spot scan.
        n_scan: number of scan points.

    Returns:
        ``G_char > 0`` in USD-gamma-per-unit-return units.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be positive, got {spot}")
    xs = np.linspace(-log_moneyness_halfwidth, log_moneyness_halfwidth, n_scan)
    g_vals = np.array(
        [abs(aggregator.compute(float(spot * np.exp(x)), float(variance), 0.0)) for x in xs]
    )
    g_char = float(np.max(g_vals))
    if g_char == 0.0:
        raise ValueError(
            "characteristic G scale is exactly zero — the OI grid produces no "
            "dealer gamma; the coupling cannot be nondimensionalised."
        )
    return g_char


def per_usd_to_dimensionless(kappa_per_usd: float, g_char: float) -> float:
    """Map a per-USD coupling to the dimensionless coupling: ``kappa_per_usd * G_char``.

    Exact change of variables: the SDE drift ``kappa_per_usd * G_USD`` equals
    ``(kappa_per_usd * G_char) * (G_USD / G_char)``, so the returned value is the
    coupling that multiplies the dimensionless ``G_tilde = G/G_char``.
    """
    if g_char <= 0.0:
        raise ValueError(f"g_char must be positive, got {g_char}")
    return kappa_per_usd * g_char


def dimensionless_to_per_usd(kappa_dimensionless: float, g_char: float) -> float:
    """Inverse map: ``kappa_dimensionless / G_char``."""
    if g_char <= 0.0:
        raise ValueError(f"g_char must be positive, got {g_char}")
    return kappa_dimensionless / g_char


@dataclass(frozen=True)
class DimensionlessKappaBand:
    """Plausible range of the dimensionless image of a per-USD coupling.

    Because ``G_char`` is unknown pre-data, the dimensionless image of a fixed
    ``kappa_per_usd`` is an interval, not a point. ``contains_kappa_star`` records
    whether ``kappa* = 17.81`` falls inside that interval — if it does, the claim
    "the market sits near the Hopf" is neither confirmed nor refuted by the
    dimensionless threshold alone.
    """

    kappa_per_usd: float
    g_char_lo: float
    g_char_hi: float
    kappa_dim_lo: float
    kappa_dim_hi: float
    kappa_star: float
    orders_of_magnitude: float
    contains_kappa_star: bool


def dimensionless_kappa_band(
    kappa_per_usd: float,
    g_char_lo: float,
    g_char_hi: float,
    *,
    kappa_star: float = 17.8065068,
) -> DimensionlessKappaBand:
    """Image interval of ``kappa_per_usd`` under the rescaling, given a ``G_char`` range.

    Args:
        kappa_per_usd: the per-USD coupling anchor (e.g. 5e-12).
        g_char_lo: smallest plausible characteristic gamma scale.
        g_char_hi: largest plausible characteristic gamma scale.
        kappa_star: the dimensionless Hopf threshold to compare against.
    """
    if not (0.0 < g_char_lo <= g_char_hi):
        raise ValueError(f"need 0 < g_char_lo <= g_char_hi, got ({g_char_lo}, {g_char_hi})")
    lo = per_usd_to_dimensionless(kappa_per_usd, g_char_lo)
    hi = per_usd_to_dimensionless(kappa_per_usd, g_char_hi)
    return DimensionlessKappaBand(
        kappa_per_usd=kappa_per_usd,
        g_char_lo=g_char_lo,
        g_char_hi=g_char_hi,
        kappa_dim_lo=lo,
        kappa_dim_hi=hi,
        kappa_star=kappa_star,
        orders_of_magnitude=float(np.log10(hi / lo)),
        contains_kappa_star=bool(lo <= kappa_star <= hi),
    )


def flat_oi_characteristic_scale(
    *,
    spot: float,
    oi_per_cell: float,
    iv: float = 0.20,
    risk_free_rate: float = 0.0,
    multiplier: float = 100.0,
    log_moneyness_halfwidth: float = 0.15,
    n_strikes: int = 7,
    maturities: tuple[float, ...] = (1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0),
) -> float:
    """Convenience: ``G_char`` for a flat OI grid, mirroring theory.md 7.1 fixture.

    Builds the same flat ``n_strikes x len(maturities)`` OI grid the stationary-density
    scan uses, then reads ``G_char`` via :func:`characteristic_gamma_scale`. Keeps the
    rescaling map anchored to the real aggregator and to a documented OI fixture.
    """
    from reflexive_options.types import SurfaceGrid  # local import: avoid cycle at module load

    logm = np.linspace(-log_moneyness_halfwidth, log_moneyness_halfwidth, n_strikes)
    mats = np.asarray(maturities, dtype=np.float64)
    grid = SurfaceGrid(log_moneyness=logm.astype(np.float64), maturities=mats)
    oi = OpenInterestGrid(
        grid=grid,
        contracts_open=np.full((n_strikes, len(maturities)), float(oi_per_cell)),
    )
    agg = GammaAggregator(
        oi_grid=oi,
        risk_free_rate=risk_free_rate,
        config=GammaAggregatorConfig(multiplier=multiplier, fixed_iv=iv),
    )
    return characteristic_gamma_scale(
        agg, spot=spot, variance=iv * iv, log_moneyness_halfwidth=log_moneyness_halfwidth
    )
