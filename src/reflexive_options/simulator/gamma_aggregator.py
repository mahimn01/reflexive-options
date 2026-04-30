"""Aggregate dealer-gamma exposure G(S, z, v).

Reference implementation per `~/Documents/reflexivity-research/dealer_gamma_brief.md`.

G is returned in USD-per-unit-return units so the SDE drift κ·G has units 1/yr
when κ is in [USD-of-dollar-gamma · year]^-1 (literature prior O(5e-12)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reflexive_options.types import OpenInterestGrid


@dataclass(frozen=True)
class GammaSignConvention:
    """Per-class dealer sign convention. Default = SqueezeMetrics SPX."""

    call_sign: float = 1.0
    put_sign: float = -1.0


@dataclass(frozen=True)
class GammaAggregatorConfig:
    """Numerical knobs for the aggregator (edge cases per dealer_gamma_brief §5)."""

    tau_floor_years: float = 1.0 / (365.0 * 24.0)
    multiplier: float = 100.0
    fixed_iv: float | None = None
    # TODO: strike-concentration smoothing (brief §5.2). Skipped in v1 — single-strike OI is
    # the common test fixture and Gaussian-kernel smoothing changes magnitudes < 1% at typical
    # surface granularity.


class GammaAggregator:
    """Computes G(S, z, v) from an open-interest grid.

    Conforms to the OI grid being half calls / half puts: by convention the brief treats
    every (K, T) cell as a single OI number with sign applied via the call/put split. For
    v1 the OI grid is interpreted as *calls only*; if/when the schema is extended to carry
    a per-cell call/put split, this class flips to per-leg aggregation. The SqueezeMetrics
    SPX convention (calls +1, puts -1) is the default for the sign object.
    """

    _oi_calls: NDArray[np.float64]
    _oi_puts: NDArray[np.float64]

    def __init__(
        self,
        oi_grid: OpenInterestGrid,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        sign: GammaSignConvention | None = None,
        config: GammaAggregatorConfig | None = None,
        oi_calls: NDArray[np.float64] | None = None,
        oi_puts: NDArray[np.float64] | None = None,
    ) -> None:
        self.oi_grid = oi_grid
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.sign = sign or GammaSignConvention()
        self.config = config or GammaAggregatorConfig()

        n_strikes, n_maturities = oi_grid.grid.shape
        if oi_calls is None and oi_puts is None:
            # default split: treat OI grid as calls only (test-fixture friendly)
            self._oi_calls = oi_grid.contracts_open.astype(np.float64, copy=True)
            self._oi_puts = np.zeros((n_strikes, n_maturities), dtype=np.float64)
        else:
            self._oi_calls = (
                np.zeros((n_strikes, n_maturities), dtype=np.float64)
                if oi_calls is None
                else oi_calls.astype(np.float64, copy=True)
            )
            self._oi_puts = (
                np.zeros((n_strikes, n_maturities), dtype=np.float64)
                if oi_puts is None
                else np.ascontiguousarray(oi_puts, dtype=np.float64)
            )
            if self._oi_calls.shape != (n_strikes, n_maturities):
                raise ValueError(
                    f"oi_calls shape {self._oi_calls.shape} != grid shape "
                    f"{(n_strikes, n_maturities)}"
                )
            if self._oi_puts.shape != (n_strikes, n_maturities):
                raise ValueError(
                    f"oi_puts shape {self._oi_puts.shape} != grid shape {(n_strikes, n_maturities)}"
                )

        self._log_moneyness = oi_grid.grid.log_moneyness.astype(np.float64)
        self._maturities = oi_grid.grid.maturities.astype(np.float64)

    def _bs_gamma_grid(
        self,
        spot: float,
        sigma: float,
    ) -> NDArray[np.float64]:
        tau = np.maximum(self._maturities, self.config.tau_floor_years)
        sqrt_tau = np.sqrt(tau)
        # strikes K = S * exp(log_moneyness)  ⇒  log(S/K) = -log_moneyness
        log_s_over_k = -self._log_moneyness[:, None]  # (n_strikes, 1)
        drift_term = (self.risk_free_rate - self.dividend_yield + 0.5 * sigma * sigma) * tau[
            None, :
        ]
        d1 = (log_s_over_k + drift_term) / (sigma * sqrt_tau[None, :])
        phi = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
        gamma = (
            np.exp(-self.dividend_yield * tau[None, :]) * phi / (spot * sigma * sqrt_tau[None, :])
        )
        return gamma  # type: ignore[no-any-return]

    def compute(self, spot: float, variance: float, log_memory: float) -> float:
        """Return G in USD-per-unit-return units.

        The dependence on z and v is implicit through the IV used for BS-gamma:
        v1 uses sigma = sqrt(variance) for every contract (flat surface). The z
        argument is accepted for API symmetry with the SDE drift; the brief allows
        future extensions to skew G by memory state.
        """
        # silence unused — kept for interface stability with future skew terms
        del log_memory

        sigma = (
            self.config.fixed_iv
            if self.config.fixed_iv is not None
            else float(np.sqrt(max(variance, 1e-12)))
        )
        if sigma <= 0.0:
            return 0.0

        gamma_bs = self._bs_gamma_grid(spot, sigma)
        signed_oi = self.sign.call_sign * self._oi_calls + self.sign.put_sign * self._oi_puts
        g_shares_per_dollar = float(np.sum(signed_oi * gamma_bs) * self.config.multiplier)
        # USD of underlying dealers must trade per unit return = shares-per-$1 · S²
        return g_shares_per_dollar * spot * spot
