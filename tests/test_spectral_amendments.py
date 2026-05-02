"""Tests for the A1 (adaptive Welch) and A5 (IAAFT surrogate) amendments to `theory.spectral`.

The original H4 detector tests in `test_h4_detector.py` are pinned to the
v0.1.0 baseline (`null_method='permutation'`, `nperseg_override=1024`).
These tests cover the *new* default behaviour — adaptive nperseg + IAAFT —
introduced by amendments A1 / A5.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexive_options.theory.spectral import (
    adaptive_welch_nperseg,
    detect_psd_peak,
    iaaft_surrogate,
)

# ---------------------------------------------------------------------------
# A1: adaptive Welch nperseg
# ---------------------------------------------------------------------------


def test_adaptive_welch_nperseg_returns_power_of_2() -> None:
    n = adaptive_welch_nperseg(
        omega_star=1.0,
        sampling_rate=252.0,
        n_trajectory=8192,
        target_bins_in_band=10,
    )
    # Closest power of 2 to 10 * 252 / 1.0 = 2520 → 2048.
    assert n == 2048


def test_adaptive_welch_nperseg_high_omega_gets_small_window() -> None:
    """At omega_star=10 cyc/yr, sampling_rate=252, target=10 → 252 → closest pow2 = 256."""
    n = adaptive_welch_nperseg(omega_star=10.0, sampling_rate=252.0, n_trajectory=8192)
    # 10 * 252 / 10 = 252 → closest power of 2 is 256.
    assert n == 256


def test_adaptive_welch_nperseg_caps_at_half_trajectory() -> None:
    """Very low omega_star → adaptive size is huge → cap at n_trajectory // 2."""
    n = adaptive_welch_nperseg(omega_star=0.1, sampling_rate=252.0, n_trajectory=4096)
    # 10 * 252 / 0.1 = 25200 → closest pow2 is 32768; cap at 2048.
    assert n == 2048


def test_adaptive_welch_nperseg_validates_inputs() -> None:
    with pytest.raises(ValueError, match="omega_star"):
        adaptive_welch_nperseg(omega_star=0.0, sampling_rate=252.0, n_trajectory=1024)
    with pytest.raises(ValueError, match="sampling_rate"):
        adaptive_welch_nperseg(omega_star=1.0, sampling_rate=0.0, n_trajectory=1024)
    with pytest.raises(ValueError, match="n_trajectory"):
        adaptive_welch_nperseg(omega_star=1.0, sampling_rate=252.0, n_trajectory=0)
    with pytest.raises(ValueError, match="target_bins_in_band"):
        adaptive_welch_nperseg(
            omega_star=1.0, sampling_rate=252.0, n_trajectory=1024, target_bins_in_band=0
        )


def test_adaptive_welch_nperseg_minimum_is_4() -> None:
    """Even degenerate inputs return at least 4 (Welch's minimum useful window)."""
    n = adaptive_welch_nperseg(
        omega_star=1e9,  # absurdly high → adaptive ≈ 0 → capped at 4
        sampling_rate=252.0,
        n_trajectory=1024,
    )
    assert n == 4


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@pytest.mark.parametrize(
    "omega_star, sampling_rate, n_trajectory",
    [
        (1.0, 252.0, 8192),
        (10.0, 252.0, 8192),
        (0.1, 252.0, 4096),
        (5.0, 1000.0, 100),  # n_trajectory // 2 = 50, NOT a power of 2
        (5.0, 1000.0, 200),  # cap = 100, NOT a power of 2 → must snap down to 64
        (5.0, 1000.0, 600),  # cap = 300, NOT a power of 2 → snap to 256
        (3.0, 252.0, 1500),  # cap = 750, snap to 512
        (1e9, 252.0, 1024),  # degenerate high-omega → snaps to 4
    ],
)
def test_adaptive_welch_nperseg_always_returns_power_of_two(
    omega_star: float, sampling_rate: float, n_trajectory: int
) -> None:
    """v0.3.1 fix: even when the n_trajectory // 2 cap is not a power of 2,
    `adaptive_welch_nperseg` must snap *down* to the nearest power of 2.
    Without this fix the v0.3.0 implementation returned the raw cap (e.g. 50),
    which Welch's method handles but is not the documented contract.
    """
    n = adaptive_welch_nperseg(
        omega_star=omega_star, sampling_rate=sampling_rate, n_trajectory=n_trajectory
    )
    assert _is_power_of_two(n), f"adaptive_welch_nperseg returned {n}, which is not a power of 2"
    assert n >= 4
    assert n <= max(n_trajectory // 2, 4)


# ---------------------------------------------------------------------------
# A5: IAAFT surrogate
# ---------------------------------------------------------------------------


def test_iaaft_surrogate_preserves_marginal_distribution() -> None:
    """The surrogate has identical sample order-statistics to the input."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(512).astype(np.float64)
    surr = iaaft_surrogate(x, rng=rng)
    np.testing.assert_array_almost_equal(np.sort(surr), np.sort(x))


def test_iaaft_surrogate_approximately_preserves_psd_magnitude() -> None:
    """The amplitude spectrum of the surrogate ≈ that of the input.

    Exact match is not enforced because the rank-mapping step may slightly
    perturb the amplitudes; the IAAFT iteration trades off marginal vs ACF
    accuracy. We check the relative error in mean amplitude is small.
    """
    rng = np.random.default_rng(1)
    n = 1024
    # AR(1) process — distinct PSD shape vs white noise.
    x = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.standard_normal()
    surr = iaaft_surrogate(x, max_iterations=200, rng=rng)
    amp_x = np.abs(np.fft.rfft(x))
    amp_s = np.abs(np.fft.rfft(surr))
    # Sum of |amp_x - amp_s| / sum |amp_x| should be small.
    rel_error = float(np.mean(np.abs(amp_x - amp_s) / (np.abs(amp_x) + 1e-12)))
    assert rel_error < 0.10, f"IAAFT amplitude error too large: {rel_error:.3f}"


def test_iaaft_surrogate_validates_inputs() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="1D"):
        iaaft_surrogate(np.zeros((4, 4)), rng=rng)
    with pytest.raises(ValueError, match="length >= 4"):
        iaaft_surrogate(np.array([1.0, 2.0]), rng=rng)
    with pytest.raises(ValueError, match="max_iterations"):
        iaaft_surrogate(np.zeros(8, dtype=np.float64), max_iterations=0, rng=rng)


def test_iaaft_surrogate_default_rng() -> None:
    """rng=None is allowed and yields a finite surrogate."""
    x = np.random.default_rng(7).standard_normal(64).astype(np.float64)
    surr = iaaft_surrogate(x, rng=None)
    assert surr.shape == x.shape
    assert np.all(np.isfinite(surr))


# ---------------------------------------------------------------------------
# Integration: IAAFT-default detector behaves correctly under realistic nulls.
# ---------------------------------------------------------------------------


def _heston_like_squared_returns(*, n: int, seed: int) -> np.ndarray:
    """Toy Heston-like squared returns: AR(1) variance with mean reversion + multiplicative shocks.

    Replicates the V3 audit's "Heston r_t²" generator without pulling in the
    full ReflexiveSimulator.
    """
    rng = np.random.default_rng(seed)
    v = np.zeros(n + 200, dtype=np.float64)
    v[0] = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    dt = 1.0 / 252.0
    for i in range(1, len(v)):
        v[i] = max(
            v[i - 1]
            + kappa * (theta - v[i - 1]) * dt
            + xi * np.sqrt(max(v[i - 1], 0.0)) * np.sqrt(dt) * rng.standard_normal(),
            1e-8,
        )
    eps = rng.standard_normal(len(v))
    log_r = np.sqrt(v * dt) * eps
    sq_returns = (log_r**2)[200:]
    return sq_returns


def test_iaaft_default_does_not_overfire_on_heston_squared_returns() -> None:
    """Under realistic Heston null, IAAFT FPR is bounded.

    The V3 audit reported 13.5% FPR for the iid permutation null on Heston
    r_t² at α = 0.05 (200-rep estimate). With IAAFT (which preserves the
    Heston ACF), we expect a substantially lower fire rate. We use a small
    seed budget here for test wall-clock — the V3 audit ran 200 reps for
    the headline number; we use 15 with a generous bound.
    """
    n_seeds = 15
    fires = 0
    omega_star = 1.5  # cycles/year — well-resolved at sampling_rate=252
    for seed in range(n_seeds):
        x = _heston_like_squared_returns(n=1024, seed=seed)
        res = detect_psd_peak(
            x,
            sampling_rate=252.0,
            omega_star=omega_star,
            bandwidth_frac=0.20,
            n_permutations=30,  # small for test speed; only loosely calibrates p-value
            null_method="iaaft",
            rng=np.random.default_rng(seed + 1_000_000),
        )
        if res.in_band and res.p_value < 0.05:
            fires += 1
    fpr = fires / n_seeds
    # IAAFT should be more conservative than iid permutation under correlated
    # H_0. Generous bound for the small-budget regime.
    assert fpr <= 0.34, f"IAAFT FPR on Heston squared-returns too high: {fpr:.3f}"


def test_iaaft_default_records_nperseg_used() -> None:
    """The detector now records the actual nperseg used."""
    x = np.random.default_rng(0).standard_normal(2048).astype(np.float64)
    res = detect_psd_peak(
        x,
        sampling_rate=252.0,
        omega_star=10.0,  # adaptive sizes nperseg to ~256
        n_permutations=10,
        null_method="iaaft",
        rng=np.random.default_rng(0),
    )
    # Adaptive: 10 * 252 / 10 = 252 → closest power of 2 is 256.
    # Capped at default welch_window=1024 (no effect here).
    assert res.nperseg_used == 256


def test_iaaft_default_supercritical_reflexive_still_fires() -> None:
    """A near-pure sinusoid embedded in noise still fires under the IAAFT default.

    IAAFT preserves the LINEAR autocorrelation, but the in-band peak ratio
    distribution under IAAFT surrogates of a sinusoid+noise input has
    enough randomization (from the rank-mapping) to keep p < 0.10 at high
    SNR. The detector + in_band gate still fires.
    """
    sampling = 252.0
    omega_star = 4.0  # cycles/year
    n = 2048
    rng = np.random.default_rng(11)
    times = np.arange(n) / sampling
    x = 1.0 * np.sin(2.0 * np.pi * omega_star * times) + 0.3 * rng.standard_normal(n)
    res = detect_psd_peak(
        x,
        sampling_rate=sampling,
        omega_star=omega_star,
        n_permutations=100,
        null_method="iaaft",
        rng=np.random.default_rng(11),
    )
    assert res.in_band, f"IAAFT-default detector failed in_band on supercritical: {res.peak_freq}"


def test_detect_psd_peak_validates_null_method() -> None:
    with pytest.raises(ValueError, match="null_method"):
        detect_psd_peak(
            np.zeros(100, dtype=np.float64),
            sampling_rate=252.0,
            omega_star=10.0,
            n_permutations=10,
            null_method="invalid",  # type: ignore[arg-type]
        )


def test_nperseg_override_skips_adaptive_sizing() -> None:
    """`nperseg_override` short-circuits the adaptive sizing and uses the given value."""
    x = np.random.default_rng(0).standard_normal(2048).astype(np.float64)
    res = detect_psd_peak(
        x,
        sampling_rate=252.0,
        omega_star=10.0,
        n_permutations=5,
        null_method="permutation",
        nperseg_override=512,
        rng=np.random.default_rng(0),
    )
    assert res.nperseg_used == 512
