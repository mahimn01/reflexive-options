"""Validation suite for the H4 PSD-peak detector (`theory.spectral`).

Three tiers of tests, mirroring `experiments/h4_validation.py`:

  1. Synthetic ground-truth controls — deterministic sinusoid in/out of band,
     Stuart-Landau oscillator, white noise, Heston-style heteroskedastic
     noise. Each is a clean test that the detector responds the way it's
     supposed to on inputs whose spectral content is known a priori.

  2. p-value calibration — under the H_0 of i.i.d. white noise, the
     permutation p-value distribution should be (approximately) uniform on
     [0, 1]. A KS test against U[0,1] backs this up.

  3. Power curves — detection rate vs trajectory length T (at fixed SNR)
     and vs SNR (at fixed T). Both should be monotone increasing.

Plus an end-to-end positive control: the reflexive simulator at κ > κ* in
the canonical (dimensionless) Hopf-exhibiting regime triggers; the same
simulator at κ ≪ κ* does not.

Per the pre-reg's H4 decision rule (paper/pre_registration.md §6), "fires"
operationally means `result.in_band` *and* `result.p_value < 0.05`. For
inputs where the squared / absolute-return transform injects strong
multiplicative-noise harmonics (which show up as out-of-band peaks
dominating the in-band fundamental), we report the *peak frequency
location* against the band as a secondary diagnostic — this is one of the
operationally underspecified pre-reg items flagged in the validation report.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.theory.spectral import PSDPeakResult, detect_psd_peak
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SurfaceGrid,
)

# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

SAMPLING_RATE = 252.0  # daily samples / year
DEFAULT_T = 2048
DEFAULT_OMEGA_STAR = 20.0  # cycles/year — well-resolved by 1024-day Welch window


def _sinusoid_with_noise(
    *,
    T: int,
    sampling_rate: float,
    frequency_cyc_per_yr: float,
    amplitude: float,
    noise_std: float,
    seed: int,
) -> NDArray[np.float64]:
    """Single sinusoid + i.i.d. Gaussian noise. SNR (per-sample) = amplitude / noise_std."""
    t = np.arange(T) / sampling_rate
    rng = np.random.default_rng(seed)
    return amplitude * np.sin(
        2.0 * np.pi * frequency_cyc_per_yr * t
    ) + noise_std * rng.standard_normal(T)


def _stuart_landau_path(
    *,
    n_steps: int,
    dt: float,
    omega_rad_per_yr: float,
    mu: float,
    noise_sigma: float,
    seed: int,
) -> NDArray[np.float64]:
    """One Stuart-Landau (canonical Hopf normal form) trajectory.

    Real form of dz = (μ + i ω - |z|²) z dt + σ dW:
        dx = (μ x - ω y - (x² + y²) x) dt + σ dW^x
        dy = (ω x + μ y - (x² + y²) y) dt + σ dW^y

    Returns the x-component (of length n_steps), with a 1000-step burn-in
    discarded so the path samples the limit cycle, not the transient.
    """
    rng = np.random.default_rng(seed)
    burn = 1000
    total = n_steps + burn
    x = np.zeros(total, dtype=np.float64)
    y = np.zeros(total, dtype=np.float64)
    x[0], y[0] = 0.5, 0.0
    sqrt_dt = float(np.sqrt(dt))
    for t in range(total - 1):
        r2 = x[t] * x[t] + y[t] * y[t]
        dx = (
            mu * x[t] - omega_rad_per_yr * y[t] - r2 * x[t]
        ) * dt + noise_sigma * sqrt_dt * rng.standard_normal()
        dy = (
            omega_rad_per_yr * x[t] + mu * y[t] - r2 * y[t]
        ) * dt + noise_sigma * sqrt_dt * rng.standard_normal()
        x[t + 1] = x[t] + dx
        y[t + 1] = y[t] + dy
    return x[burn:]


def _white_noise(*, T: int, sigma: float = 1.0, seed: int) -> NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal(T) * sigma


def _heston_squared_returns(*, T: int, seed: int) -> NDArray[np.float64]:
    """Squared log-returns from a no-feedback Heston path through ReflexiveSimulator.

    Used as a negative-control input — Heston has volatility clustering
    (heteroskedastic noise) but no Hopf cycle. The detector should not
    report `in_band=True`.
    """
    grid = SurfaceGrid(
        log_moneyness=np.array([-0.05, 0.0, 0.05]),
        maturities=np.array([30 / 365.25, 90 / 365.25]),
    )
    oi = OpenInterestGrid(grid=grid, contracts_open=np.zeros(grid.shape, dtype=np.float64))
    aggregator = GammaAggregator(oi_grid=oi, risk_free_rate=0.05)
    heston = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    params = ReflexiveParams(base=heston, coupling=0.0, drift=0.0, memory_intake=0.0, leverage=0.0)
    sim = ReflexiveSimulator(
        params=params, gamma_aggregator=aggregator, initial_spot=100.0, antithetic=False
    )
    spots, _ = sim.simulate(n_paths=1, n_steps=T + 500, dt=1.0 / SAMPLING_RATE, seed=seed)
    log_S = np.log(np.maximum(spots[0, 500:], 1e-12))
    log_returns = np.diff(log_S)
    return log_returns**2  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Canonical Hopf-exhibiting regime for the reflexive simulator (test fixture)
#
# The default parameter set in `experiments/bifurcation_scan.py` is sub-Hopf
# within the literature κ-prior (paper/theory.md §4.2 caveat). For the H4
# detector tests we need a *supercritical* fixture, so we use the dimensionless
# regime described in §4.2, with the memory-channel time constant rescaled so
# the resulting Hopf frequency falls in a band the 1024-day Welch window can
# resolve (period ~165 trading days; cf. period ~10 yr in the literal §4.2
# dimensionless example, which the 1024-day window cannot resolve — see the
# H4 validation report's "underspecified spec items").
# ---------------------------------------------------------------------------

# Cycle frequency and Hopf threshold for these parameters (computed from
# `theory.bifurcation.hopf_scan` — see test_reflexive_supercritical_triggers
# for the recomputation that locks these in test code).
CANONICAL_OMEGA_STAR_RAD_PER_YR = 9.5957
CANONICAL_OMEGA_STAR_CYC_PER_YR = CANONICAL_OMEGA_STAR_RAD_PER_YR / (2.0 * np.pi)  # ≈ 1.527
CANONICAL_KAPPA_STAR = 18.020


class _DimensionlessGammaAggregator:
    """Hand-crafted G(S, v, z) for the canonical Hopf regime tests.

    G(S, v, z) = g_0 + G_x · log(S/S_0) + G_v · (v - v_0) + G_z · z + cubic · log(S/S_0)^3

    The cubic regulariser bounds the limit-cycle amplitude past κ*. Choosing
    `cubic` large enough (≈ -2) suppresses the multiplicative-noise harmonics
    in r_t² that would otherwise dominate the in-band fundamental — see the
    "underspecified spec items" note in the validation report for context.
    """

    def __init__(
        self,
        spot0: float,
        variance0: float,
        *,
        G_x: float,
        G_v: float,
        G_z: float,
        g0: float = 0.0,
        cubic: float = -2.0,
    ) -> None:
        self.S0 = spot0
        self.v0 = variance0
        self.G_x = G_x
        self.G_v = G_v
        self.G_z = G_z
        self.g0 = g0
        self.cubic = cubic
        self._log_S0 = float(np.log(spot0))

    def compute(self, spot: float, variance: float, log_memory: float) -> float:
        y = float(np.log(max(spot, 1e-12))) - self._log_S0
        u = variance - self.v0
        return self.g0 + self.G_x * y + self.G_v * u + self.G_z * log_memory + self.cubic * y**3

    def compute_batch(
        self,
        spots: NDArray[np.float64],
        variances: NDArray[np.float64],
        memories: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        spots = np.asarray(spots, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        memories = np.asarray(memories, dtype=np.float64)
        y = np.log(np.maximum(spots, 1e-12)) - self._log_S0
        u = variances - self.v0
        return self.g0 + self.G_x * y + self.G_v * u + self.G_z * memories + self.cubic * y**3  # type: ignore[no-any-return]


def _make_canonical_reflexive_sim(
    *,
    coupling_ratio: float,
    seed_friendly_initial_v: float = 0.05,
    base_xi: float = 0.1,
) -> ReflexiveSimulator:
    """Build the canonical ReflexiveSimulator at κ = `coupling_ratio` * κ*.

    `seed_friendly_initial_v` defaults slightly off equilibrium so the
    deterministic skeleton has something to perturb past κ*; the noise term
    (`base_xi`) keeps trajectories rich without dominating the cycle.
    """
    spot0 = 100.0
    variance_eq = 0.04
    aggregator = _DimensionlessGammaAggregator(
        spot0,
        variance_eq,
        G_x=0.5,
        G_v=-0.5,
        G_z=-0.5,
        g0=0.0,
        cubic=-2.0,
    )
    heston = HestonParams(
        kappa=2.0, theta=variance_eq, xi=base_xi, rho=-0.7, v0=seed_friendly_initial_v
    )
    params = ReflexiveParams(
        base=heston,
        coupling=coupling_ratio * CANONICAL_KAPPA_STAR,
        drift=0.5 * variance_eq,  # cancel Itô drift correction at equilibrium
        memory_decay=10.0,
        memory_intake=20.0,
        leverage=0.5,
    )
    return ReflexiveSimulator(
        params=params,
        gamma_aggregator=aggregator,
        initial_spot=spot0,
        antithetic=False,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tier 1 — synthetic positive / negative controls
# ---------------------------------------------------------------------------


def test_detect_pure_sinusoid_at_omega_star() -> None:
    """High-SNR sinusoid at exactly ω* should fire decisively (in_band, p < 0.01)."""
    x = _sinusoid_with_noise(
        T=DEFAULT_T,
        sampling_rate=SAMPLING_RATE,
        frequency_cyc_per_yr=DEFAULT_OMEGA_STAR,
        amplitude=1.0,
        noise_std=0.3,
        seed=11,
    )
    res = detect_psd_peak(
        x,
        sampling_rate=SAMPLING_RATE,
        omega_star=DEFAULT_OMEGA_STAR,
        n_permutations=200,
        rng=np.random.default_rng(11),
    )
    assert res.in_band, (
        f"in_band False: peak_freq={res.peak_freq}, expected near {DEFAULT_OMEGA_STAR}"
    )
    assert res.p_value < 0.01, f"p={res.p_value} not < 0.01"
    assert res.peak_to_background_ratio > 10.0, (
        f"peak/bg ratio = {res.peak_to_background_ratio} too small for high-SNR sinusoid"
    )
    # Frequency-resolution sanity: peak_freq within one Welch bin of ω*.
    bin_width = SAMPLING_RATE / 1024.0
    assert abs(res.peak_freq - DEFAULT_OMEGA_STAR) <= bin_width, (
        f"peak_freq {res.peak_freq} not within {bin_width:.3f} of {DEFAULT_OMEGA_STAR}"
    )


def test_no_detect_white_noise() -> None:
    """Pure white noise: detector should not fire — in_band tends to False, p tends to be large."""
    x = _white_noise(T=DEFAULT_T, seed=23)
    res = detect_psd_peak(
        x,
        sampling_rate=SAMPLING_RATE,
        omega_star=DEFAULT_OMEGA_STAR,
        n_permutations=300,
        rng=np.random.default_rng(23),
    )
    # Either the in-band peak is dominated by some other random bin, or the
    # p-value is large. Most seeds give both; we assert the disjunction so
    # the test is robust to any single-seed coincidences.
    assert (not res.in_band) or res.p_value > 0.10, (
        f"unexpected fire on white noise: in_band={res.in_band}, p={res.p_value}"
    )


def test_no_detect_sinusoid_outside_band() -> None:
    """Sinusoid at 2 ω* (way outside the ω* ± 20% band) should not fire as in-band."""
    out_of_band_freq = 2.0 * DEFAULT_OMEGA_STAR
    x = _sinusoid_with_noise(
        T=DEFAULT_T,
        sampling_rate=SAMPLING_RATE,
        frequency_cyc_per_yr=out_of_band_freq,
        amplitude=2.0,
        noise_std=0.3,
        seed=31,
    )
    res = detect_psd_peak(
        x,
        sampling_rate=SAMPLING_RATE,
        omega_star=DEFAULT_OMEGA_STAR,
        n_permutations=200,
        rng=np.random.default_rng(31),
    )
    assert not res.in_band, (
        f"unexpected in_band fire: peak_freq={res.peak_freq}, expected the global max at ~{out_of_band_freq}"
    )


def test_p_value_calibrated() -> None:
    """Under H_0 (white noise) the empirical p-value distribution is approximately uniform.

    KS test against U[0, 1]; want p > 0.05 (i.e. null *not* rejected).
    """
    rng = np.random.default_rng(0)
    n_runs = 60
    p_values = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        x = _white_noise(T=DEFAULT_T, seed=int(rng.integers(1_000_000)))
        res = detect_psd_peak(
            x,
            sampling_rate=SAMPLING_RATE,
            omega_star=DEFAULT_OMEGA_STAR,
            n_permutations=100,
            rng=np.random.default_rng(int(rng.integers(1_000_000))),
        )
        p_values[i] = res.p_value
    ks_stat, ks_p = stats.kstest(p_values, "uniform")
    assert ks_p > 0.05, (
        f"p-value distribution rejects U[0,1] (KS p = {ks_p:.4f}, stat = {ks_stat:.3f}); "
        f"distribution: mean = {p_values.mean():.3f}, frac<0.05 = {np.mean(p_values < 0.05):.3f}"
    )
    # Sanity: empirical false-positive rate at α=0.05 should be ≈ 0.05 (allow ±0.05 slack).
    fpr = float(np.mean(p_values < 0.05))
    assert fpr < 0.15, f"false-positive rate at α=0.05 too high: {fpr:.3f}"


def test_power_increases_with_T_at_fixed_snr() -> None:
    """Detection power (fraction of seeds firing) is monotone in trajectory length T."""
    fixed_snr_amplitude = 0.4
    fixed_noise_std = 1.0
    n_seeds = 30

    def _power(T: int) -> float:
        fires = 0
        for seed in range(n_seeds):
            x = _sinusoid_with_noise(
                T=T,
                sampling_rate=SAMPLING_RATE,
                frequency_cyc_per_yr=DEFAULT_OMEGA_STAR,
                amplitude=fixed_snr_amplitude,
                noise_std=fixed_noise_std,
                seed=seed,
            )
            res = detect_psd_peak(
                x,
                sampling_rate=SAMPLING_RATE,
                omega_star=DEFAULT_OMEGA_STAR,
                n_permutations=80,
                rng=np.random.default_rng(seed + 5_000),
            )
            if res.in_band and res.p_value < 0.05:
                fires += 1
        return fires / n_seeds

    p_short = _power(512)
    p_long = _power(4096)
    assert p_long >= p_short - 0.02, (
        f"power not monotone in T: power(T=512)={p_short:.2f}, power(T=4096)={p_long:.2f}"
    )
    # Also expect the long-T power to be high-ish for our chosen SNR.
    assert p_long > 0.5, f"long-T power unexpectedly low: {p_long:.2f}"


def test_stuart_landau_oscillator_triggers() -> None:
    """Stuart-Landau (canonical limit-cycle SDE) at supercritical μ triggers the detector."""
    omega_rad_per_yr = 2.0 * np.pi * 4.0  # 4 cycles/year
    omega_cyc = 4.0
    x = _stuart_landau_path(
        n_steps=4096,
        dt=1.0 / SAMPLING_RATE,
        omega_rad_per_yr=omega_rad_per_yr,
        mu=0.5,  # well past Hopf
        noise_sigma=0.1,
        seed=7,
    )
    res = detect_psd_peak(
        x,
        sampling_rate=SAMPLING_RATE,
        omega_star=omega_cyc,
        n_permutations=200,
        rng=np.random.default_rng(7),
    )
    assert res.in_band, f"SL at supercritical μ failed in_band: peak={res.peak_freq}"
    assert res.p_value < 0.05, f"SL p-value not small: {res.p_value}"


# ---------------------------------------------------------------------------
# Tier 2 — reflexive-simulator integration tests
# ---------------------------------------------------------------------------


def test_canonical_kappa_star_recomputable() -> None:
    """Sanity-check the locked CANONICAL_KAPPA_STAR / OMEGA_STAR constants.

    The reflexive-simulator integration tests below pin
    CANONICAL_KAPPA_STAR ≈ 18.02 and ω* ≈ 9.60 rad/yr (≈ 1.527 cyc/yr) for
    the dimensionless {G_x, G_v, G_z} = {0.5, -0.5, -0.5}, α=10/yr, β=20,
    γ=0.5, κ_v=2 regime. If any of these change, the test fixtures break,
    so we recompute from `theory.bifurcation.hopf_scan` here as a guard.
    """
    from reflexive_options.theory.bifurcation import hopf_scan, jacobian_3d

    G_x, G_v, G_z = 0.5, -0.5, -0.5
    alpha_decay, beta_intake, gamma_lev, kappa_v = 10.0, 20.0, 0.5, 2.0

    def jac(k: float) -> NDArray[np.float64]:
        a = k * G_x
        # b includes -0.5 ∂_v σ², with σ² = v ⇒ ∂_v σ² = 1
        b = k * G_v - 0.5
        return jacobian_3d(
            kappa=k,
            a_kappa=a,
            b_kappa=b,
            G_z=G_z,
            kappa_v=kappa_v,
            alpha=alpha_decay,
            beta=beta_intake,
            gamma=gamma_lev,
        )

    grid = np.linspace(0.01, 50.0, 2001)
    res = hopf_scan(grid, jac)
    assert res.kappa_star is not None and res.omega_at_crossing is not None
    assert abs(res.kappa_star - CANONICAL_KAPPA_STAR) < 0.05, (
        f"κ* drifted from locked value: got {res.kappa_star}, expected {CANONICAL_KAPPA_STAR}"
    )
    assert abs(res.omega_at_crossing - CANONICAL_OMEGA_STAR_RAD_PER_YR) < 0.1, (
        f"ω* drifted from locked value: got {res.omega_at_crossing}, expected {CANONICAL_OMEGA_STAR_RAD_PER_YR}"
    )


def test_reflexive_supercritical_triggers() -> None:
    """At κ = 1.05·κ* in the canonical regime, the squared-returns trigger the detector.

    Operationally "trigger" = `in_band` *and* `p_value < 0.05`, matching the
    pre-reg's H4 decision rule (paper/pre_registration.md §6) reduced to a
    single-trace go/no-go.
    """
    sim = _make_canonical_reflexive_sim(coupling_ratio=1.05)
    spots, _ = sim.simulate(n_paths=1, n_steps=8000, dt=1.0 / SAMPLING_RATE, seed=42)
    assert np.all(np.isfinite(spots)), "supercritical sim blew up"

    log_returns = np.diff(np.log(np.maximum(spots[0, 2000:], 1e-12)))
    sq_returns = log_returns**2

    res = detect_psd_peak(
        sq_returns,
        sampling_rate=SAMPLING_RATE,
        omega_star=CANONICAL_OMEGA_STAR_CYC_PER_YR,
        bandwidth_frac=0.20,
        welch_window=1024,
        welch_overlap=0.5,
        n_permutations=200,
        rng=np.random.default_rng(123),
    )
    assert res.in_band, (
        f"supercritical sim did not in-band: peak_freq={res.peak_freq}, "
        f"expected near {CANONICAL_OMEGA_STAR_CYC_PER_YR}"
    )
    assert res.p_value < 0.05, f"supercritical sim p-value not significant: {res.p_value}"
    assert res.peak_to_background_ratio > 3.0, (
        f"supercritical peak/background ratio unexpectedly low: {res.peak_to_background_ratio}"
    )


def test_reflexive_subcritical_does_not_trigger() -> None:
    """At κ ≪ κ* the reflexive sim is sub-Hopf — squared returns do not show an in-band peak."""
    sim = _make_canonical_reflexive_sim(coupling_ratio=0.30)
    spots, _ = sim.simulate(n_paths=1, n_steps=8000, dt=1.0 / SAMPLING_RATE, seed=42)
    assert np.all(np.isfinite(spots))

    log_returns = np.diff(np.log(np.maximum(spots[0, 2000:], 1e-12)))
    sq_returns = log_returns**2

    res = detect_psd_peak(
        sq_returns,
        sampling_rate=SAMPLING_RATE,
        omega_star=CANONICAL_OMEGA_STAR_CYC_PER_YR,
        bandwidth_frac=0.20,
        welch_window=1024,
        welch_overlap=0.5,
        n_permutations=200,
        rng=np.random.default_rng(456),
    )
    # The discriminant per the H4 decision rule: in_band must be True for a
    # fire. Subcritical reflexive (and pure Heston, see the test below) gives
    # a non-white spectrum from vol clustering, so p<0.05 is uninformative on
    # its own — we rely on `in_band` to filter out red-noise false positives.
    assert not res.in_band, (
        f"subcritical sim incorrectly in-band: peak_freq={res.peak_freq}, ratio={res.peak_to_background_ratio}"
    )


def test_no_detect_heston_squared_returns() -> None:
    """Pure Heston (no reflexive coupling) has volatility clustering but no Hopf cycle.

    Squared returns from Heston produce a non-white spectrum (so p<0.05 fires
    on the shuffle null), but the dominant component is a 1/f-like decay,
    not a localized in-band peak — `in_band` should be False.
    """
    sq_returns = _heston_squared_returns(T=4096, seed=42)
    res = detect_psd_peak(
        sq_returns,
        sampling_rate=SAMPLING_RATE,
        omega_star=CANONICAL_OMEGA_STAR_CYC_PER_YR,
        bandwidth_frac=0.20,
        welch_window=1024,
        welch_overlap=0.5,
        n_permutations=200,
        rng=np.random.default_rng(42),
    )
    assert not res.in_band, f"Heston squared-returns incorrectly in-band: peak_freq={res.peak_freq}"


# ---------------------------------------------------------------------------
# Tier 3 — input-validation / edge-case tests on the detector itself
# ---------------------------------------------------------------------------


def test_detect_psd_peak_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        detect_psd_peak(
            np.zeros((10, 2), dtype=np.float64),
            sampling_rate=SAMPLING_RATE,
            omega_star=DEFAULT_OMEGA_STAR,
            n_permutations=10,
        )


def test_detect_psd_peak_rejects_short_input() -> None:
    with pytest.raises(ValueError, match=">= 4"):
        detect_psd_peak(
            np.zeros(2, dtype=np.float64),
            sampling_rate=SAMPLING_RATE,
            omega_star=DEFAULT_OMEGA_STAR,
            n_permutations=10,
        )


def test_detect_psd_peak_rejects_bad_sampling_rate() -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        detect_psd_peak(
            np.ones(100, dtype=np.float64),
            sampling_rate=0.0,
            omega_star=DEFAULT_OMEGA_STAR,
            n_permutations=10,
        )


def test_detect_psd_peak_rejects_bad_omega_star() -> None:
    with pytest.raises(ValueError, match="omega_star"):
        detect_psd_peak(
            np.ones(100, dtype=np.float64),
            sampling_rate=SAMPLING_RATE,
            omega_star=0.0,
            n_permutations=10,
        )


def test_detect_psd_peak_rejects_bad_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth_frac"):
        detect_psd_peak(
            np.ones(100, dtype=np.float64),
            sampling_rate=SAMPLING_RATE,
            omega_star=DEFAULT_OMEGA_STAR,
            bandwidth_frac=1.5,
            n_permutations=10,
        )


def test_detect_psd_peak_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="welch_overlap"):
        detect_psd_peak(
            np.ones(100, dtype=np.float64),
            sampling_rate=SAMPLING_RATE,
            omega_star=DEFAULT_OMEGA_STAR,
            welch_overlap=1.0,
            n_permutations=10,
        )


def test_detect_psd_peak_zero_permutations_returns_p_one() -> None:
    """n_permutations = 0 short-circuits the p-value to 1.0 (no test performed)."""
    x = _sinusoid_with_noise(
        T=DEFAULT_T,
        sampling_rate=SAMPLING_RATE,
        frequency_cyc_per_yr=DEFAULT_OMEGA_STAR,
        amplitude=1.0,
        noise_std=0.3,
        seed=99,
    )
    res = detect_psd_peak(
        x,
        sampling_rate=SAMPLING_RATE,
        omega_star=DEFAULT_OMEGA_STAR,
        n_permutations=0,
    )
    assert res.p_value == 1.0


def test_detect_psd_peak_falls_back_to_short_window_for_short_input() -> None:
    """When len(x) < welch_window, the function should fall back to a shorter nperseg."""
    x = _sinusoid_with_noise(
        T=256,  # < default welch_window=1024
        sampling_rate=SAMPLING_RATE,
        frequency_cyc_per_yr=DEFAULT_OMEGA_STAR,
        amplitude=1.0,
        noise_std=0.3,
        seed=33,
    )
    res = detect_psd_peak(
        x,
        sampling_rate=SAMPLING_RATE,
        omega_star=DEFAULT_OMEGA_STAR,
        welch_window=1024,
        n_permutations=20,
    )
    assert isinstance(res, PSDPeakResult)
    assert np.isfinite(res.peak_to_background_ratio)


def test_detect_psd_peak_uses_default_rng_when_none() -> None:
    """rng=None is allowed and yields a finite p-value."""
    x = _white_noise(T=512, seed=5)
    res = detect_psd_peak(
        x,
        sampling_rate=SAMPLING_RATE,
        omega_star=DEFAULT_OMEGA_STAR,
        n_permutations=20,
        rng=None,
    )
    assert 0.0 < res.p_value <= 1.0


def test_detect_psd_peak_falls_back_when_band_misses_all_welch_bins() -> None:
    """When the band is so narrow / mis-aligned that no Welch bin lands in it,
    `peak_freq` falls back to the global maximum and `in_band` is False.

    Forces the empty-`in_band_mask` branch in `_peak_and_background`.
    """
    # 4-sample input → Welch nperseg=4 → 3 frequencies {0, fs/4, fs/2}.
    # With fs=10 those are {0, 2.5, 5.0}. Choose ω* = 0.5 with bw=20% so the
    # band is [0.4, 0.6] — none of the Welch bins fall in it.
    x = np.array([1.0, -1.0, 1.0, -1.0])
    res = detect_psd_peak(
        x,
        sampling_rate=10.0,
        omega_star=0.5,
        bandwidth_frac=0.20,
        welch_window=4,
        n_permutations=4,
        rng=np.random.default_rng(0),
    )
    assert not res.in_band
    assert np.isfinite(res.peak_to_background_ratio)


def test_detect_psd_peak_handles_zero_background() -> None:
    """When the off-band background is identically zero (rare), the ratio is finite.

    Forces the `background_power <= 0` floor branch in `_peak_and_background`.
    Use a very short, very tonal input so almost all PSD mass is in one bin.
    """
    # 4-sample alternating signal → all PSD mass in the Nyquist bin; the
    # other off-band bins are exactly 0. With ω* = 5.0 (the Nyquist) and bw
    # large enough to include only the Nyquist bin, the off-band background
    # is the median of {0, 0} = 0, hitting the floor.
    x = np.array([1.0, -1.0, 1.0, -1.0])
    res = detect_psd_peak(
        x,
        sampling_rate=10.0,
        omega_star=5.0,
        bandwidth_frac=0.10,
        welch_window=4,
        n_permutations=2,
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(res.peak_to_background_ratio)
    assert res.background_power > 0.0  # floor kicks in
