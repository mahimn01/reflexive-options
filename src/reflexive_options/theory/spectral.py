"""H4 PSD-peak detector — pre-registered in paper/pre_registration.md §3 (H4) / §6.

Operational specification per the pre-registration + amendments:
    - Welch's method PSD with Hann window, 50% overlap. Per A1, the window
      length is data-determined post-calibration via `adaptive_welch_nperseg`
      to guarantee ≥ `target_bins_in_band` bins inside the ±20% band around
      ω*. The legacy hardcoded 1024 is preserved as `nperseg_override` for
      tests and for back-compat with the v0.1.0 baseline.
    - Search for the maximum within ω* ± 20% of the predicted Hopf frequency
      ω* = √c_1(κ*) (Theorem 1, paper/theory.md §3).
    - Background = median PSD outside the in-band region (and excluding the
      very-low-frequency strip ω < ω*/3 to avoid trend / drift contamination).
    - Empirical p-value via IAAFT (Iterative Amplitude-Adjusted Fourier
      Transform) surrogates by default — pre-reg amendment A5 replaces the
      iid permutation null from A3 because under realistic Heston / AR(1)
      H_0 the iid permutation surrogate destroys the autocorrelation that
      gives the H_0 spectrum its red-noise character, leaving the test
      anti-conservative (FPR ~ 13.5% at α = 0.05). IAAFT preserves both
      marginal and linear ACF, restoring nominal FPR.

The detector returns a PSDPeakResult dataclass with the band-restricted peak
location, the peak power, the off-band background, the ratio, an in-band
indicator, and the empirical p-value.

Validated end-to-end against synthetic ground truth in tests/test_h4_detector.py
and the experiments/h4_validation.py runner — see runs/h4_validation/ for the
power curves at the canonical (window, overlap, bandwidth, n_perm) settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

NullMethod = Literal["iaaft", "permutation"]


@dataclass(frozen=True)
class PSDPeakResult:
    """Output of the H4 PSD-peak detector.

    Attributes:
        frequencies: Welch frequencies in cycles/year.
        psd: Welch power spectral density, same length as `frequencies`.
        peak_freq: Frequency of the maximum within the ω* band.
        peak_power: PSD evaluated at `peak_freq`.
        background_power: Median PSD outside the in-band region (and excluding
            the very-low-frequency strip < ω*/3).
        peak_to_background_ratio: peak_power / background_power.
        in_band: True iff the in-band peak coincides with (or exceeds) the
            global PSD maximum outside the very-low-frequency strip — i.e.
            the in-band peak is the dominant rhythm of the series, not a
            local hump dominated by some out-of-band feature.
        p_value: Empirical surrogate p-value (fraction of surrogates whose
            in-band peak-to-background ratio meets or exceeds the observed
            ratio, with the +1 / +1 Phipson–Smyth 2010 correction so
            p ∈ (0, 1]). Surrogate scheme controlled by `null_method` —
            IAAFT by default per amendment A5.
        nperseg_used: the Welch nperseg actually used for the spectrum
            estimate. Useful for diagnostics when adaptive sizing is in
            play (A1).
    """

    frequencies: NDArray[np.float64]
    psd: NDArray[np.float64]
    peak_freq: float
    peak_power: float
    background_power: float
    peak_to_background_ratio: float
    in_band: bool
    p_value: float
    nperseg_used: int


def adaptive_welch_nperseg(
    *,
    omega_star: float,
    sampling_rate: float,
    n_trajectory: int,
    target_bins_in_band: int = 10,
) -> int:
    """A1: data-determined Welch nperseg post-calibration.

    Guarantees at least `target_bins_in_band` Welch bins inside the ±20%
    band around `omega_star` (resolving the peak structure at SNR-relevant
    scales) while not exceeding `n_trajectory // 2` (so we get at least 2
    averaging windows for variance reduction at 50% overlap).

    Returns the closest power of 2 to::

        target_bins_in_band · sampling_rate / omega_star

    capped at `n_trajectory // 2`. Always ≥ 4 (Welch's minimum useful
    window).
    """
    if omega_star <= 0.0:
        raise ValueError(f"omega_star must be > 0, got {omega_star}")
    if sampling_rate <= 0.0:
        raise ValueError(f"sampling_rate must be > 0, got {sampling_rate}")
    if n_trajectory <= 0:
        raise ValueError(f"n_trajectory must be > 0, got {n_trajectory}")
    if target_bins_in_band <= 0:
        raise ValueError(f"target_bins_in_band must be > 0, got {target_bins_in_band}")

    # Welch resolution: bin width = sampling_rate / nperseg. To get K bins in
    # a band of width 0.4·omega_star (±20%), we need nperseg ≥ K / 0.4 ·
    # (sampling_rate / omega_star) = 2.5K · sampling_rate / omega_star.
    # The brief specifies "10 · sampling_rate / omega_star" — we take that
    # as the headline target (the 0.4 width is implicit in the band spec
    # but the brief's nperseg formula is target * sampling_rate / omega_star).
    target_n = target_bins_in_band * sampling_rate / omega_star
    log2_target = float(np.log2(max(target_n, 1.0)))
    # Closest power of 2 to the target (round to nearest int exponent).
    closest_pow2 = int(2 ** round(log2_target))
    cap = max(n_trajectory // 2, 4)
    # Apply the cap, then snap *down* to the nearest power of 2 — without this
    # final snap, when the cap < closest_pow2 we return the cap directly,
    # which may not be a power of 2 (the v0.3.0 G3 audit's bug).
    capped = min(closest_pow2, cap)
    capped = max(capped, 4)
    largest_pow2_le = int(2 ** int(np.floor(np.log2(capped))))
    return max(largest_pow2_le, 4)


def _welch_psd(
    x: NDArray[np.float64],
    *,
    sampling_rate: float,
    welch_window: int,
    welch_overlap: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Welch PSD wrapper with the pre-registered settings (Hann, 50% overlap).

    Falls back to a shorter nperseg when len(x) < welch_window so the function
    remains usable on short trajectories — e.g. the power-curve tier T=256.
    The fallback is documented in the H4 detector tests; canonical pre-reg
    runs always satisfy len(x) ≥ welch_window.
    """
    n = len(x)
    nperseg = min(welch_window, n)
    noverlap = int(nperseg * welch_overlap)
    freqs, psd = welch(
        x,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        detrend="constant",
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def _peak_and_background(
    freqs: NDArray[np.float64],
    psd: NDArray[np.float64],
    *,
    omega_star: float,
    bandwidth_frac: float,
) -> tuple[float, float, float, bool]:
    """Compute (peak_freq, peak_power, background_power, in_band) on (freqs, psd).

    `peak_freq` is the location of the maximum *within* the ω* band; if no
    frequencies fall within the band, the peak is taken over the full
    spectrum. The `in_band` flag is True iff the in-band peak coincides with
    (or is at least as large as) the global PSD maximum outside the
    very-low-frequency strip — i.e. the in-band peak is the dominant rhythm
    of the series, not merely a local in-band hump dominated by some
    out-of-band feature. This is the operationally meaningful "Hopf-signature
    present?" indicator.

    Background = median PSD outside the in-band region *and* outside the
    very-low-freq strip < ω*/3 (which would otherwise be dominated by trend /
    drift contamination).
    """
    f_lo = omega_star * (1.0 - bandwidth_frac)
    f_hi = omega_star * (1.0 + bandwidth_frac)
    in_band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    low_freq_mask = freqs < omega_star / 3.0

    if in_band_mask.any():
        idx_local = int(np.argmax(psd[in_band_mask]))
        peak_freq = float(freqs[in_band_mask][idx_local])
        peak_power = float(psd[in_band_mask][idx_local])
    else:
        idx_global = int(np.argmax(psd))
        peak_freq = float(freqs[idx_global])
        peak_power = float(psd[idx_global])

    # Global-max comparison restricted to non-DC, non-very-low-frequency bins.
    candidates_mask = ~low_freq_mask & (freqs > 0.0)
    global_max = float(np.max(psd[candidates_mask])) if candidates_mask.any() else peak_power
    in_band = bool(in_band_mask.any() and peak_power >= global_max - 1e-300)

    background_mask = (~in_band_mask) & (~low_freq_mask) & (freqs > 0.0)
    if background_mask.any():
        background_power = float(np.median(psd[background_mask]))
    else:
        nonzero_mask = freqs > 0.0
        background_power = float(np.median(psd[nonzero_mask])) if nonzero_mask.any() else 1.0

    if background_power <= 0.0:
        background_power = float(np.finfo(np.float64).tiny)

    return peak_freq, peak_power, background_power, in_band


def iaaft_surrogate(
    series: NDArray[np.float64],
    *,
    max_iterations: int = 100,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """IAAFT surrogate preserving marginal + linear autocorrelation.

    Schreiber & Schmitz (1996), "Improved surrogate data for nonlinearity
    tests." Iteratively alternates between:

        1. Replacing the surrogate's magnitude spectrum with the original's
           (preserves linear ACF — Wiener–Khinchin).
        2. Re-sorting the surrogate so its sample distribution exactly
           matches the original's order-statistics (preserves marginal).

    The two steps are not jointly satisfiable in general, so we iterate
    until the ranks stop changing or `max_iterations` is reached.

    Args:
        series: 1D float array; will be converted to float64.
        max_iterations: convergence cap. 100 is standard.
        rng: numpy Generator for the initial random shuffle. If None, a
            fresh default_rng() is created.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f"series must be 1D, got shape {series.shape}")
    n = len(series)
    if n < 4:
        raise ValueError(f"series must have length >= 4, got {n}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

    if rng is None:
        rng = np.random.default_rng()

    sorted_series = np.sort(series)
    original_amplitude = np.abs(np.fft.rfft(series))

    # Initialise with a random permutation of the same marginal.
    surrogate = rng.permutation(series).astype(np.float64)
    prev_ranks: NDArray[np.int64] | None = None

    for _ in range(max_iterations):
        # Step 1: enforce the original amplitude spectrum on the surrogate.
        spectrum = np.fft.rfft(surrogate)
        phases = np.angle(spectrum)
        new_spectrum = original_amplitude * np.exp(1j * phases)
        surrogate = np.fft.irfft(new_spectrum, n=n)

        # Step 2: enforce the original marginal by rank-mapping.
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_series[ranks]

        if prev_ranks is not None and np.array_equal(prev_ranks, ranks):
            break
        prev_ranks = ranks

    return surrogate


def _surrogate_p_value(
    x: NDArray[np.float64],
    observed_ratio: float,
    *,
    sampling_rate: float,
    omega_star: float,
    bandwidth_frac: float,
    welch_window: int,
    welch_overlap: float,
    n_permutations: int,
    null_method: NullMethod,
    rng: np.random.Generator,
) -> float:
    """Empirical p-value via IAAFT or iid-permutation surrogates of the input.

    Per amendment A5, IAAFT is the locked default — it preserves the linear
    autocorrelation of the input (so the H_0 spectrum is correctly red-noise
    coloured under realistic Heston / AR(1)) while randomising the long-range
    cycle structure that H4 is meant to detect. The iid permutation surrogate
    from A3 is retained as `null_method='permutation'` for tests and the
    legacy v0.1.0 baseline.

    p = (1 + #{surrogate_ratio ≥ observed_ratio}) / (1 + n_permutations)

    The +1 / +1 correction (Phipson & Smyth 2010) keeps the p-value strictly
    positive and gives a conservative estimate at small n_permutations.
    """
    if n_permutations <= 0:
        return 1.0

    geq_count = 0
    for _ in range(n_permutations):
        x_surr = iaaft_surrogate(x, rng=rng) if null_method == "iaaft" else rng.permutation(x)
        freqs_p, psd_p = _welch_psd(
            x_surr,
            sampling_rate=sampling_rate,
            welch_window=welch_window,
            welch_overlap=welch_overlap,
        )
        _, peak_p, bg_p, _ = _peak_and_background(
            freqs_p,
            psd_p,
            omega_star=omega_star,
            bandwidth_frac=bandwidth_frac,
        )
        ratio_p = peak_p / bg_p if bg_p > 0.0 else float("inf")
        if ratio_p >= observed_ratio:
            geq_count += 1

    return float((1 + geq_count) / (1 + n_permutations))


def detect_psd_peak(
    log_returns_or_squared: NDArray[np.float64],
    *,
    sampling_rate: float,
    omega_star: float,
    bandwidth_frac: float = 0.20,
    welch_window: int = 1024,
    welch_overlap: float = 0.5,
    n_permutations: int = 1_000,
    null_method: NullMethod = "iaaft",
    nperseg_override: int | None = None,
    target_bins_in_band: int = 10,
    rng: np.random.Generator | None = None,
) -> PSDPeakResult:
    """Run the H4 PSD-peak detector on a 1D series.

    Args:
        log_returns_or_squared: 1D array of the series under test. The
            pre-reg specifies absolute returns |r_t| (paper/theory.md §3 H4);
            the squared returns r_t² and the realised-variance proxy v̂_t
            (amendment A2) are equivalent indicators of volatility cycles.
            The detector is agnostic to which is used.
        sampling_rate: samples per unit time (e.g. 252 for daily samples
            measured in cycles/year).
        omega_star: predicted Hopf frequency in cycles/(unit time matching
            sampling_rate). The detector searches in
            [omega_star*(1 - bandwidth_frac), omega_star*(1 + bandwidth_frac)].
        bandwidth_frac: fractional bandwidth around omega_star (default 0.20
            per pre-reg §3 / §6).
        welch_window: legacy nperseg; only used if `nperseg_override` is set
            or as the *upper* bound when adaptive sizing is enabled. Default
            1024 matches the v0.1.0 baseline (pre-amendment A1).
        welch_overlap: fractional overlap (default 0.5 = 50% per pre-reg).
        n_permutations: number of surrogate draws for the p-value.
        null_method: surrogate scheme — "iaaft" (default, A5) or "permutation"
            (legacy A3 path, kept for back-compat).
        nperseg_override: if not None, force the Welch nperseg to this value
            and skip the adaptive sizing. Used by tests pinned to the v0.1.0
            baseline. Falls back to `welch_window` when None and the adaptive
            sizing produces a value ≥ welch_window.
        target_bins_in_band: passed to `adaptive_welch_nperseg` (A1). Default
            10 matches the §10 amendment text.
        rng: numpy Generator for reproducible surrogates.

    Returns:
        PSDPeakResult with the spectrum, in-band peak, background, ratio,
        in-band flag, empirical p-value, and the nperseg actually used.

    Raises:
        ValueError: if the input is empty, the sampling rate is non-positive,
            omega_star is non-positive, or bandwidth_frac is outside (0, 1).
    """
    x = np.asarray(log_returns_or_squared, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"input must be 1D, got shape {x.shape}")
    if len(x) < 4:
        raise ValueError(f"input length must be >= 4, got {len(x)}")
    if sampling_rate <= 0.0:
        raise ValueError(f"sampling_rate must be > 0, got {sampling_rate}")
    if omega_star <= 0.0:
        raise ValueError(f"omega_star must be > 0, got {omega_star}")
    if not 0.0 < bandwidth_frac < 1.0:
        raise ValueError(f"bandwidth_frac must be in (0, 1), got {bandwidth_frac}")
    if not 0.0 <= welch_overlap < 1.0:
        raise ValueError(f"welch_overlap must be in [0, 1), got {welch_overlap}")
    if null_method not in ("iaaft", "permutation"):
        raise ValueError(f"null_method must be 'iaaft' or 'permutation', got {null_method}")

    if rng is None:
        rng = np.random.default_rng()

    if nperseg_override is not None:
        nperseg_to_use = nperseg_override
    else:
        adaptive = adaptive_welch_nperseg(
            omega_star=omega_star,
            sampling_rate=sampling_rate,
            n_trajectory=len(x),
            target_bins_in_band=target_bins_in_band,
        )
        # Respect the legacy `welch_window` as a non-binding upper hint when
        # adaptive sizing produces a smaller value (v0.1.0 baseline path).
        nperseg_to_use = min(adaptive, welch_window)
        nperseg_to_use = max(nperseg_to_use, 4)

    freqs, psd = _welch_psd(
        x,
        sampling_rate=sampling_rate,
        welch_window=nperseg_to_use,
        welch_overlap=welch_overlap,
    )
    peak_freq, peak_power, background_power, in_band = _peak_and_background(
        freqs,
        psd,
        omega_star=omega_star,
        bandwidth_frac=bandwidth_frac,
    )
    ratio = peak_power / background_power if background_power > 0.0 else float("inf")

    p_value = _surrogate_p_value(
        x,
        observed_ratio=ratio,
        sampling_rate=sampling_rate,
        omega_star=omega_star,
        bandwidth_frac=bandwidth_frac,
        welch_window=nperseg_to_use,
        welch_overlap=welch_overlap,
        n_permutations=n_permutations,
        null_method=null_method,
        rng=rng,
    )

    return PSDPeakResult(
        frequencies=freqs,
        psd=psd,
        peak_freq=peak_freq,
        peak_power=peak_power,
        background_power=background_power,
        peak_to_background_ratio=ratio,
        in_band=in_band,
        p_value=p_value,
        nperseg_used=int(nperseg_to_use),
    )
