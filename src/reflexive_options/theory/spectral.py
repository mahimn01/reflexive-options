"""H4 PSD-peak detector — pre-registered in paper/pre_registration.md §3 (H4) / §6.

Operational specification per the pre-registration:
    - Welch's method PSD with 1024-day Hann window, 50% overlap.
    - Search for the maximum within ω* ± 20% of the predicted Hopf frequency
      ω* = √c_1(κ*) (Theorem 1, paper/theory.md §3).
    - Background = median PSD outside the in-band region (and excluding the
      very-low-frequency strip ω < ω*/3 to avoid trend / drift contamination).
    - Empirical p-value via index-permutation surrogates: reshuffle the input
      uniformly at random (preserves marginal but breaks frequency content
      — the pre-reg names this a "circular shuffle" but a literal circular
      shift would not break the magnitude spectrum, so we permute instead).
      Recompute the in-band peak-to-background ratio under each shuffled
      spectrum, count the fraction of shuffles where the shuffled ratio is
      ≥ the observed ratio.

The detector returns a PSDPeakResult dataclass with the band-restricted peak
location, the peak power, the off-band background, the ratio, an in-band
indicator, and the empirical p-value.

Validated end-to-end against synthetic ground truth in tests/test_h4_detector.py
and the experiments/h4_validation.py runner — see runs/h4_validation/ for the
power curves at the canonical (window, overlap, bandwidth, n_perm) settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch


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
        p_value: Permutation-surrogate p-value (fraction of random shuffles
            of the input whose in-band peak-to-background ratio meets or
            exceeds the observed ratio, with the +1/+1 Phipson-Smyth 2010
            correction so p ∈ (0, 1]).
    """

    frequencies: NDArray[np.float64]
    psd: NDArray[np.float64]
    peak_freq: float
    peak_power: float
    background_power: float
    peak_to_background_ratio: float
    in_band: bool
    p_value: float


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
    runs always satisfy len(x) ≥ 1024.
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
    # Rationale: the Hopf signature is a peak at ω*; competing trend / drift
    # bins dominate the very low end of the spectrum without contradicting
    # the H4 claim, so they shouldn't disqualify the in-band peak. An
    # out-of-band high-frequency peak that beats the in-band peak DOES
    # disqualify it — that's what `in_band` is reporting.
    candidates_mask = ~low_freq_mask & (freqs > 0.0)
    global_max = float(np.max(psd[candidates_mask])) if candidates_mask.any() else peak_power
    in_band = bool(in_band_mask.any() and peak_power >= global_max - 1e-300)

    background_mask = (~in_band_mask) & (~low_freq_mask) & (freqs > 0.0)
    if background_mask.any():
        background_power = float(np.median(psd[background_mask]))
    else:
        # degenerate fallback: use the full-spectrum median sans DC
        nonzero_mask = freqs > 0.0
        background_power = float(np.median(psd[nonzero_mask])) if nonzero_mask.any() else 1.0

    if background_power <= 0.0:
        # numerical floor — keeps the ratio finite when the background is
        # exactly zero (e.g. very short single-sinusoid inputs)
        background_power = float(np.finfo(np.float64).tiny)

    return peak_freq, peak_power, background_power, in_band


def _shuffle_p_value(
    x: NDArray[np.float64],
    observed_ratio: float,
    *,
    sampling_rate: float,
    omega_star: float,
    bandwidth_frac: float,
    welch_window: int,
    welch_overlap: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Empirical p-value via random index permutation of the input.

    The pre-reg names this a "circular shuffle"; we implement it as a full
    random permutation rather than a literal np.roll because a true circular
    shift leaves the magnitude spectrum invariant (only phases rotate), so it
    cannot serve as a null for in-band peak height. A random permutation
    *does* preserve the marginal distribution and *does* break all temporal
    structure including the frequency content — which matches the pre-reg's
    stated intent ("preserves marginal but breaks frequency content").
    Under H_0 the resulting series is exchangeable, and its expected PSD is
    flat (white-noise-like), so the in-band peak-to-background ratio
    concentrates near 1 with a well-defined right tail.

    p = (1 + #{shuffled_ratio ≥ observed_ratio}) / (1 + n_permutations)

    The +1 / +1 correction (Phipson & Smyth 2010) keeps the p-value strictly
    positive and gives a conservative estimate at small n_permutations.
    """
    if n_permutations <= 0:
        return 1.0

    geq_count = 0
    for _ in range(n_permutations):
        x_shuffled = rng.permutation(x)
        freqs_p, psd_p = _welch_psd(
            x_shuffled,
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
    rng: np.random.Generator | None = None,
) -> PSDPeakResult:
    """Run the H4 PSD-peak detector on a 1D series.

    Args:
        log_returns_or_squared: 1D array of the series under test. The
            pre-reg specifies absolute returns |r_t| (paper/theory.md §3 H4);
            the squared returns r_t² are an equivalent indicator of
            volatility cycles (Marketron §5.2.3, Halperin–Itkin 2025) and can
            be passed instead. The detector is agnostic to which is used.
        sampling_rate: samples per unit time (e.g. 252 for daily samples
            measured in cycles/year).
        omega_star: predicted Hopf frequency in cycles/(unit time matching
            sampling_rate). The detector searches in
            [omega_star*(1 - bandwidth_frac), omega_star*(1 + bandwidth_frac)].
        bandwidth_frac: fractional bandwidth around omega_star (default 0.20
            per pre-reg §3 / §6).
        welch_window: nperseg for Welch's method (default 1024 per pre-reg).
        welch_overlap: fractional overlap (default 0.5 = 50% per pre-reg).
        n_permutations: number of permutation surrogates for the p-value.
        rng: numpy Generator for reproducible permutations.

    Returns:
        PSDPeakResult with the spectrum, in-band peak, background, ratio,
        in-band flag, and empirical p-value.

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

    if rng is None:
        rng = np.random.default_rng()

    freqs, psd = _welch_psd(
        x,
        sampling_rate=sampling_rate,
        welch_window=welch_window,
        welch_overlap=welch_overlap,
    )
    peak_freq, peak_power, background_power, in_band = _peak_and_background(
        freqs,
        psd,
        omega_star=omega_star,
        bandwidth_frac=bandwidth_frac,
    )
    ratio = peak_power / background_power if background_power > 0.0 else float("inf")

    p_value = _shuffle_p_value(
        x,
        observed_ratio=ratio,
        sampling_rate=sampling_rate,
        omega_star=omega_star,
        bandwidth_frac=bandwidth_frac,
        welch_window=welch_window,
        welch_overlap=welch_overlap,
        n_permutations=n_permutations,
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
    )
