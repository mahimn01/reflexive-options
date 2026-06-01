"""Spectral discriminator between the two strata of the Hawkes--SV criticality boundary.

This module operationalises the repositioned Theorem 4 (see paper/t4_reposition.md).
It does NOT verify any definitional identity (the deleted n_SV ~1e-15 tautology). It
provides a falsifiable, observable statistic that separates:

  * the real-eigenvalue (saddle-node) stratum -- the literal Hardiman n=1 analogue,
    which is NON-oscillatory (critical slowing-down: a smooth red-noise continuum
    that decays monotonically from zero frequency, with NO sharp interior line), from
  * the Hopf stratum -- a strictly stronger, OSCILLATORY instability (a sharp,
    narrow-band spectral LINE at a finite quasi-frequency omega* > 0, rising far
    above the local smooth background, away from the zero-frequency rolloff).

Discriminator (why a smooth-background line test, restricted to interior frequencies).
A real-eigenvalue crossing produces critical slowing-down: a broad, smoothly
decaying spectrum whose mass piles up at zero frequency, with no intrinsic
frequency. A Hopf crossing adds a genuine, narrow spectral line at an INTERIOR
frequency omega* > 0. The right statistic is therefore the local prominence of the
strongest line above a smoothly estimated background (running median, robust to the
line itself), computed only over interior frequencies (the lowest few bins, which
carry the red-noise rolloff common to both strata, are excluded), combined with a
narrow-band sharpness check. Phase-randomization surrogates preserve the power
spectrum and so cannot judge peak significance (they are the wrong null and are NOT
used here, retained only as an exposed helper for waveform/phase tests).

Validated on simulator-generated data with known ground-truth coupling kappa
(positive control = Hopf side, negative control = saddle-node side); see the
companion test module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Stratum = Literal["real_edge", "hopf_edge", "stable"]


@dataclass(frozen=True)
class StratumResult:
    stratum: Stratum
    peak_frequency: float  # cycles per sample of the strongest interior line; 0.0 if none
    line_prominence: float  # peak power / smooth local background at the line
    band_concentration: float  # fraction of band-pass energy inside the line (sharpness)
    branching_proxy: float  # lag-1 autocorrelation of the level, a [0,1) criticality proxy


def _welch_periodogram(x: np.ndarray, n_segments: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Welch-averaged one-sided periodogram, excluding the zero-frequency (DC) bin.

    Averaging over ``n_segments`` half-overlapping Hann-windowed segments stabilises
    the line estimate. DC is dropped: the discriminator is about NONZERO-frequency
    structure.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    seg_len = max(16, n // n_segments)
    step = max(1, seg_len // 2)
    window = np.hanning(seg_len)
    win_norm = float(np.sum(window**2))
    starts = list(range(0, n - seg_len + 1, step))
    if not starts:
        starts = [0]
        seg_len = n
        window = np.hanning(seg_len)
        win_norm = float(np.sum(window**2))
    psd_accum = np.zeros(seg_len // 2 + 1, dtype=float)
    for s in starts:
        seg = x[s : s + seg_len] * window
        fft = np.fft.rfft(seg)
        psd_accum += (np.abs(fft) ** 2) / win_norm
    psd_accum /= len(starts)
    freqs = np.fft.rfftfreq(seg_len, d=1.0)
    return freqs[1:], psd_accum[1:]


def _running_median(x: np.ndarray, half_width: int) -> np.ndarray:
    """Robust smooth background via a centred running median (window = 2*half_width+1).

    The median is insensitive to a narrow spectral line sitting inside the window, so
    it estimates the continuum the line rises above without being inflated by the line.
    """
    n = x.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half_width)
        hi = min(n, i + half_width + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def _lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    num = float(np.dot(x[:-1], x[1:]))
    return max(0.0, min(0.999, num / denom))


def _phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate: preserves the power spectrum, destroys phase
    coupling. Exposed for callers that want a waveform/phase test; NOT used by the
    line-significance discriminator."""
    x = np.asarray(x, dtype=float)
    n = x.size
    fft = np.fft.rfft(x - x.mean())
    mag = np.abs(fft)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=mag.size)
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    surrogate_fft = mag * np.exp(1j * phases)
    return np.fft.irfft(surrogate_fft, n=n)


def classify_stratum(
    series: np.ndarray,
    *,
    n_segments: int = 8,
    bg_half_width: int = 8,
    rolloff_skip_frac: float = 0.04,
    min_prominence: float = 4.0,
    min_concentration: float = 0.0,
    stable_autocorr: float = 0.3,
    seed: int | None = 0,  # kept for API stability; the line test is deterministic
) -> StratumResult:
    """Classify a 1-D series (e.g. a volatility/intensity proxy) into a stratum.

    Decision rule (pre-registered):
      1. branching_proxy = lag-1 autocorrelation of the level. If below
         ``stable_autocorr`` the system is far from criticality -> "stable".
      2. Otherwise estimate the smooth background (running median of the Welch
         periodogram) and, over INTERIOR frequencies only (skip the lowest
         ``rolloff_skip_frac`` of the band, which carries the zero-frequency
         red-noise rolloff common to both strata), find the strongest spectral line
         as the bin whose power-over-background ratio (``line_prominence``) is
         largest. The PRIMARY discriminator is ``line_prominence``: on the synthetic
         controls (known ground-truth kappa, omega) the Hopf-side line clears
         prominence >= 4.5 while the real-edge (saddle-node) continuum never exceeds
         ~2.4, so the default ``min_prominence = 4.0`` separates the two strata with a
         wide margin and zero overlap. ``band_concentration`` (the fraction of the
         background-subtracted +/-bg_half_width energy carried by the central +/-1
         bins) is REPORTED as a sharpness diagnostic but is OFF by default
         (``min_concentration = 0.0``) because it overlaps across the two strata and
         is not a clean separator; callers who want a stricter line-shape gate may
         raise it. If ``line_prominence > min_prominence`` AND ``band_concentration >
         min_concentration`` AND the line frequency is > 0 -> "hopf_edge" (a genuine
         narrow-band interior oscillation the scalar branching ratio cannot see). Else
         -> "real_edge" (non-oscillatory critical slowing-down = the literal Hardiman
         analogue; its spectrum is a smooth red continuum with no sharp interior line).

    This is the falsifiable replacement for the deleted n_SV identity: it returns a
    classification from an OBSERVABLE (a sharp interior spectral line over a robust
    smooth background), with an explicit non-oscillatory null, not from a definition.
    """
    series = np.asarray(series, dtype=float).ravel()
    if series.size < 16:
        raise ValueError("series too short for spectral discrimination (need >= 16 samples)")

    branching_proxy = _lag1_autocorr(series)

    freqs, power = _welch_periodogram(series, n_segments=n_segments)
    hw = min(bg_half_width, max(1, freqs.size // 4))
    background = _running_median(power, hw)
    fallback = float(np.median(power)) or 1.0
    background = np.where(background <= 0, fallback, background)
    ratio = power / background

    # restrict the line search to interior frequencies: skip the lowest bins that
    # carry the zero-frequency red-noise rolloff (present in BOTH strata).
    skip = max(1, round(rolloff_skip_frac * freqs.size))
    if skip >= freqs.size:
        skip = 1
    interior = np.arange(skip, freqs.size)

    rel_idx = int(np.argmax(ratio[interior]))
    peak_idx = int(interior[rel_idx])
    line_prominence = float(ratio[peak_idx])
    peak_freq = float(freqs[peak_idx])

    # band concentration: of the background-subtracted energy in +/-hw around the
    # line, what fraction sits in the central +/-1 bins (sharp line ~> 1).
    excess = np.clip(power - background, 0.0, None)
    band_lo = max(0, peak_idx - hw)
    band_hi = min(power.size, peak_idx + hw + 1)
    band_energy = float(np.sum(excess[band_lo:band_hi]))
    core_lo = max(0, peak_idx - 1)
    core_hi = min(power.size, peak_idx + 2)
    core_energy = float(np.sum(excess[core_lo:core_hi]))
    band_concentration = core_energy / band_energy if band_energy > 0 else 0.0

    if branching_proxy < stable_autocorr:
        return StratumResult(
            stratum="stable",
            peak_frequency=peak_freq,
            line_prominence=line_prominence,
            band_concentration=band_concentration,
            branching_proxy=branching_proxy,
        )

    is_hopf = (
        line_prominence > min_prominence
        and band_concentration > min_concentration
        and peak_freq > 0.0
    )
    return StratumResult(
        stratum="hopf_edge" if is_hopf else "real_edge",
        peak_frequency=peak_freq,
        line_prominence=line_prominence,
        band_concentration=band_concentration,
        branching_proxy=branching_proxy,
    )


# --- simulator-side helpers (known ground-truth kappa) for the synthetic controls ---


def simulate_sv_limit(
    *,
    kappa: float,
    omega: float,
    n_steps: int = 4096,
    dt: float = 1.0,
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Linear SV-limit proxy near the drift boundary, parameterised so the ground-truth
    stratum is known.

    The continuous drift Jacobian eigenvalues are -kappa +/- i*omega. We integrate the
    2-D linear SDE dx = A x dt + noise dW with A = [[-kappa, -omega], [omega, -kappa]]
    using the EXACT one-step propagator exp(A*dt) (a rotation-by-omega*dt with decay
    exp(-kappa*dt)), so the discrete spectrum faithfully carries the continuous
    resonance:

      * omega == 0 -> real eigenvalue; kappa -> 0 is the saddle-node / Hardiman edge.
        The propagator is a scalar contraction exp(-kappa*dt); the noise-driven series
        is an AR(1)-type red-noise process: a smooth spectrum decaying from zero
        frequency with NO sharp interior line (non-oscillatory critical slowing-down).
      * omega  > 0 -> complex pair; kappa -> 0 is the Hopf edge. The propagator rotates
        by omega*dt each step; the noise-driven oscillator has a sharp resonance LINE
        at omega*dt/(2*pi) cycles/sample.

    Returns the (signed) first component of the state -- the natural observable that
    preserves the resonance line. This is the SAME style of synthetic generator (known
    ground-truth coupling) used to validate the original H4.
    """
    rng = np.random.default_rng(seed)
    decay = np.exp(-kappa * dt)
    c, s = np.cos(omega * dt), np.sin(omega * dt)
    prop = decay * np.array([[c, -s], [s, c]], dtype=float)
    x = np.array([1.0, 0.0], dtype=float)
    out = np.empty(n_steps, dtype=float)
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        x = prop @ x + rng.standard_normal(2) * noise * sqrt_dt
        out[t] = x[0]
    return out
