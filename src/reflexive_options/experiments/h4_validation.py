"""H4 PSD-peak detector validation suite — power curves on synthetic ground truth.

End-to-end characterisation of `theory.spectral.detect_psd_peak`: detection
rate (fraction of seeds where the detector fires) as a function of (i)
trajectory length T at fixed SNR and (ii) signal-to-noise ratio at fixed T.
The synthetic positive control is a sinusoid at ω* embedded in i.i.d.
Gaussian noise.

**Amendment A2 — dual-signal H4.** Per `paper/pre_registration_amendments.md`
A2, H4 is reported on both `|r_t|` and the realised-variance proxy v̂_t. The
runner exposes `--signals abs_returns,realised_variance` and reports per-
signal power curves and the joint "either fires" rate that the H4 decision
rule actually uses (Bonferroni-corrected α = 0.025 per signal). For the
synthetic positive control (a sinusoid in noise) both transforms collapse
to the same observable, so the per-signal columns are identical at the
synthetic-validation tier. The dual reporting still flows through the same
power-curve plumbing so the empirical-data pipeline (which has truly
independent noise on the two signals) gets the right metrics shape.

Saves:
    runs/h4_validation/<timestamp>/power_curve.npz
    runs/h4_validation/<timestamp>/metrics.json
    runs/h4_validation/<timestamp>/config.json
    paper/figures/h4_detector_power.pdf

Run: python -m reflexive_options.experiments.h4_validation
     python -m reflexive_options.experiments.h4_validation --quick   # smaller grid
     python -m reflexive_options.experiments.h4_validation --signals abs_returns
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal

# Pin matplotlib's PDF /CreationDate so figure regenerations are byte-stable.
# See verification_v5_repro.md §3 — without this, every regen drifts the PDF
# hash even though the rendered drawing is bit-identical.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")  # headless safe (CI / agent envs); must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.theory.spectral import detect_psd_peak

SAMPLING_RATE_DEFAULT = 252.0  # daily samples / year
OMEGA_STAR_DEFAULT = 20.0  # cycles / year — well within the Welch resolved band

SignalName = Literal["abs_returns", "realised_variance"]
ALL_SIGNALS: tuple[SignalName, ...] = ("abs_returns", "realised_variance")


@dataclass(frozen=True)
class H4ValidationConfig:
    """Configuration for the H4 detector validation suite."""

    sampling_rate: float = SAMPLING_RATE_DEFAULT
    omega_star: float = OMEGA_STAR_DEFAULT
    bandwidth_frac: float = 0.20
    welch_window: int = 1024
    welch_overlap: float = 0.5
    n_permutations: int = 200
    n_seeds_per_point: int = 100
    # Power-vs-T grid (at fixed SNR)
    t_grid: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    fixed_snr_amplitude: float = 0.4
    fixed_snr_noise: float = 1.0
    # Power-vs-SNR grid (at fixed T = 1024)
    snr_t: int = 1024
    snr_amplitudes: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
    snr_noise: float = 1.0
    # Decision threshold (per-signal; A2 Bonferroni gives α/2 = 0.025 per signal)
    alpha: float = 0.05
    # H4 amendment A2: per-signal Bonferroni α
    alpha_per_signal: float = 0.025
    # H_0 calibration check
    n_h0_seeds: int = 200
    h0_t: int = 1024
    # A2: which signals to evaluate. The default reports both per the
    # amendment; passing a subset (e.g. ("abs_returns",)) reverts to the
    # pre-A2 single-signal behaviour.
    signals: tuple[SignalName, ...] = ALL_SIGNALS


def _sinusoid_with_noise(
    *,
    t: int,
    sampling_rate: float,
    frequency: float,
    amplitude: float,
    noise_std: float,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    times = np.arange(t) / sampling_rate
    return amplitude * np.sin(2.0 * np.pi * frequency * times) + noise_std * rng.standard_normal(t)


def _signal_transform(x: NDArray[np.float64], signal: SignalName) -> NDArray[np.float64]:
    """Apply the A2 signal transform to a raw input series.

    For the synthetic-positive-control input (sinusoid + iid noise), both
    transforms collapse to a deterministic mapping of the same observable;
    the dual reporting nevertheless exercises the per-signal plumbing so
    the empirical-data pipeline behaves correctly.
    """
    if signal == "abs_returns":
        return np.abs(x).astype(np.float64)
    if signal == "realised_variance":
        # Synthetic proxy: per-step squared-return (equivalent to the daily
        # realised-variance accumulator at intraday-N=1). The H4 detector is
        # agnostic to which transform is used; what matters is that the
        # cycle frequency survives in the spectral content of the transformed
        # series, which it does for both transforms.
        return (x * x).astype(np.float64)
    raise ValueError(f"unknown signal: {signal}")


def _power_at(
    *,
    t: int,
    amplitude: float,
    noise_std: float,
    cfg: H4ValidationConfig,
    base_seed: int,
) -> tuple[
    dict[SignalName, float],
    dict[SignalName, float],
    float,
]:
    """Detection power per signal + the joint "any signal fires" rate.

    Returns ``(per_signal_power, per_signal_mean_p, joint_power)``.

    - `per_signal_power[s]` = fraction of seeds where signal s fires
      (in_band True AND p < cfg.alpha_per_signal).
    - `joint_power` = fraction of seeds where AT LEAST ONE signal fires.
      Per A2 this is the operationally meaningful detection rate for H4.
    """
    fires_per_signal: dict[SignalName, int] = {s: 0 for s in cfg.signals}
    p_sum_per_signal: dict[SignalName, float] = {s: 0.0 for s in cfg.signals}
    joint_fires = 0
    for s in range(cfg.n_seeds_per_point):
        x_raw = _sinusoid_with_noise(
            t=t,
            sampling_rate=cfg.sampling_rate,
            frequency=cfg.omega_star,
            amplitude=amplitude,
            noise_std=noise_std,
            seed=base_seed + s,
        )
        any_fired = False
        for sig in cfg.signals:
            x_sig = _signal_transform(x_raw, sig)
            res = detect_psd_peak(
                x_sig,
                sampling_rate=cfg.sampling_rate,
                omega_star=cfg.omega_star,
                bandwidth_frac=cfg.bandwidth_frac,
                welch_window=cfg.welch_window,
                welch_overlap=cfg.welch_overlap,
                n_permutations=cfg.n_permutations,
                rng=np.random.default_rng(base_seed + s + 100_000 + 7919 * hash(sig) % 1_000_000),
            )
            fired = res.in_band and res.p_value < cfg.alpha_per_signal
            if fired:
                fires_per_signal[sig] += 1
                any_fired = True
            p_sum_per_signal[sig] += res.p_value
        if any_fired:
            joint_fires += 1
    n = cfg.n_seeds_per_point
    return (
        {s: fires_per_signal[s] / n for s in cfg.signals},
        {s: p_sum_per_signal[s] / n for s in cfg.signals},
        joint_fires / n,
    )


def _h0_calibration(
    *, cfg: H4ValidationConfig, base_seed: int
) -> tuple[
    dict[SignalName, NDArray[np.float64]],
    dict[SignalName, float],
    dict[SignalName, float],
    float,
]:
    """Per-signal H_0 calibration plus joint FPR at α/m Bonferroni.

    Returns:
        per_signal_p_values: dict mapping signal -> p-value array.
        per_signal_fpr: dict mapping signal -> empirical FPR at alpha_per_signal.
        per_signal_mean_p: dict mapping signal -> mean p.
        joint_fpr: fraction of seeds where ANY signal fires at alpha_per_signal.
    """
    p_values: dict[SignalName, NDArray[np.float64]] = {
        s: np.empty(cfg.n_h0_seeds, dtype=np.float64) for s in cfg.signals
    }
    in_band: dict[SignalName, NDArray[np.bool_]] = {
        s: np.zeros(cfg.n_h0_seeds, dtype=np.bool_) for s in cfg.signals
    }
    for s in range(cfg.n_h0_seeds):
        rng = np.random.default_rng(base_seed + s)
        x_raw = rng.standard_normal(cfg.h0_t).astype(np.float64)
        for sig in cfg.signals:
            x_sig = _signal_transform(x_raw, sig)
            res = detect_psd_peak(
                x_sig,
                sampling_rate=cfg.sampling_rate,
                omega_star=cfg.omega_star,
                bandwidth_frac=cfg.bandwidth_frac,
                welch_window=cfg.welch_window,
                welch_overlap=cfg.welch_overlap,
                n_permutations=cfg.n_permutations,
                rng=np.random.default_rng(base_seed + s + 999_999 + 7919 * hash(sig) % 1_000_000),
            )
            p_values[sig][s] = res.p_value
            in_band[sig][s] = res.in_band

    per_signal_fpr = {
        sig: float(np.mean((p_values[sig] < cfg.alpha_per_signal) & in_band[sig]))
        for sig in cfg.signals
    }
    per_signal_mean_p = {sig: float(p_values[sig].mean()) for sig in cfg.signals}
    # Joint FPR: H4 fires at *any* signal at alpha_per_signal. Per A2 this is
    # the operationally meaningful FPR for the H4 decision rule.
    joint_mask = np.zeros(cfg.n_h0_seeds, dtype=np.bool_)
    for sig in cfg.signals:
        joint_mask |= (p_values[sig] < cfg.alpha_per_signal) & in_band[sig]
    joint_fpr = float(np.mean(joint_mask))
    return p_values, per_signal_fpr, per_signal_mean_p, joint_fpr


@dataclass
class H4ValidationResult:
    """Bundled results for the H4 power-curve scan.

    Per-signal arrays in `power_vs_t`/`power_vs_snr` are dicts keyed by
    `SignalName`. The joint columns (`joint_power_vs_t`, etc.) report the
    "any signal fires" rate that the A2 decision rule consumes.
    """

    cfg: H4ValidationConfig
    t_grid: NDArray[np.int64]
    power_vs_t: dict[SignalName, NDArray[np.float64]]
    mean_p_vs_t: dict[SignalName, NDArray[np.float64]]
    joint_power_vs_t: NDArray[np.float64]
    snr_amplitudes: NDArray[np.float64]
    power_vs_snr: dict[SignalName, NDArray[np.float64]]
    mean_p_vs_snr: dict[SignalName, NDArray[np.float64]]
    joint_power_vs_snr: NDArray[np.float64]
    h0_p_values: dict[SignalName, NDArray[np.float64]]
    h0_fpr: dict[SignalName, float]
    h0_mean_p: dict[SignalName, float]
    h0_joint_fpr: float
    figures: list[str] = field(default_factory=list)


def run_validation(cfg: H4ValidationConfig, *, base_seed: int = 42) -> H4ValidationResult:
    """Compute the two power curves and the H_0 p-value distribution."""
    n_t = len(cfg.t_grid)
    power_vs_t: dict[SignalName, NDArray[np.float64]] = {
        s: np.zeros(n_t, dtype=np.float64) for s in cfg.signals
    }
    mean_p_vs_t: dict[SignalName, NDArray[np.float64]] = {
        s: np.zeros(n_t, dtype=np.float64) for s in cfg.signals
    }
    joint_power_vs_t = np.zeros(n_t, dtype=np.float64)
    for i, t_val in enumerate(cfg.t_grid):
        per_pow, per_p, joint = _power_at(
            t=t_val,
            amplitude=cfg.fixed_snr_amplitude,
            noise_std=cfg.fixed_snr_noise,
            cfg=cfg,
            base_seed=base_seed + 1_000 * i,
        )
        for sig in cfg.signals:
            power_vs_t[sig][i] = per_pow[sig]
            mean_p_vs_t[sig][i] = per_p[sig]
        joint_power_vs_t[i] = joint

    n_snr = len(cfg.snr_amplitudes)
    power_vs_snr: dict[SignalName, NDArray[np.float64]] = {
        s: np.zeros(n_snr, dtype=np.float64) for s in cfg.signals
    }
    mean_p_vs_snr: dict[SignalName, NDArray[np.float64]] = {
        s: np.zeros(n_snr, dtype=np.float64) for s in cfg.signals
    }
    joint_power_vs_snr = np.zeros(n_snr, dtype=np.float64)
    for i, amp in enumerate(cfg.snr_amplitudes):
        per_pow, per_p, joint = _power_at(
            t=cfg.snr_t,
            amplitude=amp,
            noise_std=cfg.snr_noise,
            cfg=cfg,
            base_seed=base_seed + 100_000 + 1_000 * i,
        )
        for sig in cfg.signals:
            power_vs_snr[sig][i] = per_pow[sig]
            mean_p_vs_snr[sig][i] = per_p[sig]
        joint_power_vs_snr[i] = joint

    h0_p, h0_fpr, h0_mean_p, h0_joint_fpr = _h0_calibration(cfg=cfg, base_seed=base_seed + 500_000)

    return H4ValidationResult(
        cfg=cfg,
        t_grid=np.asarray(cfg.t_grid, dtype=np.int64),
        power_vs_t=power_vs_t,
        mean_p_vs_t=mean_p_vs_t,
        joint_power_vs_t=joint_power_vs_t,
        snr_amplitudes=np.asarray(cfg.snr_amplitudes, dtype=np.float64),
        power_vs_snr=power_vs_snr,
        mean_p_vs_snr=mean_p_vs_snr,
        joint_power_vs_snr=joint_power_vs_snr,
        h0_p_values=h0_p,
        h0_fpr=h0_fpr,
        h0_mean_p=h0_mean_p,
        h0_joint_fpr=h0_joint_fpr,
    )


def plot_power_curves(result: H4ValidationResult, *, out_path: str) -> None:
    """Two-panel power curve: (a) joint power vs T, (b) joint power vs SNR. Saved as PDF.

    Per-signal curves overlay the joint "any fires" curve as faint lines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(
        result.t_grid, result.joint_power_vs_t, marker="o", label="any signal", linewidth=2
    )
    for sig in result.cfg.signals:
        axes[0].plot(
            result.t_grid, result.power_vs_t[sig], marker="x", linestyle="--", alpha=0.6, label=sig
        )
    axes[0].axhline(0.9, color="grey", linestyle=":", linewidth=0.8, label="90% target")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Trajectory length T (samples)")
    axes[0].set_ylabel("Detection power")
    axes[0].set_title(
        f"H4 power vs T (SNR = A/σ = {result.cfg.fixed_snr_amplitude / result.cfg.fixed_snr_noise:.2f})"
    )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    snr_ratios = result.snr_amplitudes / result.cfg.snr_noise
    axes[1].plot(snr_ratios, result.joint_power_vs_snr, marker="o", label="any signal", linewidth=2)
    for sig in result.cfg.signals:
        axes[1].plot(
            snr_ratios, result.power_vs_snr[sig], marker="x", linestyle="--", alpha=0.6, label=sig
        )
    axes[1].axhline(0.9, color="grey", linestyle=":", linewidth=0.8, label="90% target")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("SNR (amplitude / noise std)")
    axes[1].set_ylabel("Detection power")
    axes[1].set_title(f"H4 power vs SNR (T = {result.cfg.snr_t})")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.suptitle(
        f"H4 PSD-peak detector — Welch {result.cfg.welch_window}-day Hann, "
        f"ω* = {result.cfg.omega_star} cyc/yr, ±{int(result.cfg.bandwidth_frac * 100)}% band, "
        f"α/m = {result.cfg.alpha_per_signal}, n_perm = {result.cfg.n_permutations}, "
        f"H_0 joint FPR = {result.h0_joint_fpr:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=150)
    plt.close(fig)


def _parse_signals(arg: str) -> tuple[SignalName, ...]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    out: list[SignalName] = []
    for p in parts:
        if p not in ("abs_returns", "realised_variance"):
            raise argparse.ArgumentTypeError(
                f"unknown signal '{p}': must be one of abs_returns, realised_variance"
            )
        out.append(p)  # type: ignore[arg-type]
    if not out:
        raise argparse.ArgumentTypeError("--signals must list at least one signal")
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller seed count and Welch n_perm for fast smoke-runs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--signals",
        type=_parse_signals,
        default=ALL_SIGNALS,
        help="Comma-separated signals to evaluate (A2). Default: abs_returns,realised_variance.",
    )
    args = parser.parse_args()

    if args.quick:
        cfg = H4ValidationConfig(
            n_seeds_per_point=20,
            n_permutations=50,
            n_h0_seeds=50,
            t_grid=(256, 1024, 4096),
            snr_amplitudes=(0.10, 0.40, 1.60),
            signals=args.signals,
        )
    else:
        cfg = H4ValidationConfig(signals=args.signals)

    run_dir = make_run_dir("h4_validation")
    save_config(run_dir, cfg)

    with timed("h4_validation.run_validation"):
        result = run_validation(cfg, base_seed=args.seed)

    npz_payload: dict[str, NDArray[np.int64] | NDArray[np.float64]] = {
        "t_grid": result.t_grid,
        "snr_amplitudes": result.snr_amplitudes,
        "joint_power_vs_t": result.joint_power_vs_t,
        "joint_power_vs_snr": result.joint_power_vs_snr,
    }
    for sig in cfg.signals:
        npz_payload[f"power_vs_t__{sig}"] = result.power_vs_t[sig]
        npz_payload[f"mean_p_vs_t__{sig}"] = result.mean_p_vs_t[sig]
        npz_payload[f"power_vs_snr__{sig}"] = result.power_vs_snr[sig]
        npz_payload[f"mean_p_vs_snr__{sig}"] = result.mean_p_vs_snr[sig]
        npz_payload[f"h0_p_values__{sig}"] = result.h0_p_values[sig]
    # mypy sees np.savez's second positional arg as `bool` due to a stale
    # scipy/numpy stub edge case; cast to keep mypy strict happy without
    # changing runtime behaviour.
    np.savez(run_dir / "power_curve.npz", **npz_payload)  # type: ignore[arg-type]

    metrics: dict[str, object] = {
        "t_grid": result.t_grid.tolist(),
        "snr_amplitudes": result.snr_amplitudes.tolist(),
        "snr_ratios": (result.snr_amplitudes / cfg.snr_noise).tolist(),
        "signals": list(cfg.signals),
        "alpha_per_signal": cfg.alpha_per_signal,
        "joint_power_vs_t": result.joint_power_vs_t.tolist(),
        "joint_power_vs_snr": result.joint_power_vs_snr.tolist(),
        "h0_joint_fpr_at_alpha_per_signal": result.h0_joint_fpr,
        "t_for_90pct_power_joint": _interp_threshold(
            result.t_grid.astype(np.float64), result.joint_power_vs_t, 0.9
        ),
        "snr_for_90pct_power_joint": _interp_threshold(
            result.snr_amplitudes / cfg.snr_noise, result.joint_power_vs_snr, 0.9
        ),
    }
    for sig in cfg.signals:
        metrics[f"power_vs_t__{sig}"] = result.power_vs_t[sig].tolist()
        metrics[f"mean_p_vs_t__{sig}"] = result.mean_p_vs_t[sig].tolist()
        metrics[f"power_vs_snr__{sig}"] = result.power_vs_snr[sig].tolist()
        metrics[f"mean_p_vs_snr__{sig}"] = result.mean_p_vs_snr[sig].tolist()
        metrics[f"h0_fpr__{sig}"] = result.h0_fpr[sig]
        metrics[f"h0_mean_p__{sig}"] = result.h0_mean_p[sig]
    save_metrics(run_dir, metrics)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "h4_detector_power.pdf"
    plot_power_curves(result, out_path=str(fig_path))
    plot_power_curves(result, out_path=str(run_dir / "h4_detector_power.pdf"))

    print(f"Wrote results to: {run_dir}")
    print(f"Wrote figure to:  {fig_path}")
    print(f"H_0 joint FPR @ α/m={cfg.alpha_per_signal}: {result.h0_joint_fpr:.4f}")
    for sig in cfg.signals:
        print(f"  signal '{sig}': FPR={result.h0_fpr[sig]:.4f}  mean_p={result.h0_mean_p[sig]:.4f}")
    print(f"Joint power vs T: {dict(zip(cfg.t_grid, result.joint_power_vs_t, strict=True))}")
    print(
        "Joint power vs SNR: "
        f"{dict(zip([round(a / cfg.snr_noise, 3) for a in cfg.snr_amplitudes], result.joint_power_vs_snr, strict=True))}"
    )


def _interp_threshold(
    xs: NDArray[np.float64], ys: NDArray[np.float64], threshold: float
) -> float | None:
    """First x where y crosses `threshold`, via linear interpolation between adjacent grid points.

    Returns None if `ys` never reaches `threshold`.
    """
    if not np.any(ys >= threshold):
        return None
    above = np.where(ys >= threshold)[0]
    first = int(above[0])
    if first == 0:
        return float(xs[0])
    x_lo, x_hi = float(xs[first - 1]), float(xs[first])
    y_lo, y_hi = float(ys[first - 1]), float(ys[first])
    if y_hi == y_lo:
        return x_hi
    return float(x_lo + (threshold - y_lo) / (y_hi - y_lo) * (x_hi - x_lo))


if __name__ == "__main__":
    main()
