"""H4 PSD-peak detector validation suite — power curves on synthetic ground truth.

End-to-end characterisation of `theory.spectral.detect_psd_peak`: detection
rate (fraction of seeds where the detector fires) as a function of (i)
trajectory length T at fixed SNR and (ii) signal-to-noise ratio at fixed T.
The synthetic positive control is a sinusoid at ω* embedded in i.i.d.
Gaussian noise.

Saves:
    runs/h4_validation/<timestamp>/power_curve.npz
    runs/h4_validation/<timestamp>/metrics.json
    runs/h4_validation/<timestamp>/config.json
    paper/figures/h4_detector_power.pdf

Run: python -m reflexive_options.experiments.h4_validation
     python -m reflexive_options.experiments.h4_validation --quick   # smaller grid
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

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
    # Decision threshold
    alpha: float = 0.05
    # H_0 calibration check
    n_h0_seeds: int = 200
    h0_t: int = 1024


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


def _power_at(
    *,
    t: int,
    amplitude: float,
    noise_std: float,
    cfg: H4ValidationConfig,
    base_seed: int,
) -> tuple[float, float]:
    """Detection power and false-positive rate at one (T, SNR) point.

    Returns (power, mean_p_value). Power = fraction of seeds where the
    detector fires (in_band True AND p < cfg.alpha).
    """
    fires = 0
    p_sum = 0.0
    for s in range(cfg.n_seeds_per_point):
        x = _sinusoid_with_noise(
            t=t,
            sampling_rate=cfg.sampling_rate,
            frequency=cfg.omega_star,
            amplitude=amplitude,
            noise_std=noise_std,
            seed=base_seed + s,
        )
        res = detect_psd_peak(
            x,
            sampling_rate=cfg.sampling_rate,
            omega_star=cfg.omega_star,
            bandwidth_frac=cfg.bandwidth_frac,
            welch_window=cfg.welch_window,
            welch_overlap=cfg.welch_overlap,
            n_permutations=cfg.n_permutations,
            rng=np.random.default_rng(base_seed + s + 100_000),
        )
        if res.in_band and res.p_value < cfg.alpha:
            fires += 1
        p_sum += res.p_value
    return fires / cfg.n_seeds_per_point, p_sum / cfg.n_seeds_per_point


def _h0_calibration(
    *, cfg: H4ValidationConfig, base_seed: int
) -> tuple[NDArray[np.float64], float, float]:
    """Run the detector on `cfg.n_h0_seeds` white-noise traces; return (p_values, FPR, mean_p)."""
    p_values = np.empty(cfg.n_h0_seeds, dtype=np.float64)
    for s in range(cfg.n_h0_seeds):
        rng = np.random.default_rng(base_seed + s)
        x = rng.standard_normal(cfg.h0_t).astype(np.float64)
        res = detect_psd_peak(
            x,
            sampling_rate=cfg.sampling_rate,
            omega_star=cfg.omega_star,
            bandwidth_frac=cfg.bandwidth_frac,
            welch_window=cfg.welch_window,
            welch_overlap=cfg.welch_overlap,
            n_permutations=cfg.n_permutations,
            rng=np.random.default_rng(base_seed + s + 999_999),
        )
        p_values[s] = res.p_value
    fpr = float(np.mean(p_values < cfg.alpha))
    return p_values, fpr, float(p_values.mean())


@dataclass
class H4ValidationResult:
    """Bundled results for the H4 power-curve scan."""

    cfg: H4ValidationConfig
    t_grid: NDArray[np.int64]
    power_vs_t: NDArray[np.float64]
    mean_p_vs_t: NDArray[np.float64]
    snr_amplitudes: NDArray[np.float64]
    power_vs_snr: NDArray[np.float64]
    mean_p_vs_snr: NDArray[np.float64]
    h0_p_values: NDArray[np.float64]
    h0_fpr: float
    h0_mean_p: float
    figures: list[str] = field(default_factory=list)


def run_validation(cfg: H4ValidationConfig, *, base_seed: int = 42) -> H4ValidationResult:
    """Compute the two power curves and the H_0 p-value distribution."""
    n_t = len(cfg.t_grid)
    power_vs_t = np.zeros(n_t, dtype=np.float64)
    mean_p_vs_t = np.zeros(n_t, dtype=np.float64)
    for i, t_val in enumerate(cfg.t_grid):
        power_vs_t[i], mean_p_vs_t[i] = _power_at(
            t=t_val,
            amplitude=cfg.fixed_snr_amplitude,
            noise_std=cfg.fixed_snr_noise,
            cfg=cfg,
            base_seed=base_seed + 1_000 * i,
        )

    n_snr = len(cfg.snr_amplitudes)
    power_vs_snr = np.zeros(n_snr, dtype=np.float64)
    mean_p_vs_snr = np.zeros(n_snr, dtype=np.float64)
    for i, amp in enumerate(cfg.snr_amplitudes):
        power_vs_snr[i], mean_p_vs_snr[i] = _power_at(
            t=cfg.snr_t,
            amplitude=amp,
            noise_std=cfg.snr_noise,
            cfg=cfg,
            base_seed=base_seed + 100_000 + 1_000 * i,
        )

    h0_p, h0_fpr, h0_mean_p = _h0_calibration(cfg=cfg, base_seed=base_seed + 500_000)

    return H4ValidationResult(
        cfg=cfg,
        t_grid=np.asarray(cfg.t_grid, dtype=np.int64),
        power_vs_t=power_vs_t,
        mean_p_vs_t=mean_p_vs_t,
        snr_amplitudes=np.asarray(cfg.snr_amplitudes, dtype=np.float64),
        power_vs_snr=power_vs_snr,
        mean_p_vs_snr=mean_p_vs_snr,
        h0_p_values=h0_p,
        h0_fpr=h0_fpr,
        h0_mean_p=h0_mean_p,
    )


def plot_power_curves(result: H4ValidationResult, *, out_path: str) -> None:
    """Two-panel power curve: (a) power vs T, (b) power vs SNR. Saved as PDF."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(result.t_grid, result.power_vs_t, marker="o")
    axes[0].axhline(0.9, color="grey", linestyle=":", linewidth=0.8, label="90% target")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Trajectory length T (samples)")
    axes[0].set_ylabel("Detection power")
    axes[0].set_title(
        f"H4 power vs T (SNR = A/σ = {result.cfg.fixed_snr_amplitude / result.cfg.fixed_snr_noise:.2f})"
    )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    snr_ratios = result.snr_amplitudes / result.cfg.snr_noise
    axes[1].plot(snr_ratios, result.power_vs_snr, marker="o")
    axes[1].axhline(0.9, color="grey", linestyle=":", linewidth=0.8, label="90% target")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("SNR (amplitude / noise std)")
    axes[1].set_ylabel("Detection power")
    axes[1].set_title(f"H4 power vs SNR (T = {result.cfg.snr_t})")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"H4 PSD-peak detector — Welch {result.cfg.welch_window}-day Hann, "
        f"ω* = {result.cfg.omega_star} cyc/yr, ±{int(result.cfg.bandwidth_frac * 100)}% band, "
        f"α = {result.cfg.alpha}, n_perm = {result.cfg.n_permutations}, "
        f"H_0 FPR = {result.h0_fpr:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller seed count and Welch n_perm for fast smoke-runs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        cfg = H4ValidationConfig(
            n_seeds_per_point=20,
            n_permutations=50,
            n_h0_seeds=50,
            t_grid=(256, 1024, 4096),
            snr_amplitudes=(0.10, 0.40, 1.60),
        )
    else:
        cfg = H4ValidationConfig()

    run_dir = make_run_dir("h4_validation")
    save_config(run_dir, cfg)

    with timed("h4_validation.run_validation"):
        result = run_validation(cfg, base_seed=args.seed)

    np.savez(
        run_dir / "power_curve.npz",
        t_grid=result.t_grid,
        power_vs_t=result.power_vs_t,
        mean_p_vs_t=result.mean_p_vs_t,
        snr_amplitudes=result.snr_amplitudes,
        power_vs_snr=result.power_vs_snr,
        mean_p_vs_snr=result.mean_p_vs_snr,
        h0_p_values=result.h0_p_values,
    )

    metrics: dict[str, object] = {
        "t_grid": result.t_grid.tolist(),
        "power_vs_t": result.power_vs_t.tolist(),
        "mean_p_vs_t": result.mean_p_vs_t.tolist(),
        "snr_amplitudes": result.snr_amplitudes.tolist(),
        "snr_ratios": (result.snr_amplitudes / cfg.snr_noise).tolist(),
        "power_vs_snr": result.power_vs_snr.tolist(),
        "mean_p_vs_snr": result.mean_p_vs_snr.tolist(),
        "h0_fpr_at_alpha_005": result.h0_fpr,
        "h0_mean_p": result.h0_mean_p,
        "t_for_90pct_power": _interp_threshold(
            result.t_grid.astype(np.float64), result.power_vs_t, 0.9
        ),
        "snr_for_90pct_power": _interp_threshold(
            result.snr_amplitudes / cfg.snr_noise, result.power_vs_snr, 0.9
        ),
    }
    save_metrics(run_dir, metrics)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "h4_detector_power.pdf"
    plot_power_curves(result, out_path=str(fig_path))
    plot_power_curves(result, out_path=str(run_dir / "h4_detector_power.pdf"))

    print(f"Wrote results to: {run_dir}")
    print(f"Wrote figure to:  {fig_path}")
    print(f"H_0 FPR @ α=0.05: {result.h0_fpr:.4f} (target ≈ {cfg.alpha})")
    print(f"H_0 mean p-value: {result.h0_mean_p:.4f}")
    print(f"Power vs T: {dict(zip(cfg.t_grid, result.power_vs_t, strict=True))}")
    print(
        "Power vs SNR: "
        f"{dict(zip([round(a / cfg.snr_noise, 3) for a in cfg.snr_amplitudes], result.power_vs_snr, strict=True))}"
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
