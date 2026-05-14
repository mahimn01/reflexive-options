"""2D bimodality scan on the (log S, v) marginal — H_bimod follow-up to §7.4.

The 1D log-spot scan in `paper/theory.md` §7.4 refuted H_bimod across the
literature-prior κ range with γ = 0 (the leverage feedback channel turned
off). §7.4 itself flagged two methodological revisions for the empirical
phase:

  (a) re-run the scan with γ > 0 active so the 3D Hopf channel can close;
  (b) test bimodality on the 2D (log S, v) joint density rather than the
      1D log S marginal — the limit-cycle signature lives in 2D phase space.

This module executes both revisions:

  - Sweep κ ∈ literature-prior range × {0.5·κ★, 0.9·κ★, κ★, 1.05·κ★} where
    κ★ is the simulator's stability-envelope upper bound (computed via a
    pre-scan; see `find_stability_envelope_kappa_star`).
  - For each κ: simulate the 3D reflexive SDE with γ > 0 active, n_paths=2000,
    n_steps=4000, dt=1/(252·390); drop the first half of each path as burn-in.
  - On the joint (log S, v) sample cloud:
      * Project onto the leading PCA direction; run Hartigan dip on the
        projection (1D test on the most informative direction).
      * Compute Silverman bandwidth-test bimodality on each 1D slice.
      * Render the 2D KDE contour plot for κ ∈ {0, κ★, 1.05·κ★}.

Run: ``python -m reflexive_options.experiments.h_bimod_2d_scan``
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import diptest
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from reflexive_options.experiments._common import (
    FIGURES_DIR,
    deterministic_rng,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.simulator.gamma_aggregator import (
    GammaAggregator,
    GammaAggregatorConfig,
)
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    ReflexiveParams,
    SurfaceGrid,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HBimodScanConfig:
    """Configuration for the 2D H_bimod follow-up scan."""

    # Heston backbone — match `paper/theory.md` §7.1 setup so this is a clean
    # follow-up to the original 1D H_bimod scan.
    base_kappa_v: float = 2.0
    base_theta: float = 0.04
    base_xi: float = 0.30
    base_rho: float = -0.70
    base_v0: float = 0.04
    initial_spot: float = 100.0

    # Reflexive memory channel (active per the §7.4 flag).
    leverage_gamma: float = 0.5  # γ > 0 closes the 3D Hopf channel
    memory_decay_alpha: float = 252.0
    memory_intake_beta: float = 1.0

    # OI grid for the dealer-gamma aggregator (matches §7.1: flat 7×3, 50k/cell).
    oi_contracts_per_cell: float = 50_000.0
    oi_log_moneyness: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)
    oi_maturities_days: tuple[float, ...] = (30.0, 90.0, 180.0)

    # κ-grid spec: relative to a stability-envelope κ★ found by pre-scan.
    kappa_grid_relative: tuple[float, ...] = (0.0, 0.5, 0.9, 1.0, 1.05)

    # Stability pre-scan (used to set κ★).
    kappa_envelope_search_max: float = 1.0e-8
    kappa_envelope_n_probes: int = 16
    kappa_envelope_n_paths: int = 200
    kappa_envelope_n_steps: int = 1000
    kappa_envelope_dt: float = 1.0 / (252.0 * 390.0)

    # Main scan budget (per the spec).
    n_paths: int = 2000
    n_steps: int = 4000
    dt: float = 1.0 / (252.0 * 390.0)
    burn_in_frac: float = 0.5  # first half of each path is dropped

    # Sub-sample size for KDE / dip tests (full sample is too large for diptest).
    max_samples_for_test: int = 20_000

    # Reproducibility
    seed: int = 42


@dataclass(frozen=True)
class BimodalityOutcome:
    """Per-κ bimodality test outcome on the (log S, v) joint sample."""

    kappa: float
    n_samples: int
    pca_dip_statistic: float
    pca_dip_p_value: float
    pca_is_bimodal: bool
    silverman_log_s_bw_excess: float  # see _silverman_bandwidth_test
    silverman_log_s_bimodal: bool
    silverman_v_bw_excess: float
    silverman_v_bimodal: bool
    pca_principal_direction: list[float] = field(default_factory=list)
    pca_explained_variance_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Simulator + aggregator factory at the §7.1 setup
# ---------------------------------------------------------------------------


def _build_oi_grid(cfg: HBimodScanConfig) -> OpenInterestGrid:
    log_moneyness = np.asarray(cfg.oi_log_moneyness, dtype=np.float64)
    maturities = np.asarray(cfg.oi_maturities_days, dtype=np.float64) / 365.0
    grid = SurfaceGrid(log_moneyness=log_moneyness, maturities=maturities)
    contracts = np.full(grid.shape, cfg.oi_contracts_per_cell, dtype=np.float64)
    return OpenInterestGrid(grid=grid, contracts_open=contracts)


def _make_simulator(cfg: HBimodScanConfig, kappa: float) -> ReflexiveSimulator:
    base = HestonParams(
        kappa=cfg.base_kappa_v,
        theta=cfg.base_theta,
        xi=cfg.base_xi,
        rho=cfg.base_rho,
        v0=cfg.base_v0,
    )
    params = ReflexiveParams(
        base=base,
        coupling=float(kappa),
        drift=0.0,
        memory_decay=cfg.memory_decay_alpha,
        memory_intake=cfg.memory_intake_beta,
        leverage=cfg.leverage_gamma,
    )
    aggregator = GammaAggregator(
        oi_grid=_build_oi_grid(cfg),
        risk_free_rate=0.0,
        config=GammaAggregatorConfig(fixed_iv=0.20),
    )
    return ReflexiveSimulator(
        params=params,
        gamma_aggregator=aggregator,
        initial_spot=cfg.initial_spot,
    )


# ---------------------------------------------------------------------------
# Stability pre-scan: κ★ = largest κ where the simulator stays finite
# ---------------------------------------------------------------------------


def find_stability_envelope_kappa_star(cfg: HBimodScanConfig) -> float:
    """Bracket-search κ★ — the largest κ for which the reflexive simulator
    keeps spot+variance finite at the canonical (γ > 0, OI-grid) setup.

    Bisection over `[0, kappa_envelope_search_max]`: the upper bound is
    "blowup" iff after `kappa_envelope_n_steps` steps any path's spot or
    variance is non-finite. We refine until the bracket width is < 5% of the
    midpoint, return the midpoint as κ★.

    The literature prior is κ ≈ 5e-12 per USD-of-dealer-gamma (GPP 2009);
    this pre-scan locates the simulator-implied envelope which is typically a
    few orders of magnitude above the empirical prior at the §7.1 OI scale
    plus γ > 0.
    """
    lo = 0.0
    hi = float(cfg.kappa_envelope_search_max)
    # First, find an upper bracket where the sim DOES blow up.
    if not _simulator_blows_up(cfg, hi):
        # Envelope is wider than the search-max — return the search-max as a
        # conservative κ★. This is fine — the bimodality scan only needs a κ
        # around which to anchor the relative grid.
        return hi
    # Bisect.
    for _ in range(cfg.kappa_envelope_n_probes):
        mid = 0.5 * (lo + hi)
        if _simulator_blows_up(cfg, mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.05 * max(mid, 1e-30):
            break
    return float(lo)  # last known stable κ; κ★ = stability-envelope upper bound


def _simulator_blows_up(cfg: HBimodScanConfig, kappa: float) -> bool:
    sim = _make_simulator(cfg, kappa)
    spots, variances = sim.simulate(
        n_paths=cfg.kappa_envelope_n_paths,
        n_steps=cfg.kappa_envelope_n_steps,
        dt=cfg.kappa_envelope_dt,
        seed=cfg.seed + round(kappa * 1e16),
    )
    return bool((~np.isfinite(spots)).any() or (~np.isfinite(variances)).any())


# ---------------------------------------------------------------------------
# Joint (log S, v) sample collection
# ---------------------------------------------------------------------------


def _simulate_joint_samples(
    cfg: HBimodScanConfig,
    kappa: float,
    *,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], int, int]:
    """Run the 3D reflexive SDE; return the post-burn-in (log S, v) joint sample.

    Returns:
        joint: shape (n_kept, 2) where column 0 is log S and column 1 is v.
        n_total: total raw (path, step) cells before finiteness filter.
        n_kept: cells with finite (log S, v).
    """
    sim = _make_simulator(cfg, kappa)
    seed = int(rng.integers(low=0, high=2**31 - 1))
    spots, variances = sim.simulate(
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        seed=seed,
    )
    burn_in = round(cfg.burn_in_frac * cfg.n_steps)
    s_post = spots[:, burn_in:]
    v_post = variances[:, burn_in:]
    log_s = np.log(np.maximum(s_post, 1e-12))
    finite_mask = np.isfinite(log_s) & np.isfinite(v_post)
    log_s_flat = log_s.ravel()
    v_flat = v_post.ravel()
    finite_mask_flat = finite_mask.ravel()
    n_total = int(finite_mask_flat.size)
    n_kept = int(finite_mask_flat.sum())
    joint = np.column_stack([log_s_flat[finite_mask_flat], v_flat[finite_mask_flat]]).astype(
        np.float64
    )
    return joint, n_total, n_kept


# ---------------------------------------------------------------------------
# 2D bimodality tests — PCA-projected dip + Silverman-bandwidth slices
# ---------------------------------------------------------------------------


def _pca_projection(
    samples_2d: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Standard PCA on the 2D sample cloud.

    Returns:
        projected: shape (n,) projection onto the leading PC direction.
        principal_direction: shape (2,) unit vector in (log_s, v) space.
        explained_variance_ratio: leading PC's share of total variance.
    """
    if samples_2d.shape[0] < 2:
        raise ValueError(f"PCA needs >= 2 samples, got {samples_2d.shape[0]}")
    centred = samples_2d - samples_2d.mean(axis=0, keepdims=True)
    # Standardise channels first so the PCA isn't dominated by the scale of v
    # (typically O(0.04)) vs log S (O(1)). This is the right thing for a
    # "most informative direction" test — without standardisation the leading
    # PC degenerates to log S almost always.
    std = centred.std(axis=0, ddof=1)
    std = np.where(std > 1e-300, std, 1.0)
    standardised = centred / std
    cov = np.cov(standardised.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Eigenvalues ascending; pick the largest.
    idx = int(np.argmax(eigvals))
    direction = eigvecs[:, idx].astype(np.float64)
    projected = (standardised @ direction).astype(np.float64)
    explained = float(eigvals[idx] / max(np.sum(eigvals), 1e-300))
    return projected, direction, explained


def _silverman_bandwidth_test(
    samples_1d: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    n_bootstrap: int = 200,
) -> tuple[float, bool]:
    """Silverman (1981) bandwidth test for unimodality.

    Returns ``(bw_excess, is_bimodal)``. ``bw_excess`` is the ratio of the
    minimum bandwidth at which the bootstrap KDE is still unimodal to the
    Silverman-rule bandwidth on the original sample. is_bimodal = bootstrap
    p-value < 0.05 by the Silverman convention (proportion of bootstrap
    bandwidths exceeding the data's critical bandwidth).

    Implementation note: Silverman's exact algorithm bisects to find the
    critical bandwidth at which the KDE transitions from unimodal to bimodal.
    For tractability at our sample sizes we use the simpler proxy "ratio of
    Silverman-rule bandwidth to the bandwidth at which 95% of bootstrap KDEs
    are still unimodal"; the test is approximate but operationally consistent
    with §7.4's reporting precision.
    """
    x = np.asarray(samples_1d, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 50:
        return float("nan"), False
    sigma = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    h_silverman = 0.9 * min(sigma, iqr / 1.34) * (n ** (-0.2))
    # Critical bandwidth h_c: smallest h such that the KDE is unimodal at h.
    # Bisection on [h_silverman/10, h_silverman*10].
    lo = max(h_silverman / 10.0, 1e-12)
    hi = max(h_silverman * 10.0, lo * 10.0)
    for _ in range(20):
        mid = math.sqrt(lo * hi)
        if _kde_is_unimodal(x, mid):
            hi = mid
        else:
            lo = mid
    h_critical = hi
    # Bootstrap p-value: fraction of bootstrap samples with critical bandwidth
    # >= h_critical. Under H_0 (unimodal), critical bandwidths concentrate
    # below the data's h_critical.
    n_boot_uni = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(low=0, high=n, size=n)
        xb = x[idx]
        if _kde_is_unimodal(xb, h_critical):
            n_boot_uni += 1
    p_value = float(n_boot_uni) / float(n_bootstrap)
    bw_excess = float(h_critical / max(h_silverman, 1e-12))
    is_bimodal = bool(p_value < 0.05)
    return bw_excess, is_bimodal


def _kde_is_unimodal(x: NDArray[np.float64], h: float) -> bool:
    """Cheap unimodality test: count local maxima of a Gaussian-KDE on a fine grid.

    Uses scipy.stats.gaussian_kde with bandwidth h injected via the
    `bw_method` callable hook — avoids the numerical edge cases of a hand-
    rolled KDE.
    """
    if x.size < 4 or h <= 0.0:
        return True
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return True
    grid = np.linspace(lo - 0.5 * h, hi + 0.5 * h, 256)
    sigma = float(np.std(x, ddof=1))
    if sigma <= 1e-12:
        return True
    factor = h / sigma  # gaussian_kde uses bw_method as factor on sigma
    try:
        kde = stats.gaussian_kde(x, bw_method=factor)
    except (np.linalg.LinAlgError, ValueError):
        return True
    density = np.asarray(kde(grid), dtype=np.float64)
    peaks = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    return bool(peaks.sum() <= 1)


def evaluate_bimodality(
    joint: NDArray[np.float64],
    *,
    kappa: float,
    rng: np.random.Generator,
    max_samples: int,
) -> BimodalityOutcome:
    """Run the 3-test panel on a single κ's joint (log S, v) sample cloud."""
    if joint.shape[0] > max_samples:
        sub_idx = rng.choice(joint.shape[0], size=max_samples, replace=False)
        sub = joint[sub_idx]
    else:
        sub = joint
    projected, direction, explained = _pca_projection(sub)

    # Hartigan dip on the PCA projection.
    if projected.size >= 4:
        dip, p_value = diptest.diptest(projected)
    else:
        dip, p_value = float("nan"), float("nan")
    is_bimodal_pca = bool(p_value < 0.05) if math.isfinite(p_value) else False

    bw_log_s, is_bi_log_s = _silverman_bandwidth_test(sub[:, 0], rng=rng)
    bw_v, is_bi_v = _silverman_bandwidth_test(sub[:, 1], rng=rng)

    return BimodalityOutcome(
        kappa=kappa,
        n_samples=int(sub.shape[0]),
        pca_dip_statistic=float(dip),
        pca_dip_p_value=float(p_value),
        pca_is_bimodal=is_bimodal_pca,
        silverman_log_s_bw_excess=float(bw_log_s),
        silverman_log_s_bimodal=is_bi_log_s,
        silverman_v_bw_excess=float(bw_v),
        silverman_v_bimodal=is_bi_v,
        pca_principal_direction=[float(d) for d in direction],
        pca_explained_variance_ratio=float(explained),
    )


# ---------------------------------------------------------------------------
# 2D KDE figure — render contour panels at κ ∈ {0, κ★, 1.05·κ★}
# ---------------------------------------------------------------------------


def render_2d_kde_figure(
    samples_by_kappa: dict[float, NDArray[np.float64]],
    *,
    kappa_star: float,
    out_path: Path,
    rng: np.random.Generator,
    max_samples_per_panel: int = 8_000,
) -> Path:
    """Three-panel (log S, v) joint-density contour at κ ∈ {0, κ★, 1.05·κ★}."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels: list[tuple[float, str]] = [
        (0.0, r"$\kappa = 0$"),
        (kappa_star, r"$\kappa = \kappa^\star$"),
        (1.05 * kappa_star, r"$\kappa = 1.05\,\kappa^\star$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, (kappa, label) in zip(axes, panels, strict=True):
        # Match exact key (floats compared). Pick nearest key if not present.
        keys = list(samples_by_kappa.keys())
        if not keys:
            ax.set_title(f"{label}\n(no data)")
            continue
        k_match = min(keys, key=lambda k: abs(k - kappa))
        sub = samples_by_kappa[k_match]
        if sub.shape[0] > max_samples_per_panel:
            idx = rng.choice(sub.shape[0], size=max_samples_per_panel, replace=False)
            sub = sub[idx]
        # KDE estimate via gaussian_kde on a regular grid.
        if sub.shape[0] < 50:
            ax.set_title(f"{label}\n(too few samples)")
            continue
        try:
            kde = stats.gaussian_kde(sub.T)
        except (np.linalg.LinAlgError, ValueError):
            ax.scatter(sub[:, 0], sub[:, 1], s=2, alpha=0.3)
            ax.set_title(f"{label}\n(KDE failed)")
            continue
        log_s_min, log_s_max = np.percentile(sub[:, 0], [1, 99])
        v_min, v_max = np.percentile(sub[:, 1], [1, 99])
        log_s_grid = np.linspace(log_s_min, log_s_max, 80)
        v_grid = np.linspace(v_min, v_max, 80)
        L, V = np.meshgrid(log_s_grid, v_grid)
        density = kde(np.vstack([L.ravel(), V.ravel()])).reshape(L.shape)
        ax.contourf(L, V, density, levels=12, cmap="viridis")
        ax.contour(L, V, density, levels=8, colors="white", linewidths=0.4, alpha=0.5)
        ax.set_xlabel(r"$\log S$")
        if ax is axes[0]:
            ax.set_ylabel(r"$v$")
        ax.set_title(f"{label}  ($\\kappa$={k_match:.2e}, n={sub.shape[0]})")
    fig.suptitle(
        r"Stationary joint density $\pi^\star(\log S, v)$ — H_bimod 2D follow-up to §7.4 "
        r"(γ > 0)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_h_bimod_2d_scan(cfg: HBimodScanConfig, run_dir: Path) -> dict[str, Any]:
    save_config(run_dir, cfg)

    print("Pre-scan: locating κ★ stability envelope at γ > 0...")
    with timed("kappa_star_envelope_search"):
        kappa_star = find_stability_envelope_kappa_star(cfg)
    print(f"  κ★ ≈ {kappa_star:.4e}")

    rng = deterministic_rng(cfg.seed)
    kappa_grid = [float(r) * kappa_star for r in cfg.kappa_grid_relative]
    # Replace any duplicate (e.g. 0.0·κ★ = 0.0) by uniqueness.
    kappa_grid = sorted(set(kappa_grid))

    samples_by_kappa: dict[float, NDArray[np.float64]] = {}
    outcomes: list[BimodalityOutcome] = []

    for kappa in kappa_grid:
        with timed(f"simulate_kappa_{kappa:.3e}"):
            joint, n_total, n_kept = _simulate_joint_samples(cfg, kappa, rng=rng)
        if joint.shape[0] < 100:
            print(
                f"  κ={kappa:.3e}: only {joint.shape[0]} finite samples — skipping bimodality test"
            )
            continue
        samples_by_kappa[kappa] = joint
        with timed(f"bimodality_kappa_{kappa:.3e}"):
            outcome = evaluate_bimodality(
                joint,
                kappa=kappa,
                rng=rng,
                max_samples=cfg.max_samples_for_test,
            )
        outcomes.append(outcome)
        print(
            f"  κ={kappa:.3e}: n_kept/n_total={n_kept}/{n_total}  "
            f"PCA dip={outcome.pca_dip_statistic:.4e} (p={outcome.pca_dip_p_value:.3f})  "
            f"PCA explained={outcome.pca_explained_variance_ratio:.3f}"
        )

    fig_path = FIGURES_DIR / "stationary_density_2d_kde.pdf"
    if samples_by_kappa:
        render_2d_kde_figure(
            samples_by_kappa,
            kappa_star=kappa_star,
            out_path=fig_path,
            rng=rng,
        )

    # Determine the headline outcome: any κ in the grid with PCA dip p < 0.05?
    bimodal_kappas = [o.kappa for o in outcomes if o.pca_is_bimodal]
    headline = (
        "2D bimodality DETECTED on PCA-projected joint at κ ∈ "
        f"{[f'{k:.3e}' for k in bimodal_kappas]} (γ > 0 active)"
        if bimodal_kappas
        else "2D bimodality NOT DETECTED — H_bimod remains refuted on (log S, v) PCA projection"
    )

    metrics: dict[str, Any] = {
        "config": {
            "kappa_star_relative": list(cfg.kappa_grid_relative),
            "kappa_star_envelope": kappa_star,
            "leverage_gamma": cfg.leverage_gamma,
            "n_paths": cfg.n_paths,
            "n_steps": cfg.n_steps,
            "dt": cfg.dt,
            "burn_in_frac": cfg.burn_in_frac,
            "seed": cfg.seed,
        },
        "kappa_star_envelope": kappa_star,
        "kappa_grid": list(kappa_grid),
        "outcomes": [
            {
                "kappa": o.kappa,
                "n_samples": o.n_samples,
                "pca_dip_statistic": o.pca_dip_statistic,
                "pca_dip_p_value": o.pca_dip_p_value,
                "pca_is_bimodal": o.pca_is_bimodal,
                "silverman_log_s_bw_excess": o.silverman_log_s_bw_excess,
                "silverman_log_s_bimodal": o.silverman_log_s_bimodal,
                "silverman_v_bw_excess": o.silverman_v_bw_excess,
                "silverman_v_bimodal": o.silverman_v_bimodal,
                "pca_principal_direction": o.pca_principal_direction,
                "pca_explained_variance_ratio": o.pca_explained_variance_ratio,
            }
            for o in outcomes
        ],
        "any_pca_bimodal": bool(bimodal_kappas),
        "bimodal_kappas": bimodal_kappas,
        "headline": headline,
        "figure_path": str(fig_path),
    }
    save_metrics(run_dir, metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoketest", action="store_true")
    args = parser.parse_args()

    if args.smoketest:
        cfg = HBimodScanConfig(
            n_paths=200,
            n_steps=400,
            kappa_grid_relative=(0.0, 0.5, 1.0),
            kappa_envelope_n_paths=50,
            kappa_envelope_n_steps=400,
            kappa_envelope_n_probes=8,
            max_samples_for_test=4_000,
            seed=args.seed,
        )
    else:
        cfg = HBimodScanConfig(
            n_paths=args.n_paths,
            n_steps=args.n_steps,
            seed=args.seed,
        )

    run_dir = make_run_dir("h_bimod_2d", seed=cfg.seed)
    metrics = run_h_bimod_2d_scan(cfg, run_dir)

    print()
    print(metrics["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
