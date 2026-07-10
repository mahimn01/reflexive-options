"""Generate `tests/repro/baseline_v0.3.3.json` — the v0.3.3 Wave 1–6 receipt.

This is the *second* reproducibility receipt, complementing
`baseline_v0.1.0.json`. It pins the headline numerical claims introduced in
Wave 1–6 (paper v0.3.1 → v0.3.3): lambda scaling, supercritical limit cycle,
Hawkes–SV equivalence (including the §3.9 Brent-root anchor), codim-2 /
Bautin analysis, McKean–Vlasov propagation-of-chaos, κ★ robustness
elasticities, 2D H_bimod scan, and the H1 synthetic-validation pipeline.

Scope and rationale
-------------------
The v0.1.0 receipt locks four experiments (`bifurcation_scan`,
`phase_diagram`, `synthetic_replication`, `reflexive_transfer`) at a single
canonical seed with whole-metric-dict hashing. That schema is preserved
verbatim — its registry test (`test_every_registered_spec_appears_in_receipt`)
would break if we added new specs to `_all_specs()`. The v0.3.3 receipt is a
parallel artefact with a *per-metric* tolerance schema:

    {
      "version": "0.3.3",
      "experiments": {
        "<experiment_name>": {
          "tolerance_class": "deterministic_exact" | "stochastic_relative",
          "config": {...},
          "seed": 20260514,
          "metrics": {
            "<metric>": {"value": X, "tolerance_abs": Y}   # deterministic
            "<metric>": {"value": X, "tolerance_relative": Y}  # stochastic
          }
        },
        ...
      }
    }

Tolerances follow the CLAUDE.md "Reproducibility receipt" policy:
  * Closed-form deterministic computations (Routh–Hurwitz, Brent root,
    closed-form ℓ_1, OLS, ODE integration) lock at **1e-10 absolute**.
  * Historical stochastic simulations (the Lambda receipt is withdrawn in
    v0.4), particle SDE, BC-trained MLP, and Heston Euler lock at **5% relative**.

Headline-config locking
-----------------------
Each experiment is locked at the *exact* config that produced the paper's
headline numbers (NOT always the dataclass default). The headline configs
were chosen during the v0.3.1–v0.3.3 development cycle:

  * `mckean_vlasov_validation`: production sweep (n_grid=(10,32,100,316,
    1000), n_replicates=64, n_steps=250) — the headline 5-point Sznitman
    fit. Runs in ≈ 2 s so it fits the regression-test budget; the quick
    3-point sweep was deprecated in v0.3.5 per the 3P-CLAIMS audit.
  * `h_bimod_2d_scan`: n_paths=1000, n_steps=2000 — gave the n=15,769
    surviving-sample cell at κ_env=1.05·κ★_env that the paper cites.
  * `h1_synthetic_validation`: n_bc_train_episodes=30, n_paths_per_source=30
    — gave the SW2 = 0.005 / 0.034 / 0.054 ordering the paper cites.
  * The deterministic experiments use their dataclass defaults (those
    defaults produced the paper numbers).

Run via the wrapper:

    bash scripts/generate_repro_baseline.sh

Which invokes:

    uv run python -m reflexive_options.experiments._generate_repro_baseline_v033
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = REPO_ROOT / "tests" / "repro" / "baseline_v0.3.3.json"
SCHEMA_VERSION = "1.0"
RECEIPT_VERSION = "0.3.3"

# Tolerances per CLAUDE.md.
DETERMINISTIC_ABS_TOL = 1e-10
STOCHASTIC_REL_TOL = 0.05

ToleranceClass = str  # "deterministic_exact" | "stochastic_relative"


@dataclass(frozen=True)
class MetricSpec:
    """One headline metric to lock against drift.

    `extractor` reads the metric value from the experiment's metrics-dict
    (allows nested access without coupling to dict shape changes elsewhere).
    """

    name: str
    extractor: Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class WaveExperimentSpec:
    """One Wave 1–6 experiment + its locked headline metrics."""

    name: str
    runner: Callable[[], dict[str, Any]]
    config_dict: dict[str, Any]
    tolerance_class: ToleranceClass
    metrics: tuple[MetricSpec, ...]
    seed: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Tooling / git capture
# ---------------------------------------------------------------------------


def _tooling_versions() -> dict[str, str]:
    """Same package list as v0.1.0 + diptest (new Wave-1–6 dependency)."""
    pkgs = ("numpy", "scipy", "torch", "pandas", "gymnasium", "matplotlib", "QuantLib", "diptest")
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in pkgs:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "missing"
    return out


def _git_commit() -> str:
    """Best-effort current git HEAD; 'unknown' if git is unavailable.

    Resolves git through PATH; failure modes (missing git binary, not in a
    repo, timeout) are all caught and fall back to 'unknown' so the receipt
    generator stays usable in a non-git tarball.
    """
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        result = subprocess.run(  # noqa: S603 — git is resolved via shutil.which
            [git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


# ---------------------------------------------------------------------------
# Per-experiment runner adapters
# ---------------------------------------------------------------------------


def _run_lambda_scaling() -> dict[str, Any]:
    from reflexive_options.experiments.lambda_scaling import (
        LambdaScalingConfig,
        _fit_power_law,
        _scan_cells,
    )

    cfg = LambdaScalingConfig()
    cells = _scan_cells(cfg)
    fit = _fit_power_law(
        cells,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
        predicted_exponent=cfg.predicted_exponent,
    )
    return {
        "n_cells_total": len(cells),
        "n_cells_fit": fit.n_cells_fit,
        "fit_log_A": fit.log_A,
        "fit_B": fit.B,
        "fit_log_A_ci_low": fit.log_A_ci_low,
        "fit_log_A_ci_high": fit.log_A_ci_high,
        "fit_B_ci_low": fit.B_ci_low,
        "fit_B_ci_high": fit.B_ci_high,
        "fit_residual_std": fit.residual_std,
        "z_score_vs_two_thirds": fit.z_score_vs_two_thirds,
        "distinguishable_from_two_thirds": fit.distinguishable_from_two_thirds,
    }


def _config_lambda_scaling() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.lambda_scaling import LambdaScalingConfig

    return asdict(LambdaScalingConfig())


def _run_limit_cycle() -> dict[str, Any]:
    from reflexive_options.experiments.limit_cycle_supercritical import (
        LimitCycleConfig,
    )
    from reflexive_options.experiments.limit_cycle_supercritical import (
        run as _run,
    )

    m = _run(LimitCycleConfig())
    # Strip the trajectory blob (large arrays); keep scalar headline metrics.
    return {k: v for k, v in m.items() if k != "trajectory"}


def _config_limit_cycle() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.limit_cycle_supercritical import LimitCycleConfig

    return asdict(LimitCycleConfig())


def _run_hawkes_sv_equivalence() -> dict[str, Any]:
    from reflexive_options.experiments.hawkes_sv_equivalence import HawkesSVConfig
    from reflexive_options.experiments.hawkes_sv_equivalence import run as _run

    _, m = _run(HawkesSVConfig())
    return m


def _config_hawkes_sv_equivalence() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.hawkes_sv_equivalence import HawkesSVConfig

    return asdict(HawkesSVConfig())


def _run_codim2() -> dict[str, Any]:
    from reflexive_options.experiments.codim2_analysis import run as _run

    return _run(quick=False)


def _config_codim2() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.codim2_analysis import Codim2Config

    return asdict(Codim2Config())


def _run_mckean_vlasov() -> dict[str, Any]:
    from reflexive_options.experiments.mckean_vlasov_validation import run as _run

    # Production sweep — pins the n ∈ {10, 32, 100, 316, 1000} × 64-reps
    # numbers cited in §3.10. The full sweep runs in ≈ 2 s on a laptop,
    # so it stays inside the regression-gate budget while supplying the
    # 5-point Sznitman-slope fit the audit demanded.
    return _run(quick=False)


def _config_mckean_vlasov() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.mckean_vlasov_validation import MVValidationConfig

    return asdict(MVValidationConfig())


def _run_kappa_star_robustness() -> dict[str, Any]:
    from reflexive_options.experiments.kappa_star_robustness import run as _run

    return _run(quick=False)


def _config_kappa_star_robustness() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.kappa_star_robustness import RobustnessConfig

    return asdict(RobustnessConfig())


def _run_h_bimod_2d() -> dict[str, Any]:
    """Run the §7.4 H_bimod 2D scan at the locked headline config.

    The paper's n=15,769 surviving-sample cell at κ_env=1.05·κ★_env comes
    from n_paths=1000, n_steps=2000, seed=42. The dataclass defaults
    (n_paths=2000, n_steps=4000) would re-generate a slightly different
    sample count and dip statistic — we lock the headline-producing values.
    """
    import tempfile

    from reflexive_options.experiments.h_bimod_2d_scan import (
        HBimodScanConfig,
        run_h_bimod_2d_scan,
    )

    cfg = HBimodScanConfig(n_paths=1000, n_steps=2000, seed=42)
    with tempfile.TemporaryDirectory() as td:
        return run_h_bimod_2d_scan(cfg, Path(td))


def _config_h_bimod_2d() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.h_bimod_2d_scan import HBimodScanConfig

    return asdict(HBimodScanConfig(n_paths=1000, n_steps=2000, seed=42))


def _run_h1_synthetic() -> dict[str, Any]:
    """Run the H1 synthetic-validation pipeline at the locked headline config.

    The paper's SW2 = 0.005 / 0.034 / 0.054 ordering with disjoint CIs came
    from n_bc_train_episodes=30 + n_paths_per_source=30 at seed=42 (NOT the
    dataclass defaults of 50 / 100). Locking those.
    """
    import tempfile

    from reflexive_options.experiments.h1_synthetic_validation import (
        H1ValidationConfig,
        run_h1_synthetic_validation,
    )

    cfg = H1ValidationConfig(n_bc_train_episodes=30, n_paths_per_source=30, seed=42)
    with tempfile.TemporaryDirectory() as td:
        return run_h1_synthetic_validation(cfg, Path(td))


def _config_h1_synthetic() -> dict[str, Any]:
    from dataclasses import asdict

    from reflexive_options.experiments.h1_synthetic_validation import H1ValidationConfig

    return asdict(H1ValidationConfig(n_bc_train_episodes=30, n_paths_per_source=30, seed=42))


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------


def _get(d: dict[str, Any], *path: str) -> Any:
    """Nested-key lookup; raises KeyError with the failing path."""
    cur: Any = d
    for p in path:
        cur = cur[p]
    return cur


# ---------------------------------------------------------------------------
# Spec registry — the source of truth for what's locked
# ---------------------------------------------------------------------------


def _all_specs() -> list[WaveExperimentSpec]:
    """The Wave 1–6 spec registry.

    Order matches the CHANGELOG narrative (lambda → limit cycle →
    Hawkes-SV → codim2 → McKean-Vlasov → κ★ robustness → H_bimod 2D →
    H1 synthetic).
    """
    return [
        WaveExperimentSpec(
            name="lambda_scaling",
            runner=_run_lambda_scaling,
            config_dict=_config_lambda_scaling(),
            tolerance_class="deterministic_exact",
            seed=20260514,
            notes="WITHDRAWN v0.4: forced-state, not tangent-flow, Lambda scan.",
            metrics=(
                MetricSpec("n_cells_total", lambda m: int(m["n_cells_total"])),
                MetricSpec("n_cells_fit", lambda m: int(m["n_cells_fit"])),
                MetricSpec("fit_log_A", lambda m: float(m["fit_log_A"])),
                MetricSpec("fit_B", lambda m: float(m["fit_B"])),
                MetricSpec("fit_log_A_ci_low", lambda m: float(m["fit_log_A_ci_low"])),
                MetricSpec("fit_log_A_ci_high", lambda m: float(m["fit_log_A_ci_high"])),
                MetricSpec("fit_B_ci_low", lambda m: float(m["fit_B_ci_low"])),
                MetricSpec("fit_B_ci_high", lambda m: float(m["fit_B_ci_high"])),
                MetricSpec("fit_residual_std", lambda m: float(m["fit_residual_std"])),
                MetricSpec("z_score_vs_two_thirds", lambda m: float(m["z_score_vs_two_thirds"])),
            ),
        ),
        WaveExperimentSpec(
            name="limit_cycle_supercritical",
            runner=_run_limit_cycle,
            config_dict=_config_limit_cycle(),
            tolerance_class="deterministic_exact",
            seed=0,  # deterministic ODE — no seed
            notes="scipy.integrate.solve_ivp at rtol=1e-9, atol=1e-11; period from zero crossings.",
            metrics=(
                MetricSpec("kappa", lambda m: float(m["kappa"])),
                MetricSpec("kappa_star", lambda m: float(m["kappa_star"])),
                MetricSpec("omega_star", lambda m: float(m["omega_star"])),
                MetricSpec("period_theory", lambda m: float(m["period_theory"])),
                MetricSpec("period_measured", lambda m: float(m["period_measured"])),
                MetricSpec("period_relative_error", lambda m: float(m["period_relative_error"])),
                MetricSpec("amplitude_y", lambda m: float(m["amplitude_y"])),
                MetricSpec("amplitude_u", lambda m: float(m["amplitude_u"])),
                MetricSpec("amplitude_z", lambda m: float(m["amplitude_z"])),
                MetricSpec("n_kept_samples", lambda m: int(m["n_kept_samples"])),
            ),
        ),
        WaveExperimentSpec(
            name="hawkes_sv_equivalence",
            runner=_run_hawkes_sv_equivalence,
            config_dict=_config_hawkes_sv_equivalence(),
            tolerance_class="deterministic_exact",
            seed=0,  # deterministic eigendecomp + Brent root
            notes="3x3 Jacobian eigendecomp; Brent root on Routh-Hurwitz H(κ).",
            metrics=(
                MetricSpec("kappa_star_paper", lambda m: float(m["kappa_star_paper"])),
                MetricSpec("kappa_star_grid", lambda m: float(m["kappa_star_grid"])),
                MetricSpec("kappa_star_brent", lambda m: float(m["kappa_star_brent"])),
                MetricSpec("beta_zero", lambda m: float(m["beta_zero"])),
                MetricSpec("kappa_at_beta_zero", lambda m: float(m["kappa_at_beta_zero"])),
                MetricSpec("n_sv_at_kappa_star", lambda m: float(m["n_sv_at_kappa_star"])),
                MetricSpec(
                    "n_sv_at_kappa_star_brent",
                    lambda m: float(m["n_sv_at_kappa_star_brent"]),
                ),
                MetricSpec("n_sv_at_2_kappa_star", lambda m: float(m["n_sv_at_2_kappa_star"])),
                MetricSpec("criticality_residual", lambda m: float(m["criticality_residual"])),
                MetricSpec(
                    "criticality_residual_brent",
                    lambda m: float(m["criticality_residual_brent"]),
                ),
                MetricSpec("n_kappa", lambda m: int(m["n_kappa"])),
            ),
        ),
        WaveExperimentSpec(
            name="codim2_analysis",
            runner=_run_codim2,
            config_dict=_config_codim2(),
            tolerance_class="deterministic_exact",
            seed=0,  # deterministic closed-form scan
            notes="Closed-form ℓ_1 + saddle-node coupling on 71×97 (σ_q, γ) grid.",
            metrics=(
                MetricSpec("n_sigma_q", lambda m: int(m["n_sigma_q"])),
                MetricSpec("n_gamma", lambda m: int(m["n_gamma"])),
                MetricSpec("n_no_hopf", lambda m: int(m["n_no_hopf"])),
                MetricSpec("n_supercritical", lambda m: int(m["n_supercritical"])),
                MetricSpec("n_bautin_tube", lambda m: int(m["n_bautin_tube"])),
                MetricSpec("n_subcritical", lambda m: int(m["n_subcritical"])),
                MetricSpec(
                    "n_bt_physical_kappa_sn_positive_cells",
                    lambda m: int(m["n_bt_physical_kappa_sn_positive_cells"]),
                ),
                MetricSpec(
                    "bt_locus_empty_in_physical_range",
                    lambda m: bool(m["bt_locus_empty_in_physical_range"]),
                ),
                MetricSpec("kappa_sn_min", lambda m: float(m["kappa_sn_min"])),
                MetricSpec("kappa_sn_max", lambda m: float(m["kappa_sn_max"])),
                # All 6 Bautin anchors (σ_q, γ, κ★) — pin individually.
                MetricSpec(
                    "bautin_anchor_0_sigma_q",
                    lambda m: float(m["bautin_anchors"][0]["sigma_q"]),
                ),
                MetricSpec(
                    "bautin_anchor_0_gamma", lambda m: float(m["bautin_anchors"][0]["gamma"])
                ),
                MetricSpec(
                    "bautin_anchor_0_kappa_star",
                    lambda m: float(m["bautin_anchors"][0]["kappa_star"]),
                ),
                MetricSpec(
                    "bautin_anchor_1_sigma_q",
                    lambda m: float(m["bautin_anchors"][1]["sigma_q"]),
                ),
                MetricSpec(
                    "bautin_anchor_1_gamma", lambda m: float(m["bautin_anchors"][1]["gamma"])
                ),
                MetricSpec(
                    "bautin_anchor_1_kappa_star",
                    lambda m: float(m["bautin_anchors"][1]["kappa_star"]),
                ),
                MetricSpec(
                    "bautin_anchor_2_sigma_q",
                    lambda m: float(m["bautin_anchors"][2]["sigma_q"]),
                ),
                MetricSpec(
                    "bautin_anchor_2_gamma", lambda m: float(m["bautin_anchors"][2]["gamma"])
                ),
                MetricSpec(
                    "bautin_anchor_2_kappa_star",
                    lambda m: float(m["bautin_anchors"][2]["kappa_star"]),
                ),
                MetricSpec(
                    "bautin_anchor_3_sigma_q",
                    lambda m: float(m["bautin_anchors"][3]["sigma_q"]),
                ),
                MetricSpec(
                    "bautin_anchor_3_gamma", lambda m: float(m["bautin_anchors"][3]["gamma"])
                ),
                MetricSpec(
                    "bautin_anchor_3_kappa_star",
                    lambda m: float(m["bautin_anchors"][3]["kappa_star"]),
                ),
                MetricSpec(
                    "bautin_anchor_4_sigma_q",
                    lambda m: float(m["bautin_anchors"][4]["sigma_q"]),
                ),
                MetricSpec(
                    "bautin_anchor_4_gamma", lambda m: float(m["bautin_anchors"][4]["gamma"])
                ),
                MetricSpec(
                    "bautin_anchor_4_kappa_star",
                    lambda m: float(m["bautin_anchors"][4]["kappa_star"]),
                ),
                MetricSpec(
                    "bautin_anchor_5_sigma_q",
                    lambda m: float(m["bautin_anchors"][5]["sigma_q"]),
                ),
                MetricSpec(
                    "bautin_anchor_5_gamma", lambda m: float(m["bautin_anchors"][5]["gamma"])
                ),
                MetricSpec(
                    "bautin_anchor_5_kappa_star",
                    lambda m: float(m["bautin_anchors"][5]["kappa_star"]),
                ),
            ),
        ),
        WaveExperimentSpec(
            name="mckean_vlasov_validation",
            runner=_run_mckean_vlasov,
            config_dict=_config_mckean_vlasov(),
            tolerance_class="stochastic_relative",
            seed=20260514,
            notes="Particle SDE at locked seed; production-mode sweep (5 n-points {10,32,100,316,1000}, 64 reps, 250 steps).",
            metrics=(
                MetricSpec("kappa_star_single", lambda m: float(m["kappa_star_single"])),
                MetricSpec("kappa_star_mv", lambda m: float(m["kappa_star_mv"])),
                MetricSpec("kappa_star_shift_ratio", lambda m: float(m["kappa_star_shift_ratio"])),
                MetricSpec("omega_star", lambda m: float(m["omega_star"])),
                MetricSpec("theta_G", lambda m: float(m["theta_G"])),
                MetricSpec("tau_G_years", lambda m: float(m["tau_G_years"])),
                MetricSpec("tau_G_trading_days", lambda m: float(m["tau_G_trading_days"])),
                MetricSpec("C_T_theoretical", lambda m: float(m["C_T_theoretical"])),
                MetricSpec("rmse_sup_n10", lambda m: float(m["rmse_sup"][0])),
                MetricSpec("rmse_sup_n32", lambda m: float(m["rmse_sup"][1])),
                MetricSpec("rmse_sup_n100", lambda m: float(m["rmse_sup"][2])),
                MetricSpec("rmse_sup_n316", lambda m: float(m["rmse_sup"][3])),
                MetricSpec("rmse_sup_n1000", lambda m: float(m["rmse_sup"][4])),
                MetricSpec(
                    "fitted_slope_log_inv_sqrt_n",
                    lambda m: float(m["fitted_slope_log_inv_sqrt_n"]),
                ),
                MetricSpec("fitted_slope_log_n", lambda m: float(m["fitted_slope_log_n"])),
                MetricSpec(
                    "fitted_intercept_log_inv_sqrt_n",
                    lambda m: float(m["fitted_intercept_log_inv_sqrt_n"]),
                ),
            ),
        ),
        WaveExperimentSpec(
            name="kappa_star_robustness",
            runner=_run_kappa_star_robustness,
            config_dict=_config_kappa_star_robustness(),
            tolerance_class="deterministic_exact",
            seed=0,  # closed-form implicit differentiation
            notes="Closed-form analytical elasticities + heatmap + misspec curve.",
            metrics=(
                MetricSpec("kappa_star_canonical", lambda m: float(m["kappa_star_canonical"])),
                MetricSpec("omega_star_canonical", lambda m: float(m["omega_star_canonical"])),
                MetricSpec("G_y_canonical", lambda m: float(m["G_y_canonical"])),
                MetricSpec("G_v_canonical", lambda m: float(m["G_v_canonical"])),
                MetricSpec("elasticity_mu_q", lambda m: float(m["sensitivity"]["elasticity_mu_q"])),
                MetricSpec(
                    "elasticity_sigma_q",
                    lambda m: float(m["sensitivity"]["elasticity_sigma_q"]),
                ),
                MetricSpec("dkappa_dmu_q", lambda m: float(m["sensitivity"]["dkappa_dmu_q"])),
                MetricSpec("dkappa_dsigma_q", lambda m: float(m["sensitivity"]["dkappa_dsigma_q"])),
                MetricSpec("dkappa_dGy", lambda m: float(m["sensitivity"]["dkappa_dGy"])),
                MetricSpec("dkappa_dGv", lambda m: float(m["sensitivity"]["dkappa_dGv"])),
                MetricSpec(
                    "rel_dev_at_10pct_sigma_q_only",
                    lambda m: float(m["heatmap"]["rel_dev_at_10pct_sigma_q_only"]),
                ),
                MetricSpec(
                    "rel_dev_at_30pct_sigma_q_only",
                    lambda m: float(m["heatmap"]["rel_dev_at_30pct_sigma_q_only"]),
                ),
                MetricSpec(
                    "misspec_relative_error_first",
                    lambda m: float(m["misspec"][0]["relative_error"]),
                ),
                MetricSpec(
                    "misspec_relative_error_last",
                    lambda m: float(m["misspec"][-1]["relative_error"]),
                ),
            ),
        ),
        WaveExperimentSpec(
            name="h_bimod_2d_scan",
            runner=_run_h_bimod_2d,
            config_dict=_config_h_bimod_2d(),
            tolerance_class="stochastic_relative",
            seed=42,
            notes=(
                "Reflexive SDE at γ > 0; PCA-projected Hartigan dip + Silverman bandwidth tests. "
                "Locked config: n_paths=1000, n_steps=2000 (paper's headline n=15,769 cell)."
            ),
            metrics=(
                MetricSpec("kappa_star_envelope", lambda m: float(m["kappa_star_envelope"])),
                MetricSpec(
                    "n_samples_at_kappa_1_05_envelope",
                    lambda m: int(m["outcomes"][4]["n_samples"]),
                ),
                MetricSpec(
                    "pca_dip_p_value_at_kappa_1_05_envelope",
                    lambda m: float(m["outcomes"][4]["pca_dip_p_value"]),
                ),
                MetricSpec(
                    "pca_dip_statistic_at_kappa_1_05_envelope",
                    lambda m: float(m["outcomes"][4]["pca_dip_statistic"]),
                ),
                MetricSpec(
                    "pca_dip_p_value_at_kappa_0",
                    lambda m: float(m["outcomes"][0]["pca_dip_p_value"]),
                ),
                MetricSpec(
                    "pca_dip_p_value_at_kappa_star",
                    lambda m: float(m["outcomes"][3]["pca_dip_p_value"]),
                ),
                MetricSpec(
                    "pca_explained_variance_at_kappa_1_05_envelope",
                    lambda m: float(m["outcomes"][4]["pca_explained_variance_ratio"]),
                ),
                MetricSpec("any_pca_bimodal", lambda m: bool(m["any_pca_bimodal"])),
            ),
        ),
        WaveExperimentSpec(
            name="h1_synthetic_validation",
            runner=_run_h1_synthetic,
            config_dict=_config_h1_synthetic(),
            tolerance_class="stochastic_relative",
            seed=42,
            notes=(
                "BC-trained MLP + reflexive/Heston rollouts + sliced-W2 with block bootstrap. "
                "Locked config: n_bc_train_episodes=30, n_paths_per_source=30 "
                "(paper's headline SW2 = 0.005/0.034/0.054 ordering)."
            ),
            metrics=(
                MetricSpec(
                    "sw2_source_a",
                    lambda m: float(m["sw2"]["source_a_kappa0_deployed"]["distance"]),
                ),
                MetricSpec(
                    "sw2_source_b", lambda m: float(m["sw2"]["source_b_2kappa0"]["distance"])
                ),
                MetricSpec(
                    "sw2_source_c", lambda m: float(m["sw2"]["source_c_heston"]["distance"])
                ),
                MetricSpec(
                    "sw2_source_a_ci_low",
                    lambda m: float(m["sw2"]["source_a_kappa0_deployed"]["ci_low"]),
                ),
                MetricSpec(
                    "sw2_source_a_ci_high",
                    lambda m: float(m["sw2"]["source_a_kappa0_deployed"]["ci_high"]),
                ),
                MetricSpec(
                    "sw2_source_b_ci_low", lambda m: float(m["sw2"]["source_b_2kappa0"]["ci_low"])
                ),
                MetricSpec(
                    "sw2_source_b_ci_high",
                    lambda m: float(m["sw2"]["source_b_2kappa0"]["ci_high"]),
                ),
                MetricSpec(
                    "sw2_source_c_ci_low", lambda m: float(m["sw2"]["source_c_heston"]["ci_low"])
                ),
                MetricSpec(
                    "sw2_source_c_ci_high",
                    lambda m: float(m["sw2"]["source_c_heston"]["ci_high"]),
                ),
                MetricSpec(
                    "n_windows_kept_source_b",
                    lambda m: int(m["n_windows_kept"]["source_b"]),
                ),
                MetricSpec(
                    "n_windows_kept_source_c",
                    lambda m: int(m["n_windows_kept"]["source_c"]),
                ),
                MetricSpec("ordering_holds", lambda m: bool(m["ordering_holds"])),
                MetricSpec("pass_protocol", lambda m: bool(m["pass_protocol"])),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """JSON-friendly coercion of numpy scalars and Path objects."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):  # numpy generic
        try:
            return value.item()
        except (AttributeError, TypeError):
            return value
    if isinstance(value, Path):
        return str(value)
    return value


def _tolerance_for(spec: WaveExperimentSpec, metric_value: Any) -> dict[str, Any]:
    """Per-metric tolerance dict reflecting the spec's tolerance_class.

    Bools and ints lock at exact match (tolerance 0). Floats lock per the
    spec's class.
    """
    if isinstance(metric_value, bool):
        return {"tolerance_abs": 0.0, "tolerance_kind": "exact"}
    if isinstance(metric_value, int):
        return {"tolerance_abs": 0.0, "tolerance_kind": "exact"}
    if spec.tolerance_class == "deterministic_exact":
        return {"tolerance_abs": DETERMINISTIC_ABS_TOL, "tolerance_kind": "absolute"}
    if spec.tolerance_class == "stochastic_relative":
        return {"tolerance_relative": STOCHASTIC_REL_TOL, "tolerance_kind": "relative"}
    raise ValueError(f"unknown tolerance_class: {spec.tolerance_class!r}")


def _build_experiment_entry(spec: WaveExperimentSpec) -> dict[str, Any]:
    """Run the experiment and assemble its receipt entry."""
    raw = spec.runner()
    metrics: dict[str, Any] = {}
    for ms in spec.metrics:
        try:
            value = _coerce(ms.extractor(raw))
        except (KeyError, IndexError, TypeError) as err:
            raise RuntimeError(
                f"metric extraction failed for {spec.name}.{ms.name}: {err!r}"
            ) from err
        metrics[ms.name] = {"value": value, **_tolerance_for(spec, value)}
    return {
        "tolerance_class": spec.tolerance_class,
        "seed": spec.seed,
        "config": spec.config_dict,
        "notes": spec.notes,
        "metrics": metrics,
    }


def _build_receipt() -> dict[str, Any]:
    tooling = _tooling_versions()
    experiments: dict[str, Any] = {}
    for spec in _all_specs():
        t0 = time.perf_counter()
        entry = _build_experiment_entry(spec)
        elapsed = time.perf_counter() - t0
        print(f"  {spec.name:<28s}  {spec.tolerance_class:<22s}  {elapsed:7.2f}s")
        experiments[spec.name] = entry
    return {
        "schema_version": SCHEMA_VERSION,
        "version": RECEIPT_VERSION,
        "commit": _git_commit(),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tooling": tooling,
        "deterministic_abs_tolerance": DETERMINISTIC_ABS_TOL,
        "stochastic_rel_tolerance": STOCHASTIC_REL_TOL,
        "n_experiments": len(experiments),
        "experiments": experiments,
    }


def main() -> None:
    """Run every Wave 1–6 experiment and snapshot results to baseline_v0.3.3.json."""
    print(f"Generating v0.3.3 reproducibility receipt → {RECEIPT_PATH}")
    print(f"deterministic_abs_tol={DETERMINISTIC_ABS_TOL}; stochastic_rel_tol={STOCHASTIC_REL_TOL}")
    print()
    print(f"{'experiment':<30s}  {'tolerance_class':<22s}  elapsed")
    print(f"{'-' * 30}  {'-' * 22}  -------")

    t0 = time.perf_counter()
    receipt = _build_receipt()
    total = time.perf_counter() - t0

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    shutil.move(str(tmp), str(RECEIPT_PATH))
    size = RECEIPT_PATH.stat().st_size
    print()
    print(f"  total: {total:.2f}s  →  {RECEIPT_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
