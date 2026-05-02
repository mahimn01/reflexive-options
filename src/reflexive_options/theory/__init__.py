"""Analytical results — the novel data-free theoretical contributions of the paper.

- `bifurcation`: Hopf bifurcation analysis of the reflexive SDE.
- `stationary`: Fokker-Planck stationary marginal density.
- `sensitivity`: numerical ∂(metric)/∂κ pipeline.
- `spectral`: H4 PSD-peak detector for the Hopf signature in |r_t|.
- `inference`: shared statistical primitives — block bootstrap (V4-B3),
  BH-FDR (V4-B4), TOST equivalence (V3-B2 / A7).
"""

from reflexive_options.theory.bifurcation import (
    G_lognormal_oi,
    G_lognormal_oi_partials,
    HopfScanResult,
    build_bilinear_trilinear_tensors,
    compute_lambda_correction,
    compute_lyapunov_coefficient,
    hopf_scan,
    jacobian_eigenvalues,
    kappa_star_lognormal_oi,
    lyapunov_coefficient_lognormal_oi,
    stochastic_hopf_shift_numeric,
    top_lyapunov_exponent_linearised,
)
from reflexive_options.theory.inference import (
    benjamini_hochberg,
    block_bootstrap_ci,
    stationary_block_bootstrap,
    tost_equivalence,
)
from reflexive_options.theory.sensitivity import (
    SensitivityResult,
    kappa_sensitivity_curve,
)
from reflexive_options.theory.spectral import (
    PSDPeakResult,
    adaptive_welch_nperseg,
    detect_psd_peak,
    iaaft_surrogate,
)
from reflexive_options.theory.stationary import (
    BimodalityResult,
    StationaryDensity,
    TailIndexCurve,
    compare_to_heston,
    detect_bimodality,
    heston_log_return_cdf,
    heston_log_return_quantiles,
    heston_stationary_variance_density,
    solve_stationary,
    tail_index_vs_kappa_curve,
)

__all__ = [
    "BimodalityResult",
    "G_lognormal_oi",
    "G_lognormal_oi_partials",
    "HopfScanResult",
    "PSDPeakResult",
    "SensitivityResult",
    "StationaryDensity",
    "TailIndexCurve",
    "adaptive_welch_nperseg",
    "benjamini_hochberg",
    "block_bootstrap_ci",
    "build_bilinear_trilinear_tensors",
    "compare_to_heston",
    "compute_lambda_correction",
    "compute_lyapunov_coefficient",
    "detect_bimodality",
    "detect_psd_peak",
    "heston_log_return_cdf",
    "heston_log_return_quantiles",
    "heston_stationary_variance_density",
    "hopf_scan",
    "iaaft_surrogate",
    "jacobian_eigenvalues",
    "kappa_sensitivity_curve",
    "kappa_star_lognormal_oi",
    "lyapunov_coefficient_lognormal_oi",
    "solve_stationary",
    "stationary_block_bootstrap",
    "stochastic_hopf_shift_numeric",
    "tail_index_vs_kappa_curve",
    "top_lyapunov_exponent_linearised",
    "tost_equivalence",
]
