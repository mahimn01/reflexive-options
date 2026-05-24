"""Analytical results — the novel data-free theoretical contributions of the paper.

- `bifurcation`: Hopf bifurcation analysis of the reflexive SDE.
- `hawkes_equivalence`: Theorem 2 — mapping the Hawkes branching ratio
  n (Hardiman 2013) to the SV-Jacobian leading eigenvalue / Hopf
  threshold κ★ via the Bacry-Delattre-Hoffmann-Muzy (2013) diffusive
  limit.
- `stationary`: Fokker-Planck stationary marginal density.
- `sensitivity`: numerical ∂(metric)/∂κ pipeline.
- `spectral`: H4 PSD-peak detector for the Hopf signature in |r_t|.
- `inference`: shared statistical primitives — block bootstrap (V4-B3),
  BH-FDR (V4-B4), TOST equivalence (V3-B2 / A7).
- `info_theoretic`: Theorem 5 — closed-form excess entropy of the linearised
  3D OU and IAAFT-calibrated empirical transfer entropy for the dealer-gamma
  channel.
"""

from reflexive_options.theory.bifurcation import (
    BautinScanResult,
    G_lognormal_oi,
    G_lognormal_oi_partials,
    G_mixture_lognormal_oi,
    G_mixture_lognormal_oi_partials,
    HopfScanResult,
    MixtureOIComponent,
    bautin_curve_scan,
    bogdanov_takens_residual_lognormal_oi,
    build_bilinear_trilinear_tensors,
    compute_lambda_correction,
    compute_lyapunov_coefficient,
    find_bautin_anchors,
    hopf_scan,
    jacobian_eigenvalues,
    kappa_saddle_node_lognormal_oi,
    kappa_star_lognormal_oi,
    kappa_star_mixture_lognormal_oi,
    lyapunov_coefficient_lognormal_oi,
    lyapunov_coefficient_mixture_lognormal_oi,
    stochastic_hopf_shift_numeric,
    top_lyapunov_exponent_linearised,
)
from reflexive_options.theory.hawkes_equivalence import (
    HawkesEquivalenceResult,
    hawkes_branching_ratio_curve,
    n_sv_at_kappa,
    n_sv_from_eigenvalues,
)
from reflexive_options.theory.inference import (
    benjamini_hochberg,
    block_bootstrap_ci,
    stationary_block_bootstrap,
    tost_equivalence,
)
from reflexive_options.theory.info_theoretic import (
    CriticalExponentFit,
    ExcessEntropyCurveResult,
    TransferEntropyIAAFTResult,
    excess_entropy_curve,
    excess_entropy_linear,
    fit_critical_exponent,
    transfer_entropy_iaaft_pvalue,
    transfer_entropy_simulated,
)
from reflexive_options.theory.mckean_vlasov import (
    ChaosErrorResult,
    ChaosScalingResult,
    mckean_vlasov_kappa_star_shift,
    mean_field_limit_trajectory,
    propagation_of_chaos_constant,
    propagation_of_chaos_error,
    propagation_of_chaos_scaling,
    simulate_n_dealer_system,
)
from reflexive_options.theory.robustness import (
    KappaStarSensitivityResult,
    MisspecificationError,
    calibration_tolerance,
    fit_lognormal_to_mixture_moments,
    kappa_star_brute_force_from_G,
    kappa_star_misspecification_error,
    kappa_star_sensitivity_lognormal_oi,
    make_mixture_lognormal_density,
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
    "BautinScanResult",
    "BimodalityResult",
    "ChaosErrorResult",
    "ChaosScalingResult",
    "CriticalExponentFit",
    "ExcessEntropyCurveResult",
    "G_lognormal_oi",
    "G_lognormal_oi_partials",
    "G_mixture_lognormal_oi",
    "G_mixture_lognormal_oi_partials",
    "HawkesEquivalenceResult",
    "HopfScanResult",
    "KappaStarSensitivityResult",
    "MisspecificationError",
    "MixtureOIComponent",
    "PSDPeakResult",
    "SensitivityResult",
    "StationaryDensity",
    "TailIndexCurve",
    "TransferEntropyIAAFTResult",
    "adaptive_welch_nperseg",
    "bautin_curve_scan",
    "benjamini_hochberg",
    "block_bootstrap_ci",
    "bogdanov_takens_residual_lognormal_oi",
    "build_bilinear_trilinear_tensors",
    "calibration_tolerance",
    "compare_to_heston",
    "compute_lambda_correction",
    "compute_lyapunov_coefficient",
    "detect_bimodality",
    "detect_psd_peak",
    "excess_entropy_curve",
    "excess_entropy_linear",
    "find_bautin_anchors",
    "fit_critical_exponent",
    "fit_lognormal_to_mixture_moments",
    "hawkes_branching_ratio_curve",
    "heston_log_return_cdf",
    "heston_log_return_quantiles",
    "heston_stationary_variance_density",
    "hopf_scan",
    "iaaft_surrogate",
    "jacobian_eigenvalues",
    "kappa_saddle_node_lognormal_oi",
    "kappa_sensitivity_curve",
    "kappa_star_brute_force_from_G",
    "kappa_star_lognormal_oi",
    "kappa_star_misspecification_error",
    "kappa_star_mixture_lognormal_oi",
    "kappa_star_sensitivity_lognormal_oi",
    "lyapunov_coefficient_lognormal_oi",
    "lyapunov_coefficient_mixture_lognormal_oi",
    "make_mixture_lognormal_density",
    "mckean_vlasov_kappa_star_shift",
    "mean_field_limit_trajectory",
    "n_sv_at_kappa",
    "n_sv_from_eigenvalues",
    "propagation_of_chaos_constant",
    "propagation_of_chaos_error",
    "propagation_of_chaos_scaling",
    "simulate_n_dealer_system",
    "solve_stationary",
    "stationary_block_bootstrap",
    "stochastic_hopf_shift_numeric",
    "tail_index_vs_kappa_curve",
    "top_lyapunov_exponent_linearised",
    "tost_equivalence",
    "transfer_entropy_iaaft_pvalue",
    "transfer_entropy_simulated",
]
