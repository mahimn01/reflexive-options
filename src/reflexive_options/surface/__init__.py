"""IV surface generation, arbitrage filtering, parquet I/O, and SW2 metric."""

from reflexive_options.surface.arbitrage import ArbitrageCheck, check_arbitrage_free
from reflexive_options.surface.generator import generate_surface, make_pre_reg_grid
from reflexive_options.surface.io import load_surfaces, save_surfaces
from reflexive_options.surface.wasserstein import (
    SlicedW2Result,
    evaluate_sliced_w2_on_surface_windows,
    filter_arbitrage_free_windows,
    make_rolling_windows,
    sliced_wasserstein_2,
)

__all__ = [
    "ArbitrageCheck",
    "SlicedW2Result",
    "check_arbitrage_free",
    "evaluate_sliced_w2_on_surface_windows",
    "filter_arbitrage_free_windows",
    "generate_surface",
    "load_surfaces",
    "make_pre_reg_grid",
    "make_rolling_windows",
    "save_surfaces",
    "sliced_wasserstein_2",
]
