"""Static-arbitrage filter for simulator-generated IV surfaces.

Implements the four-check Roper / Gatheral-Jacquier characterisation:
    1. Butterfly  -- convexity of call price in K (unequal-spacing 2nd diff).
    2. Calendar   -- monotonicity of *undiscounted* forward call in T.
    3. Calendar in total variance -- non-decrease of w(k, T) on the intersection
       log-moneyness grid (strictly stronger than #2 in the overlap).
    4. Lee wing slope -- OLS slope of w against |k| in [-eps, 2 + eps].

See ../../../../reflexivity-research/arbitrage_filter_brief.md for the spec.

All checks are vectorised over the leading batch axis for the 50k-surface use case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from reflexive_options.types import SurfaceArray, SurfaceGrid

# ---------------------------------------------------------------------------
# Tolerances + result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitrageTolerances:
    """Numerical tolerances for the four arbitrage checks.

    Defaults from arbitrage_filter_brief.md §4. `*_abs` for butterfly is
    multiplied by F internally; calendar `*_abs` similarly. `lee_slope` is
    additive on both sides of the [0, 2] Lee bound.
    """

    butterfly_abs: float = 1e-8
    butterfly_rel: float = 1e-6
    calendar_abs: float = 1e-8
    calendar_rel: float = 1e-6
    w_abs: float = 1e-8
    w_rel: float = 1e-6
    lee_slope: float = 0.05
    warn_factor: float = 10.0


CheckName = str


@dataclass(frozen=True)
class ArbitrageCheck:
    """Outcome of an arbitrage-free check on one surface."""

    passes_all: bool
    is_marginal: bool
    violations: dict[CheckName, list[tuple[int, int]]] = field(default_factory=dict)
    severity: dict[CheckName, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Black-76 vectorised pricer
# ---------------------------------------------------------------------------


def _black76_call(
    F: NDArray[np.float64],
    K: NDArray[np.float64],
    T: NDArray[np.float64],
    sigma: NDArray[np.float64],
    D: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Discounted Black-76 call. All inputs broadcast-compatible.

    Switches to intrinsic when sigma*sqrt(T) < 1e-8 to avoid 0/0 in d1.
    """
    sqrtT = np.sqrt(T)
    vol_t = sigma * sqrtT
    safe = vol_t > 1e-8
    safe_vol = np.where(safe, vol_t, 1.0)
    log_FK = np.log(F / K)
    d1 = np.where(safe, (log_FK + 0.5 * vol_t**2) / safe_vol, 0.0)
    d2 = d1 - vol_t
    price = D * (F * ndtr(d1) - K * ndtr(d2))
    intrinsic = D * np.maximum(F - K, 0.0)
    return np.where(safe, price, intrinsic)


# ---------------------------------------------------------------------------
# Grid resolution
# ---------------------------------------------------------------------------


def _strikes_from_grid(grid: SurfaceGrid, spot: float) -> NDArray[np.float64]:
    """K = spot * exp(log_moneyness)."""
    return spot * np.exp(grid.log_moneyness)


def _forwards(
    spot: float, rate: float, dividend: float, T: NDArray[np.float64]
) -> NDArray[np.float64]:
    return spot * np.exp((rate - dividend) * T)


def _discounts(rate: float, T: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-rate * T)


# ---------------------------------------------------------------------------
# Single-surface checker
# ---------------------------------------------------------------------------


def check_arbitrage_free(
    iv_surface: SurfaceArray,
    grid: SurfaceGrid,
    *,
    spot: float,
    rate: float,
    dividend: float,
    tolerances: ArbitrageTolerances | None = None,
) -> ArbitrageCheck:
    """Run the four checks on one surface; return classified ArbitrageCheck."""
    tol = tolerances if tolerances is not None else ArbitrageTolerances()
    if iv_surface.shape != grid.shape:
        raise ValueError(
            f"iv_surface shape {iv_surface.shape} does not match grid shape {grid.shape}"
        )

    # 0. validity
    if not np.isfinite(iv_surface).all() or (iv_surface <= 0).any():
        return ArbitrageCheck(
            passes_all=False,
            is_marginal=False,
            violations={"input_validity": [(-1, -1)]},
            severity={"input_validity": float("inf")},
            notes=["non-finite or non-positive IV in surface"],
        )

    K = _strikes_from_grid(grid, spot)
    T = grid.maturities
    F = _forwards(spot, rate, dividend, T)
    D = _discounts(rate, T)

    # Batch dimension of 1 for shared kernel.
    iv_b = iv_surface[None, ...]
    K_b = K[None, :, None]
    T_b = T[None, None, :]
    F_b = F[None, None, :]
    D_b = D[None, None, :]

    C = _black76_call(F_b, K_b, T_b, iv_b, D_b)[0]  # (n_K, n_T)

    violations: dict[str, list[tuple[int, int]]] = {}
    severity: dict[str, float] = {}
    notes: list[str] = []
    margin_flags: list[bool] = []
    reject_flags: list[bool] = []

    # ---- Check #1: butterfly ----
    _check_butterfly(C, K, F, tol, violations, severity, margin_flags, reject_flags)

    # ---- Check #2: calendar in undiscounted call ----
    _check_calendar_call(C, D, F, tol, violations, severity, margin_flags, reject_flags)

    # ---- Check #3: calendar in total variance ----
    _check_calendar_w(
        iv_surface, K, T, F, tol, violations, severity, margin_flags, reject_flags, notes
    )

    # ---- Check #4: Lee wings ----
    _check_lee(iv_surface, K, T, F, tol, violations, severity, margin_flags, reject_flags, notes)

    # §6 classification: passes_all True iff no warn or reject; is_marginal flags warn-tier.
    is_marginal = any(margin_flags)
    passes_all = not any(reject_flags) and not is_marginal

    return ArbitrageCheck(
        passes_all=passes_all,
        is_marginal=is_marginal,
        violations=violations,
        severity=severity,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Individual check helpers (work on (n_K, n_T) for the single-surface API)
# ---------------------------------------------------------------------------


def _classify(
    max_sev: float,
    base_tol: float,
    warn_factor: float,
    margin_flags: list[bool],
    reject_flags: list[bool],
) -> None:
    if max_sev <= base_tol:
        return
    if max_sev > base_tol * warn_factor:
        reject_flags.append(True)
    else:
        margin_flags.append(True)


def _check_butterfly(
    C: NDArray[np.float64],
    K: NDArray[np.float64],
    F: NDArray[np.float64],
    tol: ArbitrageTolerances,
    violations: dict[str, list[tuple[int, int]]],
    severity: dict[str, float],
    margin_flags: list[bool],
    reject_flags: list[bool],
) -> None:
    n_K = C.shape[0]
    if n_K < 3:
        return
    h_minus = (K[1:-1] - K[:-2])[:, None]
    h_plus = (K[2:] - K[1:-1])[:, None]
    D2C = (C[2:] - C[1:-1]) / h_plus - (C[1:-1] - C[:-2]) / h_minus  # (n_K-2, n_T)
    nbhd_max = np.maximum(C[2:], np.maximum(C[1:-1], C[:-2]))
    Frow = F[None, :]
    deep_otm = nbhd_max < 1e-8 * Frow
    bf_floor = -tol.butterfly_abs * Frow - tol.butterfly_rel * np.maximum(nbhd_max, 1e-12)
    bf_viol = (bf_floor > D2C) & ~deep_otm
    if bf_viol.any():
        sev = -D2C  # how negative the second diff is
        max_sev = float(sev[bf_viol].max())
        idx = np.argwhere(bf_viol)
        violations["butterfly"] = [(int(i + 1), int(j)) for i, j in idx]
        severity["butterfly"] = max_sev
        # tolerance scale (use a representative F * abs)
        base = tol.butterfly_abs * float(F.mean()) + tol.butterfly_rel
        _classify(max_sev, base, tol.warn_factor, margin_flags, reject_flags)


def _check_calendar_call(
    C: NDArray[np.float64],
    D: NDArray[np.float64],
    F: NDArray[np.float64],
    tol: ArbitrageTolerances,
    violations: dict[str, list[tuple[int, int]]],
    severity: dict[str, float],
    margin_flags: list[bool],
    reject_flags: list[bool],
) -> None:
    n_T = C.shape[1]
    if n_T < 2:
        return
    C_undisc = C / D[None, :]
    dC = C_undisc[:, 1:] - C_undisc[:, :-1]
    cal_floor = -tol.calendar_abs * F[None, 1:] - tol.calendar_rel * np.abs(C_undisc[:, :-1])
    cal_viol = dC < cal_floor
    if cal_viol.any():
        sev = -dC
        max_sev = float(sev[cal_viol].max())
        idx = np.argwhere(cal_viol)
        violations["calendar_call"] = [(int(i), int(j)) for i, j in idx]
        severity["calendar_call"] = max_sev
        base = tol.calendar_abs * float(F.mean()) + tol.calendar_rel
        _classify(max_sev, base, tol.warn_factor, margin_flags, reject_flags)


def _check_calendar_w(
    iv: NDArray[np.float64],
    K: NDArray[np.float64],
    T: NDArray[np.float64],
    F: NDArray[np.float64],
    tol: ArbitrageTolerances,
    violations: dict[str, list[tuple[int, int]]],
    severity: dict[str, float],
    margin_flags: list[bool],
    reject_flags: list[bool],
    notes: list[str],
) -> None:
    n_K, n_T = iv.shape
    if n_T < 2 or n_K < 2:
        return
    # Per-maturity log-moneyness columns.
    k = np.log(K[:, None] / F[None, :])  # (n_K, n_T)
    k_min = float(k.min(axis=0).max())
    k_max = float(k.max(axis=0).min())
    if k_max <= k_min:
        notes.append("calendar_total_variance: empty intersection k-grid; skipped")
        return
    M = max(n_K, 25)
    kg = np.linspace(k_min, k_max, M)
    w_native = (iv**2) * T[None, :]
    w_interp = np.empty((M, n_T), dtype=np.float64)
    for j in range(n_T):
        w_interp[:, j] = np.interp(kg, k[:, j], w_native[:, j])
    dw = w_interp[:, 1:] - w_interp[:, :-1]
    w_floor = -tol.w_abs - tol.w_rel * np.abs(w_interp[:, :-1])
    w_viol = dw < w_floor
    if w_viol.any():
        sev = -dw
        max_sev = float(sev[w_viol].max())
        # Map back to nearest native (i, j).
        idx_pairs = np.argwhere(w_viol)
        native_idx: list[tuple[int, int]] = []
        for m, j in idx_pairs:
            # Native i = closest strike in column j to kg[m]
            i = int(np.argmin(np.abs(k[:, int(j)] - kg[int(m)])))
            native_idx.append((i, int(j)))
        # de-dup
        violations["calendar_total_variance"] = sorted(set(native_idx))
        severity["calendar_total_variance"] = max_sev
        base = tol.w_abs + tol.w_rel
        _classify(max_sev, base, tol.warn_factor, margin_flags, reject_flags)


def _r2(x: NDArray[np.float64], y: NDArray[np.float64], slope: float, intercept: float) -> float:
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-30:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _check_lee(
    iv: NDArray[np.float64],
    K: NDArray[np.float64],
    T: NDArray[np.float64],
    F: NDArray[np.float64],
    tol: ArbitrageTolerances,
    violations: dict[str, list[tuple[int, int]]],
    severity: dict[str, float],
    margin_flags: list[bool],
    reject_flags: list[bool],
    notes: list[str],
) -> None:
    n_K, n_T = iv.shape
    n_wing = max(3, n_K // 10)
    k = np.log(K[:, None] / F[None, :])  # (n_K, n_T)
    for j in range(n_T):
        if n_K < 7:
            notes.append(f"maturity {j}: too few strikes ({n_K}) for Lee check")
            continue
        kj = k[:, j]
        wj = (iv[:, j] ** 2) * T[j]
        for side, idx in (("left", slice(0, n_wing)), ("right", slice(n_K - n_wing, n_K))):
            x = np.abs(kj[idx])
            y = wj[idx]
            if x.std() < 1e-12:
                notes.append(f"maturity {j} {side}: degenerate wing |k|, skipped")
                continue
            slope, intercept = np.polyfit(x, y, 1)
            r2 = _r2(x, y, float(slope), float(intercept))
            slope_tol = tol.lee_slope if r2 >= 0.5 else 0.2
            if r2 < 0.5:
                notes.append(f"maturity {j} {side}: wing OLS R^2={r2:.2f}, widened tolerance")
            lo, hi = -slope_tol, 2.0 + slope_tol
            if not (lo <= slope <= hi):
                key = f"lee_{side}_wing"
                outermost_i = 0 if side == "left" else n_K - 1
                violations.setdefault(key, []).append((outermost_i, j))
                excess = float(slope - np.clip(slope, 0.0, 2.0))
                cur = severity.get(key, 0.0)
                severity[key] = max(cur, abs(excess))
                _classify(abs(excess), tol.lee_slope, tol.warn_factor, margin_flags, reject_flags)


def _lee_batched(
    iv: NDArray[np.float64],
    K: NDArray[np.float64],
    T: NDArray[np.float64],
    F: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Closed-form OLS slope/R^2 for both wings, vectorised across batch and maturity.

    Returns:
        slope_left:  (b, n_T)
        r2_left:     (b, n_T)
        slope_right: (b, n_T)
        r2_right:    (b, n_T)
    """
    n_K = iv.shape[1]
    n_wing = max(3, n_K // 10)
    k = np.log(K[:, None] / F[None, :])  # (n_K, n_T) — shared across batch
    w = (iv**2) * T[None, None, :]  # (b, n_K, n_T)

    def _ols(side: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sl = slice(0, n_wing) if side == "left" else slice(n_K - n_wing, n_K)
        x = np.abs(k[sl, :])  # (n_wing, n_T)
        y = w[:, sl, :]  # (b, n_wing, n_T)
        x_mean = x.mean(axis=0)  # (n_T,)
        y_mean = y.mean(axis=1)  # (b, n_T)
        x_c = x - x_mean[None, :]  # (n_wing, n_T)
        y_c = y - y_mean[:, None, :]  # (b, n_wing, n_T)
        sxx = (x_c**2).sum(axis=0)  # (n_T,)
        sxy = (x_c[None, :, :] * y_c).sum(axis=1)  # (b, n_T)
        slope = np.where(sxx > 1e-30, sxy / np.where(sxx > 1e-30, sxx, 1.0), 0.0)
        intercept = y_mean - slope * x_mean[None, :]
        y_pred = slope[:, None, :] * x[None, :, :] + intercept[:, None, :]
        ss_res = ((y - y_pred) ** 2).sum(axis=1)  # (b, n_T)
        ss_tot = ((y - y_mean[:, None, :]) ** 2).sum(axis=1)
        r2 = np.where(ss_tot > 1e-30, 1.0 - ss_res / np.where(ss_tot > 1e-30, ss_tot, 1.0), 1.0)
        return slope, r2

    sl_left, r2_left = _ols("left")
    sl_right, r2_right = _ols("right")
    return sl_left, r2_left, sl_right, r2_right


# ---------------------------------------------------------------------------
# Batched checker — vectorised across the leading batch dimension
# ---------------------------------------------------------------------------


def batch_check(
    surfaces: NDArray[np.float64],
    grid: SurfaceGrid,
    *,
    spot: float,
    rate: float,
    dividend: float,
    tolerances: ArbitrageTolerances | None = None,
    chunk: int = 5_000,
) -> list[ArbitrageCheck]:
    """Vectorised arbitrage check on (N, n_K, n_T) batch.

    Returns a list of N ArbitrageCheck — one per surface.

    The dominant cost (Black-76 + ndtr) is fully vectorised across the batch.
    Per-surface dict construction is a Python loop, but only over violators.
    """
    tol = tolerances if tolerances is not None else ArbitrageTolerances()
    if surfaces.ndim != 3 or surfaces.shape[1:] != grid.shape:
        raise ValueError(
            f"surfaces shape {surfaces.shape} incompatible with grid shape {grid.shape}"
        )

    N = surfaces.shape[0]
    K = _strikes_from_grid(grid, spot)
    T = grid.maturities
    F = _forwards(spot, rate, dividend, T)
    D = _discounts(rate, T)
    n_K, n_T = grid.shape

    results: list[ArbitrageCheck] = []
    for s in range(0, N, chunk):
        sl = slice(s, min(s + chunk, N))
        iv = surfaces[sl]  # (b, n_K, n_T)
        b = iv.shape[0]

        # Validate per-surface
        invalid_mask = np.asarray(
            ~np.isfinite(iv).all(axis=(1, 2)) | (iv <= 0).any(axis=(1, 2)),
            dtype=np.bool_,
        )

        K_b = K[None, :, None]
        T_b = T[None, None, :]
        F_b = F[None, None, :]
        D_b = D[None, None, :]
        # Replace bad cells with placeholder positive IV so pricing doesn't NaN; we'll override.
        iv_safe = np.where(np.isfinite(iv) & (iv > 0), iv, 0.2)
        C = _black76_call(F_b, K_b, T_b, iv_safe, D_b)  # (b, n_K, n_T)

        # ---- butterfly (vectorised) ----
        bf_present = n_K >= 3
        if bf_present:
            h_minus = (K[1:-1] - K[:-2])[None, :, None]
            h_plus = (K[2:] - K[1:-1])[None, :, None]
            D2C = (C[:, 2:] - C[:, 1:-1]) / h_plus - (C[:, 1:-1] - C[:, :-2]) / h_minus
            nbhd_max = np.maximum(C[:, 2:], np.maximum(C[:, 1:-1], C[:, :-2]))
            Frow = F[None, None, :]
            deep_otm = nbhd_max < 1e-8 * Frow
            bf_floor = -tol.butterfly_abs * Frow - tol.butterfly_rel * np.maximum(nbhd_max, 1e-12)
            bf_viol = (bf_floor > D2C) & ~deep_otm  # (b, n_K-2, n_T)
        # ---- calendar in undiscounted call (vectorised) ----
        cal_present = n_T >= 2
        if cal_present:
            C_undisc = C / D[None, None, :]
            dC = C_undisc[:, :, 1:] - C_undisc[:, :, :-1]
            cal_floor = -tol.calendar_abs * F[None, None, 1:] - tol.calendar_rel * np.abs(
                C_undisc[:, :, :-1]
            )
            cal_viol = dC < cal_floor  # (b, n_K, n_T-1)
        # ---- calendar in total variance (vectorised across batch; per-mat interp shared) ----
        w_present = n_T >= 2 and n_K >= 2
        w_viol: NDArray[np.bool_] | None = None
        dw: NDArray[np.float64] | None = None
        kg: NDArray[np.float64] | None = None
        k_native: NDArray[np.float64] | None = None
        if w_present:
            k_native = np.log(K[:, None] / F[None, :])  # (n_K, n_T) — same across batch
            k_min = float(k_native.min(axis=0).max())
            k_max = float(k_native.max(axis=0).min())
            if k_max > k_min:
                M = max(n_K, 25)
                kg = np.linspace(k_min, k_max, M).astype(np.float64)
                # For each maturity, build linear-interp weights once → matmul over batch.
                W = np.zeros((n_T, M, n_K), dtype=np.float64)
                for j in range(n_T):
                    kj = k_native[:, j]
                    lo = np.clip(np.searchsorted(kj, kg, side="right") - 1, 0, n_K - 2)
                    kj_lo = kj[lo]
                    kj_hi = kj[lo + 1]
                    denom = np.where(kj_hi > kj_lo, kj_hi - kj_lo, 1.0)
                    alpha = np.clip((kg - kj_lo) / denom, 0.0, 1.0)
                    rows = np.arange(M)
                    W[j, rows, lo] = 1.0 - alpha
                    W[j, rows, lo + 1] = alpha
                w_native_b = (iv_safe**2) * T[None, None, :]  # (b, n_K, n_T)
                w_interp = np.einsum("jmk,bkj->bmj", W, w_native_b)
                dw = w_interp[:, :, 1:] - w_interp[:, :, :-1]
                w_floor = -tol.w_abs - tol.w_rel * np.abs(w_interp[:, :, :-1])
                w_viol = dw < w_floor  # (b, M, n_T-1)
            else:
                kg = None
                k_native = None

        # ---- Lee wings (batch-vectorised) ----
        lee_active = n_K >= 7
        viol_left: NDArray[np.bool_] | None = None
        viol_right: NDArray[np.bool_] | None = None
        excess_left: NDArray[np.float64] | None = None
        excess_right: NDArray[np.float64] | None = None
        if lee_active:
            sl_left, r2_left, sl_right, r2_right = _lee_batched(iv_safe, K, T, F)
            tol_left = np.where(r2_left < 0.5, 0.2, tol.lee_slope)
            tol_right = np.where(r2_right < 0.5, 0.2, tol.lee_slope)
            excess_left = np.abs(sl_left - np.clip(sl_left, 0.0, 2.0))
            excess_right = np.abs(sl_right - np.clip(sl_right, 0.0, 2.0))
            viol_left = (sl_left < -tol_left) | (sl_left > 2.0 + tol_left)
            viol_right = (sl_right < -tol_right) | (sl_right > 2.0 + tol_right)

        # Per-surface assembly.
        for bi in range(b):
            if invalid_mask[bi]:
                results.append(
                    ArbitrageCheck(
                        passes_all=False,
                        is_marginal=False,
                        violations={"input_validity": [(-1, -1)]},
                        severity={"input_validity": float("inf")},
                        notes=["non-finite or non-positive IV"],
                    )
                )
                continue

            violations: dict[str, list[tuple[int, int]]] = {}
            severity: dict[str, float] = {}
            notes: list[str] = []
            margin_flags: list[bool] = []
            reject_flags: list[bool] = []

            if bf_present:
                bv = bf_viol[bi]
                if bv.any():
                    sev = -D2C[bi]
                    max_sev = float(sev[bv].max())
                    idx = np.argwhere(bv)
                    violations["butterfly"] = [(int(i + 1), int(j)) for i, j in idx]
                    severity["butterfly"] = max_sev
                    base = tol.butterfly_abs * float(F.mean()) + tol.butterfly_rel
                    _classify(max_sev, base, tol.warn_factor, margin_flags, reject_flags)

            if cal_present:
                cv = cal_viol[bi]
                if cv.any():
                    sev = -dC[bi]
                    max_sev = float(sev[cv].max())
                    idx = np.argwhere(cv)
                    violations["calendar_call"] = [(int(i), int(j)) for i, j in idx]
                    severity["calendar_call"] = max_sev
                    base = tol.calendar_abs * float(F.mean()) + tol.calendar_rel
                    _classify(max_sev, base, tol.warn_factor, margin_flags, reject_flags)

            if w_viol is not None and kg is not None and k_native is not None:
                assert dw is not None  # for mypy: parallel to w_viol assignment above
                wv = w_viol[bi]
                if wv.any():
                    sev = -dw[bi]
                    max_sev = float(sev[wv].max())
                    idx_pairs = np.argwhere(wv)
                    native_idx: list[tuple[int, int]] = []
                    for m, j in idx_pairs:
                        i = int(np.argmin(np.abs(k_native[:, int(j)] - kg[int(m)])))
                        native_idx.append((i, int(j)))
                    violations["calendar_total_variance"] = sorted(set(native_idx))
                    severity["calendar_total_variance"] = max_sev
                    base = tol.w_abs + tol.w_rel
                    _classify(max_sev, base, tol.warn_factor, margin_flags, reject_flags)

            # ---- Lee (already batched) ----
            if lee_active and viol_left is not None and viol_right is not None:
                assert excess_left is not None and excess_right is not None
                if viol_left[bi].any():
                    j_idx = np.argwhere(viol_left[bi]).ravel()
                    violations["lee_left_wing"] = [(0, int(j)) for j in j_idx]
                    sev_l = float(excess_left[bi].max())
                    severity["lee_left_wing"] = sev_l
                    _classify(sev_l, tol.lee_slope, tol.warn_factor, margin_flags, reject_flags)
                if viol_right[bi].any():
                    j_idx = np.argwhere(viol_right[bi]).ravel()
                    violations["lee_right_wing"] = [(n_K - 1, int(j)) for j in j_idx]
                    sev_r = float(excess_right[bi].max())
                    severity["lee_right_wing"] = sev_r
                    _classify(sev_r, tol.lee_slope, tol.warn_factor, margin_flags, reject_flags)
            elif not lee_active:
                notes.append(f"too few strikes ({n_K}) for Lee check")

            is_marginal = any(margin_flags)
            passes_all = False if any(reject_flags) else not is_marginal
            results.append(
                ArbitrageCheck(
                    passes_all=passes_all,
                    is_marginal=is_marginal,
                    violations=violations,
                    severity=severity,
                    notes=notes,
                )
            )

    return results


__all__: Sequence[str] = (
    "ArbitrageCheck",
    "ArbitrageTolerances",
    "batch_check",
    "check_arbitrage_free",
)
