"""Parquet logger for batches of IV surfaces.

Schema (long form):
    sample_id  int32   - index of the surface in the batch
    k_idx      int16   - strike index
    T_idx      int16   - maturity index
    iv         float64 - implied vol
    timestamp  int64   - microseconds since epoch (UTC), one per save call

Metadata is stored in the Parquet schema's key-value metadata as JSON, plus
the array shape so we can round-trip without an out-of-band sidecar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

_META_SHAPE_KEY = b"reflexive_options.shape"
_META_USER_KEY = b"reflexive_options.metadata"
_META_TS_KEY = b"reflexive_options.timestamp_us"


def save_surfaces(
    surfaces: NDArray[np.float64],
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a (N, n_K, n_T) batch to parquet."""
    if surfaces.ndim != 3:
        raise ValueError(f"surfaces must be (N, n_K, n_T), got shape {surfaces.shape}")
    n, n_k, n_t = surfaces.shape
    if not np.isfinite(surfaces).all():
        raise ValueError("surfaces contains non-finite values; refusing to write")

    sample_ids = np.repeat(np.arange(n, dtype=np.int32), n_k * n_t)
    k_idx = np.tile(np.repeat(np.arange(n_k, dtype=np.int16), n_t), n)
    t_idx = np.tile(np.arange(n_t, dtype=np.int16), n * n_k)
    iv_flat = surfaces.reshape(-1).astype(np.float64, copy=False)

    ts_us = int(time.time() * 1_000_000)
    timestamps = np.full(iv_flat.shape, ts_us, dtype=np.int64)

    table = pa.table(
        {
            "sample_id": pa.array(sample_ids, type=pa.int32()),
            "k_idx": pa.array(k_idx, type=pa.int16()),
            "T_idx": pa.array(t_idx, type=pa.int16()),
            "iv": pa.array(iv_flat, type=pa.float64()),
            "timestamp": pa.array(timestamps, type=pa.int64()),
        }
    )
    schema_meta: dict[bytes, bytes] = {
        _META_SHAPE_KEY: json.dumps([n, n_k, n_t]).encode("utf-8"),
        _META_TS_KEY: str(ts_us).encode("utf-8"),
        _META_USER_KEY: json.dumps(metadata or {}).encode("utf-8"),
    }
    table = table.replace_schema_metadata(schema_meta)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # pyarrow 24.0 ships py.typed but the parquet submodule's write_table/
    # read_table are still declared untyped in the bundled .pyi (same root cause
    # as apache/arrow#49831 — dynamic registration). Status as of 2026-04-23:
    # issue is OPEN, no fix released; tracked sub-issue apache/arrow#49194
    # ("Compute module annotations") is also still open. Latest pyarrow on PyPI
    # is 24.0.0 — no 24.1+ release. Maintainers acknowledge shipping py.typed
    # ahead of complete stubs may have been premature (see jorisvandenbossche
    # comment 2026-04-23). Remove these ignores once #49831 closes.
    pq.write_table(table, path, compression="snappy")  # type: ignore[no-untyped-call]


def load_surfaces(path: Path) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Read a parquet written by `save_surfaces`. Returns (surfaces, metadata)."""
    path = Path(path)
    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    schema_meta = table.schema.metadata or {}
    if _META_SHAPE_KEY not in schema_meta:
        raise ValueError(f"{path} missing reflexive_options.shape metadata")
    shape = tuple(json.loads(schema_meta[_META_SHAPE_KEY].decode("utf-8")))
    n, n_k, n_t = shape

    iv_col = table.column("iv").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    if iv_col.size != n * n_k * n_t:
        raise ValueError(f"row count {iv_col.size} does not match metadata shape {shape}")
    surfaces = iv_col.reshape(n, n_k, n_t)

    user_meta_raw = schema_meta.get(_META_USER_KEY, b"{}")
    user_meta: dict[str, Any] = json.loads(user_meta_raw.decode("utf-8"))
    return surfaces, user_meta


__all__ = ["load_surfaces", "save_surfaces"]
