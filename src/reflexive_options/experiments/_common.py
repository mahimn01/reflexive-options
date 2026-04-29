"""Shared utilities for experiment runners — paths, seeding, result logging."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def make_run_dir(experiment_name: str, *, seed: int | None = None) -> Path:
    """Create a timestamped run directory under runs/<experiment>/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_seed{seed}" if seed is not None else ""
    run_dir = RUNS_DIR / experiment_name / f"{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: Any) -> None:
    """Persist a dataclass or dict as config.json next to the run results."""
    payload = asdict(config) if is_dataclass(config) else dict(config)
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))


def save_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def deterministic_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@contextmanager
def timed(label: str) -> Generator[None, None, None]:
    """Context manager that prints elapsed wall-clock time for the block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] elapsed: {elapsed:.2f}s")
