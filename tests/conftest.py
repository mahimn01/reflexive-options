"""Keep experiment side effects out of tracked publication directories during tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_ARTIFACT_ROOT: Path | None = None
if (
    "REFLEXIVE_OPTIONS_FIGURES_DIR" not in os.environ
    or "REFLEXIVE_OPTIONS_RUNS_DIR" not in os.environ
):
    _ARTIFACT_ROOT = Path(tempfile.mkdtemp(prefix="reflexive-options-pytest-"))
    os.environ.setdefault(
        "REFLEXIVE_OPTIONS_FIGURES_DIR",
        str(_ARTIFACT_ROOT / "figures"),
    )
    os.environ.setdefault(
        "REFLEXIVE_OPTIONS_RUNS_DIR",
        str(_ARTIFACT_ROOT / "runs"),
    )


def pytest_sessionfinish() -> None:
    """Remove isolated experiment outputs after the test session."""

    if _ARTIFACT_ROOT is not None:
        shutil.rmtree(_ARTIFACT_ROOT, ignore_errors=True)
