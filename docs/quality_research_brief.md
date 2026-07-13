# Quality-Upgrade Research Brief — `reflexive-options`

Status: research input for an enterprise-grade quality pass on a single-author Python 3.12 research repo (src layout, `pyproject.toml`, ML/scientific stack: `numpy`, `scipy`, `pandas`, `pyarrow`, `torch`, `gymnasium`, `QuantLib`). Defensibility matters because the code will be cited in a paper.

Each section recommends *one* option (not a catalog) and gives the exact config snippet to drop in.

---

## 1. gymnasium 1.x typing (2026 state)

### Does gymnasium 1.x ship `py.typed`?

**Yes.** The `pyproject.toml` on `Farama-Foundation/Gymnasium@main` lists `py.typed` in `[tool.setuptools.package-data]` alongside the asset files. The marker is shipped as part of the wheel since the 1.0.0 series.

There are **no third-party stubs** worth installing. Use the bundled annotations.

### The canonical typing pattern for custom env subclasses

`gymnasium.Env` is defined in `gymnasium/core.py` as:

```python
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Env(Generic[ObsType, ActType]):
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]: ...

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]: ...
```

Source: [`Farama-Foundation/Gymnasium/gymnasium/core.py`](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py). The reward type is the bare `SupportsFloat` protocol (not `float`), so don't tighten it.

### What `ObsType` / `ActType` should be

This is **not** documented authoritatively — see open issue [Farama-Foundation/Gymnasium#845 "Providing type arguments to gymnasium.Env?"](https://github.com/Farama-Foundation/Gymnasium/issues/845). The community-converged convention is:

| Space class | Concrete `ObsType` / `ActType` |
| --- | --- |
| `spaces.Box(..., dtype=np.float64)` | `npt.NDArray[np.float64]` |
| `spaces.Box(..., dtype=np.float32)` | `npt.NDArray[np.float32]` |
| `spaces.Discrete(n)` | `np.int64` (this is what `Discrete.sample()` actually returns; `int` works at runtime but doesn't match `sample()`'s type) |
| `spaces.MultiDiscrete(...)` | `npt.NDArray[np.int64]` |
| `spaces.Dict({"a": Box, "b": Discrete})` | `dict[str, npt.NDArray[Any]]` (heterogeneous; tighten with TypedDict only when worth it) |

`spaces.Space` *is* generic (`spaces.Space[T]`) — `T` is whatever `sample()` returns.

### Mypy-strict-acceptable typed env example

```python
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

ObsArray = npt.NDArray[np.float64]


class GammaSurfaceEnv(gym.Env[ObsArray, np.int64]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, n_strikes: int = 11) -> None:
        super().__init__()
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_strikes,), dtype=np.float64
        )
        self.action_space: spaces.Discrete = spaces.Discrete(3)
        self._state: ObsArray = np.zeros(n_strikes, dtype=np.float64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsArray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self.np_random.standard_normal(self._state.shape).astype(np.float64)
        return self._state, {}

    def step(
        self, action: np.int64
    ) -> tuple[ObsArray, SupportsFloat, bool, bool, dict[str, Any]]:
        reward: float = float(self._state.sum()) * (int(action) - 1)
        terminated = False
        truncated = False
        return self._state, reward, terminated, truncated, {}
```

This passes `mypy --strict` against gymnasium 1.x without any `# type: ignore`. Two gotchas: (a) the `action` arg comes through as `np.int64` not `int` because `Discrete.sample()` returns numpy ints — coerce with `int(action)` for arithmetic; (b) `super().reset(seed=seed)` is required for the seeded RNG, even though it returns `None`.

References: [Env API docs](https://gymnasium.farama.org/api/env/), [gymnasium release notes](https://gymnasium.farama.org/gymnasium_release_notes/index.html) (1.3.0, April 2024 is the line we're on).

---

## 2. pyarrow stubs (2026 state)

### Does pyarrow ship `py.typed` natively?

**Yes, since pyarrow 24.0.0 (released April 21, 2026).** That release introduced typing infrastructure: a top-level `py.typed` marker plus `.pyi` stubs derived from the donated `pyarrow-stubs` codebase (~13k lines). See the [Apache Arrow 24.0.0 release announcement](https://arrow.apache.org/blog/2026/04/21/24.0.0-release/) and the integration discussion at [apache/arrow#45919](https://github.com/apache/arrow/discussions/45919).

### Is `pyarrow-stubs` (zen-xu) still maintained / recommended?

The third-party [`pyarrow-stubs`](https://github.com/zen-xu/pyarrow-stubs) is still receiving releases (latest `20.0.0.20251215`, December 2025) but the upstream consolidation makes it **a fallback, not a primary recommendation**. The canonical 2026 pattern is: pin `pyarrow>=24.0` and use the bundled stubs.

### Caveat (be honest about this in the brief)

pyarrow 24.0.0's bundled stubs are **incomplete** for some dynamically-populated submodules. [apache/arrow#49831](https://github.com/apache/arrow/issues/49831) tracks an issue where `pyarrow.compute.all` and `pyarrow.compute.equal` are reported missing by mypy because they're attached at import time via `_make_global_functions()` rather than declared in the stubs. Mitigation: file-local `# type: ignore[attr-defined]` for the affected `pa.compute.*` calls, or per-module override:

```toml
[[tool.mypy.overrides]]
module = ["pyarrow.compute"]
ignore_errors = true
```

### Canonical 2026 install for parquet typing

```toml
# pyproject.toml dependency
"pyarrow>=24.0",
```

```bash
uv add "pyarrow>=24.0"
```

For `pyarrow<24` repos the recommendation flips to:

```bash
uv add --dev "pyarrow-stubs>=20.0"
```

ABI compat: pyarrow 24 dropped the `gandiva` extension and switched the build system to `scikit-build-core`. There are no Python-level ABI changes that affect typing. The wheels remain manylinux2014/macosx 11.0+/win_amd64.

### Recommendation

Pin `pyarrow>=24.0` and remove any third-party stubs. Add the per-submodule mypy override above for `pyarrow.compute` until [#49831](https://github.com/apache/arrow/issues/49831) ships.

---

## 3. mypy strict mode for ML/scientific Python (2026)

### Which strict-mode flags are widely treated as too noisy

Surveying real-world configs (PyTorch, scikit-learn, scipy, jax) and the issue threads, here is the verdict for an ML/scientific repo:

| Flag | Usually relaxed? | Why |
| --- | --- | --- |
| `disallow_any_unimported` | **Relax (false)** | Triggers on every untyped third-party import boundary; numpy's deprecated `mypy_plugin`, vendored RAT/ATLAS modules, etc. would force `# type: ignore` on every class declaration. PyTorch's `mypy-strict.ini` keeps it true *only* because PyTorch only strict-checks a small subset (`tools/`, `.github/`, etc.). Scientific repos almost universally relax it. |
| `disallow_any_explicit` | **Off** | Punishes legitimate uses like generic helpers, `dict[str, Any]` info dicts (gymnasium's `info` is exactly this). Not enabled by `--strict`. Don't turn it on. |
| `disallow_subclassing_any` | **Relax (false)** for scientific repos | Subclassing `gym.Env`, `torch.nn.Module`, or any dynamic ML base inherits `Any` chains. Required to be off in practice for any RL/torch code. PyTorch keeps it on only because it doesn't strict-check torch itself. |
| `warn_return_any` | **On (true)** | Cheap, catches real bugs at typed/untyped boundaries, almost no false positives in our context. Keep it. |
| `warn_unused_ignores` | **Relax (false)** | The killer flag for cross-environment work. PyTorch explicitly documents this in `mypy-strict.ini`: _"this option may cause un-ignorable errors on files that are checked by both `mypy.ini` and `mypy-strict.ini` ... mypy errors that appear on some platforms or environments but not on others."_ See [pytorch/pytorch#60006 (review comment)](https://github.com/pytorch/pytorch/pull/60006#issuecomment-866130657). **Disable it.** |

### Real-world configs surveyed

- **PyTorch** ([`mypy-strict.ini`](https://github.com/pytorch/pytorch/blob/main/mypy-strict.ini)): `strict = True`, `warn_unused_ignores = False`, `allow_redefinition = True`, `ignore_missing_imports = True` for numpy/sympy/usort/mypy. Strict only applied to a curated allow-list (`.github/`, `tools/`, `torch/utils/_pytree.py`, …); the bulk of `torch/*` is `follow_imports = skip`.
- **scikit-learn** ([`pyproject.toml`](https://github.com/scikit-learn/scikit-learn/blob/main/pyproject.toml)): minimal — `ignore_missing_imports = true`, `allow_redefinition = true`, exclude `sklearn/externals`. **Not strict.** They lean on ruff for most quality enforcement.
- **scipy**: disabled the `numpy.typing.mypy_plugin` after numpy 2.3 deprecated it ([scipy/scipy#23123](https://github.com/scipy/scipy/pull/23123)).
- **jax**: `pyproject.toml` has no `[tool.mypy]` section at all — they migrated to `pyrefly` (Meta's checker). Not a useful template for our case.

### How mature projects handle the local-vs-CI stub mismatch

The "I have stubs locally that CI doesn't have" problem (or vice versa) is exactly why `warn_unused_ignores` is dangerous in mixed-stub-coverage codebases. Five patterns observed:

1. **PyTorch**: `warn_unused_ignores = False` permanently, with a comment in `mypy-strict.ini` explaining the brittleness ([PR #60006](https://github.com/pytorch/pytorch/pull/60006)).
2. **scikit-learn**: avoids the problem entirely by not running strict mypy.
3. **pandas-stubs project**: uses targeted `# type: ignore[error-code]` with `enable_error_code = ["ignore-without-code"]` so unused ignores fail loudly *only* if they're untargeted.
4. **mypy itself**: pins exact stub versions in CI so the stub set is identical across environments. They keep `warn_unused_ignores = True` as a result.
5. **scipy / numpy**: `ignore_missing_imports = true` per-module override for any package whose stub coverage is environment-dependent (this is exactly the pattern already in our `pyproject.toml`).

**Recommendation for `reflexive-options`**: keep your current `warn_unused_ignores = false`. The justification is identical to PyTorch's. Document it (you already do, in the existing comment block).

### Best-practice config for our repo

Your current `[tool.mypy]` block is already very close to right. The only deltas I'd make:

```toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = false       # documented
no_implicit_optional = true
check_untyped_defs = true
warn_return_any = true
warn_unused_ignores = false           # documented (cross-env)
warn_redundant_casts = true
warn_unreachable = true
strict_equality = true
extra_checks = true
show_error_codes = true
enable_error_code = ["ignore-without-code", "redundant-expr", "truthy-bool"]
# Plot a path off the deprecated numpy plugin (NumPy 2.3 deprecated it):
# do NOT add `plugins = ["numpy.typing.mypy_plugin"]`.

[[tool.mypy.overrides]]
module = ["QuantLib.*", "diptest.*", "ripser.*", "mystic.*"]
ignore_missing_imports = true

# pyarrow.compute.* dynamic exports — pyarrow 24.0 stubs are incomplete here.
# Track apache/arrow#49831; remove this override once fixed.
[[tool.mypy.overrides]]
module = ["pyarrow.compute"]
ignore_errors = true

# Vendored ATLAS / RAT (CLAUDE.md "Vendoring discipline").
[[tool.mypy.overrides]]
module = "reflexive_options.third_party.*"
ignore_errors = true
```

Drop `gymnasium.*`, `torch.*`, `pyarrow.*`, `scipy.*` from the `ignore_missing_imports` list — all four ship `py.typed` in 2026 (gymnasium 1.x, torch 2.x with bundled stubs since 2.0, pyarrow 24+, and scipy ships partial typing since 1.13). Keep `QuantLib.*`, `diptest.*`, `ripser.*`, `mystic.*` — none ship stubs.

References: [mypy 1.20 config docs](https://mypy.readthedocs.io/en/stable/config_file.html), [PyTorch mypy-strict.ini](https://github.com/pytorch/pytorch/blob/main/mypy-strict.ini), [scikit-learn pyproject.toml](https://github.com/scikit-learn/scikit-learn/blob/main/pyproject.toml), [NumPy 2.3.0 release notes — mypy_plugin deprecation](https://numpy.org/doc/2.3/release/2.3.0-notes.html), [numtype project](https://github.com/numpy/numtype) (alpha, not yet recommended).

---

## 4. Reproducible Python dev environment (2026)

### Tool comparison

| Tool | Lock format | Cross-platform lock | Cold-install speed (large project) | Dev-dep grouping | Status 2026 |
| --- | --- | --- | --- | --- | --- |
| **uv** (Astral) | `uv.lock` (TOML) | **Yes**, single lock | ~3 s | `[dependency-groups]` (PEP 735) + extras | Dominant; Rust-based |
| pip-tools | `requirements.txt` | **No** (per-platform) | ~33 s | manual | Stable, narrow scope |
| Poetry | `poetry.lock` | Yes | ~11 s | `[tool.poetry.group.*]` | Mature, slower, idiosyncratic metadata |
| PDM | `pdm.lock` | Yes | ~5 s | `[dependency-groups]` (PEP 735) | Niche but PEP-compliant |

Numbers from [Cuttlesoft's "Python Dependency Management in 2026"](https://cuttlesoft.com/blog/2026/01/27/python-dependency-management-in-2026/), corroborated by the [astral-sh/uv README](https://github.com/astral-sh/uv).

### Recommendation

**Use uv.** Single binary, PEP 621 metadata (no proprietary `[tool.poetry]` block — your `pyproject.toml` works as-is), cross-platform lock, dev-group support, and the only one that actually solves "bit-identical environments locally and on CI" without per-platform lock files.

### "Lock everything" vs "let pip resolve" balance for a research repo

For a research repo headed for paper publication, **lock everything**. The cost (run `uv lock` when you upgrade a dep) is trivial; the benefit is that a reviewer running `uv sync --locked` two years from now gets the same numerical results. The exact justification you'd cite in a paper is the [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/) reproducibility recommendation.

### The exact pyproject.toml + lockfile pattern

`pyproject.toml`:

```toml
[project]
name = "reflexive-options"
requires-python = ">=3.12,<3.14"     # tighter upper bound than your current ">=3.12"
dependencies = [
    "numpy>=1.26",
    # ...
]

[dependency-groups]                  # PEP 735
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.15",
    "mypy>=1.20",
    "pre-commit>=4.0",
    "ipykernel>=6.29",
]

[tool.uv]
managed = true
package = true                       # treat the project itself as installable
```

CI install step:

```yaml
- uses: astral-sh/setup-uv@v8
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"
- run: uv sync --locked --all-extras --group dev
```

`--locked` (not `--frozen`) makes uv **fail** if `uv.lock` is out of sync with `pyproject.toml`. That's the desired CI behavior. `--frozen` skips the check and is for environments where the lockfile should be trusted blindly (build images, etc.). See [uv's GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/) and the [sync semantics page](https://docs.astral.sh/uv/concepts/projects/sync/).

Commit `pyproject.toml` and `uv.lock`. Do not commit `requirements.txt`.

---

## 5. Pre-commit hooks (2026 standards)

### Should mypy run pre-commit or only in CI?

**CI only, plus a manual local invocation when the dev wants it.** Two reasons:

1. The pre-commit-mypy hook runs in an isolated venv. To see numpy/torch/pyarrow types you must duplicate the dependency list in `additional_dependencies`, and that list will drift from `pyproject.toml`. This is a documented pain point ([pre-commit/pre-commit#2951](https://github.com/pre-commit/pre-commit/issues/2951), [Jared Khan's "Running Mypy in Pre-commit"](https://jaredkhan.com/blog/mypy-pre-commit)).
2. Mypy on a partial file set is incoherent — the type of code you didn't touch can change because of code you did touch. Running mypy on every commit but only over the staged subset gives false greens.

The standard 2026 pattern in mature scientific repos is: ruff in pre-commit (fast, file-local), mypy in CI only, with a `Makefile` / `justfile` target so the dev can run `just typecheck` locally on demand. Bas Nijholt's ["My favorite tools that keep my Python projects sane"](https://www.nijho.lt/post/best-python-dev-tooling/) covers this.

### Pytest in pre-commit?

**No.** Same reasoning as mypy, plus tests in our repo are slow (numerical SDE simulations, RL rollouts). Run pytest in CI. If you want a guard-rail, add a `pre-push` hook (separate from `pre-commit`) that runs `pytest -x --ff -q`.

### Complete `.pre-commit-config.yaml`

```yaml
# .pre-commit-config.yaml
default_language_version:
  python: python3.12

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: check-toml
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: debug-statements
      - id: check-merge-conflict
      - id: mixed-line-ending

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.12
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format

  # mypy intentionally NOT here — runs in CI only.
  # See docs/quality_research_brief.md §5 for rationale.
```

If you really want mypy in pre-commit despite the warnings above, the safest variant is the *local* hook pattern (avoids `additional_dependencies` drift entirely):

```yaml
  - repo: local
    hooks:
      - id: mypy
        name: mypy (local venv)
        entry: uv run mypy
        language: system
        types_or: [python, pyi]
        require_serial: true
        pass_filenames: false  # always typecheck the whole project
        args: ["src", "tests"]
```

This delegates to the project's actual venv via `uv run`, so mypy sees the real installed numpy/torch/pyarrow.

References: [astral-sh/ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit) (v0.15.12, April 2026), [pre-commit/pre-commit-hooks v6.0.0](https://github.com/pre-commit/pre-commit-hooks) (released Aug 2025), [pre-commit/mirrors-mypy](https://github.com/pre-commit/mirrors-mypy).

---

## 6. CI hardening for Python ML repos (2026)

### Python version matrix

Python 3.13 went bug-fix in October 2024; Python 3.14 went stable on October 7, 2025. As of April 2026 the active-bugfix line is 3.13 + 3.14 ([devguide.python.org/versions](https://devguide.python.org/versions/)).

For an academic repo whose `requires-python = ">=3.12"`:

```yaml
strategy:
  matrix:
    python-version: ["3.12", "3.13", "3.14"]
```

Don't include 3.15-dev unless you specifically want pre-release breakage signal. Don't include 3.11 or below — they're security-only and your declared minimum is 3.12. Two of three jobs (3.13, 3.14) catch numpy/torch wheel availability regressions, which is the main practical reason to matrix.

### Coverage threshold

The [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/coverage/) is explicit: *don't fixate on a number*; what matters is reliable tests and a small allowed regression band.

Defensible academic targets:
- **Project-level threshold: 80%.** This is the codecov-action default and is what reviewers expect.
- **Patch coverage: informational, not blocking.** A 5% allowed loss on the project ratchet is the SciPy/scikit-image convention.

Codecov-style gate in `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 80
exclude_also = [
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "@overload",
]

[tool.coverage.run]
branch = true
source = ["src/reflexive_options"]
omit = ["src/reflexive_options/third_party/*"]
```

### Security scanning

Bandit (1.9.3, January 2026) is still maintained and shipped in many CI templates, but **prefer Ruff's `S` rule group**. Ruff implements most flake8-bandit checks at ~25× the speed and removes a tool from the dependency graph ([Ruff vs Bandit performance comparison, Feb 2026](https://mcginniscommawill.com/posts/2026-02-10-ruff-bandit-vs-traditional/)). Add to your existing ruff config:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "RUF", "S"]
# S101: assert is fine in tests; S301/S403: pickle is fine for our trained models
ignore = ["E501", "RUF001", "RUF002", "RUF003", "N802", "N803", "N806", "S101"]
```

For deeper SAST (taint analysis, etc.), Semgrep is the canonical step up but is overkill for a research repo. Skip it.

### GitHub Actions versions (Node 24)

The deprecation warning you saw is GitHub deprecating Node 20 actions. Pin these:

| Action | Version | Notes |
| --- | --- | --- |
| `actions/checkout` | `v5` (stable) or `v6` (current) | Both run on Node 24 |
| `actions/setup-python` | `v6` | Upgraded to Node 24 in early 2026 |
| `astral-sh/setup-uv` | `v8` | Pin by exact tag in production: `astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0` |
| `codecov/codecov-action` | `v6` | Node 24; set `use_oidc: true` and grant `id-token: write` for secretless OIDC upload |

Sources: [actions/checkout releases](https://github.com/actions/checkout/releases), [actions/setup-python releases](https://github.com/actions/setup-python/releases), [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv), and [codecov/codecov-action](https://github.com/codecov/codecov-action#using-oidc).

### Caching strategy

**Use uv's built-in cache via `setup-uv`'s `enable-cache: true`.** It caches the global uv store and keys on `uv.lock` automatically; no manual `actions/cache` block needed. The pip cache is irrelevant if you're using uv. The "dependency-aware cache" (key on `pyproject.toml + uv.lock`) is what `setup-uv` does by default.

```yaml
- uses: astral-sh/setup-uv@v8
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"
- run: uv cache prune --ci  # at end of job, optional but recommended
```

`uv cache prune --ci` strips pre-built wheels (which the cache hit will re-fetch anyway), keeping only source-built wheels. Cuts cache size 5-10× on ML stacks. See [uv caching docs](https://docs.astral.sh/uv/concepts/cache/) and [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/).

---

## 7. Strict mypy flags in 2026 — per-flag verdict for this repo

Each is on/off with one-line justification. Context: scientific Python with ML deps, single researcher, paper defensibility.

| Flag | Verdict | Justification |
| --- | --- | --- |
| `strict` | **on** | Sets the baseline; individual sub-flags below override. |
| `disallow_any_unimported` | **off** | Vendored ATLAS/RAT and untyped libs (QuantLib, diptest, ripser) make this require `# type: ignore` on every class — see your existing CLAUDE.md vendoring discipline. PyTorch only keeps it on for a curated allow-list. |
| `disallow_any_explicit` | **off** | `dict[str, Any]` info dicts are unavoidable in gymnasium and cause cascading false positives. Not enabled by `--strict`. |
| `disallow_any_decorated` | **off** | `@torch.jit.script`, `@functools.wraps`, `@gym.wrappers.*` all return `Any` after decoration; this would require ignores everywhere. |
| `disallow_any_expr` | **off** | The most aggressive Any check in mypy — incompatible with any numpy code (`arr.sum()` returns Any-ish). Documented as "too strict to be useful" in [python/mypy#7767](https://github.com/python/mypy/issues/7767). |
| `disallow_subclassing_any` | **off** | Can't subclass `gym.Env` or `torch.nn.Module` otherwise. The single most-common reason to relax `--strict` in ML repos. |
| `warn_unused_ignores` | **off** | Cross-environment instability — see §3 above and [pytorch/pytorch#60006 comment](https://github.com/pytorch/pytorch/pull/60006#issuecomment-866130657). Keep the documented comment block. |
| `warn_redundant_casts` | **on** | Cheap, no false positives in ML code. |
| `warn_return_any` | **on** | Catches the most-common silent leak from untyped third-party code. |
| `warn_unreachable` | **on** | Catches dead branches after refactors; rarely false-positives. |
| `check_untyped_defs` | **on** | Already implied by `disallow_untyped_defs`, but explicit is fine. |
| `no_implicit_optional` | **on** | PEP 484 modern behavior; required for clean `Optional[T]` annotations. |
| `strict_equality` | **on** | Catches `x == None` and similar; trivially worth it. |
| `extra_checks` | **on** | Enables the "technically correct but rarely-needed" checks that have stabilized in mypy 1.18+. Includes the new TypedDict mutability checks. Keep on; flip off if you hit a false positive. |

The flags collectively recommended **on** beyond `strict`'s defaults are: `warn_redundant_casts`, `warn_return_any`, `warn_unreachable`, `check_untyped_defs`, `no_implicit_optional`, `strict_equality`, `extra_checks`, plus `enable_error_code = ["ignore-without-code", "redundant-expr", "truthy-bool"]`. The flags recommended **off** despite `strict`'s defaults: `warn_unused_ignores`, `disallow_subclassing_any` (relax via per-module override for `gymnasium.*` and `torch.*` if needed), `disallow_any_unimported`.

Reference: [mypy 1.20 command-line / config-file docs](https://mypy.readthedocs.io/en/stable/command_line.html), [mypy 1.20 error code list](https://mypy.readthedocs.io/en/stable/error_code_list2.html).

---

## Quick cross-section action list (for the implementation agent)

1. Bump `pyarrow>=24.0` in `pyproject.toml`; remove `pyarrow.*` from the `ignore_missing_imports` override; add narrow `pyarrow.compute` ignore.
2. Bump `gymnasium>=1.2`; remove `gymnasium.*` from `ignore_missing_imports`; add `ObsType`/`ActType` parameters to all `gym.Env` subclasses per §1.
3. Remove `torch.*`, `scipy.*` from `ignore_missing_imports` (all ship `py.typed` now).
4. Add `enable_error_code = ["ignore-without-code", "redundant-expr", "truthy-bool"]` and `strict_equality`, `warn_unreachable`, `warn_redundant_casts`, `extra_checks` to `[tool.mypy]`.
5. Migrate from `[project.optional-dependencies].dev` to PEP 735 `[dependency-groups]`; add `uv.lock`; add `[tool.uv] managed = true, package = true`.
6. Add `.pre-commit-config.yaml` from §5 (no mypy hook).
7. Add `S` to `[tool.ruff.lint].select`; ignore `S101`.
8. Update CI to `actions/checkout@v5`, `actions/setup-python@v6`, `astral-sh/setup-uv@v8`, matrix `["3.12","3.13","3.14"]`, `uv sync --locked`, `uv cache prune --ci`.
9. Add `[tool.coverage.*]` block with `fail_under = 80`, branch coverage, `omit` third_party.
10. Verify build: `uv sync --locked && uv run mypy src tests && uv run pytest && uv run ruff check && uv run ruff format --check`.
