# sonde redesign implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild sonde as a small, config-driven probe toolkit. The flow is
`sonde run cfg.yaml` → extract activations (HF via nnterp, or vLLM via nnsight) → train and
select probes, reporting recall@1% FPR against text baselines → export a numpy-only probe
artifact that mechanica imports.

**Architecture:** 15 flat modules, about 2.5k LOC, replacing 6.8k. The backend seam is
`backends.forward` / `backends.generate`, with one hf and one vllm implementation. Steps
(`generate`, `extract`, `train`, `score`, `steer`) are plain functions. They share files
through a run directory and share the model through a single-slot cache. `sonde.probe` and
`sonde.fingerprint` import only numpy and the stdlib.

**Tech Stack:** Python 3.12, numpy, torch, pydantic v2, pyyaml, safetensors, HF transformers and
datasets, nnterp 1.3 with nnsight 0.7 (hf env), nnsight git `b717807` with vLLM (Spark env),
uv, pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-10-02-sonde-redesign-design.md`. Read it with this plan.
The interface contract that every task's Interfaces block follows is reproduced in the
appendix.

## Global Constraints

- `requires-python >= 3.12`. The base dependency is `numpy` only; everything else lives in the
  `core`, `hf`, `vllm` and `dev` extras. `hf` and `vllm` are declared conflicting to uv.
- `import sonde`, `sonde.probe` and `sonde.fingerprint` must never import torch, transformers,
  nnsight, vllm, pydantic, yaml or safetensors.
- Artifacts are data, never code. Write them with `np.savez` / safetensors and read them with
  `allow_pickle=False`. Never pickle.
- `Probe.block` is the residual stream after decoder block `block` (0-indexed), which is
  `vllm_aux_layer() - 1`. Never read HF `output_hidden_states`.
- `prompt_format = fingerprint.prompt_format(tokenizer.chat_template)` is a digest of the raw
  template with no kwargs. `model_fingerprint = checkpoint_fingerprint(name, revision=<as asked>,
  quantization=None)`.
- Pooling acts on logits, and the sigmoid is applied after it (float32, clipped to ±30).
  Thresholds are chosen on val using scores from the exported numpy `Probe`.
- The headline metric is recall at `max_fpr` (default 0.01). Layers are selected by it, with
  AUROC breaking ties, or by AUROC alone when `n_neg × max_fpr < 10`.
- HF forward pads on the right and passes `attention_mask`. HF generate pads on the left. Both
  run under `torch.no_grad()`, never `inference_mode`.
- vLLM: prefix caching and chunked prefill are off, and there are no taps. Extraction uses a
  per-row `tracer.invoke(ids[:end])` with at most 500 invokes per trace. Every request passes
  `edits=[...]` explicitly. Use `nnsight.save(x)`, never `.save()`.
- The extract hash check runs before any file is written and before any model loads.
- Style follows the Google Python Style Guide:
  - `import module` only, with one `from sonde import x` per line (ruff `force-single-line`);
  - 80 columns;
  - `from __future__ import annotations`;
  - Args, Returns and Raises sections on public APIs;
  - docstrings give array shapes;
  - no WHAT comments; `# ponytail:` marks deliberate corners.
- Commits have no Co-Authored-By trailer. Work stays on branch `ani/redesign`, with one PR per
  phase.

## Review Focus

These are the input classes most likely to bite a user. Each has a test in the task that owns
the code.

1. **A long prompt whose window is cut by `max_length`.** The row is dropped and counted in the
   manifest. It never becomes an empty or NaN feature. Tests:
   `test_window_cut_by_max_length_is_dropped_not_empty` (Task 2.6) and
   `test_drops_are_counted_never_empty` (Task 2.3).
2. **Re-running a run name after editing its config.** The run refuses before any model loads,
   and the error tells the user about `output.overwrite`. Test:
   `test_edited_config_refuses_before_model_load` (Task 2.6).
3. **A small dataset where a split ends up one-class.** It fails with a `ValueError` that names
   the split. Test: `test_split_missing_class_names_the_split` (Task 2.2).
4. **`format: chat` on a base model with no chat template, such as gpt2.** It fails clearly and
   suggests `format: raw`. Test: `test_chat_without_template_says_use_raw` (Task 2.3).
5. **Scoring a probe trained under a different prompt format, checkpoint or backend.** It is
   refused with the reason. Test: `test_score_refuses_mismatched_probe` (Task 3.5).

## Phase map

| Phase | Tasks | Done when |
|---|---|---|
| 1 Clean slate, packaging, probe artifact | 1.1–1.4 | `test_probe.py` passes; `uv lock` resolves |
| 2 Config, data, HF backend, extraction, storage | 2.1–2.6 | gpt2 batch invariance passes; resume and hash guard work |
| 3 Probes, training, sweep, score, runner, CLI | 3.1–3.6 | `sonde run quickstart` on CPU; headline printed |
| 4 Recipes, cleanup, CI, README | 4.1–4.4 | tree matches spec §4; CI green |
| 5 vLLM backend + Spark parity spike | 5.1–5.5 | `docs/parity-2026-10.md` filled in by script; recipes re-run |
| 6 Generation and steering | 6.1–6.5 | `test_steer.py` passes on hf; vllm steering checked on the Spark |

---

## Phase 1: Clean slate, packaging, probe artifact

**Goal.** Archive v0.1 under the tag `v0.1-legacy`, delete every old package under `sonde/` and every current test, and rewrite the packaging per spec §12: numpy-only base, `core`/`hf`/`vllm`/`dev` extras, the uv conflict, a CPU torch index for linux x86_64 only, and ruff and pyright configured in `pyproject.toml` (`pyrightconfig.json` is deleted). Then build the numpy-only artifact layer mechanica will import: `sonde/fingerprint.py` (moved verbatim) and `sonde/probe.py` (spec §5 plus the contract). **Done when:** `uv lock` resolves with the vllm extra pulling PyPI (CUDA) torch, `uv run pytest -m "not gpu"` passes `tests/test_probe.py` (13 tests), and `ruff` and `pyright` are clean on `sonde tests`. The old `.github/workflows/ci.yml` (Python 3.10 matrix) stays red until phase 4 rewrites it. That is expected under D2, and phase 1 does not touch CI.

**Cross-phase effects of this phase's config.** `force-single-line = true` means every module and test in every phase writes one `from sonde import x` per line; the combined form `from sonde import a, b` fails `ruff check` with I001. Pyright is configured under `[tool.pyright]` here, so phase 4 does not create or delete `pyrightconfig.json`.

**Verified while writing this phase** (scratch copy of the repo, uv 0.12.5, ruff 0.16.10, pyright 1.1.414, macOS arm64, 2026-10-02):
- The `pyproject.toml` below locks: `Resolved 248 packages`. vllm extra: `vllm==0.27.1`, nnsight from git `0.7.1.dev284+gb71780727`, torch `2.13.0` from PyPI. hf extra: `nnterp==1.3.0`, `nnsight==0.7.0`, torch `2.14.1+cpu` from the CPU index on linux x86_64 and `2.13.0` from PyPI elsewhere.
- **Without `extra = "hf"` on the torch source, the vllm extra on every linux machine (the aarch64 Spark included) resolved to `torch 2.13.0+cpu`.** The CPU index leaked into the vLLM env. The `extra = "hf"` scoping below is load-bearing, and Task 1.2 checks it.
- `uv sync --extra hf --extra vllm` fails with ``Extras `hf` and `vllm` are incompatible with the declared conflicts``. `uv sync --locked --extra hf --extra dev` succeeds on macOS. The lock resolves even though vllm has no macOS wheels, because of the `sys_platform == 'linux'` markers.
- ruff: mechanica's `fingerprint.py` would be reformatted and has a RUF002 en dash, so it is excluded to stay verbatim. Without `force-exclude = true`, an explicitly passed path (pre-commit passes paths) ignores the exclude; with it, `ruff format --check sonde/fingerprint.py` finds no files. ruff 0.16 formats Python blocks inside Markdown, and both the old `README.md` and the committed spec fail `ruff format --check`; with `*.md` excluded they pass. With `force-single-line = true`, `from sonde import config` then `from sonde import probe` on separate lines passes and `from sonde import config, probe` is flagged I001.
- `experiments/` and `examples/` remain until phase 4 and fail ruff under this config (110 lint errors, 10 files to reformat), so phase 1 runs ruff on `sonde tests`. With them gone, `ruff format --check .` and `ruff check .` are clean.
- pyright reads `[tool.pyright]` once `pyrightconfig.json` is gone (`--verbose`: `Loading pyproject.toml file`, `Python version: 3.12`, `Found 3 source files`) and reports `0 errors` on phase 1's code.
- pyright rejects `np.savez(handle, **arrays)` (the dict could bind `allow_pickle: bool`) unless `allow_pickle=False` is passed explicitly. It also rejects `acts @ self.q` unless `q` is narrowed. Branching on `self.q is None` does the narrowing.
- `from sonde import probe` works with a module `__getattr__` that raises `AttributeError` for unknown names, because import falls back to the submodule. Setting `sys.modules[m] = None` blocks a module: a mutation that adds `import yaml` to `probe.py` makes the isolation test fail.
- transformers 5.18: `apply_chat_template(tokenize=True)` returns a `BatchEncoding` (so `len()` is 2, the key count). `probe.last_turn_start` on `Qwen/Qwen2.5-0.5B-Instruct` returns 30, and `decode(full[30:])` is `"yo there<|im_end|>"`.
- numpy 2.5.3: `np.lib.stride_tricks.sliding_window_view` and `np.load(...)` work as a context manager.

---

### Task 1.1: Tag v0.1-legacy and clear the old package

**Files:**
- Delete: `sonde/_pkg.py`, `sonde/activation/`, `sonde/cli/`, `sonde/configs/`, `sonde/core/`, `sonde/dataset/`, `sonde/directions/`, `sonde/generation/`, `sonde/interventions/`, `sonde/probes/`, `sonde/runners/`, `tests/*.py` (all 40 current test files, `test_rjudge_*` included, which import `experiments/`)
- Rewrite: `sonde/__init__.py`
- Keep: `sonde/py.typed`. `experiments/` and `examples/` stay until phase 4; nothing collects or type-checks them.

**Interfaces:**
- Consumes: nothing.
- Produces: `sonde.__version__ = "0.2.0"`, and `sonde.__getattr__(name: str) -> typing.Any`, which lazily maps `"run"` to `sonde.runner.run` (phase 3) and `"load_config"` to `sonde.config.load` (phase 2) and raises `AttributeError` for anything else. Importing `sonde` imports only the stdlib.

- [ ] **Step 1: Tag main's tip, not this branch**

```bash
git rev-parse main
git tag -a v0.1-legacy c6095ff -m "sonde 0.1: last commit before the redesign"
git rev-parse 'v0.1-legacy^{commit}'
```

Expected: both `rev-parse` lines print `c6095ff2b6d0ed496893f1ee08ce64da9195cd89`.

- [ ] **Step 2: Remove the old packages and tests**

```bash
git rm -r -q sonde/_pkg.py sonde/activation sonde/cli sonde/configs \
  sonde/core sonde/dataset sonde/directions sonde/generation \
  sonde/interventions sonde/probes sonde/runners
git rm -q tests/*.py
find sonde tests -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
find sonde -type d -empty -delete
git ls-files sonde tests
```

Expected output, exactly:
```
sonde/__init__.py
sonde/py.typed
```

- [ ] **Step 3: Write the lazy `sonde/__init__.py`**

```python
"""sonde: train probes on LLM activations and export a portable artifact.

`sonde.probe` and `sonde.fingerprint` import only numpy and the stdlib, so
this module must not import anything heavier.
"""

from __future__ import annotations

import importlib
import typing

__version__ = "0.2.0"

_LAZY = {
    "run": ("sonde.runner", "run"),
    "load_config": ("sonde.config", "load"),
}


def __getattr__(name: str) -> typing.Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'sonde' has no attribute {name!r}")
    module, attr = _LAZY[name]
    return getattr(importlib.import_module(module), attr)
```

- [ ] **Step 4: Check that the package imports with the stdlib only**

The old `pyproject.toml` is still in place, so run this outside the project env:

```bash
uv run --no-project --python 3.12 python -c "
import sys, sonde
assert sonde.__version__ == '0.2.0'
assert not {'numpy', 'torch', 'pydantic', 'yaml'} & set(sys.modules)
try:
    sonde.nope
except AttributeError as e:
    print('ok:', e)
"
```

Expected: `ok: module 'sonde' has no attribute 'nope'`

- [ ] **Step 5: Commit**

```bash
git add sonde/__init__.py
git commit -m "refactor!: delete the v0.1 package and its tests

v0.1 is preserved under the v0.1-legacy tag (c6095ff)."
```

- [ ] **Step 6: Publish the tag with the phase PR**

When you push the branch for the phase-1 PR, also run `git push origin v0.1-legacy`. Expected: `* [new tag] v0.1-legacy -> v0.1-legacy`.

---

### Task 1.2: Packaging per spec §12 and a resolving lock

**Files:**
- Rewrite: `pyproject.toml`
- Delete: `pyrightconfig.json` (its settings move to `[tool.pyright]`, with `pythonVersion = "3.12"`)
- Regenerate: `uv.lock`. The working tree has an uncommitted `uv.lock` change from before this plan; the regenerated lock replaces it.

**Interfaces:**
- Consumes: `sonde/__init__.py` from Task 1.1.
- Produces:
  - extras `core`, `hf`, `vllm` and `dev`;
  - the uv conflict `hf` ⟂ `vllm`;
  - console script `sonde = "sonde.cli:main"` (`sonde/cli.py` arrives in phase 3, so the script fails until then);
  - pytest marker `gpu`, with no `pythonpath`;
  - package-data `sonde/recipes/*.yaml` and `sonde/recipes/data/*`, so phase 3's recipes ship without touching `pyproject.toml`;
  - ruff config: `force-single-line` imports, `force-exclude`, with `sonde/fingerprint.py` and `*.md` excluded;
  - pyright config under `[tool.pyright]` (include `sonde`, Python 3.12, basic);
  - the dev env command `uv sync --extra hf --extra dev`, which every later phase uses.

- [ ] **Step 1: Replace `pyproject.toml` entirely**

```toml
[project]
name = "sonde"
version = "0.2.0"
description = "Train portable probes on LLM activations, export them for serving."
readme = "README.md"
requires-python = ">=3.12"
license = { file = "LICENSE" }
authors = [{ name = "The sonde authors" }]
dependencies = ["numpy"]

[project.optional-dependencies]
core = [
    "transformers",
    "safetensors",
    "pydantic>=2",
    "pyyaml",
    "datasets",
]
hf = [
    "sonde[core]",
    "torch",
    "nnterp==1.3.*",
    "nnsight>=0.6,<0.8",
]
vllm = [
    "sonde[core]",
    "vllm==0.27.1; sys_platform == 'linux'",
    "nnsight[vllm] @ git+https://github.com/ndif-team/nnsight@b71780727ea9713f74ce12f76b1d3548b91a74a0 ; sys_platform == 'linux'",
]
dev = ["pytest==9.1.1", "ruff==0.16.10", "pyright==1.1.414"]

[project.urls]
Homepage = "https://github.com/aniruddh-alt/sonde"
Repository = "https://github.com/aniruddh-alt/sonde"
Changelog = "https://github.com/aniruddh-alt/sonde/blob/main/CHANGELOG.md"

[project.scripts]
sonde = "sonde.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["sonde*"]

[tool.setuptools.package-data]
sonde = ["py.typed", "recipes/*.yaml", "recipes/data/*"]

[tool.uv]
conflicts = [[{ extra = "hf" }, { extra = "vllm" }]]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "hf", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: needs a CUDA GPU; deselect with -m 'not gpu'"]

[tool.ruff]
target-version = "py312"
line-length = 80
force-exclude = true
# ponytail: fingerprint.py is a verbatim copy of mechanica's; keep it
# byte-identical until mechanica imports it from sonde.
extend-exclude = ["sonde/fingerprint.py", "*.md"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E741", "RUF012"]

[tool.ruff.lint.isort]
known-first-party = ["sonde"]
force-single-line = true

[tool.pyright]
include = ["sonde"]
pythonVersion = "3.12"
typeCheckingMode = "basic"
reportPrivateImportUsage = "none"
venvPath = "."
venv = ".venv"
```

`vllm==0.27.1` is the spec's fallback pin, the version nnsight tests on. The phase-5 spike (§13 check 1) raises it if a newer version passes, so this is not a placeholder. The `sys_platform == 'linux'` markers let the lock resolve on macOS, where vllm has no wheels. `force-exclude = true` keeps the excludes in force when pre-commit passes file paths to ruff explicitly.

- [ ] **Step 2: Delete `pyrightconfig.json`**

```bash
git rm -q pyrightconfig.json
```

Pyright gives `pyrightconfig.json` precedence over `pyproject.toml`, so it must go for `[tool.pyright]` to take effect.

- [ ] **Step 3: Lock from scratch**

```bash
rm uv.lock
uv lock
```

Expected: `Resolved <N> packages in …` with no error (N was 248 on 2026-10-02). It needs network and `git`, because uv clones nnsight at `b717807` to read its metadata.

- [ ] **Step 4: Check that the vllm extra gets CUDA torch, not the CPU index**

```bash
uv run --no-project --python 3.12 python - <<'EOF'
import tomllib
lock = tomllib.load(open("uv.lock", "rb"))
for p in lock["package"]:
    if p["name"] == "vllm":
        print([d for d in p["dependencies"] if d["name"] == "torch"])
    if p["name"] == "torch":
        print(p["version"], p["source"])
EOF
```

Expected: the `vllm` line shows torch with `'source': {'registry': 'https://pypi.org/simple'}`. A `download.pytorch.org/whl/cpu` source there means the `extra = "hf"` scoping on the torch source was lost. There is exactly one `+cpu` torch entry, and it is reached only through the hf extra on linux x86_64.

- [ ] **Step 5: Check the conflict, sync the dev env, and confirm pyright reads pyproject**

```bash
uv sync --extra hf --extra vllm 2>&1 | tail -1
uv sync --locked --extra hf --extra dev
uv run python -c "import torch, nnterp, nnsight, sonde; print(nnsight.__version__, sonde.__version__)"
uv run pyright --verbose 2>&1 | grep -E "Loading pyproject|Python version"
```

Expected:
1. ``error: Extras `hf` and `vllm` are incompatible with the declared conflicts: {`sonde[hf]`, `sonde[vllm]`}``
2. The sync succeeds.
3. `0.7.0 0.2.0`
4. `Loading pyproject.toml file at …/pyproject.toml` and `  Python version: 3.12`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: numpy-only base with core/hf/vllm/dev extras

Python >= 3.12. The hf and vllm extras conflict, and CPU torch is used
for the hf extra on linux x86_64 only. The sonde script points at
sonde.cli:main. Ruff uses one import per line and excludes Markdown and
the verbatim fingerprint.py; pyright config moves into pyproject."
```

---

### Task 1.3: `fingerprint.py` move; `Probe` validation, I/O and `load_dir`

**Files:**
- Create: `sonde/fingerprint.py` (byte-identical copy of `~/dev/personal/mechanica/mechanica/fingerprint.py`)
- Create: `sonde/probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Consumes: the dev env from Task 1.2.
- Produces:
  - `fingerprint.RAW = "raw"`
  - `fingerprint.prompt_format(template: str | None) -> str`
  - `fingerprint.artifact_digest(path) -> str`
  - `fingerprint.normalize_revision(value) -> str | None`
  - `fingerprint.checkpoint_fingerprint(model: str, revision: str | None = None, quantization: str | None = None) -> str`
  - `probe.FORMAT = 1`, `probe.WINDOWS`, `probe.KINDS`, `probe.LINEAR_POOLINGS` (frozensets)
  - `probe.last_turn_start(tokenizer, messages: list[dict]) -> int`
  - `@dataclasses.dataclass class probe.Probe`, with fields in this order: `name, kind, w, bias, block, window, pooling, threshold, model, engine, model_fingerprint, prompt_format, q=None, rolling_window=None, adapters=(), escalate_threshold=None, metrics=None`. Its `__post_init__` raises `ValueError` on any violation.
  - `Probe.save(self, path: str) -> str`: atomic; appends `.npz` if missing.
  - `Probe.load(cls, path: str) -> Probe`: `ValueError` on an unknown format or meta that doesn't build a `Probe`.
  - `probe.load_dir(directory: str) -> dict[str, Probe]`
  - Phase 3 note: `metrics` is written with `json.dumps`, so it must hold JSON-native values (Python `float`/`int`, not `np.float32`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_probe.py`:

```python
from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from sonde import probe

H = 4
BLOCKED = (
    "torch",
    "transformers",
    "nnsight",
    "nnterp",
    "vllm",
    "pydantic",
    "yaml",
    "safetensors",
    "datasets",
)


def make(**overrides) -> probe.Probe:
    fields = {
        "name": "p",
        "kind": "linear",
        "w": [0.1, 0.2, 0.3, 0.4],
        "bias": -0.25,
        "block": 3,
        "window": "prompt",
        "pooling": "mean",
        "threshold": 0.5,
        "model": "gpt2",
        "engine": "hf==5.1.0+nnterp==1.3.0",
        "model_fingerprint": "hub:abc",
        "prompt_format": "raw",
    }
    fields.update(overrides)
    return probe.Probe(**fields)


def assert_same(a: probe.Probe, b: probe.Probe) -> None:
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype == np.float32
            assert np.array_equal(x, y), field.name
        else:
            assert x == y, field.name


def test_round_trip(tmp_path):
    attention = make(
        kind="attention",
        pooling="attention",
        q=[0.5, -0.5, 0.0, 1.0],
        adapters=["lora-b", "lora-a"],
        escalate_threshold=0.25,
        metrics={"val": {"auroc": 0.9, "n_pos": 3, "n_neg": 5}},
    )
    rolling = make(pooling="rolling_mean", rolling_window=2)
    for i, original in enumerate((attention, rolling)):
        path = original.save(str(tmp_path / f"p{i}"))
        assert path == str(tmp_path / f"p{i}.npz")
        loaded = probe.Probe.load(path)
        assert_same(original, loaded)
        assert loaded.adapters == tuple(sorted(original.adapters))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p0.npz", "p1.npz"]


def test_coerces_on_construction():
    p = make(w=[1, 2, 3, 4], bias=np.float64(1), adapters=["b", "a"])
    assert p.w.dtype == np.float32
    assert type(p.bias) is float
    assert p.adapters == ("a", "b")


def test_probe_and_fingerprint_import_with_numpy_only():
    block = f"import sys\nfor m in {BLOCKED!r}:\n    sys.modules[m] = None\n"
    for stmt in (
        "import sonde.probe",
        "import sonde.fingerprint",
        "from sonde import probe, fingerprint",
    ):
        subprocess.run([sys.executable, "-c", block + stmt], check=True)
    blocked = subprocess.run(
        [sys.executable, "-c", block + "import yaml"], capture_output=True
    )
    assert blocked.returncode != 0


def test_legacy_and_unknown_format_refused(tmp_path):
    legacy_meta = {
        "name": "old",
        "bias": 0.0,
        "layer": 4,
        "model": "gpt2",
        "threshold": 0.5,
        "engine": "vllm==0.29.0",
        "window": "prompt",
        "positions": "last",
    }
    for i, meta in enumerate((legacy_meta, {**legacy_meta, "format": 1})):
        path = tmp_path / f"legacy{i}.npz"
        np.savez(path, weight=np.ones(H, np.float32), meta=json.dumps(meta))
        with pytest.raises(ValueError):
            probe.Probe.load(str(path))
    path = make().save(str(tmp_path / "future"))
    with np.load(path) as archive:
        meta = json.loads(str(archive["meta"]))
        w = archive["w"]
    np.savez(path, w=w, meta=json.dumps({**meta, "format": 2}))
    with pytest.raises(ValueError):
        probe.Probe.load(path)


def test_validation_errors():
    attention = {"kind": "attention", "pooling": "attention"}
    bad = [
        {"w": []},
        {"w": [0.1, math.nan, 0.3, 0.4]},
        {"w": np.ones((2, 2))},
        {"bias": math.inf},
        {"threshold": 1.5},
        {"threshold": math.nan},
        {"block": -1},
        {"block": True},
        {"block": 1.5},
        {"name": ""},
        {"model": " "},
        {"engine": ""},
        {"model_fingerprint": ""},
        {"window": "suffix"},
        {"kind": "softmax"},
        {"q": np.ones(H)},
        attention,
        {**attention, "q": np.ones(H + 1)},
        {"kind": "attention", "pooling": "mean", "q": np.ones(H)},
        {"pooling": "attention"},
        {"pooling": "rolling_mean"},
        {"pooling": "rolling_mean", "rolling_window": 0},
        {"pooling": "mean", "rolling_window": 2},
        {"adapters": ("",)},
        {"escalate_threshold": 0.6},
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            make(**overrides)


def test_load_dir(tmp_path):
    (tmp_path / "ok").mkdir()
    make(name="b").save(str(tmp_path / "ok" / "1"))
    make(name="a").save(str(tmp_path / "ok" / "2"))
    (tmp_path / "ok" / "nested").mkdir()
    make(name="c").save(str(tmp_path / "ok" / "nested" / "3"))
    assert list(probe.load_dir(str(tmp_path / "ok"))) == ["b", "a"]

    (tmp_path / "dup").mkdir()
    make(name="x").save(str(tmp_path / "dup" / "1"))
    make(name="x").save(str(tmp_path / "dup" / "2"))
    with pytest.raises(ValueError, match="duplicate"):
        probe.load_dir(str(tmp_path / "dup"))

    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no probes"):
        probe.load_dir(str(tmp_path / "empty"))


class CharTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(m["content"] for m in messages)
        return text + "A:" if add_generation_prompt and not tokenize else text

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return {"input_ids": [0, *ids] if add_special_tokens else ids}


def test_last_turn_start_counts_rendered_prefix():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert probe.last_turn_start(CharTokenizer(), messages) == len("hiA:")
```

The `from sonde import probe, fingerprint` inside the subprocess string is test input, not an import of this module, so `force-single-line` does not apply to it.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_probe.py -q`
Expected: collection error: `ImportError: cannot import name 'probe' from 'sonde'`

- [ ] **Step 3: Move `fingerprint.py` verbatim**

```bash
cp ~/dev/personal/mechanica/mechanica/fingerprint.py sonde/fingerprint.py
diff ~/dev/personal/mechanica/mechanica/fingerprint.py sonde/fingerprint.py && echo IDENTICAL
```

Expected: `IDENTICAL`. Do not edit or reformat this file; `pyproject.toml` excludes it from ruff (with `force-exclude`, so pre-commit leaves it alone too).

- [ ] **Step 4: Write `sonde/probe.py` (artifact, validation, I/O)**

```python
"""The portable probe artifact sonde exports and mechanica serves.

numpy and the stdlib only, so vLLM workers can import it. The artifact is
data, never code: `np.savez` out, `np.load(allow_pickle=False)` in.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

import numpy as np

FORMAT = 1
WINDOWS = frozenset({"prompt", "response", "all", "last_turn"})
KINDS = frozenset({"linear", "attention"})
LINEAR_POOLINGS = frozenset({"mean", "last", "max", "rolling_mean"})


def last_turn_start(tokenizer, messages: list[dict]) -> int:
    """Token index where a conversation's last message starts.

    The fit-time side of the `last_turn` window; mechanica computes the same
    boundary at serve time. Renders then tokenizes, because
    `apply_chat_template(tokenize=True)` returns a dict in transformers 5.
    `tokenizer` is duck-typed so this module imports nothing.

    Args:
        tokenizer: A Hugging Face tokenizer with a chat template.
        messages: The conversation; its last message is the turn to read.

    Returns:
        len(tokens of messages[:-1] rendered with the generation prompt).
    """
    text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _vector(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"probe {name} must be a nonempty 1-D array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"probe {name} must be finite")
    return value


@dataclasses.dataclass
class Probe:
    """A trained probe: `sigmoid(pool(acts @ w + bias))` at one block.

    Attributes:
        name: Probe name; the key in `load_dir`.
        kind: "linear" or "attention".
        w: [H] float32 score (linear) or value (attention) direction.
        bias: Added to the pooled logit.
        block: Reads the residual stream output by decoder block `block`.
        window: Token window read, one of WINDOWS.
        pooling: mean | last | max | rolling_mean (linear); "attention".
        threshold: Flag threshold on the `pooled_score` scale, in [0, 1].
        model: Model name the probe was fitted on.
        engine: "<backend>==<ver>[+<lib>==<ver-or-sha>]".
        model_fingerprint: `fingerprint.checkpoint_fingerprint` at fit time.
        prompt_format: `fingerprint.prompt_format(...)` or "raw".
        q: [H] float32 attention query; None for linear probes.
        rolling_window: Window length iff pooling == "rolling_mean".
        adapters: LoRA adapters the probe was validated on.
        escalate_threshold: mechanica's review band; sonde never sets it.
        metrics: Eval card of measured values.
    """

    name: str
    kind: str
    w: np.ndarray
    bias: float
    block: int
    window: str
    pooling: str
    threshold: float
    model: str
    engine: str
    model_fingerprint: str | None
    prompt_format: str | None
    q: np.ndarray | None = None
    rolling_window: int | None = None
    adapters: tuple[str, ...] = ()
    escalate_threshold: float | None = None
    metrics: dict | None = None

    def __post_init__(self) -> None:
        self.w = _vector(self.w, "w")
        self.bias = float(self.bias)
        if not np.isfinite(self.bias):
            raise ValueError("probe bias must be finite")
        self.threshold = float(self.threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("probe threshold must be finite and in [0, 1]")
        if (
            isinstance(self.block, bool)
            or int(self.block) != self.block
            or self.block < 0
        ):
            raise ValueError("probe block must be a nonnegative integer")
        self.block = int(self.block)
        for attr in ("name", "model", "engine"):
            if not _nonempty_str(getattr(self, attr)):
                raise ValueError(f"probe {attr} must be a nonempty string")
        for attr in ("model_fingerprint", "prompt_format"):
            value = getattr(self, attr)
            if value is not None and not _nonempty_str(value):
                raise ValueError(f"probe {attr} must be nonempty or None")
        if self.window not in WINDOWS:
            raise ValueError(f"probe window must be one of {sorted(WINDOWS)}")
        if self.kind not in KINDS:
            raise ValueError(f"probe kind must be one of {sorted(KINDS)}")
        if (self.q is None) != (self.kind == "linear"):
            raise ValueError("probe q is required iff kind is attention")
        if self.q is not None:
            self.q = _vector(self.q, "q")
            if self.q.shape != self.w.shape:
                raise ValueError("probe q and w must have the same shape")
        poolings = LINEAR_POOLINGS if self.kind == "linear" else {"attention"}
        if self.pooling not in poolings:
            raise ValueError(
                f"{self.kind} probe pooling must be one of {sorted(poolings)}"
            )
        if self.pooling == "rolling_mean":
            if (
                isinstance(self.rolling_window, bool)
                or not isinstance(self.rolling_window, int)
                or self.rolling_window < 1
            ):
                raise ValueError("rolling_mean needs rolling_window >= 1")
        elif self.rolling_window is not None:
            raise ValueError("rolling_window is set only for rolling_mean")
        self.adapters = tuple(sorted(self.adapters or ()))
        if not all(_nonempty_str(adapter) for adapter in self.adapters):
            raise ValueError("probe adapters must be nonempty strings")
        if self.escalate_threshold is not None:
            self.escalate_threshold = float(self.escalate_threshold)
            if not 0.0 <= self.escalate_threshold <= self.threshold:
                raise ValueError(
                    "probe escalate_threshold must be between 0 and threshold"
                )

    def save(self, path: str) -> str:
        """Writes the probe atomically.

        Args:
            path: Target path; ".npz" is appended if missing.

        Returns:
            The path written.
        """
        meta = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("w", "q")
        }
        meta["format"] = FORMAT
        arrays = {"w": self.w}
        if self.q is not None:
            arrays["q"] = self.q
        final = path if path.endswith(".npz") else f"{path}.npz"
        tmp = f"{final}.tmp"
        try:
            with open(tmp, "wb") as handle:
                np.savez(
                    handle, allow_pickle=False, meta=json.dumps(meta), **arrays
                )
            os.replace(tmp, final)
        except BaseException:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise
        return final

    @classmethod
    def load(cls, path: str) -> Probe:
        """Reads a probe written by `save`.

        Args:
            path: An `.npz` probe artifact.

        Returns:
            The validated probe.

        Raises:
            ValueError: On an unknown format or meta that does not build a
                Probe, such as a legacy mechanica artifact.
        """
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            arrays = {k: archive[k] for k in ("w", "q") if k in archive}
        if meta.pop("format", None) != FORMAT:
            raise ValueError(f"{path}: not a format-{FORMAT} sonde probe")
        try:
            return cls(**meta, **arrays)
        except TypeError as e:
            raise ValueError(f"{path}: not a sonde probe ({e})") from e


def load_dir(directory: str) -> dict[str, Probe]:
    """Loads every `*.npz` in a directory (not recursive), in filename order.

    Args:
        directory: A probe directory.

    Returns:
        Probes keyed by name.

    Raises:
        ValueError: On a duplicate probe name or an empty directory.
    """
    probes: dict[str, Probe] = {}
    for path in sorted(pathlib.Path(directory).glob("*.npz")):
        probe = Probe.load(str(path))
        if probe.name in probes:
            raise ValueError(f"duplicate probe {probe.name!r} in {directory}")
        probes[probe.name] = probe
    if not probes:
        raise ValueError(f"no probes in {directory}")
    return probes
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/test_probe.py -q`
Expected: `7 passed`

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff format --check sonde tests && uv run ruff check sonde tests && uv run pyright`
Expected: `3 files already formatted`, `All checks passed!`, `0 errors, 0 warnings, 0 informations`

(`ruff` runs on `sonde tests` because `experiments/` and `examples/` stay until phase 4 and do not pass this config.)

- [ ] **Step 7: Commit**

```bash
git add sonde/fingerprint.py sonde/probe.py tests/test_probe.py
git commit -m "feat: numpy-only probe artifact and verbatim fingerprint module

The Probe fields follow spec §5, with format 1. It refuses legacy
mechanica npz files and unknown formats, and save is atomic.
fingerprint.py is copied byte for byte from mechanica."
```

---

### Task 1.4: Scoring, pooling and serving checks on `Probe`

**Files:**
- Modify: `sonde/probe.py`. Add `_sigmoid` and `_pool` above `def _nonempty_str`, and add nine methods inside `class Probe` above `def save`.
- Test: `tests/test_probe.py` (append)

**Interfaces:**
- Consumes: `probe.Probe` and its validation from Task 1.3.
- Produces:
  - `Probe.logits(self, acts) -> np.ndarray`: `[T,H]` or `[H]` to `[T]` float32 `acts @ w + bias`. Linear only; attention raises `ValueError`.
  - `Probe.pooled_score_from_logits(self, logits) -> float`: input `[T]` logits **including bias**, as `logits()` returns them. Linear only. mechanica's old method took bias-free `w·x`; its migration must add `bias` first.
  - `Probe.pooled_logit(self, acts) -> float`: `[T,H]` or `[H]` to the pooled logit, before the sigmoid. RL rewards use it (`r' = r - λ·logit`) because the sigmoid saturates.
  - `Probe.pooled_score(self, acts) -> float`: `[T,H]` or `[H]`; `sigmoid(pooled_logit(acts))`. Phase 3's torch modules must match this within 1e-5.
  - `Probe.flag(self, acts) -> bool`
  - `Probe.escalates(self, score: float) -> bool`
  - `Probe.serves(self, fingerprint: str | None, adapter: str | None, prompt_format: str | None = "unchecked") -> str | None`
  - `Probe.vllm_aux_layer(self) -> int`: returns `block + 1`.
  - `Probe.backend(self) -> str`: returns `engine.split("==", 1)[0]`.
  - Module-private: `_pool(logits [T], pooling, rolling_window: int) -> float`, and `_sigmoid`, which works in float32 and clips to ±30.

- [ ] **Step 1: Append the failing tests to `tests/test_probe.py`**

```python
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def test_serves_refusals():
    p = make(
        model_fingerprint="fp",
        prompt_format="tmpl:x",
        adapters=("lora-a",),
    )
    assert p.serves("fp", None) is None
    assert p.serves("fp", "lora-a", "tmpl:x") is None
    refusals = [
        (make(model_fingerprint=None).serves("fp", None), "no checkpoint"),
        (p.serves(None, None), "could not be fingerprinted"),
        (p.serves("other", None), "checkpoint mismatch"),
        (p.serves("fp", "lora-b"), "adapter"),
        (p.serves("fp", None, "tmpl:y"), "prompt format mismatch"),
        (p.serves("fp", None, None), "could not be determined"),
        (
            make(model_fingerprint="fp", prompt_format=None).serves(
                "fp", None, "raw"
            ),
            "does not record",
        ),
    ]
    for reason, expected in refusals:
        assert reason is not None and expected in reason, (reason, expected)


def test_vllm_aux_layer_and_backend():
    assert make(block=3).vllm_aux_layer() == 4
    assert make().backend() == "hf"
    vllm = make(engine="vllm==0.30.0+nnsight==b717807")
    assert vllm.backend() == "vllm"


def test_linear_poolings():
    logits = [0.0, 6.0, 0.0, 0.0, 3.0, 3.0, 3.0]
    expected = {"mean": 15 / 7, "last": 3.0, "max": 6.0}
    for pooling, z in expected.items():
        p = make(pooling=pooling)
        assert p.pooled_score_from_logits(logits) == pytest.approx(sigmoid(z))
    rolling = make(pooling="rolling_mean", rolling_window=3)
    assert rolling.pooled_score_from_logits(logits) == pytest.approx(
        sigmoid(3.0)
    )
    assert rolling.pooled_score_from_logits([1.0, 2.0]) == pytest.approx(
        sigmoid(1.5)
    )
    assert make().pooled_score_from_logits([-1e9]) == pytest.approx(
        sigmoid(-30.0)
    )


def test_pooled_score_from_logits_parity():
    acts = np.random.default_rng(0).normal(size=(7, H)).astype(np.float32)
    for pooling, window in [
        ("mean", None),
        ("last", None),
        ("max", None),
        ("rolling_mean", 3),
        ("rolling_mean", 10),
    ]:
        p = make(pooling=pooling, rolling_window=window)
        score = p.pooled_score(acts)
        assert score == p.pooled_score_from_logits(p.logits(acts))
        assert score == pytest.approx(sigmoid(p.pooled_logit(acts)))
        assert p.flag(acts) == (score >= p.threshold)
        assert p.pooled_score(acts[0]) == p.pooled_score(acts[:1])


def test_attention_pooled_score():
    p = make(
        kind="attention",
        pooling="attention",
        w=[1.0, 5.0, 0.0, 0.0],
        q=[math.log(3.0), 0.0, 0.0, 0.0],
    )
    acts = np.eye(H, dtype=np.float32)[:2]
    assert p.pooled_logit(acts) == pytest.approx(0.75 + 1.25 - 0.25)
    assert p.pooled_score(acts) == pytest.approx(sigmoid(0.75 + 1.25 - 0.25))
    assert p.pooled_score(acts[0]) == pytest.approx(sigmoid(1.0 - 0.25))
    with pytest.raises(ValueError):
        p.logits(acts)
    with pytest.raises(ValueError):
        p.pooled_score_from_logits([0.0])


def test_escalates():
    p = make(threshold=0.8, escalate_threshold=0.5)
    assert p.escalates(0.6)
    assert not p.escalates(0.4)
    assert not p.escalates(0.8)
    assert not make().escalates(0.4)
```

What these pin down:
- `rolling_mean` with window 3 picks the max window mean (3.0), not the max token (6.0). With `T=2 < 3` it falls back to the mean of all T (1.5).
- `pooled_score` is `sigmoid(pooled_logit)` for every kind and pooling, so an RL reward on the logit ranks samples exactly as the score does.
- The attention case works out by hand: softmax weights `[3/4, 1/4]` over values `[1, 5]` give 2.0, and the bias brings it to 1.75.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run ruff format tests && uv run pytest tests/test_probe.py -q`
Expected: `6 failed, 7 passed`. Every failure is `AttributeError: 'Probe' object has no attribute ...`.

- [ ] **Step 3: Add the module-private pooling helpers**

In `sonde/probe.py`, insert directly above `def _nonempty_str(value) -> bool:`:

```python
def _sigmoid(z) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _pool(logits: np.ndarray, pooling: str, rolling_window: int) -> float:
    """Reduces per-token logits [T] to one logit."""
    if pooling == "mean":
        return float(logits.mean())
    if pooling == "last":
        return float(logits[-1])
    if pooling == "max":
        return float(logits.max())
    if logits.size < rolling_window:
        return float(logits.mean())
    windows = np.lib.stride_tricks.sliding_window_view(logits, rolling_window)
    return float(windows.mean(axis=1).max())


```

- [ ] **Step 4: Add the scoring and serving methods**

Inside `class Probe`, insert directly above `    def save(self, path: str) -> str:`:

```python
    def logits(self, acts) -> np.ndarray:
        """Per-token logits of a linear probe.

        Args:
            acts: [T, H] or [H] activations at `block`.

        Returns:
            [T] float32 `acts @ w + bias`.

        Raises:
            ValueError: For an attention probe, which has no per-token logit.
        """
        if self.kind != "linear":
            raise ValueError("logits() is defined for linear probes only")
        acts = np.atleast_2d(np.asarray(acts, dtype=np.float32))
        return acts @ self.w + np.float32(self.bias)

    def pooled_score_from_logits(self, logits) -> float:
        """Pools logits that already include the bias, then applies sigmoid.

        Unlike mechanica's old method, `logits` must include `bias`, as
        `logits()` returns.

        Args:
            logits: [T] per-token logits.

        Returns:
            The probe score in [0, 1].

        Raises:
            ValueError: For an attention probe.
        """
        if self.kind != "linear":
            raise ValueError("attention probes pool activations, not logits")
        logits = np.atleast_1d(np.asarray(logits, dtype=np.float32))
        return float(
            _sigmoid(_pool(logits, self.pooling, self.rolling_window or 0))
        )

    def pooled_logit(self, acts) -> float:
        """The pooled logit, before the sigmoid.

        RL rewards use this (r' = r - lambda * logit) because the sigmoid
        saturates.

        Args:
            acts: [T, H] activations over the window, or one [H] row.

        Returns:
            The window's logit; pooling acts on logits.
        """
        if self.q is None:
            logits = self.logits(acts)
            return _pool(logits, self.pooling, self.rolling_window or 0)
        acts = np.atleast_2d(np.asarray(acts, dtype=np.float32))
        attn = acts @ self.q
        attn = np.exp(attn - attn.max())
        return float(
            attn @ (acts @ self.w) / attn.sum() + np.float32(self.bias)
        )

    def pooled_score(self, acts) -> float:
        """The score the threshold was calibrated on.

        Args:
            acts: [T, H] activations over the window, or one [H] row.

        Returns:
            sigmoid(pooled_logit(acts)), in [0, 1].
        """
        return float(_sigmoid(self.pooled_logit(acts)))

    def flag(self, acts) -> bool:
        """Whether `pooled_score(acts) >= threshold`."""
        return self.pooled_score(acts) >= self.threshold

    def escalates(self, score: float) -> bool:
        """Whether a score is in the review band: uncertain, not blocked."""
        if self.escalate_threshold is None:
            return False
        return self.escalate_threshold <= score < self.threshold

    def serves(
        self,
        fingerprint: str | None,
        adapter: str | None,
        prompt_format: str | None = "unchecked",
    ) -> str | None:
        """Why this probe must not score the running model, or None if it may.

        Fail-closed: a probe or server that cannot state its checkpoint or
        prompt format is refused.

        Args:
            fingerprint: Serving checkpoint fingerprint.
            adapter: The request's LoRA adapter, or None for the base model.
            prompt_format: Served prompt format; "unchecked" skips the check.

        Returns:
            A refusal reason, or None when the probe may serve.
        """
        if self.model_fingerprint is None:
            return "probe carries no checkpoint fingerprint; retrain it"
        if fingerprint is None:
            return "serving checkpoint could not be fingerprinted"
        if fingerprint != self.model_fingerprint:
            return (
                f"checkpoint mismatch (probe={self.model_fingerprint}, "
                f"serving={fingerprint}): fine-tuning invalidates a probe"
            )
        if adapter is not None and adapter not in self.adapters:
            return f"adapter {adapter!r} is not one this probe was validated on"
        if prompt_format == "unchecked":
            return None
        if self.prompt_format is None:
            return "probe does not record its prompt format; retrain it"
        if prompt_format is None:
            return "served prompt format could not be determined"
        if prompt_format != self.prompt_format:
            return (
                f"prompt format mismatch (probe={self.prompt_format}, "
                f"serving={prompt_format})"
            )
        return None

    def vllm_aux_layer(self) -> int:
        """The vLLM `extract_hidden_states` aux id that reads `block`."""
        return self.block + 1

    def backend(self) -> str:
        """The backend token of `engine`, e.g. "vllm" or "hf"."""
        return self.engine.split("==", 1)[0]

```

`pooled_logit` branches on `self.q is None` rather than `self.kind`. `__post_init__` makes the two equivalent, and pyright can narrow `q` from the first. `rolling_window or 0` only satisfies the `int` parameter: validation guarantees a value of at least 1 whenever pooling is `rolling_mean`.

- [ ] **Step 5: Run the whole phase's checks**

```bash
uv run ruff format --check sonde tests && uv run ruff check sonde tests
uv run pyright
uv run pytest -m "not gpu" -q
diff ~/dev/personal/mechanica/mechanica/fingerprint.py sonde/fingerprint.py && echo IDENTICAL
```

Expected:
1. `3 files already formatted` and `All checks passed!`
2. `0 errors, 0 warnings, 0 informations`
3. `13 passed`
4. `IDENTICAL`

- [ ] **Step 6: Commit**

```bash
git add sonde/probe.py tests/test_probe.py
git commit -m "feat: probe scoring, pooling and serving checks

Pooling acts on logits, and the sigmoid comes last (float32, clipped to
±30). Covers mean, last, max, rolling_mean (mean over all tokens when
T < window) and attention. Also adds mechanica's fail-closed serves(),
escalates(), vllm_aux_layer() = block + 1 and backend()."
```

<!-- skipped: Issues 4 and 6 also say phase 1 can drop its ruff caveat and that "every phase can run `ruff format --check .`". Most of that fix was applied: `*.md` is excluded, and the README reason for the caveat is gone. The part that says phase 1 can lint `.` is wrong. I checked it in a scratch copy: with this config, `experiments/` and `examples/` (deleted only in phase 4) give 110 `ruff check` errors and 10 files to reformat. So phase 1 still runs ruff on `sonde tests`, and the note now gives that as the reason. Once phase 4 deletes those directories, `ruff format --check .` and `ruff check .` are clean.
Fixes outside phase 1 that are not in this markdown: phase 2 must split its combined `from sonde import a, b` lines; phase 3 should drop its force-single-line assumption bullet; phase 4 must delete Task 4.4 Step 1 and the `git add -u pyrightconfig.json` line in Step 7. -->

---

## Phase 2: Config, data, HF backend, extraction, storage

**Goal.** Build the path from YAML to activation shards on disk, using only the hf backend. That covers `config.py` (every section model, its load-time checks, `load` with overrides and recipe lookup, `score_data`, `resolve_blocks` and `run_dir`), `data.py` (loading, rendering all four windows, drop counting, splitting), the hf half of `backends.py` (the single-slot model cache, `window_pool`, and a right-padded, masked `forward` under `torch.no_grad()`), `store.py` (run.json, shards, manifest, resume, `read_layer`) and `extract.py` (`extract_hash`, `preflight`, `extract_to`, `run`). `backend: vllm` raises `NotImplementedError("vllm backend lands in phase 5")`. `generate` is not built here; it comes in phase 6. **Done when** `uv run --extra hf --extra dev pytest tests/test_config.py tests/test_data.py tests/test_extract.py -q` passes (47 tests, gpt2 batch invariance included), and `ruff check`, `ruff format --check` and `pyright` are clean on the five new modules.

**Before you start.**
- Phase 1 is merged. `sonde/probe.py` (`WINDOWS`, `last_turn_start`) and `sonde/fingerprint.py` (`RAW`, `prompt_format`, `checkpoint_fingerprint`) exist. `pyproject.toml` has the `[core]`, `[hf]` and `[dev]` extras, `line-length = 80`, `[tool.ruff.lint.isort] force-single-line = true` with `known-first-party = ["sonde"]`, and no pytest `pythonpath`.
  - Every `from sonde import …` in this phase imports one module per line, in alphabetical order. Without `force-single-line`, ruff's isort merges those lines and reports I001.
- Run every command from the repo root. `uv run --extra hf --extra dev …` uses the hf environment.
- The first run of `test_data.py` / `test_extract.py` downloads `gpt2` and the `hf-internal-testing/tiny-random-LlamaForCausalLM` tokenizer (a few MB). CI caches `~/.cache/huggingface`.

**Verified in a scratch venv** (Python 3.12.14, macOS arm64: nnterp 1.3.0, nnsight 0.7.0, transformers 5.18.0, torch 2.14.1, pydantic 2.13.5, datasets 5.0.1, safetensors 0.8.0). Every code block in this phase was run there exactly as written. All 47 tests passed with the model on MPS (nnterp's default `device_map="auto"`) and again forced to CPU. Ruff (line-length 80, `E W F I UP B SIM RUF`) and pyright (basic) were clean. Every import block in this phase (the final form of each file, plus the intermediate forms after Tasks 2.2 and 2.5) also passed `ruff check` and `ruff format --check` with isort `force-single-line = true`. Without that setting, the same blocks fail I001.
- `nnterp.StandardizedTransformer(name, revision=None, dtype=torch.float32)` leaves the model on `meta` until the first trace, so `model.device` reads `meta` and moving inputs there fails with "Cannot copy out of meta tensor". Passing `dispatch=True` loads the real weights up front. nnterp defaults `device_map` to `"auto"` (`standardized_transformer.py:99`).
- `model.num_layers`, `model.hidden_size` and `model.tokenizer` exist (gpt2: 12, 768). `model.layers_output[L]` is a `[B, T, H]` tensor.
- `model.trace({"input_ids": LongTensor[B, T], "attention_mask": LongTensor[B, T]})` takes the dict as is.
  - Right padding plus the mask, bs 1 vs bs 4: max |Δ| was 1.7e-4 on CPU fp32 and 4e-3 on MPS fp32, against activations up to 2.9e3.
  - Mutating `forward` to left-pad with an all-ones mask makes the test fail with |Δ| = 102 at block 0, so the test can tell the two apart.
- Inside the trace body, three things work: calling a plain module function (`window_pool`), `.to("cpu", copy=True)`, and appending `nnsight.save(x)` to a list or dict created before the `with`.
  - One thing does not: binding a new name inside the trace through a dict comprehension raised `UnboundLocalError` after the block. Containers are therefore always created before the `with`.
- `.to("cpu", copy=True)` matters on CPU, where `.cpu()` is a no-op. A `[T, H]` slice view would otherwise keep the whole `[B, T, H]` batch alive until its shard flushes.
- gpt2's tokenizer has `chat_template is None` and adds no BOS. The tiny-random-Llama tokenizer's template emits `<s>`, and `tok(text)` adds a second one (`[1, 1, …]`); `add_special_tokens=False` gives one.
- pydantic 2.13 does not warn about a field named `model` (`-W error::UserWarning` passed). `torch.bfloat16.itemsize == 2`.

**Deliberate choices beyond the contract** (each one small):
- `Loaded.model` and `Loaded.tokenizer` are annotated `typing.Any`, not `object`. With `object`, pyright rejects `.trace` and `.device`. Names and positions are unchanged.
- `extract_hash` also covers `seed`, but only when `data.limit` is set. The seed picks the subsample, so without it a reseeded run would resume on top of shards holding different rows.
- `forward` does not sort a batch by length: sorting inside one batch saves no padding. `extract_to` batches in sample order, which is what keeps shards resumable. A `# ponytail:` comment marks this.
- `data.system` is prepended only to `text` rows, as §6 states. `messages` rows are used exactly as given.
- `render` raises `ValueError` (not `Drop`) for `window: last_turn` with `format: raw`, because that is a config mistake, not a property of one row. `encode_all` raises when every row drops.

**Review focus pinned by tests in this phase:**
- (a) A row whose window is emptied by `max_length` is dropped and counted, never stored empty. Covered by `test_drops_are_counted_never_empty` (Task 2.3) and `test_window_cut_by_max_length_is_dropped_not_empty` (Task 2.6).
- (b) Re-running extract with an edited config raises before any model load. Covered by `test_edited_config_refuses_before_model_load` (Task 2.6).
- (c) A split that would lack a class raises a `ValueError` naming the split. Covered by `test_split_missing_class_names_the_split` (Task 2.2).
- (d) `format: chat` with gpt2's template-less tokenizer raises a `ValueError` saying `set data.format: raw`. Covered by `test_chat_without_template_says_use_raw` (Task 2.3).

---

### Task 2.1: Config models, load-time checks, overrides

**Files:**
- Create: `sonde/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from sonde (pydantic, yaml).
- Produces:
  - `ModelConfig`, `DataConfig`, `Every`, `ExtractConfig`, `ProbeConfig`, `ScoreEntry`, `GenerateConfig`, `SteerConfig`, `OutputConfig`, `RunConfig`: pydantic v2 with `extra="forbid"`, fields and defaults as in the contract.
  - `RECIPES_DIR: pathlib.Path`
  - `load(source: str, overrides: Sequence[str] = ()) -> RunConfig`
  - `apply_override(raw: dict, item: str) -> None`
  - `score_data(cfg: RunConfig, entry: ScoreEntry) -> DataConfig`
  - `resolve_blocks(spec: Literal["all"] | list[int] | Every, num_layers: int) -> list[int]`
  - `run_dir(cfg: RunConfig) -> pathlib.Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from __future__ import annotations

import pathlib

import pydantic
import pytest
import yaml

from sonde import config

BASE = {
    "name": "t",
    "model": {"name": "gpt2"},
    "data": {"path": "d.jsonl"},
    "extract": {},
    "probe": {},
}


def _cfg(**sections) -> config.RunConfig:
    raw = {k: dict(v) if isinstance(v, dict) else v for k, v in BASE.items()}
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key].update(value)
        else:
            raw[key] = value
    return config.RunConfig.model_validate(raw)


def _bad(match: str, **sections) -> None:
    with pytest.raises(pydantic.ValidationError, match=match):
        _cfg(**sections)


def test_defaults_validate():
    cfg = _cfg()
    assert cfg.steps == ["extract", "train"]
    assert cfg.extract.layers == "all"
    assert cfg.output.dir == "runs"


def test_unknown_key_error_names_its_path():
    _bad(r"data\.pathh", data={"pathh": "x"})
    _bad(r"probe\.kindd", probe={"kindd": "linear"})


def test_step_needs_its_sections():
    _bad("needs model", model=None)
    _bad("needs score", steps=["score"])
    _bad("needs generate", steps=["generate"])
    _bad("needs steer", steps=["steer"], generate={})
    _bad("needs probe", probe=None)
    _bad("pooled needs a probe", probe=None, steps=["extract"])
    cfg = _cfg(probe=None, extract={"keep": "tokens"}, steps=["extract"])
    assert cfg.probe is None


def test_data_source_and_text_checks():
    _bad("exactly one of path / hf", data={"hf": "org/ds"})
    _bad("exactly one of path / hf", data={"path": None})
    _bad("exactly one of text / messages", data={"text": None})
    _bad("raw needs text rows", data={"messages": "m", "format": "raw"})
    cfg = _cfg(data={"messages": "conv"})
    assert cfg.data.text is None and cfg.data.messages == "conv"
    _bad("exactly one of text / messages", data={"text": "t", "messages": "m"})


def test_split_must_sum_to_one():
    _bad("sum to 1", data={"split": [0.5, 0.2, 0.2]})
    _bad("sum to 1", data={"split": [1.2, -0.1, -0.1]})


def test_layers_forms():
    assert _cfg(extract={"layers": [5, 1, 5]}).extract.layers == [1, 5]
    every = _cfg(extract={"layers": {"every": 4}}).extract.layers
    assert isinstance(every, config.Every) and every.every == 4
    _bad("layers", extract={"layers": [-1]})
    _bad("layers", extract={"layers": {"every": 0}})
    _bad("layers", extract={"layers": "some"})


def test_probe_cross_field_checks():
    tokens = {"keep": "tokens"}
    _bad("pooled needs probe.kind: linear", probe={"pooling": "max"})
    _bad(
        "pooled needs probe.kind: linear",
        probe={"kind": "attention", "pooling": "attention"},
    )
    _bad(
        "rolling_window is set iff",
        extract=tokens,
        probe={"pooling": "rolling_mean"},
    )
    _bad("rolling_window is set iff", probe={"rolling_window": 3})
    _bad(
        "rolling_window must be >= 1",
        extract=tokens,
        probe={"pooling": "rolling_mean", "rolling_window": 0},
    )
    _bad(
        "kind attention goes with pooling attention",
        extract=tokens,
        probe={"kind": "attention"},
    )
    _bad("needs probe.max_fpr", probe={"max_fpr": None})
    _bad("needs data.group", probe={"select": "group_auroc"})
    grouped = _cfg(data={"group": "g"}, probe={"select": "group_auroc"})
    assert grouped.probe.select == "group_auroc"
    no_fpr = _cfg(probe={"select": "auroc", "max_fpr": None})
    assert no_fpr.probe.max_fpr is None
    _bad("diff_means needs", probe={"init": "diff_means"})
    _bad(
        "diff_means needs",
        extract=tokens,
        probe={
            "init": "diff_means",
            "epochs": 0,
            "kind": "attention",
            "pooling": "attention",
        },
    )
    ok = _cfg(
        extract=tokens, probe={"pooling": "rolling_mean", "rolling_window": 4}
    )
    assert ok.probe.rolling_window == 4
    assert _cfg(probe={"init": "diff_means", "epochs": 0}).probe.epochs == 0


def test_window_response_needs_data_response():
    _bad("needs data.response", extract={"window": "response"})
    cfg = _cfg(extract={"window": "response"}, data={"response": "r"})
    assert cfg.data.response == "r"


def test_temperature_non_negative():
    _bad("temperature", generate={"temperature": -0.1})


def test_load_applies_overrides(tmp_path: pathlib.Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(BASE))
    cfg = config.load(
        str(path),
        [
            "extract.layers=[3, 1]",
            "seed=7",
            "generate.temperature=0",
            "steps=[extract]",
        ],
    )
    assert cfg.extract.layers == [1, 3]
    assert cfg.seed == 7
    assert cfg.generate.temperature == 0.0
    assert cfg.steps == ["extract"]


def test_apply_override_parses_yaml_and_rejects_garbage():
    raw = {"data": None}
    config.apply_override(raw, "data.label_map={a: 1, b: 0}")
    assert raw == {"data": {"label_map": {"a": 1, "b": 0}}}
    config.apply_override(raw, "data.hf=")
    assert raw["data"]["hf"] is None
    with pytest.raises(ValueError, match="key=value"):
        config.apply_override(raw, "seed")


def test_recipe_lookup(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setattr(config, "RECIPES_DIR", tmp_path)
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(BASE))
    assert config.load("tiny").name == "t"
    with pytest.raises(ValueError, match=r"unknown recipe 'nope'.*\['tiny'\]"):
        config.load("nope")


def test_score_data_inherits_from_data():
    cfg = _cfg(
        data={"label_map": {"y": 1, "n": 0}, "max_length": 64},
        score=[
            {
                "name": "ood",
                "hf": "org/ood",
                "hf_split": "test",
                "messages": "conv",
                "split": [1, 0, 0],
            }
        ],
    )
    merged = config.score_data(cfg, cfg.score[0])
    assert merged.path is None and merged.hf == "org/ood"
    assert merged.text is None and merged.messages == "conv"
    assert merged.hf_split == "test"
    assert merged.label_map == {"y": 1, "n": 0} and merged.max_length == 64
    assert merged.split == cfg.data.split


def test_resolve_blocks():
    assert config.resolve_blocks("all", 3) == [0, 1, 2]
    assert config.resolve_blocks(config.Every(every=4), 10) == [0, 4, 8]
    assert config.resolve_blocks([2, 0], 3) == [0, 2]
    with pytest.raises(ValueError, match=r"\[12\] out of range"):
        config.resolve_blocks([0, 12], 12)


def test_run_dir():
    cfg = _cfg(output={"dir": "out"})
    assert config.run_dir(cfg) == pathlib.Path("out") / "t"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra hf --extra dev pytest tests/test_config.py -q`
Expected: collection error `ImportError: cannot import name 'config' from 'sonde'`.

- [ ] **Step 3: Write the implementation**

Create `sonde/config.py`:

```python
"""Run configuration: one pydantic model per YAML section, checked at load."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Literal

import pydantic
import yaml

RECIPES_DIR = pathlib.Path(__file__).parent / "recipes"

_NEEDS = {
    "extract": ("model", "data", "extract"),
    "train": ("data", "probe"),
    "score": ("model", "data", "score"),
    "generate": ("model", "data", "generate"),
    "steer": ("model", "data", "generate", "steer"),
}


class _Section(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


class ModelConfig(_Section):
    name: str
    revision: str | None = None
    dtype: str = "bfloat16"
    backend: Literal["hf", "vllm"] = "hf"
    vllm: dict = pydantic.Field(default_factory=dict)


class DataConfig(_Section):
    path: str | None = None
    hf: str | None = None
    hf_config: str | None = None
    hf_split: str = "train"
    text: str | None = "text"
    messages: str | None = None
    response: str | None = None
    label: str | None = "label"
    id: str | None = "id"
    group: str | None = None
    label_map: dict | None = None
    limit: int | None = pydantic.Field(default=None, ge=1)
    format: Literal["chat", "raw"] = "chat"
    system: str | None = None
    max_length: int = pydantic.Field(default=2048, ge=1)
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)

    @pydantic.model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.messages is not None and "text" not in self.model_fields_set:
            self.text = None
        if (self.path is None) == (self.hf is None):
            raise ValueError("data: set exactly one of path / hf")
        if (self.text is None) == (self.messages is None):
            raise ValueError("data: set exactly one of text / messages")
        if self.format == "raw" and self.messages is not None:
            raise ValueError("data.format: raw needs text rows, not messages")
        if min(self.split) < 0 or abs(sum(self.split) - 1) > 1e-6:
            raise ValueError(f"data.split {self.split} must be >= 0, sum to 1")
        return self


class Every(_Section):
    every: int = pydantic.Field(ge=1)


class ExtractConfig(_Section):
    layers: Literal["all"] | list[pydantic.NonNegativeInt] | Every = "all"
    window: Literal["prompt", "response", "all", "last_turn"] = "prompt"
    keep: Literal["pooled", "tokens"] = "pooled"
    batch_size: int = pydantic.Field(default=16, ge=1)
    shard_size: int = pydantic.Field(default=4096, ge=1)
    shard_bytes: int = pydantic.Field(default=2**31, ge=1)

    @pydantic.field_validator("layers")
    @classmethod
    def _dedupe(cls, v):
        return sorted(set(v)) if isinstance(v, list) else v


class ProbeConfig(_Section):
    kind: Literal["linear", "attention"] = "linear"
    pooling: Literal["mean", "last", "max", "rolling_mean", "attention"] = (
        "mean"
    )
    rolling_window: int | None = None
    init: Literal["random", "diff_means"] = "random"
    epochs: int = pydantic.Field(default=20, ge=0)
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = pydantic.Field(default=256, ge=1)
    patience: int = pydantic.Field(default=3, ge=0)
    max_fpr: float | None = 0.01
    select: Literal["auroc", "recall_at_fpr", "group_auroc"] = "recall_at_fpr"

    @pydantic.model_validator(mode="after")
    def _check(self) -> ProbeConfig:
        if (self.kind == "attention") != (self.pooling == "attention"):
            raise ValueError(
                "probe: kind attention goes with pooling attention, and only"
            )
        if (self.pooling == "rolling_mean") != (
            self.rolling_window is not None
        ):
            raise ValueError(
                "probe.rolling_window is set iff pooling is rolling_mean"
            )
        if self.rolling_window is not None and self.rolling_window < 1:
            raise ValueError("probe.rolling_window must be >= 1")
        if self.select == "recall_at_fpr" and self.max_fpr is None:
            raise ValueError(
                "probe.select: recall_at_fpr needs probe.max_fpr; set "
                "select: auroc to run without one"
            )
        if self.init == "diff_means" and (
            self.kind != "linear" or self.epochs != 0
        ):
            raise ValueError(
                "probe.init: diff_means needs kind: linear and epochs: 0"
            )
        return self


class ScoreEntry(_Section):
    name: str
    probe: str | None = None
    path: str | None = None
    hf: str | None = None
    hf_config: str | None = None
    hf_split: str | None = None
    text: str | None = None
    messages: str | None = None
    response: str | None = None
    label: str | None = None
    id: str | None = None
    group: str | None = None
    label_map: dict | None = None
    limit: int | None = None
    format: Literal["chat", "raw"] | None = None
    system: str | None = None
    max_length: int | None = None
    split: tuple[float, float, float] | None = None


class GenerateConfig(_Section):
    max_tokens: int = pydantic.Field(default=256, ge=1)
    temperature: float = pydantic.Field(default=0.7, ge=0)
    top_p: float = 1.0


class SteerConfig(_Section):
    probe: str | None = None
    mode: Literal["add", "ablate"] = "add"
    strengths: list[float] = pydantic.Field(default_factory=lambda: [0.0])


class OutputConfig(_Section):
    dir: str = "runs"
    overwrite: bool = False


class RunConfig(_Section):
    name: str
    seed: int = 0
    model: ModelConfig | None = None
    data: DataConfig | None = None
    extract: ExtractConfig | None = None
    probe: ProbeConfig | None = None
    score: list[ScoreEntry] = pydantic.Field(default_factory=list)
    generate: GenerateConfig | None = None
    steer: SteerConfig | None = None
    output: OutputConfig = pydantic.Field(default_factory=OutputConfig)
    steps: list[Literal["generate", "extract", "train", "score", "steer"]] = (
        pydantic.Field(default_factory=lambda: ["extract", "train"])
    )

    @pydantic.model_validator(mode="after")
    def _check(self) -> RunConfig:
        for step in self.steps:
            for section in _NEEDS[step]:
                if not getattr(self, section):
                    raise ValueError(
                        f"steps has {step!r}, which needs {section}"
                    )
        ext, probe = self.extract, self.probe
        if ext and ext.keep == "pooled":
            if probe is None and "extract" in self.steps:
                raise ValueError("extract.keep: pooled needs a probe section")
            if probe and (
                probe.kind != "linear" or probe.pooling not in ("mean", "last")
            ):
                raise ValueError(
                    "extract.keep: pooled needs probe.kind: linear with "
                    "probe.pooling mean or last; use extract.keep: tokens"
                )
        if (
            ext
            and ext.window == "response"
            and self.data
            and (self.data.response is None)
        ):
            raise ValueError("extract.window: response needs data.response")
        if (
            probe
            and probe.select == "group_auroc"
            and (self.data is None or self.data.group is None)
        ):
            raise ValueError("probe.select: group_auroc needs data.group")
        return self


def apply_override(raw: dict, item: str) -> None:
    """Sets one `a.b=v` override on a raw config dict, in place.

    Args:
        raw: The YAML mapping before validation.
        item: `dotted.key=value`; the value is parsed with `yaml.safe_load`.

    Raises:
        ValueError: If `item` has no `=` or an empty key.
    """
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise ValueError(f"override {item!r} is not key=value")
    *parents, leaf = key.split(".")
    node = raw
    for part in parents:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[leaf] = yaml.safe_load(value)


def load(source: str, overrides: Sequence[str] = ()) -> RunConfig:
    """Reads and validates a run config.

    Args:
        source: A `.yaml`/`.yml` path, or the bare name of a bundled recipe.
        overrides: `a.b=v` strings applied before validation.

    Returns:
        The validated config.

    Raises:
        ValueError: On an unknown recipe name (the message lists them).
        pydantic.ValidationError: On any schema or cross-field violation.
    """
    path = pathlib.Path(source)
    if path.suffix not in (".yaml", ".yml"):
        path = RECIPES_DIR / f"{source}.yaml"
        if not path.is_file():
            names = sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))
            raise ValueError(f"unknown recipe {source!r}; available: {names}")
    raw = yaml.safe_load(path.read_text()) or {}
    for item in overrides:
        apply_override(raw, item)
    return RunConfig.model_validate(raw)


def score_data(cfg: RunConfig, entry: ScoreEntry) -> DataConfig:
    """Merges a score entry over `cfg.data`.

    Setting either of `path`/`hf` (or `text`/`messages`) in the entry
    replaces both inherited keys of that pair.

    Args:
        cfg: The run config; `cfg.data` is the base.
        entry: Keys set here win.

    Returns:
        A validated data block for this entry.
    """
    over = entry.model_dump(
        exclude_unset=True, exclude={"name", "probe", "split"}
    )
    base = cfg.data.model_dump(exclude_unset=True) if cfg.data else {}
    for pair in (("path", "hf"), ("text", "messages")):
        if any(k in over for k in pair):
            base = {k: v for k, v in base.items() if k not in pair}
    return DataConfig.model_validate({**base, **over})


def resolve_blocks(
    spec: Literal["all"] | list[int] | Every, num_layers: int
) -> list[int]:
    """Expands `extract.layers` against a model's depth.

    Args:
        spec: `"all"`, a list of block indices, or `Every(every=k)`.
        num_layers: Decoder blocks in the model.

    Returns:
        Sorted, deduplicated block indices.

    Raises:
        ValueError: If any index is outside `[0, num_layers)`.
    """
    if spec == "all":
        blocks = range(num_layers)
    elif isinstance(spec, Every):
        blocks = range(0, num_layers, spec.every)
    else:
        blocks = spec
    blocks = sorted(set(blocks))
    bad = [b for b in blocks if not 0 <= b < num_layers]
    if bad:
        raise ValueError(
            f"extract.layers {bad} out of range: model has {num_layers} blocks"
        )
    return blocks


def run_dir(cfg: RunConfig) -> pathlib.Path:
    """Returns `<output.dir>/<name>`."""
    return pathlib.Path(cfg.output.dir) / cfg.name
```

Why some lines are as they are:
- A `messages` block that leaves `text` at its default clears `text`. That way `messages: conv` alone is valid, while an explicit `text` plus `messages` is still an error.
- `score_data` drops the inherited half of a source pair, so an `hf` score entry over a `path` data block works without a `path: null`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra hf --extra dev pytest tests/test_config.py -q`
Expected: `15 passed`.

- [ ] **Step 5: Lint**

Run: `uv run --extra hf --extra dev ruff format --check sonde/config.py tests/test_config.py && uv run --extra hf --extra dev ruff check sonde/config.py tests/test_config.py`
Expected: `2 files already formatted` then `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add sonde/config.py tests/test_config.py
git commit -m "feat(config): pydantic run config with load-time checks and overrides"
```

---

### Task 2.2: Row loading and the stratified group split

**Files:**
- Create: `sonde/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `config.DataConfig` (Task 2.1).
- Produces:
  - `Sample` dataclass: `id: str; text: str | None; messages: list[dict] | None; response: str | None; response_ids: list[int] | None; label: int | None; group: str | None; raw: dict`
  - `load_samples(src: config.DataConfig, seed: int) -> list[Sample]`
  - `split(labels: Sequence[int], groups: Sequence[str | None], fractions: Sequence[float], seed: int) -> dict[str, list[int]]`, with keys `train` / `val` / `test` mapping to sorted sample indices. It raises `ValueError` naming any split that lacks a class.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:

```python
from __future__ import annotations

import json
import pathlib

import datasets
import pytest

from sonde import config
from sonde import data


def _jsonl(tmp_path: pathlib.Path, rows: list[dict]) -> str:
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_load_jsonl_label_map_and_ids(tmp_path):
    path = _jsonl(
        tmp_path,
        [
            {"q": "a", "y": "yes", "uid": "u1"},
            {"q": "b", "y": "no"},
        ],
    )
    src = config.DataConfig(
        path=path, text="q", label="y", id="uid", label_map={"yes": 1, "no": 0}
    )
    s = data.load_samples(src, seed=0)
    assert [(x.id, x.text, x.label) for x in s] == [
        ("u1", "a", 1),
        ("1", "b", 0),
    ]


def test_load_csv_labels_are_strings(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("text,label\nhi,1\nyo,0\n")
    s = data.load_samples(config.DataConfig(path=str(path)), seed=0)
    assert [x.label for x in s] == [1, 0]


def test_bad_label_names_the_row(tmp_path):
    path = _jsonl(
        tmp_path, [{"text": "a", "label": 0}, {"text": "b", "label": "maybe"}]
    )
    with pytest.raises(ValueError, match="row 1: label 'maybe'"):
        data.load_samples(config.DataConfig(path=path), seed=0)
    path = _jsonl(tmp_path, [{"text": "a"}])
    with pytest.raises(ValueError, match="row 0 has no 'label' key"):
        data.load_samples(config.DataConfig(path=path), seed=0)


def test_limit_is_a_seeded_subsample(tmp_path):
    path = _jsonl(
        tmp_path, [{"text": str(i), "label": i % 2} for i in range(20)]
    )
    src = config.DataConfig(path=path, limit=5)
    a = [x.id for x in data.load_samples(src, seed=1)]
    assert a == [x.id for x in data.load_samples(src, seed=1)]
    assert len(a) == 5 and a == sorted(a, key=int)
    assert a != [x.id for x in data.load_samples(src, seed=2)]


def test_load_hf_dataset(monkeypatch):
    calls = []

    def fake(name, cfg_name, split):
        calls.append((name, cfg_name, split))
        return [{"inputs": "x", "labels": "high-stakes", "ids": 9}]

    monkeypatch.setattr(datasets, "load_dataset", fake)
    src = config.DataConfig(
        hf="org/ds",
        hf_config="training",
        text="inputs",
        label="labels",
        id="ids",
        label_map={"high-stakes": 1, "low-stakes": 0},
    )
    s = data.load_samples(src, seed=0)
    assert calls == [("org/ds", "training", "train")]
    assert (s[0].id, s[0].label) == ("9", 1)


def test_split_is_stratified_and_group_disjoint():
    labels = [i % 2 for i in range(200)]
    groups = [f"g{i % 50}" for i in range(200)]
    parts = data.split(labels, groups, (0.6, 0.2, 0.2), seed=0)
    assert sorted(i for idx in parts.values() for i in idx) == list(range(200))
    where = {}
    for name, idx in parts.items():
        assert {labels[i] for i in idx} == {0, 1}
        for i in idx:
            assert where.setdefault(groups[i], name) == name
    assert abs(len(parts["train"]) / 200 - 0.6) < 0.05
    assert abs(sum(labels[i] for i in parts["train"]) / 100 - 0.6) < 0.05
    assert parts == data.split(labels, groups, (0.6, 0.2, 0.2), seed=0)


def test_split_missing_class_names_the_split():
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="split 'val' has n_pos=0"):
        data.split(labels, [None] * 9, (0.7, 0.15, 0.15), seed=0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra hf --extra dev pytest tests/test_data.py -q`
Expected: collection error `ImportError: cannot import name 'data' from 'sonde'`.

- [ ] **Step 3: Write the implementation**

Create `sonde/data.py`:

```python
"""Labeled rows in, token ids with a window span out."""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib
from collections.abc import Sequence

import datasets
import numpy as np

from sonde import config

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


@dataclasses.dataclass
class Sample:
    id: str
    text: str | None
    messages: list[dict] | None
    response: str | None
    response_ids: list[int] | None
    label: int | None
    group: str | None
    raw: dict


def _rows(src: config.DataConfig) -> list:
    if src.hf is not None:
        return list(
            datasets.load_dataset(src.hf, src.hf_config, split=src.hf_split)
        )
    path = pathlib.Path(str(src.path))
    with path.open(newline="") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if path.suffix == ".csv":
            return list(csv.DictReader(f))
    raise ValueError(f"data.path must be .jsonl or .csv, got {path}")


def _field(row: dict, key: str, i: int):
    if key not in row:
        raise ValueError(f"row {i} has no {key!r} key; keys: {sorted(row)}")
    return row[key]


def load_samples(src: config.DataConfig, seed: int) -> list[Sample]:
    """Reads, subsamples and validates the rows of a data block.

    Args:
        src: The `data` block (or a merged `score` entry).
        seed: Drives the `limit` subsample.

    Returns:
        Samples in file order; `id` falls back to the row index.

    Raises:
        ValueError: On a missing key or a label outside {0, 1} after
            `label_map`; the message names the row.
    """
    rows = _rows(src)
    index = range(len(rows))
    if src.limit is not None and src.limit < len(rows):
        rng = np.random.default_rng(seed)
        index = sorted(rng.choice(len(rows), src.limit, replace=False))
    samples = []
    for i in index:
        row = rows[i]
        label = None
        if src.label is not None:
            label = _field(row, src.label, i)
            if src.label_map:
                label = src.label_map.get(label, label)
            if label not in (0, 1, "0", "1"):
                raise ValueError(
                    f"row {i}: label {label!r} is not 0/1; set data.label_map"
                )
            label = int(label)
        has_id = src.id is not None and src.id in row
        samples.append(
            Sample(
                id=str(row[src.id]) if has_id else str(i),
                text=_field(row, src.text, i) if src.text else None,
                messages=(
                    _field(row, src.messages, i) if src.messages else None
                ),
                response=(
                    _field(row, src.response, i) if src.response else None
                ),
                response_ids=row.get("response_ids") if src.response else None,
                label=label,
                group=str(_field(row, src.group, i)) if src.group else None,
                raw=row,
            )
        )
    return samples


def split(
    labels: Sequence[int],
    groups: Sequence[str | None],
    fractions: Sequence[float],
    seed: int,
) -> dict[str, list[int]]:
    """Stratified, group-disjoint train / val / test split.

    A group is stratified by its majority label; a `None` group is its own
    unit.

    Args:
        labels: [n] in {0, 1}.
        groups: [n] group keys or None.
        fractions: (train, val, test), summing to 1.
        seed: Seeds the one generator that orders units.

    Returns:
        `{"train", "val", "test"}` -> sorted sample indices.

    Raises:
        ValueError: If any split lacks a class; the message names it.
    """
    y = np.asarray(labels)
    units = {}
    for i, g in enumerate(groups):
        units.setdefault(("i", i) if g is None else ("g", g), []).append(i)
    rng = np.random.default_rng(seed)
    out = {name: [] for name in _SPLITS}
    cuts = np.cumsum(fractions)[:2]
    for cls in (0, 1):
        members = [u for u in units.values() if round(y[u].mean()) == cls]
        total, seen = sum(map(len, members)), 0
        for j in rng.permutation(len(members)):
            k = int(np.searchsorted(cuts, seen / total, side="right"))
            out[_SPLITS[k]] += members[j]
            seen += len(members[j])
    for name, idx in out.items():
        n_pos = int(y[idx].sum())
        n_neg = len(idx) - n_pos
        logger.info(
            "split %s: n=%d (%.3f) n_pos=%d n_neg=%d",
            name,
            len(idx),
            len(idx) / len(y),
            n_pos,
            n_neg,
        )
        if not n_pos or not n_neg:
            raise ValueError(
                f"split {name!r} has n_pos={n_pos}, n_neg={n_neg}; it needs "
                "both classes. Add data or change data.split"
            )
    return {name: sorted(idx) for name, idx in out.items()}
```

Notes:
- CSV values arrive as strings, so `"0"` and `"1"` are accepted as labels. JSON booleans are rejected unless `label_map` maps them.
- A unit goes to the split whose cumulative fraction its starting offset falls in. With 3 positives at 0.7 / 0.15 / 0.15, all 3 land in train and the split raises; that is the intended tiny-data failure.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra hf --extra dev pytest tests/test_data.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Lint**

Run: `uv run --extra hf --extra dev ruff format --check sonde/data.py tests/test_data.py && uv run --extra hf --extra dev ruff check sonde/data.py tests/test_data.py`
Expected: `2 files already formatted` then `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add sonde/data.py tests/test_data.py
git commit -m "feat(data): load jsonl/csv/hf rows with label_map and limit; group split"
```

---

### Task 2.3: Rendering: chat and raw, four windows, counted drops

**Files:**
- Modify: `sonde/data.py` (import block; append)
- Test: `tests/test_data.py` (import block; append)

**Interfaces:**
- Consumes:
  - `config.DataConfig` (Task 2.1) and `data.Sample` (Task 2.2).
  - From phase 1: `probe.last_turn_start(tokenizer, messages: list[dict]) -> int`, `fingerprint.RAW` and `fingerprint.prompt_format(template: str | None) -> str`.
- Produces:
  - `Encoded` dataclass: `id: str; ids: list[int]; span: tuple[int, int]; label: int | None; group: str | None`
  - `Drop(Exception)` with `.reason` in `{"empty_window", "not_assistant_last", "last_turn_prefix_mismatch", "empty_response"}`
  - `prompt_format_of(src: config.DataConfig, tokenizer) -> str`
  - `prompt_ids(sample: Sample, src: config.DataConfig, tokenizer) -> list[int]`
  - `render(sample: Sample, src: config.DataConfig, window: str, tokenizer) -> Encoded` (raises `Drop`)
  - `encode_all(samples, src, window, tokenizer) -> tuple[list[Encoded], dict[str, int]]`

- [ ] **Step 1: Write the failing tests**

In `tests/test_data.py`, replace:

```python
import datasets
import pytest

from sonde import config
from sonde import data
```

with:

```python
import datasets
import pytest
import transformers

from sonde import config
from sonde import data
from sonde import fingerprint
```

Append to `tests/test_data.py`:

```python


TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _tok(template: str | None = None):
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = template
    return tok


def _src(**kw) -> config.DataConfig:
    return config.DataConfig(path="unused.jsonl", **kw)


def _sample(**kw) -> data.Sample:
    fields = dict(
        id="0",
        text=None,
        messages=None,
        response=None,
        response_ids=None,
        label=1,
        group=None,
        raw={},
    )
    return data.Sample(**{**fields, **kw})


def test_raw_windows():
    tok = _tok()
    p = tok("Hello world").input_ids
    r = tok(" yes sir", add_special_tokens=False).input_ids
    src = _src(format="raw", response="r")
    s = _sample(text="Hello world", response=" yes sir")
    assert data.render(s, src, "prompt", tok).span == (0, len(p))
    resp = data.render(s, src, "response", tok)
    assert resp.ids == p + r and resp.span == (len(p), len(p) + len(r))
    assert data.render(s, src, "all", tok).span == (0, len(p) + len(r))
    assert data.prompt_format_of(src, tok) == fingerprint.RAW


def test_chat_windows_and_prompt_format():
    tok = _tok(TEMPLATE)
    src = _src(system="be brief")
    s = _sample(text="hi")
    enc = data.render(s, src, "prompt", tok)
    want = "<|system|>\nbe brief\n<|user|>\nhi\n<|assistant|>\n"
    assert tok.decode(enc.ids) == want
    assert enc.span == (0, len(enc.ids))
    assert data.prompt_format_of(src, tok) == fingerprint.prompt_format(
        TEMPLATE
    )
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    turn = data.render(
        _sample(messages=msgs), _src(messages="m"), "last_turn", tok
    )
    assert tok.decode(turn.ids[turn.span[0] :]) == "hello there\n"


def test_response_ids_are_reused_exactly():
    tok = _tok()
    s = _sample(text="Q", response="ignored", response_ids=[11, 12, 13])
    enc = data.render(s, _src(format="raw", response="r"), "response", tok)
    assert enc.ids[enc.span[0] :] == [11, 12, 13]


def test_chat_render_has_no_double_bos():
    tok = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    enc = data.render(_sample(text="hi"), _src(), "prompt", tok)
    assert enc.ids[0] == tok.bos_token_id
    assert enc.ids[1] != tok.bos_token_id


def test_chat_without_template_says_use_raw():
    tok = _tok()
    with pytest.raises(ValueError, match=r"set data\.format: raw"):
        data.render(_sample(text="hi"), _src(), "prompt", tok)
    with pytest.raises(ValueError, match=r"set data\.format: raw"):
        data.prompt_format_of(_src(), tok)


def test_drops_are_counted_never_empty():
    tok = _tok(TEMPLATE)
    long_prompt = _sample(text="word " * 50, response="yes")
    no_response = _sample(text="q", response="")
    ok = _sample(text="q", response="yes")
    src = _src(response="r", max_length=16)
    encoded, drops = data.encode_all(
        [long_prompt, no_response, ok], src, "response", tok
    )
    assert drops == {"empty_window": 1, "empty_response": 1}
    assert len(encoded) == 1
    start, end = encoded[0].span
    assert 0 <= start < end == len(encoded[0].ids) <= 16


def test_max_length_clips_span():
    tok = _tok()
    s = _sample(text="one two three four five six")
    enc = data.render(s, _src(format="raw", max_length=3), "prompt", tok)
    assert len(enc.ids) == 3 and enc.span == (0, 3)


def test_last_turn_drops():
    tok = _tok(TEMPLATE)
    user_last = [{"role": "user", "content": "hi"}]
    with pytest.raises(data.Drop) as e:
        data.render(
            _sample(messages=user_last), _src(messages="m"), "last_turn", tok
        )
    assert e.value.reason == "not_assistant_last"
    thinking = TEMPLATE.replace(
        "<|assistant|>\n{% endif %}", "<|assistant|>\n<think>{% endif %}"
    )
    msgs = [*user_last, {"role": "assistant", "content": "ok"}]
    with pytest.raises(data.Drop) as e:
        data.render(
            _sample(messages=msgs),
            _src(messages="m"),
            "last_turn",
            _tok(thinking),
        )
    assert e.value.reason == "last_turn_prefix_mismatch"


def test_all_rows_dropped_raises():
    with pytest.raises(ValueError, match="every row was dropped"):
        data.encode_all(
            [_sample(text="q", response="")],
            _src(format="raw", response="r"),
            "response",
            _tok(),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra hf --extra dev pytest tests/test_data.py -q`
Expected: `9 failed, 7 passed`. The new tests fail with `AttributeError: module 'sonde.data' has no attribute 'render'` (or `'prompt_format_of'`, `'encode_all'`, `'Drop'`).

- [ ] **Step 3: Write the implementation**

In `sonde/data.py`, replace:

```python
import csv
import dataclasses
import json
import logging
import pathlib
from collections.abc import Sequence

import datasets
import numpy as np

from sonde import config
```

with:

```python
import collections
import csv
import dataclasses
import json
import logging
import pathlib
from collections.abc import Sequence

import datasets
import numpy as np

from sonde import config
from sonde import fingerprint
from sonde import probe
```

Append to `sonde/data.py`:

```python


@dataclasses.dataclass
class Encoded:
    id: str
    ids: list[int]
    span: tuple[int, int]
    label: int | None
    group: str | None


class Drop(Exception):
    """A row that cannot yield a non-empty window; `reason` names why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _template(tokenizer) -> str:
    if tokenizer.chat_template is None:
        raise ValueError(
            f"data.format is chat but the {tokenizer.name_or_path} tokenizer "
            "has no chat_template; set data.format: raw"
        )
    return tokenizer.chat_template


def _chat_ids(tokenizer, messages: list[dict], gen_prompt: bool) -> list[int]:
    _template(tokenizer)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=gen_prompt
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def _messages(sample: Sample, src: config.DataConfig) -> list[dict]:
    if sample.messages is not None:
        return sample.messages
    system = [{"role": "system", "content": src.system}] if src.system else []
    return [*system, {"role": "user", "content": sample.text}]


def prompt_format_of(src: config.DataConfig, tokenizer) -> str:
    """The `prompt_format` mechanica serves under for this data block.

    Raises:
        ValueError: If `format: chat` and the tokenizer has no template.
    """
    if src.format == "raw":
        return fingerprint.RAW
    return fingerprint.prompt_format(_template(tokenizer))


def prompt_ids(sample: Sample, src: config.DataConfig, tokenizer) -> list[int]:
    """Prompt token ids, generation prompt included for `format: chat`."""
    if src.format == "raw":
        return tokenizer(sample.text).input_ids
    return _chat_ids(tokenizer, _messages(sample, src), True)


def render(
    sample: Sample, src: config.DataConfig, window: str, tokenizer
) -> Encoded:
    """Tokenizes one sample and locates its window.

    Args:
        sample: One loaded row.
        src: Format, system prompt and `max_length`.
        window: One of `probe.WINDOWS`.
        tokenizer: A HF tokenizer.

    Returns:
        `ids` truncated to `max_length` and `span = (start, end)` with
        `0 <= start < end <= len(ids)`.

    Raises:
        Drop: If the window is empty, or `last_turn` is not usable.
        ValueError: On `window: last_turn` with `format: raw`, or a chat
            format without a chat template.
    """
    if window == "last_turn":
        if src.format == "raw":
            raise ValueError(
                "extract.window: last_turn needs data.format: chat"
            )
        msgs = _messages(sample, src)
        if msgs[-1]["role"] != "assistant":
            raise Drop("not_assistant_last")
        ids = _chat_ids(tokenizer, msgs, False)
        start = probe.last_turn_start(tokenizer, msgs)
        if ids[:start] != _chat_ids(tokenizer, msgs[:-1], True):
            raise Drop("last_turn_prefix_mismatch")
    else:
        ids = prompt_ids(sample, src, tokenizer)
        start = 0
        if window != "prompt":
            r = sample.response_ids or (
                tokenizer(sample.response, add_special_tokens=False).input_ids
                if sample.response
                else []
            )
            if window == "response":
                if not r:
                    raise Drop("empty_response")
                start = len(ids)
            ids = ids + list(r)
    ids = ids[: src.max_length]
    if start >= len(ids):
        raise Drop("empty_window")
    return Encoded(
        sample.id, ids, (start, len(ids)), sample.label, sample.group
    )


def encode_all(
    samples: Sequence[Sample], src: config.DataConfig, window: str, tokenizer
) -> tuple[list[Encoded], dict[str, int]]:
    """Renders every sample, counting drops by reason.

    Returns:
        `(encoded, drops)`, encoded in sample order.

    Raises:
        ValueError: If every row is dropped.
    """
    encoded, drops = [], collections.Counter()
    for sample in samples:
        try:
            encoded.append(render(sample, src, window, tokenizer))
        except Drop as e:
            drops[e.reason] += 1
    if drops:
        logger.warning(
            "dropped %d of %d rows: %s",
            drops.total(),
            len(samples),
            dict(drops),
        )
    if not encoded:
        raise ValueError(f"every row was dropped: {dict(drops)}")
    return encoded, dict(drops)
```

Notes:
- The chat path never calls `tokenizer(..., add_special_tokens=True)` on rendered text. The template already carries BOS, and this is how vLLM and mechanica tokenize a rendered prompt.
- `window: prompt` sends no response tokens: the model is causal, so tokens after the window cannot change it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra hf --extra dev pytest tests/test_data.py -q`
Expected: `16 passed`.

- [ ] **Step 5: Lint**

Run: `uv run --extra hf --extra dev ruff format --check sonde/data.py tests/test_data.py && uv run --extra hf --extra dev ruff check sonde/data.py tests/test_data.py`
Expected: `2 files already formatted` then `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add sonde/data.py tests/test_data.py
git commit -m "feat(data): chat/raw render for four windows; counted drops"
```

---

### Task 2.4: Shard store: run.json, shards, manifest, read_layer

**Files:**
- Create: `sonde/store.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: nothing from sonde (torch, safetensors).
- Produces:
  - `FORMAT = 1`
  - `config_hash(sections: dict) -> str`: 16 hex chars.
  - `prepare(out_dir: pathlib.Path, config_hash: str, overwrite: bool) -> str`: returns `"fresh" | "resume" | "done"`. It raises `ValueError` on a hash mismatch unless `overwrite`, which `rmtree`s `out_dir` first.
  - `ShardWriter(out_dir: pathlib.Path, keep: str, shard_size: int, shard_bytes: int)`, with `.done: int`, `.add(features: dict[int, list[torch.Tensor]]) -> None` and `.close() -> list[str]`.
  - `write_manifest(out_dir: pathlib.Path, manifest: dict) -> None` and `read_manifest(out_dir: pathlib.Path) -> dict`.
  - `read_layer(out_dir: pathlib.Path, block: int) -> tuple[torch.Tensor, torch.Tensor | None]`.
  - Shard tensor names are `B{block}` and `offsets`. Shard metadata `{"n": str}` holds the sample count.

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract.py`:

```python
from __future__ import annotations

import json

import pytest
import torch

from sonde import store


def test_prepare_states(tmp_path):
    out = tmp_path / "extract"
    assert store.prepare(out, "abc", overwrite=False) == "fresh"
    assert json.loads((out / "run.json").read_text()) == {"config_hash": "abc"}
    assert store.prepare(out, "abc", overwrite=False) == "resume"
    store.write_manifest(out, {"config_hash": "abc"})
    assert store.prepare(out, "abc", overwrite=False) == "done"
    with pytest.raises(ValueError, match="another config"):
        store.prepare(out, "xyz", overwrite=False)
    assert store.prepare(out, "xyz", overwrite=True) == "fresh"
    assert sorted(p.name for p in out.iterdir()) == ["run.json"]


def test_config_hash_ignores_key_order():
    assert store.config_hash({"a": 1, "b": [2]}) == store.config_hash(
        {"b": [2], "a": 1}
    )
    assert len(store.config_hash({})) == 16


def test_unknown_manifest_format_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": 2}))
    with pytest.raises(ValueError, match="format 2"):
        store.read_manifest(tmp_path)


def test_shards_round_trip_tokens(tmp_path):
    writer = store.ShardWriter(
        tmp_path, "tokens", shard_size=2, shard_bytes=2**31
    )
    rows = [torch.full((n, 3), float(n)) for n in (1, 2, 3)]
    for row in rows:
        writer.add({4: [row], 7: [row * 2]})
    shards = writer.close()
    assert shards == ["shard_00000.safetensors", "shard_00001.safetensors"]
    store.write_manifest(
        tmp_path, {"shards": shards, "blocks": [4, 7], "keep": "tokens"}
    )
    x, offsets = store.read_layer(tmp_path, 7)
    assert offsets.tolist() == [0, 1, 3, 6]
    assert torch.equal(x, torch.cat(rows) * 2)
    assert store.ShardWriter(tmp_path, "tokens", 2, 2**31).done == 3
    with pytest.raises(ValueError, match="block 5"):
        store.read_layer(tmp_path, 5)


def test_shard_bytes_closes_shards(tmp_path):
    writer = store.ShardWriter(
        tmp_path, "pooled", shard_size=100, shard_bytes=1
    )
    for _ in range(3):
        writer.add({0: [torch.ones(4)]})
    assert writer.close() == [f"shard_{k:05d}.safetensors" for k in range(3)]
    assert store.ShardWriter(tmp_path, "pooled", 100, 1).done == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra hf --extra dev pytest tests/test_extract.py -q`
Expected: collection error `ImportError: cannot import name 'store' from 'sonde'`.

- [ ] **Step 3: Write the implementation**

Create `sonde/store.py`:

```python
"""Extraction on disk: run.json, safetensors shards, manifest.json."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil

import safetensors
import safetensors.torch
import torch

FORMAT = 1


def _write_json(path: pathlib.Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def config_hash(sections: dict) -> str:
    """sha256 of the JSON-dumped sections, first 16 hex chars."""
    blob = json.dumps(sections, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def prepare(out_dir: pathlib.Path, config_hash: str, overwrite: bool) -> str:
    """Decides what an extraction into `out_dir` must do. Loads no model.

    Args:
        out_dir: The extraction directory.
        config_hash: Hash of the config that would write it.
        overwrite: Delete `out_dir` first.

    Returns:
        `"done"` (finished manifest), `"resume"` (run.json only) or
        `"fresh"` (run.json just written).

    Raises:
        ValueError: If run.json or the manifest has another hash.
    """
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    for name, state in (("manifest.json", "done"), ("run.json", "resume")):
        path = out_dir / name
        if path.exists():
            found = json.loads(path.read_text())["config_hash"]
            if found != config_hash:
                raise ValueError(
                    f"{path} was written by another config (hash {found}, "
                    f"now {config_hash}); set output.overwrite: true or "
                    "change name"
                )
            return state
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "run.json", {"config_hash": config_hash})
    return "fresh"


class ShardWriter:
    """Buffers per-sample features and writes them as ordered shards.

    A shard closes once it holds `shard_size` samples or `shard_bytes`
    bytes, checked after each batch. Complete shards found in `out_dir`
    count toward `done`, so a resumed run continues at sample `done`.
    """

    def __init__(
        self,
        out_dir: pathlib.Path,
        keep: str,
        shard_size: int,
        shard_bytes: int,
    ):
        self.out_dir = out_dir
        self.keep = keep
        self.shard_size = shard_size
        self.shard_bytes = shard_bytes
        self.shards = sorted(
            p.name for p in out_dir.glob("shard_*.safetensors")
        )
        self.done = 0
        for name in self.shards:
            with safetensors.safe_open(str(out_dir / name), "pt") as f:
                self.done += int(f.metadata()["n"])
        self._buf: dict[int, list[torch.Tensor]] = {}
        self._n = 0
        self._bytes = 0

    def add(self, features: dict[int, list[torch.Tensor]]) -> None:
        """Appends one batch: block -> per-sample tensors in sample order."""
        for block, rows in features.items():
            self._buf.setdefault(block, []).extend(rows)
            self._bytes += sum(t.nbytes for t in rows)
        self._n += len(next(iter(features.values())))
        if self._n >= self.shard_size or self._bytes >= self.shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._n:
            return
        tensors = {}
        for block, rows in self._buf.items():
            if self.keep == "pooled":
                tensors[f"B{block}"] = torch.stack(rows)
            else:
                tensors[f"B{block}"] = torch.cat(rows)
                lengths = torch.tensor([0] + [len(t) for t in rows])
                tensors["offsets"] = lengths.cumsum(0)
        name = f"shard_{len(self.shards):05d}.safetensors"
        tmp = self.out_dir / f"{name}.tmp"
        safetensors.torch.save_file(tensors, str(tmp), {"n": str(self._n)})
        os.replace(tmp, self.out_dir / name)
        self.shards.append(name)
        self._buf, self._n, self._bytes = {}, 0, 0

    def close(self) -> list[str]:
        """Flushes the last shard; returns all shard file names in order."""
        self._flush()
        return self.shards


def write_manifest(out_dir: pathlib.Path, manifest: dict) -> None:
    """Atomically writes manifest.json with `format` added."""
    _write_json(out_dir / "manifest.json", {"format": FORMAT, **manifest})


def read_manifest(out_dir: pathlib.Path) -> dict:
    """Reads manifest.json.

    Raises:
        ValueError: On a `format` other than 1.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"{out_dir}/manifest.json has format {manifest.get('format')!r}; "
            f"this sonde reads format {FORMAT}"
        )
    return manifest


def read_layer(
    out_dir: pathlib.Path, block: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reads one block across all shards.

    Args:
        out_dir: A finished extraction directory.
        block: A block listed in the manifest.

    Returns:
        `(X, offsets)`: pooled X [n, H] with offsets None, or tokens X
        [sum T, H] with offsets [n + 1] (row i is X[offsets[i]:offsets[i+1]]).

    Raises:
        ValueError: If `block` was not extracted.
    """
    # ponytail: one tokens-mode block must fit in RAM; stream by shard if a
    # dataset outgrows it.
    manifest = read_manifest(out_dir)
    if block not in manifest["blocks"]:
        raise ValueError(f"block {block} not in {manifest['blocks']}")
    xs, offsets = [], [torch.zeros(1, dtype=torch.long)]
    for name in manifest["shards"]:
        with safetensors.safe_open(str(out_dir / name), "pt") as f:
            xs.append(f.get_tensor(f"B{block}"))
            if manifest["keep"] == "tokens":
                offsets.append(f.get_tensor("offsets")[1:] + offsets[-1][-1])
    if manifest["keep"] == "pooled":
        return torch.cat(xs), None
    return torch.cat(xs), torch.cat(offsets)
```

Notes:
- A shard is complete exactly when its final name exists: it is written to `shard_k.safetensors.tmp` and then `os.replace`d. The resume glob never matches a `.tmp` file.
- Offsets are per shard on disk, and `read_layer` makes them global.
- Shards may overshoot `shard_size` by less than one batch. Sample order is what keeps resume correct; exact boundaries are not needed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra hf --extra dev pytest tests/test_extract.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Lint**

Run: `uv run --extra hf --extra dev ruff format --check sonde/store.py tests/test_extract.py && uv run --extra hf --extra dev ruff check sonde/store.py tests/test_extract.py`
Expected: `2 files already formatted` then `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add sonde/store.py tests/test_extract.py
git commit -m "feat(store): run.json hash guard, atomic safetensors shards, read_layer"
```

---

### Task 2.5: HF backend: model cache, window_pool, masked right-padded forward

**Files:**
- Create: `sonde/backends.py`
- Test: `tests/test_extract.py` (import line; append)

**Interfaces:**
- Consumes: `config.ModelConfig` (Task 2.1) and `data.Encoded` (Task 2.3).
- Produces:
  - `Loaded` dataclass: `backend: str; model: Any; tokenizer: Any; num_layers: int; hidden: int; engine: str`. The engine reads `hf==<transformers>+nnterp==<nnterp>`.
  - `load_model(cfg: config.ModelConfig) -> Loaded`: a single-slot cache keyed by `cfg.model_dump_json()`. `backend: vllm` raises `NotImplementedError("vllm backend lands in phase 5")`.
  - `window_pool(h: torch.Tensor, span: tuple[int, int], keep: str, pooling: str | None) -> torch.Tensor`: `[T, H]` in; `[H]` when pooled, else `[end - start, H]`.
  - `forward(loaded: Loaded, batch: list[data.Encoded], blocks: list[int], keep: str, pooling: str | None) -> dict[int, list[torch.Tensor]]`: CPU tensors in the model dtype, in batch order.
  - Phase 5 adds the vllm branch and phase 6 adds `Steering` and `generate`; neither exists yet.

- [ ] **Step 1: Write the failing tests**

In `tests/test_extract.py`, replace:

```python
from sonde import store
```

with:

```python
from sonde import backends
from sonde import config
from sonde import data
from sonde import store

TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Hi",
    "Paris is the capital of",
    "One two three four five six seven",
    "A much longer sentence that will certainly be cut by max_length.",
]
```

Append to `tests/test_extract.py`:

```python


def test_window_pool():
    h = torch.arange(12.0).reshape(4, 3)
    assert torch.equal(backends.window_pool(h, (1, 3), "tokens", None), h[1:3])
    assert torch.equal(backends.window_pool(h, (1, 3), "pooled", "last"), h[2])
    assert torch.equal(
        backends.window_pool(h, (1, 3), "pooled", "mean"), h[1:3].mean(0)
    )
    with pytest.raises(ValueError, match="mean or last"):
        backends.window_pool(h, (1, 3), "pooled", "max")


def test_vllm_backend_is_not_here_yet():
    with pytest.raises(NotImplementedError, match="phase 5"):
        backends.load_model(config.ModelConfig(name="x", backend="vllm"))


def test_load_model_is_cached():
    cfg = config.ModelConfig(name="gpt2", dtype="float32")
    loaded = backends.load_model(cfg)
    assert backends.load_model(cfg.model_copy()) is loaded
    assert (loaded.num_layers, loaded.hidden) == (12, 768)
    assert loaded.engine.startswith("hf==") and "+nnterp==" in loaded.engine


def test_gpt2_batch_invariance():
    loaded = backends.load_model(
        config.ModelConfig(name="gpt2", dtype="float32")
    )
    batch = []
    for i, text in enumerate(TEXTS[:4]):
        ids = loaded.tokenizer(text).input_ids
        batch.append(
            data.Encoded(str(i), ids, (len(ids) // 2, len(ids)), 0, None)
        )
    blocks = [0, 5, 11]
    four = backends.forward(loaded, batch, blocks, "tokens", None)
    last = backends.forward(loaded, batch, [11], "pooled", "last")
    for i, e in enumerate(batch):
        one = backends.forward(loaded, [e], blocks, "tokens", None)
        for b in blocks:
            assert four[b][i].shape == (e.span[1] - e.span[0], 768)
            assert torch.isfinite(four[b][i]).all()
            # fp32 CPU differs by ~2e-4, MPS by ~4e-3; the padding bug was
            # ~2.6e3.
            assert torch.allclose(
                one[b][0], four[b][i], atol=1e-2, rtol=1e-4
            ), (b, i)
        assert torch.equal(last[11][i], four[11][i][-1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra hf --extra dev pytest tests/test_extract.py -q`
Expected: collection error `ImportError: cannot import name 'backends' from 'sonde'`.

- [ ] **Step 3: Write the implementation**

Create `sonde/backends.py`:

```python
"""The backend seam: load a model once, prefill token ids, pool a window."""

from __future__ import annotations

import dataclasses
from importlib import metadata
from typing import Any

import nnsight
import torch

from sonde import config
from sonde import data


@dataclasses.dataclass
class Loaded:
    backend: str
    model: Any
    tokenizer: Any
    num_layers: int
    hidden: int
    engine: str


_cache: dict[str, Loaded] = {}


def load_model(cfg: config.ModelConfig) -> Loaded:
    """Loads the model, reusing the last one loaded for an identical config.

    Args:
        cfg: The `model` section.

    Returns:
        The loaded model, tokenizer and shape facts.

    Raises:
        NotImplementedError: For `backend: vllm` until phase 5.
    """
    key = cfg.model_dump_json()
    if key not in _cache:
        if cfg.backend != "hf":
            raise NotImplementedError("vllm backend lands in phase 5")
        _cache.clear()
        _cache[key] = _load_hf(cfg)
    return _cache[key]


def _load_hf(cfg: config.ModelConfig) -> Loaded:
    import nnterp

    model = nnterp.StandardizedTransformer(
        cfg.name,
        revision=cfg.revision,
        dtype=getattr(torch, cfg.dtype),
        dispatch=True,
    )
    engine = "hf=={}+nnterp=={}".format(
        metadata.version("transformers"), metadata.version("nnterp")
    )
    return Loaded(
        "hf",
        model,
        model.tokenizer,
        model.num_layers,
        model.hidden_size,
        engine,
    )


def window_pool(
    h: torch.Tensor, span: tuple[int, int], keep: str, pooling: str | None
) -> torch.Tensor:
    """Cuts one row's window out of a block output.

    Args:
        h: [T, H] block output for one row.
        span: `(start, end)` token window, `start < end <= T`.
        keep: `"pooled"` or `"tokens"`.
        pooling: `"mean"` or `"last"` when `keep == "pooled"`.

    Returns:
        [H] if pooled, else [end - start, H]; same dtype as `h`.
    """
    start, end = span
    if keep == "tokens":
        return h[start:end]
    if pooling == "last":
        return h[end - 1]
    if pooling == "mean":
        return h[start:end].mean(0)
    raise ValueError(f"keep: pooled needs pooling mean or last, not {pooling}")


def forward(
    loaded: Loaded,
    batch: list[data.Encoded],
    blocks: list[int],
    keep: str,
    pooling: str | None,
) -> dict[int, list[torch.Tensor]]:
    """One prefill of a batch, read at each block.

    Rows are right-padded with an attention mask, so a row's values do not
    depend on its batch-mates.

    Args:
        loaded: From `load_model`.
        batch: Rows to run, in sample order.
        blocks: Ascending block indices.
        keep: `"pooled"` or `"tokens"`.
        pooling: `"mean"` or `"last"` when pooled.

    Returns:
        block -> one CPU tensor per row in batch order: [H] if pooled, else
        [T_window, H], in the model's dtype.
    """
    model = loaded.model
    width = max(len(e.ids) for e in batch)
    ids = torch.zeros(len(batch), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, e in enumerate(batch):
        ids[i, : len(e.ids)] = torch.tensor(e.ids)
        mask[i, : len(e.ids)] = 1
    inputs = {
        "input_ids": ids.to(model.device),
        "attention_mask": mask.to(model.device),
    }
    out = {b: [] for b in blocks}
    with torch.no_grad(), model.trace(inputs):
        for b in blocks:
            h = model.layers_output[b]
            for i, e in enumerate(batch):
                row = window_pool(h[i], e.span, keep, pooling)
                out[b].append(nnsight.save(row.to("cpu", copy=True)))
    return out
```

Notes:
- The code uses `torch.no_grad()`, not `inference_mode` (§7): phase 6 steering writes in place inside the trace.
- `nnterp` is imported inside `_load_hf` so that the vllm environment never imports it (§7).
- `out` is built before the `with` (see the verified facts above).
- The pad id 0 is never attended to: padding is on the right and masked.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra hf --extra dev pytest tests/test_extract.py -q`
Expected: `9 passed`. The first run downloads gpt2.

- [ ] **Step 5: Lint**

Run: `uv run --extra hf --extra dev ruff format --check sonde/backends.py tests/test_extract.py && uv run --extra hf --extra dev ruff check sonde/backends.py tests/test_extract.py`
Expected: `2 files already formatted` then `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add sonde/backends.py tests/test_extract.py
git commit -m "feat(backends): hf forward, right-padded and masked; single-slot model cache"
```

---

### Task 2.6: The extract step: hash guard, resume, manifest

**Files:**
- Create: `sonde/extract.py`
- Test: `tests/test_extract.py` (import block; append)

**Interfaces:**
- Consumes:
  - `config.RunConfig`, `config.run_dir` and `config.resolve_blocks` (Task 2.1).
  - `data.load_samples` (Task 2.2), and `data.prompt_format_of`, `data.encode_all` and `data.Encoded` (Task 2.3).
  - `store.prepare`, `store.config_hash`, `store.ShardWriter` and `store.write_manifest` (Task 2.4).
  - `backends.load_model`, `backends.forward` and `backends.Loaded` (Task 2.5).
  - From phase 1: `fingerprint.checkpoint_fingerprint(model, revision=...)`.
- Produces (phase 3's runner and score consume these):
  - `extract_hash(cfg: config.RunConfig) -> str`: covers model, data and extract; `probe.pooling` when `keep: pooled`; `seed` when `data.limit` is set.
  - `preflight(cfg: config.RunConfig) -> str`: `store.prepare(config.run_dir(cfg) / "extract", …)`. It never loads a model.
  - `extract_to(loaded: backends.Loaded, encoded: list[data.Encoded], blocks: list[int], keep: str, pooling: str | None, out_dir: pathlib.Path, batch_size: int, shard_size: int, shard_bytes: int, meta: dict) -> None`
  - `run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None`: writes `run_dir/extract/{run.json, shard_*.safetensors, manifest.json}`.
  - Manifest keys: `format, model, revision, model_fingerprint, prompt_format, engine, blocks, window, keep, pooling, dtype, ids, labels, groups, shards, drops, config_hash`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_extract.py`, replace:

```python
import json

import pytest
import torch

from sonde import backends
from sonde import config
from sonde import data
from sonde import store
```

with:

```python
import json
import pathlib

import pytest
import torch

from sonde import backends
from sonde import config
from sonde import data
from sonde import extract
from sonde import store
```

Append to `tests/test_extract.py`:

```python


def _cfg(tmp_path: pathlib.Path, **extract_kw) -> config.RunConfig:
    rows = [
        {"text": t, "label": i % 2, "r": " ok" * (i + 1)}
        for i, t in enumerate(TEXTS)
    ]
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return config.RunConfig.model_validate(
        {
            "name": "t",
            "model": {"name": "gpt2", "dtype": "float32"},
            "data": {
                "path": str(path),
                "format": "raw",
                "response": "r",
                "max_length": 12,
            },
            "extract": {"layers": [0, 5, 11], "window": "all", **extract_kw},
            "probe": {},
            "output": {"dir": str(tmp_path / "runs")},
            "steps": ["extract"],
        }
    )


def _run(cfg: config.RunConfig) -> pathlib.Path:
    extract.run(cfg, config.run_dir(cfg))
    return config.run_dir(cfg) / "extract"


def _no_load(monkeypatch) -> None:
    def boom(_):
        raise AssertionError("model loaded")

    monkeypatch.setattr(backends, "load_model", boom)


def test_shard_and_manifest_layout(tmp_path):
    cfg = _cfg(tmp_path, keep="pooled", batch_size=2, shard_size=2)
    out = _run(cfg)
    manifest = store.read_manifest(out)
    shards = [f"shard_{k:05d}.safetensors" for k in range(3)]
    assert manifest["shards"] == shards
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [*shards, "manifest.json", "run.json"]
    )
    assert manifest["format"] == 1
    assert manifest["ids"] == ["0", "1", "2", "3", "4"]
    assert manifest["labels"] == [0, 1, 0, 1, 0]
    assert manifest["groups"] == [None] * 5
    assert manifest["blocks"] == [0, 5, 11]
    assert (manifest["keep"], manifest["pooling"]) == ("pooled", "mean")
    assert (manifest["window"], manifest["dtype"]) == ("all", "float32")
    assert manifest["prompt_format"] == "raw"
    assert (manifest["model"], manifest["revision"]) == ("gpt2", None)
    assert manifest["engine"].startswith("hf==")
    assert manifest["model_fingerprint"].startswith("hub:")
    assert manifest["config_hash"] == extract.extract_hash(cfg)
    assert manifest["drops"] == {}
    x, offsets = store.read_layer(out, 5)
    assert x.shape == (5, 768) and x.dtype == torch.float32
    assert offsets is None


def test_tokens_offsets_match_spans(tmp_path):
    out = _run(_cfg(tmp_path, keep="tokens", batch_size=2, shard_size=2))
    x, offsets = store.read_layer(out, 0)
    tok = backends.load_model(
        config.ModelConfig(name="gpt2", dtype="float32")
    ).tokenizer
    want = [min(len(tok(t).input_ids) + i + 1, 12) for i, t in enumerate(TEXTS)]
    assert offsets[0] == 0 and offsets[-1] == len(x)
    assert (offsets[1:] - offsets[:-1]).tolist() == want


def test_window_cut_by_max_length_is_dropped_not_empty(tmp_path):
    out = _run(_cfg(tmp_path, keep="tokens", window="response"))
    manifest = store.read_manifest(out)
    assert manifest["drops"] == {"empty_window": 2}
    assert manifest["ids"] == ["1", "2", "3"]
    x, offsets = store.read_layer(out, 11)
    assert (offsets[1:] > offsets[:-1]).all() and torch.isfinite(x).all()


def test_resume_skips_complete_shards(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, keep="pooled", batch_size=2, shard_size=2)
    out = _run(cfg)
    before = store.read_layer(out, 11)[0]
    kept = (out / "shard_00000.safetensors").read_bytes()
    for name in (
        "manifest.json",
        "shard_00001.safetensors",
        "shard_00002.safetensors",
    ):
        (out / name).unlink()
    seen = []
    forward = backends.forward

    def counting(loaded, batch, *args):
        seen.extend(e.id for e in batch)
        return forward(loaded, batch, *args)

    monkeypatch.setattr(backends, "forward", counting)
    _run(cfg)
    assert seen == ["2", "3", "4"]
    assert (out / "shard_00000.safetensors").read_bytes() == kept
    assert torch.allclose(store.read_layer(out, 11)[0], before, atol=1e-4)


def test_finished_extract_never_loads_the_model(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, keep="pooled")
    _run(cfg)
    _no_load(monkeypatch)
    _run(cfg)


def test_edited_config_refuses_before_model_load(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, keep="pooled")
    assert extract.preflight(cfg) == "fresh"
    _no_load(monkeypatch)
    edited = cfg.model_copy(deep=True)
    edited.data.max_length = 64
    with pytest.raises(ValueError, match="another config"):
        extract.preflight(edited)
    with pytest.raises(ValueError, match="another config"):
        _run(edited)
    edited.output.overwrite = True
    assert extract.preflight(edited) == "fresh"


def test_hash_covers_pooling_and_limit_seed(tmp_path):
    cfg = _cfg(tmp_path, keep="pooled")
    other = cfg.model_copy(deep=True)
    other.probe.pooling = "last"
    assert extract.extract_hash(other) != extract.extract_hash(cfg)
    other = cfg.model_copy(deep=True)
    other.seed = 1
    assert extract.extract_hash(other) == extract.extract_hash(cfg)
    other.data.limit = 3
    limited = other.model_copy(deep=True)
    limited.seed = 2
    assert extract.extract_hash(limited) != extract.extract_hash(other)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra hf --extra dev pytest tests/test_extract.py -q`
Expected: collection error `ImportError: cannot import name 'extract' from 'sonde'`.

- [ ] **Step 3: Write the implementation**

Create `sonde/extract.py`:

```python
"""The extract step: labeled data to activation shards."""

from __future__ import annotations

import logging
import pathlib

import torch

from sonde import backends
from sonde import config
from sonde import data
from sonde import fingerprint
from sonde import store

logger = logging.getLogger(__name__)


def extract_hash(cfg: config.RunConfig) -> str:
    """Hash of everything that changes what extraction writes."""
    sections = {
        k: getattr(cfg, k).model_dump(mode="json")
        for k in ("model", "data", "extract")
    }
    if sections["extract"]["keep"] == "pooled" and cfg.probe:
        sections["pooling"] = cfg.probe.pooling
    if sections["data"]["limit"] is not None:
        sections["seed"] = cfg.seed
    return store.config_hash(sections)


def preflight(cfg: config.RunConfig) -> str:
    """Checks the run's extract dir against the config; loads no model.

    Returns:
        `store.prepare`'s state: "fresh", "resume" or "done".

    Raises:
        ValueError: If the dir was written by another config.
    """
    out_dir = config.run_dir(cfg) / "extract"
    return store.prepare(out_dir, extract_hash(cfg), cfg.output.overwrite)


def extract_to(
    loaded: backends.Loaded,
    encoded: list[data.Encoded],
    blocks: list[int],
    keep: str,
    pooling: str | None,
    out_dir: pathlib.Path,
    batch_size: int,
    shard_size: int,
    shard_bytes: int,
    meta: dict,
) -> None:
    """Runs the forward passes and writes shards, then the manifest.

    Resumes after the samples already in complete shards. `meta` holds the
    manifest keys this function cannot know (model, engine, drops, ...).
    """
    writer = store.ShardWriter(out_dir, keep, shard_size, shard_bytes)
    # ponytail: batches follow sample order, so each pads to its own longest
    # row; length-sort within a shard if padding waste ever shows up.
    for i in range(writer.done, len(encoded), batch_size):
        batch = encoded[i : i + batch_size]
        writer.add(backends.forward(loaded, batch, blocks, keep, pooling))
    store.write_manifest(
        out_dir,
        {
            **meta,
            "blocks": blocks,
            "keep": keep,
            "pooling": pooling,
            "ids": [e.id for e in encoded],
            "labels": [e.label for e in encoded],
            "groups": [e.group for e in encoded],
            "shards": writer.close(),
        },
    )


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """The extract step: writes `run_dir/extract/`.

    Raises:
        ValueError: Before any model load, if `run_dir/extract` was written
            by another config and `output.overwrite` is off.
    """
    model, src, ext = cfg.model, cfg.data, cfg.extract
    if model is None or src is None or ext is None:
        raise ValueError("extract needs the model, data and extract sections")
    out_dir = run_dir / "extract"
    h = extract_hash(cfg)
    if store.prepare(out_dir, h, cfg.output.overwrite) == "done":
        logger.info("extract: %s is complete; skipping", out_dir)
        return
    samples = data.load_samples(src, cfg.seed)
    loaded = backends.load_model(model)
    blocks = config.resolve_blocks(ext.layers, loaded.num_layers)
    prompt_format = data.prompt_format_of(src, loaded.tokenizer)
    window, keep = ext.window, ext.keep
    encoded, drops = data.encode_all(samples, src, window, loaded.tokenizer)
    tokens = (
        sum(e.span[1] - e.span[0] for e in encoded)
        if keep == "tokens"
        else len(encoded)
    )
    size = tokens * loaded.hidden * len(blocks)
    size *= getattr(torch, model.dtype).itemsize
    logger.info(
        "extract: %d samples x %d blocks, ~%.2f GB",
        len(encoded),
        len(blocks),
        size / 1e9,
    )
    meta = {
        "model": model.name,
        "revision": model.revision,
        "model_fingerprint": fingerprint.checkpoint_fingerprint(
            model.name, revision=model.revision
        ),
        "prompt_format": prompt_format,
        "engine": loaded.engine,
        "window": window,
        "dtype": model.dtype,
        "drops": drops,
        "config_hash": h,
    }
    extract_to(
        loaded,
        encoded,
        blocks,
        keep,
        cfg.probe.pooling if cfg.probe and keep == "pooled" else None,
        out_dir,
        ext.batch_size,
        ext.shard_size,
        ext.shard_bytes,
        meta,
    )
```

Order inside `run`:
1. The hash check comes first: no I/O, no model.
2. Then `load_samples`, so a bad row fails before minutes of weight loading.
3. Then the model.
4. Then `resolve_blocks`, which is the §7 layer-range check on first load.
5. Then `prompt_format_of`, which is where a template-less `format: chat` fails.

The disk estimate is `Σ window tokens × H × n_blocks × dtype bytes`, with one token per sample when pooled.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra hf --extra dev pytest tests/test_extract.py -q`
Expected: `16 passed`.

- [ ] **Step 5: Run the whole phase, lint and type-check**

Run: `uv run --extra hf --extra dev pytest tests/test_config.py tests/test_data.py tests/test_extract.py -q`
Expected: `47 passed`.

Run: `uv run --extra hf --extra dev ruff format --check sonde tests && uv run --extra hf --extra dev ruff check sonde tests`
Expected: `… files already formatted` then `All checks passed!`.

Run: `uv run --extra hf --extra dev pyright sonde/config.py sonde/data.py sonde/backends.py sonde/store.py sonde/extract.py`
Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 6: Commit**

```bash
git add sonde/extract.py tests/test_extract.py
git commit -m "feat(extract): hash-guarded, resumable extract step with manifest"
```

<!-- Applied the single review issue: every `from sonde import` in phase 2 now imports one module per line, alphabetically. That covers data.py, backends.py, extract.py, tests/test_data.py and tests/test_extract.py, the replace-old/new pairs in Tasks 2.3, 2.5 and 2.6, and the Task 2.2 test header those pairs build on. Checked with ruff (line-length 80, E W F I UP B SIM RUF, isort force-single-line = true, known-first-party sonde): the final import block of all five modules and both test files passed `ruff check` and `ruff format --check`, and so did the intermediate test headers after Tasks 2.2 and 2.5. With force-single-line removed, the same blocks fail with 7 I001 errors, so "Before you start" now lists force-single-line as a phase-1 pyproject precondition. config.py and store.py import only single names, so they are unchanged. Nothing skipped. -->

---

## Phase 3: Probes, training, sweep, score, runner, CLI

**Goal.** Turn an extraction directory (phase 2) into a trusted probe artifact and run it on new data, all behind one command. Phase 3 adds:
- `probes.py`: torch modules that compute exactly what the numpy `probe.Probe` computes, plus the standardisation fold at export.
- `train.py`: `fit`, thresholds, numpy metrics (recall at `max_fpr`, AUROC, `group_auroc`, bootstrap CIs, `small_n`) and the bag-of-words baseline features.
- `sweep.py`: the `train` step, which splits, fits every block, selects one, tests it once, runs a shuffled-label control and a bag-of-words / length baseline, and writes `metrics.json["headline"]`.
- `score.py`: the `score` step, which refuses a probe whose backend, model fingerprint or prompt format differs from the run.
- `runner.py`, `cli.py` and a bundled `quickstart` recipe (gpt2 with tiny data), so `sonde run quickstart` works on a laptop CPU.

**Done when:**
- `uv run sonde run quickstart -o output.dir=/tmp/sonde-qs` exits 0 on CPU in under 2 min.
- `uv run pytest tests/test_probes.py tests/test_train.py tests/test_sweep.py tests/test_e2e.py -v` passes (43 tests).
- `uv run ruff check sonde tests && uv run ruff format --check sonde tests && uv run pyright` is clean.

**Verified on 2026-10-02.** Environment: scratch venv, Python 3.12, torch 2.14.1, nnsight 0.7.0, nnterp 1.3, transformers 5.18.0, pydantic 2.13.5, ruff 0.16.10.

All code in this phase was run there. Phase 1/2 modules were replaced by minimal stand-ins written to the interface contract. `store.prepare` and `config.load` use phase 2's exact code (Task 2.4 and Task 2.1, with Task 3.4's change).
- Results:
  - All 43 tests passed in about 8 s. The honest-eval additions (Tasks 3.2, 3.3, 3.5, 3.6) were rerun against the real phase 1 `probe.py`, the real phase 2 `config.py` (new `select`/`max_fpr` defaults) and `data.load_samples`/`data.split`.
  - `ruff check` (E, W, F, I with `force-single-line`, UP, B, SIM, RUF; line-length 80), `ruff format --check` and pyright (basic mode, tests included) are clean.
  - A mutation check passed: disabling the `prompt_format` and `model_fingerprint` checks in `score.run` makes the Task 3.5 refusal tests fail with `DID NOT RAISE`.
  - `sonde run quickstart` was run twice into the same `output.dir`. The second run skips extract (`"done"`), retrains, and deletes and re-extracts `score/heldout/extract/`. Both runs give identical metrics.
- nnterp `StandardizedTransformer("gpt2", dtype=torch.float32)`:
  - loads in 0.7 s;
  - `layers_output[L]` is `[B, T, 768]`, and there are 12 blocks;
  - 64 rows run forward in about 0.5 s on an M-series CPU;
  - the whole `sonde run quickstart` takes 4.6 s wall time.
- gpt2's tokenizer adds no BOS and its `chat_template` is `None`, so quickstart must use `format: raw`.
- In nnsight 0.7, names bound inside a list comprehension in a trace body are not pushed out (`NameError`); use explicit loops. This matters to phase 2; phase 3 code never traces.
- `transformers.AutoConfig.from_pretrained("gpt2")` exposes `hidden_size == 768` and `num_hidden_layers == 12`, through attribute_map.
- `torch.float32.itemsize == 4` and `getattr(torch, "bfloat16").itemsize == 2`.
- `RunConfig.model_dump(mode="json")` dumps the `split` tuple as a list and `Every` as `{every: k}`; `yaml.safe_dump` handles both.
- Quickstart behaviour:
  - Val has 8 negatives, so `8 × 0.01 < 10` marks it `small_n` and selection falls back to AUROC (`select_fallback: auroc`). All 12 blocks reach val AUROC 1.0, and the tie-break picks block 0.
  - The headline was `recall_at_fpr` 1.0, `auroc` 1.0, `baseline_auroc` 0.78, `length_auroc` 0.5 (seed 0). The bag-of-words probe trains with the recipe's `lr: 1.0e-3` for 30 epochs, so it underfits on 32 train rows; that is the spec's "same `fit`" rule, not a bug.
  - Held-out AUROC is 1.0 for seeds 0–3.
  - `control_auroc` on 16 val rows ranges from 0.05 to 0.75 by seed. It is recorded and never raised.
  - At `lr: 1.0e-2` the float32 sigmoid saturates at 1.0, so the threshold becomes 1.0. `lr: 1.0e-3` keeps it near 0.99, so the recipe uses that.

**Cross-phase assumptions (from the contract and the phase 1/2 drafts).** Phase 3 relies on these phase 1/2 behaviours. A reviewer should confirm them before starting.
- `store.prepare(out_dir, hash, overwrite)` creates `out_dir` when it writes `run.json`. With `overwrite=True` it first rmtrees `out_dir`, so it always returns `"fresh"` (Task 2.4).
- `store.ShardWriter(out_dir, ...)` writes into an existing directory.
- `extract.extract_to(..., meta)` writes the manifest as `meta`, plus `ids`, `labels`, `groups`, `shards` and `format`.
- `config.load` has exactly the body shown in Task 2.1: for a bare name it reads `RECIPES_DIR / f"{name}.yaml"`, and the YAML is parsed once after the if-block. Task 3.4 replaces that body.
- `extract.run` computes its disk estimate inline (Task 2.6). Task 3.6 moves that calculation into `extract.disk_bytes` so the CLI shares it.
- Phase 1's `[tool.setuptools.package-data]` is already `sonde = ["py.typed", "recipes/*.yaml", "recipes/data/*"]`, so phase 3 does not touch `pyproject.toml`.
- `RunConfig(steps=["train"])` validates without `model`.
- `RunConfig(steps=["score"])` validates without `extract` or `probe`.
- `data.split` returns lists of indices.
- Ruff isort uses `force-single-line = true` (Google style, one `from sonde import x` per line).

**Review focus (phase 3).** Each item is pinned by a test in the task that owns it.
1. A probe fitted under another prompt format, checkpoint or backend must be refused by `score`, with a message that names the field (Task 3.5, `test_score_refuses_mismatched_probe`).
2. With `rolling_mean`, a sample shorter than the window scores as the mean over all its tokens in both torch and numpy (Task 3.1, the `rolling_window=9` case).
3. Re-running only `train` after changing `probe.pooling` must refuse pooled features that were pooled differently, not silently fit them (Task 3.3, `test_refuses_pooling_mismatch`).
4. `sonde run quickstart` must find its bundled data from any working directory (Task 3.4, `test_quickstart_data_resolves_from_any_cwd`).
5. A step missing from `STEPS` must refuse before creating the run dir (Task 3.6, `test_runner_refuses_unregistered_step`).

---

### Task 3.1: Torch probe modules, export fold, padding

**Files:**
- Create: `sonde/probes.py`
- Test: `tests/test_probes.py`

**Interfaces:**
- Consumes:
  - `config.ProbeConfig`: `kind: Literal["linear","attention"]`, `pooling: Literal["mean","last","max","rolling_mean","attention"]`, `rolling_window: int | None`.
  - `probe.Probe(name, kind, w, bias, block, window, pooling, threshold, model, engine, model_fingerprint, prompt_format, q=None, rolling_window=None, adapters=(), escalate_threshold=None, metrics=None)`.
  - `Probe.pooled_score(acts [T, H] | [H]) -> float`.
- Produces:
  - `class LinearProbe(torch.nn.Module)`:
    - `__init__(self, hidden: int, pooling: str, rolling_window: int | None = None)`;
    - `forward(self, x, mask=None) -> torch.Tensor`, where x is `[B, H]` (mask None) or `[B, T, H]` and the result is `[B]`;
    - attributes `.linear` (`torch.nn.Linear(hidden, 1)`), `.pooling` and `.rolling_window`.
  - `class AttentionProbe(torch.nn.Module)`:
    - `__init__(self, hidden: int)`;
    - `forward(self, x, mask) -> torch.Tensor`, taking `[B, T, H]` and `[B, T]` and returning `[B]`;
    - attributes `.linear` (value with bias) and `.q` (`Linear(hidden, 1, bias=False)`).
  - `build(cfg: config.ProbeConfig, hidden: int) -> LinearProbe | AttentionProbe`. This is narrower than the contract's `torch.nn.Module` and compatible with it.
  - `export(module: LinearProbe | AttentionProbe, mu: torch.Tensor, sigma: torch.Tensor, **meta) -> probe.Probe`.
    - `meta` must supply `name`, `block`, `window`, `threshold`, `model`, `engine`, `model_fingerprint` and `prompt_format`, and may supply `metrics`.
  - `pad(X: torch.Tensor, offsets: torch.Tensor | None, idx: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor | None]`.
    - The contract has `Sequence[int]`. `Iterable[int]` is wider, and pyright needs it to accept numpy index arrays.
    - It returns float32 `x` and a bool `mask` that is right-padded.

- [ ] **Step 1: Write the failing test**

`tests/test_probes.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import torch

from sonde import config
from sonde import probes

CASES = [
    ("linear", "mean", None),
    ("linear", "last", None),
    ("linear", "max", None),
    ("linear", "rolling_mean", 3),
    ("linear", "rolling_mean", 9),
    ("attention", "attention", None),
]
META = dict(
    name="p",
    block=2,
    window="prompt",
    threshold=0.5,
    model="m",
    engine="hf==0",
    model_fingerprint=None,
    prompt_format="raw",
)


@pytest.mark.parametrize("kind,pooling,rw", CASES)
def test_torch_matches_numpy_probe(kind, pooling, rw):
    torch.manual_seed(0)
    cfg = config.ProbeConfig(kind=kind, pooling=pooling, rolling_window=rw)
    module = probes.build(cfg, 8)
    lengths = [1, 4, 7]
    X = torch.randn(sum(lengths), 8) * 3 + 1
    offsets = torch.tensor([0, 1, 5, 12])
    mu, sigma = X.mean(0), X.std(0)
    x, mask = probes.pad(X, offsets, [0, 1, 2])
    with torch.no_grad():
        want = torch.sigmoid(module((x - mu) / sigma, mask)).numpy()
    p = probes.export(module, mu, sigma, **META)
    got = [
        p.pooled_score(X[offsets[i] : offsets[i + 1]].numpy()) for i in range(3)
    ]
    assert p.kind == kind and p.pooling == pooling
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_pooled_input_matches_numpy_probe():
    torch.manual_seed(0)
    module = probes.LinearProbe(8, "mean")
    X = torch.randn(5, 8)
    mu, sigma = X.mean(0), X.std(0)
    x, mask = probes.pad(X, None, [0, 3])
    assert mask is None and x.shape == (2, 8)
    with torch.no_grad():
        want = torch.sigmoid(module((x - mu) / sigma)).numpy()
    p = probes.export(module, mu, sigma, **META)
    got = [p.pooled_score(X[i].numpy()) for i in (0, 3)]
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_pad_shapes():
    X = torch.arange(12.0).reshape(6, 2)
    x, mask = probes.pad(X, torch.tensor([0, 2, 6]), [1, 0])
    assert x.shape == (2, 4, 2) and x.dtype == torch.float32
    assert mask is not None
    assert mask.tolist() == [[True] * 4, [True, True, False, False]]
    assert x[1, :2].tolist() == [[0.0, 1.0], [2.0, 3.0]]
```

The case lengths are `[1, 4, 7]`. With `rolling_window=3`, the 1-token sample takes the `T < window` rule and the other two take the windowed max. With `rolling_window=9`, the whole padded batch is shorter than the window.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probes.py -v`
Expected: FAIL during collection, with `ImportError: cannot import name 'probes' from 'sonde'`.

- [ ] **Step 3: Write minimal implementation**

`sonde/probes.py`:

```python
"""Torch probe modules that compute exactly what `probe.Probe` scores."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sonde import config
from sonde import probe


def _pool(
    z: torch.Tensor,
    mask: torch.Tensor,
    pooling: str,
    rolling_window: int | None,
) -> torch.Tensor:
    """Pools per-token logits z [B, T] over the valid tokens of mask [B, T]."""
    n = mask.sum(1)
    if pooling == "last":
        return z.gather(1, (n - 1)[:, None])[:, 0]
    if pooling == "max":
        return z.masked_fill(~mask, -torch.inf).max(1).values
    zm = z * mask
    mean = zm.sum(1) / n
    if pooling == "mean" or rolling_window is None:
        return mean
    if z.shape[1] < rolling_window:
        return mean
    w = rolling_window
    c = torch.nn.functional.pad(zm.cumsum(1), (1, 0))
    win = (c[:, w:] - c[:, :-w]) / w
    start = torch.arange(win.shape[1], device=z.device)
    win = win.masked_fill(start + w > n[:, None], -torch.inf)
    return torch.where(n < w, mean, win.max(1).values)


class LinearProbe(torch.nn.Module):
    """Linear logit per token, pooled: mean | last | max | rolling_mean."""

    def __init__(
        self, hidden: int, pooling: str, rolling_window: int | None = None
    ):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 1)
        self.pooling = pooling
        self.rolling_window = rolling_window

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pooled logits.

        Args:
            x: [B, H] pre-pooled features (mask None), or [B, T, H] tokens.
            mask: [B, T] bool, True on valid tokens, right-padded.

        Returns:
            [B] logits.
        """
        z = self.linear(x)[..., 0]
        if mask is None:
            return z
        return _pool(z, mask, self.pooling, self.rolling_window)


class AttentionProbe(torch.nn.Module):
    """softmax(x @ q) weighted sum of per-token logits x @ w + b."""

    def __init__(self, hidden: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 1)
        self.q = torch.nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-pooled logits.

        Args:
            x: [B, T, H] tokens.
            mask: [B, T] bool, True on valid tokens.

        Returns:
            [B] logits.
        """
        a = self.q(x)[..., 0].masked_fill(~mask, -torch.inf)
        return (torch.softmax(a, 1) * self.linear(x)[..., 0]).sum(1)


def build(cfg: config.ProbeConfig, hidden: int) -> LinearProbe | AttentionProbe:
    """Builds the untrained module that `cfg` describes."""
    if cfg.kind == "attention":
        return AttentionProbe(hidden)
    return LinearProbe(hidden, cfg.pooling, cfg.rolling_window)


def export(
    module: LinearProbe | AttentionProbe,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    **meta,
) -> probe.Probe:
    """Folds standardisation into raw-space weights and builds the artifact.

    The module saw (x - mu) / sigma, so w' = w / sigma and
    b' = b - sum(w * mu / sigma). q folds to q / sigma; its constant term is
    dropped because softmax is shift-invariant.

    Args:
        module: trained LinearProbe or AttentionProbe.
        mu: [H] train mean.
        sigma: [H] floored train std.
        **meta: the remaining Probe fields (name, block, window, threshold,
            model, engine, model_fingerprint, prompt_format, metrics, ...).

    Returns:
        The numpy Probe.
    """
    w = module.linear.weight.detach().cpu()[0].double()
    mu, sigma = mu.double().cpu(), sigma.double().cpu()
    bias = float(module.linear.bias.detach().cpu()[0]) - float(
        (w * mu / sigma).sum()
    )
    attention = isinstance(module, AttentionProbe)
    q = None
    if attention:
        q = (module.q.weight.detach().cpu()[0].double() / sigma).numpy()
    return probe.Probe(
        kind="attention" if attention else "linear",
        w=(w / sigma).numpy(),
        bias=bias,
        pooling="attention" if attention else module.pooling,
        q=q,
        rolling_window=None if attention else module.rolling_window,
        **meta,
    )


def pad(
    X: torch.Tensor, offsets: torch.Tensor | None, idx: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gathers samples idx into a float32 batch.

    Args:
        X: [n, H] pooled features, or flat [sum T, H] tokens.
        offsets: None for pooled, else [n + 1] row offsets into X.
        idx: sample indices.

    Returns:
        (x, mask): x [B, H] and None for pooled; x [B, T, H] right-padded
        with zeros and mask [B, T] bool for tokens.
    """
    idx = list(idx)
    if offsets is None:
        return X[idx].float(), None
    rows = [X[offsets[i] : offsets[i + 1]].float() for i in idx]
    x = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    n = torch.tensor([len(r) for r in rows])
    return x, torch.arange(x.shape[1])[None, :] < n[:, None]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probes.py -v && uv run ruff check sonde/probes.py tests/test_probes.py && uv run ruff format --check sonde/probes.py tests/test_probes.py`
Expected: `8 passed`, then `All checks passed!` and `2 files already formatted`.

- [ ] **Step 5: Commit**

```bash
git add sonde/probes.py tests/test_probes.py
git commit -m "feat(probes): torch linear/attention probes with exact numpy export"
```

---

### Task 3.2: Training, thresholds and metrics

**Files:**
- Create: `sonde/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes:
  - `probes.build`, `probes.LinearProbe` (`.linear`), `probes.pad` and `probes.export` from Task 3.1;
  - `config.ProbeConfig` fields `init`, `epochs`, `lr`, `weight_decay`, `batch_size`, `patience`, `pooling` and `rolling_window`;
  - `probe.Probe.pooled_score`;
  - `data.Sample` fields `text`, `messages` and `response` (phase 2).
- Produces:
  - `SIGMA_FLOOR = 1e-6`.
  - `standardise(X, offsets, train_idx: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]` returns `(mu [H], sigma [H])` in float32.
  - `fit(cfg: config.ProbeConfig, X, offsets, labels: np.ndarray, train_idx, val_idx, seed: int) -> tuple[probes.LinearProbe | probes.AttentionProbe, torch.Tensor, torch.Tensor]`.
    - It returns `(module on CPU, mu, sigma)`.
    - For `init: diff_means` it returns `mu = 0` and `sigma = 1`.
  - `diff_means(X, offsets, labels, train_idx) -> tuple[np.ndarray, float]` returns `(w, b)` in raw space.
  - `auroc(y, scores) -> float` raises `ValueError` when `y` lacks a class.
  - `choose_threshold(y, scores, max_fpr: float | None) -> tuple[float, bool]` returns `(threshold, threshold_failed)`.
  - `recall_at_fpr(y, scores, max_fpr: float) -> float`: recall at the set's own lowest threshold with FPR ≤ `max_fpr`; 0.0 when none qualifies. Not in the contract: `metrics` and sweep's bootstrap share it.
  - `group_auroc(y, scores, groups) -> tuple[float, int]` returns `(mean AUROC over groups holding both classes, n_groups)`, and `(nan, 0)` when no group qualifies.
  - `bootstrap_ci(y, scores, fn, seed: int, n: int = 1000) -> tuple[float, float]` returns the 95% percentile interval of `fn(y, scores)`. Resamples that lose a class are skipped.
  - `metrics(y, scores, threshold: float, max_fpr: float | None, groups=None) -> dict`. It never raises on a single class.
    - Keys: `n_pos`, `n_neg`, `threshold`, `auroc`, `f1`, `precision`, `recall`, `fpr`, `small_n`; `recall_at_fpr` when `max_fpr` is set; `group_auroc` and `n_groups` when `groups` is given.
    - A benign-only set gets `fpr` with `recall`, `auroc`, `f1`, `precision` and `recall_at_fpr` set to `None`. A positive-only set gets `recall` with `fpr` and the rest set to `None`.
    - `small_n = max_fpr is not None and n_neg * max_fpr < 10`; when true it logs a warning.
  - `probe_scores(p: probe.Probe, X, offsets, idx) -> np.ndarray` returns `[len(idx)]` float32 scores.
  - `bow_features(texts: list[str], train_idx, vocab_size: int = 5000) -> np.ndarray` returns `[n, V]` float32 counts of lowercase `re.findall(r"\w+")` words over the train split's `vocab_size` most frequent words.
  - `window_text(sample: data.Sample, window: str) -> str`: the prompt for `prompt` (the message contents joined by newlines, or the text), the response for `response`, both for `all`, and the last message for `last_turn`.
- Simplification: shuffling uses `torch.randperm(..., generator=local_gen)`, not a `DataLoader`. Reproducibility is the same and it is less code.

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from sonde import config
from sonde import data
from sonde import probes
from sonde import train

META = dict(
    name="p",
    block=0,
    window="prompt",
    threshold=0.5,
    model="m",
    engine="hf==0",
    model_fingerprint=None,
    prompt_format="raw",
)


def _data(n=80, h=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = np.arange(n) % 2
    X = torch.randn(n, h, generator=g) * 5 + 10
    X[:, 0] += torch.as_tensor(y, dtype=torch.float32) * 15
    return X, y


def test_auroc_rank_based_with_ties():
    y = np.array([0, 0, 1, 1])
    assert train.auroc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert train.auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert train.auroc(y, np.array([0.1, 0.2, 0.3, 0.3])) == 1.0


def test_threshold_at_max_fpr():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
    assert train.choose_threshold(y, s, 0.25) == (0.4, False)
    assert train.choose_threshold(y, s, 0.0) == (0.8, False)


def test_threshold_failed_falls_back_to_best_f1():
    y = np.array([0, 1, 1])
    s = np.array([0.9, 0.9, 0.2])
    thr, failed = train.choose_threshold(y, s, 0.0)
    assert failed and thr == 0.2


def test_best_f1_threshold():
    y = np.array([0, 0, 1, 1])
    assert train.choose_threshold(y, np.array([0.1, 0.6, 0.5, 0.9]), None) == (
        0.5,
        False,
    )


def test_metrics_counts():
    y = np.array([0, 0, 0, 1, 1])
    m = train.metrics(y, np.array([0.1, 0.2, 0.9, 0.8, 0.95]), 0.5, 0.0)
    assert (m["n_pos"], m["n_neg"]) == (2, 3)
    assert m["recall"] == 1.0 and m["fpr"] == 1 / 3
    assert m["precision"] == 2 / 3 and m["recall_at_fpr"] == 0.5
    assert train.metrics(np.zeros(3), np.ones(3), 0.5, None)["auroc"] is None


def test_recall_at_fpr():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
    assert train.recall_at_fpr(y, s, 0.25) == 1.0
    assert train.recall_at_fpr(y, s, 0.0) == 0.5


def test_single_class_reports_one_side_without_raising():
    s = np.array([0.1, 0.2, 0.6, 0.9])
    benign = train.metrics(np.zeros(4), s, 0.5, 0.01)
    assert benign["fpr"] == 0.5 and benign["recall"] is None
    assert benign["auroc"] is None and benign["recall_at_fpr"] is None
    positive = train.metrics(np.ones(4), s, 0.5, 0.01)
    assert positive["recall"] == 0.5 and positive["fpr"] is None


def test_small_n_flag(caplog):
    y = np.r_[np.zeros(2000), np.ones(10)]
    assert not train.metrics(y, y, 0.5, 0.01)["small_n"]
    assert not train.metrics(y, y, 0.5, None)["small_n"]
    assert train.metrics(y[1500:], y[1500:], 0.5, 0.01)["small_n"]
    assert "small_n: 500 negatives" in caplog.text


def test_group_auroc_on_hand_built_groups():
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.2, 0.5, 0.6, 0.3, 0.4])
    g = ["a", "a", "b", "b", "c", "c", "d", "d"]
    assert train.group_auroc(y, s, g) == (0.5, 2)
    assert train.group_auroc(y[4:], s[4:], g[4:])[1] == 0
    m = train.metrics(y, s, 0.5, None, groups=g)
    assert (m["group_auroc"], m["n_groups"]) == (0.5, 2)


def test_bootstrap_ci_contains_point_estimate():
    y = np.arange(200) % 2
    s = y + np.random.default_rng(0).normal(size=200)
    low, high = train.bootstrap_ci(y, s, train.auroc, seed=0)
    assert low < train.auroc(y, s) < high
    assert train.bootstrap_ci(y, s, train.auroc, seed=0) == (low, high)


def test_sigma_floor():
    X = torch.ones(6, 3)
    mu, sigma = train.standardise(X, None, [0, 1, 2])
    assert torch.all(sigma == train.SIGMA_FLOOR) and torch.all(mu == 1)


def test_standardise_tokens_uses_train_tokens_only():
    X = torch.tensor([[1.0], [3.0], [100.0]])
    mu, _ = train.standardise(X, torch.tensor([0, 2, 3]), [0])
    assert mu.item() == 2.0


def test_standardisation_fold_is_exact():
    X, y = _data()
    cfg = config.ProbeConfig(epochs=3, lr=1e-2, batch_size=16)
    module, mu, sigma = train.fit(cfg, X, None, y, range(60), range(60, 80), 0)
    p = probes.export(module, mu, sigma, **META)
    with torch.no_grad():
        want = torch.sigmoid(module((X - mu) / sigma)).numpy()
    got = train.probe_scores(p, X, None, range(80))
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_fit_is_seed_reproducible_and_learns():
    X, y = _data()
    cfg = config.ProbeConfig(epochs=20, lr=1e-2, batch_size=16, patience=0)
    a, _, _ = train.fit(cfg, X, None, y, range(60), range(60, 80), 7)
    b, _, _ = train.fit(cfg, X, None, y, range(60), range(60, 80), 7)
    assert torch.equal(a.linear.weight, b.linear.weight)
    with torch.no_grad():
        s = a((X[60:] - X[:60].mean(0)) / X[:60].std(0)).numpy()
    assert train.auroc(y[60:], s) > 0.9


def test_diff_means_closed_form():
    X = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 0.0], [2.0, 0.0]])
    y = np.array([1, 1, 0, 0])
    w, b = train.diff_means(X, None, y, [0, 1, 2, 3])
    np.testing.assert_allclose(w, [1.0, 1.0])
    assert b == -(1.0 * 3.0 + 1.0 * 1.0) / 2
    cfg = config.ProbeConfig(init="diff_means", epochs=0)
    module, mu, sigma = train.fit(cfg, X, None, y, [0, 1, 2, 3], [0], 0)
    p = probes.export(module, mu, sigma, **META)
    np.testing.assert_allclose(p.w, [1.0, 1.0])
    assert p.bias == b


def test_diff_means_tokens_mean_pools_each_sample():
    X = torch.tensor([[0.0], [2.0], [5.0]])
    w, _ = train.diff_means(
        X, torch.tensor([0, 2, 3]), np.array([0, 1]), [0, 1]
    )
    np.testing.assert_allclose(w, [4.0])


def test_bow_baseline_separates_keyword_only_data():
    np.testing.assert_array_equal(
        train.bow_features(["Bomb bomb!", "x"], [0]), [[2.0], [0.0]]
    )
    rng = np.random.default_rng(0)
    filler = ["the", "a", "cat", "dog", "ran", "sat", "on", "mat"]
    y = np.arange(80) % 2
    texts = [
        " ".join(
            [*rng.choice(filler, 6), "bomb"] if label else rng.choice(filler, 7)
        )
        for label in y
    ]
    X = torch.from_numpy(train.bow_features(texts, range(60)))
    assert X.shape == (80, 9)
    cfg = config.ProbeConfig(epochs=30, lr=1e-2, batch_size=16)
    module, mu, sigma = train.fit(cfg, X, None, y, range(60), range(60, 80), 0)
    with torch.no_grad():
        s = module((X[60:] - mu) / sigma).numpy()
    assert train.auroc(y[60:], s) == 1.0


def test_window_text():
    s = data.Sample(
        id="0",
        text=None,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
        response="ok",
        response_ids=None,
        label=1,
        group=None,
        raw={},
    )
    assert train.window_text(s, "prompt") == "hi\nyo"
    assert train.window_text(s, "response") == "ok"
    assert train.window_text(s, "all") == "hi\nyo\nok"
    assert train.window_text(s, "last_turn") == "yo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL during collection, with `ImportError: cannot import name 'train' from 'sonde'`.

- [ ] **Step 3: Write minimal implementation**

`sonde/train.py`:

```python
"""Probe fitting, thresholds and numpy metrics."""

from __future__ import annotations

import collections
import copy
import logging
import re
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
import torch

from sonde import config
from sonde import data
from sonde import probe
from sonde import probes

SIGMA_FLOOR = 1e-6

_log = logging.getLogger(__name__)


def _rows(X: torch.Tensor, offsets: torch.Tensor | None, i: int):
    """[H] for pooled, [T, H] for tokens."""
    return X[i] if offsets is None else X[offsets[i] : offsets[i + 1]]


def standardise(
    X: torch.Tensor, offsets: torch.Tensor | None, train_idx: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-split feature statistics.

    Args:
        X: [n, H] pooled, or flat [sum T, H] tokens.
        offsets: None, or [n + 1] token offsets.
        train_idx: train sample indices.

    Returns:
        (mu [H], sigma [H]) float32; sigma floored at SIGMA_FLOOR. Token data
        uses every window token of the train samples.
    """
    if offsets is None:
        rows = X[list(train_idx)].float()
    else:
        rows = torch.cat([_rows(X, offsets, i) for i in train_idx]).float()
    return rows.mean(0), rows.std(0).clamp_min(SIGMA_FLOOR)


def diff_means(
    X: torch.Tensor,
    offsets: torch.Tensor | None,
    labels: np.ndarray,
    train_idx: Iterable[int],
) -> tuple[np.ndarray, float]:
    """Raw-space difference of class means: w = mu_pos - mu_neg.

    Args:
        X: [n, H] pooled, or flat [sum T, H] tokens.
        offsets: None, or [n + 1]; token samples are mean-pooled first.
        labels: [n] in {0, 1}.
        train_idx: train sample indices.

    Returns:
        (w [H] float64, b) with b = -w . (mu_pos + mu_neg) / 2.
    """
    idx = list(train_idx)
    if offsets is None:
        feats = X[idx].double().numpy()
    else:
        feats = torch.stack(
            [_rows(X, offsets, i).double().mean(0) for i in idx]
        ).numpy()
    y = labels[idx]
    pos, neg = feats[y == 1].mean(0), feats[y == 0].mean(0)
    w = pos - neg
    return w, float(-w @ (pos + neg) / 2)


def fit(
    cfg: config.ProbeConfig,
    X: torch.Tensor,
    offsets: torch.Tensor | None,
    labels: np.ndarray,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    seed: int,
) -> tuple[
    probes.LinearProbe | probes.AttentionProbe, torch.Tensor, torch.Tensor
]:
    """Trains one probe on one block.

    AdamW on CUDA when available. The seed is set before the module is
    built; a local generator shuffles batches. pos_weight = n_neg / n_pos;
    weight decay skips biases. Early stopping on val loss with
    cfg.patience (0 = off); the best epoch's weights are returned.

    Args:
        cfg: probe section.
        X: [n, H] pooled, or flat [sum T, H] tokens.
        offsets: None, or [n + 1].
        labels: [n] in {0, 1}.
        train_idx: train sample indices.
        val_idx: val sample indices.
        seed: run seed.

    Returns:
        (module on CPU, mu [H], sigma [H]); diff_means returns mu = 0 and
        sigma = 1 so export leaves its raw-space weights untouched.
    """
    hidden = X.shape[1]
    if cfg.init == "diff_means":
        w, b = diff_means(X, offsets, labels, train_idx)
        module = probes.LinearProbe(hidden, cfg.pooling, cfg.rolling_window)
        with torch.no_grad():
            module.linear.weight[0] = torch.from_numpy(w)
            module.linear.bias[0] = b
        return module, torch.zeros(hidden), torch.ones(hidden)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sigma = standardise(X, offsets, train_idx)
    torch.manual_seed(seed)
    module = probes.build(cfg, hidden).to(device)
    y = torch.as_tensor(labels, dtype=torch.float32)
    tr = np.asarray(list(train_idx), dtype=np.int64)
    va = np.asarray(list(val_idx), dtype=np.int64)
    n_pos = float(labels[tr].sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(tr) - n_pos) / n_pos, device=device)
    )
    params = list(module.named_parameters())
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in params if not n.endswith("bias")],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in params if n.endswith("bias")],
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.lr,
    )
    gen = torch.Generator().manual_seed(seed)
    mu_d, sigma_d = mu.to(device), sigma.to(device)
    val_chunks = [
        va[i : i + cfg.batch_size] for i in range(0, len(va), cfg.batch_size)
    ]

    def loss_on(idx: np.ndarray) -> torch.Tensor:
        x, mask = probes.pad(X, offsets, idx)
        x = (x.to(device) - mu_d) / sigma_d
        mask = None if mask is None else mask.to(device)
        return loss_fn(module(x, mask), y[idx].to(device))

    best, best_loss, bad = copy.deepcopy(module.state_dict()), np.inf, 0
    for _ in range(cfg.epochs):
        module.train()
        for b in torch.randperm(len(tr), generator=gen).split(cfg.batch_size):
            loss = loss_on(tr[b.numpy()])
            opt.zero_grad()
            loss.backward()
            opt.step()
        module.eval()
        with torch.no_grad():
            val_loss = sum(
                float(loss_on(c)) * len(c) for c in val_chunks
            ) / len(va)
        if val_loss < best_loss:
            best, best_loss = copy.deepcopy(module.state_dict()), val_loss
            bad = 0
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                break
    module.load_state_dict(best)
    return module.cpu(), mu, sigma


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUROC; tied scores share their mean rank.

    Raises:
        ValueError: y lacks a class.
    """
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("auroc needs both classes")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inv]
    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _counts(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray):
    """(tp, fp) at each threshold for the rule score >= threshold."""
    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    return tp, fp


def choose_threshold(
    y: np.ndarray, scores: np.ndarray, max_fpr: float | None
) -> tuple[float, bool]:
    """Operating threshold on probe scores in [0, 1].

    With max_fpr: the lowest score threshold whose FPR <= max_fpr. When
    none qualifies (negatives tie at the top score), falls back to best F1
    and reports failure. Without max_fpr: the best-F1 threshold.

    Returns:
        (threshold, threshold_failed).
    """
    y, scores = np.asarray(y), np.asarray(scores)
    cand = np.unique(scores)
    tp, fp = _counts(y, scores, cand)
    if max_fpr is not None:
        ok = fp / max(int((y == 0).sum()), 1) <= max_fpr
        if ok.any():
            return float(cand[ok.argmax()]), False
    f1 = 2 * tp / np.maximum(tp + fp + int(y.sum()), 1)
    return float(cand[f1.argmax()]), max_fpr is not None


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Recall at this set's own lowest threshold with FPR <= max_fpr.

    Args:
        y: [n] labels with both classes.
        scores: [n] scores.
        max_fpr: the FPR budget.

    Returns:
        The recall; 0.0 when no threshold meets the budget.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    thr, failed = choose_threshold(y, scores, max_fpr)
    return 0.0 if failed else float((scores[y == 1] >= thr).mean())


def group_auroc(
    y: np.ndarray, scores: np.ndarray, groups: Sequence
) -> tuple[float, int]:
    """Mean AUROC within each group that holds both classes.

    GRPO's group-relative advantage only uses the ranking among one
    prompt's completions, so this is the headline for reward probes.

    Args:
        y: [n] labels in {0, 1}.
        scores: [n] scores.
        groups: [n] group keys.

    Returns:
        (mean within-group AUROC, n_groups); (nan, 0) when no group holds
        both classes.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    members = collections.defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    vals = [
        auroc(y[i], scores[i])
        for i in members.values()
        if 0 < y[i].sum() < len(i)
    ]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    seed: int,
    n: int = 1000,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of fn(y, scores).

    Args:
        y: [m] labels with both classes.
        scores: [m] scores.
        fn: the statistic, e.g. auroc.
        seed: numpy seed for the resamples.
        n: resamples; those that lose a class are skipped.

    Returns:
        (low, high).
    """
    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            stats.append(fn(y[i], scores[i]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_fpr: float | None,
    groups: Sequence | None = None,
) -> dict:
    """Eval card for one labelled set. Never raises on a single class.

    A benign-only set reports fpr only and a positive-only set recall
    only; metrics that need both classes are None.

    Args:
        y: [n] labels in {0, 1}.
        scores: [n] scores.
        threshold: the operating threshold.
        max_fpr: FPR budget for recall_at_fpr and small_n, or None.
        groups: [n] group keys, or None to skip group_auroc.

    Returns:
        n_pos, n_neg, threshold, auroc, f1, precision, recall and fpr at
        threshold, small_n (n_neg * max_fpr < 10), recall_at_fpr when
        max_fpr is set, and group_auroc with n_groups when groups is set.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    both = n_pos > 0 and n_neg > 0
    (tp,), (fp,) = _counts(y, scores, np.array([threshold]))
    out = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": float(threshold),
        "auroc": auroc(y, scores) if both else None,
        "f1": float(2 * tp / (tp + fp + n_pos)) if both else None,
        "precision": float(tp / max(tp + fp, 1)) if both else None,
        "recall": float(tp / n_pos) if n_pos else None,
        "fpr": float(fp / n_neg) if n_neg else None,
        "small_n": max_fpr is not None and n_neg * max_fpr < 10,
    }
    if out["small_n"]:
        _log.warning(
            "small_n: %d negatives x max_fpr %s < 10; the FPR threshold "
            "rests on too few negatives",
            n_neg,
            max_fpr,
        )
    if max_fpr is not None:
        out["recall_at_fpr"] = (
            recall_at_fpr(y, scores, max_fpr) if both else None
        )
    if groups is not None:
        value, out["n_groups"] = group_auroc(y, scores, groups)
        out["group_auroc"] = value if out["n_groups"] else None
    return out


def probe_scores(
    p: probe.Probe,
    X: torch.Tensor,
    offsets: torch.Tensor | None,
    idx: Iterable[int],
) -> np.ndarray:
    """[len(idx)] float32 scores from the numpy artifact's pooled_score."""
    return np.array(
        [p.pooled_score(_rows(X, offsets, i).float().numpy()) for i in idx],
        dtype=np.float32,
    )


def bow_features(
    texts: list[str], train_idx: Iterable[int], vocab_size: int = 5000
) -> np.ndarray:
    """Lowercase word counts over the train split's most frequent words.

    Args:
        texts: [n] window texts.
        train_idx: train sample indices; only they build the vocabulary.
        vocab_size: words kept.

    Returns:
        [n, V] float32 counts, V <= vocab_size.
    """
    words = [re.findall(r"\w+", t.lower()) for t in texts]
    top = collections.Counter(w for i in train_idx for w in words[i])
    vocab = {w: j for j, (w, _) in enumerate(top.most_common(vocab_size))}
    # ponytail: dense [n, V]; use scipy.sparse if n * V outgrows RAM.
    out = np.zeros((len(texts), len(vocab)), np.float32)
    for i, row in enumerate(words):
        for w in row:
            if w in vocab:
                out[i, vocab[w]] += 1
    return out


def window_text(sample: data.Sample, window: str) -> str:
    """The text a window covers, for the bag-of-words baseline.

    Args:
        sample: a loaded row.
        window: one of probe.WINDOWS.

    Returns:
        The prompt for "prompt", the response for "response", both for
        "all", and the last message for "last_turn".
    """
    turns = [m["content"] for m in sample.messages or []] or [sample.text]
    prompt = "\n".join(t or "" for t in turns)
    if window == "prompt":
        return prompt
    if window == "response":
        return sample.response or ""
    if window == "all":
        return f"{prompt}\n{sample.response or ''}"
    return turns[-1] or ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v && uv run ruff check sonde/train.py tests/test_train.py && uv run ruff format --check sonde/train.py tests/test_train.py`
Expected: `18 passed`, then `All checks passed!` and `2 files already formatted`.

- [ ] **Step 5: Commit**

```bash
git add sonde/train.py tests/test_train.py
git commit -m "feat(train): seeded fit, numpy metrics with recall@FPR, group AUROC, bootstrap CIs and bag-of-words features"
```

---

### Task 3.3: The train step (`sweep.run`)

**Files:**
- Create: `sonde/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes:
  - `store.read_manifest(out_dir) -> dict`.
  - `store.read_layer(out_dir, block) -> (X, offsets | None)`.
  - `store.ShardWriter(out_dir, keep, shard_size, shard_bytes)` with `.add(features: dict[int, list[Tensor]])` and `.close() -> list[str]` (test only).
  - `store.write_manifest(out_dir, manifest)` (test only).
  - `data.split(labels, groups, fractions, seed) -> dict[str, list[int]]`.
  - `data.load_samples(src, seed) -> list[Sample]`, for the raw texts. Samples are matched to the manifest by `id`, because extract drops rows.
  - `fingerprint.checkpoint_fingerprint(model, revision=None, quantization=None) -> str`.
  - `train.fit`, `train.choose_threshold`, `train.metrics`, `train.probe_scores`, `train.auroc`, `train.recall_at_fpr`, `train.bootstrap_ci`, `train.bow_features` and `train.window_text` (Task 3.2), and `probes.export` (Task 3.1).
  - `probe.Probe.save` / `load`.
  - `config.RunConfig`, `config.DataConfig`, `config.ExtractConfig` and `config.ProbeConfig`.
- Produces:
  - `run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None`, which writes:
    - `run_dir/"splits.json"` as `{train|val|test: [sample ids]}`;
    - `run_dir/"layers"/f"B{b}.npz"` for every block, each named `f"{cfg.name}@B{b}"` with `metrics={"val": ...}`;
    - `run_dir/"probes"/f"{cfg.name}.npz"`, the only file there;
    - `run_dir/"metrics.json"`.
  - `metrics.json` keys are:
    - `headline`, on test, with exactly the spec §9 step 7 fields: `max_fpr`; `recall_at_fpr` and `recall_at_fpr_ci` (`None` when `max_fpr` is unset); `auroc` and `auroc_ci`; `baseline_recall_at_fpr`, `baseline_auroc`, `length_auroc`; `group_auroc` (only when `data.group` is set); `n_pos`, `n_neg`, `small_n`. CIs are `[low, high]` from `train.bootstrap_ci` with `cfg.seed`;
    - `block`, `select`, `select_fallback` (`"auroc"` when `select: recall_at_fpr` meets a `small_n` val, else `None`), `max_fpr`, `threshold`, `threshold_failed`;
    - `val` (`{str(block): metrics}`), `test` and `baseline` (the bag-of-words probe's test metrics at its val threshold);
    - `control_auroc`, `controls_passed`;
    - `n` (`{train, val, test}` counts).
  - The last log line of the step is `headline {json}`.
  - The exported probe's `metrics` equals that dict.
  - `model_fingerprint = fingerprint.checkpoint_fingerprint(manifest["model"], revision=manifest["revision"], quantization=None)`.
  - `prompt_format`, `engine`, `window` and `model` come from the manifest.

- [ ] **Step 1: Write the failing test**

`tests/test_sweep.py`:

```python
from __future__ import annotations

import json
import logging

import pytest
import torch

from sonde import config
from sonde import probe
from sonde import store
from sonde import sweep


def _fake_extract(run_dir, keep, n=120, h=8):
    ext = run_dir / "extract"
    ext.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    y = [i % 2 for i in range(n)]
    (run_dir / "rows.jsonl").write_text(
        "".join(
            json.dumps({"id": f"r{i}", "text": f"row {i}", "label": y[i]})
            + "\n"
            for i in range(n)
        )
    )
    lengths = [1 + i % 5 for i in range(n)]
    feats = {}
    for b in (0, 1, 2):
        rows = []
        for label, t in zip(y, lengths, strict=True):
            x = torch.randn(t, h, generator=g)
            if b == 1:
                x[:, 0] += 4.0 * label
            rows.append(x.mean(0) if keep == "pooled" else x)
        feats[b] = rows
    writer = store.ShardWriter(ext, keep, shard_size=50, shard_bytes=2**31)
    writer.add(feats)
    store.write_manifest(
        ext,
        {
            "model": "gpt2",
            "revision": None,
            "model_fingerprint": None,
            "prompt_format": "raw",
            "engine": "hf==0+nnterp==1.3.0",
            "blocks": [0, 1, 2],
            "window": "prompt",
            "keep": keep,
            "pooling": "mean" if keep == "pooled" else None,
            "dtype": "float32",
            "ids": [f"r{i}" for i in range(n)],
            "labels": y,
            "groups": [None] * n,
            "shards": writer.close(),
            "drops": {},
            "config_hash": "x",
        },
    )


def _cfg(run_dir, keep, **probe_kw):
    return config.RunConfig(
        name="syn",
        data=config.DataConfig(
            path=str(run_dir / "rows.jsonl"), split=(0.5, 0.25, 0.25)
        ),
        extract=config.ExtractConfig(keep=keep),
        probe=config.ProbeConfig(lr=1e-2, epochs=30, batch_size=16, **probe_kw),
        steps=["train"],
    )


@pytest.mark.parametrize(
    "keep,probe_kw",
    [
        ("pooled", {}),
        ("tokens", {"kind": "attention", "pooling": "attention"}),
        ("tokens", {"pooling": "rolling_mean", "rolling_window": 2}),
        ("pooled", {"select": "auroc", "max_fpr": None}),
    ],
)
def test_selects_planted_block(tmp_path, keep, probe_kw):
    _fake_extract(tmp_path, keep)
    sweep.run(_cfg(tmp_path, keep, **probe_kw), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["block"] == 1
    assert m["test"]["auroc"] > 0.9
    assert [f.name for f in (tmp_path / "probes").iterdir()] == ["syn.npz"]
    p = probe.Probe.load(str(tmp_path / "probes" / "syn.npz"))
    assert (p.name, p.block, p.prompt_format) == ("syn", 1, "raw")
    assert (p.model_fingerprint or "").startswith("hub:")
    layers = sorted(f.name for f in (tmp_path / "layers").iterdir())
    assert layers == ["B0.npz", "B1.npz", "B2.npz"]
    assert probe.Probe.load(str(tmp_path / "layers" / "B2.npz")).name == (
        "syn@B2"
    )
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert sum(map(len, splits.values())) == 120


def test_control_is_recorded_not_raised(tmp_path):
    _fake_extract(tmp_path, "pooled")
    sweep.run(_cfg(tmp_path, "pooled"), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert 0.0 <= m["control_auroc"] <= 1.0
    assert m["controls_passed"] == (m["control_auroc"] < 0.6)


def test_refuses_pooling_mismatch(tmp_path):
    _fake_extract(tmp_path, "pooled")
    with pytest.raises(ValueError, match="pooling"):
        sweep.run(_cfg(tmp_path, "pooled", pooling="last"), tmp_path)


def test_headline_holds_probe_and_baseline(tmp_path, caplog):
    _fake_extract(tmp_path, "pooled")
    with caplog.at_level(logging.INFO):
        sweep.run(_cfg(tmp_path, "pooled"), tmp_path)
    h = json.loads((tmp_path / "metrics.json").read_text())["headline"]
    assert h["max_fpr"] == 0.01
    assert 0.0 <= h["recall_at_fpr"] <= 1.0
    assert 0.0 <= h["baseline_recall_at_fpr"] <= 1.0
    low, high = h["auroc_ci"]
    assert low <= h["auroc"] <= high
    assert {"baseline_auroc", "length_auroc", "n_pos", "small_n"} <= h.keys()
    assert caplog.records[-1].getMessage() == f"headline {json.dumps(h)}"


def test_small_n_val_falls_back_to_auroc(tmp_path):
    _fake_extract(tmp_path, "pooled")
    sweep.run(_cfg(tmp_path, "pooled"), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["val"]["1"]["small_n"] and m["select_fallback"] == "auroc"
    assert m["block"] == 1
    sweep.run(_cfg(tmp_path, "pooled", max_fpr=0.9), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert not m["val"]["1"]["small_n"] and m["select_fallback"] is None
    assert m["block"] == 1
```

What the tests pin down:
- `_fake_extract` also writes `rows.jsonl`, because the baseline loads the raw texts with `data.load_samples`. The texts (`row {i}`) carry no label signal.
- With the new defaults (`select: recall_at_fpr`, `max_fpr: 0.01`), val has about 15 negatives, so it is `small_n` and selection falls back to AUROC. `max_fpr: 0.9` lifts `n_neg × max_fpr` above 10 and the fallback goes away.
- The last parametrize case runs with `max_fpr: null` and `select: auroc`, the best-F1 path.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: FAIL during collection, with `ImportError: cannot import name 'sweep' from 'sonde'`.

- [ ] **Step 3: Write minimal implementation**

`sonde/sweep.py`:

```python
"""The train step: split, fit every block, select, test once, control."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import re
import shutil

import numpy as np
import torch

from sonde import config
from sonde import data
from sonde import fingerprint
from sonde import probes
from sonde import store
from sonde import train

_log = logging.getLogger(__name__)


def _at(groups, idx):
    """groups restricted to idx, or None when the run has no groups."""
    return None if groups is None else [groups[i] for i in idx]


def _fit(pc, X, offsets, y, sp, seed, groups, **meta):
    """Fits on train, thresholds on val; returns (Probe, val metrics)."""
    module, mu, sigma = train.fit(
        pc, X, offsets, y, sp["train"], sp["val"], seed
    )
    p = probes.export(module, mu, sigma, threshold=0.5, **meta)
    s = train.probe_scores(p, X, offsets, sp["val"])
    thr, failed = train.choose_threshold(y[sp["val"]], s, pc.max_fpr)
    val = train.metrics(
        y[sp["val"]], s, thr, pc.max_fpr, _at(groups, sp["val"])
    )
    val["threshold_failed"] = failed
    return dataclasses.replace(p, threshold=thr), val


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Writes splits.json, layers/B{b}.npz, probes/{name}.npz, metrics.json.

    Raises:
        ValueError: a sample has no label, or pooled features were pooled
            differently from cfg.probe.pooling.
    """
    pc, dc = cfg.probe, cfg.data
    if pc is None or dc is None:
        raise ValueError("train needs the data and probe sections")
    ext = run_dir / "extract"
    man = store.read_manifest(ext)
    if man["keep"] == "pooled" and man["pooling"] != pc.pooling:
        raise ValueError(
            f"extract pooled with {man['pooling']!r} but probe.pooling is "
            f"{pc.pooling!r}; re-extract or match them"
        )
    if None in man["labels"]:
        raise ValueError("train needs a label on every sample")
    y = np.asarray(man["labels"])
    groups = man["groups"] if dc.group else None
    sp = data.split(man["labels"], man["groups"], dc.split, cfg.seed)
    (run_dir / "splits.json").write_text(
        json.dumps({k: [man["ids"][i] for i in v] for k, v in sp.items()})
    )
    meta = {
        "window": man["window"],
        "model": man["model"],
        "engine": man["engine"],
        "model_fingerprint": fingerprint.checkpoint_fingerprint(
            man["model"], revision=man["revision"], quantization=None
        ),
        "prompt_format": man["prompt_format"],
    }
    layers = run_dir / "layers"
    shutil.rmtree(layers, ignore_errors=True)
    layers.mkdir()
    fitted, table = {}, {}
    for b in man["blocks"]:
        X, offsets = store.read_layer(ext, b)
        p, table[b] = _fit(
            pc,
            X,
            offsets,
            y,
            sp,
            cfg.seed,
            groups,
            name=f"{cfg.name}@B{b}",
            block=b,
            **meta,
        )
        fitted[b] = dataclasses.replace(p, metrics={"val": table[b]})
        fitted[b].save(str(layers / f"B{b}.npz"))
        _log.info("block %d: val %s", b, table[b])
    small_val = table[man["blocks"][0]]["small_n"]
    fallback = pc.select == "recall_at_fpr" and small_val
    key = "auroc" if fallback else pc.select
    best = max(table, key=lambda b: (table[b][key] or 0.0, table[b]["auroc"]))
    X, offsets = store.read_layer(ext, best)
    p = fitted[best]
    te = sp["test"]
    scores = train.probe_scores(p, X, offsets, te)
    test = train.metrics(
        y[te], scores, p.threshold, pc.max_fpr, _at(groups, te)
    )
    shuffled = np.random.default_rng(cfg.seed).permutation(y)
    control, _ = _fit(
        pc,
        X,
        offsets,
        shuffled,
        sp,
        cfg.seed,
        None,
        name="control",
        block=best,
        **meta,
    )
    control_auroc = train.auroc(
        y[sp["val"]], train.probe_scores(control, X, offsets, sp["val"])
    )
    by_id = {s.id: s for s in data.load_samples(dc, cfg.seed)}
    texts = [train.window_text(by_id[i], man["window"]) for i in man["ids"]]
    bow = torch.from_numpy(train.bow_features(texts, sp["train"]))
    bow_pc = pc.model_copy(
        update={"kind": "linear", "pooling": "mean", "rolling_window": None}
    )
    bow_probe, _ = _fit(
        bow_pc, bow, None, y, sp, cfg.seed, None, name="bow", block=0, **meta
    )
    baseline = train.metrics(
        y[te],
        train.probe_scores(bow_probe, bow, None, te),
        bow_probe.threshold,
        pc.max_fpr,
    )
    # ponytail: word count stands in for the window's token count, which
    # the manifest does not store; add per-sample lengths there if needed.
    length = np.array([len(re.findall(r"\w+", t)) for t in texts])
    max_fpr = pc.max_fpr
    headline = {
        "max_fpr": max_fpr,
        "recall_at_fpr": test.get("recall_at_fpr"),
        "recall_at_fpr_ci": None
        if max_fpr is None
        else train.bootstrap_ci(
            y[te],
            scores,
            lambda a, s: train.recall_at_fpr(a, s, max_fpr),
            cfg.seed,
        ),
        "auroc": test["auroc"],
        "auroc_ci": train.bootstrap_ci(y[te], scores, train.auroc, cfg.seed),
        "baseline_recall_at_fpr": baseline.get("recall_at_fpr"),
        "baseline_auroc": baseline["auroc"],
        "length_auroc": train.auroc(y[te], length[te]),
        "n_pos": test["n_pos"],
        "n_neg": test["n_neg"],
        "small_n": test["small_n"],
    }
    if groups is not None:
        headline["group_auroc"] = test["group_auroc"]
    card = {
        "headline": headline,
        "block": best,
        "select": pc.select,
        "select_fallback": "auroc" if fallback else None,
        "max_fpr": max_fpr,
        "threshold": p.threshold,
        "threshold_failed": table[best]["threshold_failed"],
        "val": {str(b): v for b, v in table.items()},
        "test": test,
        "baseline": baseline,
        "control_auroc": control_auroc,
        "controls_passed": control_auroc < 0.6,
        "n": {k: len(v) for k, v in sp.items()},
    }
    out = run_dir / "probes"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    dataclasses.replace(p, name=cfg.name, metrics=card).save(
        str(out / f"{cfg.name}.npz")
    )
    (run_dir / "metrics.json").write_text(json.dumps(card, indent=2))
    _log.info(
        "selected block %d; test %s; control_auroc %.3f",
        best,
        test,
        control_auroc,
    )
    _log.info("headline %s", json.dumps(headline))
```

Notes for the implementer:
- `pc.select` is `"auroc"`, `"recall_at_fpr"` or `"group_auroc"`, which are exactly the metric keys, so no mapping is needed. `or 0.0` ranks a block whose val has no mixed group (`group_auroc: None`) last.
- `small_n` depends only on val's `n_neg`, which every block shares, so the first block's flag decides the fallback.
- `groups` is passed to `metrics` only when `data.group` is set; `data.split` always gets the manifest groups.
- `data.split` gets `man["labels"]` (a list) because its parameter is `Sequence[int]`; pyright rejects the ndarray.
- The baseline reuses `_fit`: a `LinearProbe` (`kind: linear`, `pooling: mean`, pooled features) with the run's `fit` settings, thresholded on val at `max_fpr`, scored once on test. With `init: diff_means` it is the closed-form difference of means on word counts.
- The control permutes all labels, so its early stopping is blind too. Its AUROC is measured on val against the true labels.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep.py -v && uv run ruff check sonde/sweep.py tests/test_sweep.py && uv run ruff format --check sonde/sweep.py tests/test_sweep.py`
Expected: `8 passed`, then `All checks passed!` and `2 files already formatted`.

- [ ] **Step 5: Commit**

```bash
git add sonde/sweep.py tests/test_sweep.py
git commit -m "feat(sweep): train step with selection, test-once, control, text baselines and headline"
```

---

### Task 3.4: Quickstart recipe, bundled data, recipe-relative paths

**Files:**
- Create: `sonde/recipes/quickstart.yaml`, `sonde/recipes/data/quickstart.jsonl`, `sonde/recipes/data/quickstart_heldout.jsonl`
- Modify: `sonde/config.py` (add `_resolve_bundled` and replace the body of `load`)
- Test: `tests/test_e2e.py` (created here; later tasks append to it)

**Interfaces:**
- Consumes:
  - `config.load(source: str, overrides: Sequence[str] = ()) -> RunConfig`;
  - `config.RECIPES_DIR: pathlib.Path`;
  - `config.apply_override(raw: dict, item: str) -> None`;
  - phase 1's package-data (`recipes/*.yaml`, `recipes/data/*`).
- Produces:
  - The recipe `quickstart`: gpt2, `format: raw`, every block, `keep: pooled`, linear mean probe, one score entry `heldout`.
  - For a recipe loaded by name, `config.load` resolves a relative `data.path` or `score[*].path` against `RECIPES_DIR` when the file exists there. Explicit `.yaml` paths and `-o` overrides keep the spec's cwd-relative rule.
  - `tests/test_e2e.py` defines `QUICKSTART = config.RECIPES_DIR / "data" / "quickstart.jsonl"`, used by Tasks 3.5 and 3.6.

- [ ] **Step 1: Write the failing test**

`tests/test_e2e.py`:

```python
from __future__ import annotations

import pathlib

from sonde import config

QUICKSTART = config.RECIPES_DIR / "data" / "quickstart.jsonl"


def test_quickstart_data_resolves_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config.load("quickstart")
    paths = [cfg.data.path if cfg.data else None, cfg.score[0].path]
    assert all(p is not None and pathlib.Path(p).is_file() for p in paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: FAIL. The `ValueError` from `config.load` says `quickstart` is an unknown recipe and lists the available ones.

- [ ] **Step 3: Generate the bundled data (deterministic)**

Run from the repo root:

```bash
uv run python - <<'EOF'
import json
import pathlib

THINGS = ["movie", "book", "song", "meal", "game", "phone", "car", "chair",
          "show", "hotel", "camera", "jacket", "laptop", "garden", "coffee",
          "bike", "album", "city", "beach", "train", "lamp", "watch", "pizza",
          "park", "museum", "sofa", "tent", "guitar", "printer", "bakery",
          "podcast", "puzzle"]
OUT = pathlib.Path("sonde/recipes/data")
OUT.mkdir(parents=True, exist_ok=True)


def write(name, things, pos, neg):
    rows = []
    for thing in things:
        for text, label in ((pos, 1), (neg, 0)):
            rows.append({"id": f"{name}-{len(rows)}",
                         "text": text.format(thing), "label": label})
    (OUT / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))


write("quickstart", THINGS, "I love this {}, it is wonderful.",
      "I hate this {}, it is awful.")
write("quickstart_heldout", THINGS[:16], "What a great {} this is!",
      "What a terrible {} this is!")
EOF
wc -l sonde/recipes/data/*.jsonl
md5sum sonde/recipes/data/*.jsonl   # macOS: md5 -q sonde/recipes/data/*.jsonl
```

Expected:
- `wc -l` reports `64 sonde/recipes/data/quickstart.jsonl` and `32 sonde/recipes/data/quickstart_heldout.jsonl`.
- The md5 sums are `17bc67ab512cd33146bf4f488b1b6a46` and `6416a65257f4882c84839601a6b37052`.
- The first line is `{"id": "quickstart-0", "text": "I love this movie, it is wonderful.", "label": 1}`.
- All 64 training sentences are distinct, so no sentence can sit in two splits.
- The script is not kept in the repo; the two `.jsonl` files are committed.

- [ ] **Step 4: Write the recipe**

`sonde/recipes/quickstart.yaml`:

```yaml
# CPU smoke run: gpt2 learns positive vs negative sentiment in seconds.
# sonde run quickstart [-o output.dir=/tmp/runs]
name: quickstart
seed: 0
model: {name: gpt2, dtype: float32, backend: hf}
data:
  path: data/quickstart.jsonl
  format: raw
  split: [0.5, 0.25, 0.25]
extract: {layers: all, window: prompt, keep: pooled, batch_size: 16}
probe: {kind: linear, pooling: mean, lr: 1.0e-3, epochs: 30, batch_size: 16}
score:
  - {name: heldout, path: data/quickstart_heldout.jsonl}
steps: [extract, train, score]
```

Run: `uv run pytest tests/test_e2e.py -v`
Expected: FAIL on the `assert all(...)` line, because `data/quickstart.jsonl` is still resolved against `tmp_path`.

- [ ] **Step 5: Resolve bundled paths in `config.load`**

1. In `sonde/config.py`, add this function directly below `RECIPES_DIR`:

```python
def _resolve_bundled(raw: dict) -> None:
    """Points a bundled recipe's relative data paths at RECIPES_DIR.

    Only for recipes loaded by name, and only when the file exists there,
    so `sonde run quickstart` works from any cwd while user configs keep
    the cwd-relative rule.
    """
    for src in [raw.get("data") or {}, *(raw.get("score") or [])]:
        path = src.get("path")
        if path and (RECIPES_DIR / path).is_file():
            src["path"] = str(RECIPES_DIR / path)
```

2. In `config.load`, keep the signature and docstring and replace everything after the docstring with:

```python
    path = pathlib.Path(source)
    bundled = path.suffix not in (".yaml", ".yml")
    if bundled:
        path = RECIPES_DIR / f"{source}.yaml"
        if not path.is_file():
            names = sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))
            raise ValueError(f"unknown recipe {source!r}; available: {names}")
    raw = yaml.safe_load(path.read_text()) or {}
    if bundled:
        _resolve_bundled(raw)
    for item in overrides:
        apply_override(raw, item)
    return RunConfig.model_validate(raw)
```

`_resolve_bundled` runs before the overrides, so `-o data.path=...` still wins and stays cwd-relative. An explicit `.yaml` path never goes through it.

- [ ] **Step 6: Check that the wheel ships the recipes**

Phase 1's package-data (`"recipes/*.yaml", "recipes/data/*"`) already covers the new files, so `pyproject.toml` does not change.

Run: `uv build --wheel -o /tmp/sonde-whl && unzip -l /tmp/sonde-whl/sonde-*.whl | grep recipes/`
Expected: the listing includes `sonde/recipes/quickstart.yaml`, `sonde/recipes/data/quickstart.jsonl` and `sonde/recipes/data/quickstart_heldout.jsonl`.

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_e2e.py tests/test_config.py -v`
Expected: `test_quickstart_data_resolves_from_any_cwd PASSED`, and phase 2's `test_config.py` still passes (unknown-recipe message and override behaviour are unchanged).

- [ ] **Step 8: Commit**

```bash
git add sonde/recipes/quickstart.yaml sonde/recipes/data/quickstart.jsonl \
  sonde/recipes/data/quickstart_heldout.jsonl sonde/config.py \
  tests/test_e2e.py
git commit -m "feat(recipes): bundled gpt2 quickstart with recipe-relative data paths"
```

---

### Task 3.5: The score step with refusal on mismatched probes

**Files:**
- Create: `sonde/score.py`
- Modify: `tests/test_e2e.py` (replace its import block and append tests)

**Interfaces:**
- Consumes:
  - `backends.load_model(cfg: ModelConfig) -> Loaded`, using `.tokenizer` and `.engine`; its cache is shared with extract.
  - `data.prompt_format_of(src, tokenizer) -> str`, `data.load_samples(src, seed) -> list[Sample]` and `data.encode_all(samples, src, window, tokenizer) -> (list[Encoded], dict[str, int])`.
  - `config.score_data(cfg, entry) -> DataConfig`, `config.ExtractConfig()` and `config.ScoreEntry(name, probe=None, ...)`.
  - `store.config_hash(sections) -> str`, `store.prepare(out_dir, hash, overwrite) -> "fresh" | "resume" | "done"`, `store.read_manifest` and `store.read_layer`.
  - `extract.extract_to(loaded, encoded, blocks, keep, pooling, out_dir, batch_size, shard_size, shard_bytes, meta)`.
  - `probe.Probe.load` and `.backend()`.
  - `fingerprint.checkpoint_fingerprint`.
  - `train.probe_scores` and `train.metrics(..., groups)` (Task 3.2).
- Produces `run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None`:
  - For each entry it writes `run_dir/"score"/entry.name/{extract/, scores.jsonl, metrics.json}`.
  - `scores.jsonl` rows are `{"id", "score", "flag"}`.
  - `metrics.json` is written only when every sample has a label. It reports the deployed operating point: `fpr` and `recall` at the probe's **frozen** `threshold`, which is what mechanica sees. It also reports `recall_at_fpr` re-thresholded on this set, `auroc`, and `group_auroc` when the entry's data has `group`.
    - `max_fpr` comes from `cfg.probe`, else from the probe's own card (`p.metrics["max_fpr"]`), so a `steps: [score]` run without a probe section still reports recall at the budget the probe was trained for.
    - A benign-only set reports `fpr` only, and a positive-only set `recall` only (`train.metrics` sets the rest to `None`). Neither raises. Task 3.6 adds the benign-only test.
  - The probe path defaults to `run_dir/"probes"/f"{cfg.name}.npz"`.
  - Features are extracted at the probe's block, window and pooling: `keep` is `pooled` for mean or last, else `tokens`.
  - A rerun deletes and re-extracts `score/<entry>/extract/` (`store.prepare(..., overwrite=True)`). Spec §11 says score overwrites its own outputs.
  - It raises `ValueError` naming `backend` or `model_fingerprint` before any model loads, and naming `prompt_format` after the tokenizer is available.

- [ ] **Step 1: Write the failing tests (review focus)**

1. Replace the import block of `tests/test_e2e.py` with:

```python
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from sonde import config
from sonde import fingerprint
from sonde import probe
from sonde import score
```

2. Append:

```python
def _planted(tmp_path, **override):
    good = probe.Probe(
        name="planted",
        kind="linear",
        w=np.zeros(768, np.float32),
        bias=0.0,
        block=3,
        window="prompt",
        pooling="mean",
        threshold=0.5,
        model="gpt2",
        engine="hf==5.1.0+nnterp==1.3.0",
        model_fingerprint=fingerprint.checkpoint_fingerprint("gpt2"),
        prompt_format=fingerprint.RAW,
    )
    path = dataclasses.replace(good, **override).save(
        str(tmp_path / "planted.npz")
    )
    return config.RunConfig(
        name="refuse",
        model=config.ModelConfig(name="gpt2", dtype="float32"),
        data=config.DataConfig(path=str(QUICKSTART), format="raw"),
        score=[config.ScoreEntry(name="s", probe=path)],
        output=config.OutputConfig(dir=str(tmp_path)),
        steps=["score"],
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("prompt_format", f"tmpl:{'0' * 32}", "prompt_format"),
        ("model_fingerprint", f"hub:{'0' * 32}", "model_fingerprint"),
        ("engine", "vllm==0.30.0+nnsight==b717807", "backend"),
    ],
)
def test_score_refuses_mismatched_probe(tmp_path, field, value, match):
    cfg = _planted(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=match):
        score.run(cfg, tmp_path / "refuse")
    assert not (tmp_path / "refuse" / "score" / "s" / "scores.jsonl").exists()


def test_score_accepts_matching_probe(tmp_path):
    score.run(_planted(tmp_path), tmp_path / "refuse")
    rows = (tmp_path / "refuse" / "score" / "s" / "scores.jsonl").read_text()
    assert len(rows.splitlines()) == 64
    assert json.loads(rows.splitlines()[0]) == {
        "id": "quickstart-0",
        "score": 0.5,
        "flag": True,
    }
```

The all-zero `w` makes every score exactly `sigmoid(0) = 0.5`, so the accepted path checks the row format exactly. The refusal cases use a valid gpt2 run and change one field of an otherwise matching probe.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: FAIL during collection, with `ImportError: cannot import name 'score' from 'sonde'`.

- [ ] **Step 3: Write minimal implementation**

`sonde/score.py`:

```python
"""The score step: apply a probe to new data."""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np

from sonde import backends
from sonde import config
from sonde import data
from sonde import extract
from sonde import fingerprint
from sonde import probe
from sonde import store
from sonde import train

_log = logging.getLogger(__name__)


def _mismatch(path: str, field: str, probe_value, run_value) -> ValueError:
    return ValueError(
        f"probe {path} has {field}={probe_value!r} but this run has "
        f"{run_value!r}; its scores would be meaningless here, so score "
        "refuses it"
    )


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Writes score/<entry>/{extract/, scores.jsonl, metrics.json}.

    metrics.json reports the deployed operating point: fpr and recall at
    the probe's frozen threshold. It also reports recall at max_fpr
    re-thresholded on this set (from the probe section, else the probe's
    own card), auroc, and group_auroc when the entry has a group column.
    A benign-only set reports fpr only, a positive-only set recall only.

    Raises:
        ValueError: the probe's backend, model_fingerprint or prompt_format
            disagrees with this run.
    """
    if cfg.model is None:
        raise ValueError("score needs the model section")
    ex = cfg.extract or config.ExtractConfig()
    fp = fingerprint.checkpoint_fingerprint(
        cfg.model.name, revision=cfg.model.revision, quantization=None
    )
    for entry in cfg.score:
        path = entry.probe or str(run_dir / "probes" / f"{cfg.name}.npz")
        p = probe.Probe.load(path)
        if p.backend() != cfg.model.backend:
            raise _mismatch(path, "backend", p.backend(), cfg.model.backend)
        if p.model_fingerprint != fp:
            raise _mismatch(path, "model_fingerprint", p.model_fingerprint, fp)
        src = config.score_data(cfg, entry)
        loaded = backends.load_model(cfg.model)
        fmt = data.prompt_format_of(src, loaded.tokenizer)
        if p.prompt_format != fmt:
            raise _mismatch(path, "prompt_format", p.prompt_format, fmt)
        keep = "pooled" if p.pooling in ("mean", "last") else "tokens"
        pooling = p.pooling if keep == "pooled" else None
        out = run_dir / "score" / entry.name
        ext = out / "extract"
        h = store.config_hash(
            {
                "model": cfg.model.model_dump(mode="json"),
                "data": src.model_dump(mode="json"),
                "probe": [p.block, p.window, keep, pooling],
            }
        )
        store.prepare(ext, h, overwrite=True)
        encoded, drops = data.encode_all(
            data.load_samples(src, cfg.seed),
            src,
            p.window,
            loaded.tokenizer,
        )
        extract.extract_to(
            loaded,
            encoded,
            [p.block],
            keep,
            pooling,
            ext,
            ex.batch_size,
            ex.shard_size,
            ex.shard_bytes,
            {
                "model": cfg.model.name,
                "revision": cfg.model.revision,
                "model_fingerprint": fp,
                "prompt_format": fmt,
                "engine": loaded.engine,
                "blocks": [p.block],
                "window": p.window,
                "keep": keep,
                "pooling": pooling,
                "dtype": cfg.model.dtype,
                "drops": drops,
                "config_hash": h,
            },
        )
        man = store.read_manifest(ext)
        X, offsets = store.read_layer(ext, p.block)
        scores = train.probe_scores(p, X, offsets, range(len(man["ids"])))
        with open(out / "scores.jsonl", "w") as f:
            for i, s in zip(man["ids"], scores, strict=True):
                row = {
                    "id": i,
                    "score": float(s),
                    "flag": bool(s >= p.threshold),
                }
                f.write(f"{json.dumps(row)}\n")
        labels = man["labels"]
        if labels and None not in labels:
            card = p.metrics or {}
            m = train.metrics(
                np.asarray(labels),
                scores,
                p.threshold,
                cfg.probe.max_fpr if cfg.probe else card.get("max_fpr"),
                man["groups"] if src.group else None,
            )
            (out / "metrics.json").write_text(json.dumps(m, indent=2))
            _log.info("score %s: %s", entry.name, m)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_e2e.py -v && uv run ruff check sonde/score.py tests/test_e2e.py && uv run ruff format --check sonde/score.py tests/test_e2e.py`
Expected: `5 passed`, including `test_score_refuses_mismatched_probe[prompt_format-...]`, `[model_fingerprint-...]` and `[engine-...]`. Then `All checks passed!` and `2 files already formatted`.

Optional sanity check: replace `if p.prompt_format != fmt:` with `if False:` and rerun. The prompt_format case must fail with `DID NOT RAISE`. Revert afterwards.

- [ ] **Step 5: Commit**

```bash
git add sonde/score.py tests/test_e2e.py
git commit -m "feat(score): score step that refuses probes from another backend, checkpoint or prompt format"
```

---

### Task 3.6: Runner, CLI and the end-to-end quickstart

**Files:**
- Create: `sonde/runner.py`, `sonde/cli.py`
- Modify: `sonde/extract.py` (factor the disk estimate into `disk_bytes`)
- Modify: `tests/test_e2e.py` (replace its import block and append tests)

**Interfaces:**
- Consumes:
  - `extract.run(cfg, run_dir)` and `extract.preflight(cfg) -> str` (phase 2);
  - `sweep.run` (Task 3.3) and `score.run` (Task 3.5);
  - `config.load`, `config.run_dir(cfg) -> pathlib.Path` and `config.resolve_blocks(spec, num_layers) -> list[int]`;
  - `data.load_samples`, `data.encode_all` and `data.Encoded.span`;
  - `transformers.AutoTokenizer` / `AutoConfig` (tokenizer files and config.json only).
- Produces:
  - `extract.disk_bytes(encoded: list[data.Encoded], keep: str, hidden: int, n_blocks: int, dtype: str) -> int`. Both `extract.run`'s log line and the CLI's `--dry-run` use it.
  - `runner.STEPS: dict[str, Callable[[config.RunConfig, pathlib.Path], None]] = {"extract": extract.run, "train": sweep.run, "score": score.run}`. Phase 6 adds `generate` and `steer`.
  - `runner.run(cfg: config.RunConfig) -> pathlib.Path`:
    1. Refuse any step that is not in `STEPS`.
    2. Run `extract.preflight` if `extract` is in the steps.
    3. Create the run dir and write `config.yaml`.
    4. Run the steps in order.
    5. Return the run dir.
  - `cli.main(argv: list[str] | None = None) -> int`.
    - Usage: `sonde run CFG|RECIPE [-o k=v]... [--dry-run]`.
    - `--dry-run` prints the resolved YAML and `# extract disk estimate: X MiB` (when `extract` is in the steps), creates nothing and loads no weights.
    - The module ends with `if __name__ == "__main__": raise SystemExit(main())`. Phase 1 already set `[project.scripts] sonde = "sonde.cli:main"`.

- [ ] **Step 1: Write the failing tests**

1. Replace the import block of `tests/test_e2e.py` with:

```python
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from sonde import cli
from sonde import config
from sonde import fingerprint
from sonde import probe
from sonde import runner
from sonde import score
```

2. Insert these tests between the `QUICKSTART = ...` line and `test_quickstart_data_resolves_from_any_cwd`:

```python
def test_quickstart_extract_train_score(tmp_path):
    assert cli.main(["run", "quickstart", "-o", f"output.dir={tmp_path}"]) == 0
    run = tmp_path / "quickstart"
    assert (run / "config.yaml").is_file()
    assert list((run / "extract").glob("shard_*.safetensors"))
    p = probe.Probe.load(str(run / "probes" / "quickstart.npz"))
    assert p.prompt_format == fingerprint.RAW and p.backend() == "hf"
    m = json.loads((run / "metrics.json").read_text())
    assert len(m["val"]) == 12 and m["test"]["auroc"] > 0.9
    held = run / "score" / "heldout"
    assert len((held / "scores.jsonl").read_text().splitlines()) == 32
    hm = json.loads((held / "metrics.json").read_text())
    assert (hm["n_pos"], hm["n_neg"]) == (16, 16) and hm["auroc"] > 0.9


def test_dry_run_prints_config_and_creates_nothing(tmp_path, capsys):
    argv = ["run", "quickstart", "--dry-run", "-o", f"output.dir={tmp_path}"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "name: quickstart" in out and "disk estimate" in out
    assert not (tmp_path / "quickstart").exists()
```

3. Insert this test after `test_quickstart_data_resolves_from_any_cwd`. It removes a registered step, so it still holds once phase 6 registers `generate` and `steer`:

```python
def test_runner_refuses_unregistered_step(tmp_path, monkeypatch):
    monkeypatch.delitem(runner.STEPS, "score")
    cfg = config.RunConfig(
        name="x",
        model=config.ModelConfig(name="gpt2"),
        data=config.DataConfig(path=str(QUICKSTART)),
        score=[config.ScoreEntry(name="s")],
        output=config.OutputConfig(dir=str(tmp_path)),
        steps=["score"],
    )
    with pytest.raises(ValueError, match="score"):
        runner.run(cfg)
    assert not (tmp_path / "x").exists()
```

4. Append this test at the end of the file. It pins Task 3.5's deployed operating point on a benign-only set: every score is 0.5 and the frozen threshold is 0.5, so the realized FPR is 1.0. `recall`, `auroc` and `recall_at_fpr` are `None` rather than raising, and `max_fpr` comes from the probe's card because the run has no probe section. The refusal tests from Task 3.5 stay as they are.

```python
def test_score_benign_only_reports_fpr_at_frozen_threshold(tmp_path):
    rows = QUICKSTART.read_text().splitlines()
    benign = tmp_path / "benign.jsonl"
    benign.write_text("".join(f"{r}\n" for r in rows if '"label": 0' in r))
    cfg = _planted(tmp_path, metrics={"max_fpr": 0.01})
    probe_path = cfg.score[0].probe
    cfg.score = [
        config.ScoreEntry(name="s", probe=probe_path, path=str(benign))
    ]
    score.run(cfg, tmp_path / "refuse")
    out = tmp_path / "refuse" / "score" / "s" / "metrics.json"
    m = json.loads(out.read_text())
    assert (m["n_pos"], m["n_neg"], m["threshold"]) == (0, 32, 0.5)
    assert m["fpr"] == 1.0 and m["recall"] is None
    assert m["auroc"] is None and m["recall_at_fpr"] is None
    assert m["small_n"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: FAIL during collection, with an `ImportError` while importing the test module (`sonde.cli` / `sonde.runner` do not exist yet).

- [ ] **Step 3: Factor the disk estimate into `extract.disk_bytes`**

1. In `sonde/extract.py`, add this function directly above `def run(`:

```python
def disk_bytes(
    encoded: list[data.Encoded],
    keep: str,
    hidden: int,
    n_blocks: int,
    dtype: str,
) -> int:
    """Bytes extract writes: window tokens x H x blocks x dtype size.

    A pooled sample counts as one token.
    """
    rows = (
        len(encoded)
        if keep == "pooled"
        else sum(e.span[1] - e.span[0] for e in encoded)
    )
    return rows * hidden * n_blocks * getattr(torch, dtype).itemsize
```

2. In `extract.run`, replace these lines:

```python
    tokens = (
        sum(e.span[1] - e.span[0] for e in encoded)
        if keep == "tokens"
        else len(encoded)
    )
    size = tokens * loaded.hidden * len(blocks)
    size *= getattr(torch, model.dtype).itemsize
```

with:

```python
    size = disk_bytes(encoded, keep, loaded.hidden, len(blocks), model.dtype)
```

The `logger.info` call that follows them does not change.

Run: `uv run pytest tests/test_extract.py -q`
Expected: phase 2's extract tests still pass.

- [ ] **Step 4: Write the runner**

`sonde/runner.py`:

```python
"""Step registry and run order."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import yaml

from sonde import config
from sonde import extract
from sonde import score
from sonde import sweep

STEPS: dict[str, Callable[[config.RunConfig, pathlib.Path], None]] = {
    "extract": extract.run,
    "train": sweep.run,
    "score": score.run,
}


def run(cfg: config.RunConfig) -> pathlib.Path:
    """Runs cfg.steps in order.

    Validates, refuses a stale extraction before any model loads, writes
    the resolved config.yaml, then runs each step.

    Returns:
        The run directory.

    Raises:
        ValueError: a step is not available, or extract/ holds a different
            config (and output.overwrite is false).
    """
    missing = [s for s in cfg.steps if s not in STEPS]
    if missing:
        raise ValueError(f"steps not available yet: {missing}")
    if "extract" in cfg.steps:
        extract.preflight(cfg)
    out = config.run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    )
    for step in cfg.steps:
        STEPS[step](cfg, out)
    return out
```

- [ ] **Step 5: Write the CLI**

`sonde/cli.py`:

```python
"""sonde run CFG|RECIPE [-o key=value ...] [--dry-run]."""

from __future__ import annotations

import argparse
import logging

import transformers
import yaml

from sonde import config
from sonde import data
from sonde import extract
from sonde import runner


def _disk_estimate(cfg: config.RunConfig) -> int:
    """Bytes extract will write, via extract.disk_bytes.

    Reads the tokenizer and config.json only, never weights.
    """
    model, src, ex = cfg.model, cfg.data, cfg.extract
    if model is None or src is None or ex is None:
        raise ValueError("extract needs the model, data and extract sections")
    tok = transformers.AutoTokenizer.from_pretrained(
        model.name, revision=model.revision
    )
    hf = transformers.AutoConfig.from_pretrained(
        model.name, revision=model.revision
    )
    encoded, _ = data.encode_all(
        data.load_samples(src, cfg.seed), src, ex.window, tok
    )
    blocks = config.resolve_blocks(ex.layers, hf.num_hidden_layers)
    return extract.disk_bytes(
        encoded, ex.keep, hf.hidden_size, len(blocks), model.dtype
    )


def main(argv: list[str] | None = None) -> int:
    """Console entry point; returns the exit code."""
    parser = argparse.ArgumentParser(prog="sonde")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a config or bundled recipe")
    run.add_argument("config", help="path to a .yaml, or a recipe name")
    run.add_argument(
        "-o",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="e.g. -o probe.lr=0.01",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved config and disk estimate",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = config.load(args.config, args.overrides or ())
    if args.dry_run:
        print(
            yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), end=""
        )
        if "extract" in cfg.steps:
            print(
                f"# extract disk estimate: "
                f"{_disk_estimate(cfg) / 2**20:.1f} MiB"
            )
        return 0
    print(runner.run(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: `9 passed`. The whole file took about 5 s on an M-series CPU with gpt2 cached.

- [ ] **Step 7: Run the real command**

```bash
time uv run sonde run quickstart -o output.dir=/tmp/sonde-qs
uv run sonde run quickstart --dry-run | tail -5
ls /tmp/sonde-qs/quickstart /tmp/sonde-qs/quickstart/probes
```

Expected:
- The run's last line is `/tmp/sonde-qs/quickstart`, in well under 2 min. It was 4.6 s when verified.
- The INFO logs include `selected block 0` and `score heldout:` with `'auroc': 1.0`. The train step's last line is `headline {"max_fpr": 0.01, "recall_at_fpr": 1.0, ...}` with `baseline_recall_at_fpr`, `baseline_auroc` and `length_auroc`.
- WARNING lines `small_n: 8 negatives x max_fpr 0.01 < 10 ...` are expected: quickstart's splits are far too small for a 1% FPR, so `metrics.json` records `select_fallback: auroc`.
- The dry run ends with:

```
steps:
- extract
- train
- score
# extract disk estimate: 2.2 MiB
```

- The run dir holds `config.yaml extract layers metrics.json probes score splits.json`, and `probes/` holds only `quickstart.npz`.
- `control_auroc` in `metrics.json` varies by seed on only 16 val rows. That is expected and recorded, not raised.
- Running the first command a second time also exits 0. Extract is skipped as `done`, and `score/heldout/extract/` is rebuilt.

- [ ] **Step 8: Full phase gate**

Run: `uv run pytest tests/test_probes.py tests/test_train.py tests/test_sweep.py tests/test_e2e.py -v && uv run ruff check sonde tests && uv run ruff format --check sonde tests && uv run pyright`
Expected: `43 passed`, then `All checks passed!`, the files reported as already formatted, and `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 9: Commit**

```bash
git add sonde/runner.py sonde/cli.py sonde/extract.py tests/test_e2e.py
git commit -m "feat(cli): sonde run with dry-run, step runner, end-to-end gpt2 quickstart"
```

<!-- skipped:
- Issue 1 (reuse score/<entry>/extract/ through prepare(overwrite=False) with a fallback to overwrite=True): not applied. Issues 6 and 8 fix the same false claim in a different way, and that way was applied: re-extract every time. Spec §11 says score overwrites its own outputs. Reuse would add an except branch, plus a "resume" path into extract_to, for a step that only extracts one block.
- Issue 10's parts for Task 4.1 Steps 5/7/9, Task 4.4 Step 3 and Task 6.2 Step 4, and issues 3/5's Task 4.1 Step 7 part: these are in phases 4 and 6, not in this phase's markdown, so they are left to those phases. Only Task 3.4 Step 6 was applied here.
- Issues 2, 4 and 9 are duplicates (the full load body). Issues 3, 5 and the Task 3.4 part of 10 are duplicates (package-data). Each was applied once.
-->

---

## Phase 4: Recipes, repo cleanup, CI, README

**Goal.** Ship the two GPU recipes (`refusal`, `high_stakes`) next to `quickstart`. Delete the legacy experiments, activations and docs listed in spec §15. Correct the linear-probes primer. Rewrite README, CHANGELOG and CONTRIBUTING for `sonde run` and the two-venv setup. Replace CI with a single locked uv job. **Done when:** `git ls-files sonde` matches spec §4 (generate/steer arrive in phase 6); `tests/test_config.py` loads every shipped recipe with sockets blocked; `uv run --no-sync pytest -m "not gpu"`, ruff and pyright pass locally exactly as CI runs them; and the pushed phase branch's CI run is green.

**Facts verified while writing this phase (2026-10-02):**
- `experiments/refusal_probing/data/labeled.jsonl` has keys `original_prompt, refusal_reasoning, refusal_label, original_response, sample_id`. It has 256 rows: 35 `"LABEL: refusal"` and 221 `"LABEL: non-refusal"`. There are 200 unique prompts, and 83 rows sit in duplicated-prompt groups (this matches spec §15). 4 duplicated prompts carry conflicting labels. `sample_id` runs 0..255 and is unique; there is no `id` key.
- The responses came from `meta-llama/Llama-3.1-8B-Instruct` with greedy decoding and `max_new_tokens: 256` (`generate_responses.yaml`) as raw continuations. The first response starts `" here's an example…"`, which is a continuation and not a chat turn.
- HF datasets-server shows `Arrrlex/models-under-pressure` config `training` has `train`/`test` splits with columns `inputs, ids, labels` (`labels` ∈ {`high-stakes`, `low-stakes`}). All five `*_balanced` OOD configs have a `test` split with `inputs, ids, labels`, but their `inputs` are **JSON-encoded message lists stored as strings**. The old pipeline probed that string as raw text.
- The `meta-llama/Llama-3.1-8B` base tokenizer has `chat_template = None`. Checked on the ungated mirrors `unsloth/Meta-Llama-3.1-8B` and `NousResearch/Meta-Llama-3.1-8B`; the meta-llama repo is gated. So `format: chat` would raise in `data.render`, and `high_stakes` must say `format: raw`. This is also how the RESULTS.md numbers were produced (v0.1 had no template, and ran at `batch_size: 1`).
- The old OOD evaluation applied **no** sample limit (`experiment_runner.py:518-529`).
- Contract-shaped pydantic 2.13 stand-ins (scratch venv) validate both recipe files below. An explicit `limit: null` in a `ScoreEntry` lands in `model_fields_set`, so phase 2's `model_dump(exclude_unset=True)` merge in `config.score_data` yields `limit=None`. `Model.model_construct().model_dump()` prints the defaults.
- `uv 0.12.5`: `UV_PROJECT_ENVIRONMENT=<dir> uv sync --locked --extra X --python 3.12 --managed-python` creates and syncs that venv. `uv run` is inexact by default, and `--no-sync` skips syncing.
- `astral-sh/setup-uv@v6` has the `python-version` and `enable-cache` inputs; `actions/cache@v4` has `path` and `key`. Phase 1 pins ruff 0.16.10 and pyright 1.1.414. The `ruff-pre-commit` tag `v0.16.10` exists (`git ls-remote` → `f12be1eb…`), with hook ids `ruff-check`, `ruff-format` and the legacy alias `ruff`.
- ruff 0.16.10 formats Python blocks inside Markdown, so `ruff format --check .` covers README.md. A README block with column-aligned inline comments (`(acts)   # …`) is reported as `File would be reformatted`; two spaces before `#` passes. The kept Markdown files (`docs/linear-probes-primer.md`, `experiments/detecting_high_stakes/RESULTS.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `experiments/refusal_probing/README.md`) already pass at 80 columns. The spec under `docs/superpowers/` does not pass, and phase 1's `extend-exclude` excludes it.
- With only `[tool.pyright]` in pyproject, `pyright --verbose` prints `Loading pyproject.toml file at …/pyproject.toml`. A `pyrightconfig.json` takes precedence over pyproject, which is why the duplicate must go.
- `.gitignore` already ignores `runs/` (`git check-ignore -v runs/refusal/metrics.json` → `.gitignore:61:runs/`). It does **not** ignore `.venv-hf/`.
- Primer claims were checked against arXiv abstracts and HTML: 2310.06824 (LR, MM and CCS; PCA used only for visualisation; LLaMA-2 7B/13B/70B; MM most causal in 7/8), 2306.03341 (mass-mean shift 42.3% vs probe weight 34.8%), 2210.13382 ("nonlinear internal representation"), 2309.00941 (Nanda, Lee & Wattenberg; mine/theirs), 2310.02207 (R² plateaus about halfway), 1905.05950 (ordering only), 2305.01610 (7 models, 70M–6.9B), 2312.06681 (first author Nina Panickssery), 2306.03819 (LEACE guarantee), 2502.03407 (1% FPR on unrelated chat data) and 2109.09234 (Hewitt et al. conditional probing).

**Deliberate deviations from spec §15, each forced by a verified fact:**
1. `high_stakes` adds `format: raw`, because the base tokenizer has no chat template.
2. Both recipes spell out `extract:` and `probe:` sections, because RunConfig requires a section for every listed step.
3. `refusal` adds `id: sample_id`, because rows have no `id` key.
4. Each `high_stakes` score entry sets `limit: null`, so OOD sets are scored in full as in RESULTS.md instead of inheriting `limit: 1000`.
5. `labeled.jsonl` stays in place (spec: "Keep"), and its directory README gets the provenance note.
6. The dead v0.1 scripts in the two kept experiment dirs are deleted as well (`run_pipeline.py`, `run_steering.py`, `run_diff_means_steering.py`, `generate_responses.yaml`, and the old `detecting_high_stakes/config.yaml`). They import packages phase 1 deleted, and `ruff format --check .` at 80 columns would fail on them.

---

### Task 4.1: Port the refusal and high_stakes recipes

**Files:**
- Create: `sonde/recipes/refusal.yaml`
- Create: `sonde/recipes/high_stakes.yaml`
- Modify: `experiments/refusal_probing/README.md` (full rewrite: data provenance note)
- Modify: `tests/test_config.py` (append three tests)
- Possibly modify: `pyproject.toml` (package-data, Step 7), only if the stated check fails

**Interfaces:**
- Consumes (phases 2–3, contract):
  - `config.RECIPES_DIR: pathlib.Path`
  - `config.load(source: str, overrides: Sequence[str] = ()) -> RunConfig`
  - `config.score_data(cfg: RunConfig, entry: ScoreEntry) -> DataConfig`
  - `data.load_samples(src: DataConfig, seed: int) -> list[Sample]`, where `Sample` has `label: int | None` and `group: str | None`
- Produces: recipe names `"refusal"` and `"high_stakes"`, resolvable by `config.load(name)`. Their run names (and so their run dirs and probe names) are `refusal` and `high_stakes`. Phase 5 re-runs both on the Spark without any `-o name=` override.

- [ ] **Step 1: Write the failing tests.** Append these to `tests/test_config.py`. Merge `import pathlib`, `import socket` and `from sonde import data` into the file's import block (`from sonde import config` is already there from phase 2).

```python
def test_every_shipped_recipe_loads_offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("config.load opened a socket")

    monkeypatch.setattr(socket, "socket", no_network)
    names = sorted(p.stem for p in config.RECIPES_DIR.glob("*.yaml"))
    assert {"quickstart", "refusal", "high_stakes"} <= set(names)
    for name in names:
        assert config.load(name).name == name


def test_refusal_recipe_reads_its_labeled_data(monkeypatch):
    monkeypatch.chdir(pathlib.Path(__file__).parents[1])
    cfg = config.load("refusal")
    samples = data.load_samples(cfg.data, cfg.seed)
    assert len(samples) == 256
    assert sum(s.label for s in samples) == 35
    assert len({s.group for s in samples}) == 200


def test_high_stakes_scores_full_ood_test_splits():
    cfg = config.load("high_stakes")
    srcs = [config.score_data(cfg, e) for e in cfg.score]
    assert [s.hf_config for s in srcs] == [
        "anthropic_hh_balanced",
        "mt_balanced",
        "toolace_balanced",
        "mental_health_balanced",
        "aya_redteaming_balanced",
    ]
    assert all(s.hf_split == "test" and s.limit is None for s in srcs)
    assert all(s.hf == cfg.data.hf and s.text == "inputs" for s in srcs)
    assert all(s.format == "raw" for s in srcs)
```

`config.load(name).name == name` pins each recipe's `name:` to its file stem, so the run dir and probe name match the recipe a user typed. This assumes phase 2's `quickstart.yaml` says `name: quickstart`; if Step 5 fails only on quickstart, fix its `name:` line.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "recipe or high_stakes" -v`
Expected: 3 FAILED. The first fails on the `<= set(names)` assert; the other two fail with `ValueError` whose message lists the available recipes (only `quickstart`).

- [ ] **Step 3: Create `sonde/recipes/refusal.yaml`**

```yaml
# Refusal probe on Llama-3.1-8B-Instruct (data: experiments/refusal_probing).
# The labels judge raw continuations generated without a chat template, so
# format is raw and the probe only serves completion-style requests.
# group keeps the 83 rows that share a prompt inside one split.
# Relative data path: run from the repo root.
name: refusal
model:
  name: meta-llama/Llama-3.1-8B-Instruct
  backend: vllm
  vllm: {gpu_memory_utilization: 0.3, max_model_len: 4096}
data:
  path: experiments/refusal_probing/data/labeled.jsonl
  text: original_prompt
  label: refusal_label
  id: sample_id
  label_map: {"LABEL: refusal": 1, "LABEL: non-refusal": 0}
  group: original_prompt
  format: raw
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean}
steps: [extract, train]
```

- [ ] **Step 4: Create `sonde/recipes/high_stakes.yaml`**

```yaml
# Replicates arXiv:2506.10805 on Llama-3.1-8B (base). The base tokenizer has
# no chat template, so format is raw. OOD `inputs` are JSON-encoded message
# lists and are probed as raw text, as in the v0.1 RESULTS.md numbers.
# limit: null scores each OOD test split in full instead of inheriting 1000.
name: high_stakes
model:
  name: meta-llama/Llama-3.1-8B
  backend: vllm
  vllm: {gpu_memory_utilization: 0.3, max_model_len: 4096}
data:
  hf: Arrrlex/models-under-pressure
  hf_config: training
  text: inputs
  label: labels
  id: ids
  label_map: {high-stakes: 1, low-stakes: 0}
  limit: 1000
  format: raw
extract: {layers: [10, 13, 15, 17, 20], keep: tokens}
probe: {kind: attention, pooling: attention, weight_decay: 0.1, patience: 5}
score:
  - name: anthropic_hh_balanced
    hf_config: anthropic_hh_balanced
    hf_split: test
    limit: null
  - name: mt_balanced
    hf_config: mt_balanced
    hf_split: test
    limit: null
  - name: toolace_balanced
    hf_config: toolace_balanced
    hf_split: test
    limit: null
  - name: mental_health_balanced
    hf_config: mental_health_balanced
    hf_split: test
    limit: null
  - name: aya_redteaming_balanced
    hf_config: aya_redteaming_balanced
    hf_split: test
    limit: null
steps: [extract, train, score]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_config.py -k "recipe or high_stakes" -v`
Expected: 3 PASSED.

- [ ] **Step 6: Rewrite `experiments/refusal_probing/README.md`**

````markdown
# Refusal probing data

`data/labeled.jsonl` is the labeled set behind the `refusal` recipe
(`sonde run refusal`, from the repo root).

| | |
|---|---|
| Rows | 256 (35 `LABEL: refusal`, 221 `LABEL: non-refusal`) |
| Unique prompts | 200; 83 rows share a prompt with another row, and 4 of those prompts carry conflicting labels |
| Keys | `original_prompt`, `original_response`, `refusal_label`, `refusal_reasoning`, `sample_id` |

## How it was made

1. `1_generate_prompts.yaml` (oumi synth) wrote `data/prompts.jsonl`, a mix
   of harmful and harmless prompts.
2. sonde v0.1 (tag `v0.1-legacy`) generated `data/responses.jsonl` from
   `meta-llama/Llama-3.1-8B-Instruct`, greedy, 256 new tokens.
   **No chat template was applied**: each response is a raw continuation of
   the prompt text.
3. `2_label_responses.yaml` (oumi synth, LLM judge) wrote `data/labeled.jsonl`.

## Consequences for probing

- The recipe uses `format: raw`, because the labels describe raw
  continuations. The probe's `prompt_format` is therefore `raw`, and
  mechanica will refuse to score templated chat traffic with it.
- `group: original_prompt` keeps duplicated prompts inside one split.
- `window: prompt` reads the model before it answers. A probe there measures
  what predicts refusal (mostly "is this prompt harmful?"), not the refusal
  itself; see `docs/linear-probes-primer.md` §4.3.
````

- [ ] **Step 7: Check that the recipes ship in the wheel**

Run: `uv build --wheel -o dist && unzip -l dist/sonde-0.2.0-py3-none-any.whl | grep 'sonde/recipes/'`
Expected: lines for `sonde/recipes/quickstart.yaml`, `sonde/recipes/refusal.yaml` and `sonde/recipes/high_stakes.yaml` (plus quickstart's bundled data).
If the two new files are missing and pyproject uses setuptools, set this table and rerun:

```toml
[tool.setuptools.package-data]
sonde = ["py.typed", "recipes/*.yaml", "recipes/data/*"]
```

- [ ] **Step 8: Run the whole config test file**

Run: `uv run pytest tests/test_config.py -q`
Expected: all passed, 0 failed.

- [ ] **Step 9: Commit**

```bash
git add sonde/recipes/refusal.yaml sonde/recipes/high_stakes.yaml \
  experiments/refusal_probing/README.md tests/test_config.py
git add -u pyproject.toml
git commit -m "feat(recipes): port refusal and high_stakes recipes"
```

---

### Task 4.2: Delete legacy experiments, activations and docs

**Files:**
- Delete:
  - `experiments/geometry_of_truth/`, `experiments/rjudge_dissociation/`, `experiments/scaleJSD_probing/`, `experiments/refusal_probe_qwen.py`, `examples/`
  - `experiments/detecting_high_stakes/{k8s_job.yaml,config.yaml,run_pipeline.py}`
  - `experiments/refusal_probing/{run_pipeline.py,run_steering.py,run_diff_means_steering.py,generate_responses.yaml}`
  - `experiments/refusal_probing/data/{activations.safetensors,activations_manifest.pt}`
  - `docs/{toolkit_audit.md,intervention_design.md,oumi_integration_plan.md,research_gaps_and_extensions.md}`
- Modify: `experiments/detecting_high_stakes/RESULTS.md` ("How to Run" section and the References line)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: recipe `high_stakes` (Task 4.1).
- Produces: a tree with no references to the deleted paths outside `CHANGELOG.md` history and `docs/superpowers/`.

- [ ] **Step 1: Remove the files from git**

```bash
git rm -r -q experiments/geometry_of_truth experiments/rjudge_dissociation \
  experiments/scaleJSD_probing examples
git rm -q experiments/refusal_probe_qwen.py \
  experiments/detecting_high_stakes/k8s_job.yaml \
  experiments/detecting_high_stakes/config.yaml \
  experiments/detecting_high_stakes/run_pipeline.py \
  experiments/refusal_probing/run_pipeline.py \
  experiments/refusal_probing/run_steering.py \
  experiments/refusal_probing/run_diff_means_steering.py \
  experiments/refusal_probing/generate_responses.yaml \
  experiments/refusal_probing/data/activations.safetensors \
  experiments/refusal_probing/data/activations_manifest.pt
git rm -q docs/toolkit_audit.md docs/intervention_design.md \
  docs/oumi_integration_plan.md docs/research_gaps_and_extensions.md
```

Gitignored leftovers such as `experiments/rjudge_dissociation/data/` stay on disk untouched. Do not `rm -rf` them; they may hold run outputs.

- [ ] **Step 2: Verify what remains**

Run: `git ls-files experiments examples docs | grep -v '^docs/superpowers/'`
Expected (exactly):
```
docs/img/hero.svg
docs/linear-probes-primer.md
experiments/detecting_high_stakes/RESULTS.md
experiments/refusal_probing/1_generate_prompts.yaml
experiments/refusal_probing/2_label_responses.yaml
experiments/refusal_probing/README.md
experiments/refusal_probing/data/labeled.jsonl
experiments/refusal_probing/data/prompts.jsonl
experiments/refusal_probing/data/responses.jsonl
```

- [ ] **Step 3: Fix RESULTS.md "How to Run".** In `experiments/detecting_high_stakes/RESULTS.md`, replace everything from the line `## How to Run` up to (not including) `## Key Takeaways` with:

````markdown
## How to Run

```bash
sonde run high_stakes                      # .venv-vllm on the Spark
sonde run high_stakes -o model.backend=hf  # .venv-hf
```

The recipe (`sonde/recipes/high_stakes.yaml`):
1. Loads a seeded 1,000-row subsample of `Arrrlex/models-under-pressure` (`training`).
2. Extracts every token of the raw (untemplated) input at blocks 10, 13, 15, 17 and 20 of Llama-3.1-8B.
3. Fits one attention probe per block and selects by validation recall at 1% FPR (validation AUROC when val has too few negatives for 1% FPR).
4. Records a shuffled-label control.
5. Scores the selected probe on the five `*_balanced` OOD test splits.
6. Writes `runs/high_stakes/probes/high_stakes.npz` plus `metrics.json` and `score/*/metrics.json`.

The numbers above were produced by sonde v0.1 (tag `v0.1-legacy`) at batch
size 1. They have not yet been reproduced with the current pipeline.

````

Then replace the References line
`- Research gaps and extensions: [`docs/research_gaps_and_extensions.md`](../../docs/research_gaps_and_extensions.md)`
with
`- Open extensions: [spec §17](../../docs/superpowers/specs/2026-10-02-sonde-redesign-design.md#17-deferred-add-when-a-config-needs-it)`.

- [ ] **Step 4: Update `.gitignore`.** Delete the two lines

```
# R-Judge dissociation experiment outputs
experiments/rjudge_dissociation/data/
```

and add `.venv-*/` directly under the existing `.venv/` line:

```
.venv/
.venv-*/
venv/
```

Run: `git check-ignore -v runs/x/metrics.json .venv-hf/bin/python .venv-vllm/bin/python`
Expected: three lines, the first matched by `runs/` and the other two by `.venv-*/`.

- [ ] **Step 5: Verify that lint and tests still pass**

Run: `uv run ruff check . && uv run ruff format --check sonde tests && uv run pytest -m "not gpu" -q`
Expected: `All checks passed!`, `N files already formatted`, then pytest reports passed with 0 failed. Formatting is checked only on `sonde` and `tests` here, because ruff also formats Python blocks in Markdown and the old README.md is rewritten in Task 4.3. The full-tree `ruff format --check .` runs in Task 4.3 Step 3.

Run: `git grep -n -e research_gaps -e toolkit_audit -e intervention_design -e oumi_integration -e causal_loop -e k8s_job -e rjudge -e scaleJSD -e geometry_of_truth -e refusal_probe_qwen -- ':!docs/superpowers' ':!CHANGELOG.md' ':!README.md'`
Expected: no output. README.md is rewritten in Task 4.3.

- [ ] **Step 6: Commit**

```bash
git add -A experiments examples docs .gitignore
git commit -m "chore: delete legacy experiments, activations and docs (spec §15)"
```

---

### Task 4.3: Primer accuracy pass, README, CHANGELOG, CONTRIBUTING

**Files:**
- Modify: `docs/linear-probes-primer.md` (23 targeted edits)
- Modify: `README.md` (full rewrite)
- Modify: `CHANGELOG.md` (new 0.2.0 entry)
- Modify: `CONTRIBUTING.md` (full rewrite)

**Interfaces:**
- Consumes (documented, not called):
  - `sonde.run`, `sonde.load_config` (lazy `__getattr__`)
  - `probe.Probe.load`, `probe.load_dir`, `Probe.pooled_score`, `Probe.flag`, `Probe.serves`, `Probe.vllm_aux_layer`
  - `fingerprint.checkpoint_fingerprint`, `fingerprint.RAW`
  - the `RunConfig` fields and defaults from the contract
  - phase 1's ruff `extend-exclude = ["sonde/fingerprint.py", "docs/superpowers"]`, so the design docs' illustrative code is not format-checked
- Produces: user docs. Phase 5 must check the README's vllm lines against the Spark run. Phase 6 adds `generate` and `steer` rows to the README tables and the CHANGELOG.

- [ ] **Step 1: Apply the primer corrections.** Run from the repo root. The script refuses to write if any target string does not match exactly once. The corrections are:
  - Qwen 0.5B width: "≈" becomes "=", and the model is named Qwen2.5.
  - LR calibration: calibrated only without class weighting, and sonde uses `pos_weight`.
  - Fisher/LDA: optimal only for Gaussian classes.
  - V-information: cite Hewitt et al. 2021 (arXiv:2109.09234).
  - Tenney: drop the invented layer numbers and the unsupported decoder-LLM claim.
  - Marks & Tegmark: the methods were LR, MM and CCS (PCA only for visualisation), on LLaMA-2 7B/13B/70B; MM was most causal in 7/8.
  - ITI: it steers along the mass-mean shift, not the probe direction.
  - Gurnee 2023: Pythia only, not OPT.
  - §6.2: DoM is not "near-optimal" or "unbiased"; rewrite to state what it is.
  - CAA: the first author is Panickssery (formerly Rimsky), one person.
  - LEACE: state its actual guarantee.
  - Apollo: the 1% FPR is set on unrelated control chat.
  - Kadavath: P(True) and P(IK) are not frozen-activation probes, and the "RAG routing" claim is unsourced.
  - Delete three unsourced claims: hallucination AUC > 0.85, Arditi AUROC > 0.99, and the "training-run canaries" bullet.
  - OthelloGPT: Li et al. used nonlinear probes, and the linear mine/theirs result is Nanda, Lee & Wattenberg 2023.
  - Gurnee & Tegmark: R² plateaus around the middle layers instead of rising monotonically.
  - Add the two new references.

```bash
uv run python - <<'PYEOF'
import pathlib

path = pathlib.Path("docs/linear-probes-primer.md")
text = path.read_text()
edits = [
    (
        r"For Qwen-0.5B, $d \approx 896$;",
        r"For Qwen2.5-0.5B, $d = 896$;",
    ),
    (
        "| default in interp; calibrated probabilities |",
        "| default in interp; calibrated only on the training distribution"
        " and only without class weighting (sonde trains with `pos_weight`,"
        " so read scores against the val-chosen threshold) |",
    ),
    (
        r"| Bayes-optimal under shared $\Sigma$ |",
        r"| Bayes-optimal for Gaussian classes with shared $\Sigma$ |",
    ),
    (
        "and V-information (Hewitt et al. 2021) account",
        r"and conditional $\mathcal{V}$-information ([Hewitt et al. 2021]"
        "(https://arxiv.org/abs/2109.09234)) account",
    ),
    (
        'pipeline": POS in layers ~3-4, parsing/NER in ~5-8, semantic roles'
        " in ~9-11, coreference at the top. Modern decoder-only LLMs show the"
        " same depth profile for semantic features.",
        'pipeline": in BERT-large (24 layers) the regions that matter for each'
        " task appear in the order POS tagging, parsing, NER, semantic roles,"
        " then coreference. The ordering is the finding; the regions overlap,"
        " and the paper makes no claim about decoder-only LLMs.",
    ),
    (
        "compare logistic regression, mass-mean (DoM), and PCA probes on"
        " Llama-2 residual streams;",
        "compare logistic regression, mass-mean (DoM) and CCS probes on"
        " LLaMA-2-7B/13B/70B residual streams (PCA is used only to"
        " visualise);",
    ),
    (
        "DoM generalizes as well as or better than LR across topics.",
        "Mass-mean probes generalise about as well as LR and CCS, and their"
        " directions are the most causally implicated (MM beats LR and CCS"
        " in 7 of 8 intervention settings).",
    ),
    (
        r"shift top-$K$ heads along the probe direction by a coefficient"
        r" $\alpha$.",
        r"shift the top-$K$ heads by $\alpha$ along the mass-mean shift (the"
        " class-mean difference), which beat the probe-weight direction in"
        " their ablation (42.3% vs 34.8% true*informative).",
    ),
    ("in Pythia/OPT middle layers", "in Pythia middle layers"),
    (
        "### 6.2 Difference-of-Means Is Near-Optimal",
        "### 6.2 Why Difference-of-Means Steers Well",
    ),
    (
        r"Under shared class covariance $\Sigma$, the Bayes-optimal linear"
        r" classifier is Fisher's direction $w^* = \Sigma^{-1}(\mu_+ -"
        r" \mu_-)$. Logistic regression approximates this *but* is pulled"
        " toward any axis that gives low-noise separability — including axes"
        r" merely *correlated* with the feature. The raw DoM $\mu_+ - \mu_-$"
        " is unbiased for the *causal* feature direction when the label is"
        " generated by the feature. This is why [Marks & Tegmark]"
        "(https://arxiv.org/abs/2310.06824) argue DoM generalizes and steers"
        " better than LR despite being a weaker classifier.",
        r"For Gaussian classes with shared covariance $\Sigma$, the"
        r" Bayes-optimal linear classifier is Fisher's direction"
        r" $w^* = \Sigma^{-1}(\mu_+ - \mu_-)$. DoM equals it only when"
        r" $\Sigma \propto I$, so DoM is usually the weaker classifier."
        " Logistic regression is pulled toward any axis that gives low-noise"
        " separability, including axes merely *correlated* with the feature."
        " DoM ignores $\\Sigma$ and points where the class means actually"
        " differ, which is what an intervention moves. Empirically, [Marks &"
        " Tegmark](https://arxiv.org/abs/2310.06824) find mass-mean"
        " directions more causally implicated than LR directions at similar"
        " accuracy, and [ITI](https://arxiv.org/abs/2306.03341) steers best"
        " along the mass-mean shift.",
    ),
    (
        "[**CAA** (Panickssery, Rimsky et al. 2023)]",
        "[**CAA** (Panickssery et al. 2023)]",
    ),
    ("- Panickssery, Rimsky et al. (2023).", "- Panickssery et al. (2023)."),
    (
        "closed-form optimal linear concept erasure via mean-difference"
        " subspaces.",
        "closed-form erasure that provably stops every linear classifier"
        " from detecting a concept while changing the representation as"
        " little as possible.",
    ),
    (
        "Catch 95-99% of deceptive responses at 1% FPR",
        "Catch 95-99% of deceptive responses at a threshold set for 1% FPR"
        " on unrelated control chat data",
    ),
    (
        "P(True) and P(IK) probes; large models well-calibrated under correct"
        ' formatting. Underpins "I don\'t know" routing in RAG stacks.',
        "P(True) is the model's own probability that a proposed answer is"
        " true; P(IK) comes from a head fine-tuned with the model, so neither"
        " is a probe on frozen activations. Large models are well calibrated"
        " on multiple-choice and true/false questions in the right format.",
    ),
    (
        "- Real-time hallucination probes stream entity-level fabrication"
        " scores during generation at AUC > 0.85.\n",
        "",
    ),
    (
        " Refusal probes (Arditi et al.) separate harmful from harmless"
        " prompts at AUROC > 0.99 on Llama-2/3 chat.",
        "",
    ),
    (
        "\n**Concept-specific monitors.**\n- Power-seeking, scheming,"
        " self-preservation probes. Used as *training-run canaries*: watch a"
        " probe score during fine-tuning and halt when the concept's norm"
        ' rises (motivated by "emergent misalignment" work).\n',
        "",
    ),
    (
        "(ICLR 2023 oral)](https://arxiv.org/abs/2210.13382). OthelloGPT"
        ' probed for board state: reparameterized ("my color" vs "opponent'
        ' color") linear probes recover per-square state; intervening on'
        " probe-identified directions causally flips predicted legal moves.",
        "(ICLR 2023)](https://arxiv.org/abs/2210.13382) recovered OthelloGPT's"
        " board state with *nonlinear* (MLP) probes, and intervening on the"
        " probed representation flips predicted legal moves. [Nanda, Lee &"
        " Wattenberg (2023)](https://arxiv.org/abs/2309.00941) then showed"
        ' the state is *linear* once squares are read as "mine" vs "theirs"'
        " instead of black vs white.",
    ),
    (
        "with $R^2$ rising monotonically with depth and scale.",
        "with $R^2$ rising through the first half of the layers, plateauing"
        " near the middle, and higher in larger models.",
    ),
    (
        "[arXiv:2102.12452](https://arxiv.org/abs/2102.12452)\n",
        "[arXiv:2102.12452](https://arxiv.org/abs/2102.12452)\n"
        "- Hewitt et al. (2021). *Conditional probing: measuring usable"
        " information beyond a baseline*."
        " [arXiv:2109.09234](https://arxiv.org/abs/2109.09234)\n",
    ),
    (
        "[arXiv:2210.13382](https://arxiv.org/abs/2210.13382)\n",
        "[arXiv:2210.13382](https://arxiv.org/abs/2210.13382)\n"
        "- Nanda, Lee & Wattenberg (2023). *Emergent Linear Representations"
        " in World Models of Self-Supervised Sequence Models*."
        " [arXiv:2309.00941](https://arxiv.org/abs/2309.00941)\n",
    ),
]
for old, new in edits:
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one match for: {old[:60]!r}")
    text = text.replace(old, new)
path.write_text(text)
print(f"applied {len(edits)} edits")
PYEOF
```

Expected: `applied 23 edits`. (This script was dry-run on a copy of the current primer; all 23 matched.)

Run: `grep -n -e "Pythia/OPT" -e "Rimsky et al" -e "monotonically" -e "Near-Optimal" -e "AUC > 0.85" -e "AUROC > 0.99" -e "canaries" docs/linear-probes-primer.md`
Expected: no output.

- [ ] **Step 2: Replace `README.md` with exactly:**

````markdown
<p align="center">
  <img src="docs/img/hero.svg" alt="A probe descending through transformer layers" width="100%"/>
</p>

<h1 align="center">sonde</h1>

<p align="center">
  <a href="https://github.com/aniruddh-alt/sonde/actions/workflows/ci.yml"><img src="https://github.com/aniruddh-alt/sonde/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+"/>
</p>

sonde trains activation probes on language models. You write one YAML file,
run `sonde run cfg.yaml`, and get back a probe artifact with a measured eval
card. [mechanica](https://github.com/aniruddh-alt/mechanica) loads that
artifact to score live vLLM traffic.

A run goes through these stages:
1. Labeled rows are rendered with the chat template or as raw text.
2. Residual-stream activations are extracted at the blocks you choose.
3. One probe is fitted per block.
4. The block with the best validation recall at 1% FPR is selected (AUROC
   breaks ties, and replaces it when val has too few negatives).
5. That probe gets a threshold chosen on validation scores, test metrics
   with bootstrap CIs, a shuffled-label control, and a bag-of-words and
   length baseline to beat.

Model access goes through [nnsight](https://nnsight.net) and
[nnterp](https://github.com/Butanium/nnterp). sonde does not reimplement
hooks, model loading or the vLLM runtime.

## Install

sonde needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

**Mac, CI, or any single GPU (`hf` backend):**

```bash
uv sync --extra hf --extra dev
uv run sonde run quickstart
```

**DGX Spark: two environments.** The `hf` and `vllm` extras conflict. nnterp
1.3 needs nnsight < 0.8, and the `vllm` backend needs a pinned nnsight
commit, so each extra gets its own venv. Use a uv-managed Python, because
nnsight builds from source on aarch64 and needs `Python.h`.

```bash
UV_PROJECT_ENVIRONMENT=.venv-hf uv sync --locked --extra hf --python 3.12 --managed-python
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --locked --extra vllm --python 3.12 --managed-python

.venv-vllm/bin/sonde run refusal
.venv-hf/bin/sonde run refusal -o model.backend=hf
```

The Spark is shared. Always set `model.vllm.gpu_memory_utilization`, as the
shipped recipes do: vLLM's default of 0.9 can freeze the box.

To load and score artifacts you need only the base package, with no extras:
`import sonde.probe` and `import sonde.fingerprint` use only numpy and the
stdlib.

## Quickstart

```bash
sonde run quickstart --dry-run   # validate, print resolved YAML + disk estimate
sonde run quickstart
sonde run my.yaml -o probe.epochs=50 -o "extract.layers=[4, 8]"
```

A minimal config, reading rows like
`{"id": "1", "prompt": "...", "label": 1}`:

```yaml
name: my-probe
model: {name: Qwen/Qwen3-0.6B}
data: {path: data/rows.jsonl, text: prompt, label: label}
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean}
steps: [extract, train]
```

The same run from Python:

```python
import sonde
from sonde import probe

run_dir = sonde.run(sonde.load_config("quickstart"))
for name, p in probe.load_dir(str(run_dir / "probes")).items():
    print(name, p.block, p.pooling, round(p.threshold, 3))
```

A run writes `runs/<name>/`:

```
config.yaml              resolved config
extract/                 run.json, manifest.json, shard_00000.safetensors, ...
splits.json              train / val / test sample indices
probes/<name>.npz        the selected probe; the only file here
layers/B<block>.npz      every block's probe, named <name>@B<block>
metrics.json             headline, per-block val table, test, baseline, control, n counts
score/<entry>/           scores.jsonl, metrics.json (when labels exist)
```

Reruns behave as follows:
- `extract` resumes from its last complete shard. If the model, data or
  extract settings changed, it refuses before loading anything; set
  `output.overwrite: true` to start over.
- `train` and `score` overwrite their own outputs.

## Config reference

| Key | Default | Notes |
|---|---|---|
| `name` | required | run dir is `<output.dir>/<name>` |
| `seed` | `0` | drives subsampling, splits, init and data order |
| `steps` | `[extract, train]` | `extract`, `train`, `score`, run in the listed order |
| `model.name` | required | HF hub id or local path |
| `model.revision` | `null` | recorded in the fingerprint as given |
| `model.dtype` | `bfloat16` | activations are stored in this dtype |
| `model.backend` | `hf` | `hf` or `vllm`; never auto-detected |
| `model.vllm` | `{}` | engine kwargs, e.g. `gpu_memory_utilization`, `max_model_len` |
| `data.path` / `data.hf` | `null` | exactly one: local `.jsonl` / `.csv`, or an HF dataset id |
| `data.hf_config`, `data.hf_split` | `null`, `train` | HF config and split |
| `data.text` / `data.messages` | `text` / `null` | exactly one; setting messages alone switches off the text default |
| `data.response` | `null` | response column; required for `window: response` |
| `data.label`, `data.id`, `data.group` | `label`, `id`, `null` | column names; rows sharing a group stay in one split |
| `data.label_map` | `null` | raw value → 0/1; labels must end in {0, 1} |
| `data.limit` | `null` | seeded subsample size |
| `data.format` | `chat` | `chat` (tokenizer template) or `raw` |
| `data.system` | `null` | system prompt for `chat`; serving must send the same one |
| `data.max_length` | `2048` | keep the first N tokens |
| `data.split` | `[0.7, 0.15, 0.15]` | train / val / test, stratified |
| `extract.layers` | `all` | `all`, a list of block indices, or `{every: k}` |
| `extract.window` | `prompt` | `prompt`, `response`, `all`, `last_turn` |
| `extract.keep` | `pooled` | `pooled`: one vector per row; `tokens`: every window token |
| `extract.batch_size` | `16` | |
| `extract.shard_size`, `extract.shard_bytes` | `4096`, `2147483648` | a shard closes at whichever comes first |
| `probe.kind` | `linear` | `linear` or `attention` |
| `probe.pooling` | `mean` | linear: `mean`, `last`, `max`, `rolling_mean`; attention: `attention` |
| `probe.rolling_window` | `null` | required iff `rolling_mean` |
| `probe.init` | `random` | `diff_means` needs `kind: linear` and `epochs: 0` |
| `probe.epochs`, `probe.lr`, `probe.weight_decay`, `probe.batch_size` | `20`, `1e-3`, `0.0`, `256` | AdamW |
| `probe.patience` | `3` | early stopping on val loss; `0` turns it off |
| `probe.max_fpr` | `0.01` | set: lowest threshold with val FPR ≤ `max_fpr`; `null`: best F1 |
| `probe.select` | `recall_at_fpr` | `recall_at_fpr` needs `max_fpr`; `auroc`; `group_auroc` needs `data.group` |
| `score` | `[]` | `{name, probe?, <any data key>}`; unset keys inherit from `data`, explicit `null` overrides |
| `output.dir`, `output.overwrite` | `runs`, `false` | |

A section is required only when a listed step needs it:

| Step | Sections |
|---|---|
| `extract` | `model`, `data`, `extract`, and `probe` when `keep: pooled` |
| `train` | `data`, `probe` |
| `score` | `model`, `data`, `score` |

These checks run when the config loads, and errors name the offending key:
- Unknown keys are rejected.
- `keep: pooled` needs `kind: linear` with `pooling` `mean` or `last`.
- `attention`, `max` and `rolling_mean` need `keep: tokens`.
- `window: response` needs `data.response`.
- `split` must sum to 1.

## Recipes

`sonde run <name>` resolves `sonde/recipes/<name>.yaml`. Recipe data paths
are relative, so run recipes from the repo root.

| Recipe | Model | What it does |
|---|---|---|
| `quickstart` | small, CPU | tiny bundled dataset; extract, train and score in one go |
| `refusal` | Llama-3.1-8B-Instruct | linear probe at every block on `experiments/refusal_probing/data/labeled.jsonl`, `format: raw` |
| `high_stakes` | Llama-3.1-8B | attention probe at blocks 10–20 on `Arrrlex/models-under-pressure`, scored on five OOD test sets, `format: raw` |

The Llama models are gated. Run `huggingface-cli login` or set `HF_TOKEN`.

Both GPU recipes are `format: raw`, so their probes record
`prompt_format = "raw"`. mechanica refuses to use them on chat-templated
traffic, because a raw-fitted probe scores templated text near 1.0.

## The probe artifact

`probes/<name>.npz` holds `w` (`[H]` float32), `q` (`[H]`, attention only)
and a JSON `meta` with `format: 1`. It loads with numpy alone:

```python
import numpy as np
from sonde import probe

p = probe.Probe.load("runs/refusal/probes/refusal.npz")
acts = np.zeros((12, p.w.shape[0]), dtype=np.float32)  # [T, H] at p.block
score = p.pooled_score(acts)  # sigmoid of pooled logits, in [0, 1]
flagged = p.flag(acts)  # score >= p.threshold
```

- `threshold` lives on the `pooled_score` scale. It is chosen on validation
  scores of the exported numpy probe, the same function mechanica calls.
- `metrics` is the eval card: val and test metrics, the control, and the
  `n_pos` / `n_neg` behind each number.
- `engine` records the backend and library versions, for example
  `hf==5.1.0+nnterp==1.3.0`.

**Using it from mechanica.** mechanica imports `sonde.probe` and
`sonde.fingerprint` and loads a probe directory with `probe.load_dir`.
Before scoring a request it calls
`p.serves(fingerprint, adapter, prompt_format)`. That returns a reason
string when the checkpoint fingerprint, the LoRA adapter or the
prompt-format digest disagrees with what the probe was fitted on, and
`None` when the probe may score.

## Layer numbering

`block = L` is the residual stream right after decoder block `L` (0-indexed),
before any final norm. The last block is included.

| Where | Same tensor |
|---|---|
| nnterp (`hf` backend) | `model.layers_output[L]` |
| nnsight on vLLM (`vllm` backend) | `out = model.model.layers[L].output; out[0] + out[1]` |
| vLLM `extract_hidden_states` (mechanica) | aux layer `L + 1`, i.e. `p.vllm_aux_layer()` |
| HF `output_hidden_states` | `hidden_states[L + 1]` only for `L < N - 1`; the last entry is post-norm |

sonde never reads `output_hidden_states`. The `vllm` backend accepts Llama,
Mistral, Qwen2 and Qwen3 decoder layers and refuses any other architecture
when it loads.

## More

- [docs/linear-probes-primer.md](docs/linear-probes-primer.md): what linear
  probes measure, and the papers behind them.
- [CONTRIBUTING.md](CONTRIBUTING.md): dev setup, tests and style.
````

- [ ] **Step 3: Check the README against the code**

Run: `uv run ruff format --check .`
Expected: `N files already formatted`. ruff formats the README's Python blocks too (verified on 0.16.10: the two blocks above pass, and column-aligned inline comments do not). If it names a file under `docs/superpowers/`, phase 1's `extend-exclude` is missing that entry; add it there instead of reformatting the design docs.

Run: `uv run sonde run quickstart --dry-run; echo "exit=$?"`
Expected: the resolved YAML, a disk estimate line, and `exit=0`. No `runs/quickstart/` is created; `ls runs/quickstart 2>/dev/null` prints nothing if it did not exist before.

Run:
```bash
uv run python -c "
from sonde import config
for m in (config.DataConfig, config.ExtractConfig, config.ProbeConfig,
          config.OutputConfig):
    print(m.__name__, m.model_construct().model_dump())"
```
Expected: the defaults in the README table, i.e.
`DataConfig {'path': None, 'hf': None, 'hf_config': None, 'hf_split': 'train', 'text': 'text', 'messages': None, 'response': None, 'label': 'label', 'id': 'id', 'group': None, 'label_map': None, 'limit': None, 'format': 'chat', 'system': None, 'max_length': 2048, 'split': (0.7, 0.15, 0.15)}`, then
`ExtractConfig {'layers': 'all', 'window': 'prompt', 'keep': 'pooled', 'batch_size': 16, 'shard_size': 4096, 'shard_bytes': 2147483648}`, then
`ProbeConfig {'kind': 'linear', 'pooling': 'mean', 'rolling_window': None, 'init': 'random', 'epochs': 20, 'lr': 0.001, 'weight_decay': 0.0, 'batch_size': 256, 'patience': 3, 'max_fpr': 0.01, 'select': 'recall_at_fpr'}`, then
`OutputConfig {'dir': 'runs', 'overwrite': False}`.
If any value differs, fix the README row, not the code.

Run the README's Python block:
```bash
uv run python -c "
import sonde
from sonde import probe
run_dir = sonde.run(sonde.load_config('quickstart'))
for name, p in probe.load_dir(str(run_dir / 'probes')).items():
    print(name, p.block, p.pooling, round(p.threshold, 3))"
```
Expected: exactly one line, `<quickstart probe name> <int> <pooling> <float in [0, 1]>`.

- [ ] **Step 4: Add the CHANGELOG entry.** In `CHANGELOG.md`, replace the line `## [Unreleased]` with:

```markdown
## [Unreleased]

## [0.2.0] - Unreleased

A rewrite; nothing from 0.1 stays compatible. 0.1 is preserved at the git
tag `v0.1-legacy`.

### Added
- `sonde run CFG|RECIPE [-o k=v ...] [--dry-run]`, driven by one pydantic
  config (`extra="forbid"`) with an explicit `steps:` list.
- `sonde.probe.Probe` (`.npz`, format 1) and `sonde.fingerprint`, importable
  with numpy and the stdlib only. This is the artifact mechanica loads.
- Two backends, chosen explicitly: `hf` (nnterp 1.3 on nnsight 0.7) and
  `vllm` (nnsight's `VLLM`).
- Sharded, resumable extraction with a config-hash check that runs before
  any model loads.
- Linear (`mean`, `last`, `max`, `rolling_mean`) and attention probes,
  `diff_means` init, a validation-chosen threshold (`max_fpr`, default 1%,
  or best F1), recall at `max_fpr` as the headline with bootstrap CIs,
  rank and group AUROC, a shuffled-label control, and bag-of-words and
  length baselines.
- Recipes: `quickstart`, `refusal`, `high_stakes`.

### Fixed
- Activations at batch size > 1 were corrupted by padding without an
  attention mask. Chat templates are now applied, and the BOS token is no
  longer doubled.
- The seed is now set before initialisation, and training uses CUDA when
  available.

### Removed
- The old packages (`activation`, `core`, `dataset`, `directions`,
  `interventions`, `generation`, `probes`, `runners`, `configs`, `cli`),
  PCA, the extra activation kinds and the selector classes.
- The OmegaConf, einops and torchmetrics dependencies, and support for
  Python 3.10 and 3.11.
- The experiments `geometry_of_truth`, `rjudge_dissociation`,
  `scaleJSD_probing` and `refusal_probe_qwen.py`, plus `examples/`.
- The docs `toolkit_audit.md`, `intervention_design.md`,
  `oumi_integration_plan.md` and `research_gaps_and_extensions.md`, and the
  committed refusal activations.

### Changed
- CI is one job: `uv sync --locked`, then ruff, pyright and
  `pytest -m "not gpu"` on Python 3.12.
- `docs/linear-probes-primer.md`: an accuracy pass on its paper summaries
  and citations.
```

- [ ] **Step 5: Replace `CONTRIBUTING.md` with exactly:**

````markdown
# Contributing to sonde

## Dev setup

```bash
git clone https://github.com/aniruddh-alt/sonde.git
cd sonde
uv sync --extra hf --extra dev
uvx pre-commit install   # optional: ruff on every commit
```

GPU work happens on the DGX Spark in two venvs (`.venv-hf` and `.venv-vllm`);
see the README.

## The inner loop

These are the commands CI runs:

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright
uv run pytest -m "not gpu"
```

Tests that need a CUDA GPU carry `@pytest.mark.gpu`. Run them on the Spark
with `pytest -m gpu`. Skip a test by its marker, never by catching an
exception.

## Style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):
- 80 columns.
- `import module` or `from package import module`, never imported names
  (`typing` and `collections.abc` are the exceptions). In tests too: write
  `from sonde import probe`, then `probe.Probe(...)`.
- `from __future__ import annotations` at the top of every module.
- Args / Returns / Raises docstrings on public functions, giving array
  shapes.
- No comments that restate the code. Mark a deliberate shortcut with
  `# ponytail:` and name its limit.
- Tests use plain asserts and few fixtures.

`sonde/probe.py` and `sonde/fingerprint.py` must import only numpy and the
stdlib; `tests/test_probe.py` enforces this. mechanica imports them inside
vLLM workers.

## Common changes

**A new pooling or probe kind.** Change both halves together:
- the numpy scoring in `sonde/probe.py`;
- the torch module in `sonde/probes.py`;
- the parity case in `tests/test_probes.py`, which asserts that
  `Probe.pooled_score == sigmoid(module(x))` within 1e-5.

**A new vLLM architecture.** Add its decoder-layer class to the allowlist in
`sonde/backends.py` only after you check in vLLM's source that the layer
returns `(mlp_out, residual)`. Record that check in the PR.

**A new recipe.** Add `sonde/recipes/<name>.yaml` with `name: <name>`.
`tests/test_config.py` loads every recipe offline.

## Commits and PRs

Branch off `main`, keep commits atomic, and open a PR that says what changed
and why. CI must be green before merge.

## Reporting bugs

Open an issue with:
- the model and revision;
- the config (`runs/<name>/config.yaml`);
- the backend and `engine` string;
- the traceback.
````

- [ ] **Step 6: Check for dangling references**

Run: `git grep -n -e research_gaps -e toolkit_audit -e intervention_design -e oumi_integration -e causal_loop -e k8s_job -e InterventionContext -e LayerProbeSweepRunner -- ':!docs/superpowers' ':!CHANGELOG.md'`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add docs/linear-probes-primer.md README.md CHANGELOG.md CONTRIBUTING.md
git commit -m "docs: rewrite README for sonde run, fix primer, changelog 0.2.0"
```

---

### Task 4.4: CI, pre-commit pin, single pyright config

**Files:**
- Modify: `.github/workflows/ci.yml` (full rewrite)
- Modify: `.pre-commit-config.yaml` (full rewrite)
- Delete: `pyrightconfig.json`
- Modify: `pyproject.toml` (`[tool.pyright]` table)

**Interfaces:**
- Consumes (phase 1 packaging):
  - the `[dev]` pins (`ruff==0.16.10`, `pyright==1.1.414`, `pytest==…`)
  - the `[hf]` extra
  - the CPU-torch index for `sys_platform == 'linux' and platform_machine == 'x86_64'`
  - `[tool.ruff] extend-exclude = ["sonde/fingerprint.py", "docs/superpowers"]`
  - the `gpu` marker in `[tool.pytest.ini_options]`
  - `uv.lock`
- Produces: a green CI job named `check`.

- [ ] **Step 1: Make pyproject the only pyright config**

Run: `git rm -q pyrightconfig.json`

In `pyproject.toml`, replace the whole `[tool.pyright]` table (whatever phase 1 left) with:

```toml
[tool.pyright]
include = ["sonde"]
pythonVersion = "3.12"
typeCheckingMode = "basic"
reportPrivateImportUsage = "none"
venvPath = "."
venv = ".venv"
```

Run: `uv run pyright --verbose 2>&1 | grep -e "Loading pyproject.toml" -e "errors,"`
Expected: `Loading pyproject.toml file at <repo>/pyproject.toml` and `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 2: Pin the pre-commit ruff to the dev pin**

Run: `grep -n '"ruff==' pyproject.toml`
Expected: `"ruff==0.16.10",`

Replace `.pre-commit-config.yaml` with:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.10  # must equal the ruff== pin in pyproject [dev]
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=500]
```

Run: `uvx pre-commit run --all-files`
Expected: every hook prints `Passed`. If `trailing-whitespace` or `end-of-file-fixer` rewrites a file, it prints `Failed` with "files were modified". Stage those rewrites and rerun until every hook passes.

- [ ] **Step 3: Check that the `gpu` marker is registered**

Run: `uv run pytest --markers | grep "@pytest.mark.gpu"`
Expected: one line (phase 1 registers the marker).

- [ ] **Step 4: Replace `.github/workflows/ci.yml` with:**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.12"
          enable-cache: true
      - name: Install (CPU torch via the lock)
        run: uv sync --locked --extra hf --extra dev
      - uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-${{ runner.os }}-gpt2
      - run: uv run --no-sync ruff check .
      - run: uv run --no-sync ruff format --check .
      - run: uv run --no-sync pyright
      - run: uv run --no-sync pytest -m "not gpu"
```

The single job installs once. The fixed HF cache key saves gpt2 on the first run and restores it afterwards.

- [ ] **Step 5: Run the CI commands locally**

```bash
uv sync --locked --extra hf --extra dev \
  && uv run --no-sync ruff check . \
  && uv run --no-sync ruff format --check . \
  && uv run --no-sync pyright \
  && uv run --no-sync pytest -m "not gpu" -q
```

Expected: `All checks passed!`, `N files already formatted`, `0 errors, 0 warnings, 0 informations`, then a pytest summary with 0 failed. GPU tests appear as deselected.

- [ ] **Step 6: Check that the tree matches spec §4 and §14**

Run: `git ls-files sonde | grep -v -e '^sonde/recipes/data/' -e '^sonde/py.typed$' | sort`
Expected (exactly; `generate.py` and `steer.py` arrive in phase 6):
```
sonde/__init__.py
sonde/backends.py
sonde/cli.py
sonde/config.py
sonde/data.py
sonde/extract.py
sonde/fingerprint.py
sonde/probe.py
sonde/probes.py
sonde/recipes/high_stakes.yaml
sonde/recipes/quickstart.yaml
sonde/recipes/refusal.yaml
sonde/runner.py
sonde/score.py
sonde/store.py
sonde/sweep.py
sonde/train.py
```

Run: `git ls-files tests | sort`
Expected: `tests/test_config.py`, `tests/test_data.py`, `tests/test_e2e.py`, `tests/test_extract.py`, `tests/test_probe.py`, `tests/test_probes.py`, `tests/test_sweep.py`, `tests/test_train.py`, plus `tests/conftest.py` only if an earlier phase added one.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/ci.yml .pre-commit-config.yaml pyproject.toml
git add -u pyrightconfig.json
git commit -m "ci: single locked uv job; pin pre-commit ruff; one pyright config"
```

- [ ] **Step 8: Push and confirm CI is green (needs the user's go-ahead to push)**

```bash
git push -u origin HEAD
gh pr view --json number,url || gh pr create --fill --base main
gh run watch "$(gh run list --branch "$(git branch --show-current)" --limit 1 --json databaseId -q '.[0].databaseId')" --exit-status
```

Expected: the `check` job finishes with every step marked ✓ and `gh run watch` exits 0. If it fails, fix the cause, commit, and push again. Never skip a step or mark a test `gpu` to get green.

<!-- skipped: none of the issues were wrong. Two parts of the fixes fall outside phase 4 and were not edited here:
(1) Task 1.2's pyproject `extend-exclude = ["sonde/fingerprint.py", "docs/superpowers"]`, with its ponytail comment "docs/superpowers holds design docs with illustrative, unformatted code". Phase 4 now lists it under Consumes and checks it in Task 4.3 Step 3.
(2) Task 5.5 Steps 5 and 6 must drop both `-o name=...` overrides, now that the recipes say `name: refusal` and `name: high_stakes`. Task 4.1's test now pins each recipe's name to its file stem.
Verified here: ruff 0.16.10 passes the corrected README Python blocks and fails the aligned-comment version; the kept primer, RESULTS.md, CHANGELOG.md, CONTRIBUTING.md and refusal README all pass `ruff format --check` at 80 columns; `git ls-remote` shows ruff-pre-commit tag v0.16.10 at f12be1eb. -->

---

## Phase 5: vLLM backend + Spark parity spike

Goal: give `sonde/backends.py` its `vllm` branch. That branch is an nnsight `VLLM` engine running eagerly. It refuses any decoder layer whose output is not `(mlp_out, residual)`, and it runs one prefill per row that pools inside the worker. Then measure it on the DGX Spark against sonde's hf backend and against vLLM's native `extract_hidden_states`, the capture mechanica serves with. Every number comes from committed scripts. Done when:
1. the nnsight `tests/vllm` smoke has picked the `[vllm]` pin and `uv.lock` matches it;
2. `tests/test_vllm.py` passes on the Spark (`-m gpu`) and the CPU suite still passes;
3. `docs/parity-2026-10.md` has measured numbers and was written by `scripts/spike/compare.py` alone;
4. the refusal and high_stakes recipes have re-run on the Spark with the chosen backend, and `scripts/spike/results.py` has written the commit, backend and command into each `RESULTS.md`.

**Checked while writing this phase.** I ran the library facts in a scratch venv (nnterp 1.3.0, nnsight 0.7.0, torch 2.14.1, transformers 5.18.0) or read them from source at the pinned commits. Nothing that executes vLLM can run here, so the GPU tests and the spike are the check for those parts.
- transformers 5.18 accepts `AutoModelForCausalLM.from_pretrained(..., dtype=...)`. On gpt2, `hidden_states[L + 1]` equals decoder block L's hooked output for every L < N-1. It differs at N-1, because the last entry is post-norm.
- `Qwen/Qwen3-0.6B` has 28 layers, hidden size 1024 and `Qwen3ForCausalLM`. Its chat render has no BOS. `scripts/spike/tokens.py` ran on the real `labeled.jsonl` (256 rows).
- `importlib.metadata.distribution(x).read_text("direct_url.json")` returns `None` for a PyPI install. I could not check here whether uv writes `vcs_info.commit_id` for a git install. `test_load` therefore pins the exact engine string, and the Spark run checks it.
- `compare.py` and `results.py` ran on synthetic shards with a stub `sonde.store`, and `results.py` splices idempotently. ruff at 80 columns and pyright in basic mode are clean on every new file, apart from the phase-2 symbols that the stubs lack.
- Phase 2's `sonde/backends.py` (read from its plan) already imports `from importlib import metadata`, `from typing import Any`, `import nnsight` and `from sonde import config, data`. Its `load_model` guards with `raise NotImplementedError("vllm backend lands in phase 5")`, its `forward` has no vllm branch and starts with `model = loaded.model`, and `tests/test_extract.py` has `test_vllm_backend_is_not_here_yet`. Phase 1's `[vllm]` extra carries `; sys_platform == 'linux'` markers, and its torch source is scoped with `extra = "hf"`. Phase 4 sets `[tool.pyright] include = ["sonde"]`.
- Checked in nnsight b717807 source and its own tests:
  - `VLLM(repo, **kw)` forwards `revision` and `dtype` to `EngineArgs` and `vllm.LLM`. It `setdefault`s `enable_chunked_prefill=False` and `enable_prefix_caching=False`, and runs with `enforce_eager=not taps`. It sets `VLLM_USE_V2_MODEL_RUNNER=0` and raises if that variable is set to anything else.
  - `dispatch=False` builds only the meta tree. `model.dispatch()` reuses that tree and leaves it at `model._module` (`test_dispatch_does_not_rebuild_the_meta_tree`).
  - A list saved above the invokes with `nnsight.save([None] * n)` comes back merged slot-wise (`test_a_container_saved_above_the_invokes_still_merges`). A name saved separately in each invoke comes back as a list, except when there is only one invoke (`VLLM._collect`).
  - `Invoker.execute` snapshots `dict(frame.f_locals)` per invoke, so loop variables are per-row.
  - `tests/vllm/conftest.py` prepends `src/` to `sys.path`.
- vLLM v0.27.1 and v0.30.0 source: `LlamaDecoderLayer`, `Qwen2DecoderLayer`, `Qwen3DecoderLayer` and `MistralDecoderLayer(LlamaDecoderLayer)` all `return hidden_states, residual` after the MLP. In the v0.30.0 registry, `MistralForCausalLM` maps to `mistral.MistralForCausalLM`.
- The native capture config is copied from mechanica's `scripts/train_probe_pipeline.py`: `speculative_config` with `extract_hidden_states`, `KVTransferConfig(ExampleHiddenStatesConnector)`, `load_hidden_states`, the token-id check and `cleanup_hidden_states`.
- From the dgx-spark helper:
  - a `run` job starts in `~/experiments/<name>/code`;
  - `sync` does an `rsync --delete` that skips `.venv`, `outputs` and `runs`;
  - the sitecustomize cap is `SPARK_MEM_GB/121`, and every vLLM `gpu_memory_utilization` below stays under its job's cap.

**Deviations from the spec, each deliberate.**
- The allowlist also names `MistralDecoderLayer`. The spec says "Mistral uses Llama's class", but since vLLM 0.27.1 Mistral has its own subclass. The source check above records this.
- Per-row results go into one slot list saved above the invokes, because per-invoke `nnsight.save` names would come back unwrapped for a one-row chunk. Saving still uses `nnsight.save`, never `.save()`.
- The vllm tokenizer is `transformers.AutoTokenizer`, which reads the same files the hf backend tokenizes with. vLLM's tokenizer wrapper is not used.
- The pin search tries 0.30.0, then 0.29.0, then 0.27.1, because §12 says "highest ≥ 0.29".
- `ninja` joins the `[vllm]` extra (linux only, like vLLM itself), because the dgx-spark skill requires it in any venv that JIT-compiles kernels.

---

### Task 5.1: Pick the vLLM pin on the Spark and build the two venvs

**Files:**
- Create: `scripts/spike/smoke.sh`, `scripts/spike/venvs.sh`
- Modify: `pyproject.toml` (the `vllm` list under `[project.optional-dependencies]`, plus `[tool.uv.sources]`/`[[tool.uv.index]]` only if Step 11 runs), `uv.lock`
- Modify only if every smoke fails: `sonde/recipes/refusal.yaml`, `sonde/recipes/high_stakes.yaml`, `README.md`

**Interfaces:**
- Consumes: phase 1's `[vllm]` extra, `[tool.uv] conflicts` and the `extra = "hf"`-scoped torch source; phase 4's recipes `refusal` and `high_stakes` (their `model.backend: vllm`).
- Produces, for Tasks 5.2–5.5:
  - the pinned `vllm==<version>` in `pyproject.toml`;
  - on the Spark, `~/experiments/sonde-spike/{code/, .venv-hf/, .venv-vllm/, outputs/spike/smoke-vllm-<version>.txt}`.
  - Every later Spark command uses `S=~/.claude/skills/dgx-spark/scripts/spark`.

- [ ] **Step 1: Check the Spark**

```bash
~/.claude/skills/dgx-spark/scripts/spark check
```

Expected: `safe_to_use_gb` of at least 40. If another GPU compute process is listed, tell the user and ask before going on.

- [ ] **Step 2: Write `scripts/spike/smoke.sh`**

```bash
#!/usr/bin/env bash
# Spec §13 check 1: nnsight's own vLLM tests, at the pinned nnsight commit,
# against one vLLM version. On the Spark, from ~/experiments/sonde-spike/code:
#   bash scripts/spike/smoke.sh 0.30.0
set -eu
v=$1
sha=b71780727ea9713f74ce12f76b1d3548b91a74a0
uv=~/.local/bin/uv
venv=../.venv-smoke-$v
mkdir -p ../outputs/spike
if [ ! -d ../nnsight ]; then
  git clone -q https://github.com/ndif-team/nnsight ../nnsight
  git -C ../nnsight checkout -q $sha
  # tests/vllm/conftest.py puts src/ first on sys.path, and src/ has no built
  # C extension; without src/ the tests import the installed same-commit copy.
  rm -rf ../nnsight/src
fi
if [ ! -d $venv ]; then
  $uv venv $venv --python 3.12 --managed-python
  $uv pip install --python $venv/bin/python "vllm==$v" ninja pytest \
    "nnsight[vllm] @ git+https://github.com/ndif-team/nnsight@$sha"
  cp ~/experiments/.venv/lib/python3.12/site-packages/sitecustomize.py \
    $venv/lib/python3.12/site-packages/
fi
$venv/bin/python -c "import nnsight._c, vllm; print('vllm', vllm.__version__)"
cd ../nnsight
$venv/bin/python -m pytest tests/vllm/test_registration.py \
  tests/vllm/test_tracing.py tests/vllm/test_chunked_prefill.py -q 2>&1 \
  | tee ../outputs/spike/smoke-vllm-$v.txt
```

Run: `bash -n scripts/spike/smoke.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Commit the smoke script and sync**

```bash
git add scripts/spike/smoke.sh
git commit -m "chore(spike): nnsight tests/vllm smoke script for the Spark"
git status --porcelain
~/.claude/skills/dgx-spark/scripts/spark sync "$(git rev-parse --show-toplevel)" sonde-spike
```

Expected: `git status --porcelain` prints nothing, then `synced ... -> 10.97.110.177:~/experiments/sonde-spike/code`.

- [ ] **Step 4: Smoke vLLM 0.30.0**

The test fixtures hold up to three gpt2 engines at `gpu_memory_utilization=0.1` each (about 12 GB apiece), so the job declares 40 GB.

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S run sonde-spike 40 'bash scripts/spike/smoke.sh 0.30.0'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
ssh 10.97.110.177 'grep -m1 "^vllm " ~/experiments/sonde-spike/runs/latest.log; tail -n 1 ~/experiments/sonde-spike/outputs/spike/smoke-vllm-0.30.0.txt'
```

Expected: `vllm 0.30.0`, then a last line of the form `N passed[, M skipped] in Ts`.
- The result is clean when that line contains neither `failed` nor `error`.
- If `tail` reports that the file does not exist, read `$S logs sonde-spike 30`. An `ImportError` at `import nnsight._c` means nnsight's optional C extension did not build on aarch64. Stop and report it. The tests use `.save()`, so in that case the smoke would fail because of the missing extension, not because of vLLM.

- [ ] **Step 5: If 0.30.0 is not clean, smoke 0.29.0**

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S run sonde-spike 40 'bash scripts/spike/smoke.sh 0.29.0'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
ssh 10.97.110.177 'tail -n 1 ~/experiments/sonde-spike/outputs/spike/smoke-vllm-0.29.0.txt'
```

Expected: same as Step 4.

- [ ] **Step 6: If 0.29.0 is not clean either, smoke 0.27.1**

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S run sonde-spike 40 'bash scripts/spike/smoke.sh 0.27.1'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
ssh 10.97.110.177 'tail -n 1 ~/experiments/sonde-spike/outputs/spike/smoke-vllm-0.27.1.txt'
```

Expected: same as Step 4.

- [ ] **Step 7: Set the pin**

The pin is the first clean version among 0.30.0, 0.29.0 and 0.27.1. If none is clean, pin `0.27.1`, the version nnsight tests on, and ship the backend as experimental. In `pyproject.toml`, make the `vllm` list under `[project.optional-dependencies]` exactly the block below. It is shown for 0.30.0; write the version you chose. The `sys_platform == 'linux'` markers stay: they are what let `uv lock` resolve on macOS, where vllm has no wheels.

```toml
vllm = [
    "sonde[core]",
    "vllm==0.30.0; sys_platform == 'linux'",
    "ninja; sys_platform == 'linux'",
    "nnsight[vllm] @ git+https://github.com/ndif-team/nnsight@b71780727ea9713f74ce12f76b1d3548b91a74a0 ; sys_platform == 'linux'",
]
```

Run:

```bash
uv lock
grep -A2 '^name = "torch"$' uv.lock
uv sync --locked --extra hf --extra dev
```

Expected:
- `uv lock` prints `Resolved <n> packages in ...` with no conflict error.
- The `grep` finds two torch entries, the same check as phase 1 Task 1.2 Step 4. One is vLLM's torch with `source = { registry = "https://pypi.org/simple" }`. The other is `<version>+cpu` with `source = { registry = "https://download.pytorch.org/whl/cpu" }`.
- The Mac sync succeeds.

- [ ] **Step 8: Only if no version was clean, default the recipes to hf**

In both `sonde/recipes/refusal.yaml` and `sonde/recipes/high_stakes.yaml`, change the model section's `backend: vllm` to `backend: hf`. Add this sentence to README's vLLM setup section: "The vllm backend is experimental: nnsight's `tests/vllm` failed on vLLM 0.30.0, 0.29.0 and 0.27.1 at nnsight b717807 (see `docs/parity-2026-10.md`), so the recipes default to `hf`."

Run: `uv run --extra hf --extra dev pytest tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 9: Write `scripts/spike/venvs.sh`**

```bash
#!/usr/bin/env bash
# Builds the Spark's two sonde venvs from uv.lock (spec §12). On the Spark,
# from ~/experiments/sonde-spike/code: bash scripts/spike/venvs.sh
set -eu
uv=~/.local/bin/uv
for extra in hf vllm; do
  venv=../.venv-$extra
  [ -d $venv ] || $uv venv $venv --python 3.12 --managed-python
  UV_PROJECT_ENVIRONMENT=$venv $uv sync --locked --extra $extra --extra dev
  cp ~/experiments/.venv/lib/python3.12/site-packages/sitecustomize.py \
    $venv/lib/python3.12/site-packages/
  $venv/bin/python -c "import torch; print('$extra', torch.__version__,
    torch.cuda.is_available(), torch.cuda.get_device_capability())"
done
```

Run: `bash -n scripts/spike/venvs.sh`
Expected: no output, exit 0.

- [ ] **Step 10: Commit the pin and build the venvs**

```bash
git add pyproject.toml uv.lock scripts/spike/venvs.sh
git add sonde/recipes README.md   # only if Step 8 ran
git commit -m "build: pin the vllm extra from the Spark nnsight tests/vllm smoke"
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 8 'bash scripts/spike/venvs.sh'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 4
```

Expected: the last two lines before `[spark] exit code 0` are `hf <torch version> True (12, 1)` and `vllm <the torch version vLLM pins> True (12, 1)`.

- [ ] **Step 11: Only if the hf line printed `False`, add a CUDA torch source for aarch64**

A `False` means the venv got CPU-only aarch64 torch from PyPI, and spec §12 asks the spike to confirm CUDA torch on aarch64. In `pyproject.toml`, add this index next to phase 1's `pytorch-cpu` index:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Then make the `torch` entry under `[tool.uv.sources]` exactly the block below. Both entries keep `extra = "hf"`, so neither index leaks into the vllm extra, which must keep vLLM's own PyPI torch.

```toml
torch = [
    { index = "pytorch-cpu", extra = "hf", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
    { index = "pytorch-cu130", extra = "hf", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
]
```

```bash
uv lock
grep -A2 '^name = "torch"$' uv.lock
git commit -am "build: CUDA torch for linux aarch64 (the Spark)"
ssh 10.97.110.177 'rm -rf ~/experiments/sonde-spike/.venv-hf ~/experiments/sonde-spike/.venv-vllm'
```

Expected: `grep` shows three torch entries: the pypi.org one (still vLLM's), the `+cpu` one from `whl/cpu` and a `+cu130` one from `whl/cu130`. Then repeat Step 10's sync, run, wait and logs commands. Both lines should end in `True (12, 1)`.

---

### Task 5.2: `load_model` for vllm

**Files:**
- Modify: `sonde/backends.py`, `tests/test_extract.py`
- Create: `tests/test_vllm.py`

**Interfaces:**
- Consumes (phase 2): `config.ModelConfig` (`name`, `revision`, `dtype`, `backend`, `vllm: dict`); `backends.Loaded(backend, model, tokenizer, num_layers, hidden, engine)`; `backends.load_model(cfg: ModelConfig) -> Loaded`, with its single-slot cache keyed by `cfg.model_dump_json()`; `backends._load_hf`.
- Produces:
  - `backends._load_vllm(cfg: config.ModelConfig) -> Loaded`, reached only through `load_model`. It returns `Loaded.backend == "vllm"` and `Loaded.engine == f"vllm=={vllm.__version__}+nnsight=={sha7}"`. `Loaded.model` is an nnsight `VLLM`, with decoder layers at `.model.layers[L]`.
  - `backends._VLLM_LAYERS: frozenset[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vllm.py`. Keep `test_refuses_unlisted_layers` as the last test in the file, because a refused build leaves nnsight's client process group up.

```python
"""vLLM backend checks. Run on the Spark in .venv-vllm with -m gpu."""

from __future__ import annotations

import pytest

from sonde import backends
from sonde import config

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen3-0.6B"
CFG = config.ModelConfig(
    name=MODEL,
    backend="vllm",
    vllm={"gpu_memory_utilization": 0.15, "max_model_len": 1024},
)


def test_load():
    import vllm  # pyright: ignore[reportMissingImports]

    loaded = backends.load_model(CFG)
    assert loaded.backend == "vllm"
    assert (loaded.num_layers, loaded.hidden) == (28, 1024)
    assert loaded.engine == f"vllm=={vllm.__version__}+nnsight==b717807"


def test_refuses_unlisted_layers():
    # Last: a refused build leaves nnsight's client process group up.
    gpt2 = config.ModelConfig(
        name="gpt2", backend="vllm", vllm={"gpu_memory_utilization": 0.1}
    )
    with pytest.raises(ValueError, match="GPT2"):
        backends.load_model(gpt2)
```

- [ ] **Step 2: Run on the Spark to verify both fail**

```bash
git status --porcelain   # only tests/test_vllm.py; sync copies the working tree
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 24 '../.venv-vllm/bin/python -m pytest -m gpu tests/test_vllm.py -v'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 40
```

Expected: `2 failed`, both with `NotImplementedError: vllm backend lands in phase 5` from phase 2's guard in `load_model`. `test_refuses_unlisted_layers` fails because `pytest.raises(ValueError)` does not catch that error.

- [ ] **Step 3: Implement the vllm loader**

In `sonde/backends.py`, add `import json` to the stdlib imports. Phase 2 already imports `from importlib import metadata`, `from typing import Any` and `nnsight`, so add nothing else. In `load_model`, replace

```python
    Raises:
        NotImplementedError: For `backend: vllm` until phase 5.
    """
    key = cfg.model_dump_json()
    if key not in _cache:
        if cfg.backend != "hf":
            raise NotImplementedError("vllm backend lands in phase 5")
        _cache.clear()
        _cache[key] = _load_hf(cfg)
    return _cache[key]
```

with

```python
    Raises:
        ValueError: For `backend: vllm`, a decoder layer outside the
            allowlist.
    """
    key = cfg.model_dump_json()
    if key not in _cache:
        _cache.clear()
        load = _load_vllm if cfg.backend == "vllm" else _load_hf
        _cache[key] = load(cfg)
    return _cache[key]
```

Then add these definitions after `_load_hf`:

```python
_VLLM_LAYERS = frozenset(
    {
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "Qwen3DecoderLayer",
    }
)


def _nnsight_ref() -> str:
    dist = metadata.distribution("nnsight")
    url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = url.get("vcs_info", {}).get("commit_id")
    return commit[:7] if commit else dist.version


def _load_vllm(cfg: config.ModelConfig) -> Loaded:
    """Builds an eager nnsight VLLM engine for an allowlisted architecture.

    The meta tree is checked before dispatch, so a refused model never
    allocates GPU memory.

    Args:
      cfg: the run's model section; cfg.vllm passes through to vllm.LLM.

    Returns:
      A Loaded whose engine is "vllm==<version>+nnsight==<sha>".

    Raises:
      ValueError: a decoder layer's output is not (mlp_out, residual).
    """
    import transformers
    import vllm  # pyright: ignore[reportMissingImports]
    from nnsight.modeling import vllm as nnsight_vllm

    model = nnsight_vllm.VLLM(
        cfg.name,
        revision=cfg.revision,
        dtype=cfg.dtype,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        **cfg.vllm,
    )
    root = model._module
    layers = getattr(getattr(root, "model", None), "layers", ())
    kinds = {type(layer).__name__ for layer in layers}
    if not kinds or not kinds <= _VLLM_LAYERS:
        found = sorted(kinds) or type(root).__name__
        raise ValueError(
            f"{cfg.name}: vLLM decoder layers {found} are not in "
            f"{sorted(_VLLM_LAYERS)}, whose output is (mlp_out, residual)"
        )
    model.dispatch()
    hf = transformers.AutoConfig.from_pretrained(
        cfg.name, revision=cfg.revision
    )
    return Loaded(
        backend="vllm",
        model=model,
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            cfg.name, revision=cfg.revision
        ),
        num_layers=hf.num_hidden_layers,
        hidden=hf.hidden_size,
        engine=f"vllm=={vllm.__version__}+nnsight=={_nnsight_ref()}",
    )
```

In `tests/test_extract.py`, delete the whole `test_vllm_backend_is_not_here_yet` function. In the hf env it would now reach `import vllm` and fail with `ModuleNotFoundError`, and the GPU tests in `tests/test_vllm.py` cover the vllm branch.

- [ ] **Step 4: Check that the CPU suite, lint and types still pass on the Mac**

```bash
uv run --extra hf --extra dev pytest -m "not gpu" -q
uv run --extra hf --extra dev ruff check sonde tests
uv run --extra hf --extra dev ruff format --check sonde tests
uv run --extra hf --extra dev pyright
```

Expected:
- pytest: every test passes. The pass count is one lower than before this task, because `test_vllm_backend_is_not_here_yet` is gone, and `tests/test_vllm.py` is deselected.
- `ruff check`: `All checks passed!`.
- `ruff format --check`: `... files already formatted`.
- pyright: `0 errors`. It checks the configured include, `sonde` only, which matches CI.

- [ ] **Step 5: Run on the Spark to verify both pass**

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 24 '../.venv-vllm/bin/python -m pytest -m gpu tests/test_vllm.py -v'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 40
```

Expected: `2 passed`.
- If only the engine assert fails, and its right side shows a version string instead of `b717807`, uv did not write `vcs_info` into `direct_url.json`. Print `importlib.metadata.distribution("nnsight").read_text("direct_url.json")` in `.venv-vllm` and fix `_nnsight_ref` to read what is actually there. Do not relax the test.

- [ ] **Step 6: Commit**

```bash
git add sonde/backends.py tests/test_vllm.py tests/test_extract.py
git commit -m "feat(backends): vllm load_model with decoder-layer allowlist"
```

---

### Task 5.3: vllm forward

**Files:**
- Modify: `sonde/backends.py`
- Modify: `tests/test_vllm.py`

**Interfaces:**
- Consumes:
  - phase 2: `backends.window_pool(h: torch.Tensor, span: tuple[int, int], keep: str, pooling: str | None) -> torch.Tensor`, which maps `[T, H]` to `[H]` or `[end - start, H]`;
  - phase 2: `data.Encoded(id, ids, span, label, group)`;
  - phase 2: `backends.forward(loaded, batch, blocks, keep, pooling) -> dict[int, list[torch.Tensor]]`;
  - Task 5.2: `Loaded.model.model.layers[L]`.
- Produces: `backends.vllm_forward(loaded: Loaded, batch: list[data.Encoded], blocks: list[int], keep: str, pooling: str | None) -> dict[int, list[torch.Tensor]]`. It returns CPU tensors in batch order and in the model dtype. `forward` dispatches to it when `loaded.backend == "vllm"`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_vllm.py`, replace the import block and constants with the block below. Keep `test_load` and `test_refuses_unlisted_layers`.

```python
"""vLLM backend checks. Run on the Spark in .venv-vllm with -m gpu."""

from __future__ import annotations

import typing

import pytest
import torch
import transformers

from sonde import backends
from sonde import config
from sonde import data

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen3-0.6B"
BLOCK = 14
CFG = config.ModelConfig(
    name=MODEL,
    backend="vllm",
    vllm={"gpu_memory_utilization": 0.15, "max_model_len": 1024},
)
# HF and vLLM run different bf16 kernels; the spike measures the real gap.
TOL = 2e-2


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _rows(tokenizer) -> list[data.Encoded]:
    pairs = [
        ("The capital of France is", " Paris, which sits on the Seine."),
        ("List three primes:", " 2, 3 and 5."),
    ]
    rows = []
    for i, (prompt, response) in enumerate(pairs):
        p = tokenizer(prompt).input_ids
        r = tokenizer(response, add_special_tokens=False).input_ids
        rows.append(
            data.Encoded(
                id=str(i),
                ids=p + r,
                span=(len(p), len(p) + len(r)),
                label=None,
                group=None,
            )
        )
    return rows
```

Then insert these two tests between `test_load` and `test_refuses_unlisted_layers`:

```python
def test_pooled_response_rows_differ_and_match_hf():
    loaded = backends.load_model(CFG)
    rows = _rows(loaded.tokenizer)
    got = backends.forward(loaded, rows, [BLOCK], "pooled", "mean")[BLOCK]
    hf: typing.Any = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    )
    hf.cuda()
    for row, x in zip(rows, got, strict=True):
        ids = torch.tensor([row.ids], device="cuda")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        start, end = row.span
        # hidden_states[L + 1] is block L's raw output for L < N - 1.
        ref = hs[BLOCK + 1][0, start:end].float().mean(0).cpu()
        assert x.shape == (1024,) and x.device.type == "cpu"
        assert _rel(x, ref) < TOL
    assert _rel(got[0], got[1]) > 0.1


def test_token_rows_cover_the_window():
    loaded = backends.load_model(CFG)
    rows = _rows(loaded.tokenizer)
    for batch in (rows[:1], rows):
        got = backends.forward(loaded, batch, [3, BLOCK], "tokens", None)
        want = [(r.span[1] - r.span[0], 1024) for r in batch]
        assert [tuple(x.shape) for x in got[3]] == want
        assert [tuple(x.shape) for x in got[BLOCK]] == want
```

`rows[:1]` covers a trace with one invoke. `rows` covers two invokes with different response spans.

- [ ] **Step 2: Run on the Spark to verify the new tests fail**

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 24 '../.venv-vllm/bin/python -m pytest -m gpu tests/test_vllm.py -v'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 40
```

Expected: `2 failed, 2 passed`. Phase 2's `forward` assumes an nnterp model and runs its hf path on the VLLM object. The two forward tests therefore fail with an `AttributeError` inside that path, for example on `model.device` or `model.layers_output`.

- [ ] **Step 3: Implement `vllm_forward`**

In `sonde/backends.py`, add `_MAX_INVOKES = 500` next to `_VLLM_LAYERS`. Then add this function after `_load_vllm`. `nnsight` and `data` are already module-level imports from phase 2.

```python
def vllm_forward(
    loaded: Loaded,
    batch: list[data.Encoded],
    blocks: list[int],
    keep: str,
    pooling: str | None,
) -> dict[int, list[torch.Tensor]]:
    """Prefills each row as its own vLLM request and pools its window.

    A row sends only ids[:end]: the model is causal, so later tokens cannot
    change the window.

    Args:
      loaded: a Loaded with backend "vllm".
      batch: rows to prefill.
      blocks: ascending block indices.
      keep: "pooled" or "tokens".
      pooling: "mean" or "last" when keep is "pooled", else None.

    Returns:
      block -> one CPU tensor per row, in batch order and the model dtype:
      [H] when pooled, else [end - start, H].

    Raises:
      RuntimeError: a request saw a row count other than its prompt length.
    """
    model = loaded.model
    layers = [model.model.layers[b] for b in blocks]
    feats: dict[int, list[torch.Tensor]] = {b: [] for b in blocks}
    for k in range(0, len(batch), _MAX_INVOKES):
        chunk = batch[k : k + _MAX_INVOKES]
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            got = nnsight.save([None] * len(chunk))
            for i, row in enumerate(chunk):
                start, end = map(int, row.span)
                with tracer.invoke(row.ids[:end]):
                    saved = []
                    for layer in layers:
                        out = layer.output
                        h = out[0] + out[1]
                        x = window_pool(h, (start, end), keep, pooling)
                        saved.append((x.cpu(), h.shape[0]))
                    got[i] = saved
        for row, saved in zip(chunk, got, strict=True):
            for b, (x, n) in zip(blocks, saved, strict=True):
                if n != row.span[1]:
                    raise RuntimeError(
                        f"row {row.id} block {b}: {n} rows for a "
                        f"{row.span[1]}-token prompt"
                    )
                feats[b].append(x)
    return feats
```

nnsight captures the trace body from source and ships each invoke body to the worker. So:
- keep the body inline;
- inside the invoke, reference only `layers`, `window_pool`, `start`, `end`, `keep`, `pooling`, `got` and `i`. Never reference `loaded` or `model`, which would ship the whole model with every request;
- do not move the envoy lookup into the trace.

Then insert this as the first statement of `forward`'s body, before `model = loaded.model`:

```python
    if loaded.backend == "vllm":
        return vllm_forward(loaded, batch, blocks, keep, pooling)
```

- [ ] **Step 4: Check that the CPU suite, lint and types still pass on the Mac**

```bash
uv run --extra hf --extra dev pytest -m "not gpu" -q
uv run --extra hf --extra dev ruff check sonde tests
uv run --extra hf --extra dev ruff format --check sonde tests
uv run --extra hf --extra dev pyright
```

Expected: pytest passes with the same count as Task 5.2 Step 4. ruff prints `All checks passed!` and reports the files already formatted. pyright reports `0 errors`.

- [ ] **Step 5: Run on the Spark to verify all pass**

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 24 '../.venv-vllm/bin/python -m pytest -m gpu tests/test_vllm.py -v'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 40
```

Expected: `4 passed`. If only `_rel(x, ref) < TOL` fails, print the ratio:
- A ratio of about 0.1 or more means a wrong block index (compare spec §5 layer semantics). That is a bug to fix.
- A ratio between `TOL` and 0.1 is the bf16 engine gap. Set `TOL` to twice the largest observed ratio, rounded up to one significant figure, and put the observed value in the commit message. Task 5.4 Step 9 re-checks it against the spike's block-14 number.

- [ ] **Step 6: Commit**

```bash
git add sonde/backends.py tests/test_vllm.py
git commit -m "feat(backends): vllm forward, one prefill request per row, pooled in the worker"
```

---

### Task 5.4: Parity spike scripts and `docs/parity-2026-10.md`

**Files:**
- Create: `scripts/spike/tokens.py`, `scripts/spike/dump_sonde.py`, `scripts/spike/dump_native.py`, `scripts/spike/compare.py`, `scripts/spike/run.sh`
- Create (generated on the Spark): `docs/parity-2026-10.md`

**Interfaces:**
- Consumes:
  - `backends.load_model(cfg: config.ModelConfig) -> backends.Loaded`;
  - `config.resolve_blocks(spec, num_layers: int) -> list[int]`;
  - `data.Encoded(id, ids, span, label, group)`;
  - `extract.extract_to(loaded, encoded, blocks, keep, pooling, out_dir, batch_size, shard_size, shard_bytes, meta) -> None`;
  - `store.read_layer(out_dir, block) -> (X [ΣT, H], offsets [n + 1] | None)`;
  - `store.read_manifest(out_dir) -> dict`, with keys `engine`, `model`, `blocks` and `ids`;
  - Task 5.1's Spark venvs and `outputs/spike/smoke-vllm-*.txt`.
- Produces:
  - on the Spark, in `outputs/spike/`:
    - `tokens.jsonl` (`{"id", "ids"}` per prompt);
    - sonde shard dirs `hf-bf16`, `hf-fp32`, `vllm-bf16`, `vllm-fp32`, `tp-hf` and `tp-vllm`, each with `manifest.json` and `timing.json`;
    - `native-{eager,compiled}.safetensors` (`B{block}` `[ΣT, H]` plus `offsets`), each with a `.json` sidecar that names the engine;
  - in the repo, `docs/parity-2026-10.md`.

- [ ] **Step 1: Write `scripts/spike/tokens.py`**

```python
"""Writes the spike's token ids: Qwen3 chat renders of the refusal prompts.

Every dump reads this one file, so all sides see identical token ids.
Run in .venv-hf from the repo root: python scripts/spike/tokens.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import transformers

MODEL = "Qwen/Qwen3-0.6B"
DATA = "experiments/refusal_probing/data/labeled.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="../outputs/spike/tokens.jsonl")
    args = parser.parse_args()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    with open(DATA) as f:
        prompts = [json.loads(line)["original_prompt"] for line in f]
    prompts = prompts[: args.limit]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, prompt in enumerate(prompts):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tok(text, add_special_tokens=False).input_ids
            f.write(json.dumps({"id": str(i), "ids": ids}) + "\n")
    print(f"wrote {len(prompts)} prompts to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/spike/dump_sonde.py`**

```python
"""Extracts the spike's token ids through sonde's own backend into shards.

Run from the repo root in .venv-hf (--backend hf) or .venv-vllm
(--backend vllm). Writes OUT/{shard_*.safetensors, manifest.json,
timing.json}.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

from sonde import backends
from sonde import config
from sonde import data
from sonde import extract

MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--keep", choices=["tokens", "pooled"], default="tokens"
    )
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mem-gb", type=float, default=16)
    parser.add_argument("--tokens", default="../outputs/spike/tokens.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.tokens) as f:
        rows = [json.loads(line) for line in f]
    encoded = [
        data.Encoded(
            id=f"{row['id']}.{k}",
            ids=row["ids"],
            span=(0, len(row["ids"])),
            label=None,
            group=None,
        )
        for k in range(args.repeat)
        for row in rows
    ]
    loaded = backends.load_model(
        config.ModelConfig(
            name=MODEL,
            dtype=args.dtype,
            backend=args.backend,
            vllm={
                "gpu_memory_utilization": args.mem_gb / 121,
                "max_model_len": 4096,
            },
        )
    )
    spec = (
        "all"
        if args.blocks == "all"
        else [int(b) for b in args.blocks.split(",")]
    )
    blocks = config.resolve_blocks(spec, loaded.num_layers)
    pooling = "mean" if args.keep == "pooled" else None
    out = pathlib.Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    meta = {
        "model": MODEL,
        "engine": loaded.engine,
        "dtype": args.dtype,
        "blocks": blocks,
        "window": "prompt",
        "keep": args.keep,
        "pooling": pooling,
    }
    began = time.perf_counter()
    extract.extract_to(
        loaded,
        encoded,
        blocks,
        args.keep,
        pooling,
        out,
        args.batch_size,
        4096,
        2**31,
        meta,
    )
    seconds = time.perf_counter() - began
    timing = {
        "engine": loaded.engine,
        "prompts": len(encoded),
        "seconds": seconds,
        "prompts_per_s": len(encoded) / seconds,
    }
    (out / "timing.json").write_text(json.dumps(timing))
    print(f"{loaded.engine}: {len(encoded)} prompts in {seconds:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `scripts/spike/dump_native.py`**

This script runs only in the Spark's own vLLM venv. That venv is mechanica's serving engine (vLLM 0.30.0) and has no sonde install, so the script imports nothing from sonde.

```python
"""Captures the spike's token ids with vLLM's native extract_hidden_states.

This is mechanica's serving capture: aux id k is the stream after k blocks,
so sonde block b is aux id b + 1. Run in the Spark's own vLLM venv
(~/experiments/vllm/.venv), never in a sonde venv:
  python scripts/spike/dump_native.py --out OUT.safetensors
  VLLM_USE_V2_MODEL_RUNNER=0 python scripts/spike/dump_native.py --eager
      --out OUT.safetensors
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

import torch
import transformers
import vllm
from safetensors import torch as st
from vllm.config import kv_transfer
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as hsc,
)

MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--mem-gb", type=float, default=16)
    parser.add_argument("--tokens", default="../outputs/spike/tokens.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.tokens) as f:
        prompts = [json.loads(line)["ids"] for line in f]
    n = transformers.AutoConfig.from_pretrained(MODEL).num_hidden_layers
    aux = list(range(1, n + 1))
    slabs = []
    with tempfile.TemporaryDirectory(dir="/dev/shm") as shm:
        llm = vllm.LLM(
            model=MODEL,
            dtype="bfloat16",
            enforce_eager=args.eager,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            gpu_memory_utilization=args.mem_gb / 121,
            max_model_len=4096,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {"eagle_aux_hidden_state_layer_ids": aux}
                },
            },
            kv_transfer_config=kv_transfer.KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={"shared_storage_path": shm},
            ),
        )
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in prompts],
            vllm.SamplingParams(temperature=0.0, max_tokens=1),
        )
        for ids, output in zip(prompts, outputs, strict=True):
            path = output.kv_transfer_params["hidden_states_path"]
            slab = hsc.load_hidden_states(path)
            if slab["token_ids"].tolist() != ids:
                raise ValueError(f"slab token ids differ from request: {path}")
            slabs.append(slab["hidden_states"].cpu())
            hsc.cleanup_hidden_states(path)
    flat = torch.cat(slabs)
    tensors = {f"B{a - 1}": flat[:, j].contiguous() for j, a in enumerate(aux)}
    tensors["offsets"] = torch.tensor([0] + [len(p) for p in prompts]).cumsum(0)
    st.save_file(tensors, args.out)
    meta = {"engine": f"vllm=={vllm.__version__}+native", "eager": args.eager}
    pathlib.Path(args.out).with_suffix(".json").write_text(json.dumps(meta))
    print(f"wrote {len(prompts)} prompts x {len(aux)} blocks to {args.out}")


if __name__ == "__main__":
    main()
```

The eager run uses `VLLM_USE_V2_MODEL_RUNNER=0` with `enforce_eager`. That is the same runner and mode as nnsight's engine, so the only difference is the capture path. The compiled run is mechanica's serving default.

- [ ] **Step 4: Write `scripts/spike/compare.py`**

The doc template lives in this script, and every number in `docs/parity-2026-10.md` comes from it.

```python
"""Writes docs/parity-2026-10.md from the phase-5 spike outputs.

Run in .venv-hf from the repo root, after scripts/spike/run.sh's dumps:
  python scripts/spike/compare.py --commit SHA
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re

import safetensors
import torch

from sonde import store

PAIRS = (
    ("hf ↔ vllm bf16", "hf-bf16", "vllm-bf16"),
    ("hf ↔ vllm fp32", "hf-fp32", "vllm-fp32"),
    ("native eager ↔ vllm", "native-eager", "vllm-bf16"),
    ("native compiled ↔ vllm", "native-compiled", "vllm-bf16"),
)
THROUGHPUT = ("tp-hf", "tp-vllm")


def delta(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    """Returns (max_t |Δ_t| / |x_t|, mean |Δ|) for [ΣT, H] rows, x = ref."""
    ref, got = ref.float(), got.float()
    d = got - ref
    rel = (d.norm(dim=-1) / ref.norm(dim=-1)).max().item()
    return rel, d.abs().mean().item()


def _present(root: pathlib.Path, name: str) -> bool:
    return (root / name / "manifest.json").exists() or (
        root / f"{name}.safetensors"
    ).exists()


def _read(
    root: pathlib.Path, name: str, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    native = root / f"{name}.safetensors"
    if native.exists():
        with safetensors.safe_open(str(native), "pt") as f:
            return f.get_tensor(f"B{block}"), f.get_tensor("offsets")
    x, offsets = store.read_layer(root / name, block)
    if offsets is None:
        raise ValueError(f"{name} was not extracted with keep: tokens")
    return x, offsets


def _engine(root: pathlib.Path, name: str) -> str:
    meta = root / f"{name}.json"
    if meta.exists():
        return json.loads(meta.read_text())["engine"]
    return store.read_manifest(root / name)["engine"]


def _cell(root: pathlib.Path, left: str, right: str, block: int) -> str:
    if not (_present(root, left) and _present(root, right)):
        return "not run | not run"
    a, a_off = _read(root, left, block)
    b, b_off = _read(root, right, block)
    if not torch.equal(a_off.long(), b_off.long()):
        raise ValueError(f"{left} and {right} hold different token rows")
    rel, mean = delta(a, b)
    return f"{rel:.2e} | {mean:.2e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--root", default="../outputs/spike")
    parser.add_argument("--doc", default="docs/parity-2026-10.md")
    args = parser.parse_args()
    root = pathlib.Path(args.root)
    manifest = store.read_manifest(root / "hf-bf16")
    blocks = manifest["blocks"]
    pin = re.search(
        r'"vllm==([^";]+)', pathlib.Path("pyproject.toml").read_text()
    )
    names = [n for _, a, b in PAIRS for n in (a, b)] + list(THROUGHPUT)
    present = [n for n in dict.fromkeys(names) if _present(root, n)]
    engines = {n: _engine(root, n) for n in present}
    versions = {
        e.split("+", 1)[0] for e in engines.values() if e.startswith("vllm==")
    }

    lines = [
        "# hf ↔ vLLM parity, 2026-10",
        "",
        f"Generated by `scripts/spike/compare.py` on "
        f"{datetime.date.today().isoformat()} from sonde `{args.commit}`. "
        "Do not edit by hand: re-run `scripts/spike/run.sh` on the Spark.",
        "",
        f"Model `{manifest['model']}`, {len(manifest['ids'])} prompts (chat "
        "render of `experiments/refusal_probing/data/labeled.jsonl`), every "
        "block, every prompt token, identical token ids on every side.",
        "",
        "`rel` is max over tokens of ‖Δ‖₂ / ‖x‖₂ with x the left run; "
        "`mean` is mean |Δ|. mechanica's engine-axis band is relative "
        "max|Δ| 1e-5 to 4e-4 on mid layers (Qwen3-8B, vLLM 0.19 + "
        "vllm-lens vs HF).",
        "",
        f"Pinned: `vllm=={pin.group(1).strip() if pin else '?'}` "
        "(pyproject `[vllm]` extra).",
        "",
        "## Smoke: nnsight tests/vllm at b717807",
        "",
        "| vLLM | result |",
        "|---|---|",
    ]
    for path in sorted(root.glob("smoke-vllm-*.txt")):
        last = path.read_text().strip().splitlines()[-1]
        lines.append(f"| {path.stem.removeprefix('smoke-vllm-')} | {last} |")
    lines += ["", "## Engines", "", "| run | engine |", "|---|---|"]
    lines += [f"| {n} | `{e}` |" for n, e in engines.items()]
    if len(versions) > 1:
        lines += [
            "",
            "The vLLM versions differ, so the native columns mix "
            "engine version and capture path.",
        ]
    lines += ["", "## Activations", ""]
    lines.append(
        "| block | "
        + " | ".join(f"{title} rel | mean" for title, _, _ in PAIRS)
        + " |"
    )
    lines.append("|---|" + "---|---|" * len(PAIRS))
    for block in blocks:
        cells = [_cell(root, a, b, block) for _, a, b in PAIRS]
        lines.append(f"| {block} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Throughput: 10k prompts, mean pooling, 3 blocks",
        "",
        "| run | engine | prompts | seconds | prompts/s |",
        "|---|---|---|---|---|",
    ]
    for name in THROUGHPUT:
        timing = root / name / "timing.json"
        if not timing.exists():
            lines.append(f"| {name} | not run | | | |")
            continue
        t = json.loads(timing.read_text())
        lines.append(
            f"| {name} | `{t['engine']}` | {t['prompts']} | "
            f"{t['seconds']:.1f} | {t['prompts_per_s']:.1f} |"
        )
    pathlib.Path(args.doc).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
```

The pin regex stops at `;` because the `[vllm]` extra now carries a `; sys_platform == 'linux'` marker.

- [ ] **Step 5: Write `scripts/spike/run.sh`**

```bash
#!/usr/bin/env bash
# The phase-5 parity spike, on the Spark, from ~/experiments/sonde-spike/code.
# Steps run one at a time so the throughput rows never share the GPU, and a
# failed step (e.g. vLLM fp32) does not stop the rest. Usage: run.sh COMMIT
set -u
out=../outputs/spike
hf=../.venv-hf/bin/python
vl=../.venv-vllm/bin/python
native=~/experiments/vllm/.venv/bin/python
spike=scripts/spike
$hf $spike/tokens.py || exit 1
$hf $spike/dump_sonde.py --backend hf --batch-size 64 --out $out/hf-bf16
$hf $spike/dump_sonde.py --backend hf --dtype float32 --batch-size 64 \
  --out $out/hf-fp32
$vl $spike/dump_sonde.py --backend vllm --batch-size 500 --out $out/vllm-bf16
$vl $spike/dump_sonde.py --backend vllm --dtype float32 --batch-size 500 \
  --out $out/vllm-fp32
$native $spike/dump_native.py --out $out/native-compiled.safetensors
VLLM_USE_V2_MODEL_RUNNER=0 $native $spike/dump_native.py --eager \
  --out $out/native-eager.safetensors
$hf $spike/dump_sonde.py --backend hf --keep pooled --blocks 7,14,21 \
  --repeat 40 --batch-size 64 --out $out/tp-hf
$vl $spike/dump_sonde.py --backend vllm --keep pooled --blocks 7,14,21 \
  --repeat 40 --batch-size 500 --out $out/tp-vllm
$hf $spike/compare.py --commit "$1"
```

The 10k-prompt throughput rows are the 256 prompts repeated 40 times (10,240 prompts). With prefix caching off, every repeat is a full prefill.

- [ ] **Step 6: Smoke the scripts on the Mac's CPU**

This check uses the hf backend only. An fp32 dump stands in for "hf-bf16" and a bf16 dump stands in for "vllm-bf16", so `compare.py` runs end to end on real shards.

```bash
d=/tmp/sonde-spike-cpu; rm -rf $d; mkdir -p $d
uv run --extra hf --extra dev python scripts/spike/tokens.py --limit 4 --out $d/tokens.jsonl
uv run --extra hf --extra dev python scripts/spike/dump_sonde.py --backend hf --dtype float32 --blocks 0,27 --tokens $d/tokens.jsonl --out $d/hf-bf16
uv run --extra hf --extra dev python scripts/spike/dump_sonde.py --backend hf --blocks 0,27 --tokens $d/tokens.jsonl --out $d/vllm-bf16
uv run --extra hf --extra dev python scripts/spike/compare.py --commit cpu-check --root $d --doc $d/parity.md
grep -A3 '^| block' $d/parity.md
grep '^Pinned' $d/parity.md
bash -n scripts/spike/run.sh
```

Expected:
- `wrote 4 prompts to /tmp/sonde-spike-cpu/tokens.jsonl`.
- Two `hf==...: 4 prompts in ...s` lines.
- `wrote /tmp/sonde-spike-cpu/parity.md`.
- Block rows of the form `| 0 | <x>e-0<y> | <x>e-0<y> | not run | not run | not run | not run | not run | not run |`, and the same for block 27. Because this compares fp32 against bf16, the first `rel` lies between 1e-4 and 1e-1.
- ``Pinned: `vllm==<Task 5.1's version>` (pyproject `[vllm]` extra).``
- `bash -n` prints nothing.

- [ ] **Step 7: Lint, type-check and commit the scripts**

```bash
uv run --extra hf --extra dev ruff check --fix scripts/spike
uv run --extra hf --extra dev ruff format scripts/spike
uv run --extra hf --extra dev pyright scripts/spike
git add scripts/spike
git commit -m "feat(spike): hf/vllm/native parity and throughput scripts"
```

Expected: ruff reports `All checks passed!` after `--fix`, which only reorders imports if phase 1's isort settings differ. pyright reports `0 errors`.

- [ ] **Step 8: Run the spike on the Spark**

The job declares 24 GB. vLLM gets `gpu_memory_utilization = 16/121`, so no step uses more than 16 GB.

```bash
git status --porcelain
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 24 "bash scripts/spike/run.sh $(git rev-parse --short HEAD)"
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 120
```

Expected:
- `git status --porcelain` prints nothing.
- The log has one `<engine>: N prompts in Ts` line per sonde dump. N is 256 for `hf-bf16`, `hf-fp32`, `vllm-bf16` and `vllm-fp32`, and 10240 for `tp-hf` and `tp-vllm`.
- The log has two `wrote 256 prompts x 28 blocks` lines and ends with `wrote docs/parity-2026-10.md`.
- A failed step shows its traceback in the log and appears in the doc as `not run`. The most likely one is `vllm-fp32`, if GB10 has no fp32 attention backend. Name the reason in the PR description, not in the doc.

- [ ] **Step 9: Fetch and read the doc**

```bash
~/.claude/skills/dgx-spark/scripts/spark fetch sonde-spike code/docs/parity-2026-10.md docs/
sed -n '1,80p' docs/parity-2026-10.md
```

Expected:
- The doc has the smoke rows from Task 5.1, an engines table and one Activations row for each block 0–27.
- `hf ↔ vllm bf16 rel` is well under 0.1 on every block.
- `native eager ↔ vllm` should be at or near 0, because it uses the same runner and kernels. That expectation comes from reading source, not from a measurement, so a larger value is a finding for the PR description.
- If `native ↔ vllm` `rel` is 0.1 or more at mid blocks, the block index is off by one. Stop and debug `vllm_forward` before Task 5.5.
- The test pools over far fewer tokens, so `TOL` is usually well clear of the spike's number. If `TOL` in `tests/test_vllm.py` is below twice the block-14 `hf ↔ vllm bf16 rel`, raise it to that value and rerun Task 5.3 Step 5.

- [ ] **Step 10: Commit the doc**

```bash
git add docs/parity-2026-10.md tests/test_vllm.py
git commit -m "docs: hf/vLLM/native parity and throughput from the Spark spike"
```

---

### Task 5.5: Re-run both recipes on the Spark and record RESULTS.md

**Files:**
- Create: `scripts/spike/results.py`, `experiments/refusal_probing/RESULTS.md`
- Modify: `experiments/detecting_high_stakes/RESULTS.md`

**Interfaces:**
- Consumes:
  - phase 3's `sonde run <recipe> [-o k=v]...`;
  - a run dir layout of `extract/manifest.json` (keys `engine`, `model`), `metrics.json` and `score/<entry>/metrics.json`;
  - phase 4's recipes `refusal` and `high_stakes`, and its rewrite of `experiments/detecting_high_stakes/RESULTS.md` (How to Run, references);
  - Task 5.1's pin and venvs.
- Produces: `results.render(run_dir: pathlib.Path, commit: str, cmd: str) -> str` and `results.splice(text: str, block: str) -> str`. The block sits between `<!-- sonde:results -->` and `<!-- /sonde:results -->`.

- [ ] **Step 1: Write `scripts/spike/results.py`**

```python
"""Splices a sonde run's measured results into an experiment's RESULTS.md.

Run on the Mac after fetching the run dir (shards excluded):
  uv run python scripts/spike/results.py RUN_DIR --commit SHA --cmd CMD
      --into experiments/<experiment>/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import pathlib

BEGIN = "<!-- sonde:results -->"
END = "<!-- /sonde:results -->"


def render(run_dir: pathlib.Path, commit: str, cmd: str) -> str:
    """Returns the marked markdown block for one run dir."""
    manifest = json.loads((run_dir / "extract" / "manifest.json").read_text())
    engine = manifest["engine"]
    lines = [
        BEGIN,
        f"## Results: sonde `{commit}`, backend `{engine.split('==', 1)[0]}`",
        "",
        "Generated by `scripts/spike/results.py` from the run's JSON files.",
        "",
        f"- engine: `{engine}`",
        f"- model: `{manifest['model']}`",
        f"- command: `{cmd}`",
        "",
    ]
    paths = [
        run_dir / "metrics.json",
        *sorted(run_dir.glob("score/*/metrics.json")),
    ]
    for path in paths:
        body = json.dumps(json.loads(path.read_text()), indent=1)
        lines += [
            f"### `{path.relative_to(run_dir)}`",
            "",
            "```json",
            body,
            "```",
            "",
        ]
    return "\n".join([*lines, END])


def splice(text: str, block: str) -> str:
    """Replaces the marked block in text, or inserts it after the title."""
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        return head + block + rest.split(END, 1)[1]
    title, _, body = text.partition("\n")
    return f"{title}\n\n{block}\n{body}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--into", required=True)
    args = parser.parse_args()
    target = pathlib.Path(args.into)
    block = render(pathlib.Path(args.run_dir), args.commit, args.cmd)
    target.write_text(splice(target.read_text(), block))
    print(f"updated {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check that it splices idempotently on a synthetic run dir**

```bash
d=/tmp/sonde-results-check; rm -rf $d; mkdir -p $d/run/extract $d/run/score/x
echo '{"engine": "vllm==0.30.0+nnsight==b717807", "model": "m"}' > $d/run/extract/manifest.json
echo '{"test": {"auroc": 0.9}}' > $d/run/metrics.json
echo '{"auroc": 0.8}' > $d/run/score/x/metrics.json
printf '# Title\n\nbody\n' > $d/R.md
uv run --extra hf --extra dev python scripts/spike/results.py $d/run --commit a --cmd c --into $d/R.md
uv run --extra hf --extra dev python scripts/spike/results.py $d/run --commit b --cmd c --into $d/R.md
grep -c '<!-- sonde:results -->' $d/R.md; grep -c 'sonde `b`' $d/R.md; tail -n 1 $d/R.md
```

Expected: `updated ...` twice, then `1`, `1` and `body`.

- [ ] **Step 3: Create the refusal RESULTS.md and commit the tooling**

Create `experiments/refusal_probing/RESULTS.md`:

```markdown
# Refusal probe results

Recipe `sonde/recipes/refusal.yaml` on `data/labeled.jsonl` (256 rows). The labels came from raw-format continuations, so the recipe uses `format: raw`; `group: original_prompt` keeps duplicate prompts inside one split.
```

```bash
uv run --extra hf --extra dev ruff check --fix scripts/spike/results.py
uv run --extra hf --extra dev ruff format scripts/spike/results.py
git add scripts/spike/results.py experiments/refusal_probing/RESULTS.md
git commit -m "feat(spike): results.py splices measured run results into RESULTS.md"
```

- [ ] **Step 4: Check that the Llama weights are reachable on the Spark**

```bash
ssh 10.97.110.177 'ls ~/experiments/hf-cache/hub 2>/dev/null | grep -i "llama-3.1-8b"; ls ~/experiments/hf-cache/token 2>/dev/null'
```

Expected: either both `models--meta-llama--Llama-3.1-8B` and `models--meta-llama--Llama-3.1-8B-Instruct`, or a `token` file. If neither appears, stop. Ask the user to log in on the Spark with a token that has Llama 3.1 access (`HF_HOME=~/experiments/hf-cache huggingface-cli login`).

- [ ] **Step 5: Dry-run both recipes on the Spark**

Use `.venv-vllm` if Task 5.1 pinned a clean vLLM. Otherwise use `.venv-hf` and drop the `gpu_memory_utilization` override.

```bash
git status --porcelain
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
ssh 10.97.110.177 'cd ~/experiments/sonde-spike/code && for r in refusal high_stakes; do ../.venv-vllm/bin/sonde run $r --dry-run -o name=$r -o output.dir=../outputs/runs -o model.vllm.gpu_memory_utilization=0.33 | grep -E "backend|gpu_memory_utilization"; done'
```

Expected: `git status --porcelain` prints nothing. For each recipe, the dry run prints `backend: vllm` and `gpu_memory_utilization: 0.33` (40 GB of 121), then exits 0 without loading weights.

- [ ] **Step 6: Run both recipes**

Each job declares 48 GB: 40 GB for the engine, plus room for probe training in the client process. The ledger runs the jobs one at a time.

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S run sonde-spike 48 '../.venv-vllm/bin/sonde run refusal -o name=refusal -o output.dir=../outputs/runs -o model.vllm.gpu_memory_utilization=0.33'
$S run sonde-spike 48 '../.venv-vllm/bin/sonde run high_stakes -o name=high_stakes -o output.dir=../outputs/runs -o model.vllm.gpu_memory_utilization=0.33'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S status
$S logs sonde-spike 30
```

Expected: `pueue wait` returns once the second job finishes. `status` shows both `sonde-spike` tasks as `Success`, and the log ends with `[spark] exit code 0`. If the refusal task failed, find its log with `ssh 10.97.110.177 'ls -t ~/experiments/sonde-spike/runs/*.log'`, fix the cause and rerun only that recipe.

- [ ] **Step 7: Fetch the run dirs without shards and splice the results**

Use the commit that was synced in Step 5. Nothing has been committed since then.

```bash
rm -rf /tmp/sonde-runs; mkdir -p /tmp/sonde-runs
rsync -az --exclude '*.safetensors' 10.97.110.177:experiments/sonde-spike/outputs/runs/ /tmp/sonde-runs/
C=$(git rev-parse --short HEAD)
uv run --extra hf --extra dev python scripts/spike/results.py /tmp/sonde-runs/refusal --commit $C --cmd "sonde run refusal -o model.vllm.gpu_memory_utilization=0.33" --into experiments/refusal_probing/RESULTS.md
uv run --extra hf --extra dev python scripts/spike/results.py /tmp/sonde-runs/high_stakes --commit $C --cmd "sonde run high_stakes -o model.vllm.gpu_memory_utilization=0.33" --into experiments/detecting_high_stakes/RESULTS.md
grep -n '^## Results: sonde' experiments/refusal_probing/RESULTS.md experiments/detecting_high_stakes/RESULTS.md
```

Expected: `updated ...` twice, then one matching line per file that names the commit and backend `vllm`. Under the Task 5.1 fallback the backend is `hf`, and the `--cmd` strings drop the override.

- [ ] **Step 8: Mark the old high_stakes results as legacy**

Phase 4 Task 4.2 Step 3 already rewrote `## How to Run` and removed the research_gaps reference. The one edit left is in `experiments/detecting_high_stakes/RESULTS.md`. Replace

```markdown
## Results

### Run v1: Last-Token Probing (8K samples, 17 layers)
```

with

```markdown
## Legacy results (v0.1-legacy)

These came from the pre-redesign pipeline at `batch_size: 1`; the generated block above supersedes them and is the one to compare against.

### Run v1: Last-Token Probing (8K samples, 17 layers)
```

Run: `grep -nE "kubectl|run_pipeline|research_gaps" experiments/detecting_high_stakes/RESULTS.md`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add experiments/refusal_probing/RESULTS.md experiments/detecting_high_stakes/RESULTS.md
git commit -m "docs: re-run refusal and high_stakes on the Spark; RESULTS record commit and backend"
```

- [ ] **Step 10: Clean the Spark**

Everything phase 6 needs is now in the repo: the parity doc, both RESULTS.md files, and `venvs.sh`, which rebuilds the venvs.

```bash
S=~/.claude/skills/dgx-spark/scripts/spark
$S clean sonde-spike
$S check
```

Expected: `removed ~/experiments/sonde-spike`, then `check` lists no leftover experiment dirs and no queued `sonde-spike` jobs. Ask the user whether to run `spark clean --models`, which deletes the cached Qwen3-0.6B, gpt2 and Llama-3.1-8B weights.

<!-- skipped: one fix applied in part. The "use `model: Any = loaded.model`" suggestion (Task 5.2/5.3 imports issue) was not applied as written: phase 2 already declares `Loaded.model: Any`, so vllm_forward uses plain `model = loaded.model`, which matches phase 2's forward and needs no annotation. The rest of that issue was applied: no `import typing`, no `import importlib.metadata`, and `metadata.distribution` in `_nnsight_ref`. The redundant local `import nnsight` in vllm_forward was also dropped, because phase 2 imports it at module level. All other issues (several were duplicates) were applied. Two follow-on changes: compare.py's pin regex now stops at `;` because of the new platform marker, and the Task 5.1 Step 10 expectation no longer hard-codes torch 2.13.0 for vLLM. -->

---

## Phase 6: Generation and steering

**Goal.** Add generation and steering as first-class steps (spec §7 "Generation", §10, §16 phase 6). `backends.generate` gets an hf path and a vllm path. The hf path left-pads, builds its `GenerationConfig` only from sonde config and seeds every batch. The vllm path passes explicit sampling params and always sets `edits=[...]`. Steering writes block `L`'s residual inside `tracer.all()`, so it covers prefill and every decode step. `generate.py` writes `generations.jsonl`, appending rows and skipping ids already present. `steer.py` writes one `steer/s{strength}.jsonl` per strength, and strength 0 runs with no edit. The runner registers both steps, and the README and CHANGELOG document them. **Done when:** `uv run pytest tests/test_steer.py -m "not gpu" -v` passes (8 tests on gpt2, hf); `uv run pytest -m "not gpu" -q` passes; `sonde run quickstart` with `steps: [generate, steer]` writes `generations.jsonl` and `steer/s*.jsonl` on CPU (Task 6.5 Step 3); and the `@pytest.mark.gpu` vllm steer test passes on the Spark in `.venv-vllm`.

**Verified on 2026-10-02** in a scratch venv (nnterp 1.3.0, nnsight 0.7.0, transformers 5.18.0, torch 2.14.1, gpt2). The planned `_hf_generate`, `generate.py`, `steer.py` (guard removed) and the full `tests/test_steer.py` were run against stand-in versions of `data`, `probe`, `config` and `runner`. Result: `8 passed, 1 deselected`. They are also clean under ruff (line length 80) and pyright (basic).
1. A `{input_ids, attention_mask}` dict passed to `tracer.invoke(...)` is used exactly as given, because nnsight wraps it in a `BatchEncoding` and does not re-pad it. Left-padded batched greedy output equals the output of generating one prompt at a time.
2. Calling `model.generate(dict, ...)` without a `with` block skips nnsight's input preparation and fails with `'dict' object has no attribute 'shape'`. Always use the `with` form.
3. In nnsight 0.7, `tracer.all()` runs to the end of generation, and any code after the loop in the same invoke is dropped. So `tracer.result.save()` goes in a second, empty `tracer.invoke()`. A steered call warns "`...output.i8` was not provided"; this is harmless.
4. Assigning `model.layers_output[L] = ...` inside `tracer.all()` under `torch.no_grad()` edits the prefill and every decode step. A forward pre-hook on block L+1 ran 8 times for `max_new_tokens=8`. After ablation, |x·v| was ≤ 4e-5; without ablation it was about 158.
5. nnterp loads lazily: the device is `meta` until the first trace, and dispatch then swaps in new modules. Torch hooks on `model.layers[i]._module` therefore have to be registered after one generate call.
6. transformers 5 fills every `GenerationConfig` field left as `None` from the checkpoint's `generation_config` (`_prepare_generation_config`). Its global default is `top_k=50`, so sampling needs an explicit `top_k=0`. With `do_sample=False`, passing `top_k` only triggers a "generation flags are not valid" warning. **Deviation from the spec wording:** `top_k=0` is passed only when sampling. Greedy decoding is unaffected.
7. Calling `torch.manual_seed(seed)` before the call makes sampled output reproducible, and a different seed changes it. `model.generation_config.eos_token_id` can be reached through the nnterp envoy. The ruff SIM117 form `with torch.no_grad(), model.generate(...) as tracer:` is captured correctly.
8. vLLM could not be run here. The following come from nnsight source at b717807: `VLLM.edit(*, name=None, inplace=True, ...)` yields `(tracer, edit)`; a plain `model.generate(prompts, **kw)` builds `SamplingParams(**kw)` with every field explicit; `edits=[...]` is routed to `extra_args["nnsight_edits"]`. The steering block follows the research doc's §5. **Not verified:** whether vLLM's `outputs[0].token_ids` include the EOS id. The hf path keeps EOS so the two backends match.
9. Inputs stay on the CPU on purpose. On the Mac, nnterp dispatched gpt2 to `mps:0`, and CPU inputs generated correctly, with the steered and unsteered tests all passing. The move is done by transformers 5's `prepare_inputs_for_generation` (step 6, "Move the tensors the forward consumes onto the model device"), which works the same way on CUDA. Calling `.to(model.device)` before the first trace moves the inputs to `meta`, and the first call fails with `Cannot copy out of meta tensor`.

**Deviations from the spec wording:**
- A `generations.jsonl` row is the source row (its configured keys and raw values) plus `id`, `response` and `response_ids`. This covers every key the spec lists, and the file can be reloaded with only `response: response` changed.
- The tests steer at strength 100 (gpt2) and 1000 (vLLM), not at spec §14's strength 1. A unit vector at strength 1, against a residual norm of about 10², does not change greedy output.

---

### Task 6.1: hf `backends.generate` with per-step steering

**Files:**
- Modify: `sonde/backends.py` (add `Steering`, `generate`, `_hf_generate`)
- Create: `tests/test_steer.py`

**Interfaces:**
- Consumes:
  - `backends.Loaded(backend, model, tokenizer, num_layers, hidden, engine)`
  - `backends.load_model(cfg: config.ModelConfig) -> Loaded`, which is cached and returns an nnterp `StandardizedTransformer` as `.model` for `backend="hf"`
  - `config.ModelConfig(name, dtype, backend, vllm)`
  - `config.GenerateConfig(max_tokens: int = 256, temperature: float = 0.7, top_p: float = 1.0)`
- Produces:
  - `backends.Steering(block: int, vector: torch.Tensor, mode: str, strength: float)`
  - `backends.generate(loaded: Loaded, prompts: list[list[int]], gen: config.GenerateConfig, seed: int, steering: Steering | None = None) -> list[list[int]]`, which returns response ids up to and including the first EOS

- [ ] **Step 1: Write the failing tests**

Create `tests/test_steer.py`:

```python
from __future__ import annotations

from typing import Any

import torch

from sonde import backends
from sonde import config

BLOCK = 5
GREEDY = config.GenerateConfig(max_tokens=8, temperature=0.0)
TEXTS = ["The cat sat on the", "Hello, my name is", "In 1999 the"]


def _gpt2() -> tuple[backends.Loaded, list[list[int]], torch.Tensor]:
    loaded = backends.load_model(
        config.ModelConfig(name="gpt2", dtype="float32")
    )
    tok: Any = loaded.tokenizer
    v = torch.randn(loaded.hidden, generator=torch.Generator().manual_seed(0))
    return loaded, [tok(t).input_ids for t in TEXTS], v / v.norm()


def test_hf_strength_zero_equals_unsteered():
    loaded, prompts, v = _gpt2()
    plain = backends.generate(loaded, prompts, GREEDY, 0)
    zero = backends.Steering(BLOCK, v, "add", 0.0)
    assert backends.generate(loaded, prompts, GREEDY, 0, zero) == plain
    assert [len(r) for r in plain] == [8, 8, 8]


def test_hf_batched_equals_one_by_one():
    loaded, prompts, _ = _gpt2()
    batched = backends.generate(loaded, prompts, GREEDY, 0)
    single = [backends.generate(loaded, [p], GREEDY, 0)[0] for p in prompts]
    assert single == batched


def test_hf_add_changes_output_and_does_not_leak():
    loaded, prompts, v = _gpt2()
    plain = backends.generate(loaded, prompts, GREEDY, 0)
    add = backends.Steering(BLOCK, v, "add", 100.0)
    assert backends.generate(loaded, prompts, GREEDY, 0, add) != plain
    assert backends.generate(loaded, prompts, GREEDY, 0) == plain


def test_hf_ablate_removes_projection_at_every_step():
    loaded, prompts, v = _gpt2()
    backends.generate(loaded, prompts, GREEDY, 0)
    seen = []

    def grab(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        seen.append((h.float() @ v.to(h.device)).abs().max().item())

    model: Any = loaded.model
    nxt = model.layers[BLOCK + 1]._module
    hook = nxt.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        backends.generate(loaded, prompts, GREEDY, 0)
        plain = list(seen)
        seen.clear()
        ablate = backends.Steering(BLOCK, v, "ablate", 1.0)
        backends.generate(loaded, prompts, GREEDY, 0, ablate)
    finally:
        hook.remove()
    assert len(seen) == len(plain) == 8
    assert min(plain) > 1.0
    assert max(seen) < 1e-3


def test_hf_sampling_is_seeded():
    loaded, prompts, _ = _gpt2()
    hot = config.GenerateConfig(max_tokens=8, temperature=1.0, top_p=0.95)
    first = backends.generate(loaded, prompts, hot, 3)
    assert backends.generate(loaded, prompts, hot, 3) == first
    assert backends.generate(loaded, prompts, hot, 4) != first
```

The ablation test calls `generate` once before registering the hook because nnterp dispatches lazily (verified fact 5). Block 6's input is block 5's output in gpt2, so the pre-hook reads the steered residual at every forward pass: the prefill plus 7 decode steps, 8 passes in total.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_steer.py -v`
Expected: 5 failed, each with `AttributeError: module 'sonde.backends' has no attribute 'generate'`.

- [ ] **Step 3: Implement**

In `sonde/backends.py`, make sure the import block contains these lines alongside phase 2's imports:

```python
import dataclasses
from typing import Any

import torch
import transformers

from sonde import config
```

Append:

```python
@dataclasses.dataclass
class Steering:
    """A residual-stream edit at one block, applied at every forward pass.

    Attributes:
        block: Decoder block whose output is edited (spec §5 numbering).
        vector: [H] unit-norm direction v.
        mode: "add" (x += s·v) or "ablate" (x -= s·(x·v)·v).
        strength: The multiple s.
    """

    block: int
    vector: torch.Tensor
    mode: str
    strength: float


def generate(
    loaded: Loaded,
    prompts: list[list[int]],
    gen: config.GenerateConfig,
    seed: int,
    steering: Steering | None = None,
) -> list[list[int]]:
    """Generates one response per prompt.

    Args:
        loaded: The model from `load_model`.
        prompts: Token ids per prompt, generation prompt included.
        gen: Sampling settings; temperature 0 is greedy.
        seed: Sampler seed for this call.
        steering: Edit applied at every forward pass (prefill and each
            decode step), or None for plain generation.

    Returns:
        Generated token ids per prompt, in prompt order, up to and including
        the first EOS.
    """
    if loaded.backend != "hf":
        raise NotImplementedError(f"generate on {loaded.backend}")
    return _hf_generate(loaded, prompts, gen, seed, steering)


def _hf_generate(
    loaded: Loaded,
    prompts: list[list[int]],
    gen: config.GenerateConfig,
    seed: int,
    steering: Steering | None,
) -> list[list[int]]:
    model: Any = loaded.model
    tok: Any = loaded.tokenizer
    n = max(len(p) for p in prompts)
    pad = tok.pad_token_id or 0
    inputs = {
        "input_ids": torch.tensor([[pad] * (n - len(p)) + p for p in prompts]),
        "attention_mask": torch.tensor(
            [[0] * (n - len(p)) + [1] * len(p) for p in prompts]
        ),
    }
    sampling = gen.temperature > 0
    knobs = {"temperature": gen.temperature, "top_p": gen.top_p, "top_k": 0}
    # ponytail: transformers fills fields left unset here from the
    # checkpoint's generation_config (e.g. repetition_penalty); pin one here
    # if a recipe model ships it.
    gc = transformers.GenerationConfig(
        do_sample=sampling,
        max_new_tokens=gen.max_tokens,
        **(knobs if sampling else {}),
    )
    torch.manual_seed(seed)
    with torch.no_grad(), model.generate(generation_config=gc) as tracer:
        with tracer.invoke(inputs):
            if steering is not None:
                b, s = steering.block, steering.strength
                for _ in tracer.all():
                    h = model.layers_output[b]
                    v = steering.vector.to(h.device, h.dtype)
                    if steering.mode == "add":
                        model.layers_output[b] = h + s * v
                    else:
                        model.layers_output[b] = h - s * (h @ v)[..., None] * v
        with tracer.invoke():
            out = tracer.result.save()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    responses = []
    for row in out[:, n:].tolist():
        end = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
        responses.append(row[:end])
    return responses
```

`pad` can be any id because the mask hides it, which is why `or 0` is safe even when the pad id is `0`. The inputs are built on the CPU and left there, because transformers moves them to the model device (verified fact 9). Spec §7 requires `torch.no_grad()` rather than `inference_mode`, because nnsight's worker thread writes in place.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_steer.py -v`
Expected: 5 passed. A `UserWarning: Execution complete but ... was not provided` is expected (verified fact 3).

- [ ] **Step 5: Commit**

```bash
git add sonde/backends.py tests/test_steer.py
git commit -m "feat(backends): hf generate with per-step add/ablate steering"
```

---

### Task 6.2: `generate` step (`generations.jsonl`, resume, exact reload)

**Files:**
- Create: `sonde/generate.py`
- Modify: `tests/test_steer.py`

**Interfaces:**
- Consumes:
  - `backends.generate`, `backends.Steering` and `backends.Loaded` from Task 6.1
  - `data.Sample(id: str, text, messages, response, response_ids: list[int] | None, label, group, raw: dict)`
  - `data.load_samples(src: config.DataConfig, seed: int) -> list[Sample]`, which phase 2 has fill `response_ids` from the row when `src.response` is set
  - `data.prompt_ids(sample, src, tokenizer) -> list[int]`
  - `data.render(sample, src, window, tokenizer) -> Encoded`, which phase 2 has use `sample.response_ids` when present
  - `config.RunConfig`, `config.DataConfig`
- Produces:
  - `generate.BATCH: int`
  - `generate.write_rows(loaded: backends.Loaded, samples: list[data.Sample], src: config.DataConfig, gen: config.GenerateConfig, seed: int, path: pathlib.Path, steering: backends.Steering | None = None) -> None`
  - `generate.run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None`, which writes `run_dir/"generations.jsonl"`

- [ ] **Step 1: Write the failing tests**

In `tests/test_steer.py`, replace the import block with:

```python
from __future__ import annotations

import json
from typing import Any

import torch

from sonde import backends
from sonde import config
from sonde import data
from sonde import generate
```

After `_gpt2`, add the helpers:

```python
def _write_rows(path, rows) -> None:
    path.write_text("".join(f"{json.dumps(r)}\n" for r in rows))


def _read_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _rows() -> list[dict]:
    return [
        {"id": k, "text": t, "label": i % 2}
        for i, (k, t) in enumerate(zip("abc", TEXTS, strict=True))
    ]
```

Append the tests:

```python
def test_generate_resume_skips_done_ids(tmp_path):
    _write_rows(tmp_path / "in.jsonl", _rows())
    cfg = config.RunConfig.model_validate(
        {
            "name": "g",
            "model": {"name": "gpt2", "dtype": "float32"},
            "data": {"path": str(tmp_path / "in.jsonl"), "format": "raw"},
            "generate": {"max_tokens": 4, "temperature": 0.0},
            "steps": ["generate"],
        }
    )
    out = tmp_path / "generations.jsonl"
    _write_rows(out, [{"id": "a", "response": "SENTINEL"}])
    generate.run(cfg, tmp_path)
    got = _read_rows(out)
    assert [r["id"] for r in got] == ["a", "b", "c"]
    assert got[0]["response"] == "SENTINEL"
    assert got[1]["text"] == TEXTS[1] and got[1]["label"] == 1
    generate.run(cfg, tmp_path)
    assert _read_rows(out) == got


def test_generations_reload_with_exact_response_ids(tmp_path):
    _write_rows(tmp_path / "in.jsonl", _rows())
    cfg = config.RunConfig.model_validate(
        {
            "name": "g",
            "model": {"name": "gpt2", "dtype": "float32"},
            "data": {"path": str(tmp_path / "in.jsonl"), "format": "raw"},
            "generate": {"max_tokens": 4, "temperature": 0.0},
            "steps": ["generate"],
        }
    )
    generate.run(cfg, tmp_path)
    rows = _read_rows(tmp_path / "generations.jsonl")
    src = config.DataConfig(
        path=str(tmp_path / "generations.jsonl"),
        response="response",
        format="raw",
    )
    tok = backends.load_model(cfg.model).tokenizer
    for row, s in zip(rows, data.load_samples(src, 0), strict=True):
        assert s.response_ids == row["response_ids"]
        enc = data.render(s, src, "response", tok)
        p = data.prompt_ids(s, src, tok)
        assert enc.ids == p + row["response_ids"]
        assert enc.span == (len(p), len(enc.ids))
```

The second test checks that phase 2 carries `response_ids` through `load_samples` and `render` (spec §6). It asserts `s.response_ids` directly because a decode/re-encode round trip can pass by accident on gpt2.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_steer.py -v`
Expected: a collection error, `ImportError: cannot import name 'generate' from 'sonde'`.

- [ ] **Step 3: Implement**

Create `sonde/generate.py`:

```python
"""The generate step: sample responses into generations.jsonl."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from sonde import backends
from sonde import config
from sonde import data

# ponytail: fixed chunk; add generate.batch_size when a config needs it.
BATCH = 64


def write_rows(
    loaded: backends.Loaded,
    samples: list[data.Sample],
    src: config.DataConfig,
    gen: config.GenerateConfig,
    seed: int,
    path: pathlib.Path,
    steering: backends.Steering | None = None,
) -> None:
    """Generates for `samples` in chunks and appends one JSON row each.

    A row is the sample's source row with its id under the id key, plus
    `response` (decoded, special tokens skipped) and `response_ids` (the
    exact generated ids). Each chunk is flushed as it finishes.

    Args:
        loaded: The model from `backends.load_model`.
        samples: Rows to generate for, written in this order.
        src: The data block the samples came from (render and id key).
        gen: Sampling settings.
        seed: Sampler seed, reset for every chunk.
        path: The .jsonl file to append to; parents are created.
        steering: Edit applied while generating, or None.
    """
    tok: Any = loaded.tokenizer
    key = src.id or "id"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for k in range(0, len(samples), BATCH):
            chunk = samples[k : k + BATCH]
            prompts = [data.prompt_ids(s, src, tok) for s in chunk]
            responses = backends.generate(loaded, prompts, gen, seed, steering)
            for s, r in zip(chunk, responses, strict=True):
                row = {
                    **s.raw,
                    key: s.id,
                    "response": tok.decode(r, skip_special_tokens=True),
                    "response_ids": r,
                }
                f.write(f"{json.dumps(row)}\n")
            f.flush()


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Appends a generation for every sample whose id is not in the file yet.

    Raises:
        ValueError: If the model, data or generate section is missing.
    """
    if cfg.model is None or cfg.data is None or cfg.generate is None:
        raise ValueError("generate needs the model, data and generate sections")
    path = run_dir / "generations.jsonl"
    key = cfg.data.id or "id"
    done = set()
    if path.exists():
        with path.open() as f:
            done = {str(json.loads(line)[key]) for line in f}
    samples = data.load_samples(cfg.data, cfg.seed)
    todo = [s for s in samples if s.id not in done]
    if todo:
        loaded = backends.load_model(cfg.model)
        write_rows(loaded, todo, cfg.data, cfg.generate, cfg.seed, path)
```

The `None` guard narrows the optional sections for pyright; `RunConfig`'s validator has already enforced them. When every id is already done, `run` returns without loading the model.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_steer.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add sonde/generate.py tests/test_steer.py
git commit -m "feat(generate): generate step with resume and exact response_ids"
```

---

### Task 6.3: `steer` step and runner registration

**Files:**
- Create: `sonde/steer.py`
- Modify: `sonde/runner.py` (imports, `STEPS`, `run`)
- Modify: `tests/test_steer.py`
- Modify: `tests/test_e2e.py` (delete `test_runner_refuses_unregistered_step`)

**Interfaces:**
- Consumes:
  - `generate.write_rows(...)` and `generate.run(...)` from Task 6.2
  - `backends.Steering` and `backends.load_model`
  - `probe.Probe.load(path: str) -> Probe`, with fields `w [H] float32` and `block: int`
  - `probe.Probe.save(path: str) -> str`, which does not create parent directories
  - `runner.STEPS: dict[str, Callable[[config.RunConfig, pathlib.Path], None]]`
  - `runner.run(cfg) -> pathlib.Path`, which returns `Path(cfg.output.dir) / cfg.name`
  - `config.SteerConfig(probe: str | None, mode: "add" | "ablate", strengths: list[float])`
- Produces:
  - `steer.run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None`, which writes `run_dir/"steer"/f"s{strength:g}.jsonl"` (`s0`, `s4`, `s0.5`)
  - `runner.STEPS["generate"] is generate.run` and `runner.STEPS["steer"] is steer.run`

- [ ] **Step 1: Write the failing test**

In `tests/test_steer.py`, add these two imports to the `sonde` group in alphabetical order, so the block ends with `generate`, `probe`, `runner`:

```python
from sonde import probe
from sonde import runner
```

Append:

```python
def test_steer_run_strength_zero_matches_generate(tmp_path):
    _, _, v = _gpt2()
    _write_rows(tmp_path / "in.jsonl", _rows())
    cfg = config.RunConfig.model_validate(
        {
            "name": "t",
            "model": {"name": "gpt2", "dtype": "float32"},
            "data": {"path": str(tmp_path / "in.jsonl"), "format": "raw"},
            "generate": {"max_tokens": 6, "temperature": 0.0},
            "steer": {"mode": "add", "strengths": [0, 100]},
            "output": {"dir": str(tmp_path)},
            "steps": ["generate", "steer"],
        }
    )
    (tmp_path / "t" / "probes").mkdir(parents=True)
    probe.Probe(
        name="t",
        kind="linear",
        w=(3.0 * v).numpy(),
        bias=0.0,
        block=BLOCK,
        window="prompt",
        pooling="mean",
        threshold=0.5,
        model="gpt2",
        engine="hf==test",
        model_fingerprint=None,
        prompt_format="raw",
    ).save(str(tmp_path / "t" / "probes" / "t.npz"))
    run_dir = runner.run(cfg)
    gen = _read_rows(run_dir / "generations.jsonl")
    s0 = _read_rows(run_dir / "steer" / "s0.jsonl")
    s100 = _read_rows(run_dir / "steer" / "s100.jsonl")
    assert [r["response_ids"] for r in s0] == [r["response_ids"] for r in gen]
    assert [r["response_ids"] for r in s100] != [r["response_ids"] for r in gen]
    assert [r["id"] for r in s100] == ["a", "b", "c"]
    runner.run(cfg)
    assert len(_read_rows(run_dir / "steer" / "s0.jsonl")) == 3
```

This test covers six things:
- the runner registers both steps;
- the default probe path `run_dir/probes/{name}.npz` is used;
- the vector is normalized: `w` is 3·v, and steering uses w/|w| = v;
- strength 0 matches unsteered generation along the real code path;
- a large add changes the output;
- a rerun overwrites `steer/` and does not append to it.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/test_steer.py::test_steer_run_strength_zero_matches_generate -v`
Expected: FAIL with `ValueError: steps not available yet: ['generate', 'steer']`, raised by `runner.run`.

- [ ] **Step 3: Implement**

Create `sonde/steer.py`:

```python
"""The steer step: generate along a probe direction at several strengths."""

from __future__ import annotations

import pathlib

import torch

from sonde import backends
from sonde import config
from sonde import data
from sonde import generate
from sonde import probe


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Writes steer/s{strength}.jsonl per strength; strength 0 is unedited.

    The direction is the probe's w / |w|, applied at the probe's block.

    Raises:
        ValueError: If a needed section is missing.
    """
    if (
        cfg.model is None
        or cfg.data is None
        or cfg.generate is None
        or cfg.steer is None
    ):
        raise ValueError(
            "steer needs the model, data, generate, steer sections"
        )
    path = cfg.steer.probe or str(run_dir / "probes" / f"{cfg.name}.npz")
    p = probe.Probe.load(path)
    loaded = backends.load_model(cfg.model)
    w = torch.tensor(p.w)
    unit = w / w.norm()
    samples = data.load_samples(cfg.data, cfg.seed)
    for s in cfg.steer.strengths:
        out = run_dir / "steer" / f"s{s:g}.jsonl"
        out.unlink(missing_ok=True)
        steering = (
            None
            if s == 0
            else backends.Steering(p.block, unit, cfg.steer.mode, s)
        )
        generate.write_rows(
            loaded, samples, cfg.data, cfg.generate, cfg.seed, out, steering
        )
```

A probe whose width or block does not fit the model already fails loudly: a wrong H breaks the matmul, and an out-of-range block raises `IndexError`.

In `sonde/runner.py`, add these imports to the `sonde` group in alphabetical order:

```python
from sonde import generate
from sonde import steer
```

Then make `STEPS` read:

```python
STEPS: dict[str, Callable[[config.RunConfig, pathlib.Path], None]] = {
    "extract": extract.run,
    "train": sweep.run,
    "score": score.run,
    "generate": generate.run,
    "steer": steer.run,
}
```

Every value `RunConfig.steps` accepts is now registered, so phase 3's `missing` check can never fire. Replace `run` with:

```python
def run(cfg: config.RunConfig) -> pathlib.Path:
    """Runs cfg.steps in order.

    Validates, refuses a stale extraction before any model loads, writes
    the resolved config.yaml, then runs each step.

    Returns:
        The run directory.

    Raises:
        ValueError: extract/ holds a different config (and output.overwrite
            is false).
    """
    if "extract" in cfg.steps:
        extract.preflight(cfg)
    out = config.run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    )
    for step in cfg.steps:
        STEPS[step](cfg, out)
    return out
```

In `tests/test_e2e.py`, delete `test_runner_refuses_unregistered_step` along with the two blank lines after it. After this task the test would fail: generate is registered, so the run dir gets created, and the default `format: chat` then raises a chat-template error that does not mention `generate`. `pytest`, `config.GenerateConfig` and `QUICKSTART` are still used elsewhere in that file, so no imports change.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_steer.py -v`
Expected: 8 passed.

Run: `uv run pytest -m "not gpu" -q`
Expected: the whole suite passes. `test_runner_refuses_unregistered_step` no longer appears.

- [ ] **Step 5: Commit**

```bash
git add sonde/steer.py sonde/runner.py tests/test_steer.py tests/test_e2e.py
git commit -m "feat(steer): steer step over strengths; register generate and steer"
```

---

### Task 6.4: vllm `backends.generate` with the `sonde_steer` edit (checked on the Spark)

**Files:**
- Modify: `sonde/backends.py` (dispatch in `generate`; add `_vllm_generate`)
- Modify: `tests/test_steer.py` (one `@pytest.mark.gpu` test)

**Interfaces:**
- Consumes:
  - `backends.load_model(config.ModelConfig(backend="vllm", ...))` from phase 5, which returns an nnsight `VLLM` engine as `.model` that runs eager with prefix caching off
  - the pytest marker `gpu`, registered in pyproject in phase 1
  - `scripts/spike/venvs.sh` from phase 5, which builds `~/experiments/sonde-spike/.venv-{hf,vllm}` from `uv.lock`
  - `backends.Steering` and `backends.generate` from Task 6.1
- Produces:
  - `backends.generate` for `loaded.backend == "vllm"`, with the same contract as hf. Every request passes `edits=[]`, or `edits=["sonde_steer"]` when steering. The edit is cleared in a `finally`.

- [ ] **Step 1: Write the failing GPU test**

In `tests/test_steer.py`, add `import pytest` after `from typing import Any`, separated by a blank line as an isort third-party group with `torch`. Append:

```python
@pytest.mark.gpu
def test_vllm_steer_zero_add_and_clear():
    loaded = backends.load_model(
        config.ModelConfig(
            name="Qwen/Qwen3-0.6B",
            backend="vllm",
            vllm={"gpu_memory_utilization": 0.2, "max_model_len": 512},
        )
    )
    tok: Any = loaded.tokenizer
    prompts = [tok(t).input_ids for t in TEXTS]
    v = torch.randn(loaded.hidden, generator=torch.Generator().manual_seed(0))
    v = v / v.norm()
    plain = backends.generate(loaded, prompts, GREEDY, 0)
    zero = backends.Steering(14, v, "add", 0.0)
    assert backends.generate(loaded, prompts, GREEDY, 0, zero) == plain
    add = backends.Steering(14, v, "add", 1000.0)
    assert backends.generate(loaded, prompts, GREEDY, 0, add) != plain
    ablate = backends.Steering(14, v, "ablate", 1.0)
    assert len(backends.generate(loaded, prompts, GREEDY, 0, ablate)) == 3
    assert backends.generate(loaded, prompts, GREEDY, 0) == plain
```

The final assert checks that `edit.clear()` ran, so steering does not leak into later requests.

- [ ] **Step 2: Confirm the local suite deselects it, and that it fails on the Spark**

Run (Mac or CI): `uv run pytest tests/test_steer.py -m "not gpu" -q`
Expected: `8 passed, 1 deselected`.

Phase 5 removed `~/experiments/sonde-spike` at its end, so sync the tree and rebuild the venvs first. `sync` copies the working tree, including the uncommitted test.

```bash
git status --porcelain   # only tests/test_steer.py
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 8 'bash scripts/spike/venvs.sh'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S run sonde-spike 32 '../.venv-vllm/bin/python -m pytest -m gpu tests/test_steer.py -k vllm -v'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 40
```

Expected: the log shows `NotImplementedError: generate on vllm` and `1 failed, 8 deselected`, then `[spark] exit code 1`. A 32 GB job has a cap of 32/121 ≈ 0.26, which leaves room for the test's `gpu_memory_utilization` of 0.2.

- [ ] **Step 3: Implement**

In `sonde/backends.py`, replace the last three lines of `generate` (the `NotImplementedError` guard and the hf return) with:

```python
    if loaded.backend == "hf":
        return _hf_generate(loaded, prompts, gen, seed, steering)
    return _vllm_generate(loaded, prompts, gen, seed, steering)
```

Append:

```python
def _vllm_generate(
    loaded: Loaded,
    prompts: list[list[int]],
    gen: config.GenerateConfig,
    seed: int,
    steering: Steering | None,
) -> list[list[int]]:
    model: Any = loaded.model
    requests = [{"prompt_token_ids": p} for p in prompts]
    params = {
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "top_k": -1,
        "max_tokens": gen.max_tokens,
        "seed": seed,
    }
    if steering is None:
        outs = model.generate(requests, edits=[], **params)
    else:
        layer = model.model.layers[steering.block]
        v, s = steering.vector, steering.strength
        ablate = steering.mode == "ablate"
        with model.edit(name="sonde_steer") as (tracer, edit):
            for _ in tracer.all():
                out = layer.output
                u = v.to(out[0].device, out[0].dtype)
                if ablate:
                    out[0][:] -= s * ((out[0] + out[1]) @ u)[:, None] * u
                else:
                    out[0][:] += s * u
        try:
            outs = model.generate(requests, edits=["sonde_steer"], **params)
        finally:
            edit.clear()
    return [list(o.outputs[0].token_ids) for o in outs]
```

Notes from the research doc and nnsight b717807:
- A plain `model.generate(requests, **params)` builds an explicit `SamplingParams(**params)`, so the checkpoint's `generation_config.json` never applies.
- The envoy `layer` is bound outside the edit, so the model is not shipped with the block.
- A vLLM value is `[tokens, H]` with no batch axis. `out[0] + out[1]` is the stream after block L, and writing `out[0]` alone adds to that sum exactly once.
- The code uses no `.skip()` and no `taps`.

- [ ] **Step 4: Run on the Spark and confirm it passes**

The venvs from Step 2 persist, because `sync` skips them.

```bash
git status --porcelain   # only sonde/backends.py and tests/test_steer.py
S=~/.claude/skills/dgx-spark/scripts/spark
$S sync "$(git rev-parse --show-toplevel)" sonde-spike
$S run sonde-spike 32 '../.venv-vllm/bin/python -m pytest -m gpu tests/test_steer.py -k vllm -v'
ssh 10.97.110.177 '~/.local/bin/pueue wait $(cat ~/experiments/sonde-spike/runs/task_id)'
$S logs sonde-spike 40
ssh 10.97.110.177 '~/experiments/sonde-spike/.venv-vllm/bin/python -c "import vllm; print(vllm.__version__)"'
```

Expected: `1 passed, 8 deselected`, then `[spark] exit code 0`. The last command prints the vLLM version for the commit message.

Run locally: `uv run pytest tests/test_steer.py -m "not gpu" -q`
Expected: `8 passed, 1 deselected`.

- [ ] **Step 5: Commit, record the result and clean the Spark**

Put the version printed in Step 4 in place of `<version>`:

```bash
git add sonde/backends.py tests/test_steer.py
git commit -m "feat(backends): vllm generate with sonde_steer edit, cleared in finally" \
  -m "test_vllm_steer_zero_add_and_clear passed on the Spark (Qwen3-0.6B, block 14), vllm==<version>."
git rev-parse --short HEAD
~/.claude/skills/dgx-spark/scripts/spark clean sonde-spike
```

Expected: `removed ~/experiments/sonde-spike`. Add this line to the PR description, filling in the printed sha: `vllm steer: test_vllm_steer_zero_add_and_clear passed on the Spark at <sha>, vllm==<version>.` Do not record it in `docs/parity-2026-10.md`. `scripts/spike/compare.py` regenerates that file, so any hand edit would be lost.

---

### Task 6.5: README, CHANGELOG, CLI check

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: phase 4's README sections (Config reference table, step table, run-dir listing, rerun bullets) and CHANGELOG `0.2.0` → `Added`; `sonde run` from phase 3; `generate.run` and `steer.run` from Tasks 6.2 and 6.3
- Produces: documentation only

- [ ] **Step 1: README**

In the Config reference table, change the `steps` row's notes cell to: `` `generate`, `extract`, `train`, `score`, `steer`, run in the listed order ``.

Insert these rows directly before the `output.dir` row:

```markdown
| `generate.max_tokens`, `generate.temperature`, `generate.top_p` | `256`, `0.7`, `1.0` | `temperature: 0` is greedy; top-k is off |
| `steer.probe` | `null` | defaults to this run's `probes/<name>.npz`; direction is w/‖w‖ at its block |
| `steer.mode`, `steer.strengths` | `add`, `[0.0]` | `add`: x += s·v; `ablate`: x −= s·(x·v)·v; strength 0 is the unedited baseline |
```

Append these rows to the step table (step and the sections it needs):

```markdown
| `generate` | `model`, `data`, `generate` |
| `steer` | `model`, `data`, `generate`, `steer` |
```

Add these lines to the run-dir listing:

```text
generations.jsonl        source row + response, response_ids
steer/s<strength>.jsonl  one file per strength
```

Add this bullet to the rerun bullets: `` `generate` skips ids already in generations.jsonl; `steer` overwrites its files. ``

- [ ] **Step 2: CHANGELOG**

Under `## 0.2.0`, in `### Added`, append:

```markdown
- `generate` (resumable `generations.jsonl` with exact `response_ids`) and `steer` (add / ablate at every decode step, one file per strength) on both backends.
```

- [ ] **Step 3: Run the CLI end to end on CPU**

The first run trains the quickstart probe. The second run generates and steers along it.

```bash
uv run sonde run quickstart -o output.dir=/tmp/sonde-qs6
uv run sonde run quickstart -o output.dir=/tmp/sonde-qs6 -o "steps=[generate, steer]" -o generate.max_tokens=8 -o generate.temperature=0 -o "steer.strengths=[0, 4]"
wc -l /tmp/sonde-qs6/quickstart/generations.jsonl; ls /tmp/sonde-qs6/quickstart/steer
```

Expected: `64 /tmp/sonde-qs6/quickstart/generations.jsonl`, then `s0.jsonl  s4.jsonl`.

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document generate and steer"
```

<!-- skipped: Task 6.1 _hf_generate device move (`.to(model.device)`). Verified wrong in the scratch venv. nnterp's `model.device` is `meta` until the first trace, so moving the inputs there makes the first generate call fail with "NotImplementedError: Cannot copy out of meta tensor; no data!". CPU inputs already work on a non-CPU device: the model was dispatched to mps:0 on the Mac and all 8 tests passed, because transformers 5's prepare_inputs_for_generation moves the forward's tensors to self.device (generation/utils.py, step 6). That mechanism also covers CUDA. Recorded as verified fact 9. Phase 2's `forward` should check the same meta trap if it calls `.to(model.device)` before any trace has run. -->

---

## Appendix: interface contract


Spec: docs/superpowers/specs/2026-10-02-sonde-redesign-design.md. Where this file and the spec
differ on a NAME or SIGNATURE, this file wins (it resolves placement details the spec left open);
on BEHAVIOR the spec wins.

Style: Google Python style. `import x` / `from pkg import module` only (never import functions,
classes or constants; `typing`, `collections.abc` exempt). In tests: `from sonde import probe`,
then `probe.Probe(...)`. 80-col lines, 4-space indent. Args/Returns/Raises docstrings on public
functions; docstrings give array shapes. No WHAT comments. `# ponytail:` marks deliberate corners.
`from __future__ import annotations` at top of every module.

## sonde/__init__.py
```python
__version__ = "0.2.0"
def __getattr__(name: str):  # lazy: sonde.run, sonde.load_config
    ...  # "run" -> runner.run, "load_config" -> config.load; else AttributeError
```
Must not import torch/pydantic/yaml at import time.

## sonde/fingerprint.py  (verbatim move from ~/dev/personal/mechanica/mechanica/fingerprint.py)
`RAW = "raw"`; `prompt_format(template: str | None) -> str`; `artifact_digest(path) -> str`;
`normalize_revision(value) -> str | None`;
`checkpoint_fingerprint(model: str, revision: str | None = None, quantization: str | None = None) -> str`.

## sonde/probe.py  (numpy + stdlib only)
```python
FORMAT = 1
WINDOWS = frozenset({"prompt", "response", "all", "last_turn"})
KINDS = frozenset({"linear", "attention"})
LINEAR_POOLINGS = frozenset({"mean", "last", "max", "rolling_mean"})

def last_turn_start(tokenizer, messages: list[dict]) -> int   # mechanica's rule, duck-typed

@dataclasses.dataclass
class Probe:  # fields exactly as spec §5, in that order
    def logits(self, acts) -> np.ndarray            # [T,H] -> [T]; linear only
    def pooled_logit(self, acts) -> float           # [T,H] or [H]
    def pooled_score(self, acts) -> float           # sigmoid(pooled_logit)
    def pooled_score_from_logits(self, logits) -> float   # linear only
    def flag(self, acts) -> bool
    def escalates(self, score: float) -> bool
    def serves(self, fingerprint: str | None, adapter: str | None,
               prompt_format: str | None = "unchecked") -> str | None
    def vllm_aux_layer(self) -> int                 # block + 1
    def backend(self) -> str                        # engine.split("==", 1)[0]
    def save(self, path: str) -> str                # atomic; appends .npz if missing
    @classmethod
    def load(cls, path: str) -> "Probe"
def load_dir(directory: str) -> dict[str, Probe]
```
Pooling math (module-private `_pool(logits [T], pooling, rolling_window) -> float` on logits,
float32, then `_sigmoid` clipped ±30). attention: `a = acts@q; v = acts@w; softmax(a)·v + bias`.

## sonde/config.py  (pydantic v2, every model `model_config = ConfigDict(extra="forbid")`)
```python
class ModelConfig: name: str; revision: str | None = None; dtype: str = "bfloat16";
    backend: Literal["hf", "vllm"] = "hf"; vllm: dict = {}   # pydantic copies defaults; ok
class DataConfig: path: str | None = None; hf: str | None = None; hf_config: str | None = None;
    hf_split: str = "train"; text: str | None = "text"; messages: str | None = None;
    response: str | None = None; label: str | None = "label"; id: str | None = "id";
    group: str | None = None; label_map: dict | None = None; limit: int | None = None;
    format: Literal["chat", "raw"] = "chat"; system: str | None = None;
    max_length: int = 2048; split: tuple[float, float, float] = (0.7, 0.15, 0.15)
class Every: every: int
class ExtractConfig: layers: Literal["all"] | list[int] | Every = "all";
    window: Literal["prompt","response","all","last_turn"] = "prompt";
    keep: Literal["pooled","tokens"] = "pooled"; batch_size: int = 16;
    shard_size: int = 4096; shard_bytes: int = 2**31
class ProbeConfig: kind: Literal["linear","attention"] = "linear";
    pooling: Literal["mean","last","max","rolling_mean","attention"] = "mean";
    rolling_window: int | None = None; init: Literal["random","diff_means"] = "random";
    epochs: int = 20; lr: float = 1e-3; weight_decay: float = 0.0; batch_size: int = 256;
    patience: int = 3; max_fpr: float | None = 0.01;
    select: Literal["auroc","recall_at_fpr","group_auroc"] = "recall_at_fpr"
class ScoreEntry: name: str; probe: str | None = None; + every DataConfig field as Optional
    with default None (unset = inherit from data)
class GenerateConfig: max_tokens: int = 256; temperature: float = 0.7; top_p: float = 1.0
class SteerConfig: probe: str | None = None; mode: Literal["add","ablate"] = "add";
    strengths: list[float] = [0.0]
class OutputConfig: dir: str = "runs"; overwrite: bool = False
class RunConfig: name: str; seed: int = 0; model: ModelConfig | None = None;
    data: DataConfig | None = None; extract: ExtractConfig | None = None;
    probe: ProbeConfig | None = None; score: list[ScoreEntry] = [];
    generate: GenerateConfig | None = None; steer: SteerConfig | None = None;
    output: OutputConfig = OutputConfig(); steps: list[Literal["generate","extract","train",
    "score","steer"]] = ["extract", "train"]
    # model_validator(mode="after"): required sections per step (spec §11 table) + every
    # cross-field check in spec §11 "Load-time checks". Errors name the field path.

RECIPES_DIR: pathlib.Path   # sonde/recipes
def load(source: str, overrides: Sequence[str] = ()) -> RunConfig
    # source = path to .yaml or bare recipe name; unknown name -> ValueError listing recipes
def apply_override(raw: dict, item: str) -> None     # "a.b=v", v via yaml.safe_load
def score_data(cfg: RunConfig, entry: ScoreEntry) -> DataConfig   # entry fields over cfg.data
def resolve_blocks(spec, num_layers: int) -> list[int]   # sorted, deduped; ValueError if out of range
def run_dir(cfg: RunConfig) -> pathlib.Path              # Path(cfg.output.dir) / cfg.name
```

## sonde/data.py
```python
@dataclasses.dataclass
class Sample: id: str; text: str | None; messages: list[dict] | None; response: str | None;
    response_ids: list[int] | None; label: int | None; group: str | None; raw: dict
@dataclasses.dataclass
class Encoded: id: str; ids: list[int]; span: tuple[int, int]; label: int | None; group: str | None
class Drop(Exception): reason: str      # raised by render; reasons: "empty_window",
    # "not_assistant_last", "last_turn_prefix_mismatch", "empty_response"
def load_samples(src: DataConfig, seed: int) -> list[Sample]
def prompt_format_of(src: DataConfig, tokenizer) -> str
def prompt_ids(sample: Sample, src: DataConfig, tokenizer) -> list[int]   # with generation prompt
def render(sample: Sample, src: DataConfig, window: str, tokenizer) -> Encoded   # raises Drop
def encode_all(samples, src, window, tokenizer) -> tuple[list[Encoded], dict[str, int]]
    # drop counts by reason; logs them
def split(labels: Sequence[int], groups: Sequence[str | None], fractions, seed: int
          ) -> dict[str, list[int]]   # keys train/val/test -> sample indices; ValueError if a
          # split lacks a class; logs achieved fractions and n_pos/n_neg
```

## sonde/backends.py  (nnterp imported only inside hf functions; nnsight.modeling.vllm only
inside vllm functions; torch imported at module top is fine)
```python
@dataclasses.dataclass
class Loaded: backend: str; model: object; tokenizer: object; num_layers: int; hidden: int;
    engine: str     # spec §5 engine grammar
@dataclasses.dataclass
class Steering: block: int; vector: torch.Tensor  # [H] unit norm
    mode: str; strength: float
def load_model(cfg: ModelConfig) -> Loaded   # single-slot cache keyed by cfg.model_dump_json()
def window_pool(h: torch.Tensor, span: tuple[int, int], keep: str, pooling: str | None
                ) -> torch.Tensor           # h [T,H] -> [H] (pooled, mean|last) or [end-start, H]
def forward(loaded: Loaded, batch: list[Encoded], blocks: list[int], keep: str,
            pooling: str | None) -> dict[int, list[torch.Tensor]]   # dispatch hf/vllm
def generate(loaded: Loaded, prompts: list[list[int]], gen: GenerateConfig, seed: int,
             steering: Steering | None = None) -> list[list[int]]   # response token ids
```
Stored values keep the model dtype; `.cpu()` before returning.

## sonde/store.py
```python
def config_hash(sections: dict) -> str        # sha256 of json.dumps(sort_keys=True)[:16]
def prepare(out_dir: pathlib.Path, config_hash: str, overwrite: bool) -> str
    # "fresh" | "resume" | "done"; raises ValueError on hash mismatch unless overwrite
    # (overwrite rmtree's out_dir). Writes run.json on fresh. Never loads a model.
class ShardWriter:
    def __init__(self, out_dir: pathlib.Path, keep: str, shard_size: int, shard_bytes: int)
    done: int                                  # samples already in complete shards (resume)
    def add(self, features: dict[int, list[torch.Tensor]]) -> None   # one batch, sample order
    def close(self) -> list[str]               # flush; returns shard file names in order
def write_manifest(out_dir: pathlib.Path, manifest: dict) -> None   # adds "format": 1; atomic
def read_manifest(out_dir: pathlib.Path) -> dict   # ValueError on unknown format
def read_layer(out_dir: pathlib.Path, block: int) -> tuple[torch.Tensor, torch.Tensor | None]
    # (X [n,H] or flat [ΣT,H], offsets [n+1] or None)
```
Shard tensor names: `B{block}`, `offsets`.

## sonde/extract.py
```python
def extract_hash(cfg: RunConfig) -> str      # model, data, extract (+ probe.pooling if pooled)
def preflight(cfg: RunConfig) -> str         # store.prepare(run_dir/"extract", ...); no model
def extract_to(loaded: backends.Loaded, encoded: list[data.Encoded], blocks: list[int],
               keep: str, pooling: str | None, out_dir: pathlib.Path, batch_size: int,
               shard_size: int, shard_bytes: int, meta: dict) -> None
    # resumes via ShardWriter.done; writes manifest last with meta + ids/labels/groups/shards
def run(cfg: RunConfig, run_dir: pathlib.Path) -> None   # the "extract" step
```
Manifest keys: format, model, revision, model_fingerprint, prompt_format, engine, blocks, window,
keep, pooling, dtype, ids, labels, groups, shards, drops, config_hash.

## sonde/probes.py  (torch)
```python
class LinearProbe(torch.nn.Module):
    def __init__(self, hidden: int, pooling: str, rolling_window: int | None = None)
    def forward(self, x, mask=None) -> torch.Tensor    # x [B,H] (mask None) or [B,T,H]; -> [B]
class AttentionProbe(torch.nn.Module):
    def __init__(self, hidden: int)
    def forward(self, x, mask) -> torch.Tensor         # [B,T,H],[B,T] -> [B]
def build(cfg: ProbeConfig, hidden: int) -> torch.nn.Module
def export(module, mu: torch.Tensor, sigma: torch.Tensor, **meta) -> probe.Probe
    # folds standardisation (spec §9); meta = Probe fields other than w/q/bias/kind/pooling/
    # rolling_window
def pad(X: torch.Tensor, offsets: torch.Tensor | None, idx: Sequence[int]
        ) -> tuple[torch.Tensor, torch.Tensor | None]   # -> (x [B,H] or [B,T,H], mask or None)
```

## sonde/train.py  (numpy metrics; torch fit)
```python
def standardise(X, offsets, train_idx) -> tuple[torch.Tensor, torch.Tensor]  # (mu [H], sigma [H])
def fit(cfg: ProbeConfig, X, offsets, labels: np.ndarray, train_idx, val_idx, seed: int
        ) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]   # (module, mu, sigma)
def diff_means(X, offsets, labels, train_idx) -> tuple[np.ndarray, float]   # (w, b) raw space;
    # tokens: per-sample mean over window first
def auroc(y: np.ndarray, scores: np.ndarray) -> float
def choose_threshold(y, scores, max_fpr: float | None) -> tuple[float, bool]   # (thr, failed)
def recall_at_fpr(y, scores, max_fpr: float) -> float   # val-free: threshold on these scores
def group_auroc(y, scores, groups) -> tuple[float, int]   # (mean within-group AUROC, n_groups)
def bootstrap_ci(y, scores, fn, seed: int, n: int = 1000) -> tuple[float, float]
def metrics(y, scores, threshold: float, max_fpr: float | None, groups=None) -> dict
    # handles single-class y: FPR-only or recall-only; adds small_n flag
def probe_scores(p: probe.Probe, X, offsets, idx) -> np.ndarray   # numpy Probe.pooled_score
def bow_features(texts: list[str], train_idx, vocab_size: int = 5000) -> np.ndarray  # [n, V] float32 counts
def window_text(sample: data.Sample, window: str) -> str
```

## sonde/sweep.py, score.py, generate.py, steer.py
Each: `def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None`.
sweep writes run_dir/"splits.json", "probes/{name}.npz", "layers/B{b}.npz", "metrics.json".
score writes run_dir/"score"/<entry>/{extract/, scores.jsonl, metrics.json}.
generate writes run_dir/"generations.jsonl". steer writes run_dir/"steer"/"s{strength}.jsonl".

## sonde/runner.py
```python
STEPS: dict[str, Callable[[config.RunConfig, pathlib.Path], None]]
def run(cfg: config.RunConfig) -> pathlib.Path   # validate → extract.preflight if "extract"
    # in steps → mkdir run dir, write config.yaml → steps in order → return run dir
```
Phase 3 registers extract/train/score; phase 6 adds generate/steer.

## sonde/cli.py
`def main(argv: list[str] | None = None) -> int`; pyproject `sonde = "sonde.cli:main"`.
`if __name__ == "__main__": raise SystemExit(main())`.

## Notes
- pydantic mutable defaults use `pydantic.Field(default_factory=...)` (Google style: no mutable
  default values); shown as `= {}` / `= []` above only for brevity.
- Tests live flat in tests/ with names from spec §14. Run with `uv run pytest ...`.
