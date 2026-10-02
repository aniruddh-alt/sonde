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

CI runs the same checks, with ruff in check-only mode:

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
