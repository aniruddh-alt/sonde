# Changelog

All notable changes to `sondekit` are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project is pre-1.0 and the
API may change between minor versions.

## [0.2.0] - Unreleased

A rewrite; nothing from 0.1 stays compatible. 0.1 is preserved at the git
tag `v0.1-legacy`. The package is renamed from `sonde` to `sondekit`
(`pip install sondekit`, `import sondekit`), because `sonde` is taken on
PyPI.

### Added
- `sondekit run CFG|RECIPE [-o k=v ...] [--dry-run]`, driven by one pydantic
  config (`extra="forbid"`) with an explicit `steps:` list.
- `sondekit.probe.Probe` (`.npz`, format 1) and `sondekit.fingerprint`, importable
  with numpy and the stdlib only. This is the artifact mechanica loads.
- Two backends, chosen explicitly: `hf` (nnterp 1.3 on nnsight 0.7) and
  `vllm` (nnsight's `VLLM`; experimental: it loads, but extraction and
  generation raise `NotImplementedError`).
- Sharded, resumable extraction with a config-hash check that runs before
  any model loads.
- Linear (`mean`, `last`, `max`, `rolling_mean`) and attention probes,
  `diff_means` init, a validation-chosen threshold (`max_fpr`, default 1%,
  or best F1), recall at `max_fpr` as the headline with bootstrap CIs,
  rank and group AUROC, a shuffled-label control, and bag-of-words and
  length baselines.
- Recipes: `quickstart`, `refusal`, `high_stakes`.
- `generate` (resumable `generations.jsonl` with exact `response_ids`) and
  `steer` (add / ablate at every decode step, one file per strength), on
  the `hf` backend.

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
- `experiments/`, `examples/`, the internal design documents and the
  committed refusal activations. The refusal dataset moved to
  `data/refusal/`.

### Changed
- CI is one job: `uv sync --locked`, then ruff, pyright and
  `pytest -m "not gpu"` on Python 3.12.
- `docs/linear-probes-primer.md`: an accuracy pass on its paper summaries
  and citations.

## [0.1.0] - 2026-06-28

### Packaging
- Moved every module under a single `sonde/` package (`sonde.activation`,
  `sonde.dataset`, `sonde.probes`, `sonde.interventions`, …). Previously the
  modules squatted top-level names (`dataset`, `core`, `configs`) and a
  non-editable install was broken.
- `sonde/__init__.py` re-exports the public API; added `sonde.__version__`.
- Added `py.typed`; `matplotlib` moved to a `viz` extra; `transformers` and
  `pyyaml` declared explicitly.
- Config aliases and recipes load via `importlib.resources` (CWD-independent);
  `sonde run quickstart` works offline from any directory.
- CLI accepts both `sonde <cfg>` and `sonde run -c <cfg>`.

### Tensor storage
- Extraction artifacts are a JSON manifest (`_manifest.json`) + safetensors, not
  a pickle. Loading a manifest is no longer code execution.
- The safetensors path is stored relative to the manifest, so artifact
  directories are relocatable.
- `save_path=None` (new default) skips persistence cleanly instead of crashing.
- Variable-length (sequence-mode) activations now persist (padded + lengths) and
  round-trip exactly.
- Overwrite protection: existing artifacts are not clobbered unless
  `overwrite=True`. Legacy `_manifest.pt` pickles remain readable.

### Datasets
- `ProbingDataset`: a list of equal-length 2-D `(S, D)` tensors is now sequence
  data by default (no silent flatten to `(N, S*D)`); `sequence_mode=False` opts
  into pooled flattening.
- `from_extraction_result` raises when no labels are resolvable instead of
  silently labelling every sample 0.
- `ProbingSampleBuilder.to_samples` logs dropped empty-text rows.
- `SampleBundle.prompts` is a `list[str]`; the split's grouping regime is logged.

### Probes & runner
- Wired the `probe_sweep` and `diff_means` runner actions (previously
  `NotImplementedError`); both read a pre-extracted artifact and save a
  `ProbeArtifact`.
- `ProbeArtifact`: a single safetensors file with the concept direction (+bias,
  layer, metadata) — the contract between probing and intervention/deploy.

### Causal interventions
- `sonde.interventions.InterventionContext` with additive steering and
  directional ablation (`project_subtract`), applied via `ctx.apply()` inside a
  user `with model.trace/generate(...)` block (verified against gpt2).
- `docs/intervention_design.md` updated with verified findings and a required
  scientific-controls section.
- `examples/causal_loop_gpt2.py`: full extract → probe → ablate → measure loop
  with a random-direction specificity control.
