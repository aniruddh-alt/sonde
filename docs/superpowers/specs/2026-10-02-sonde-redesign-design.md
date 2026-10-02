# sonde redesign: design spec

Date: 2026-10-02. Status: the owner approved each section in conversation. Five independent
reviewers (audit coverage, mechanica contract, nnsight feasibility, YAGNI, consistency) then
checked it, and each review was adversarially verified. This revision folds in their 86
confirmed issues.

## 1. Goal

A user should be able to write one YAML file, run `sonde run cfg.yaml` on a GPU, and get back a
probe artifact they can trust. That artifact is what mechanica (`~/dev/personal/mechanica`,
real-time vLLM monitoring and probe-reward RL) loads and serves.

Success means:

- One config and one command take you from labeled data to an exported probe with a measured
  eval card.
- Activations are correct at every batch size and on both backends, and tests prove it.
- `import sonde.probe` and `import sonde.fingerprint` need only numpy and the stdlib, so
  mechanica can import them inside vLLM workers.
- The package shrinks from 6.8k LOC to about 2.5k, and a contributor can read the whole path in
  an afternoon.

## 2. Decisions (owner-approved)

| # | Decision |
|---|---|
| D1 | "Production" means train, then export a portable probe. Serving and monitoring are mechanica's job. |
| D2 | No backward compatibility is owed. The Python API, YAML schema and `experiments/` may break. |
| D3 | Steering and generation stay first-class features. |
| D4 | Target is a single GPU and small models (≤ 8B) on one DGX Spark (aarch64, GB10, sm_121). No tensor parallelism and no multi-node. |
| D5 | sonde owns the canonical probe artifact and fingerprints. mechanica will import them from sonde. |
| D6 | Wrap NDIF's nnsight and nnterp. Never re-implement model loading, module naming, hooks, the vLLM runtime or interventions. |
| D7 | The default GPU backend is nnsight's `VLLM`. vllm-lens is retired: it forces eager mode, runs only on the V1 runner, and is stale. |
| D8 | Architecture B: a flat, small core behind a two-function backend seam. |
| D9 | Two probe kinds: `linear` (with `pooling`) and `attention`. |
| D10 | Extraction is always a prefill of given token ids. Capturing activations during generation is deferred. |
| D11 | Config is pydantic with `extra="forbid"`, plus an explicit `steps:` list. OmegaConf is dropped. |
| D12 | Headline metric is recall at `max_fpr` (default 1% FPR), as Goodfire's probe-monitor guide recommends. Layers are selected by it (AUROC breaks ties), and every run is reported against a bag-of-words text baseline. |
| D13 | Steering strength is a plain multiple of the unit vector, applied at every step (prefill and decode). |
| D14 | Python floor is 3.12. |

## 3. Problems this fixes

These come from the 2026-10-02 audit: 6 subsystem auditors, each adversarially verified, with
211 findings kept and 1 refuted.

- **R1. Padding corrupts activations.**
  - nnsight left-pads and sonde passes `attention_mask=None`. On gpt2, activations at batch size
    1 and batch size 2 differ by 2653 against a magnitude of 2676.
  - There is no chat template, and the BOS token is doubled.
  - Every result produced at batch size > 1 is suspect.
- **R2. The config hierarchy is broken.**
  - It has 17 dataclasses.
  - `__post_init__` runs on the defaults, so `action: pipeline` can't load.
  - Layers can be set in 3 places and seeds in 4.
- **R3. Everything is implemented twice.**
  - `directions/sweep.py` copies `probes/sweep.py`.
  - There are 2 model loaders, 2 steering paths and 2 vector loaders.
  - The pipeline re-implements extract and sweep.
  - The OOD loop reloads the model.
- **R4. Extraction holds everything in RAM.**
  - There is no resume, and autograd stays on.
  - The overwrite check runs only after the GPU work.
  - Ragged sequences are padded to the global max length.
- **R5. Artifacts are not self-describing.**
  - The probe stores a unit direction next to the raw bias.
  - There are 5 save formats.
  - No revision, template or backend is recorded.
- **R6. Features nobody uses:** 12 activation kinds, 7 selector classes, PCA, and an unused,
  buggy `interventions/` package.
- **R7. Training defaults are wrong.**
  - The seed is set after initialisation.
  - Training defaults to CPU.
  - The threshold is fixed at 0.5, and there is no low-FPR metric.
  - The sanity check raises and throws away the sweep.

## 4. Package layout

```
sonde/
  __init__.py     __version__ + lazy __getattr__ (numpy-only import path)
  probe.py        Probe artifact, WINDOWS, last_turn_start, load_dir (numpy + stdlib)
  fingerprint.py  RAW, prompt_format, checkpoint_fingerprint, normalize_revision,
                  artifact_digest (moved verbatim from mechanica; stdlib only)
  config.py       RunConfig (pydantic), load(path_or_recipe, overrides)
  data.py         load_samples, render (chat|raw), window spans, split
  backends.py     load_model (single-slot cache), hf_forward, vllm_forward,
                  hf_generate, vllm_generate
  extract.py      window_pool, extract step (data → shards)
  store.py        ShardWriter, manifest, resume, read_layer
  probes.py       torch Linear / Attention modules, export() → Probe
  train.py        fit, threshold, metrics, shuffled-label control
  sweep.py        train step: split → per-layer fit → select → test
  score.py        score step: apply a probe to new data
  generate.py     generate step → generations.jsonl
  steer.py        steer step: add / ablate at every step
  runner.py       step registry and run order
  cli.py          sonde run CFG|RECIPE [-o k=v ...] [--dry-run]
  recipes/        quickstart.yaml (+ bundled tiny data), high_stakes.yaml, refusal.yaml
```

Everything in `sonde/activation`, `core`, `dataset`, `directions`, `interventions`, `generation`,
`probes/` (the old package), `runners`, `configs` and `cli/` is deleted in phase 1 (§16).

## 5. Probe artifact (`probe.py`, `fingerprint.py`)

Adapted from `mechanica/probe.py`. `mechanica/fingerprint.py` moves verbatim. Both import only
numpy and the stdlib. A test blocks torch, transformers, nnsight, vllm, pydantic, yaml and
safetensors on both import paths.

```python
FORMAT = 1
WINDOWS = frozenset({"prompt", "response", "all", "last_turn"})

@dataclasses.dataclass
class Probe:
    name: str
    kind: str                  # "linear" | "attention"
    w: np.ndarray              # [H] float32, value / score direction
    bias: float
    block: int                 # residual stream after decoder block `block` (0-indexed)
    window: str                # one of WINDOWS
    pooling: str               # linear: mean | last | max | rolling_mean; attention: "attention"
    threshold: float           # in [0, 1], on the pooled_score scale
    model: str
    engine: str                # "<backend>==<ver>[+<lib>==<ver-or-sha>]"
    model_fingerprint: str | None
    prompt_format: str | None  # fingerprint.prompt_format(...) or "raw"
    q: np.ndarray | None = None          # [H] float32, attention only
    rolling_window: int | None = None    # >= 1 iff pooling == "rolling_mean"
    adapters: tuple[str, ...] = ()
    escalate_threshold: float | None = None   # mechanica's review band; sonde never sets it
    metrics: dict | None = None               # eval card: measured values only
```

**Validation.** `__post_init__` keeps mechanica's checks:
- float32 coercion; finite `w` and `bias`;
- threshold finite and in [0, 1];
- non-empty `name`, `model` and `engine`;
- `adapters` sorted into a tuple;
- `0 ≤ escalate_threshold ≤ threshold`.

It adds:
- `kind` must be valid;
- `q` is present iff `kind == "attention"`, with the same H as `w`;
- `pooling` must be valid for `kind`;
- the `rolling_window` rule above.

Any violation raises `ValueError`.

**Scoring.** All pooling acts on logits, and the sigmoid is applied after pooling.
- `logits(acts [T, H]) -> [T]` is `acts @ w + bias`. Linear only; attention raises.
- `pooled_logit(acts [T, H] | [H]) -> float` pools the logits; `pooled_score` is
  `sigmoid(pooled_logit)`. RL rewards use the logit (`r' = r − λ·logit`, Goodfire's Anatomy of
  Post-Training, 2606.12360) because the sigmoid saturates. The poolings:
  - `mean`, `last`, `max`;
  - `rolling_mean`: max over contiguous stride-1 windows of the window mean. If
    `T < rolling_window`, it is the mean over all T;
  - `attention`: `Σ softmax(acts @ q) · (acts @ w) + bias`.
- `pooled_score_from_logits(logits [T]) -> float` is the linear-only path mechanica's instream
  scorer uses. `pooled_score` is that function applied to `logits(acts)`.
- `flag(acts) -> bool` means `pooled_score >= threshold`, and `escalates(score) -> bool` is as in
  mechanica.
- The sigmoid runs in float32 with clipping to ±30, matching mechanica.

**Other methods and helpers.**
- `serves(fingerprint, adapter, prompt_format) -> str | None`: mechanica's semantics, fail-closed
  on `None`.
- `save(path)` writes atomically (temp file, then `os.replace`). The `.npz` holds the `w` and `q`
  arrays plus a JSON `meta` that includes `format: 1`.
- `load(path)` uses `allow_pickle=False`. It raises `ValueError` on an unknown `format` or on
  meta keys that don't build a `Probe`, so a legacy mechanica npz (`weight`, `layer`,
  `positions`) is refused.
- `load_dir(dir)`: mechanica's semantics. Non-recursive `*.npz`, keyed by name, in filename
  order. It raises on a duplicate name or an empty directory.
- `last_turn_start(tokenizer, messages)`: mechanica's duck-typed rule, used at fit time and by
  mechanica at serve time.
- `vllm_aux_layer() -> int` returns `block + 1`, the vLLM `extract_hidden_states` aux id that
  mechanica uses. This is verified in the vLLM v0.29 and v0.30 Llama, Qwen2 and Qwen3 sources.
  mechanica can serve it natively only on architectures whose vLLM model calls
  `_maybe_add_hidden_state` (Llama, Mistral, Qwen2 and Qwen3 at v0.29; not Gemma 2 or 3).

**Field naming against mechanica.** The field is `block`, not `layer`, so any mechanica call site
missed during migration raises `AttributeError` instead of silently reading the neighbouring
block.

**Engine grammar.** `<backend>==<version>[+<lib>==<ver-or-sha>]`, for example
`vllm==0.30.0+nnsight==b717807` or `hf==5.1.0+nnterp==1.3.0`. The backend is the token before the
first `==`.

**Fingerprint at export.**
`model_fingerprint = checkpoint_fingerprint(cfg.model.name, revision=cfg.model.revision, quantization=None)`.
This uses the revision as requested, never a resolved commit sha, which matches mechanica's
serving computation.

**Layer semantics.** `block = L` is the raw output of decoder block L: nnterp `layers_output[L]`,
or nnsight-vLLM `out = layers[L].output; out[0] + out[1]`. It is pre-norm on both, last block
included.
- HF `hidden_states[L + 1]` matches only for `L < N - 1`; `hidden_states[N]` is post-norm.
- sonde never reads `output_hidden_states`.

## 6. Data (`data.py`)

**Loading.**
- `load_samples(src)` takes a `data` block or a `score` entry. Source is exactly one of
  `path` (local `.jsonl` / `.csv`) or `hf` (with `hf_config` and `hf_split`, default `"train"`).
- Rows have exactly one of a `text` key or a `messages` key, plus `label`, `id` and an optional
  `group` and `response`. All key names are configurable. `id` defaults to the row index.
- `label_map` maps raw values to {0, 1}. After mapping, labels must be in {0, 1}; otherwise the
  load raises and names the row.
- `limit: n` takes a seeded subsample.
- Relative paths resolve against the cwd.

**Rendering.** `render(row, cfg.data, tokenizer) -> Encoded(ids, span)` or `None`, where
`None` means the row is dropped.
- `format: chat`
  - Raises if `tokenizer.chat_template` is None.
  - `prompt_format = fingerprint.prompt_format(tokenizer.chat_template)`: the raw template
    string, no kwargs. This is the same digest mechanica serves under.
  - A `system` prompt and template kwargs are not covered by the digest, so serving must send
    the same system prompt. Template kwargs such as `enable_thinking` are not exposed; template
    defaults apply.
  - A `text` row becomes a single user message, preceded by the system message if one is set.
  - `p = tokenizer(apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids`.
  - `prompt`: span `[0, len(p))`, generation prompt included.
  - `response`: `ids = p + r`, where `r` is the row's `response_ids` if present (from
    `generations.jsonl`), else `tokenizer(response, add_special_tokens=False).input_ids`. Span is
    `[len(p), len(ids))`. No end-of-turn token is appended, which matches mechanica's
    prompt-plus-generated layout.
  - `all`: span `[0, len(ids))`.
  - `last_turn`: render the full conversation with `add_generation_prompt=False`; the last
    message must be the assistant's. Span is `[last_turn_start(tok, msgs), len(ids))`. The row
    is dropped if the prefix ids differ from the full render, which happens with reasoning
    templates.
- `format: raw`: `tokenizer(text).input_ids`, and `prompt_format = "raw"`. `messages` rows are a
  config error.

**Truncation and drops.**
- `max_length` keeps the first `max_length` tokens and clips the span.
- A row is dropped, and counted by reason in the log and the manifest, when its window is empty
  after clipping, its `last_turn` is invalid, or its response is empty.

**Splitting.**
- `split(ids, labels, groups, fractions, seed)` produces stratified train / val / test splits.
- Groups never cross a split (any group-disjoint assignment is allowed).
- One `np.random.default_rng(seed)` drives it.
- It logs the achieved fractions and the per-split `n_pos` / `n_neg`, and raises if a split
  lacks a class.

## 7. Backends (`backends.py`)

There are two backends, `hf` and `vllm`, selected by `cfg.model.backend`. They are never
auto-detected. The code default is `hf`; the shipped GPU recipes say `vllm`. `nnterp` is
imported only inside the hf functions and `nnsight.modeling.vllm` only inside the vllm ones, so
each environment imports cleanly.

**Model loading.**
- `load_model(cfg.model)` is the only loader in the package.
- It keeps a single-slot module cache keyed by `cfg.model.model_dump_json()`, so a run loads
  the model at most once, lazily, at the first step that needs it. A resumed, already-finished
  extract never loads it.
- The layer-range check (`0 ≤ L < num_hidden_layers`) runs here, on first load.

**Forward pass.**

```python
def forward(model, batch: list[Encoded], blocks: list[int], keep: str, pooling: str | None
            ) -> dict[int, list[torch.Tensor]]   # [H] if keep == "pooled" else [T_window, H]
```

`window_pool(h [T, H], span, keep, pooling)` is shared, plain torch, and called by both backends.

*hf* (nnterp 1.3 `StandardizedTransformer`, nnsight 0.7)
- Runs under `torch.no_grad()`, not `inference_mode`: nnsight runs intervention code on another
  thread, and an in-place steering write on an inference tensor raises. This was reproduced.
- Batches are sorted by length and right-padded. The tracer gets an `{input_ids, attention_mask}`
  dict.
- For each block it reads `model.layers_output[L]`, slices each row to its span, and applies
  `window_pool` on the GPU.
- Rows are restored to sample order before being returned.

*vllm* (nnsight `VLLM`, pinned per §12)
- Engine settings:
  - `dtype` comes from config;
  - `enable_prefix_caching=False` and `enable_chunked_prefill=False`;
  - no `taps`, so the engine runs eager;
  - `cfg.model.vllm` is passed through (`gpu_memory_utilization`, `max_model_len`).
  - `ponytail:` eager decode is slower than CUDA graphs. Add `taps` for generate and steer only
    after a Spark run shows it matters and an edit+taps parity check passes, since upstream has
    no test for that combination.
- Extraction:
  - Uses `model.trace(temperature=0.0, max_tokens=1)` with one `tracer.invoke(ids[:end])` per
    row and at most 500 invokes per trace. The model is causal, so rows after the window end
    cannot affect it and are not sent.
  - Inside each invoke, `start` is a plain Python int. For each block in ascending order:
    `h = out[0] + out[1]`, then `nnsight.save(window_pool(h, (start, end), keep, pooling))`,
    then `nnsight.save(h.shape[0])`.
  - Envoys are bound outside the trace. The block never reads `model.logits`, `model.samples`
    or `tracer.result`.
  - It raises if any invoke's row count is not `end`.
- Always uses `nnsight.save(x)`, never `.save()`, because the C extension is optional on aarch64.
- nnsight itself refuses a non-V1 model runner; sonde adds no check of its own.
- Allowlist of vLLM decoder-layer classes whose output is `(mlp_out, residual)`:
  `{LlamaDecoderLayer, Qwen2DecoderLayer, Qwen3DecoderLayer}`. Mistral uses Llama's class.
  Anything else, such as GPT-2 (single tensor), Exaone4 (`(full, post-attn)`) or the
  Transformers backend (`[1, T, H]`), is refused at load. A class is added only with a recorded
  source check.

**Generation (used by §10).**
- `hf_generate` left-pads and passes one `{input_ids, attention_mask}` dict.
  - It builds a `GenerationConfig` from sonde config only: `do_sample = temperature > 0`;
    `temperature` and `top_p` only when sampling; `top_k = 0`; `max_new_tokens = max_tokens`.
  - It calls `torch.manual_seed(seed)` before each batch.
- `vllm_generate` uses an explicit `SamplingParams` (`temperature`, `top_p`, `top_k=-1`,
  `max_tokens`, `seed`). nnsight bypasses `generation_config.json` on vLLM.
- Every vLLM request passes `edits=[...]` explicitly: `[]` for plain generation and
  `["sonde_steer"]` for steering.

## 8. Extraction and storage (`extract.py`, `store.py`)

**What is stored.** `extract.keep` sets this; pooling has a single source of truth.
- `keep: pooled` stores `[H]` per sample, pooled with `probe.pooling`. That pooling must be
  `mean` or `last` and is recorded in the manifest.
- `keep: tokens` stores every window token and leaves pooling to train time, so different
  poolings can be tried without re-extracting.
- The `attention`, `max` and `rolling_mean` probes require `keep: tokens`.
- Values are stored in the dtype the model ran in, and the manifest records it. No second dtype
  knob.

**Shard layout.**
- `ShardWriter` closes a shard at `extract.shard_size` samples or `extract.shard_bytes`
  (default 2 GiB), whichever comes first.
- A shard is written to a temp file and renamed: `extract/shard_{k:05d}.safetensors`.
- Contents: `B{block}` as `[n, H]` for pooled, or as flat `[ΣT, H]` plus `offsets [n + 1]` for
  tokens.
- Shards follow sample order, so sample-to-shard assignment is deterministic across resumes.
- The extract step and `--dry-run` log an estimated disk size:
  `n × mean window tokens × H × n_blocks × dtype bytes`.

**Run metadata and resume.**
- `extract/run.json` holds `config_hash` and is written before the first shard. The hash covers
  the `model` and `data` sections, `extract`, and `probe.pooling` when `keep: pooled`.
- `extract/manifest.json` is written last. It holds `format: 1`, model, revision, fingerprint,
  `prompt_format`, engine, blocks, window, keep, pooling, dtype, the per-sample `ids` / `labels`
  / `groups`, the shard list, drop counts and `config_hash`.
- Resume:
  1. A `run.json` or manifest whose hash differs raises before any model loads, unless
     `output.overwrite` is set; overwrite deletes `extract/` first.
  2. With a matching hash, existing complete shards are skipped.
  3. A matching finished manifest makes the step a no-op.

**Reading.**
- `read_layer(dir, block) -> (X, offsets | None)` reads one block across shards via `safe_open`.
- It raises on an unknown manifest `format`.
- `ponytail:` one tokens-mode block must fit in RAM; stream by shard if a dataset outgrows it.

## 9. Probes, training, sweep (`probes.py`, `train.py`, `sweep.py`)

**Modules.** Each is a torch `forward(x [B, T, H], mask [B, T]) -> logit [B]` that computes the
same thing as `Probe.pooled_score`. Pooling acts on logits, and `rolling_mean` uses the
`T < rolling_window` rule from §5.
- With `keep: pooled`, `x` is `[B, H]`; this applies only to `linear` with `mean` or `last`.
- `export(module, ...) -> Probe` writes numpy arrays.
- A test asserts `Probe.pooled_score == sigmoid(module(x))` within 1e-5 for every kind and
  pooling, including the `T < rolling_window` case.

**Training (`fit`).**
- AdamW, on CUDA if available, else CPU.
- `torch.manual_seed(seed)` runs before the module is built. A local `torch.Generator` drives the
  DataLoader.
- Standardisation:
  - Statistics come from train-split samples for pooled data, or from unmasked train-split tokens
    for token data.
  - `σ = max(std, 1e-6)`, the same in training and export.
  - Export folds it in exactly: `w' = w/σ`, `b' = b − Σ w·μ/σ`. `q` folds the same way, and its
    constant term is dropped because softmax is shift-invariant.
- `pos_weight = n_neg / n_pos`, and weight decay applies to weights only.
- Early stopping on val loss with `patience`; `0` means off. The best epoch is kept in memory.
- `init: diff_means` requires `kind: linear` and `epochs: 0`. It is computed on raw train
  features with no standardisation and no fold: `w = (μ₊ − μ₋) / std(X_train·(μ₊ − μ₋))`,
  `b = −w·(μ₊ + μ₋)/2`. The scaling gives unit-std train logits, so the float32 sigmoid does not
  saturate (unscaled logits reach ±1000 on a residual stream). This
  replaces `directions/`.

**Thresholds** are computed on val, using scores from the **exported numpy `Probe`**, so the
threshold lives on exactly the scale mechanica compares against.
- If `max_fpr` is set (default `0.01`): the lowest threshold whose val FPR is at most `max_fpr`.
  If none exists, record `threshold_failed: true` and fall back to best F1. Never raise.
- Otherwise (`max_fpr: null`): the best-F1 threshold.

**Metrics** are numpy, with no torchmetrics. They are:
- **recall at `max_fpr`**, the headline;
- rank-based AUROC;
- F1, precision, recall and FPR at the threshold;
- `group_auroc` when `data.group` is set: AUROC within each group that has both classes,
  averaged. It measures whether the probe ranks completions *of the same prompt* correctly,
  which is all that GRPO's group-relative advantage uses. This is the headline for reward probes.

Each metric records the `n_pos` / `n_neg` it was computed on (and `n_groups` for
`group_auroc`). The test headline also carries 95% bootstrap CIs: 1,000 resamples with numpy
and a fixed seed, for recall at `max_fpr` and AUROC. When `n_neg × max_fpr < 10` in a split, the
metrics log a warning and record `small_n: true`, because a 1% FPR threshold on 200 negatives
rests on 2 samples.

**Train step (`sweep.py`).**
1. Read the manifest. `data.split` produces the splits, written to `splits.json`.
2. For each extracted block, fit on train and score on val.
3. Select by `probe.select`:
   - `recall_at_fpr` (default), at `max_fpr`; ties break by AUROC. If val is `small_n`, it
     selects by AUROC instead and records `select_fallback: auroc`;
   - `auroc`;
   - `group_auroc` (requires `data.group`), for RL-reward probes.
4. Score test once, on the selected block only.
5. Run the control: refit on train with shuffled labels, then report AUROC on val with the true
   labels as `control_auroc`. `controls_passed = |control_auroc − 0.5| < 0.1` (an inverted AUROC leaks too). Never raise.
6. Fit the text baseline on the same splits, a check that the probe beats surface features
   (Wang et al., 2509.03888):
   - Features: lowercase word counts (`re.findall(r"\w+")`) over the window's text, using a
     vocabulary of the 5,000 most frequent train-split words.
   - The window's text is the prompt for `prompt`, the response for `response`, both for `all`,
     and the last message for `last_turn`.
   - The model is a `LinearProbe` (`pooling: mean`, `keep: pooled`) trained with the same `fit`.
     Its threshold is chosen on val at `max_fpr`, and it is scored on test.
   - A length-confound baseline also runs: the AUROC of the window's word count (the train step
     has no tokenizer). The Silico
     report's first probe looked strong (AUROC 0.98) but was mostly picking up a confound.
   - `ponytail:` bag-of-words and length only; add TF-IDF or an LLM-judge baseline when a
     recipe needs a stronger bar.
7. Write the headline: `metrics.json["headline"]` on test, printed as the last line of the
   step's log. Fields:
   - `max_fpr`;
   - `recall_at_fpr` and `auroc`, each with a CI;
   - `baseline_recall_at_fpr`, `baseline_auroc`, `length_auroc`;
   - `group_auroc` (if set);
   - `n_pos`, `n_neg`, `small_n`.

Outputs:
- `probes/{name}.npz` is the selected probe and the only file in `probes/`, so it is the
  directory to hand to mechanica.
- `layers/B{block}.npz` holds every block's probe, named `{name}@B{block}`.
- `metrics.json` holds the headline, the per-block val table, test results, the baseline,
  the control, threshold info and n counts.

## 10. Score, generation, steering (`score.py`, `generate.py`, `steer.py`)

**Score.** This is the only path for OOD evaluation and for applying an existing probe.
- `score: [{name, probe?, ...data overrides}]`. Each entry is a partial `data` block whose
  unspecified keys inherit from `data`. `split` is ignored.
- `probe` defaults to `probes/{name}.npz` from this run.
- It extracts at the probe's block, window and pooling (`keep` follows from the probe), using
  the same cached model.
- It refuses when the probe's backend token, `prompt_format` or `model_fingerprint` disagrees
  with the run.
- It writes `score/<entry>/{scores.jsonl, metrics.json}`. Metrics are written only when labels
  exist.
- The metrics report the deployed operating point: realized FPR and recall at the probe's
  **frozen** `threshold`, which is what mechanica will see. They also report recall at
  `max_fpr` re-thresholded on that set, AUROC and `group_auroc`.
  - A benign-only set reports FPR only, and a positive-only set reports recall only. Neither
    raises.
  - Benign-only sets are the main measure of overtriggering. In the Silico report, the
    intended ~1% FPR came out as 2.9% on new data.

**Generate.**
- Renders prompts through `data.render`, then calls the backend's generate with `generate`
  params and the top-level `seed`.
- Appends `{id, <text or messages under the input's key>, response, response_ids, group, label?}`
  to `generations.jsonl` as rows finish.
- A rerun skips ids that are already present.
- The file can be used directly as `data.path` with `data.response: response`. `response_ids`
  are reused exactly.

**Steer.**
- Config is `steer: {probe?, mode, strengths}`. `probe` defaults to this run's probe, and the
  vector is `w` at `block`.
- Modes, with `v̂ = w/|w|`:
  - `add`: `x += s·v̂`
  - `ablate`: `x -= s·(x·v̂)·v̂`, where `s = 1` is full directional ablation.
- Prompts come from `data` and sampling from `generate`.
- Output is one `steer/s{strength}.jsonl` per strength. Strength 0 runs with no edit and is the
  baseline.
- Applied inside `tracer.all()`, covering prefill and every decode step.
  - vLLM: for each strength, install `sonde_steer` with the delta, generate with
    `edits=["sonde_steer"]`, then `edit.clear()` in a `finally`. It writes `out[0][:] += …`;
    `ablate` projects on `out[0] + out[1]`.
  - HF: writes nnterp `layers_output[L]` under `torch.no_grad()`.
- `.skip()` is never used.
- This deliberately changes today's behaviour, which steers only the prefill.

## 11. Config, runner, CLI (`config.py`, `runner.py`, `cli.py`)

```yaml
name: refusal-qwen
seed: 0
model:   {name: Qwen/Qwen3-1.7B, revision: null, dtype: bfloat16, backend: vllm,
          vllm: {gpu_memory_utilization: 0.3, max_model_len: 4096}}
data:    {path: data/labeled.jsonl, hf: null, hf_config: null, hf_split: train,
          text: prompt, messages: null, response: null, label: label, id: id,
          group: null, label_map: null, limit: null, format: chat, system: null,
          max_length: 2048, split: [0.7, 0.15, 0.15]}
extract: {layers: all, window: prompt, keep: pooled, batch_size: 16,
          shard_size: 4096, shard_bytes: 2147483648}
probe:   {kind: linear, pooling: mean, rolling_window: null, init: random,
          epochs: 20, lr: 1.0e-3, weight_decay: 0.0, batch_size: 256, patience: 3,
          max_fpr: 0.01, select: recall_at_fpr}
score:   [{name: xstest, path: data/xstest.jsonl}]
generate: {max_tokens: 256, temperature: 0.7, top_p: 0.95}
steer:   {mode: add, strengths: [0, 4, 8]}
output:  {dir: runs, overwrite: false}
steps:   [extract, train, score]
```

**Sections each step needs.** A section is required only if a listed step needs it.

| Step | Sections |
|---|---|
| extract | model, data, extract, probe (when `keep: pooled`) |
| train | data (split), probe |
| score | model, data, score |
| generate | model, data, generate |
| steer | model, data, generate, steer |

**Load-time checks** (pydantic `extra="forbid"`):
- `config.load()` does no I/O beyond reading the YAML.
- Exactly one of `path` / `hf` and one of `text` / `messages`.
- `split` fractions sum to 1.
- `layers` is `all`, a list of non-negative ints (deduplicated and sorted), or `{every: k}`.
- `keep: pooled` requires `kind: linear` with `pooling` in {mean, last}.
- `attention`, `max` and `rolling_mean` require `keep: tokens`.
- `rolling_window ≥ 1` iff `rolling_mean`.
- `select: group_auroc` requires `data.group`.
- `select: recall_at_fpr` (the default) requires `max_fpr`; set `select: auroc` to run with
  `max_fpr: null`.
- `window: response` requires `data.response`.
- `init: diff_means` requires `kind: linear` and `epochs: 0`.
- `temperature ≥ 0`.

**Overrides.** `-o a.b=v` parses `v` with `yaml.safe_load` and sets it on the raw dict before
validation. `load("quickstart")` resolves `sonde/recipes/quickstart.yaml`; an unknown name lists
the available recipes.

**`runner.run(cfg)`:**
1. Validate.
2. If `extract` is in `steps`, run the extract hash check.
3. Create `runs/<name>/` and write the resolved `config.yaml`.
4. Run each step `(cfg, run_dir) -> None` in order.

Steps share files only through the run dir; the model is shared through the `load_model` cache.
Reruns: extract resumes or refuses (§8), generate skips done ids, and train, score and steer
overwrite their own outputs.

**CLI.** `sonde run <cfg|recipe> [-o k=v]... [--dry-run]` is plain argparse. `--dry-run`
validates, prints the resolved YAML and the disk estimate, and exits 0 without loading weights or
creating the run dir. `cli.main` is the `if __name__ == "__main__"`-safe entry point that vLLM's
spawned engine needs.

## 12. Packaging and environments

- `requires-python >= 3.12`. Base dependency: `numpy` only.
- Extras:
  - `[core]`: `transformers`, `safetensors`, `pydantic>=2`, `pyyaml`, `datasets`.
  - `[hf]`: `sonde[core]`, `torch`, `nnterp==1.3.*`, `nnsight>=0.6,<0.8`. For Mac, CI and the
    hf backend.
  - `[vllm]`: `sonde[core]`, `vllm==<pin>`, and
    `nnsight[vllm] @ git+https://github.com/ndif-team/nnsight@b71780727ea9713f74ce12f76b1d3548b91a74a0`.
    vLLM brings its own torch. `<pin>` is the highest vLLM ≥ 0.29 that passes §13 check 1,
    which also matches mechanica's serving floor. Fallback is `0.27.1`, the version nnsight
    tests on.
  - `[dev]`: pinned `pytest`, `ruff` and `pyright`.
- `[tool.uv] conflicts = [[{extra = "hf"}, {extra = "vllm"}]]`, because nnterp 1.3 breaks on
  nnsight 0.8.
- CPU torch comes from a `pytorch-cpu` index source only for
  `sys_platform == 'linux' and platform_machine == 'x86_64'` (CI). The Spark is aarch64 and gets
  CUDA torch.
- The Spark has two venvs: `.venv-hf` (`--extra hf`; confirm CUDA aarch64 torch in the spike) and
  `.venv-vllm` (`--extra vllm`), both from `uv venv --python 3.12 --managed-python`.
- Dropped: `einops`, `torchmetrics`, `omegaconf`.
- The git direct URL blocks a PyPI upload. Pin nnsight `0.8.0` final before publishing.
  mechanica depends only on the numpy base, so it can install from git meanwhile.

## 13. Spark spike (phase 5)

The spike measures and records; it asserts nothing. Results go to `docs/parity-2026-10.md`. Each
side extracts to safetensors shards in its own venv, and one script compares them offline.

1. **Smoke.** Run nnsight's `tests/vllm` (`test_registration`, `test_tracing`,
   `test_chunked_prefill`) at the pinned nnsight commit on vLLM 0.30, falling back to 0.27.1.
   The result picks `<pin>` (§12). If both fail, the vllm backend ships experimental and recipes
   default to `hf` until upstream fixes it.
2. **hf ↔ vllm.** Qwen3-0.6B, 256 prompts, every block, `keep: tokens`. Record `max|Δ|/|x|` and
   mean abs Δ per block, in fp32 and bf16.
3. **sonde-vllm ↔ mechanica native `extract_hidden_states`** (aux id `block + 1`), run in
   mechanica's own vLLM ≥ 0.29 venv on identical token ids. Record both vLLM versions; if they
   differ, the delta mixes engine version and capture path. Compare against mechanica's
   engine-axis band (relative `max|Δ|` 1e-5 to 4e-4 on mid layers).
4. **Throughput,** record only: prompts per second, hf vs vllm, on 10k prompts, mean pooling,
   3 blocks.

## 14. Tests

Plain asserts and few fixtures. GPU tests carry `@pytest.mark.gpu` and are skipped by marker,
never by catching exceptions. `pythonpath` is removed from the pytest config.

| File | Proves |
|---|---|
| `test_probe.py` | round-trip; `serves()` refusals; numpy-only import of `probe` and `fingerprint`; legacy and unknown-format npz refused; validation errors; `pooled_logit` and `pooled_score` agree; `vllm_aux_layer`; `load_dir` duplicate and empty refusals |
| `test_data.py` | chat render has no double BOS; window spans for all four windows; drops are counted; `label_map` and label validation; group-disjoint stratified split |
| `test_extract.py` | **gpt2 batch invariance (bs 1 vs bs 4, allclose)**; shard and manifest layout; resume skips shards; a hash mismatch refuses before load |
| `test_train.py` | threshold at `max_fpr` on numpy scores; recall@FPR; standardisation fold is exact; σ floor; diff-means closed form; seed reproducibility; bag-of-words baseline separates keyword-only synthetic data; `group_auroc` on hand-built groups; bootstrap CI contains the point estimate; `small_n` flag |
| `test_probes.py` | torch forward equals numpy `Probe` for every kind and pooling, including `T < rolling_window` |
| `test_sweep.py` | selects the planted block on synthetic data; control is recorded, not raised; `metrics.json["headline"]` holds probe and baseline recall@FPR; small-n val falls back to AUROC selection |
| `test_config.py` | every shipped recipe loads with no network; unknown-key error names its path; overrides; each cross-field check |
| `test_e2e.py` | `sonde run quickstart` on CPU; gpt2 extract → train → score; score on a benign-only set reports FPR at the frozen threshold without raising; score refuses a mismatched `prompt_format` / fingerprint |
| `test_steer.py` (phase 6) | strength 0 equals unsteered; gpt2 `add` at strength 1 changes the output through the backend's own path; `ablate` removes the projection |
| gpu, Spark | two prompts with different `response` spans give different pooled rows, and each matches hf |

## 15. Repo cleanup

**Recipes.**
- `refusal.yaml`: `data: {path: …/labeled.jsonl, text: original_prompt, label: refusal_label, label_map: {"LABEL: refusal": 1, "LABEL: non-refusal": 0}, group: original_prompt, format: raw}`.
  It uses `format: raw` because the labels came from raw-format continuations, and
  `group: original_prompt` because 83 of the 256 rows are duplicate prompts.
- `high_stakes.yaml`: `model: meta-llama/Llama-3.1-8B`,
  `data: {hf: Arrrlex/models-under-pressure, hf_config: training, text: inputs, label: labels, id: ids, label_map: {high-stakes: 1, low-stakes: 0}, limit: 1000}`,
  `extract: {layers: [10, 13, 15, 17, 20], keep: tokens}`, `probe: {kind: attention, weight_decay: 0.1, patience: 5}`.
  It has one `score` entry per OOD config, all with `hf_split: test`: `anthropic_hh_balanced`,
  `mt_balanced`, `toolace_balanced`, `mental_health_balanced` and `aya_redteaming_balanced`.

**Experiments.** Delete `geometry_of_truth`, `refusal_probe_qwen.py`, both `k8s_job.yaml`,
`rjudge_dissociation`, `scaleJSD_probing` and `examples/causal_loop_gpt2.py`. They remain under
the `v0.1-legacy` tag.

**Data.** `git rm` `refusal_probing/data/activations.safetensors` and `activations_manifest.pt`.
Keep `labeled.jsonl`, with a README note that it was generated without a chat template.

**Docs.**
- Delete `toolkit_audit.md`, `intervention_design.md` and `oumi_integration_plan.md`.
- Move the open items in `research_gaps_and_extensions.md` into §17, then delete it.
- Keep `linear-probes-primer.md` after an accuracy pass.
- Rewrite `README.md` around `sonde run` and the two-venv setup.

**CI.**
- `uv sync --locked --extra hf --extra dev` with CPU torch, installed once.
- Pinned ruff and pyright.
- `pytest -m "not gpu"` on 3.12.
- Cache `~/.cache/huggingface` for gpt2.
- Fix the pre-commit ruff pin.

**Style** (Google Python Style Guide, global CLAUDE.md):
- `line-length = 80`;
- `import module`, never imported names (except `typing` / `collections.abc`);
- Args / Returns docstrings on public APIs only;
- docstrings give array shapes;
- no WHAT comments; `# ponytail:` marks deliberate corners.

## 16. Build phases

Each phase is one PR, and nothing merges until that phase's tests pass. No intermediate working
state is owed (D2).

| # | Phase | Done when |
|---|---|---|
| 1 | Tag `v0.1-legacy`. `git rm` every old package under `sonde/` and every test not in §14. New lazy `__init__.py`. `[project.scripts] sonde = "sonde.cli:main"`. Packaging per §12. Build `probe.py` and `fingerprint.py`. | `test_probe.py` passes; `uv lock` resolves |
| 2 | `config.py` (section models, `load`, overrides, checks), `data.py`, `backends.py` (hf forward), `extract.py`, `store.py` | `test_config.py` (minus recipes), `test_data.py` and `test_extract.py` pass, including batch invariance |
| 3 | `probes.py`, `train.py`, `sweep.py`, `score.py`, `runner.py` (extract, train, score), `cli.py`, quickstart recipe | `sonde run quickstart` on CPU; `test_train`, `test_probes`, `test_sweep` and `test_e2e` pass |
| 4 | Port the recipes; delete experiments, docs and data per §15; CI; README | the tree matches §4; `test_config` loads every recipe; CI is green |
| 5 | vllm backend + Spark spike (§13) | `docs/parity-2026-10.md` has measured numbers; both recipes re-run on the Spark with the chosen backend; each `RESULTS.md` records the commit and backend |
| 6 | `generate.py`, `steer.py`, hf and vllm generate, runner registers generate and steer | `test_steer.py` passes on hf; vllm steer checked on the Spark |

**Out of scope** (separate plan, in the mechanica repo): moving mechanica onto
`from sonde import probe, fingerprint`.
- mechanica deletes its own `probe.py`, `fingerprint.py` and `train.py`, then retrains its probes.
- Field mapping:
  - `weight` → `w`;
  - `layer` (aux id) → `vllm_aux_layer()`;
  - `positions: "last"` → `pooling: "last"`;
  - `positions: "suffix"` is dropped.
- mechanica's exact engine-string checks (`_vllm_plugin._register`, `scoring._checked`,
  `cli calibrate`) must become a backend-token match, with the version gap bounded by §13 check 3.
- Depending on sonde raises mechanica's `requires-python` to ≥ 3.12, which drops vLLM containers
  running 3.10 or 3.11.

## 17. Deferred (add when a config needs it)

- Single-pass capture during generation, to match mechanica's live `response` window exactly
  (its slab is `all_token_ids[:-1]`, computed by KV-cached decode).
- vLLM `taps` for generate and steer throughput.
- `softmax` probes, EMA and multimax aggregators, and Platt calibration (refitting two scalars per
  aggregator).
- Multi-class and regression probes; token-level labels; `generate.n > 1`.
- Multi-layer ensembles, hard-negative or contrastive training, and adversarially trained probes
  (from `research_gaps_and_extensions.md`).
- Recall at fixed FPRs beyond `max_fpr`.
- An automatic steering-efficacy metric; steering across multiple blocks.
- Raw vector files as steering input (today: probe artifacts only).
- nnterp 2 plus `StandardizedVLLM`. This is a migration across both environments: nnsight 0.8 on
  both, accessors `layers[i].layer_output`, and values `[1, T, H]` squeezed before `window_pool`.
- Tensor parallelism, multi-node, NDIF remote.
