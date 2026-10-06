# Configuration

New to sondekit? Start with [getting started](getting-started.md); this page
is the reference.

A run is one YAML file. `sondekit run cfg.yaml` validates it, creates
`<output.dir>/<name>/`, writes the resolved `config.yaml` there, and runs the
listed `steps` in order. A name without a `.yaml` or `.yml` suffix loads a
bundled recipe from `sondekit/recipes/`. Override any key from the command
line with `-o key=value`, where the value is parsed as YAML, for example
`-o probe.max_fpr=0.001 -o "extract.layers=[4, 8]"`. `--dry-run` prints the
resolved config and, when `extract` is a step, a disk estimate computed from
the tokenizer and the model's `config.json` without loading weights.

## Steps

| Step | What it does | Needs |
|---|---|---|
| `extract` | saves residual-stream activations to `extract/` | `model`, `data`, `extract`, and `probe` when `keep: pooled` |
| `train` | fits one probe per layer, keeps the best, and evaluates it once on the test split | `data`, `probe`, and a finished `extract/` |
| `score` | applies a trained probe to each `score` entry | `model`, `data`, `score` |
| `generate` | samples model responses into `generations.jsonl` | `model`, `data`, `generate` |
| `steer` | generates while adding or removing the probe direction | `model`, `data`, `generate`, `steer` |

`score` and `steer` use this run's `probes/<name>.npz` unless an entry or
`steer.probe` names another file, so either list `train` before them or point
at an existing probe.

## Options

| Key | Default | Notes |
|---|---|---|
| `name` | required | run directory is `<output.dir>/<name>` |
| `seed` | `0` | drives the `data.limit` subsample, the split, probe initialization and batch order, the shuffled-label control, bootstrap intervals and the generation sampler |
| `steps` | `[extract, train]` | any of `generate`, `extract`, `train`, `score`, `steer`, run in the listed order |
| `model.name` | required | Hugging Face id or local path |
| `model.revision` | `null` | recorded in the probe's fingerprint |
| `model.dtype` | `bfloat16` | a torch dtype name; the model runs and activations are stored in this dtype |
| `model.backend` | `hf` | `hf` or `vllm`; `vllm` only loads the model so far, and extract, score, generate and steer raise `NotImplementedError` on it |
| `model.vllm` | `{}` | passed to the vLLM engine, e.g. `gpu_memory_utilization` |
| `data.path` / `data.hf` | `null` | exactly one: a local `.jsonl`/`.csv` file or a Hugging Face dataset |
| `data.hf_config`, `data.hf_split` | `null`, `train` | dataset config and split |
| `data.text` / `data.messages` | `text` / `null` | exactly one: a text column or a chat-messages column; setting `messages` clears the `text` default |
| `data.response` | `null` | response column, needed for `window: response`; a `response_ids` column, as `generate` writes, is used instead of re-tokenizing |
| `data.label` | `label` | label column; `extract` and `train` need it on every row, `score` and `generate` do not |
| `data.id` | `id` | id column; a row without it, or `id: null`, uses the row index. Ids must be unique |
| `data.group` | `null` | rows with the same group stay in one split |
| `data.label_map` | `null` | maps raw label values to 0 and 1; any other label is an error |
| `data.limit` | `null` | size of a seeded random subsample |
| `data.format` | `chat` | `chat` applies the tokenizer's chat template; `raw` tokenizes the text as is and cannot use `messages` |
| `data.system` | `null` | system prompt added to text rows under `format: chat` |
| `data.chat_template_kwargs` | `{}` | passed to the chat template, e.g. `{enable_thinking: false}` for Qwen3; serving must pass the same |
| `data.max_length` | `2048` | keeps the first N tokens, capped at the tokenizer's `model_max_length`; rows whose window is cut off entirely are dropped and counted |
| `data.split` | `[0.7, 0.15, 0.15]` | train, validation and test fractions, stratified by label; each split needs both classes |
| `extract.layers` | `all` | `all`, a list of layer indices, or `{every: k}` for layers 0, k, 2k, ... |
| `extract.window` | `prompt` | `prompt`, `response`, `all` (prompt and response) or `last_turn` (the final assistant message; needs `format: chat`) |
| `extract.keep` | `pooled` | `pooled` stores one vector per row; `tokens` stores every token in the window |
| `extract.batch_size` | `16` | rows per forward pass |
| `extract.shard_size`, `extract.shard_bytes` | `4096`, `2147483648` | a shard file closes at whichever limit comes first, checked after each batch |
| `probe.kind` | `linear` | `linear` or `attention` |
| `probe.pooling` | `mean` | `mean`, `last`, `max` or `rolling_mean` for linear probes; `attention` for attention probes |
| `probe.rolling_window` | `null` | window size in tokens, required for `rolling_mean` and not allowed otherwise |
| `probe.init` | `random` | `diff_means` sets the weights to the difference of class means instead of training (needs `kind: linear`, `epochs: 0`) |
| `probe.epochs` | `20` | AdamW epochs; see [Optimizers](#optimizers) |
| `probe.lr` | `1e-2` | AdamW learning rate |
| `probe.weight_decay` | `1e-4` | AdamW weight decay (biases excluded), or the L2 penalty under L-BFGS |
| `probe.batch_size` | `256` | AdamW batch size |
| `probe.patience` | `3` | AdamW early stopping on validation loss; `0` disables it |
| `probe.max_fpr` | `0.01` | the threshold is the lowest one with validation FPR at or below this; `null` uses best F1 and needs `select: auroc` or `group_auroc` |
| `probe.select` | `recall_at_fpr` | how the layer is chosen: `recall_at_fpr`, `auroc`, or `group_auroc` (needs `data.group`) |
| `score` | `[]` | list of entries, see [Score entries](#score-entries) |
| `generate.max_tokens` | `256` | new tokens per response |
| `generate.temperature`, `generate.top_p` | `0.7`, `1.0` | `temperature: 0` is greedy |
| `steer.probe` | `null` | probe path; defaults to this run's `probes/<name>.npz` |
| `steer.mode` | `add` | `add` adds `s·v`; `ablate` removes `s` times the projection on `v` |
| `steer.strengths` | `[0.0]` | one output file per strength; strength 0 is the unedited baseline |
| `output.dir` | `runs` | parent of the run directory |
| `output.overwrite` | `false` | see [Rerunning](#rerunning) |

The config is checked when it loads. Unknown keys and invalid combinations
fail with a message that names the key. For example, `keep: pooled` works
only with linear `mean` or `last` probes, and `attention`, `max` and
`rolling_mean` need `keep: tokens`. Layer indices outside the model fail when
the model's depth is known, at `--dry-run` or at the start of `extract`.

## Optimizers

Which optimizer fits a probe depends on how the activations were stored:

| Probe | Fit | Options used |
|---|---|---|
| linear, `keep: pooled` | full-batch L-BFGS on standardized features, up to 500 iterations | `weight_decay` as an L2 penalty on the weights |
| linear, `keep: tokens`, and attention | AdamW over shuffled batches, keeping the epoch with the lowest validation loss | `epochs`, `lr`, `weight_decay`, `batch_size`, `patience` |
| `init: diff_means` | difference of class means, scaled so train logits have unit standard deviation | none |

Both trained fits weight the positive class by `n_neg / n_pos`. Training runs
on CUDA when it is available.

## Score entries

Each entry under `score` names a dataset to run a probe on:

| Key | Notes |
|---|---|
| `name` | required; outputs go to `score/<name>/` |
| `probe` | probe path; defaults to this run's `probes/<name>.npz` |
| any `data` key | overrides that key of `data`; unset keys are inherited |

Setting `path` or `hf` replaces both inherited keys of that pair, and the same
holds for `text` and `messages`. Set a key to `null` to clear an inherited
value, as the `high_stakes` recipe does with `limit: null`. `split` is
accepted but ignored, because every row is scored. The window, layer and
pooling come from the probe. Batch and shard sizes come from `extract`, or
its defaults when the run has no `extract` section.

`score` refuses a probe whose backend, model fingerprint or prompt format
differs from this run's. Each entry writes `scores.jsonl` (`id`, `score`,
`flag`) and, when every row has a label, `metrics.json`.

## Run outputs

The quickstart runs `extract`, `train` and `score`, so it writes the first
seven entries; `generate` and `steer` write the last three:

```
runs/<name>/
  config.yaml              the resolved config
  extract/                 run.json, shard_*.safetensors and manifest.json
  splits.json              train, validation and test ids
  layers/B<layer>.npz      the probe trained at each layer
  probes/<name>.npz        the selected probe
  metrics.json             evaluation of the selected probe
  score/<entry>/           extract/, scores.jsonl and metrics.json
  generate.json            hash of the config that wrote generations.jsonl
  generations.jsonl        generated responses with their token ids
  steer/s<strength>.jsonl  steered generations, one file per strength
```

A row of `generations.jsonl` or a `steer/` file is the source row with its
id under the `data.id` key, plus `response` and `response_ids`, the exact
generated token ids. Steer files are named after the strength, e.g. `s0.jsonl`, `s4.jsonl` and `s-0.5.jsonl` for
strengths 0, 4 and -0.5.

## Metrics

`train` writes `metrics.json` with the headline on the test split, the
selected layer, the per-layer validation table, the test and bag-of-words
baseline metrics, and the shuffled-label control. The same dictionary is
stored in the probe file. [Evaluation](evaluation.md#metricsjson) describes
every field and how to read it.

## Rerunning

Running the same config again continues where it stopped:

- `extract` skips when its manifest is complete and otherwise resumes after
  the last complete shard. If the `model`, `data` or `extract` section
  changed (including `extract.batch_size`), or `probe.pooling` under
  `keep: pooled`, or `seed` under `data.limit`, it stops before loading the
  model.
- `generate` skips rows whose id is already in `generations.jsonl`, and stops
  if the `model`, `data` or `generate` section or the seed changed.
- `train`, `score` and `steer` replace their outputs.

`output.overwrite: true` deletes `extract/` and re-extracts on every run
while it is set, and deletes `generations.jsonl` when the generate config
changed. It leaves the rest of the run directory in place, so changing
`name` is the way to start clean.

## Layer numbering

Layer `L` is the residual stream right after decoder block `L`, counting
from 0 and before the final norm. Logs, `metrics.json` and the probe file
call it the block (`block`, `layers/B<layer>.npz`). The same tensor appears
under different names:

| Where | Tensor |
|---|---|
| nnterp | `model.layers_output[L]` |
| nnsight on vLLM | `out = model.model.layers[L].output; out[0] + out[1]` |
| vLLM `extract_hidden_states` | auxiliary layer `L + 1`, see `Probe.vllm_aux_layer()` |
| transformers `output_hidden_states` | `hidden_states[L + 1]`, except for the last layer, where transformers applies the final norm |

[Python API](python-api.md#getting-the-right-activations) shows how to read
this tensor when serving a probe.
