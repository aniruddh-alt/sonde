# Configuration

A run is one YAML file. `sondekit run cfg.yaml` validates it, creates
`runs/<name>/`, and runs the listed `steps` in order. Override any key from
the command line with `-o key=value`, for example
`-o probe.max_fpr=0.001 -o "extract.layers=[4, 8]"`.

## Steps

| Step | What it does | Needs |
|---|---|---|
| `extract` | saves residual-stream activations to `extract/` | `model`, `data`, `extract`, and `probe` when `keep: pooled` |
| `train` | fits one probe per layer and keeps the best | `data`, `probe` |
| `score` | applies a probe to other datasets | `model`, `data`, `score` |
| `generate` | samples model responses into `generations.jsonl` | `model`, `data`, `generate` |
| `steer` | generates while adding or removing the probe direction | `model`, `data`, `generate`, `steer` |

## Options

| Key | Default | Notes |
|---|---|---|
| `name` | required | run directory is `<output.dir>/<name>` |
| `seed` | `0` | drives subsampling, splits, initialization and data order |
| `steps` | `[extract, train]` | any of `generate`, `extract`, `train`, `score`, `steer` |
| `model.name` | required | Hugging Face id or local path |
| `model.revision` | `null` | recorded in the probe's fingerprint |
| `model.dtype` | `bfloat16` | activations are stored in this dtype |
| `model.backend` | `hf` | `hf` or `vllm` (experimental) |
| `model.vllm` | `{}` | passed to the vLLM engine, e.g. `gpu_memory_utilization` |
| `data.path` / `data.hf` | `null` | exactly one: a local `.jsonl`/`.csv` file or a Hugging Face dataset |
| `data.hf_config`, `data.hf_split` | `null`, `train` | dataset config and split |
| `data.text` / `data.messages` | `text` / `null` | exactly one: a text column or a chat-messages column |
| `data.response` | `null` | response column, needed for `window: response` |
| `data.label`, `data.id`, `data.group` | `label`, `id`, `null` | column names; rows with the same group stay in one split |
| `data.label_map` | `null` | maps raw label values to 0 and 1 |
| `data.limit` | `null` | size of a seeded random subsample |
| `data.format` | `chat` | `chat` applies the tokenizer's chat template; `raw` does not |
| `data.system` | `null` | system prompt for `format: chat` |
| `data.chat_template_kwargs` | `{}` | passed to the chat template, e.g. `{enable_thinking: false}` for Qwen3; serving must pass the same |
| `data.max_length` | `2048` | keeps the first N tokens; rows whose window is cut off entirely are dropped and counted |
| `data.split` | `[0.7, 0.15, 0.15]` | train, validation and test fractions, stratified by label |
| `extract.layers` | `all` | `all`, a list of layer indices, or `{every: k}` |
| `extract.window` | `prompt` | `prompt`, `response`, `all` or `last_turn` |
| `extract.keep` | `pooled` | `pooled` stores one vector per row; `tokens` stores every token in the window |
| `extract.batch_size` | `16` | |
| `extract.shard_size`, `extract.shard_bytes` | `4096`, `2147483648` | a shard file closes at whichever limit comes first |
| `probe.kind` | `linear` | `linear` or `attention` |
| `probe.pooling` | `mean` | `mean`, `last`, `max` or `rolling_mean` for linear probes; `attention` for attention probes |
| `probe.rolling_window` | `null` | window size, required for `rolling_mean` |
| `probe.init` | `random` | `diff_means` fits a difference-of-means direction (linear, `epochs: 0`) |
| `probe.epochs`, `probe.lr`, `probe.weight_decay`, `probe.batch_size` | `20`, `1e-3`, `0.0`, `256` | AdamW |
| `probe.patience` | `3` | early stopping on validation loss; `0` disables it |
| `probe.max_fpr` | `0.01` | the threshold is the lowest one with validation FPR at or below this; `null` uses best F1 |
| `probe.select` | `recall_at_fpr` | how the layer is chosen: `recall_at_fpr`, `auroc`, or `group_auroc` (needs `data.group`) |
| `score` | `[]` | list of `{name, probe, <any data key>}`; unset keys come from `data` |
| `generate.max_tokens`, `generate.temperature`, `generate.top_p` | `256`, `0.7`, `1.0` | `temperature: 0` is greedy |
| `steer.probe` | `null` | defaults to this run's probe |
| `steer.mode`, `steer.strengths` | `add`, `[0.0]` | `add` adds `s·v`; `ablate` removes `s` times the projection on `v`; strength 0 is the baseline |
| `output.dir`, `output.overwrite` | `runs`, `false` | |

The config is checked when it loads. Unknown keys and invalid combinations
fail with a message that names the key. For example, `keep: pooled` works
only with linear `mean` or `last` probes, and `attention`, `max` and
`rolling_mean` need `keep: tokens`.

## Run outputs

```
runs/<name>/
  config.yaml              the resolved config
  extract/                 activation shards and manifest.json
  splits.json              train, validation and test indices
  probes/<name>.npz        the selected probe
  layers/B<layer>.npz      the probe trained at each layer
  metrics.json             headline, per-layer validation table, test, baselines, control
  score/<entry>/           scores.jsonl and metrics.json
  generations.jsonl        generated responses with their token ids
  steer/s<strength>.jsonl  steered generations, one file per strength
```

`metrics.json["headline"]` reports, on the test split, recall at `max_fpr`
and AUROC with 95% bootstrap intervals, recall and FPR at the saved
threshold, the bag-of-words and length baselines, and the number of positive
and negative examples. `small_n: true` marks a split with too few negatives
for a reliable 1% FPR threshold. When the validation split is that small,
the layer is chosen by AUROC instead.

## Rerunning

Running the same config again continues where it stopped:

- `extract` resumes from its last complete shard. If the model, data or
  extraction settings changed, it stops before loading the model.
- `generate` skips rows that are already in `generations.jsonl`, and stops
  if the model, data, generation settings or seed changed.
- `train`, `score` and `steer` overwrite their outputs.

Set `output.overwrite: true` to start a run over.

## Layer numbering

Layer `L` is the residual stream right after decoder block `L`, counting
from 0 and before the final norm. The same tensor appears under different
names:

| Where | Tensor |
|---|---|
| nnterp | `model.layers_output[L]` |
| nnsight on vLLM | `out = model.model.layers[L].output; out[0] + out[1]` |
| vLLM `extract_hidden_states` | auxiliary layer `L + 1`, see `Probe.vllm_aux_layer()` |
| transformers `output_hidden_states` | `hidden_states[L + 1]`, except for the last layer, where transformers applies the final norm |
