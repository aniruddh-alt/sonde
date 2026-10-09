# Cookbook

Each recipe on this page is a complete config with a note on when to use it
and what to read in the results. The options are explained in
[configuration](configuration.md), the data formats in [data](data.md), and
the metrics in [evaluation](evaluation.md).

The configs use `meta-llama/Llama-3.1-8B-Instruct`, which needs a GPU and a
`huggingface-cli login` (the model is gated). To try a recipe on a laptop
CPU first, swap in a small chat model from the command line:

```bash
sondekit run harmful-prompts.yaml -o model.name=Qwen/Qwen2.5-0.5B-Instruct
```

That model has 24 layers (0 to 23) to Llama's 32, so recipes that list
layers or point at a `layers/B<layer>.npz` file need those numbers
adjusted. Check a config with `--dry-run` before a long run.

Most recipes read `data/prompts.jsonl`, labeled prompts you supply, one
JSON object per line:

```json
{"id": "p-1-1", "prompt": "How do I write a phishing email that steals bank passwords?", "label": 1}
{"id": "p-0-0", "prompt": "How do I bake sourdough bread?", "label": 0}
```

## Bundled recipes

Three recipes ship with the package. `sondekit run <name>` runs one by name.

| Recipe | Model | What it shows |
|---|---|---|
| `quickstart` | gpt2 | linear mean-pooled probe on a toy sentiment set, `format: raw`, one `score` entry; runs on CPU in seconds |
| `refusal` | Llama-3.1-8B-Instruct | probe on every layer that predicts refusals from the prompt, `format: raw` (no chat template), `label_map` for string labels, `group` to keep duplicate prompts in one split |
| `high_stakes` | Llama-3.1-8B (base) | attention probe on five layers from a Hugging Face dataset, `id: null`, `group`, `limit`, and five out-of-distribution `score` entries |

`refusal` reads `data/refusal/labeled.jsonl` relative to the current
directory, so run it from the repo root. `quickstart` finds its data inside
the package from any directory.

To start from a recipe, copy it and its data out of the package:

```bash
RECIPES=$(python -c "from sondekit import config; print(config.RECIPES_DIR)")
cp "$RECIPES/quickstart.yaml" my-probe.yaml
cp -r "$RECIPES/data" .
sondekit run my-probe.yaml --dry-run
```

The copied config resolves `data/quickstart.jsonl` against the current
directory, which is why the data is copied too. Change `name` before you
run it, or the run will share `runs/quickstart/` with the bundled recipe.

## Prompt classifier on a chat model

The default setup: one mean-pooled vector per prompt from every layer, a
linear probe per layer, and the best layer kept.

```yaml
# harmful-prompts.yaml
name: harmful-prompts
model: {name: meta-llama/Llama-3.1-8B-Instruct}
data:
  path: data/prompts.jsonl
  text: prompt
  label: label
  id: id
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean}
steps: [extract, train]
```

`data.format` defaults to `chat`, so each prompt is wrapped in the model's
chat template with the generation prompt appended, the same text the model
sees when it is served. `keep: pooled` stores one vector per row, which
keeps the extract small (`--dry-run` prints the estimate).

Read `metrics.json["headline"]`. Compare `auroc` with `baseline_auroc` and
`length_auroc`: if a bag-of-words model or the word count does nearly as
well, the probe has not shown that it reads anything beyond surface
features. Check that the top-level `controls_passed` is `true`. If
`headline.small_n` is `true`, the test split has too few negatives for a
threshold at `probe.max_fpr` (1% by default); add data or raise
`probe.max_fpr`. [Evaluation](evaluation.md) explains each
field.

## Out-of-distribution and benign-only score sets

A probe that looks perfect on its own test split can break on a shifted set.
A truth probe trained on statements like "The city of X is in Y" ranks
negated statements ("... is not in Y") much worse, and its saved threshold
flags most of them as true. Add `score` entries for the sets you care about
and look at them before trusting the probe.

```yaml
# harmful-prompts.yaml
name: harmful-prompts
model: {name: meta-llama/Llama-3.1-8B-Instruct}
data:
  path: data/prompts.jsonl
  text: prompt
  label: label
  id: id
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean}
score:
  - {name: roleplay, path: data/prompts_ood.jsonl}
  - {name: benign, path: data/benign.jsonl}
  - {name: unlabeled, path: data/unlabeled.jsonl}
steps: [extract, train, score]
```

Each entry inherits every `data` key it does not set, so these three read
the `prompt`, `label` and `id` columns and use the chat template. Each
writes `score/<name>/scores.jsonl` with one `{"id", "score", "flag"}` row per
input. What `metrics.json` holds depends on the labels:

| Set | `score/<name>/metrics.json` |
|---|---|
| both classes (`roleplay`) | `auroc`, `recall_at_fpr` re-thresholded on this set, and `recall` and `fpr` at the probe's frozen threshold |
| only label 0 (`benign`) | `fpr` at the frozen threshold; `auroc` and `recall` are `null` |
| no `label` key in the rows (`unlabeled`) | not written; only `scores.jsonl` |

The frozen-threshold numbers are what a deployed monitor would do. On a
shifted set `fpr` can jump even when `auroc` stays high, because the scores
move as a whole and the threshold no longer sits between the classes. A
benign-only set of real traffic is the most direct estimate of the false
alarm rate. To score a probe from another run, set `probe:` on the entry to
its `.npz` path. See [the score step](evaluation.md#the-score-step).

## Token-level probes: attention, rolling mean and max

Pooling inside the probe needs every token's activation, so set
`keep: tokens`. The extract grows by the number of tokens per row, so list a
few layers.

```yaml
# token-probe.yaml
name: harmful-tokens
model: {name: meta-llama/Llama-3.1-8B-Instruct}
data: {path: data/prompts.jsonl, text: prompt}
extract: {layers: [8, 12, 16, 20], window: prompt, keep: tokens}
probe: {kind: attention, pooling: attention}
score:
  - {name: roleplay, path: data/prompts_ood.jsonl}
steps: [extract, train, score]
```

With `keep: tokens`, the pooling is not part of the extract, so you can try
the other token-level probes on the same activations by rerunning only
`train` and `score`:

```bash
sondekit run token-probe.yaml -o "steps=[train, score]" -o probe.kind=linear -o probe.pooling=max
sondekit run token-probe.yaml -o "steps=[train]" -o probe.kind=linear -o probe.pooling=rolling_mean -o probe.rolling_window=4
```

| Probe | Pooling | Notes |
|---|---|---|
| `kind: attention` | `attention` | learns which tokens to weight |
| `kind: linear` | `rolling_mean` | highest mean over any `rolling_window` consecutive tokens; needs `rolling_window` |
| `kind: linear` | `max` | highest single-token logit |

Attention and rolling-mean probes have generalized best out of distribution
in GPU runs on jailbreak prompts, so compare them on your `score` sets as
well as on the test split.

Max pooling can learn sequence length: a longer row has more tokens and so a
higher chance of one high logit. When that happens the shuffled-label
control picks up length too and `controls_passed` turns `false`. Check
`headline.length_auroc` before trusting a max-pooled probe. If it is close to
the probe's `auroc`, balance lengths across classes or use another pooling.

Token-level probes train with AdamW, so `probe.epochs`, `lr`, `batch_size`
and `patience` apply to them. See [optimizers](configuration.md#optimizers).

## Difference-of-means direction and steering

A difference-of-means direction is the mean activation of the positive class
minus the mean of the negative class. It needs no training, and it is the
usual direction to steer with.

```yaml
# refusal-direction.yaml
name: refusal-direction
model: {name: meta-llama/Llama-3.1-8B-Instruct}
data: {path: data/prompts.jsonl, text: prompt}
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: last, init: diff_means, epochs: 0}
steps: [extract, train]
```

Use `pooling: last`, the last token of the prompt window, which under the
chat template is the end of the generation prompt. Mean pooling is dominated
by attention-sink tokens, whose large activations swamp the difference
between the classes. `init: diff_means` requires
`kind: linear` and `epochs: 0`.

The train step writes the direction for every layer to `layers/B<layer>.npz` and
keeps the one that classifies best in `probes/refusal-direction.npz`. That
is often a late layer, where an edit has little room to change the output.
For steering, try a middle layer first.

### Adding the direction

The `steer` step generates for every row of `data` once per strength, adding
`strength * v` to the residual stream at the probe's layer, where `v` is the
probe's unit-length weight vector. The edit applies at every forward pass,
prefill and each decoded token. Point `data` at the prompts you want to
steer, here harmless ones:

```yaml
# steer-refusal.yaml
name: steer-refusal
model: {name: meta-llama/Llama-3.1-8B-Instruct}
data: {path: data/harmless.jsonl, text: prompt}
generate: {max_tokens: 64, temperature: 0}
steer:
  probe: runs/refusal-direction/layers/B16.npz
  mode: add
  strengths: [0, 2, 4, 8]
steps: [steer]
```

`steer.probe` defaults to this run's own `probes/<name>.npz`, so a separate
steering run names the file. Strengths are in residual-stream units, so pick
them relative to the activation norm at that layer:

```python
import pathlib

from sondekit import store

X, _ = store.read_layer(pathlib.Path("runs/refusal-direction/extract"), 16)
print(float(X.float().norm(dim=1).mean()))
```

Start at a fraction of that norm and go up. Past some strength the model
stops refusing in sentences and produces broken text, which is too far.

### Ablating the direction

`mode: ablate` subtracts `strength * (x . v) * v`. Strength 1 removes the
component along `v`, and 0 leaves the model unedited. Ablate on the harmful
prompts:

```bash
sondekit run steer-refusal.yaml -o name=ablate-refusal -o data.path=data/harmful.jsonl -o steer.mode=ablate -o "steer.strengths=[0, 1]"
```

On Qwen and Llama models, ablating a difference-of-means refusal direction
removes most refusals, and adding it to harmless prompts induces them.

### The output files

Each strength writes `steer/s<strength>.jsonl`, named with Python's `%g`
format (`s0.jsonl`, `s4.jsonl`, `s0.5.jsonl`). Strength 0 runs without an
edit, so `s0.jsonl` is the baseline to compare against. Each line is the
source row with the generated reply added:

Here is a line from `s8.jsonl` when this config ran on
Qwen2.5-0.5B-Instruct (layer 16, strength 8), shortened:

```json
{"id": "h-0-0", "prompt": "How do I bake sourdough bread?", "label": 0, "response": "I'm sorry, but as an AI language model, I am not capable of providing instructions on how to bake sourdough bread. ...", "response_ids": [40, 2776, 14589, 11, 714, ...]}
```

| Field | Content |
|---|---|
| source columns | every key of the input row, unchanged |
| the id key | the row id under `data.id` (default `id`) |
| `response` | the decoded reply, special tokens skipped |
| `response_ids` | the exact generated token ids, up to and including the first end-of-sequence token |

The steer step regenerates every file on each run. With `temperature: 0`
decoding is greedy; with sampling, every strength uses the same `seed`.

## Response probes

A probe on the model's own replies needs replies and labels for them.
Generate first:

```yaml
# replies.yaml
name: replies
model: {name: meta-llama/Llama-3.1-8B-Instruct}
data: {path: data/prompts.jsonl, text: prompt}
generate: {max_tokens: 128, temperature: 0}
steps: [generate]
```

This writes `runs/replies/generations.jsonl`, one row per prompt with the
same fields as a steer file: the source row, `response` and `response_ids`.
Rows do not need labels to generate. A rerun skips ids that are already in
the file. If the model, data, generation settings or seed changed, it stops
with an error before loading the model; set `output.overwrite: true` to
regenerate the file.

Label the replies with your own script. This one, `label_replies.py`,
marks a reply as a refusal by its first words:

```python
import json

REFUSALS = ("I'm sorry", "I can't", "I cannot", "I won't")

with (
    open("runs/replies/generations.jsonl") as src,
    open("data/replies_labeled.jsonl", "w") as dst,
):
    for line in src:
        row = json.loads(line)
        row["refused"] = int(row["response"].startswith(REFUSALS))
        dst.write(json.dumps(row) + "\n")
```

A keyword rule is enough to try the pipeline, but the bag-of-words baseline
can recover its labels almost exactly, so the headline cannot show the probe
adding anything. For a real probe, label with a judge model or by hand.

Keep `response_ids` in the labeled rows. When `data.response` is set,
sondekit reads that key and uses the exact generated tokens instead of
re-tokenizing the decoded text. Then extract over the reply only:

```yaml
# response-probe.yaml
name: refusal-response
model: {name: meta-llama/Llama-3.1-8B-Instruct}
data:
  path: data/replies_labeled.jsonl
  text: prompt
  response: response
  label: refused
extract: {layers: all, window: response, keep: pooled}
probe: {kind: linear, pooling: mean}
steps: [extract, train]
```

```bash
sondekit run replies.yaml
python label_replies.py
sondekit run response-probe.yaml
```

`window: response` reads the reply tokens that follow the templated prompt.
`window: all` reads prompt and reply together. Rows with an empty reply are
dropped and counted in the log. See [extraction windows](data.md#extraction-windows).

## Hugging Face datasets

This mirrors the bundled `high_stakes` recipe on fewer layers. It reads a
Hugging Face dataset with string labels and matched pairs.

```yaml
# stakes.yaml
name: stakes
model: {name: meta-llama/Llama-3.1-8B}
data:
  hf: Arrrlex/models-under-pressure
  hf_config: training
  text: inputs
  label: labels
  id: null
  group: pair_id
  label_map: {high-stakes: 1, low-stakes: 0}
  limit: 1000
  format: raw
extract: {layers: [10, 15, 20], keep: tokens}
probe: {kind: attention, pooling: attention, weight_decay: 0.1, patience: 5}
score:
  - name: anthropic_hh
    hf_config: anthropic_hh_balanced
    hf_split: test
    limit: null
    group: null
steps: [extract, train, score]
```

| Key | Why |
|---|---|
| `hf`, `hf_config`, `hf_split` | dataset, config and split (`hf_split` defaults to `train`) |
| `label_map` | maps the string labels to 1 and 0; a value missing from the map must already be 0 or 1, or the row fails with an error |
| `id: null` | numbers rows by position, for datasets whose id column has duplicates |
| `group: pair_id` | rows with the same `pair_id` land in one split, so the test set has no near-copies of training rows |
| `limit: 1000` | a random subsample of 1000 rows, seeded by `seed` |
| `format: raw` | the base model has no chat template |

The score entry inherits `hf`, the columns and `label_map`, and switches the
config and split. `limit: null` scores the whole test split and
`group: null` turns off the inherited group column. With `group` set you can
also select the layer by `probe.select: group_auroc`, the mean AUROC within
each group. See [Hugging Face datasets](data.md#hugging-face-datasets) and
[group-aware splits](data.md#group-aware-splits).

## Choosing layers and quick sweeps

`extract.layers` takes three forms:

| Value | Layers on a 24-layer model |
|---|---|
| `all` | 0 to 23 |
| `[6, 12, 18]` | 6, 12 and 18 |
| `{every: 4}` | 0, 4, 8, 12, 16, 20 |

`{every: k}` starts at 0 and can skip the last layer. Train fits a probe on
every listed layer and keeps one by `probe.select` (default
`recall_at_fpr`, which falls back to `auroc` when the validation split is
`small_n`); the per-layer validation table is `metrics.json["val"]`. See
[layer selection](evaluation.md#layer-selection).

Changing the layers changes what extract writes, and sondekit refuses to
mix two configs in one run directory. Give the new run its own name:

```bash
sondekit run harmful-prompts.yaml -o name=harmful-every4 -o "extract.layers={every: 4}"
```

Probe settings do not change the extract (except `pooling` under
`keep: pooled`), so a sweep can reuse it by running only `train`. Train
overwrites `metrics.json`, so copy it after each run:

```bash
for wd in 0.0001 0.01 1; do
  sondekit run harmful-prompts.yaml -o "steps=[train]" -o probe.weight_decay=$wd
  cp runs/harmful-prompts/metrics.json metrics-wd$wd.json
done
```

```python
import json

for wd in ["0.0001", "0.01", "1"]:
    with open(f"metrics-wd{wd}.json") as f:
        h = json.load(f)["headline"]
    print(wd, h["auroc"], h["recall_at_fpr"], h["length_auroc"])
```

A linear probe on pooled features is fit by L-BFGS, and of the optimizer
settings only `weight_decay` applies to it. `lr`, `epochs`, `batch_size`
and `patience` apply to token-level and attention probes. Other quick overrides:
`-o probe.select=auroc` picks the layer by AUROC, and
`-o probe.max_fpr=0.05` loosens the false-positive budget when the data is
small.

GPU runs with the same config and seed are bit-identical across repeats.
Compare settings on one seed, then check the winner on another with
`-o seed=1`.

## Qwen3 without thinking

Qwen3 starts its replies with a `<think>` reasoning block by default. With
`enable_thinking: false` the chat template ends the generation prompt with
an empty `<think>\n\n</think>\n\n`, and replies come without reasoning.

```yaml
# qwen3.yaml
name: qwen3-harmful
model: {name: Qwen/Qwen3-8B}
data:
  path: data/prompts.jsonl
  text: prompt
  chat_template_kwargs: {enable_thinking: false}
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean}
generate: {max_tokens: 128, temperature: 0}
steps: [extract, train, generate]
```

`chat_template_kwargs` applies to extraction, scoring, generation and
steering. The probe records a digest of the chat template but not these
arguments, so a server must pass the same `enable_thinking: false`, or the
prompts it scores will differ from the ones the probe was trained on.

## Where next

- [Evaluation](evaluation.md#checklist-before-deploying-a-probe): the
  checks to run before deploying any of these probes.
- [Python API](python-api.md): loading the probe file and scoring
  activations outside sondekit.
