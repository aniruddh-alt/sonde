# Getting started

This page takes you from install to a probe trained on your own data. It
runs the bundled quickstart, walks through the run directory it writes,
then builds a config from a `.jsonl` file step by step. The last sections
cover reruns and the errors you are most likely to see.

## Install

sondekit needs Python 3.12 or newer. What you install depends on what the
machine does:

| Install | Pulls in | Use it for |
|---|---|---|
| `pip install sondekit` | numpy | loading a trained probe and scoring activations you already have |
| `pip install "sondekit[hf]"` | torch, transformers, nnsight, nnterp, pydantic, pyyaml, datasets, safetensors | the `sondekit` command: extracting, training, scoring, generating, steering |
| `pip install "sondekit[vllm]"` | vLLM 0.30.0 and nnsight's vLLM support (Linux only) | experimental, see below |

The `core` extra holds the dependencies that `hf` and `vllm` share. You do
not need to install it on its own.

To run the bundled recipes or work on sondekit itself, install from source
with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/aniruddh-alt/sondekit.git
cd sondekit
uv sync --extra hf
```

The `hf` and `vllm` extras are declared as conflicting, and uv refuses to
install both into one environment. On a GPU machine that needs both, give
each its own environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-hf uv sync --locked --extra hf
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --locked --extra vllm
```

The `vllm` backend can load a model, but extraction and generation on it
are not implemented yet and raise `NotImplementedError`. Use the default
`hf` backend for real runs. It works on CPU or GPU.

Gated models such as `meta-llama/Llama-3.1-8B-Instruct` need a Hugging
Face token with access to the model. Request access on the model page,
then run `huggingface-cli login` once on the machine.

## Run the quickstart

```bash
sondekit run quickstart
```

`quickstart` is a bundled recipe. It trains a sentiment probe on gpt2 using
64 short sentences shipped with the package, labeled 1 for positive and 0
for negative. Here is the recipe:

```yaml
name: quickstart
model: {name: gpt2, dtype: float32}
data:
  path: data/quickstart.jsonl
  format: raw
  split: [0.5, 0.25, 0.25]
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean, epochs: 30, batch_size: 16}
score:
  - {name: heldout, path: data/quickstart_heldout.jsonl}
steps: [extract, train, score]
```

Read top to bottom, it says:

- Load gpt2 in float32. gpt2 has no chat template, so `format: raw` feeds
  the text to the tokenizer as is.
- Split the 64 rows 50/25/25 into train, validation and test, stratified
  by label.
- `extract`: run every row through the model and save the residual stream
  after each of gpt2's 12 layers, mean-pooled over the prompt tokens
  (`keep: pooled` stores one vector per row and layer).
- `train`: fit a linear probe at every layer, pick the best layer on the
  validation split, and evaluate it once on the test split.
- `score`: apply the chosen probe to a second file of 32 sentences written
  in a different style.

It takes a few seconds on a laptop CPU. The log goes to stderr. With the
per-layer lines for layers 1 to 11 and the repeated `small_n` warnings
cut, it looks like this:

```
Loading weights: 100%|██████████| 148/148 [00:00<00:00, 35807.39it/s]
INFO sondekit.extract: extract: 64 samples x 12 blocks, ~0.00 GB
INFO sondekit.data: split train: n=32 (0.500) n_pos=16 n_neg=16
INFO sondekit.data: split val: n=16 (0.250) n_pos=8 n_neg=8
INFO sondekit.data: split test: n=16 (0.250) n_pos=8 n_neg=8
WARNING sondekit.train: small_n: 8 negatives x max_fpr 0.01 < 10; the FPR threshold rests on too few negatives
INFO sondekit.sweep: block 0: val {'n_pos': 8, 'n_neg': 8, 'threshold': 0.9999285340309143, 'auroc': 1.0, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0, 'fpr': 0.0, 'small_n': True, 'recall_at_fpr': 1.0, 'threshold_failed': False, 'val_loss': 2.8390242945598068e-05}
...
INFO sondekit.sweep: selected block 8; test {'n_pos': 8, 'n_neg': 8, 'threshold': 0.9999760985374451, 'auroc': 1.0, 'f1': 0.8571428571428571, 'precision': 1.0, 'recall': 0.75, 'fpr': 0.0, 'small_n': True, 'recall_at_fpr': 1.0}; control_auroc 0.562
INFO sondekit.sweep: headline {"max_fpr": 0.01, "recall_at_fpr": 1.0, "recall_at_fpr_ci": [1.0, 1.0], "threshold": 0.9999760985374451, "recall": 0.75, "fpr": 0.0, "auroc": 1.0, "auroc_ci": [1.0, 1.0], "baseline_recall_at_fpr": 1.0, "baseline_auroc": 1.0, "length_auroc": 0.5, "n_pos": 8, "n_neg": 8, "small_n": true}
WARNING sondekit.train: small_n: 16 negatives x max_fpr 0.01 < 10; the FPR threshold rests on too few negatives
INFO sondekit.score: score heldout: {'n_pos': 16, 'n_neg': 16, 'threshold': 0.9999760985374451, 'auroc': 1.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'fpr': 0.0, 'small_n': True, 'recall_at_fpr': 1.0}
```

When it finishes, sondekit prints the run directory on stdout:

```
runs/quickstart
```

A few things in this log are worth knowing before you train on real data:

- `small_n` warnings mean the split has too few negatives to place a 1%
  false-positive threshold reliably (fewer than 10 expected false
  positives). On a 64-row toy set that is expected.
- Every layer reaches validation AUROC 1.0 here, so the choice comes down
  to tie-breaking. Because the validation split is `small_n`, sondekit
  ranks layers by AUROC instead of recall at 1% FPR, then by lower
  validation loss. Layer 8 had the lowest.
- `control_auroc` is the validation AUROC of the same probe trained on
  shuffled labels, averaged over 5 shuffles. It should sit near 0.5.
- `baseline_auroc` is 1.0 too. A bag-of-words classifier separates "I love
  this" from "I hate this" as well as the probe does, so this dataset
  cannot show that the probe has learned anything beyond word identity.
  [Evaluation](evaluation.md) covers how to read the baselines.
- On the heldout set the probe still ranks every example correctly
  (`auroc` 1.0), but no score reaches the threshold picked on the
  validation split, so `recall` at that threshold is 0.0. A threshold
  chosen on one distribution does not carry over to another for free,
  which is why `score:` entries exist.

Recipes resolve their bundled data files themselves, so `sondekit run
quickstart` works from any directory. The `refusal` recipe reads
`data/refusal/labeled.jsonl` relative to the current directory, so run it
from the repo root. `high_stakes` loads its data from the Hugging Face Hub.

## The run directory

The quickstart writes `runs/quickstart/`, about 2.5 MB:

```
runs/quickstart/
  config.yaml                    the resolved config, every default filled in
  extract/
    run.json                     config hash, written when extraction starts
    shard_00000.safetensors      activations, one tensor per layer (B0 ... B11)
    manifest.json                ids, labels, blocks, model fingerprint, drops
  splits.json                    row ids in train, val and test
  layers/
    B0.npz ... B11.npz           the probe fitted at each layer
  probes/
    quickstart.npz               the selected probe, with its metrics inside
  metrics.json                   headline, per-layer validation, test, baselines, control
  score/
    heldout/
      extract/                   activations for the heldout set, selected layer only
      scores.jsonl               one {"id", "score", "flag"} row per example
      metrics.json               metrics on this set
```

`probes/quickstart.npz` is the file you deploy. It needs only numpy to
load and score, and it records the model fingerprint and prompt format it
was trained under. [Python API](python-api.md) shows how to use it.

`metrics.json` has these top-level keys: `headline`, `block`, `select`,
`select_fallback`, `max_fpr`, `threshold`, `threshold_failed`, `val` (one
entry per layer), `test`, `baseline`, `control_auroc`, `control_aurocs`,
`controls_passed` and `n` (split sizes). The quickstart run has
`"select_fallback": "auroc"` and `"controls_passed": true`.

The first lines of `score/heldout/scores.jsonl`:

```
{"id": "quickstart_heldout-0", "score": 0.47642144560813904, "flag": false}
{"id": "quickstart_heldout-1", "score": 2.6355279260315e-05, "flag": false}
{"id": "quickstart_heldout-2", "score": 0.8687804937362671, "flag": false}
```

`flag` is `score >= threshold`, using the threshold stored in the probe.

## Your first config

This example trains a probe that tells questions from statements, on gpt2,
from a local file.

### 1. The data file

Write one JSON object per line. Each row needs a text column and a 0/1
label. An `id` column is optional; without one, sondekit uses the row
index.

```
{"id": "r0", "prompt": "When does the train start?", "label": 1}
{"id": "r1", "prompt": "I think the train starts at noon.", "label": 0}
{"id": "r2", "prompt": "Why is the train so popular?", "label": 1}
{"id": "r3", "prompt": "Everyone says the train is popular.", "label": 0}
```

The file used below has 80 such rows in `data/rows.jsonl`, 40 of each
label. CSV files and Hugging Face datasets work too; [data](data.md)
covers every source, chat-format rows, response windows and label maps.

### 2. The config

```yaml
name: question
model: {name: gpt2, dtype: float32}
data:
  path: data/rows.jsonl
  text: prompt
  label: label
  format: raw
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: last}
steps: [extract, train]
```

Save it as `question.yaml`. Section by section:

- `name` names the run directory, `runs/question/`.
- `model.dtype: float32` because this runs on CPU. The default is
  `bfloat16`, which is what you want on a GPU. Activations are stored in
  this dtype.
- `data.text` and `data.label` are column names. Their defaults are `text`
  and `label`, so `label: label` could be left out. `data.path` is
  relative to the directory you run `sondekit` from, not to the config
  file.
- `format: raw` because gpt2 has no chat template. For a chat model, leave
  it at the default `chat` and sondekit wraps each text as a user turn and
  applies the tokenizer's chat template.
- `pooling: last` takes the activation at the last prompt token, which has
  read the whole sentence, including the question mark. A linear probe on
  pooled activations is fitted with full-batch L-BFGS, so `epochs`, `lr`,
  `batch_size` and `patience` do not apply to it; `weight_decay` does.
- `steps` lists what to run, in order. `[extract, train]` is the default.

Every option, with its default, is in [configuration](configuration.md).

### 3. Check it with --dry-run

```bash
sondekit run question.yaml --dry-run
```

`--dry-run` validates the config, prints it with every default filled in,
and estimates the disk space `extract` will use. It reads the data file,
the tokenizer and the model's `config.json`, but no weights. The output
ends like this:

```
probe:
  kind: linear
  pooling: last
  rolling_window: null
  init: random
  epochs: 20
  lr: 0.01
  weight_decay: 0.0001
  batch_size: 256
  patience: 3
  max_fpr: 0.01
  select: recall_at_fpr
score: []
generate: null
steer: null
output:
  dir: runs
  overwrite: false
steps:
- extract
- train
# extract disk estimate: 2.8 MiB
```

The estimate is rows x hidden size x layers x bytes per value: 80 x 768 x
12 x 4 bytes here. With `keep: tokens`, every token in the window is
stored, and the estimate grows with the number of tokens per row. The same
config with `-o extract.keep=tokens -o probe.pooling=max` estimates
21.7 MiB. On a 7B model with thousands of long rows, `keep: tokens` can
reach hundreds of gigabytes, so check the estimate before a large run.

Because it reads the data, `--dry-run` also catches a missing file, a
missing column, a bad label or an out-of-range layer before any model
loads.

### 4. Run it

```bash
sondekit run question.yaml
```

```
INFO sondekit.extract: extract: 80 samples x 12 blocks, ~0.00 GB
INFO sondekit.data: split train: n=56 (0.700) n_pos=28 n_neg=28
INFO sondekit.data: split val: n=12 (0.150) n_pos=6 n_neg=6
INFO sondekit.data: split test: n=12 (0.150) n_pos=6 n_neg=6
...
INFO sondekit.sweep: selected block 1; test {'n_pos': 6, 'n_neg': 6, 'threshold': 0.9999922513961792, 'auroc': 1.0, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0, 'fpr': 0.0, 'small_n': True, 'recall_at_fpr': 1.0}; control_auroc 0.550
INFO sondekit.sweep: headline {"max_fpr": 0.01, "recall_at_fpr": 1.0, "recall_at_fpr_ci": [1.0, 1.0], "threshold": 0.9999922513961792, "recall": 1.0, "fpr": 0.0, "auroc": 1.0, "auroc_ci": [1.0, 1.0], "baseline_recall_at_fpr": 1.0, "baseline_auroc": 1.0, "length_auroc": 0.16666666666666666, "n_pos": 6, "n_neg": 6, "small_n": true}
runs/question
```

The probe is in `runs/question/probes/question.npz`.

### 5. Change settings with -o

`-o key=value` sets any key for one run without editing the file. Keys are
dotted paths into the YAML, and values are parsed as YAML, so numbers,
`null`, `true`, lists and mappings all work. Repeat `-o` for several keys.

```bash
sondekit run question.yaml -o probe.weight_decay=1e-3
sondekit run question.yaml --dry-run -o probe.weight_decay=1e-3 -o probe.max_fpr=0.05
sondekit run question.yaml --dry-run -o "extract.layers=[4, 8]"
sondekit run question.yaml --dry-run -o "extract.layers={every: 4}"
sondekit run question.yaml --dry-run -o probe.max_fpr=null -o probe.select=auroc
sondekit run question.yaml --dry-run -o "data.chat_template_kwargs={enable_thinking: false}"
sondekit run question.yaml --dry-run -o name=question-v2
```

Quote any value that contains spaces, brackets or braces, so the shell
passes it through unchanged. `extract.layers=[4, 8]` extracts layers 4 and
8 only, and its disk estimate drops to 0.5 MiB. `{every: 4}` takes layers
0, 4, 8 and so on.

Changing `name` is the easy way to keep two variants side by side, since
each name gets its own run directory.

### 6. Recipe names and paths

`sondekit run` treats its argument as a file path when it ends in `.yaml`
or `.yml`, and as a bundled recipe name otherwise. `sondekit run question`
looks for a recipe called `question` and fails:

```
sondekit: error: unknown recipe 'question'; available: ['high_stakes', 'quickstart', 'refusal']
```

To start from a recipe, copy its YAML out of the package and edit the
copy. This prints where the recipes live:

```bash
python -c "from sondekit import config; print(config.RECIPES_DIR)"
```

The quickstart also reads two data files from that directory.
[Cookbook](cookbook.md#bundled-recipes) shows how to copy a recipe together
with its data.

### Using a chat model

The same data on Qwen3-0.6B, a chat model, needs the chat template and the
Qwen3 switch that turns off the thinking block:

```yaml
name: question-qwen
model: {name: Qwen/Qwen3-0.6B}
data:
  path: data/rows.jsonl
  text: prompt
  chat_template_kwargs: {enable_thinking: false}
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: last}
steps: [extract, train]
```

`--dry-run` on this config estimates 4.4 MiB (28 layers, hidden size 1024,
bfloat16). Whatever serves the probe later must render prompts with the
same template and the same `chat_template_kwargs`. The probe records a
digest of the chat template, and the `score` step refuses a probe whose
digest differs from the run's. Non-empty `chat_template_kwargs` are part of
that digest.

## Rerunning

Running the same config again picks up where it left off:

- `extract` hashes the `model`, `data` and `extract` sections (plus
  `probe.pooling` when `keep: pooled`, and `seed` when `data.limit` is
  set). If the hash matches a finished extraction, it logs
  `extract: runs/quickstart/extract is complete; skipping` and loads no
  model for it. If the hash matches an unfinished one, it resumes from the
  last complete shard.
- `generate` skips rows whose id is already in `generations.jsonl`.
- `train`, `score` and `steer` always run again and replace their outputs.

So a change to training settings, such as `-o probe.weight_decay=1e-3`,
reuses the extracted activations and only retrains. A change that affects extraction
stops the run before any model loads:

```bash
sondekit run question.yaml -o probe.pooling=mean
```

```
sondekit: error: runs/question/extract/manifest.json was written by another config (hash 2753a59878f304c5, now 7d210407687d18a1); set output.overwrite: true or change name
```

Here the existing activations were pooled at the last token, and `mean`
needs them pooled differently. Either pick a new `name`, or pass
`-o output.overwrite=true`, which deletes `extract/` (and
`generations.jsonl`, when it was generated under a different config) and
starts it again. Pass it on the command line for the one run that needs it. Left in
the YAML, it makes every run re-extract.

A run that fails partway still leaves `config.yaml` and `extract/run.json`
behind. If you then fix the config, for example a wrong `data.path`, the
next run sees a different hash and stops with the same error, naming
`extract/run.json`. Rerun once with `-o output.overwrite=true`.

With the same config and seed, repeated runs produce identical results.
Two quickstart runs on CPU wrote byte-identical `metrics.json` files.

## Reading the headline

The `headline` log line, also stored as `metrics.json["headline"]`, sums
up the selected probe on the test split. [Evaluation](evaluation.md#the-headline)
lists every field.

`recall_at_fpr` re-picks the threshold on the test set, so it describes
how well the probe ranks examples. `recall` and `fpr` use the threshold
you will deploy, so they describe how it behaves in use. When the two
disagree, the saved threshold is off for this data.

Compare the probe with the baselines before trusting it. In the question
example, the bag-of-words baseline also scores 1.0, because words like
"when" and "why" give the label away. `length_auroc` is 0.17, which is as
far from chance as 0.83: questions in this file are shorter than
statements, and a length AUROC far from 0.5 in either direction means
sequence length predicts the label. This matters most for `max` pooling,
which can pick up length. [Evaluation](evaluation.md) goes through each
metric and the shuffled-label control in detail.

## Common errors

Configuration and data mistakes print a single line starting with
`sondekit: error:` and exit with code 2. Anything else prints a full
traceback. Apart from the `NotImplementedError` of the vllm backend, that
is a bug in sondekit; please report it.

Config errors come from pydantic. For an unknown or mistyped key, the
second line names the key.

**Unknown key.** Usually a typo:

```bash
sondekit run question.yaml -o probe.epoch=50
```

```
sondekit: error: 1 validation error for RunConfig
probe.epoch
  Extra inputs are not permitted [type=extra_forbidden, input_value=50, input_type=int]
    For further information visit https://errors.pydantic.dev/2.13/v/extra_forbidden
```

**Invalid combination.** `keep: pooled` pools at extraction time, which
works only for linear probes with `mean` or `last` pooling:

```bash
sondekit run question.yaml -o probe.pooling=max
```

```
sondekit: error: 1 validation error for RunConfig
  Value error, extract.keep: pooled needs probe.kind: linear with probe.pooling mean or last; use extract.keep: tokens [type=value_error, input_value={'name': 'question', 'mod...': ['extract', 'train']}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
```

Other combinations are checked the same way. Two examples:

```
Value error, probe.select: recall_at_fpr needs probe.max_fpr; set select: auroc to run without one
Value error, data: set exactly one of path / hf
```

**Missing data file.**

```bash
sondekit run question.yaml --dry-run -o data.path=data/missing.jsonl
```

```
sondekit: error: [Errno 2] No such file or directory: 'data/missing.jsonl'
```

Relative paths resolve against the current directory. The `refusal`
recipe run outside the repo root fails this way.

**Missing column.** The message lists the keys the row does have:

```
sondekit: error: row 0 has no 'sentence' key; keys: ['id', 'label', 'prompt']
```

**Label that is not 0 or 1.**

```
sondekit: error: row 0: label 'yes' is not 0/1; set data.label_map
```

Map the values with `data.label_map`. Quote `yes` and `no` as keys:
YAML reads them unquoted as booleans, and the map then matches nothing.

```bash
sondekit run question.yaml --dry-run -o "data.label_map={'yes': 1, 'no': 0}"
```

**Chat format on a model without a chat template.**

```
sondekit: error: data.format is chat but the gpt2 tokenizer has no chat_template; set data.format: raw
```

**Layer out of range.** Layers count from 0, so gpt2's are 0 to 11:

```
sondekit: error: extract.layers [12] out of range: model has 12 blocks
```

**Bad override.**

```
sondekit: error: override 'probe.epochs' is not key=value
```

**Model weights not available.** `--dry-run` only needs the tokenizer and
`config.json`, so a config can pass it and still fail at extraction when
the weights cannot be downloaded or are not cached:

```
sondekit: error: Qwen/Qwen3-0.6B does not appear to have a file named pytorch_model.bin or model.safetensors.
```

A gated model fails with Hugging Face's own message until you have been
granted access and have run `huggingface-cli login`.

**Config written by another run.** See [Rerunning](#rerunning).

**A split with only one class.** Each of train, validation and test needs
both labels. With very little data, or a skewed `data.split`, the `train`
step stops and names the split. With `-o data.limit=8` on the question
config:

```
sondekit: error: split 'val' has n_pos=0, n_neg=1; it needs both classes. Add data or change data.split
```

## Where next

- [Data](data.md): every data source and column option, chat rows,
  response windows, groups and label maps.
- [Configuration](configuration.md): the full option table, steps and
  layer numbering.
- [Evaluation](evaluation.md): what each metric means, the baselines, the
  shuffled-label control, and scoring on shifted data.
- [Cookbook](cookbook.md): configs for common tasks, such as attention
  probes, diff-means directions, generating and steering.
- [Python API](python-api.md): loading a `.npz` probe, scoring
  activations, and running configs from Python.
- [Linear probes primer](linear-probes-primer.md): what probes measure,
  and the papers behind them.
