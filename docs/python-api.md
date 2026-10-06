# Python API

This page covers using sondekit from Python: running a pipeline without the
CLI, loading and scoring a trained probe, the probe file format, the
compatibility checks a probe carries, and how to read the activations a probe
expects at serve time.

The examples assume you ran the quickstart from the directory you run Python
in, so that `runs/quickstart/` exists:

```bash
sondekit run quickstart
```

See [getting started](getting-started.md) for that first run and
[configuration](configuration.md) for every config option.

## Install

| Install | What you get |
|---|---|
| `pip install sondekit` | `sondekit.probe` and `sondekit.fingerprint`, which need only numpy and the standard library |
| `pip install "sondekit[hf]"` | the training pipeline as well: `sondekit.run`, `sondekit.load_config` and the `sondekit` command |

A machine that only scores probes needs the plain install. Importing
`sondekit`, `sondekit.probe` or `sondekit.fingerprint` does not import torch,
transformers, pydantic or yaml, so the probe code can run inside a vLLM worker
or any other process without adding dependencies.

`sondekit.run` and `sondekit.load_config` are imported lazily. Without the
extras, touching either one raises an `ImportError` that names the extra:

```python
import sondekit

try:
    sondekit.run
except ImportError as e:
    print(e)
```

```
sondekit.run needs the training extras: pip install 'sondekit[hf]' (or 'sondekit[vllm]')
```

## Running pipelines from Python

`sondekit.load_config(source, overrides=())` reads and validates a config,
the same way `sondekit run` does. `source` is a `.yaml` or `.yml` path, or
the bare name of a bundled recipe. `overrides` is a list of the strings you
would pass to `-o`; each value is parsed as YAML.

`sondekit.run(cfg)` runs `cfg.steps` in order and returns the run directory
as a `pathlib.Path`.

```python
import json
import logging

import sondekit

logging.basicConfig(level=logging.INFO)  # the CLI does this for you

cfg = sondekit.load_config(
    "quickstart",
    ["output.dir=pyruns", "probe.epochs=10", "extract.layers=[4, 8]"],
)
print(cfg.probe.epochs, cfg.extract.layers, cfg.steps)
run_dir = sondekit.run(cfg)
print(run_dir)
card = json.loads((run_dir / "metrics.json").read_text())
print(card["block"], card["headline"]["auroc"])
```

```
10 [4, 8] ['extract', 'train', 'score']
pyruns/quickstart
8 1.0
```

sondekit logs progress through the `logging` module and configures nothing
itself, so call `logging.basicConfig` if you want the log lines the CLI
prints.

The config object is a pydantic model, `sondekit.config.RunConfig`. You can
build one from a dict instead of a file. Here `data/rows.jsonl` stands for
your own data file, and `steps` is left at its default, `[extract, train]`:

```python
import sondekit
from sondekit import config

cfg = config.RunConfig.model_validate({
    "name": "from-dict",
    "model": {"name": "gpt2", "dtype": "float32"},
    "data": {"path": "data/rows.jsonl", "format": "raw",
             "split": [0.5, 0.25, 0.25]},
    "extract": {"layers": [6, 8], "keep": "pooled"},
    "probe": {"kind": "linear", "pooling": "last", "epochs": 30},
})
print(sondekit.run(cfg))
```

```
runs/from-dict
```

Errors are the same ones the CLI prints after `sondekit: error:`:

| Situation | Exception |
|---|---|
| unknown recipe name | `ValueError: unknown recipe 'nope'; available: ['high_stakes', 'quickstart', 'refusal']` |
| invalid key or combination | `pydantic.ValidationError`, naming the key |
| `extract/` or `generations.jsonl` written by a different config | `ValueError` from `sondekit.run`, before any model loads |
| missing file, gated model | `OSError` |

Rerunning the same config resumes, as described in
[configuration](configuration.md#rerunning). A second `sondekit.run(cfg)` in
the same process reuses the loaded model when the `model` section is
identical.

## The probe file

A trained probe is a `sondekit.probe.Probe`, a dataclass saved as one `.npz`
file. A run writes the selected probe to `runs/<name>/probes/<name>.npz` and
one probe per layer to `runs/<name>/layers/B<layer>.npz`.

```python
from sondekit import probe

p = probe.Probe.load("runs/quickstart/probes/quickstart.npz")
print(p.name, p.kind, p.pooling, p.window, p.block)
print(p.w.shape, p.w.dtype, p.q, p.rolling_window)
print(p.bias, p.threshold)
print(p.model, p.engine, p.backend())
print(p.model_fingerprint)
print(p.prompt_format)
print(p.adapters, p.escalate_threshold)
print(sorted(p.metrics))
```

```
quickstart linear mean prompt 8
(768,) float32 None None
6.455459596336729 0.9999760985374451
gpt2 hf==5.18.0+nnterp==1.3.0 hf
hub:c9a84be385409e6005a141e99416aefc
raw
() None
['baseline', 'block', 'control_auroc', 'control_aurocs', 'controls_passed', 'headline', 'max_fpr', 'n', 'select', 'select_fallback', 'test', 'threshold', 'threshold_failed', 'val']
```

Your numbers and the `engine` string may differ with other hardware or
library versions.

### Fields

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | probe name; the key in `load_dir`. A run's selected probe is named after the run; per-layer probes are `<name>@B<layer>` |
| `kind` | `str` | `linear` or `attention` |
| `w` | `[H]` float32 | the score direction (linear) or value direction (attention) |
| `bias` | `float` | added to the pooled logit |
| `block` | `int` | reads the residual stream output by decoder block `block` (see [layer numbering](#layer-numbering)) |
| `window` | `str` | `prompt`, `response`, `all` or `last_turn` |
| `pooling` | `str` | `mean`, `last`, `max` or `rolling_mean` for linear probes; `attention` for attention probes |
| `threshold` | `float` | flag threshold on the `pooled_score` scale, in [0, 1] |
| `model` | `str` | `model.name` the probe was trained on |
| `engine` | `str` | backend and library versions, `hf==<transformers>+nnterp==<nnterp>` or `vllm==<vllm>+nnsight==<commit or version>` |
| `model_fingerprint` | `str` or `None` | checkpoint digest, see [fingerprint](#the-fingerprint-module) |
| `prompt_format` | `str` or `None` | chat template digest `tmpl:<hex>`, or `raw` for `data.format: raw` |
| `q` | `[H]` float32 or `None` | attention query; set for attention probes only |
| `rolling_window` | `int` or `None` | window length; set for `rolling_mean` only |
| `adapters` | `tuple[str, ...]` | LoRA adapters the probe is allowed to score; sorted; sondekit writes `()` |
| `escalate_threshold` | `float` or `None` | lower edge of a review band below `threshold`; sondekit never sets it |
| `metrics` | `dict` or `None` | the run's `metrics.json` for the selected probe; `{"val": ...}` for a per-layer probe |

`H` is the model's hidden size. Standardization is folded into `w`, `bias`
and `q` when the probe is exported, so the probe reads raw activations.

Constructing a `Probe` validates it. `w` and `q` are converted to float32
and must be finite, nonempty 1-D vectors of the same shape. `q` must be set
exactly for attention probes, `pooling` must be valid for `kind`, `window`
must be one of the four windows, `rolling_window` must be set exactly for
`rolling_mean`, `threshold` must be in [0, 1], and `escalate_threshold`
must be between 0 and `threshold`. A violation raises `ValueError`, for example
`probe escalate_threshold must be between 0 and threshold`.

### Methods

| Method | Input | Returns |
|---|---|---|
| `logits(acts)` | `[T, H]` or `[H]` | `[T]` float32 `acts @ w + bias`; linear probes only |
| `pooled_logit(acts)` | `[T, H]` or `[H]` | the pooled logit as a `float`, before the sigmoid |
| `pooled_score(acts)` | `[T, H]` or `[H]` | `sigmoid(pooled_logit(acts))`, a `float` in [0, 1]; the scale `threshold` is on |
| `pooled_score_from_logits(logits)` | `[T]` logits from `logits()` | the same score as `pooled_score`; linear probes only |
| `flag(acts)` | `[T, H]` or `[H]` | `pooled_score(acts) >= threshold` |
| `escalates(score)` | a score | `escalate_threshold <= score < threshold`; always `False` when `escalate_threshold` is `None` |
| `serves(fingerprint, adapter, prompt_format="unchecked")` | serving facts | `None` if the probe may score this model, else the reason it may not |
| `vllm_aux_layer()` | | `block + 1`, the vLLM `extract_hidden_states` layer id for `block` |
| `backend()` | | the backend part of `engine`, e.g. `hf` |
| `save(path)` | a `str` path | writes the `.npz` atomically, appending `.npz` if missing, and returns the path written; a `pathlib.Path` raises `AttributeError` |
| `Probe.load(path)` | path | the validated probe |

`acts` are the activations at `block` over the probe's `window`, one row per
token. A single `[H]` row is treated as `T = 1`. The sigmoid is computed in
float64 with the logit clipped to [-50, 50], so confident scores do not tie
at 1.0 as early as they would in float32.

`pooled_logit` is the value to use when a score feeds an optimizer, such as
an RL reward penalty, because the sigmoid saturates.

```python
import numpy as np

from sondekit import probe

p = probe.Probe.load("runs/quickstart/probes/quickstart.npz")
acts = np.zeros((12, p.w.shape[0]), dtype=np.float32)  # [tokens, hidden]
print(p.pooled_score(acts), p.flag(acts))
```

```
0.9984305503731531 False
```

`sondekit.probe.load_dir(directory)` loads every `*.npz` in a directory (not
recursive) in filename order and returns a dict keyed by `Probe.name`. It
raises `ValueError` on two files with the same probe name
(`duplicate probe 'sentiment' in probe_dir`) or an empty directory
(`no probes in <dir>`).

### Pooling at serve time

Pass `pooled_score` the activations for every token in the probe's window.
The pooling then matches training:

| `pooling` | What `pooled_logit` computes over the `[T]` per-token logits |
|---|---|
| `mean` | their mean |
| `last` | the last one; passing only the last row `[H]` gives the same result |
| `max` | their maximum |
| `rolling_mean` | the largest mean over any `rolling_window` consecutive tokens; the plain mean when `T < rolling_window` |
| `attention` | `softmax(acts @ q)`-weighted mean of `acts @ w`, plus `bias` |

For linear probes, pooling acts on per-token logits, so a scorer that sees
tokens one at a time can call `logits()` on each new row, keep the `[T]`
logits, and call `pooled_score_from_logits` on them. This gives the same
score as `pooled_score` on the stacked activations and never stores the
activations. Attention probes need all `[T, H]` activations, so
`logits()` and `pooled_score_from_logits` raise `ValueError` for them.

When a run uses `extract.keep: pooled`, sondekit stores the mean (or last)
activation per row during extraction and scores that one row. For a linear
probe, the logit of the mean activation equals the mean of the per-token
logits, so passing the full `[T, H]` window at serve time gives the same
score up to floating point rounding.

### Editing and saving

`Probe` is a dataclass, so `dataclasses.replace` makes an edited copy and
re-runs validation. This is how you add a review band or allow a LoRA
adapter before deploying a probe:

```python
import dataclasses
import pathlib

from sondekit import probe

p = probe.Probe.load("runs/quickstart/probes/quickstart.npz")
pathlib.Path("probe_dir").mkdir(exist_ok=True)
deployed = dataclasses.replace(
    p, name="sentiment", escalate_threshold=0.5, adapters=("my-lora",)
)
print(deployed.escalates(0.7), deployed.escalates(0.3))
print(deployed.save("probe_dir/sentiment"))
print(sorted(probe.load_dir("probe_dir")))
```

```
True False
probe_dir/sentiment.npz
['sentiment']
```

The `adapters` list only widens what `serves` accepts. The probe was trained
on the base model, so check that it still works on the adapter before you
add one, for example with a `score:` entry (see [evaluation](evaluation.md)).

## File format

A probe file is a numpy `.npz` archive written with `allow_pickle=False` and
read back the same way, so loading a probe never executes code. It holds:

| Array | Contents |
|---|---|
| `meta` | a JSON string with every field except `w` and `q`, plus `"format": 1` |
| `w` | `[H]` float32 |
| `q` | `[H]` float32, attention probes only |

The current format is 1 (`sondekit.probe.FORMAT`). `Probe.load` refuses any
other format and any meta that does not build a `Probe`:

```
legacy.npz: not a format-1 sondekit probe
bad.npz: not a sondekit probe (Probe.__init__() got an unexpected keyword argument 'colour')
```

Three fields record what the probe was trained against:

- `model_fingerprint`: which checkpoint produced the activations.
- `prompt_format`: which chat template rendered the prompts, or `raw`.
- `engine`: the backend and the transformers and nnterp (or vLLM and
  nnsight) versions.

## Compatibility checks

A probe only means something on the weights and prompt format it was trained
on. A fine-tuned model usually keeps its name, and a probe trained on raw
text scores templated prompts very differently, so neither failure shows up
as an error unless something checks for it.

### `serves`

`p.serves(fingerprint, adapter, prompt_format="unchecked")` returns `None`
when the probe may score the serving model, and otherwise a string saying
why not. It fails closed: a probe or server that cannot state its checkpoint
or prompt format is refused. Pass `prompt_format="unchecked"` (the default)
to skip the prompt format check, and `adapter=None` for the base model.

```python
from sondekit import fingerprint
from sondekit import probe

p = probe.Probe.load("runs/quickstart/probes/quickstart.npz")
fp = fingerprint.checkpoint_fingerprint("gpt2")
print(p.serves(fp, None, prompt_format=fingerprint.RAW))
```

```
None
```

Each refusal, as returned for the quickstart probe:

| Case | Message |
|---|---|
| different checkpoint (here, `revision="main"` where training used none) | `checkpoint mismatch (probe=hub:c9a84be385409e6005a141e99416aefc, serving=hub:e6129f485b2dcd25073d9cd000ad064b): fine-tuning invalidates a probe` |
| serving fingerprint is `None` | `serving checkpoint could not be fingerprinted` |
| adapter not in `p.adapters` | `adapter 'my-lora' is not one this probe was validated on` |
| served prompt format is `None` | `served prompt format could not be determined` |
| different prompt format | `prompt format mismatch (probe=raw, serving=tmpl:f24189f08c85a1eb19a737306c3a13e8)` |
| probe has no `model_fingerprint` | `probe carries no checkpoint fingerprint; retrain it` |
| probe has no `prompt_format` (and the format is checked) | `probe does not record its prompt format; retrain it` |

`serves` does not compare `engine`. The `score` step does compare the
backend: it refuses a probe whose `backend()` differs from `model.backend`,
or whose fingerprint or prompt format differs from the run's. A `score`
entry that points at the quickstart probe, in a run on
`sshleifer/tiny-gpt2`, stops with:

```
sondekit: error: probe runs/quickstart/probes/quickstart.npz has model_fingerprint='hub:c9a84be385409e6005a141e99416aefc' but this run has 'hub:caf008f07f3a29bdd080111b0be59a2d'; its scores would be meaningless here, so score refuses it
```

### The fingerprint module

`sondekit.fingerprint` needs only the standard library.

| Function | Returns |
|---|---|
| `checkpoint_fingerprint(model, revision=None, quantization=None)` | `local:<hex>` for a local directory, from the names and sizes of its weight and config files; `hub:<hex>` for anything else, from the name and revision |
| `prompt_format(template)` | `tmpl:` plus a sha256 prefix of the chat template text, or `RAW` (`"raw"`) when `template` is `None` |
| `normalize_revision(value)` | `value.initial` when the object has it (vLLM hands back a revision object that keeps the requested revision there), otherwise `value` unchanged |
| `artifact_digest(path)` | a 16-hex-character sha256 prefix of a probe file's bytes, to tell two builds of a same-named probe apart |

sondekit fingerprints with `checkpoint_fingerprint(model.name,
revision=model.revision)` at training time. To match it when serving:

- Pass the same `model` string. A Hub id and a local directory holding the
  same weights fingerprint differently (`hub:` vs `local:`), so a probe
  trained on `gpt2` is refused by a server that loaded the snapshot from a
  local path, and the other way round.
- Pass the same revision. `None` and `"main"` give different digests.
- Leave `quantization` as `None`. sondekit never passes it, so any other
  value gives a different digest.
- `revision` and `quantization` must be exact `str` or `None`. A `str`
  subclass raises `ValueError`, because it would digest as its string value
  and silently change every fingerprint.

For chat probes, compute the served format with
`fingerprint.prompt_format(tokenizer.chat_template)`. The digest covers the
template text only. `data.chat_template_kwargs` (such as
`{enable_thinking: false}` for Qwen3) and `data.system` are not part of it,
so the serving side has to apply the same values itself.

## Getting the right activations

### Layer numbering

Layer `L` is the residual stream right after decoder block `L`, counting
from 0, before the final norm. The same tensor has different names in
different libraries:

| Where | Tensor for `p.block = L` |
|---|---|
| nnterp (the `hf` backend) | `model.layers_output[L]` |
| transformers `output_hidden_states=True` | `hidden_states[L + 1]`; for the last block, transformers applies the final norm to this entry, so read the block's output with a forward hook instead |
| vLLM `extract_hidden_states` | auxiliary layer `L + 1`, which `p.vllm_aux_layer()` returns |
| nnsight on vLLM | `out = model.model.layers[L].output; out[0] + out[1]` |

`hidden_states[0]` is the embedding output, which is why transformers is off
by one.

### Token ids and the window

The activations must come from the same token ids sondekit extracted from.
sondekit builds them as follows (`sondekit/data.py`):

| `data.format`, `window` | Token ids | Window |
|---|---|---|
| `raw`, `prompt` | `tokenizer(text).input_ids` (with the tokenizer's special tokens) | every token |
| `chat`, `prompt` | `tokenizer(apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs), add_special_tokens=False).input_ids` | every token |
| either, `response` | the prompt ids above, then the response ids | the response ids |
| either, `all` | the prompt ids, then the response ids | every token |
| `chat`, `last_turn` | the whole conversation rendered without a generation prompt | from `probe.last_turn_start(tokenizer, messages)` to the end |

For `text` rows in chat format, `messages` is the user turn, after a system
turn when `data.system` is set. Response ids come from the row's
`response_ids` when present (as written by the `generate` step), otherwise
from `tokenizer(response, add_special_tokens=False)`. Sequences are cut to
the first `min(data.max_length, tokenizer.model_max_length)` tokens.

`probe.last_turn_start(tokenizer, messages)` returns the number of tokens in
`messages[:-1]` rendered with the generation prompt, which is where the last
message starts. It renders without `chat_template_kwargs`. sondekit drops a
row as `last_turn_prefix_mismatch` when that prefix differs from the one
rendered with the kwargs. With the Qwen3 tokenizer and
`chat_template_kwargs: {enable_thinking: false}` the prefixes differ for
every row, so `window: last_turn` currently drops all of them. Rows whose
last message is not from the assistant are dropped as `not_assistant_last`.

### Reproducing a score with transformers

This script scores the quickstart's held-out set with plain transformers and
checks the result against `runs/quickstart/score/heldout/scores.jsonl`,
which the `score` step wrote. The quickstart probe uses `format: raw`,
`window: prompt` and `pooling: mean`, so the activations are every token of
the tokenized text at `hidden_states[p.block + 1]`.

```python
import json
import pathlib

import torch
import transformers

import sondekit
from sondekit import fingerprint
from sondekit import probe

p = probe.Probe.load("runs/quickstart/probes/quickstart.npz")
reason = p.serves(
    fingerprint.checkpoint_fingerprint("gpt2"),
    adapter=None,
    prompt_format=fingerprint.RAW,  # the quickstart uses data.format: raw
)
if reason is not None:
    raise SystemExit(reason)

tok = transformers.AutoTokenizer.from_pretrained("gpt2")
model = transformers.AutoModelForCausalLM.from_pretrained(
    "gpt2", dtype=torch.float32
)
model.eval()


def score(text: str) -> float:
    ids = tok(text, return_tensors="pt")  # format: raw, window: prompt
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    acts = out.hidden_states[p.block + 1][0].numpy()  # [T, H] at p.block
    return p.pooled_score(acts)


written = {}
for line in open("runs/quickstart/score/heldout/scores.jsonl"):
    row = json.loads(line)
    written[row["id"]] = row["score"]

data = pathlib.Path(sondekit.__file__).parent / "recipes/data"
diffs = []
for line in open(data / "quickstart_heldout.jsonl"):
    row = json.loads(line)
    diffs.append(abs(score(row["text"]) - written[row["id"]]))
print(f"{len(diffs)} rows, max |difference| {max(diffs):.1e}")
```

```
32 rows, max |difference| 3.0e-06
```

The small difference is float32 rounding: the run stored the mean
activation and scored it, while this script scores each token and averages
the logits.

If the probe's layer is the model's last one, `hidden_states[p.block + 1]`
is the normed output. Read the block output with a forward hook instead.
For gpt2 the blocks are `model.transformer.h`; for Llama and Qwen models
they are `model.model.layers`:

```python
seen = {}


def keep(module, args, output):
    seen["h"] = output[0] if isinstance(output, tuple) else output


handle = model.transformer.h[p.block].register_forward_hook(keep)
```

### With nnterp

The `hf` backend extracts through nnterp, so reading `layers_output` gives
the training-time tensor directly:

```python
import nnsight
import nnterp
import torch

from sondekit import probe

p = probe.Probe.load("runs/quickstart/probes/quickstart.npz")
model = nnterp.StandardizedTransformer("gpt2", dtype=torch.float32)
ids = model.tokenizer("What a great movie this is!", return_tensors="pt")
with torch.no_grad(), model.trace(ids):
    h = nnsight.save(model.layers_output[p.block])
print(p.pooled_score(h[0].float().cpu().numpy()))
```

```
0.4764220309212402
```

`scores.jsonl` has `0.47642144560813904` for this row.

### With vLLM

vLLM's `extract_hidden_states` returns auxiliary hidden states by layer id,
and layer `L` is id `L + 1`. Request `p.vllm_aux_layer()` for each probe,
then pool the window's rows with `pooled_score` or, token by token, with
`logits` and `pooled_score_from_logits`. sondekit's own `vllm` backend only
loads the model so far, so the recipes train on `hf`.

## mechanica

[mechanica](https://github.com/aniruddh-alt/mechanica) is the real-time
consumer of these files: it loads probe `.npz` files and scores live vLLM
traffic. The parts of this page it relies on:

- `serves()` and the fingerprint module decide whether a probe may score the
  running model. `sondekit/fingerprint.py` is kept identical to mechanica's
  copy, so both sides compute the same digests.
- Its in-stream scorer uses `logits()` and `pooled_score_from_logits`.
- `escalate_threshold` is mechanica's review band; sondekit leaves it unset.
- It computes the `last_turn` boundary with the same rule as
  `probe.last_turn_start`.
- `artifact_digest` is stamped on its score rows, so a threshold is never
  calibrated across two builds of a probe with the same name.

See the mechanica repository for how to configure and run it.
