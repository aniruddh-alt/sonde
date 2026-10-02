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

.venv-hf/bin/sonde run refusal
```

The `vllm` backend is experimental: its loader is in, but extraction,
generation and the parity spike have not run on a GPU yet, so the recipes
default to `hf`, and `forward` and `generate` on vllm raise
`NotImplementedError`.

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
generations.jsonl        source row + response, response_ids
steer/s<strength>.jsonl  one file per strength
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
- `generate` skips ids already in generations.jsonl. If the model, data,
  generate or seed settings changed, it refuses before loading anything;
  `output.overwrite: true` regenerates.
- `steer` overwrites its files.

## Config reference

| Key | Default | Notes |
|---|---|---|
| `name` | required | run dir is `<output.dir>/<name>` |
| `seed` | `0` | drives subsampling, splits, init and data order |
| `steps` | `[extract, train]` | `generate`, `extract`, `train`, `score`, `steer`, run in the listed order |
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
| `generate.max_tokens`, `generate.temperature`, `generate.top_p` | `256`, `0.7`, `1.0` | `temperature: 0` is greedy; top-k is off |
| `steer.probe` | `null` | defaults to this run's `probes/<name>.npz`; direction is w/‖w‖ at its block |
| `steer.mode`, `steer.strengths` | `add`, `[0.0]` | `add`: x += s·v; `ablate`: x −= s·(x·v)·v; strength 0 is the unedited baseline |
| `output.dir`, `output.overwrite` | `runs`, `false` | |

A section is required only when a listed step needs it:

| Step | Sections |
|---|---|
| `extract` | `model`, `data`, `extract`, and `probe` when `keep: pooled` |
| `train` | `data`, `probe` |
| `score` | `model`, `data`, `score` |
| `generate` | `model`, `data`, `generate` |
| `steer` | `model`, `data`, `generate`, `steer` |

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
