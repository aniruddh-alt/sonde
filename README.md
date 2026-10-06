<p align="center">
  <img src="https://raw.githubusercontent.com/aniruddh-alt/sondekit/main/docs/img/hero.svg" alt="A probe descending through transformer layers" width="100%"/>
</p>

<h1 align="center">sondekit</h1>

<p align="center">
  <a href="https://github.com/aniruddh-alt/sondekit/actions/workflows/ci.yml"><img src="https://github.com/aniruddh-alt/sondekit/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+"/>
</p>

sondekit trains activation probes for language models from a YAML config. It
extracts residual-stream activations, fits a linear or attention probe at
each layer, picks the best layer, and writes the probe as a small `.npz`
file together with its evaluation metrics.

Probes are evaluated the way monitors are used. The headline number is
recall at 1% false-positive rate, reported with confidence intervals and
next to a bag-of-words baseline, a length baseline and a shuffled-label
control. Model access goes through [nnsight](https://nnsight.net) and
[nnterp](https://github.com/Butanium/nnterp).

## Install

sondekit needs Python 3.12 or newer.

```bash
pip install "sondekit[hf]"
```

Loading and scoring a trained probe needs only numpy, so `pip install
sondekit` without extras is enough on a machine that only serves probes.

To develop or run the bundled recipes, install from source with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/aniruddh-alt/sondekit.git
cd sondekit
uv sync --extra hf
```

## Quickstart

```bash
sondekit run quickstart
```

This trains a sentiment probe on gpt2 with a tiny bundled dataset and
finishes in a few seconds on a laptop CPU. The results go to
`runs/quickstart/`, and the train step logs the headline:

```
headline {"max_fpr": 0.01, "recall_at_fpr": 1.0, ..., "auroc": 1.0, ..., "baseline_auroc": 1.0, ...}
```

To train your own probe, write a config. This one reads rows such as
`{"id": "1", "prompt": "...", "label": 1}`:

```yaml
name: my-probe
model: {name: Qwen/Qwen3-0.6B}
data: {path: data/rows.jsonl, text: prompt, label: label}
extract: {layers: all, window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean}
steps: [extract, train]
```

```bash
sondekit run my-probe.yaml --dry-run   # validate and print the resolved config
sondekit run my-probe.yaml
sondekit run my-probe.yaml -o probe.epochs=50 -o "extract.layers=[4, 8]"
```

Besides `extract` and `train`, a run can `score` other datasets with the
trained probe, `generate` model responses, and `steer` generation along the
probe direction. [docs/configuration.md](https://github.com/aniruddh-alt/sondekit/blob/main/docs/configuration.md) lists every
option.

## Recipes

`sondekit run <name>` loads `sondekit/recipes/<name>.yaml`. The `refusal`
recipe reads `data/refusal/` relative to the current directory, so run it
from the repo root.

| Recipe | Model | Task |
|---|---|---|
| `quickstart` | gpt2 | toy sentiment probe on bundled data, CPU |
| `refusal` | Llama-3.1-8B-Instruct | predicts refusals from the prompt ([data](https://github.com/aniruddh-alt/sondekit/tree/main/data/refusal)) |
| `high_stakes` | Llama-3.1-8B | attention probe for high-stakes requests, scored on five out-of-distribution sets |

The Llama models are gated on Hugging Face, so run `huggingface-cli login`
first.

## Using a trained probe

```python
import numpy as np
from sondekit import probe

p = probe.Probe.load("runs/refusal/probes/refusal.npz")
acts = np.zeros((12, p.w.shape[0]), dtype=np.float32)  # [tokens, hidden] at p.block
score = p.pooled_score(acts)  # in [0, 1]
flagged = p.flag(acts)  # score >= p.threshold
```

The file records the model fingerprint, the chat-template digest and the
library versions the probe was trained with, and `p.serves(...)` refuses to
score a model or prompt format that differs from them. `p.metrics` holds the
evaluation metrics. [mechanica](https://github.com/aniruddh-alt/mechanica)
uses these files to score live vLLM traffic.

## GPU backends

The default `hf` backend runs on any GPU through nnterp. The `vllm` backend
is experimental: model loading works, but extraction and generation on vLLM
are not finished, so the recipes use `hf`. The `hf` and `vllm` extras
conflict, so each gets its own environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-hf uv sync --locked --extra hf
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --locked --extra vllm
```

## Documentation

- [Getting started](https://github.com/aniruddh-alt/sondekit/blob/main/docs/getting-started.md): install, the quickstart, and a first probe on your own data.
- [Data](https://github.com/aniruddh-alt/sondekit/blob/main/docs/data.md): dataset formats, columns, chat templates and extraction windows.
- [Evaluation](https://github.com/aniruddh-alt/sondekit/blob/main/docs/evaluation.md): reading `metrics.json`, baselines, controls and scoring shifted data.
- [Cookbook](https://github.com/aniruddh-alt/sondekit/blob/main/docs/cookbook.md): worked configs for common probing tasks.
- [Python API](https://github.com/aniruddh-alt/sondekit/blob/main/docs/python-api.md): loading and serving a trained probe from Python.
- [Configuration](https://github.com/aniruddh-alt/sondekit/blob/main/docs/configuration.md): every option, run outputs, rerun behavior and layer numbering.
- [Linear probes primer](https://github.com/aniruddh-alt/sondekit/blob/main/docs/linear-probes-primer.md): what linear probes measure, and the papers behind them.
- [CONTRIBUTING.md](https://github.com/aniruddh-alt/sondekit/blob/main/CONTRIBUTING.md): development setup, tests and style.

sondekit is released under the [Apache 2.0 license](https://github.com/aniruddh-alt/sondekit/blob/main/LICENSE).
