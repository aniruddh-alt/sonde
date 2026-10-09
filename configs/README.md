# Configs

Example configs for common probing tasks. Each one is a complete `sondekit
run` config whose header lists its requirements, the commands to run it, and
the results to expect.

Run everything from the repo root:

```bash
python configs/examples/prepare_data.py          # builds configs/examples/data/
sondekit run configs/examples/truth/train.yaml
```

`prepare_data.py` downloads and assembles the datasets that are not on the
Hugging Face Hub in a usable form. `sentiment/` reads the Hub directly and
needs no preparation. The data directory is gitignored.

## Examples

| Folder | Model | What it shows |
|---|---|---|
| [`sentiment/`](examples/sentiment) | Qwen3-0.6B, Qwen3-8B | SST-2 probe scored on IMDB; data straight from the Hub |
| [`truth/`](examples/truth) | Qwen3-1.7B | true vs false statements, group-aware splits, failed transfer to negations |
| [`keyword_control/`](examples/keyword_control) | Qwen3-0.6B | a task the bag-of-words baseline solves, as a sanity check |
| [`harmful_prompts/`](examples/harmful_prompts) | Qwen3-1.7B | harmful vs harmless prompts with six poolings, out-of-distribution and benign-only score sets |
| [`refusal_direction/`](examples/refusal_direction) | Qwen2.5-1.5B-Instruct | a difference-of-means direction, then ablating and adding it during generation |
| [`response_probe/`](examples/response_probe) | Qwen2.5-1.5B-Instruct | generate replies, label them, probe the reply tokens |
| [`qwen3_no_thinking/`](examples/qwen3_no_thinking) | Qwen3-1.7B | `chat_template_kwargs` to turn off Qwen3's thinking block |

Steps that depend on each other are numbered in their headers ("step 1 of
3"). Run them in that order.

## Bundled recipes

`sondekit/recipes/` holds three configs that run by name, without a path:
`sondekit run quickstart`, `sondekit run refusal` and `sondekit run
high_stakes`. They ship inside the package.

## Writing your own

Copy the closest example and change `name`, `model` and `data`. The
[configuration reference](../docs/configuration.md) lists every option and
the [cookbook](../docs/cookbook.md) explains the choices behind these
examples.
