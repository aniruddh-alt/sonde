# Refusal probing data

`data/labeled.jsonl` is the labeled set behind the `refusal` recipe
(`sonde run refusal`, from the repo root).

| | |
|---|---|
| Rows | 256 (35 `LABEL: refusal`, 221 `LABEL: non-refusal`) |
| Unique prompts | 200; 83 rows share a prompt with another row, and 4 of those prompts carry conflicting labels |
| Keys | `original_prompt`, `original_response`, `refusal_label`, `refusal_reasoning`, `sample_id` |

## How it was made

1. `1_generate_prompts.yaml` (oumi synth) wrote `data/prompts.jsonl`, a mix
   of harmful and harmless prompts.
2. sonde v0.1 (tag `v0.1-legacy`) generated `data/responses.jsonl` from
   `meta-llama/Llama-3.1-8B-Instruct`, greedy, 256 new tokens.
   **No chat template was applied**: each response is a raw continuation of
   the prompt text.
3. `2_label_responses.yaml` (oumi synth, LLM judge) wrote `data/labeled.jsonl`.

## Consequences for probing

- The recipe uses `format: raw`, because the labels describe raw
  continuations. The probe's `prompt_format` is therefore `raw`, and
  mechanica will refuse to score templated chat traffic with it.
- `group: original_prompt` keeps duplicated prompts inside one split.
- `window: prompt` reads the model before it answers. A probe there measures
  what predicts refusal (mostly "is this prompt harmful?"), not the refusal
  itself; see `docs/linear-probes-primer.md` §4.3.
