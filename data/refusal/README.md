# Refusal dataset

`labeled.jsonl` is the labeled set behind the `refusal` recipe
(`sondekit run refusal`, run from the repo root).

| | |
|---|---|
| Rows | 256 (35 `LABEL: refusal`, 221 `LABEL: non-refusal`) |
| Unique prompts | 200; 83 rows share a prompt with another row |
| Keys | `original_prompt`, `original_response`, `refusal_label`, `refusal_reasoning`, `sample_id` |

## Provenance

1. Prompts, a mix of harmful and harmless requests, were generated with
   [oumi](https://github.com/oumi-ai/oumi) synth.
2. Responses came from `meta-llama/Llama-3.1-8B-Instruct`, greedy decoding,
   256 new tokens, **without a chat template**: each response is a raw
   continuation of the prompt text.
3. An LLM judge labeled each response as a refusal or not.

## Consequences for probing

- The recipe uses `format: raw`, because the labels describe raw
  continuations. The probe's `prompt_format` is therefore `raw`, and it
  should not score chat-templated traffic.
- `group: original_prompt` keeps duplicated prompts inside one split.
- `window: prompt` reads the model before it answers, so the probe measures
  what predicts refusal (mostly "is this prompt harmful?") rather than the
  refusal itself; see `docs/linear-probes-primer.md`.
