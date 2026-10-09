# Preparing data

A run reads one table of rows from the `data` block. Each row needs a text
(or a chat conversation), a binary label for training, and ideally a unique
id. This page covers the file shapes sondekit reads, how columns map to
config keys, how rows are split, which tokens each extraction window covers,
and the errors bad data produces. [configuration.md](configuration.md) has the
full option table.

`sondekit run cfg.yaml --dry-run` is the fastest way to check a dataset. When
`steps` includes `extract`, the dry run loads every row, renders it with the
model's tokenizer and prints a disk estimate, so missing columns, bad labels
and dropped rows show up without loading the model. It needs the tokenizer
and `config.json` of the model, not its weights.

## Example datasets

### Text rows in JSONL

One JSON object per line. Blank lines are skipped. With the default column
names (`text`, `label`, `id`) the config needs only the path.

`reviews.jsonl`:

```json
{"id": "r1", "text": "I loved this film, the acting was superb.", "label": 1}
{"id": "r2", "text": "A dull, lifeless movie. I walked out.", "label": 0}
{"id": "r3", "text": "Great soundtrack and a moving story.", "label": 1}
{"id": "r4", "text": "The plot made no sense and the jokes fell flat.", "label": 0}
{"id": "r5", "text": "One of the best books I have read this year.", "label": 1}
{"id": "r6", "text": "Boring from start to finish.", "label": 0}
{"id": "r7", "text": "Charming, funny and beautifully shot.", "label": 1}
{"id": "r8", "text": "I want my money back.", "label": 0}
```

`reviews.yaml`:

```yaml
name: reviews
model: {name: gpt2, dtype: float32}
data:
  path: reviews.jsonl
  format: raw
  split: [0.5, 0.25, 0.25]
extract: {layers: [4, 8], window: prompt, keep: pooled}
probe: {kind: linear, pooling: mean, epochs: 5}
score:
  - {name: unlabeled, path: unlabeled.jsonl}
steps: [extract, train, score]
```

gpt2 has no chat template, so this config uses `format: raw`. Eight rows are
enough to see the run work end to end on a CPU, but far too few for a probe
you would trust; the quickstart recipe uses 64.

### Text rows in CSV

The first line is the header. The same config reads it with
`-o data.path=reviews.csv`.

`reviews.csv`:

```
id,text,label
r1,"I loved this film, the acting was superb.",1
r2,"A dull, lifeless movie. I walked out.",0
r3,Great soundtrack and a moving story.,1
r4,The plot made no sense and the jokes fell flat.,0
r5,One of the best books I have read this year.,1
r6,Boring from start to finish.,0
r7,"Charming, funny and beautifully shot.",1
r8,I want my money back.,0
```

Every CSV value is a string. Labels `"0"` and `"1"` are accepted as they are,
and `label_map` keys must match the strings in the file. A CSV cell cannot
hold a chat conversation, so use JSONL or a Hugging Face dataset for
`messages`.

### Prompt and response with string labels

A `response` column holds the model's reply. Labels here are strings, mapped
to 0 and 1 with `label_map`, and `group` keeps rows about the same topic in
one split.

`replies.jsonl`:

```json
{"id": "q1", "prompt": "How do I pick a lock?", "response": "I can't help with that.", "verdict": "refusal", "topic": "locks"}
{"id": "q2", "prompt": "How do I pick a lock?", "response": "Insert a tension wrench, then rake the pins.", "verdict": "complied", "topic": "locks"}
{"id": "q3", "prompt": "Write a poem about rain.", "response": "Soft rain on the roof, a quiet drum.", "verdict": "complied", "topic": "poems"}
{"id": "q4", "prompt": "Write a poem insulting my coworker.", "response": "I'd rather not write something meant to hurt someone.", "verdict": "refusal", "topic": "poems"}
```

`replies.yaml`:

```yaml
name: replies
model: {name: Qwen/Qwen3-0.6B}
data:
  path: replies.jsonl
  text: prompt
  response: response
  label: verdict
  label_map: {refusal: 1, complied: 0}
  group: topic
  chat_template_kwargs: {enable_thinking: false}
extract: {layers: all, window: response, keep: pooled}
probe: {kind: linear, pooling: last, init: diff_means, epochs: 0}
steps: [extract, train]
```

This file and the chat one below show the shape of the data only. Four rows
cannot fill three splits with both classes, so `train` needs more rows than
this.

### Chat conversations

A `messages` column holds a list of `{"role": ..., "content": ...}` objects,
passed to the tokenizer's chat template unchanged. Conversations can have a
system message and several turns. Setting `messages` clears the default
`text: text`, so the config sets only `messages`.

`chats.jsonl`:

```json
{"id": "c1", "messages": [{"role": "user", "content": "How do I pick a lock?"}, {"role": "assistant", "content": "I can't help with that."}], "label": 1}
{"id": "c2", "messages": [{"role": "user", "content": "Write a poem about rain."}, {"role": "assistant", "content": "Soft rain on the roof, a quiet drum."}], "label": 0}
{"id": "c3", "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Explain how to hotwire a car."}, {"role": "assistant", "content": "Sorry, I won't explain that."}], "label": 1}
{"id": "c4", "messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}, {"role": "user", "content": "And of Spain?"}, {"role": "assistant", "content": "Madrid."}], "label": 0}
```

`chats.yaml`:

```yaml
name: chats
model: {name: Qwen/Qwen2.5-0.5B-Instruct}
data:
  path: chats.jsonl
  messages: messages
extract: {layers: all, window: last_turn, keep: tokens}
probe: {kind: linear, pooling: rolling_mean, rolling_window: 4}
steps: [extract, train]
```

### Hugging Face datasets

Set `hf` instead of `path`. `hf_config` picks the dataset configuration and
`hf_split` the split (default `train`). This is the `data` block of the
`high_stakes` recipe:

```yaml
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
```

The rows come from `datasets.load_dataset(hf, hf_config, split=hf_split)`,
so the whole split is downloaded before `limit` subsamples it. Gated datasets
need `huggingface-cli login`.

### Unlabeled rows

Rows for `score`, `generate` or `steer` can leave the label out:

`unlabeled.jsonl`:

```json
{"id": "u1", "text": "An instant classic."}
{"id": "u2", "text": "Two hours I will never get back."}
```

The `reviews` run above scores this file and writes
`score/unlabeled/scores.jsonl` with one `{"id", "score", "flag"}` row each,
and no `metrics.json`, because there are no labels to measure against.

## Column mapping

Each key names a column in the rows. Relative `path` values resolve against
the directory you run `sondekit` from (bundled recipes resolve their own data
paths).

| Key | Default | Column holds |
|---|---|---|
| `text` | `text` | the prompt as a string |
| `messages` | `null` | a list of chat messages; set this or `text`, not both |
| `response` | `null` | the model's reply as a string; needed for `window: response` |
| `label` | `label` | 0 or 1, or a value `label_map` maps to 0 or 1; `null` reads no labels |
| `id` | `id` | a unique id per row; `null` numbers the rows |
| `group` | `null` | a key that keeps related rows in one split |

Extra columns are ignored by extraction. `generate` copies them into its
output rows.

### Labels and `label_map`

After mapping, a label must be `0`, `1`, `"0"` or `"1"`. `label_map` maps raw
values to those, and values missing from the map pass through unchanged, so
an unmapped value still fails the check:

```yaml
label: verdict
label_map: {refusal: 1, complied: 0}
```

YAML reads unquoted keys with their natural type. Quote a key that looks like
a number or contains a colon, as the `refusal` recipe does:
`label_map: {"LABEL: refusal": 1, "LABEL: non-refusal": 0}`.

### Ids

Ids are stored as strings and appear in `splits.json`, `scores.jsonl`, the
extract manifest and `generations.jsonl`. A row without the id key falls back
to its row number, counted from 0 in file order before any `limit`
subsample. `id: null` numbers every row that way, which is useful when a
dataset reuses ids for different rows. Two rows with the same id are an
error.

## Subsampling with `limit`

`limit: N` keeps a random subsample of N rows, drawn with the run's `seed`
and kept in file order. The same seed always picks the same rows. A `limit`
larger than the dataset keeps every row. Only the kept rows are validated.

## Splits

`train` divides the extracted rows into train, validation and test with
`data.split` (default `[0.7, 0.15, 0.15]`). The fractions must be
non-negative and sum to 1. The probe is fitted on train, its threshold and
layer are chosen on validation, and test is used only for the reported
metrics.

The split is stratified by label and seeded by `seed`. Each class is shuffled
on its own and dealt into the three splits by the fractions, so every split
has about the same positive rate. The ids of each split are written to
`splits.json`, and the log reports the result:

```
INFO sondekit.data: split train: n=4 (0.500) n_pos=2 n_neg=2
INFO sondekit.data: split val: n=2 (0.250) n_pos=1 n_neg=1
INFO sondekit.data: split test: n=2 (0.250) n_pos=1 n_neg=1
```

Every split needs at least one positive and one negative row, or the run
stops with an error that names the split.

### Group-aware splits

Rows that share a `group` value always land in the same split. Use it when
rows are near-duplicates of each other, such as several responses to one
prompt or a matched pair of situations, so the test split does not contain
rows the probe has effectively already seen. The `refusal` recipe groups by
`original_prompt` and `high_stakes` by `pair_id`.

With groups, whole groups are dealt into splits, so the actual fractions can
drift from `data.split` when groups are large; the log line shows the actual
share. A group is stratified by the majority label of its rows, and a group
with as many positives as negatives counts as negative. Group values are
compared as strings, so a row whose group value is `null` gets the group
`"None"`, and all such rows travel together. `probe.select: group_auroc`
needs `data.group`.

## Chat or raw format

`data.format` decides how a row becomes tokens.

| Format | Rendering |
|---|---|
| `chat` (default) | the tokenizer's chat template, with the generation prompt appended |
| `raw` | the text tokenized as is, with the tokenizer's default special tokens (BOS for Llama) |

Under `chat`, a `text` row becomes one user message, preceded by a system
message when `data.system` is set. `data.system` does not apply to
`messages` rows, which carry their own system message if they have one. A
tokenizer without a chat template fails with `format: chat`, so base models
such as gpt2 and Llama-3.1-8B need `format: raw`. `raw` works only with
`text` rows.

`data.chat_template_kwargs` is passed to the chat template. For Qwen3, set
`{enable_thinking: false}` to probe prompts and replies without a thinking
block. The generation prompt then ends in an empty `<think>\n\n</think>\n\n`
block, and those tokens are part of the prompt window.

Choose the format the probe will be served under. The probe file records a
digest of the chat template and any `chat_template_kwargs` (or `raw`), and
refuses to score prompts rendered another way.

## Extraction windows

`extract.window` picks which tokens of each rendered row the probe reads.

| Window | Tokens | Needs |
|---|---|---|
| `prompt` (default) | the rendered prompt, including the system message and the generation prompt under `chat` | |
| `response` | only the response tokens, which follow the prompt | `data.response` |
| `all` | the prompt followed by the response | |
| `last_turn` | the final message of a conversation as the template renders it, including its end-of-turn tokens | `format: chat`, `messages` rows ending in an `assistant` message |

For `response` and `all`, the response is tokenized without special tokens
and appended straight after the prompt, with no end-of-turn token. When
`data.response` is set and a row also has a `response_ids` column (which
`generate` writes), those exact ids are used instead of re-tokenizing the
response text. With no response, `all`
covers only the prompt.

`last_turn` renders the whole conversation and starts the window where the
conversation without its last message, rendered with a generation prompt,
ends. A `text` row renders as a single user message, so it is always dropped
under `last_turn`.

### Worked example

This row, with `system: Answer briefly.` and `response: response`, rendered
by the Qwen2.5-0.5B-Instruct tokenizer:

```json
{"text": "Is 7 prime?", "response": "Yes."}
```

The full prompt is 21 tokens:

```
<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\nIs 7 prime?<|im_end|>\n<|im_start|>assistant\n
```

| Window | Span | Text the probe reads |
|---|---|---|
| `prompt` | 0 to 21 | `<\|im_start\|>system\nAnswer briefly.<\|im_end\|>\n<\|im_start\|>user\nIs 7 prime?<\|im_end\|>\n<\|im_start\|>assistant\n` |
| `response` | 21 to 23 | `Yes.` |
| `all` | 0 to 23 | the prompt followed by `Yes.` |
| `last_turn` | 21 to 25 | `Yes.<\|im_end\|>\n` |

The `last_turn` row is the same conversation given as `messages` (system,
user, assistant). It reads the reply plus the template's end-of-turn tokens,
which `response` does not. With the Qwen3-0.6B tokenizer and default
settings, the template adds an empty thinking block to the final assistant
turn, so the same `last_turn` window reads
`<think>\n\n</think>\n\nYes.<|im_end|>\n`.

Under `format: raw` with gpt2, there is no template and nothing is inserted
between prompt and response, so a response needs its own leading space or
newline. With the response written as `" Yes."`:

| Window | Span | Tokens |
|---|---|---|
| `prompt` | 0 to 4 | `Is`, ` 7`, ` prime`, `?` |
| `response` | 4 to 6 | ` Yes`, `.` |
| `all` | 0 to 6 | `Is`, ` 7`, ` prime`, `?`, ` Yes`, `.` |

## Using generated responses

`generate` writes `generations.jsonl` with every source column plus
`response` and `response_ids`. That file can be the `data.path` of a later
run with `response: response`, after you label the responses. Because the
exact generated ids are reused, the response window holds the tokens the
model produced, even where decoding and re-encoding would differ.

## Length limits and dropped rows

Each rendered row is cut to its first `min(data.max_length,
tokenizer.model_max_length)` tokens. `max_length` defaults to 2048, and the
tokenizer cap stops a model from running past its position embeddings
(gpt2's is 1024). Truncation keeps the start of the row, so a long prompt
loses its end, including the generation prompt under `chat`, and under
`response` a long prompt uses up the budget before the response starts.

A row whose window ends up empty is dropped, counted by reason:

| Reason | Cause |
|---|---|
| `empty_response` | `window: response` and the row's response is empty or null |
| `empty_window` | truncation removed every token of the window |
| `not_assistant_last` | `window: last_turn` and the last message is not from `assistant` |
| `last_turn_prefix_mismatch` | `window: last_turn` and the template renders the earlier turns differently once the last message follows them, so the boundary is unreliable |

Drops are logged as a warning and recorded under `drops` in
`extract/manifest.json` (and in `score/<entry>/extract/manifest.json` for
score entries):

```
WARNING sondekit.data: dropped 2 of 8 rows: {'empty_window': 2}
```

If every row is dropped the run stops:

```
sondekit: error: every row was dropped: {'empty_window': 8}
```

## Which steps need labels

| Step | Labels |
|---|---|
| `extract` | the label column must exist in every row, unless `data.label: null` |
| `train` | every row needs a label; with `data.label: null` it fails with `train needs a label on every sample` |
| `score` | optional; `metrics.json` is written only when every scored row has a label |
| `generate`, `steer` | not needed |

In `score`, `generate` and `steer`, a row without the label column gets no
label, but a label that is present must still be 0 or 1 after `label_map`.

Score entries inherit every `data` key they do not set. Setting `path` or
`hf` in an entry replaces both of the inherited pair, and the same holds for
`text` and `messages`. An inherited `limit` applies too, so set `limit: null`
in an entry to score a whole set. `split` is ignored in score entries. See
[evaluation](evaluation.md) for reading the results.

## Errors

Data errors print one line and exit with status 2. Rows are numbered from 0
in file order; JSONL parse errors give the file line, counted from 1.

| Problem | Message |
|---|---|
| duplicate id | `row 1: duplicate id 'a'` |
| empty file | `bad/empty.jsonl has no rows` |
| malformed JSONL line | `bad/badline.jsonl:2: not valid JSON: Illegal trailing comma before end of object: line 1 column 36 (char 35)` |
| label column missing in a row | `row 1 has no 'label' key; keys: ['id', 'text']` |
| text column missing | `row 0 has no 'text' key; keys: ['id', 'label', 'prompt']` |
| label not 0/1 | `row 0: label 'positive' is not 0/1; set data.label_map` |
| wrong file type | `data.path must be .jsonl or .csv, got bad/rows.txt` |
| file not found | `[Errno 2] No such file or directory: 'bad/missing.jsonl'` |
| chat format, no template | `data.format is chat but the gpt2 tokenizer has no chat_template; set data.format: raw` |
| `last_turn` with raw | `extract.window: last_turn needs data.format: chat` |
| a split lacks a class | `split 'val' has n_pos=0, n_neg=0; it needs both classes. Add data or change data.split` |

Each message follows `sondekit: error: `. Config mistakes are caught when the
config loads, before any data is read, and the pydantic message contains one
of these:

| Problem | Message |
|---|---|
| both or neither of `path` and `hf` | `data: set exactly one of path / hf` |
| both `text` and `messages` | `data: set exactly one of text / messages` |
| `messages` with `format: raw` | `data.format: raw needs text rows, not messages` |
| `window: response` without a response column | `extract.window: response needs data.response` |
| fractions that do not sum to 1 | `data.split (0.8, 0.2, 0.1) must be >= 0, sum to 1` |

Once the data loads, [evaluation](evaluation.md) explains what `train` and
`score` report, and [cookbook](cookbook.md) has complete configs that use
these formats.
