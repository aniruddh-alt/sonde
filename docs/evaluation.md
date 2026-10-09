# Evaluation

This page explains what the `train` and `score` steps measure, where each
number is written, and how to decide whether a probe can be trusted. For the
options themselves see [configuration](configuration.md). For the ideas
behind probes see the [primer](linear-probes-primer.md).

The examples come from the quickstart recipe:

```bash
sondekit run quickstart
```

The run writes to `runs/quickstart/`.

## What the train step does

For every extracted layer, `train` fits one probe on the train split, picks
its threshold on the validation split, and records validation metrics. It
then selects one layer, scores the test split once with that layer's probe,
and runs three comparisons on the same splits:

- a bag-of-words baseline,
- a length baseline,
- five shuffled-label controls.

Everything goes to `metrics.json` in the run directory. The same dictionary
is stored inside `probes/<name>.npz`, so `Probe.load(path).metrics` returns
it wherever the `.npz` travels. Each `layers/B<layer>.npz` carries only its own
validation metrics, under `metrics["val"]`.

The splits come from `data.split` (default `[0.7, 0.15, 0.15]`), stratified
by label. Rows that share a `data.group` value stay in one split.
`splits.json` lists the ids in each split. The run seed fixes the split, the
initialization and the batch order, so running the same config twice gives
the same numbers.

## metrics.json

This is the quickstart's `metrics.json`, trimmed to one layer of the
validation table:

```json
{
  "headline": {
    "max_fpr": 0.01,
    "recall_at_fpr": 1.0,
    "recall_at_fpr_ci": [1.0, 1.0],
    "threshold": 0.9999760985374451,
    "recall": 0.75,
    "fpr": 0.0,
    "auroc": 1.0,
    "auroc_ci": [1.0, 1.0],
    "baseline_recall_at_fpr": 1.0,
    "baseline_auroc": 1.0,
    "length_auroc": 0.5,
    "n_pos": 8,
    "n_neg": 8,
    "small_n": true
  },
  "block": 8,
  "select": "recall_at_fpr",
  "select_fallback": "auroc",
  "max_fpr": 0.01,
  "threshold": 0.9999760985374451,
  "threshold_failed": false,
  "val": {
    "8": {
      "n_pos": 8,
      "n_neg": 8,
      "threshold": 0.9999760985374451,
      "auroc": 1.0,
      "f1": 1.0,
      "precision": 1.0,
      "recall": 1.0,
      "fpr": 0.0,
      "small_n": true,
      "recall_at_fpr": 1.0,
      "threshold_failed": false,
      "val_loss": 1.0650228318752821e-05
    }
  },
  "test": {"n_pos": 8, "n_neg": 8, "threshold": 0.9999760985374451, "auroc": 1.0, "...": "..."},
  "baseline": {"n_pos": 8, "n_neg": 8, "threshold": 0.9984102845191956, "auroc": 1.0, "...": "..."},
  "control_auroc": 0.5625,
  "control_aurocs": [0.78125, 0.890625, 0.421875, 0.421875, 0.296875],
  "controls_passed": true,
  "n": {"train": 32, "val": 16, "test": 16}
}
```

The quickstart data is tiny, so every number here is saturated and
`small_n` is true. The file shows the layout; the sections below explain
why numbers like these should not be trusted.

### Top-level fields

| Field | Meaning |
|---|---|
| `headline` | the summary on the test split, described in the next table |
| `block` | the selected layer |
| `select` | the configured selection metric (`probe.select`) |
| `select_fallback` | `"auroc"` when selection fell back from `recall_at_fpr` because the validation split is `small_n`, else `null` |
| `max_fpr` | the FPR budget (`probe.max_fpr`) |
| `threshold` | the selected probe's threshold, chosen on validation |
| `threshold_failed` | true when no validation threshold met `max_fpr` and the best-F1 threshold was used instead |
| `val` | per-layer validation metrics, keyed by layer as a string |
| `test` | metrics of the selected probe on the test split |
| `baseline` | the bag-of-words baseline's metrics on the test split |
| `control_auroc` | mean validation AUROC of the shuffled-label fits |
| `control_aurocs` | the AUROC of each shuffled-label fit |
| `controls_passed` | whether the shuffled fits are consistent with chance |
| `n` | number of rows in train, val and test |

### Per-set metrics

`val[<layer>]`, `test`, `baseline` and `score/<entry>/metrics.json` share one
set of fields, computed on one labelled set at one threshold. A row is
flagged when `score >= threshold`.

| Field | Meaning |
|---|---|
| `n_pos`, `n_neg` | positives and negatives in the set |
| `threshold` | the threshold the counts below use |
| `auroc` | rank-based AUROC; tied scores share their mean rank |
| `f1`, `precision`, `recall`, `fpr` | at `threshold` |
| `small_n` | `n_neg * max_fpr < 10` |
| `recall_at_fpr` | recall at this set's own lowest threshold with FPR at or below `max_fpr`; `0.0` when no threshold meets it; present only when `max_fpr` is set |
| `group_auroc`, `n_groups` | only when the set has a `group` column, see [layer selection](#layer-selection) |

A set with only one class does not raise. A benign-only set reports `fpr`
and leaves `auroc`, `f1`, `precision`, `recall` and `recall_at_fpr` as
`null`. A positive-only set reports `recall` and leaves `fpr` `null`.

`val[<layer>]` adds two fields: `threshold_failed` and `val_loss`, the mean
binary cross-entropy of the probe's scores on validation.

### The headline

All headline numbers are on the test split, which is scored once with the
selected probe.

| Field | Meaning |
|---|---|
| `max_fpr` | the FPR budget |
| `recall_at_fpr` | recall at `max_fpr`, with the threshold re-chosen on test |
| `recall_at_fpr_ci` | 95% bootstrap interval of `recall_at_fpr`; `null` when `max_fpr` is `null` |
| `threshold` | the frozen threshold from validation |
| `recall`, `fpr` | recall and FPR on test at the frozen threshold |
| `auroc`, `auroc_ci` | test AUROC and its 95% bootstrap interval |
| `baseline_recall_at_fpr`, `baseline_auroc` | the same two metrics for the bag-of-words baseline |
| `length_auroc` | AUROC of the window's word count used as the score |
| `n_pos`, `n_neg` | test positives and negatives |
| `small_n` | test has too few negatives for `max_fpr` |
| `group_auroc` | test group AUROC, only when `data.group` is set |

## Thresholds

The threshold is chosen on validation and frozen. With `probe.max_fpr` set
(default `0.01`), it is the lowest validation score whose validation FPR is
at or below `max_fpr`, which gives the highest recall that fits the budget.
If no threshold qualifies, which happens when negatives tie with the top
score, the best-F1 threshold is used and `threshold_failed` is true. With
`max_fpr: null` the threshold is the best-F1 one, and `probe.select` must be
`auroc` or `group_auroc`. Otherwise the config fails to load with a
validation error whose message reads:

```
probe.select: recall_at_fpr needs probe.max_fpr; set select: auroc to run without one
```

The threshold is saved in the `.npz` and used unchanged by `Probe.flag`, the
`score` step and serving. Two kinds of recall follow from this, and the
headline reports both:

- `recall` and `fpr` use the frozen threshold. This is what a deployed probe
  does on new data from the same distribution.
- `recall_at_fpr` re-chooses the threshold on test. It measures how well the
  test scores rank, assuming you could recalibrate.

When the two disagree, the threshold did not transfer. In the quickstart,
`recall_at_fpr` is 1.0 and `recall` is 0.75: the scores rank perfectly, but
the threshold is the lowest validation score above all 8 validation
negatives, and two test positives score below it.

### small_n

`small_n` is true when `n_neg * max_fpr < 10`, meaning the FPR budget allows
fewer than ten false positives on the set. At 1% FPR a split needs at least
1000 negatives to clear it. Below that the threshold rests on the few
highest-scoring negatives, and `fpr` at the frozen threshold can move a lot
on new data. sondekit logs a warning each time:

```
WARNING sondekit.train: small_n: 8 negatives x max_fpr 0.01 < 10; the FPR threshold rests on too few negatives
```

With a small validation split, selecting layers by `recall_at_fpr` would
compare layers on one or two negatives, so selection falls back to AUROC and
`select_fallback` is set to `"auroc"`. Raise the data size, give validation
a larger fraction with `data.split`, or loosen `probe.max_fpr` to get
`small_n: false`.

### Confidence intervals

`auroc_ci` and `recall_at_fpr_ci` are 95% percentile bootstrap intervals
over the test split: 1000 resamples with replacement, seeded by the run
seed. Resamples that lose a class are skipped. `recall_at_fpr_ci`
re-chooses the threshold in each resample.

A bootstrap cannot show uncertainty that the sample does not contain. The
quickstart's `[1.0, 1.0]` comes from 16 test rows that all rank correctly,
and says nothing about the next thousand rows. Read the intervals together
with `n_pos`, `n_neg` and `small_n`.

## Layer selection

`val` holds one entry per extracted layer. This prints it as a table:

```python
import json

m = json.load(open("runs/quickstart/metrics.json"))
print("block  auroc  recall_at_fpr  val_loss")
for block, v in m["val"].items():
    print(f"{block:>5}  {v['auroc']:.3f}  {v['recall_at_fpr']:.3f}  "
          f"{v['val_loss']:.2e}")
print("selected", m["block"], "by", m["select_fallback"] or m["select"])
```

```
block  auroc  recall_at_fpr  val_loss
    0  1.000  1.000  2.84e-05
    1  1.000  1.000  2.26e-05
    2  1.000  1.000  1.98e-05
    3  1.000  1.000  1.26e-05
    4  1.000  1.000  1.15e-05
    5  1.000  1.000  1.21e-05
    6  1.000  1.000  1.13e-05
    7  1.000  1.000  1.12e-05
    8  1.000  1.000  1.07e-05
    9  1.000  1.000  1.22e-05
   10  1.000  1.000  1.53e-05
   11  1.000  1.000  1.65e-05
selected 8 by auroc
```

`probe.select` sets the metric that picks the layer:

| `select` | Picks the layer with the highest validation |
|---|---|
| `recall_at_fpr` (default) | recall at `max_fpr`; falls back to `auroc` when validation is `small_n` |
| `auroc` | AUROC |
| `group_auroc` | mean AUROC within groups; needs `data.group` |

Ties are broken by validation AUROC, then by the lower `val_loss`. AUROC
often saturates at 1.0 on easy tasks, and the lower log-loss picks the layer
that separates the classes with the larger margin. In the table above every
layer ties on AUROC, and layer 8 wins on `val_loss`.

`group_auroc` computes AUROC separately inside each group that holds both
classes and averages them. `n_groups` counts those groups. Use it when rows
come in sets that are compared with each other, such as several completions
of one prompt or matched pairs of situations. It asks whether the probe
ranks within a group, which a probe can fail while ranking well across
groups.

A flat table near 1.0 means the task is easy to
read from any layer, and the baselines decide whether the probe is reading
anything beyond the words.

## Baselines

Both baselines are evaluated on the same test split as the probe.

**Bag-of-words.** The text of each row's window (the prompt for `window:
prompt`, the response for `response`, both for `all`, the last message for
`last_turn`) is lowercased and split into words. The 5000 most frequent
words in the train split form the vocabulary. A linear probe is fitted on
word counts with the same probe settings, its threshold is chosen on
validation, and its test metrics go to `baseline`, with
`baseline_auroc` and `baseline_recall_at_fpr` copied to the headline.

**Length.** `length_auroc` is the test AUROC of the window's word count used
directly as the score. 0.5 means length carries no information about the
label. Values far from 0.5 in either direction mean it does: 0.2 means
shorter rows tend to be positive.

How to read them:

- A probe that matches the bag-of-words baseline has not shown that it reads
  anything the surface words do not already give away. It can still be
  useful as a monitor, but a word classifier would do the same job, and it
  is likely to fail on rephrased inputs. The quickstart is such a case: its
  sentiment words make the task trivial for both.
- A probe that beats the baseline on AUROC or recall at `max_fpr` is
  reading something in the activations beyond word identity. Check that the
  lower end of the confidence interval also clears the baseline.
- If `length_auroc` is far from 0.5, check whether the probe is just
  measuring length. `max` pooling is the usual culprit: the maximum of more
  token logits tends to be higher, so a max-pooled probe can learn sequence
  length, and the shuffled-label control then fails too. Balance lengths
  across classes in the data, or switch to `mean`, `last`, `rolling_mean` or
  `attention`.

## Shuffled-label control

The control asks whether the probe setup scores above chance when there is
no label to learn. At the selected layer, sondekit permutes all labels five
times (seeds `seed`, `seed + 1`, ..., `seed + 4`), refits the probe on the
train split with each permutation, and computes its validation AUROC against
the true labels. `control_aurocs` lists the five values and `control_auroc`
is their mean.

The control passes when both of these hold:

```
|mean - 0.5| <= max(2 * SE, 0.05)
|a - 0.5|    <= max(4 * sd0, 0.05)   for every shuffle AUROC a
```

SE is the standard error of the five AUROCs (sample standard deviation
divided by the square root of 5). sd0 is the standard deviation of AUROC
when scores are independent of labels,
`sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))` on the validation split.

The first check catches shuffles that sit off chance in the same direction.
The probe can then score the true labels without learning them, through a
property that correlates with the label, such as length under `max`
pooling.

The second check catches a single shuffle far from 0.5. That happens when
the label is the strongest direction of variance in the activations: a fit
to random labels lands on that axis with either sign and separates the
classes anyway. The probe's own score then says little about what it
learned. Difference-of-means probes fail this check on most easy tasks,
because a mean difference over random labels points along the top direction
of variance.

When `controls_passed` is false, treat the headline as suspect until you
find the cause. Check `length_auroc` and the bag-of-words baseline, and set
`data.group` if related rows could land in different splits.

Both checks are loose on small validation sets. In the quickstart (8
positives and 8 negatives) sd0 is 0.15, so shuffles from 0.30 to 0.89 still
pass. A pass on a few dozen rows tells you little.

## The score step

`score` applies a trained probe to other datasets. Each entry in the `score`
list names a dataset; keys it does not set are inherited from `data`. The
quickstart scores a held-out file:

```yaml
score:
  - {name: heldout, path: data/quickstart_heldout.jsonl}
steps: [extract, train, score]
```

`probe:` in an entry points at another `.npz`; by default the entry uses
this run's `probes/<name>.npz`. The entry inherits `data.limit` too, so set
`limit: null` on an entry to score a set in full when `data` subsamples (the
`high_stakes` recipe does this).

Before extracting, `score` checks that the probe's backend, model
fingerprint and prompt format match this run, and refuses otherwise:

```
sondekit: error: probe runs/quickstart/probes/quickstart.npz has model_fingerprint='hub:c9a84be385409e6005a141e99416aefc' but this run has 'hub:caf008f07f3a29bdd080111b0be59a2d'; its scores would be meaningless here, so score refuses it
```

That is the gpt2 quickstart probe scored with `model.name` set to
`sshleifer/tiny-gpt2`. The backend is checked first, then the fingerprint,
then the prompt format, and each refusal names the field that differs.

It extracts only the probe's layer and window and writes, under
`score/<entry>/`:

| File | Contents |
|---|---|
| `extract/` | the activations for this set |
| `scores.jsonl` | one row per sample: `{"id", "score", "flag"}`, with `flag = score >= threshold` |
| `metrics.json` | the per-set metrics above, only when every row has a label |

```
{"id": "quickstart_heldout-0", "score": 0.47642144560813904, "flag": false}
{"id": "quickstart_heldout-1", "score": 2.6355279260315e-05, "flag": false}
```

Rows without the label key are scored and get no metrics, so an unlabelled
set produces `scores.jsonl` only. `max_fpr` comes from this run's `probe`
section when it has one, else from the probe's own `metrics["max_fpr"]`.

### Frozen and re-thresholded numbers

`score/<entry>/metrics.json` reports the deployed operating point: `recall`
and `fpr` at the probe's frozen threshold. `recall_at_fpr` and `auroc`
re-rank on this set and ignore the threshold. The quickstart's held-out set
shows why both are reported:

```json
{
  "n_pos": 16,
  "n_neg": 16,
  "threshold": 0.9999760985374451,
  "auroc": 1.0,
  "f1": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "fpr": 0.0,
  "small_n": true,
  "recall_at_fpr": 1.0
}
```

The probe ranks every held-out row correctly, yet flags none of them,
because its threshold was fitted to 8 validation negatives and every
held-out score falls below it. A monitor built on this probe would catch
nothing, and `recall_at_fpr` alone would hide that.

### Out-of-distribution sets

The test split comes from the same distribution as the training data, so a
high test score shows the probe works on more of the same. Monitors meet
other phrasings, other topics and adversarial prompts. A probe can look
perfect in distribution and break on a shifted set. The
[truth example](../configs/examples/truth/train.yaml) scores AUROC 0.995 on
its test split, but on negated statements its saved threshold flags 70% of
the false ones as true. Score entries exist to catch this before deployment.

Pick sets that differ from training in the ways deployment will: other
sources, other templates, jailbreak or paraphrased prompts. Compare
`auroc` across entries for ranking, and `recall` and `fpr` at the frozen
threshold for what the deployed probe would do. In the
[harmful-prompt examples](../configs/examples/harmful_prompts),
`rolling_mean` and `attention` probes transfer best to jailbreak prompts,
so they are worth trying when OOD recall drops.

### Measuring false positives on benign traffic

A set of benign rows labelled `0` measures the false-positive rate the
deployed threshold would produce. This config scores a probe on such a set
without retraining:

```yaml
name: quickstart
model: {name: gpt2, dtype: float32}
data: {path: data/benign.jsonl, format: raw}
score:
  - {name: benign, probe: runs/quickstart/probes/quickstart.npz}
steps: [score]
```

```json
{
  "n_pos": 0,
  "n_neg": 16,
  "threshold": 0.9999760985374451,
  "auroc": null,
  "f1": null,
  "precision": null,
  "recall": null,
  "fpr": 0.0,
  "small_n": true,
  "recall_at_fpr": null
}
```

`fpr` is the realized false-positive rate at the frozen threshold. Use as
many benign rows as you can; with `max_fpr: 0.01`, the set needs 1000
negatives before `small_n` turns false. Running `score` again overwrites the
entry's outputs and the run's `config.yaml`.

## How training works

The fitting method depends on the probe and on what `extract` stored.
[Configuration](configuration.md#optimizers) has the table of which
optimizer runs and which options it reads.

**Pooled linear probes** are logistic regression, which is convex, so
sondekit solves it with full-batch L-BFGS on the whole train split, up to
500 iterations. `weight_decay` times the squared weight norm keeps weights
finite when the classes separate perfectly. `epochs`, `lr`, `batch_size` and `patience` are
ignored on this path.

**Token-level probes** (`keep: tokens`) train with AdamW on minibatches of
`batch_size` samples, shuffled with the run seed. Weight decay skips the
bias. After each epoch the validation loss is computed. Training stops
after `patience` epochs without improvement (`0` disables early stopping),
and the weights from the best epoch are kept. Training uses CUDA when it
is available. The defaults are listed in
[configuration](configuration.md).

Both optimizers weight the positive class by `n_neg / n_pos`, so imbalanced
data does not push the probe toward always predicting the majority class.

**Standardization.** Before fitting, features are standardized with the
mean and standard deviation of the train split (for token data, over every
window token of the train samples; standard deviations are floored at
`1e-6`). On export the scaling is folded into the weights and bias, and for
attention probes into the query vector, so the `.npz` scores raw
activations and needs no preprocessing.

**diff_means.** `init: diff_means` with `kind: linear` and `epochs: 0`
skips training. The direction is the difference between the mean positive
and mean negative train activation, scaled so the train logits have unit
standard deviation, with the bias at the midpoint between the two class
means. Any other combination fails validation with:

```
probe.init: diff_means needs kind: linear and epochs: 0
```

The threshold, layer selection, baselines and controls work as for any
other probe. On `keep: tokens` data each sample's tokens are mean-pooled
before the means are taken, whatever `probe.pooling` says. Mean pooling over
a chat prompt is dominated by attention-sink tokens, so for a diff-means
direction use `extract.keep: pooled` with `probe.pooling: last`. The
quickstart already uses `keep: pooled`:

```bash
sondekit run quickstart -o probe.init=diff_means -o probe.epochs=0 -o probe.pooling=last
```

`steer` uses the probe's `w` as its direction, and diff-means directions
steer well. In the [refusal direction
example](../configs/examples/refusal_direction), ablating the direction
removes most refusals and adding it induces them.

## Checklist before deploying a probe

- `small_n` is false on validation, test and every score set, so the
  threshold and FPR rest on at least ten expected false positives.
- `threshold_failed` is false.
- On test, `recall` and `fpr` at the frozen threshold are close to
  `recall_at_fpr` and `max_fpr`.
- The probe beats `baseline_auroc` and `baseline_recall_at_fpr`, and the
  confidence interval does not reach down to the baseline.
- `length_auroc` is near 0.5, or you have checked that the probe is not
  just measuring length (see [baselines](#baselines)).
- `controls_passed` is true and `control_aurocs` cluster near 0.5.
- At least one out-of-distribution score set keeps a useful `auroc` and
  `recall` at the frozen threshold.
- A benign set in the shape of real traffic gives an `fpr` you can accept.
- Serving uses the same checkpoint and chat template, which `Probe.serves`
  checks when given the served prompt format, and the same
  `data.chat_template_kwargs`, which it does not check. For Qwen3, train
  with `chat_template_kwargs: {enable_thinking: false}` and serve with it
  too.

To serve a probe that passes these checks, see [Python API](python-api.md).
[Cookbook](cookbook.md) has configs for the score sets and probe variants
mentioned on this page.
