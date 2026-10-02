"""Labeled rows in, token ids with a window span out."""

from __future__ import annotations

import collections
import csv
import dataclasses
import json
import logging
import pathlib
from collections.abc import Sequence

import datasets
import numpy as np

from sonde import config
from sonde import fingerprint
from sonde import probe

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


@dataclasses.dataclass
class Sample:
    id: str
    text: str | None
    messages: list[dict] | None
    response: str | None
    response_ids: list[int] | None
    label: int | None
    group: str | None
    raw: dict


def _rows(src: config.DataConfig) -> list:
    if src.hf is not None:
        return list(
            datasets.load_dataset(src.hf, src.hf_config, split=src.hf_split)
        )
    path = pathlib.Path(str(src.path))
    with path.open(newline="") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if path.suffix == ".csv":
            return list(csv.DictReader(f))
    raise ValueError(f"data.path must be .jsonl or .csv, got {path}")


def _field(row: dict, key: str | None, i: int):
    if key is None:
        return None
    if key not in row:
        raise ValueError(f"row {i} has no {key!r} key; keys: {sorted(row)}")
    return row[key]


def load_samples(
    src: config.DataConfig, seed: int, require_label: bool = True
) -> list[Sample]:
    """Reads, subsamples and validates the rows of a data block.

    Args:
        src: The `data` block (or a merged `score` entry).
        seed: Drives the `limit` subsample.
        require_label: If false, a row without the label key gets label
            None (scoring unlabeled data).

    Returns:
        Samples in file order; `id` falls back to the row index.

    Raises:
        ValueError: On a missing key, a duplicate id, or a label outside
            {0, 1} after `label_map`; the message names the row.
    """
    rows = _rows(src)
    index = range(len(rows))
    if src.limit is not None and src.limit < len(rows):
        rng = np.random.default_rng(seed)
        index = sorted(rng.choice(len(rows), src.limit, replace=False))
    samples, seen = [], set()
    for i in index:
        row = rows[i]
        label = None
        if src.label is not None and (require_label or src.label in row):
            label = _field(row, src.label, i)
            if src.label_map:
                label = src.label_map.get(label, label)
            if label not in (0, 1, "0", "1"):
                raise ValueError(
                    f"row {i}: label {label!r} is not 0/1; set data.label_map"
                )
            label = int(label)
        sid = str(row.get(src.id, i) if src.id else i)
        if sid in seen:
            raise ValueError(f"row {i}: duplicate id {sid!r}")
        seen.add(sid)
        samples.append(
            Sample(
                id=sid,
                text=_field(row, src.text, i),
                messages=_field(row, src.messages, i),
                response=_field(row, src.response, i),
                response_ids=row.get("response_ids") if src.response else None,
                label=label,
                group=str(_field(row, src.group, i)) if src.group else None,
                raw=row,
            )
        )
    return samples


def split(
    labels: Sequence[int],
    groups: Sequence[str | None],
    fractions: Sequence[float],
    seed: int,
) -> dict[str, list[int]]:
    """Stratified, group-disjoint train / val / test split.

    A group is stratified by its majority label; a `None` group is its own
    unit.

    Args:
        labels: [n] in {0, 1}.
        groups: [n] group keys or None.
        fractions: (train, val, test), summing to 1.
        seed: Seeds the one generator that orders units.

    Returns:
        `{"train", "val", "test"}` -> sorted sample indices.

    Raises:
        ValueError: If any split lacks a class; the message names it.
    """
    y = np.asarray(labels)
    units = {}
    for i, g in enumerate(groups):
        units.setdefault(("i", i) if g is None else ("g", g), []).append(i)
    rng = np.random.default_rng(seed)
    out = {name: [] for name in _SPLITS}
    cuts = np.cumsum(fractions)[:2]
    for cls in (0, 1):
        members = [u for u in units.values() if round(y[u].mean()) == cls]
        total, seen = sum(map(len, members)), 0
        for j in rng.permutation(len(members)):
            k = int(np.searchsorted(cuts, seen / total, side="right"))
            out[_SPLITS[k]] += members[j]
            seen += len(members[j])
    for name, idx in out.items():
        n_pos = int(y[idx].sum())
        n_neg = len(idx) - n_pos
        logger.info(
            "split %s: n=%d (%.3f) n_pos=%d n_neg=%d",
            name,
            len(idx),
            len(idx) / len(y),
            n_pos,
            n_neg,
        )
        if not n_pos or not n_neg:
            raise ValueError(
                f"split {name!r} has n_pos={n_pos}, n_neg={n_neg}; it needs "
                "both classes. Add data or change data.split"
            )
    return {name: sorted(idx) for name, idx in out.items()}


@dataclasses.dataclass
class Encoded:
    id: str
    ids: list[int]
    span: tuple[int, int]
    label: int | None
    group: str | None


class Drop(Exception):
    """A row that cannot yield a non-empty window; the message is why."""


def _template(tokenizer) -> str:
    if tokenizer.chat_template is None:
        raise ValueError(
            f"data.format is chat but the {tokenizer.name_or_path} tokenizer "
            "has no chat_template; set data.format: raw"
        )
    return tokenizer.chat_template


def _chat_ids(tokenizer, messages: list[dict], gen_prompt: bool) -> list[int]:
    _template(tokenizer)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=gen_prompt
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def _messages(sample: Sample, src: config.DataConfig) -> list[dict]:
    if sample.messages is not None:
        return sample.messages
    system = [{"role": "system", "content": src.system}] if src.system else []
    return [*system, {"role": "user", "content": sample.text}]


def prompt_format_of(src: config.DataConfig, tokenizer) -> str:
    """The `prompt_format` mechanica serves under for this data block.

    Raises:
        ValueError: If `format: chat` and the tokenizer has no template.
    """
    if src.format == "raw":
        return fingerprint.RAW
    return fingerprint.prompt_format(_template(tokenizer))


def prompt_ids(sample: Sample, src: config.DataConfig, tokenizer) -> list[int]:
    """Prompt token ids, generation prompt included for `format: chat`."""
    if src.format == "raw":
        return tokenizer(sample.text).input_ids
    return _chat_ids(tokenizer, _messages(sample, src), True)


def render(
    sample: Sample, src: config.DataConfig, window: str, tokenizer
) -> Encoded:
    """Tokenizes one sample and locates its window.

    Args:
        sample: One loaded row.
        src: Format, system prompt and `max_length`.
        window: One of `probe.WINDOWS`.
        tokenizer: A HF tokenizer.

    Returns:
        `ids` truncated to `max_length` and `span = (start, end)` with
        `0 <= start < end <= len(ids)`.

    Raises:
        Drop: If the window is empty, or `last_turn` is not usable.
        ValueError: On `window: last_turn` with `format: raw`, or a chat
            format without a chat template.
    """
    if window == "last_turn":
        if src.format == "raw":
            raise ValueError(
                "extract.window: last_turn needs data.format: chat"
            )
        msgs = _messages(sample, src)
        if not msgs or msgs[-1]["role"] != "assistant":
            raise Drop("not_assistant_last")
        ids = _chat_ids(tokenizer, msgs, False)
        start = probe.last_turn_start(tokenizer, msgs)
        if ids[:start] != _chat_ids(tokenizer, msgs[:-1], True):
            raise Drop("last_turn_prefix_mismatch")
    else:
        ids = prompt_ids(sample, src, tokenizer)
        start = 0
        if window != "prompt":
            r = sample.response_ids or (
                tokenizer(sample.response, add_special_tokens=False).input_ids
                if sample.response
                else []
            )
            if window == "response":
                if not r:
                    raise Drop("empty_response")
                start = len(ids)
            ids = ids + r
    ids = ids[: src.max_length]
    if start >= len(ids):
        raise Drop("empty_window")
    return Encoded(
        sample.id, ids, (start, len(ids)), sample.label, sample.group
    )


def encode_all(
    samples: Sequence[Sample], src: config.DataConfig, window: str, tokenizer
) -> tuple[list[Encoded], dict[str, int]]:
    """Renders every sample, counting drops by reason.

    Returns:
        `(encoded, drops)`, encoded in sample order.

    Raises:
        ValueError: If every row is dropped.
    """
    encoded, drops = [], collections.Counter()
    for sample in samples:
        try:
            encoded.append(render(sample, src, window, tokenizer))
        except Drop as e:
            drops[str(e)] += 1
    if drops:
        logger.warning(
            "dropped %d of %d rows: %s",
            drops.total(),
            len(samples),
            dict(drops),
        )
    if not encoded:
        raise ValueError(f"every row was dropped: {dict(drops)}")
    return encoded, dict(drops)
