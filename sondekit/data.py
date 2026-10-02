"""Labeled rows in, token ids with a window span out."""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib
from collections.abc import Sequence

import datasets
import numpy as np

from sondekit import config

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
    if path.suffix not in (".jsonl", ".csv"):
        raise ValueError(f"data.path must be .jsonl or .csv, got {path}")
    with path.open(newline="") as f:
        if path.suffix == ".csv":
            rows = list(csv.DictReader(f))
        else:
            rows = []
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: not valid JSON: {e}") from e
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


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
