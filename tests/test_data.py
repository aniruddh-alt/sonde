from __future__ import annotations

import json
import pathlib

import datasets
import pytest

from sonde import config
from sonde import data


def _jsonl(tmp_path: pathlib.Path, rows: list[dict]) -> str:
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_load_jsonl_label_map_and_ids(tmp_path):
    path = _jsonl(
        tmp_path,
        [
            {"q": "a", "y": "yes", "uid": "u1"},
            {"q": "b", "y": "no"},
        ],
    )
    src = config.DataConfig(
        path=path, text="q", label="y", id="uid", label_map={"yes": 1, "no": 0}
    )
    s = data.load_samples(src, seed=0)
    assert [(x.id, x.text, x.label) for x in s] == [
        ("u1", "a", 1),
        ("1", "b", 0),
    ]


def test_load_csv_labels_are_strings(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("text,label\nhi,1\nyo,0\n")
    s = data.load_samples(config.DataConfig(path=str(path)), seed=0)
    assert [x.label for x in s] == [1, 0]


def test_bad_label_names_the_row(tmp_path):
    path = _jsonl(
        tmp_path, [{"text": "a", "label": 0}, {"text": "b", "label": "maybe"}]
    )
    with pytest.raises(ValueError, match="row 1: label 'maybe'"):
        data.load_samples(config.DataConfig(path=path), seed=0)
    path = _jsonl(tmp_path, [{"text": "a"}])
    with pytest.raises(ValueError, match="row 0 has no 'label' key"):
        data.load_samples(config.DataConfig(path=path), seed=0)


def test_duplicate_ids_are_refused(tmp_path):
    rows = [
        {"id": "x", "text": "a", "label": 0},
        {"id": "y", "text": "b", "label": 1},
        {"id": "x", "text": "c", "label": 1},
    ]
    with pytest.raises(ValueError, match="row 2: duplicate id 'x'"):
        data.load_samples(config.DataConfig(path=_jsonl(tmp_path, rows)), 0)


def test_limit_is_a_seeded_subsample(tmp_path):
    path = _jsonl(
        tmp_path, [{"text": str(i), "label": i % 2} for i in range(20)]
    )
    src = config.DataConfig(path=path, limit=5)
    a = [x.id for x in data.load_samples(src, seed=1)]
    assert a == [x.id for x in data.load_samples(src, seed=1)]
    assert len(a) == 5 and a == sorted(a, key=int)
    assert a != [x.id for x in data.load_samples(src, seed=2)]


def test_load_hf_dataset(monkeypatch):
    calls = []

    def fake(name, cfg_name, split):
        calls.append((name, cfg_name, split))
        return [{"inputs": "x", "labels": "high-stakes", "ids": 9}]

    monkeypatch.setattr(datasets, "load_dataset", fake)
    src = config.DataConfig(
        hf="org/ds",
        hf_config="training",
        text="inputs",
        label="labels",
        id="ids",
        label_map={"high-stakes": 1, "low-stakes": 0},
    )
    s = data.load_samples(src, seed=0)
    assert calls == [("org/ds", "training", "train")]
    assert (s[0].id, s[0].label) == ("9", 1)


def test_split_is_stratified_and_group_disjoint():
    labels = [i % 2 for i in range(200)]
    groups = [f"g{i % 50}" for i in range(200)]
    parts = data.split(labels, groups, (0.6, 0.2, 0.2), seed=0)
    assert sorted(i for idx in parts.values() for i in idx) == list(range(200))
    where = {}
    for name, idx in parts.items():
        assert {labels[i] for i in idx} == {0, 1}
        for i in idx:
            assert where.setdefault(groups[i], name) == name
    assert abs(len(parts["train"]) / 200 - 0.6) < 0.05
    assert abs(sum(labels[i] for i in parts["train"]) / 100 - 0.6) < 0.05
    assert parts == data.split(labels, groups, (0.6, 0.2, 0.2), seed=0)


def test_split_missing_class_names_the_split():
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="split 'val' has n_pos=0"):
        data.split(labels, [None] * 9, (0.7, 0.15, 0.15), seed=0)
