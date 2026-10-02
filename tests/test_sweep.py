from __future__ import annotations

import json
import logging

import pytest
import torch

from sondekit import config
from sondekit import probe
from sondekit import store
from sondekit import sweep
from sondekit import train


def _fake_extract(run_dir, keep, n=120, h=8, shifts=None, keyword=False):
    ext = run_dir / "extract"
    ext.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    y = [i % 2 for i in range(n)]
    (run_dir / "rows.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": f"r{i}",
                    "text": f"{'dog' if keyword and y[i] else 'cat'} row {i}",
                    "label": y[i],
                }
            )
            + "\n"
            for i in range(n)
        )
    )
    lengths = [1 + i % 5 for i in range(n)]
    feats = {}
    for b in (0, 1, 2):
        rows = []
        for label, t in zip(y, lengths, strict=True):
            x = torch.randn(t, h, generator=g)
            x[:, 0] += (shifts or {1: 4.0}).get(b, 0.0) * label
            rows.append(x.mean(0) if keep == "pooled" else x)
        feats[b] = rows
    writer = store.ShardWriter(ext, keep, shard_size=50, shard_bytes=2**31)
    writer.add(feats)
    store.write_manifest(
        ext,
        {
            "model": "gpt2",
            "revision": None,
            "model_fingerprint": "hub:planted",
            "prompt_format": "raw",
            "engine": "hf==0+nnterp==1.3.0",
            "blocks": [0, 1, 2],
            "window": "prompt",
            "keep": keep,
            "pooling": "mean" if keep == "pooled" else None,
            "dtype": "float32",
            "ids": [f"r{i}" for i in range(n)],
            "labels": y,
            "groups": [None] * n,
            "shards": writer.close(),
            "drops": {},
            "config_hash": "x",
        },
    )


def _cfg(run_dir, keep, **probe_kw):
    return config.RunConfig(
        name="syn",
        data=config.DataConfig(
            path=str(run_dir / "rows.jsonl"), split=(0.5, 0.25, 0.25)
        ),
        extract=config.ExtractConfig(keep=keep),
        probe=config.ProbeConfig(lr=1e-2, epochs=30, batch_size=16, **probe_kw),
        steps=["train"],
    )


@pytest.mark.parametrize(
    "keep,probe_kw",
    [
        ("pooled", {}),
        ("tokens", {"kind": "attention", "pooling": "attention"}),
        ("tokens", {"pooling": "rolling_mean", "rolling_window": 2}),
        ("pooled", {"select": "auroc", "max_fpr": None}),
    ],
)
def test_selects_planted_block(tmp_path, keep, probe_kw):
    _fake_extract(tmp_path, keep)
    sweep.run(_cfg(tmp_path, keep, **probe_kw), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["block"] == 1
    assert m["test"]["auroc"] > 0.9
    assert [f.name for f in (tmp_path / "probes").iterdir()] == ["syn.npz"]
    p = probe.Probe.load(str(tmp_path / "probes" / "syn.npz"))
    assert (p.name, p.block, p.prompt_format) == ("syn", 1, "raw")
    assert p.model_fingerprint == "hub:planted"
    layers = sorted(f.name for f in (tmp_path / "layers").iterdir())
    assert layers == ["B0.npz", "B1.npz", "B2.npz"]
    assert probe.Probe.load(str(tmp_path / "layers" / "B2.npz")).name == (
        "syn@B2"
    )
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert sum(map(len, splits.values())) == 120


def test_refuses_pooling_mismatch(tmp_path):
    _fake_extract(tmp_path, "pooled")
    with pytest.raises(ValueError, match="pooling"):
        sweep.run(_cfg(tmp_path, "pooled", pooling="last"), tmp_path)


def test_headline_holds_probe_and_baseline(tmp_path, caplog):
    _fake_extract(tmp_path, "pooled")
    with caplog.at_level(logging.INFO):
        sweep.run(_cfg(tmp_path, "pooled"), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    h = m["headline"]
    assert h["max_fpr"] == 0.01
    assert 0.0 <= h["recall_at_fpr"] <= 1.0
    assert 0.0 <= h["baseline_recall_at_fpr"] <= 1.0
    low, high = h["auroc_ci"]
    assert low <= h["auroc"] <= high
    assert {"baseline_auroc", "length_auroc", "n_pos", "small_n"} <= h.keys()
    assert h["threshold"] == m["threshold"]
    assert (h["recall"], h["fpr"]) == (m["test"]["recall"], m["test"]["fpr"])
    assert caplog.records[-1].getMessage() == f"headline {json.dumps(h)}"


def test_small_n_val_falls_back_to_auroc(tmp_path):
    _fake_extract(tmp_path, "pooled")
    sweep.run(_cfg(tmp_path, "pooled"), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["val"]["1"]["small_n"] and m["select_fallback"] == "auroc"
    assert m["block"] == 1
    assert 0.0 <= m["control_auroc"] <= 1.0
    assert m["controls_passed"] == train.control_passed(m["control_aurocs"])
    sweep.run(_cfg(tmp_path, "pooled", max_fpr=0.9), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert not m["val"]["1"]["small_n"] and m["select_fallback"] is None
    assert m["block"] == 1


def test_auroc_ties_break_by_validation_loss(tmp_path):
    _fake_extract(tmp_path, "pooled", shifts={0: 6.0, 1: 20.0})
    sweep.run(_cfg(tmp_path, "pooled"), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["val"]["0"]["auroc"] == m["val"]["1"]["auroc"] == 1.0
    assert m["val"]["1"]["val_loss"] < m["val"]["0"]["val_loss"]
    assert m["block"] == 1


def test_control_averages_five_shuffles(tmp_path):
    _fake_extract(tmp_path, "pooled")
    sweep.run(_cfg(tmp_path, "pooled"), tmp_path)
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert len(m["control_aurocs"]) == 5
    assert m["control_auroc"] == pytest.approx(sum(m["control_aurocs"]) / 5)


def test_bow_baseline_learns_a_keyword_label(tmp_path):
    _fake_extract(tmp_path, "pooled", n=1000, shifts={}, keyword=True)
    cfg = _cfg(tmp_path, "pooled")
    cfg.probe = config.ProbeConfig()
    sweep.run(cfg, tmp_path)
    h = json.loads((tmp_path / "metrics.json").read_text())["headline"]
    assert h["baseline_auroc"] >= 0.95
