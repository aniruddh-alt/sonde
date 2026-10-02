from __future__ import annotations

import json

import pytest
import torch

from sondekit import store


def test_prepare_states(tmp_path):
    out = tmp_path / "extract"
    assert store.prepare(out, "abc", overwrite=False) == "fresh"
    assert json.loads((out / "run.json").read_text()) == {"config_hash": "abc"}
    assert store.prepare(out, "abc", overwrite=False) == "resume"
    store.write_manifest(out, {"config_hash": "abc"})
    assert store.prepare(out, "abc", overwrite=False) == "done"
    with pytest.raises(ValueError, match="another config"):
        store.prepare(out, "xyz", overwrite=False)
    assert store.prepare(out, "xyz", overwrite=True) == "fresh"
    assert sorted(p.name for p in out.iterdir()) == ["run.json"]


def test_config_hash_ignores_key_order():
    assert store.config_hash({"a": 1, "b": [2]}) == store.config_hash(
        {"b": [2], "a": 1}
    )
    assert len(store.config_hash({})) == 16


def test_unknown_manifest_format_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": 2}))
    with pytest.raises(ValueError, match="format 2"):
        store.read_manifest(tmp_path)


def test_shards_round_trip_tokens(tmp_path):
    writer = store.ShardWriter(
        tmp_path, "tokens", shard_size=2, shard_bytes=2**31
    )
    rows = [torch.full((n, 3), float(n)) for n in (1, 2, 3)]
    for row in rows:
        writer.add({4: [row], 7: [row * 2]})
    shards = writer.close()
    assert shards == ["shard_00000.safetensors", "shard_00001.safetensors"]
    store.write_manifest(
        tmp_path, {"shards": shards, "blocks": [4, 7], "keep": "tokens"}
    )
    x, offsets = store.read_layer(tmp_path, 7)
    assert offsets.tolist() == [0, 1, 3, 6]
    assert torch.equal(x, torch.cat(rows) * 2)
    assert store.ShardWriter(tmp_path, "tokens", 2, 2**31).done == 3
    with pytest.raises(ValueError, match="block 5"):
        store.read_layer(tmp_path, 5)


def test_shard_bytes_closes_shards(tmp_path):
    writer = store.ShardWriter(
        tmp_path, "pooled", shard_size=100, shard_bytes=1
    )
    for _ in range(3):
        writer.add({0: [torch.ones(4)]})
    assert writer.close() == [f"shard_{k:05d}.safetensors" for k in range(3)]
    assert store.ShardWriter(tmp_path, "pooled", 100, 1).done == 3
