from __future__ import annotations

import json

import pytest
import torch

from sondekit import backends
from sondekit import config
from sondekit import data
from sondekit import store

TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Hi",
    "Paris is the capital of",
    "One two three four five six seven",
    "A much longer sentence that will certainly be cut by max_length.",
]


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


def test_window_pool():
    h = torch.arange(12.0).reshape(4, 3)
    assert torch.equal(backends.window_pool(h, (1, 3), "tokens", None), h[1:3])
    assert torch.equal(backends.window_pool(h, (1, 3), "pooled", "last"), h[2])
    assert torch.equal(
        backends.window_pool(h, (1, 3), "pooled", "mean"), h[1:3].mean(0)
    )
    with pytest.raises(ValueError, match="mean or last"):
        backends.window_pool(h, (1, 3), "pooled", "max")


def test_vllm_backend_is_not_here_yet():
    with pytest.raises(NotImplementedError, match="not implemented"):
        backends.load_model(config.ModelConfig(name="x", backend="vllm"))


def test_load_model_is_cached():
    cfg = config.ModelConfig(name="gpt2", dtype="float32")
    loaded = backends.load_model(cfg)
    assert backends.load_model(cfg.model_copy()) is loaded
    assert (loaded.num_layers, loaded.hidden) == (12, 768)
    assert loaded.engine.startswith("hf==") and "+nnterp==" in loaded.engine


def test_gpt2_batch_invariance():
    loaded = backends.load_model(
        config.ModelConfig(name="gpt2", dtype="float32")
    )
    batch = []
    for i, text in enumerate(TEXTS[:4]):
        ids = loaded.tokenizer(text).input_ids
        batch.append(
            data.Encoded(str(i), ids, (len(ids) // 2, len(ids)), 0, None)
        )
    blocks = [0, 5, 11]
    four = backends.forward(loaded, batch, blocks, "tokens", None)
    last = backends.forward(loaded, batch, [11], "pooled", "last")
    for i, e in enumerate(batch):
        one = backends.forward(loaded, [e], blocks, "tokens", None)
        for b in blocks:
            assert four[b][i].shape == (e.span[1] - e.span[0], 768)
            assert torch.isfinite(four[b][i]).all()
            # fp32 CPU differs by ~2e-4, MPS by ~4e-3; the padding bug was
            # ~2.6e3.
            assert torch.allclose(
                one[b][0], four[b][i], atol=1e-2, rtol=1e-4
            ), (b, i)
        assert torch.equal(last[11][i], four[11][i][-1])
