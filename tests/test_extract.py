from __future__ import annotations

import json
import pathlib

import pytest
import torch

from sondekit import backends
from sondekit import config
from sondekit import data
from sondekit import extract
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


def _cfg(tmp_path: pathlib.Path, **extract_kw) -> config.RunConfig:
    rows = [
        {"text": t, "label": i % 2, "r": " ok" * (i + 1)}
        for i, t in enumerate(TEXTS)
    ]
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return config.RunConfig.model_validate(
        {
            "name": "t",
            "model": {"name": "gpt2", "dtype": "float32"},
            "data": {
                "path": str(path),
                "format": "raw",
                "response": "r",
                "max_length": 12,
            },
            "extract": {"layers": [0, 5, 11], "window": "all", **extract_kw},
            "probe": {},
            "output": {"dir": str(tmp_path / "runs")},
            "steps": ["extract"],
        }
    )


def _run(cfg: config.RunConfig) -> pathlib.Path:
    extract.run(cfg, config.run_dir(cfg))
    return config.run_dir(cfg) / "extract"


def _no_load(monkeypatch) -> None:
    def boom(_):
        raise AssertionError("model loaded")

    monkeypatch.setattr(backends, "load_model", boom)


def test_shard_and_manifest_layout(tmp_path):
    cfg = _cfg(tmp_path, keep="pooled", batch_size=2, shard_size=2)
    out = _run(cfg)
    manifest = store.read_manifest(out)
    shards = [f"shard_{k:05d}.safetensors" for k in range(3)]
    assert manifest["shards"] == shards
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [*shards, "manifest.json", "run.json"]
    )
    assert manifest["format"] == 1
    assert manifest["ids"] == ["0", "1", "2", "3", "4"]
    assert manifest["labels"] == [0, 1, 0, 1, 0]
    assert manifest["groups"] == [None] * 5
    assert manifest["blocks"] == [0, 5, 11]
    assert (manifest["keep"], manifest["pooling"]) == ("pooled", "mean")
    assert (manifest["window"], manifest["dtype"]) == ("all", "float32")
    assert manifest["prompt_format"] == "raw"
    assert (manifest["model"], manifest["revision"]) == ("gpt2", None)
    assert manifest["engine"].startswith("hf==")
    assert manifest["model_fingerprint"].startswith("hub:")
    assert manifest["config_hash"] == extract.extract_hash(cfg)
    assert manifest["drops"] == {}
    x, offsets = store.read_layer(out, 5)
    assert x.shape == (5, 768) and x.dtype == torch.float32
    assert offsets is None


def test_tokens_offsets_match_spans(tmp_path):
    out = _run(_cfg(tmp_path, keep="tokens", batch_size=2, shard_size=2))
    x, offsets = store.read_layer(out, 0)
    tok = backends.load_model(
        config.ModelConfig(name="gpt2", dtype="float32")
    ).tokenizer
    want = [min(len(tok(t).input_ids) + i + 1, 12) for i, t in enumerate(TEXTS)]
    assert offsets[0] == 0 and offsets[-1] == len(x)
    assert (offsets[1:] - offsets[:-1]).tolist() == want


def test_window_cut_by_max_length_is_dropped_not_empty(tmp_path):
    out = _run(_cfg(tmp_path, keep="tokens", window="response"))
    manifest = store.read_manifest(out)
    assert manifest["drops"] == {"empty_window": 2}
    assert manifest["ids"] == ["1", "2", "3"]
    x, offsets = store.read_layer(out, 11)
    assert (offsets[1:] > offsets[:-1]).all() and torch.isfinite(x).all()


def test_resume_skips_complete_shards(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, keep="pooled", batch_size=2, shard_size=2)
    out = _run(cfg)
    before = store.read_layer(out, 11)[0]
    kept = (out / "shard_00000.safetensors").read_bytes()
    for name in (
        "manifest.json",
        "shard_00001.safetensors",
        "shard_00002.safetensors",
    ):
        (out / name).unlink()
    seen = []
    forward = backends.forward

    def counting(loaded, batch, *args):
        seen.extend(e.id for e in batch)
        return forward(loaded, batch, *args)

    monkeypatch.setattr(backends, "forward", counting)
    _run(cfg)
    assert seen == ["2", "3", "4"]
    assert (out / "shard_00000.safetensors").read_bytes() == kept
    assert torch.allclose(store.read_layer(out, 11)[0], before, atol=1e-4)


def test_finished_extract_never_loads_the_model(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, keep="pooled")
    _run(cfg)
    _no_load(monkeypatch)
    _run(cfg)


def test_edited_config_refuses_before_model_load(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, keep="pooled")
    assert extract.preflight(cfg) == "fresh"
    _no_load(monkeypatch)
    edited = cfg.model_copy(deep=True)
    edited.data.max_length = 64
    with pytest.raises(ValueError, match="another config"):
        extract.preflight(edited)
    with pytest.raises(ValueError, match="another config"):
        _run(edited)
    edited.output.overwrite = True
    assert extract.preflight(edited) == "fresh"


def test_hash_covers_pooling_and_limit_seed(tmp_path):
    cfg = _cfg(tmp_path, keep="pooled")
    other = cfg.model_copy(deep=True)
    other.probe.pooling = "last"
    assert extract.extract_hash(other) != extract.extract_hash(cfg)
    other = cfg.model_copy(deep=True)
    other.seed = 1
    assert extract.extract_hash(other) == extract.extract_hash(cfg)
    other.data.limit = 3
    limited = other.model_copy(deep=True)
    limited.seed = 2
    assert extract.extract_hash(limited) != extract.extract_hash(other)


def test_vllm_forward_is_refused_until_the_follow_up():
    loaded = backends.Loaded("vllm", object(), None, 1, 1, "vllm==0.30.0")
    row = data.Encoded("a", [1, 2], (0, 2), 1, None)
    with pytest.raises(NotImplementedError, match="vllm"):
        backends.forward(loaded, [row], [0], "pooled", "mean")


def test_forward_stops_before_the_lm_head():
    loaded = backends.load_model(
        config.ModelConfig(name="gpt2", dtype="float32")
    )
    calls = []
    hook = loaded.model.lm_head._module.register_forward_hook(
        lambda *a: calls.append(1)
    )
    try:
        row = data.Encoded("a", [464, 3797, 3332], (0, 3), 1, None)
        out = backends.forward(loaded, [row], [0, 2], "pooled", "mean")
    finally:
        hook.remove()
    assert calls == []
    assert out[2][0].shape == (loaded.hidden,)
