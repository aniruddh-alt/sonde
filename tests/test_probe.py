from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from sondekit import probe

H = 4
BLOCKED = (
    "torch",
    "transformers",
    "nnsight",
    "nnterp",
    "vllm",
    "pydantic",
    "yaml",
    "safetensors",
    "datasets",
)


def make(**overrides) -> probe.Probe:
    fields = {
        "name": "p",
        "kind": "linear",
        "w": [0.1, 0.2, 0.3, 0.4],
        "bias": -0.25,
        "block": 3,
        "window": "prompt",
        "pooling": "mean",
        "threshold": 0.5,
        "model": "gpt2",
        "engine": "hf==5.1.0+nnterp==1.3.0",
        "model_fingerprint": "hub:abc",
        "prompt_format": "raw",
    }
    fields.update(overrides)
    return probe.Probe(**fields)


def assert_same(a: probe.Probe, b: probe.Probe) -> None:
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype == np.float32
            assert np.array_equal(x, y), field.name
        else:
            assert x == y, field.name


def test_round_trip(tmp_path):
    attention = make(
        kind="attention",
        pooling="attention",
        q=[0.5, -0.5, 0.0, 1.0],
        adapters=["lora-b", "lora-a"],
        escalate_threshold=0.25,
        metrics={"val": {"auroc": 0.9, "n_pos": 3, "n_neg": 5}},
    )
    rolling = make(pooling="rolling_mean", rolling_window=2)
    for i, original in enumerate((attention, rolling)):
        path = original.save(str(tmp_path / f"p{i}"))
        assert path == str(tmp_path / f"p{i}.npz")
        loaded = probe.Probe.load(path)
        assert_same(original, loaded)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p0.npz", "p1.npz"]


def test_coerces_on_construction():
    p = make(w=[1, 2, 3, 4], bias=np.float64(1), adapters=["b", "a"])
    assert p.w.dtype == np.float32
    assert type(p.bias) is float
    assert p.adapters == ("a", "b")


def test_probe_and_fingerprint_import_with_numpy_only():
    code = (
        f"import sys; sys.modules.update(dict.fromkeys({BLOCKED!r})); "
        "from sondekit import probe, fingerprint"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_run_without_extras_names_the_extra():
    code = (
        f"import sys; sys.modules.update(dict.fromkeys({BLOCKED!r})); "
        "import sondekit\n"
        "try:\n    sondekit.run\nexcept ImportError as e:\n    print(e)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    ).stdout
    assert "sondekit[hf]" in out


def test_legacy_and_unknown_format_refused(tmp_path):
    legacy_meta = {
        "name": "old",
        "bias": 0.0,
        "layer": 4,
        "model": "gpt2",
        "threshold": 0.5,
        "engine": "vllm==0.29.0",
        "window": "prompt",
        "positions": "last",
    }
    for i, meta in enumerate((legacy_meta, {**legacy_meta, "format": 1})):
        path = tmp_path / f"legacy{i}.npz"
        np.savez(path, weight=np.ones(H, np.float32), meta=json.dumps(meta))
        with pytest.raises(ValueError):
            probe.Probe.load(str(path))
    path = make().save(str(tmp_path / "future"))
    with np.load(path) as archive:
        meta = json.loads(str(archive["meta"]))
        w = archive["w"]
    np.savez(path, w=w, meta=json.dumps({**meta, "format": 2}))
    with pytest.raises(ValueError):
        probe.Probe.load(path)


def test_validation_errors():
    attention = {"kind": "attention", "pooling": "attention"}
    bad = [
        {"w": []},
        {"w": [0.1, math.nan, 0.3, 0.4]},
        {"w": np.ones((2, 2))},
        {"bias": math.inf},
        {"threshold": 1.5},
        {"threshold": math.nan},
        {"block": -1},
        {"block": True},
        {"block": 1.5},
        {"name": ""},
        {"model": " "},
        {"engine": ""},
        {"model_fingerprint": ""},
        {"window": "suffix"},
        {"kind": "softmax"},
        {"q": np.ones(H)},
        attention,
        {**attention, "q": np.ones(H + 1)},
        {"kind": "attention", "pooling": "mean", "q": np.ones(H)},
        {"pooling": "attention"},
        {"pooling": "rolling_mean"},
        {"pooling": "rolling_mean", "rolling_window": 0},
        {"pooling": "mean", "rolling_window": 2},
        {"adapters": ("",)},
        {"escalate_threshold": 0.6},
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            make(**overrides)


def test_load_dir(tmp_path):
    (tmp_path / "ok").mkdir()
    make(name="b").save(str(tmp_path / "ok" / "1"))
    make(name="a").save(str(tmp_path / "ok" / "2"))
    (tmp_path / "ok" / "nested").mkdir()
    make(name="c").save(str(tmp_path / "ok" / "nested" / "3"))
    assert list(probe.load_dir(str(tmp_path / "ok"))) == ["b", "a"]

    (tmp_path / "dup").mkdir()
    make(name="x").save(str(tmp_path / "dup" / "1"))
    make(name="x").save(str(tmp_path / "dup" / "2"))
    with pytest.raises(ValueError, match="duplicate"):
        probe.load_dir(str(tmp_path / "dup"))

    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no probes"):
        probe.load_dir(str(tmp_path / "empty"))


class CharTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(m["content"] for m in messages)
        return text + "A:" if add_generation_prompt and not tokenize else text

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return {"input_ids": [0, *ids] if add_special_tokens else ids}


def test_last_turn_start_counts_rendered_prefix():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert probe.last_turn_start(CharTokenizer(), messages) == len("hiA:")
