from __future__ import annotations

import json

import torch

from sonde import backends
from sonde import config
from sonde import data
from sonde import generate

BLOCK = 5
GREEDY = config.GenerateConfig(max_tokens=8, temperature=0.0)
TEXTS = ["The cat sat on the", "Hello, my name is", "In 1999 the"]


def _gpt2() -> tuple[backends.Loaded, list[list[int]], torch.Tensor]:
    loaded = backends.load_model(
        config.ModelConfig(name="gpt2", dtype="float32")
    )
    tok = loaded.tokenizer
    v = torch.randn(loaded.hidden, generator=torch.Generator().manual_seed(0))
    return loaded, [tok(t).input_ids for t in TEXTS], v / v.norm()


def _write_rows(path, rows) -> None:
    path.write_text("".join(f"{json.dumps(r)}\n" for r in rows))


def _read_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _rows() -> list[dict]:
    return [
        {"id": k, "text": t, "label": i % 2}
        for i, (k, t) in enumerate(zip("abc", TEXTS, strict=True))
    ]


def _generate_cfg(tmp_path) -> config.RunConfig:
    _write_rows(tmp_path / "in.jsonl", _rows())
    return config.RunConfig.model_validate(
        {
            "name": "g",
            "model": {"name": "gpt2", "dtype": "float32"},
            "data": {"path": str(tmp_path / "in.jsonl"), "format": "raw"},
            "generate": {"max_tokens": 4, "temperature": 0.0},
            "output": {"dir": str(tmp_path)},
            "steps": ["generate"],
        }
    )


def test_hf_strength_zero_equals_unsteered():
    loaded, prompts, v = _gpt2()
    plain = backends.generate(loaded, prompts, GREEDY, 0)
    zero = backends.Steering(BLOCK, v, "add", 0.0)
    assert backends.generate(loaded, prompts, GREEDY, 0, zero) == plain
    assert [len(r) for r in plain] == [8, 8, 8]


def test_hf_batched_equals_one_by_one():
    loaded, prompts, _ = _gpt2()
    batched = backends.generate(loaded, prompts, GREEDY, 0)
    single = [backends.generate(loaded, [p], GREEDY, 0)[0] for p in prompts]
    assert single == batched


def test_hf_add_changes_output_and_does_not_leak():
    loaded, prompts, v = _gpt2()
    plain = backends.generate(loaded, prompts, GREEDY, 0)
    add = backends.Steering(BLOCK, v, "add", 100.0)
    assert backends.generate(loaded, prompts, GREEDY, 0, add) != plain
    assert backends.generate(loaded, prompts, GREEDY, 0) == plain


def test_hf_ablate_removes_projection_at_every_step():
    loaded, prompts, v = _gpt2()
    seen = []

    def grab(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        seen.append((h.float() @ v.to(h.device)).abs().max().item())

    nxt = loaded.model.layers[BLOCK + 1]._module
    hook = nxt.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        backends.generate(loaded, prompts, GREEDY, 0)
        plain = list(seen)
        seen.clear()
        ablate = backends.Steering(BLOCK, v, "ablate", 1.0)
        backends.generate(loaded, prompts, GREEDY, 0, ablate)
    finally:
        hook.remove()
    assert len(seen) == len(plain) == 8
    assert min(plain) > 1.0
    assert max(seen) < 1e-3


def test_hf_sampling_is_seeded():
    loaded, prompts, _ = _gpt2()
    hot = config.GenerateConfig(max_tokens=8, temperature=1.0, top_p=0.95)
    first = backends.generate(loaded, prompts, hot, 3)
    assert backends.generate(loaded, prompts, hot, 3) == first
    assert backends.generate(loaded, prompts, hot, 4) != first


def test_generate_resume_skips_done_ids(tmp_path):
    cfg = _generate_cfg(tmp_path)
    out = tmp_path / "generations.jsonl"
    _write_rows(out, [{"id": "a", "response": "SENTINEL"}])
    generate.run(cfg, tmp_path)
    got = _read_rows(out)
    assert [r["id"] for r in got] == ["a", "b", "c"]
    assert got[0]["response"] == "SENTINEL"
    assert got[1]["text"] == TEXTS[1] and got[1]["label"] == 1
    generate.run(cfg, tmp_path)
    assert _read_rows(out) == got


def test_generations_reload_with_exact_response_ids(tmp_path):
    cfg = _generate_cfg(tmp_path)
    generate.run(cfg, tmp_path)
    rows = _read_rows(tmp_path / "generations.jsonl")
    src = config.DataConfig(
        path=str(tmp_path / "generations.jsonl"),
        response="response",
        format="raw",
    )
    tok = _gpt2()[0].tokenizer
    for row, s in zip(rows, data.load_samples(src, 0), strict=True):
        assert s.response_ids == row["response_ids"]
        enc = data.render(s, src, "response", tok)
        p = data.prompt_ids(s, src, tok)
        assert enc.ids == p + row["response_ids"]
        assert enc.span == (len(p), len(enc.ids))
