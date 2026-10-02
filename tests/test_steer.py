from __future__ import annotations


import torch

from sondekit import backends
from sondekit import config

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
