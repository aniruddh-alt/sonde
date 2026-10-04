"""The backend seam: load a model once, prefill token ids, pool a window."""

from __future__ import annotations

import dataclasses
import json
from importlib import metadata
from typing import Any

import nnsight
import torch
import transformers

from sonde import config
from sonde import data


@dataclasses.dataclass
class Loaded:
    backend: str
    model: Any
    tokenizer: Any
    num_layers: int
    hidden: int
    engine: str


_cache: dict[str, Loaded] = {}


def load_model(cfg: config.ModelConfig) -> Loaded:
    """Loads the model, reusing the last one loaded for an identical config.

    Args:
        cfg: The `model` section.

    Returns:
        The loaded model, tokenizer and shape facts.

    Raises:
        ValueError: For `backend: vllm`, a decoder layer outside the
            allowlist.
    """
    key = cfg.model_dump_json()
    if key not in _cache:
        _cache.clear()
        load = _load_vllm if cfg.backend == "vllm" else _load_hf
        _cache[key] = load(cfg)
    return _cache[key]


def _load_hf(cfg: config.ModelConfig) -> Loaded:
    import nnterp

    model = nnterp.StandardizedTransformer(
        cfg.name,
        revision=cfg.revision,
        dtype=getattr(torch, cfg.dtype),
        dispatch=True,
    )
    engine = "hf=={}+nnterp=={}".format(
        metadata.version("transformers"), metadata.version("nnterp")
    )
    return Loaded(
        "hf",
        model,
        model.tokenizer,
        model.num_layers,
        model.hidden_size,
        engine,
    )


_VLLM_LAYERS = frozenset(
    {
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "Qwen3DecoderLayer",
    }
)


def _nnsight_ref() -> str:
    dist = metadata.distribution("nnsight")
    url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = url.get("vcs_info", {}).get("commit_id")
    return commit[:7] if commit else dist.version


def _load_vllm(cfg: config.ModelConfig) -> Loaded:
    """Builds an eager nnsight VLLM engine for an allowlisted architecture.

    The meta tree is checked before dispatch, so a refused model never
    allocates GPU memory.
    """
    import vllm  # pyright: ignore[reportMissingImports]
    from nnsight.modeling import vllm as nnsight_vllm

    model = nnsight_vllm.VLLM(
        cfg.name,
        revision=cfg.revision,
        dtype=cfg.dtype,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        **cfg.vllm,
    )
    root = model._module
    layers = getattr(getattr(root, "model", None), "layers", ())
    kinds = {type(layer).__name__ for layer in layers}
    if not kinds or not kinds <= _VLLM_LAYERS:
        found = sorted(kinds) or type(root).__name__
        raise ValueError(
            f"{cfg.name}: vLLM decoder layers {found} are not in "
            f"{sorted(_VLLM_LAYERS)}, whose output is (mlp_out, residual)"
        )
    model.dispatch()
    hf = transformers.AutoConfig.from_pretrained(
        cfg.name, revision=cfg.revision
    )
    return Loaded(
        backend="vllm",
        model=model,
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            cfg.name, revision=cfg.revision
        ),
        num_layers=hf.num_hidden_layers,
        hidden=hf.hidden_size,
        engine=f"vllm=={vllm.__version__}+nnsight=={_nnsight_ref()}",
    )


def window_pool(
    h: torch.Tensor, span: tuple[int, int], keep: str, pooling: str | None
) -> torch.Tensor:
    """Cuts one row's window out of a block output.

    Args:
        h: [T, H] block output for one row.
        span: `(start, end)` token window, `start < end <= T`.
        keep: `"pooled"` or `"tokens"`.
        pooling: `"mean"` or `"last"` when `keep == "pooled"`.

    Returns:
        [H] if pooled, else [end - start, H]; same dtype as `h`.
    """
    start, end = span
    if keep == "tokens":
        return h[start:end]
    if pooling == "last":
        return h[end - 1]
    if pooling == "mean":
        return h[start:end].mean(0)
    raise ValueError(f"keep: pooled needs pooling mean or last, not {pooling}")


def forward(
    loaded: Loaded,
    batch: list[data.Encoded],
    blocks: list[int],
    keep: str,
    pooling: str | None,
) -> dict[int, list[torch.Tensor]]:
    """One prefill of a batch, read at each block.

    Rows are right-padded with an attention mask, so a row's values do not
    depend on its batch-mates.

    Args:
        loaded: From `load_model`.
        batch: Rows to run, in sample order.
        blocks: Ascending block indices.
        keep: `"pooled"` or `"tokens"`.
        pooling: `"mean"` or `"last"` when pooled.

    Returns:
        block -> one CPU tensor per row in batch order: [H] if pooled, else
        [T_window, H], in the model's dtype.

    Raises:
        NotImplementedError: For the vllm backend.
    """
    if loaded.backend != "hf":
        raise NotImplementedError(
            f"forward on {loaded.backend} is not implemented yet; use "
            "model.backend: hf"
        )
    model = loaded.model
    inputs = loaded.tokenizer.pad(
        {"input_ids": [e.ids for e in batch]},
        padding_side="right",
        return_tensors="pt",
    )
    out = {b: [] for b in blocks}
    with torch.no_grad(), model.trace(inputs):
        for b in blocks:
            h = model.layers_output[b]
            for i, e in enumerate(batch):
                row = window_pool(h[i], e.span, keep, pooling)
                out[b].append(nnsight.save(row.to("cpu", copy=True)))
    return out


@dataclasses.dataclass
class Steering:
    """A residual-stream edit at one block, applied at every forward pass.

    Attributes:
        block: Decoder block whose output is edited (spec §5 numbering).
        vector: [H] unit-norm direction v.
        mode: "add" (x += s·v) or "ablate" (x -= s·(x·v)·v).
        strength: The multiple s.
    """

    block: int
    vector: torch.Tensor
    mode: str
    strength: float


def generate(
    loaded: Loaded,
    prompts: list[list[int]],
    gen: config.GenerateConfig,
    seed: int,
    steering: Steering | None = None,
) -> list[list[int]]:
    """Generates one response per prompt.

    Args:
        loaded: The model from `load_model`.
        prompts: Token ids per prompt, generation prompt included.
        gen: Sampling settings; temperature 0 is greedy.
        seed: Sampler seed for this call.
        steering: Edit applied at every forward pass (prefill and each
            decode step), or None for plain generation.

    Returns:
        Generated token ids per prompt, in prompt order, up to and including
        the first EOS.
    """
    if loaded.backend != "hf":
        raise NotImplementedError(f"generate on {loaded.backend}")
    model = loaded.model
    inputs = loaded.tokenizer.pad(
        {"input_ids": prompts}, padding_side="left", return_tensors="pt"
    )
    n = inputs["input_ids"].shape[1]
    sampling = gen.temperature > 0
    knobs = {"temperature": gen.temperature, "top_p": gen.top_p, "top_k": 0}
    # ponytail: transformers fills fields left unset here from the
    # checkpoint's generation_config (e.g. repetition_penalty); pin one here
    # if a recipe model ships it.
    gc = transformers.GenerationConfig(
        do_sample=sampling,
        max_new_tokens=gen.max_tokens,
        **(knobs if sampling else {}),
    )
    torch.manual_seed(seed)
    with torch.no_grad(), model.generate(generation_config=gc) as tracer:
        with tracer.invoke(inputs):
            if steering is not None:
                b, s = steering.block, steering.strength
                for _ in tracer.all():
                    h = model.layers_output[b]
                    v = steering.vector.to(h.device, h.dtype)
                    if steering.mode == "add":
                        model.layers_output[b] = h + s * v
                    else:
                        model.layers_output[b] = h - s * (h @ v)[..., None] * v
        with tracer.invoke():
            out = nnsight.save(tracer.result)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    responses = []
    for row in out[:, n:].tolist():
        end = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
        responses.append(row[:end])
    return responses
