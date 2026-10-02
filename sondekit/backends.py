"""The backend seam: load a model once, prefill token ids, pool a window."""

from __future__ import annotations

import dataclasses
from importlib import metadata
from typing import Any

import nnsight
import torch

from sondekit import config
from sondekit import data


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
        NotImplementedError: For `backend: vllm`, which is not implemented yet.
    """
    key = cfg.model_dump_json()
    if key not in _cache:
        if cfg.backend != "hf":
            raise NotImplementedError("the vllm backend is not implemented yet")
        _cache.clear()
        _cache[key] = _load_hf(cfg)
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
    with torch.no_grad(), model.trace(inputs) as tracer:
        for b in blocks:
            h = model.layers_output[b]
            for i, e in enumerate(batch):
                row = window_pool(h[i], e.span, keep, pooling)
                out[b].append(nnsight.save(row.to("cpu", copy=True)))
        # Skip the remaining blocks and the LM head: logits over the vocab
        # are the largest tensor of a forward pass and nothing reads them.
        tracer.stop()
    return out
