"""The extract step: labeled data to activation shards."""

from __future__ import annotations

import logging
import pathlib

import torch

from sondekit import backends
from sondekit import config
from sondekit import data
from sondekit import fingerprint
from sondekit import store

logger = logging.getLogger(__name__)


def extract_hash(cfg: config.RunConfig) -> str:
    """Hash of everything that changes what extraction writes."""
    sections = {
        k: getattr(cfg, k).model_dump(mode="json")
        for k in ("model", "data", "extract")
    }
    if sections["extract"]["keep"] == "pooled" and cfg.probe:
        sections["pooling"] = cfg.probe.pooling
    if sections["data"]["limit"] is not None:
        sections["seed"] = cfg.seed
    return store.config_hash(sections)


def preflight(cfg: config.RunConfig) -> str:
    """Checks the run's extract dir against the config; loads no model.

    Returns:
        `store.prepare`'s state: "fresh", "resume" or "done".

    Raises:
        ValueError: If the dir was written by another config.
    """
    out_dir = config.run_dir(cfg) / "extract"
    return store.prepare(out_dir, extract_hash(cfg), cfg.output.overwrite)


def extract_to(
    loaded: backends.Loaded,
    encoded: list[data.Encoded],
    blocks: list[int],
    keep: str,
    pooling: str | None,
    out_dir: pathlib.Path,
    ex: config.ExtractConfig,
    meta: dict,
) -> None:
    """Runs the forward passes and writes shards, then the manifest.

    Resumes after the samples already in complete shards. `meta` holds the
    manifest keys this function cannot know (model, engine, drops, ...).
    """
    writer = store.ShardWriter(out_dir, keep, ex.shard_size, ex.shard_bytes)
    # NOTE: batches follow sample order, so each pads to its own longest
    # row; length-sort within a shard if padding waste ever shows up.
    for i in range(writer.done, len(encoded), ex.batch_size):
        batch = encoded[i : i + ex.batch_size]
        writer.add(backends.forward(loaded, batch, blocks, keep, pooling))
    store.write_manifest(
        out_dir,
        {
            **meta,
            "blocks": blocks,
            "keep": keep,
            "pooling": pooling,
            "ids": [e.id for e in encoded],
            "labels": [e.label for e in encoded],
            "groups": [e.group for e in encoded],
            "shards": writer.close(),
        },
    )


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """The extract step: writes `run_dir/extract/`.

    Raises:
        ValueError: Before any model load, if `run_dir/extract` was written
            by another config and `output.overwrite` is off.
    """
    model, src, ext = cfg.model, cfg.data, cfg.extract
    if model is None or src is None or ext is None:
        raise ValueError("extract needs the model, data and extract sections")
    out_dir = run_dir / "extract"
    h = extract_hash(cfg)
    if store.prepare(out_dir, h, cfg.output.overwrite) == "done":
        logger.info("extract: %s is complete; skipping", out_dir)
        return
    samples = data.load_samples(src, cfg.seed)
    loaded = backends.load_model(model)
    blocks = config.resolve_blocks(ext.layers, loaded.num_layers)
    window, keep = ext.window, ext.keep
    encoded, drops = data.encode_all(samples, src, window, loaded.tokenizer)
    tokens = (
        sum(e.span[1] - e.span[0] for e in encoded)
        if keep == "tokens"
        else len(encoded)
    )
    size = tokens * loaded.hidden * len(blocks)
    size *= getattr(torch, model.dtype).itemsize
    logger.info(
        "extract: %d samples x %d blocks, ~%.2f GB",
        len(encoded),
        len(blocks),
        size / 1e9,
    )
    meta = {
        "model": model.name,
        "revision": model.revision,
        "model_fingerprint": fingerprint.checkpoint_fingerprint(
            model.name, revision=model.revision
        ),
        "prompt_format": data.prompt_format_of(src, loaded.tokenizer),
        "engine": loaded.engine,
        "window": window,
        "dtype": model.dtype,
        "drops": drops,
        "config_hash": h,
    }
    extract_to(
        loaded,
        encoded,
        blocks,
        keep,
        cfg.probe.pooling if cfg.probe and keep == "pooled" else None,
        out_dir,
        ext,
        meta,
    )
