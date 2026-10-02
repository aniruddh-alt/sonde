"""The generate step: sample responses into generations.jsonl."""

from __future__ import annotations

import json
import pathlib

from sondekit import backends
from sondekit import config
from sondekit import data
from sondekit import store

# NOTE: fixed chunk; add generate.batch_size when a config needs it.
BATCH = 64


def write_rows(
    loaded: backends.Loaded,
    samples: list[data.Sample],
    src: config.DataConfig,
    gen: config.GenerateConfig,
    seed: int,
    path: pathlib.Path,
    steering: backends.Steering | None = None,
) -> None:
    """Generates for `samples` in chunks and appends one JSON row each.

    A row is the sample's source row with its id under the id key, plus
    `response` (decoded, special tokens skipped) and `response_ids` (the
    exact generated ids). Each chunk is flushed as it finishes.

    Args:
        loaded: The model from `backends.load_model`.
        samples: Rows to generate for, written in this order.
        src: The data block the samples came from (render and id key).
        gen: Sampling settings.
        seed: Sampler seed, reset for every chunk.
        path: The .jsonl file to append to; parents are created.
        steering: Edit applied while generating, or None.
    """
    tok = loaded.tokenizer
    key = src.id or "id"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for k in range(0, len(samples), BATCH):
            chunk = samples[k : k + BATCH]
            prompts = [data.prompt_ids(s, src, tok) for s in chunk]
            responses = backends.generate(loaded, prompts, gen, seed, steering)
            for s, r in zip(chunk, responses, strict=True):
                row = {
                    **s.raw,
                    key: s.id,
                    "response": tok.decode(r, skip_special_tokens=True),
                    "response_ids": r,
                }
                f.write(f"{json.dumps(row)}\n")
            f.flush()


def preflight(cfg: config.RunConfig) -> None:
    """Refuses to append to generations.jsonl made under another config.

    The resume skip-set is keyed by id only, so without this check an edited
    model, data, sampling or seed would mix two configs in one file. Loads
    no model.

    Raises:
        ValueError: generate.json holds a different hash and
            output.overwrite is false.
    """
    out = config.run_dir(cfg)
    sidecar, rows = out / "generate.json", out / "generations.jsonl"
    sections = {
        k: getattr(cfg, k).model_dump(mode="json")
        for k in ("model", "data", "generate")
    }
    want = store.config_hash({**sections, "seed": cfg.seed})
    if sidecar.exists() and json.loads(sidecar.read_text())["hash"] != want:
        if not cfg.output.overwrite:
            raise ValueError(
                f"{rows} was generated under a different config; set "
                "output.overwrite: true to regenerate it"
            )
        rows.unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"hash": want}))


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Appends a generation for every sample whose id is not in the file yet.

    Raises:
        ValueError: If the model, data or generate section is missing.
    """
    if cfg.model is None or cfg.data is None or cfg.generate is None:
        raise ValueError("generate needs the model, data and generate sections")
    path = run_dir / "generations.jsonl"
    key = cfg.data.id or "id"
    done = set()
    if path.exists():
        with path.open() as f:
            done = {json.loads(line)[key] for line in f}
    samples = data.load_samples(cfg.data, cfg.seed, require_label=False)
    todo = [s for s in samples if s.id not in done]
    if todo:
        loaded = backends.load_model(cfg.model)
        write_rows(loaded, todo, cfg.data, cfg.generate, cfg.seed, path)
