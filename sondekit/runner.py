"""Step registry and run order."""

from __future__ import annotations

import pathlib

import yaml

from sondekit import config
from sondekit import extract
from sondekit import generate
from sondekit import score
from sondekit import steer
from sondekit import sweep

STEPS = {
    "extract": extract.run,
    "train": sweep.run,
    "score": score.run,
    "generate": generate.run,
    "steer": steer.run,
}


def run(cfg: config.RunConfig) -> pathlib.Path:
    """Runs cfg.steps in order, writing the resolved config.yaml first.

    Preflights run before anything is written, so a stale extract/ or
    generations.jsonl refuses before any model loads.

    Returns:
        The run directory.

    Raises:
        ValueError: extract/ or generations.jsonl holds a different
            config (and output.overwrite is false).
    """
    if "extract" in cfg.steps:
        extract.preflight(cfg)
    if "generate" in cfg.steps:
        generate.preflight(cfg)
    out = config.run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    )
    for step in cfg.steps:
        STEPS[step](cfg, out)
    return out
