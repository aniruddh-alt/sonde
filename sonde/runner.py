"""Step registry and run order."""

from __future__ import annotations

import pathlib

import yaml

from sonde import config
from sonde import extract
from sonde import score
from sonde import sweep

STEPS = {
    "extract": extract.run,
    "train": sweep.run,
    "score": score.run,
}


def run(cfg: config.RunConfig) -> pathlib.Path:
    """Runs cfg.steps in order, writing the resolved config.yaml first.

    The extract preflight runs before anything is written, so a stale
    extract/ refuses before any model loads.

    Returns:
        The run directory.

    Raises:
        ValueError: a step is not available, or extract/ holds a different
            config (and output.overwrite is false).
    """
    missing = [s for s in cfg.steps if s not in STEPS]
    if missing:
        raise ValueError(f"steps not available yet: {missing}")
    if "extract" in cfg.steps:
        extract.preflight(cfg)
    out = config.run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    )
    for step in cfg.steps:
        STEPS[step](cfg, out)
    return out
