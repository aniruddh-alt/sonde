"""The steer step: generate along a probe direction at several strengths."""

from __future__ import annotations

import pathlib

import torch

from sondekit import backends
from sondekit import config
from sondekit import data
from sondekit import generate
from sondekit import probe


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Writes steer/s{strength}.jsonl per strength; strength 0 is unedited.

    The direction is the probe's w / |w|, applied at the probe's block.

    Raises:
        ValueError: If a needed section is missing.
    """
    if (
        cfg.model is None
        or cfg.data is None
        or cfg.generate is None
        or cfg.steer is None
    ):
        raise ValueError(
            "steer needs the model, data, generate, steer sections"
        )
    path = cfg.steer.probe or str(run_dir / "probes" / f"{cfg.name}.npz")
    p = probe.Probe.load(path)
    loaded = backends.load_model(cfg.model)
    w = torch.tensor(p.w)
    v = w / w.norm()
    samples = data.load_samples(cfg.data, cfg.seed, require_label=False)
    for s in cfg.steer.strengths:
        out = run_dir / "steer" / f"s{s:g}.jsonl"
        out.unlink(missing_ok=True)
        edit = backends.Steering(p.block, v, cfg.steer.mode, s) if s else None
        generate.write_rows(
            loaded, samples, cfg.data, cfg.generate, cfg.seed, out, edit
        )
