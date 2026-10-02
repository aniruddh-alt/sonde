"""The score step: apply a probe to new data."""

from __future__ import annotations

import json
import logging
import pathlib
import shutil

import numpy as np

from sonde import backends
from sonde import config
from sonde import data
from sonde import extract
from sonde import fingerprint
from sonde import probe
from sonde import store
from sonde import train

_log = logging.getLogger(__name__)


def _mismatch(path: str, field: str, probe_value, run_value) -> ValueError:
    return ValueError(
        f"probe {path} has {field}={probe_value!r} but this run has "
        f"{run_value!r}; its scores would be meaningless here, so score "
        "refuses it"
    )


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Writes score/<entry>/{extract/, scores.jsonl, metrics.json}.

    metrics.json reports the deployed operating point: fpr and recall at
    the probe's frozen threshold. It also reports recall at max_fpr
    re-thresholded on this set (from the probe section, else the probe's
    own card), auroc, and group_auroc when the entry has a group column.
    A benign-only set reports fpr only, a positive-only set recall only.

    Raises:
        ValueError: the probe's backend, model_fingerprint or prompt_format
            disagrees with this run.
    """
    if cfg.model is None:
        raise ValueError("score needs the model section")
    ex = cfg.extract or config.ExtractConfig()
    fp = fingerprint.checkpoint_fingerprint(cfg.model.name, cfg.model.revision)
    for entry in cfg.score:
        path = entry.probe or str(run_dir / "probes" / f"{cfg.name}.npz")
        p = probe.Probe.load(path)
        if p.backend() != cfg.model.backend:
            raise _mismatch(path, "backend", p.backend(), cfg.model.backend)
        if p.model_fingerprint != fp:
            raise _mismatch(path, "model_fingerprint", p.model_fingerprint, fp)
        src = config.score_data(cfg, entry)
        loaded = backends.load_model(cfg.model)
        fmt = data.prompt_format_of(src, loaded.tokenizer)
        if p.prompt_format != fmt:
            raise _mismatch(path, "prompt_format", p.prompt_format, fmt)
        keep = "pooled" if p.pooling in ("mean", "last") else "tokens"
        pooling = p.pooling if keep == "pooled" else None
        out = run_dir / "score" / entry.name
        ext = out / "extract"
        shutil.rmtree(ext, ignore_errors=True)
        ext.mkdir(parents=True)
        encoded, drops = data.encode_all(
            data.load_samples(src, cfg.seed, require_label=False),
            src,
            p.window,
            loaded.tokenizer,
        )
        extract.extract_to(
            loaded,
            encoded,
            [p.block],
            keep,
            pooling,
            ext,
            ex,
            {
                "model": cfg.model.name,
                "revision": cfg.model.revision,
                "model_fingerprint": fp,
                "prompt_format": fmt,
                "engine": loaded.engine,
                "window": p.window,
                "dtype": cfg.model.dtype,
                "drops": drops,
            },
        )
        man = store.read_manifest(ext)
        X, offsets = store.read_layer(ext, p.block)
        scores = train.probe_scores(p, X, offsets, range(len(man["ids"])))
        with open(out / "scores.jsonl", "w") as f:
            for i, s in zip(man["ids"], scores, strict=True):
                row = {
                    "id": i,
                    "score": float(s),
                    "flag": bool(s >= p.threshold),
                }
                f.write(f"{json.dumps(row)}\n")
        labels = man["labels"]
        if labels and None not in labels:
            card = p.metrics or {}
            m = train.metrics(
                np.asarray(labels),
                scores,
                p.threshold,
                cfg.probe.max_fpr if cfg.probe else card.get("max_fpr"),
                man["groups"] if src.group else None,
            )
            (out / "metrics.json").write_text(json.dumps(m, indent=2))
            _log.info("score %s: %s", entry.name, m)
