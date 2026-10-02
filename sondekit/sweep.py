"""The train step: split, fit every block, select, test once, control."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import re
import shutil

import numpy as np
import torch

from sondekit import config
from sondekit import data
from sondekit import probes
from sondekit import store
from sondekit import train

_log = logging.getLogger(__name__)


_CONTROL_SHUFFLES = 5


def _log_loss(y: np.ndarray, scores: np.ndarray) -> float:
    """Mean binary cross-entropy of [n] scores in [0, 1] against [n] labels."""
    s = np.clip(scores.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))


def _at(groups, idx):
    """groups restricted to idx, or None when the run has no groups."""
    return None if groups is None else [groups[i] for i in idx]


def _fit(pc, X, offsets, y, sp, seed, **meta):
    """Fits on train, thresholds on val; returns (Probe, val scores, failed)."""
    module, mu, sigma = train.fit(
        pc, X, offsets, y, sp["train"], sp["val"], seed
    )
    p = probes.export(module, mu, sigma, threshold=0.5, **meta)
    s = train.probe_scores(p, X, offsets, sp["val"])
    thr, failed = train.choose_threshold(y[sp["val"]], s, pc.max_fpr)
    return dataclasses.replace(p, threshold=thr), s, failed


def run(cfg: config.RunConfig, run_dir: pathlib.Path) -> None:
    """Writes splits.json, layers/B{b}.npz, probes/{name}.npz, metrics.json.

    Raises:
        ValueError: a sample has no label, or pooled features were pooled
            differently from cfg.probe.pooling.
    """
    pc, dc = cfg.probe, cfg.data
    if pc is None or dc is None:
        raise ValueError("train needs the data and probe sections")
    ext = run_dir / "extract"
    man = store.read_manifest(ext)
    if man["keep"] == "pooled" and man["pooling"] != pc.pooling:
        raise ValueError(
            f"extract pooled with {man['pooling']!r} but probe.pooling is "
            f"{pc.pooling!r}; re-extract or match them"
        )
    if None in man["labels"]:
        raise ValueError("train needs a label on every sample")
    y = np.asarray(man["labels"])
    groups = man["groups"] if dc.group else None
    sp = data.split(man["labels"], man["groups"], dc.split, cfg.seed)
    (run_dir / "splits.json").write_text(
        json.dumps({k: [man["ids"][i] for i in v] for k, v in sp.items()})
    )
    keys = ("window", "model", "engine", "model_fingerprint", "prompt_format")
    meta = {"name": cfg.name, **{k: man[k] for k in keys}}
    layers = run_dir / "layers"
    shutil.rmtree(layers, ignore_errors=True)
    layers.mkdir()
    fitted, table = {}, {}
    for b in man["blocks"]:
        X, offsets = store.read_layer(ext, b)
        p, s, failed = _fit(pc, X, offsets, y, sp, cfg.seed, block=b, **meta)
        table[b] = train.metrics(
            y[sp["val"]], s, p.threshold, pc.max_fpr, _at(groups, sp["val"])
        )
        table[b]["threshold_failed"] = failed
        table[b]["val_loss"] = _log_loss(y[sp["val"]], s)
        fitted[b] = dataclasses.replace(
            p, name=f"{cfg.name}@B{b}", metrics={"val": table[b]}
        )
        fitted[b].save(str(layers / f"B{b}.npz"))
        _log.info("block %d: val %s", b, table[b])
    small_val = table[man["blocks"][0]]["small_n"]
    fallback = pc.select == "recall_at_fpr" and small_val
    key = "auroc" if fallback else pc.select
    # Ties are common once AUROC saturates at 1.0; the lower validation
    # log-loss picks the block that separates with the larger margin.
    best = max(
        table,
        key=lambda b: (
            table[b][key] or 0.0,
            table[b]["auroc"] or 0.0,
            -table[b]["val_loss"],
        ),
    )
    X, offsets = store.read_layer(ext, best)
    p = fitted[best]
    te = sp["test"]
    scores = train.probe_scores(p, X, offsets, te)
    test = train.metrics(
        y[te], scores, p.threshold, pc.max_fpr, _at(groups, te)
    )
    # One shuffled-label fit can land on a dominant feature axis with a
    # random sign, so the control averages several shuffles.
    control_aurocs = []
    for k in range(_CONTROL_SHUFFLES):
        shuffled = np.random.default_rng(cfg.seed + k).permutation(y)
        _, s, _ = _fit(
            pc, X, offsets, shuffled, sp, cfg.seed, block=best, **meta
        )
        control_aurocs.append(train.auroc(y[sp["val"]], s))
    control_auroc = float(np.mean(control_aurocs))
    by_id = {s.id: s for s in data.load_samples(dc, cfg.seed)}
    texts = [train.window_text(by_id[i], man["window"]) for i in man["ids"]]
    bow = torch.from_numpy(train.bow_features(texts, sp["train"]))
    bow_pc = pc.model_copy(
        update={"kind": "linear", "pooling": "mean", "rolling_window": None}
    )
    bow_probe, _, _ = _fit(bow_pc, bow, None, y, sp, cfg.seed, block=0, **meta)
    baseline = train.metrics(
        y[te],
        train.probe_scores(bow_probe, bow, None, te),
        bow_probe.threshold,
        pc.max_fpr,
    )
    # NOTE: word count stands in for the window's token count, which
    # the manifest does not store; add per-sample lengths there if needed.
    length = np.array([len(re.findall(r"\w+", t)) for t in texts])
    max_fpr = pc.max_fpr
    headline = {
        "max_fpr": max_fpr,
        "recall_at_fpr": test.get("recall_at_fpr"),
        "recall_at_fpr_ci": None
        if max_fpr is None
        else train.bootstrap_ci(
            y[te],
            scores,
            lambda a, s: train.recall_at_fpr(a, s, max_fpr),
            cfg.seed,
        ),
        "threshold": p.threshold,
        "recall": test["recall"],
        "fpr": test["fpr"],
        "auroc": test["auroc"],
        "auroc_ci": train.bootstrap_ci(y[te], scores, train.auroc, cfg.seed),
        "baseline_recall_at_fpr": baseline.get("recall_at_fpr"),
        "baseline_auroc": baseline["auroc"],
        "length_auroc": train.auroc(y[te], length[te]),
        "n_pos": test["n_pos"],
        "n_neg": test["n_neg"],
        "small_n": test["small_n"],
    }
    if groups is not None:
        headline["group_auroc"] = test["group_auroc"]
    card = {
        "headline": headline,
        "block": best,
        "select": pc.select,
        "select_fallback": "auroc" if fallback else None,
        "max_fpr": max_fpr,
        "threshold": p.threshold,
        "threshold_failed": table[best]["threshold_failed"],
        "val": {str(b): v for b, v in table.items()},
        "test": test,
        "baseline": baseline,
        "control_auroc": control_auroc,
        "control_aurocs": control_aurocs,
        "controls_passed": train.control_passed(control_aurocs),
        "n": {k: len(v) for k, v in sp.items()},
    }
    out = run_dir / "probes"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    dataclasses.replace(p, name=cfg.name, metrics=card).save(
        str(out / f"{cfg.name}.npz")
    )
    (run_dir / "metrics.json").write_text(json.dumps(card, indent=2))
    _log.info(
        "selected block %d; test %s; control_auroc %.3f",
        best,
        test,
        control_auroc,
    )
    _log.info("headline %s", json.dumps(headline))
