from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from sonde import config
from sonde import fingerprint
from sonde import probe
from sonde import score

QUICKSTART = config.RECIPES_DIR / "data" / "quickstart.jsonl"


def test_quickstart_data_resolves_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config.load("quickstart")
    paths = [cfg.data.path if cfg.data else None, cfg.score[0].path]
    assert all(p is not None and pathlib.Path(p).is_file() for p in paths)


def _planted(tmp_path, **override):
    good = probe.Probe(
        name="planted",
        kind="linear",
        w=np.zeros(768, np.float32),
        bias=0.0,
        block=3,
        window="prompt",
        pooling="mean",
        threshold=0.5,
        model="gpt2",
        engine="hf==5.1.0+nnterp==1.3.0",
        model_fingerprint=fingerprint.checkpoint_fingerprint("gpt2"),
        prompt_format=fingerprint.RAW,
    )
    path = dataclasses.replace(good, **override).save(
        str(tmp_path / "planted.npz")
    )
    return config.RunConfig(
        name="refuse",
        model=config.ModelConfig(name="gpt2", dtype="float32"),
        data=config.DataConfig(path=str(QUICKSTART), format="raw"),
        score=[config.ScoreEntry(name="s", probe=path)],
        output=config.OutputConfig(dir=str(tmp_path)),
        steps=["score"],
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("prompt_format", f"tmpl:{'0' * 32}", "prompt_format"),
        ("model_fingerprint", f"hub:{'0' * 32}", "model_fingerprint"),
        ("engine", "vllm==0.30.0+nnsight==b717807", "backend"),
    ],
)
def test_score_refuses_mismatched_probe(tmp_path, field, value, match):
    cfg = _planted(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=match):
        score.run(cfg, tmp_path / "refuse")
    assert not (tmp_path / "refuse" / "score" / "s" / "scores.jsonl").exists()


def test_score_accepts_matching_probe(tmp_path):
    score.run(_planted(tmp_path), tmp_path / "refuse")
    rows = (tmp_path / "refuse" / "score" / "s" / "scores.jsonl").read_text()
    assert len(rows.splitlines()) == 64
    assert json.loads(rows.splitlines()[0]) == {
        "id": "quickstart-0",
        "score": 0.5,
        "flag": True,
    }
