from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from sonde import cli
from sonde import config
from sonde import fingerprint
from sonde import probe
from sonde import score

QUICKSTART = config.RECIPES_DIR / "data" / "quickstart.jsonl"


def test_quickstart_extract_train_score(tmp_path):
    assert cli.main(["run", "quickstart", "-o", f"output.dir={tmp_path}"]) == 0
    run = tmp_path / "quickstart"
    assert (run / "config.yaml").is_file()
    assert list((run / "extract").glob("shard_*.safetensors"))
    p = probe.Probe.load(str(run / "probes" / "quickstart.npz"))
    assert p.prompt_format == fingerprint.RAW and p.backend() == "hf"
    m = json.loads((run / "metrics.json").read_text())
    assert len(m["val"]) == 12 and m["test"]["auroc"] > 0.9
    held = run / "score" / "heldout"
    assert len((held / "scores.jsonl").read_text().splitlines()) == 32
    hm = json.loads((held / "metrics.json").read_text())
    assert (hm["n_pos"], hm["n_neg"]) == (16, 16) and hm["auroc"] > 0.9


def test_dry_run_prints_config_and_creates_nothing(tmp_path, capsys):
    argv = ["run", "quickstart", "--dry-run", "-o", f"output.dir={tmp_path}"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "name: quickstart" in out and "disk estimate" in out
    assert not (tmp_path / "quickstart").exists()


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


def test_score_benign_only_reports_fpr_at_frozen_threshold(tmp_path):
    rows = QUICKSTART.read_text().splitlines()
    benign = tmp_path / "benign.jsonl"
    benign.write_text("".join(f"{r}\n" for r in rows if '"label": 0' in r))
    cfg = _planted(tmp_path, metrics={"max_fpr": 0.01})
    cfg.score[0].path = str(benign)
    score.run(cfg, tmp_path / "refuse")
    out = tmp_path / "refuse" / "score" / "s" / "metrics.json"
    m = json.loads(out.read_text())
    assert (m["n_pos"], m["n_neg"], m["threshold"]) == (0, 32, 0.5)
    assert m["fpr"] == 1.0 and m["recall"] is None
    assert m["auroc"] is None and m["recall_at_fpr"] is None
    assert m["small_n"]


def test_score_unlabeled_set_writes_scores_without_metrics(tmp_path):
    rows = [json.loads(r) for r in QUICKSTART.read_text().splitlines()[:6]]
    unlabeled = tmp_path / "unlabeled.jsonl"
    unlabeled.write_text(
        "".join(
            json.dumps({"id": r["id"], "text": r["text"]}) + "\n" for r in rows
        )
    )
    cfg = _planted(tmp_path, metrics={"max_fpr": 0.01})
    cfg.score[0].path = str(unlabeled)
    score.run(cfg, tmp_path / "refuse")
    out = tmp_path / "refuse" / "score" / "s"
    assert len((out / "scores.jsonl").read_text().splitlines()) == 6
    assert not (out / "metrics.json").exists()
