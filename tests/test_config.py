from __future__ import annotations

import pathlib

import pydantic
import pytest
import yaml

from sonde import config

BASE = {
    "name": "t",
    "model": {"name": "gpt2"},
    "data": {"path": "d.jsonl"},
    "extract": {},
    "probe": {},
}


def _cfg(**sections) -> config.RunConfig:
    raw = dict(BASE)
    for key, value in sections.items():
        if isinstance(value, dict) and key in BASE:
            value = {**BASE[key], **value}
        raw[key] = value
    return config.RunConfig.model_validate(raw)


def _bad(match: str, **sections) -> None:
    with pytest.raises(pydantic.ValidationError, match=match):
        _cfg(**sections)


def test_defaults_validate():
    cfg = _cfg()
    assert cfg.steps == ["extract", "train"]
    assert cfg.extract.layers == "all"
    assert cfg.output.dir == "runs"


def test_unknown_key_error_names_its_path():
    _bad(r"data\.pathh", data={"pathh": "x"})
    _bad(r"probe\.kindd", probe={"kindd": "linear"})


def test_step_needs_its_sections():
    _bad("needs model", model=None)
    _bad("needs score", steps=["score"])
    _bad("needs generate", steps=["generate"])
    _bad("needs steer", steps=["steer"], generate={})
    _bad("needs probe", probe=None)
    _bad("pooled needs a probe", probe=None, steps=["extract"])
    cfg = _cfg(probe=None, extract={"keep": "tokens"}, steps=["extract"])
    assert cfg.probe is None


def test_data_source_and_text_checks():
    _bad("exactly one of path / hf", data={"hf": "org/ds"})
    _bad("exactly one of path / hf", data={"path": None})
    _bad("exactly one of text / messages", data={"text": None})
    _bad("raw needs text rows", data={"messages": "m", "format": "raw"})
    cfg = _cfg(data={"messages": "conv"})
    assert cfg.data.text is None and cfg.data.messages == "conv"
    _bad("exactly one of text / messages", data={"text": "t", "messages": "m"})


def test_split_must_sum_to_one():
    _bad("sum to 1", data={"split": [0.5, 0.2, 0.2]})
    _bad("sum to 1", data={"split": [1.2, -0.1, -0.1]})


def test_layers_forms():
    assert _cfg(extract={"layers": [5, 1, 5]}).extract.layers == [1, 5]
    every = _cfg(extract={"layers": {"every": 4}}).extract.layers
    assert isinstance(every, config.Every) and every.every == 4
    _bad("layers", extract={"layers": [-1]})
    _bad("layers", extract={"layers": {"every": 0}})
    _bad("layers", extract={"layers": "some"})


def test_probe_cross_field_checks():
    tokens = {"keep": "tokens"}
    _bad("pooled needs probe.kind: linear", probe={"pooling": "max"})
    _bad(
        "pooled needs probe.kind: linear",
        probe={"kind": "attention", "pooling": "attention"},
    )
    _bad(
        "rolling_window is set iff",
        extract=tokens,
        probe={"pooling": "rolling_mean"},
    )
    _bad("rolling_window is set iff", probe={"rolling_window": 3})
    _bad(
        "rolling_window must be >= 1",
        extract=tokens,
        probe={"pooling": "rolling_mean", "rolling_window": 0},
    )
    _bad(
        "kind attention goes with pooling attention",
        extract=tokens,
        probe={"kind": "attention"},
    )
    _bad("needs probe.max_fpr", probe={"max_fpr": None})
    _bad("needs data.group", probe={"select": "group_auroc"})
    grouped = _cfg(data={"group": "g"}, probe={"select": "group_auroc"})
    assert grouped.probe.select == "group_auroc"
    no_fpr = _cfg(probe={"select": "auroc", "max_fpr": None})
    assert no_fpr.probe.max_fpr is None
    _bad("diff_means needs", probe={"init": "diff_means"})
    _bad(
        "diff_means needs",
        extract=tokens,
        probe={
            "init": "diff_means",
            "epochs": 0,
            "kind": "attention",
            "pooling": "attention",
        },
    )
    ok = _cfg(
        extract=tokens, probe={"pooling": "rolling_mean", "rolling_window": 4}
    )
    assert ok.probe.rolling_window == 4
    assert _cfg(probe={"init": "diff_means", "epochs": 0}).probe.epochs == 0


def test_window_response_needs_data_response():
    _bad("needs data.response", extract={"window": "response"})
    cfg = _cfg(extract={"window": "response"}, data={"response": "r"})
    assert cfg.data.response == "r"


def test_temperature_non_negative():
    _bad("temperature", generate={"temperature": -0.1})


def test_load_applies_overrides(tmp_path: pathlib.Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(BASE))
    cfg = config.load(
        str(path),
        [
            "extract.layers=[3, 1]",
            "seed=7",
            "generate.temperature=0",
            "steps=[extract]",
        ],
    )
    assert cfg.extract.layers == [1, 3]
    assert cfg.seed == 7
    assert cfg.generate.temperature == 0.0
    assert cfg.steps == ["extract"]


def test_apply_override_parses_yaml_and_rejects_garbage():
    raw = {"data": None}
    config.apply_override(raw, "data.label_map={a: 1, b: 0}")
    assert raw == {"data": {"label_map": {"a": 1, "b": 0}}}
    config.apply_override(raw, "data.hf=")
    assert raw["data"]["hf"] is None
    with pytest.raises(ValueError, match="key=value"):
        config.apply_override(raw, "seed")


def test_recipe_lookup(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setattr(config, "RECIPES_DIR", tmp_path)
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(BASE))
    assert config.load("tiny").name == "t"
    with pytest.raises(ValueError, match=r"unknown recipe 'nope'.*\['tiny'\]"):
        config.load("nope")


def test_score_data_inherits_from_data():
    cfg = _cfg(
        data={"label_map": {"y": 1, "n": 0}, "max_length": 64},
        score=[
            {
                "name": "ood",
                "hf": "org/ood",
                "hf_split": "test",
                "messages": "conv",
                "split": [1, 0, 0],
            }
        ],
    )
    merged = config.score_data(cfg, cfg.score[0])
    assert merged.path is None and merged.hf == "org/ood"
    assert merged.text is None and merged.messages == "conv"
    assert merged.hf_split == "test"
    assert merged.label_map == {"y": 1, "n": 0} and merged.max_length == 64
    assert merged.split == cfg.data.split


def test_resolve_blocks():
    assert config.resolve_blocks("all", 3) == [0, 1, 2]
    assert config.resolve_blocks(config.Every(every=4), 10) == [0, 4, 8]
    assert config.resolve_blocks([2, 0], 3) == [0, 2]
    with pytest.raises(ValueError, match=r"\[12\] out of range"):
        config.resolve_blocks([0, 12], 12)


def test_run_dir():
    cfg = _cfg(output={"dir": "out"})
    assert config.run_dir(cfg) == pathlib.Path("out") / "t"
