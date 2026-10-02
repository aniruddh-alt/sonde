from __future__ import annotations

import json
import pathlib

import datasets
import pytest
import transformers

from sonde import config
from sonde import data
from sonde import fingerprint


def _jsonl(tmp_path: pathlib.Path, rows: list[dict]) -> str:
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_load_jsonl_label_map_and_ids(tmp_path):
    path = _jsonl(
        tmp_path,
        [
            {"q": "a", "y": "yes", "uid": "u1"},
            {"q": "b", "y": "no"},
        ],
    )
    src = config.DataConfig(
        path=path, text="q", label="y", id="uid", label_map={"yes": 1, "no": 0}
    )
    s = data.load_samples(src, seed=0)
    assert [(x.id, x.text, x.label) for x in s] == [
        ("u1", "a", 1),
        ("1", "b", 0),
    ]


def test_load_csv_labels_are_strings(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("text,label\nhi,1\nyo,0\n")
    s = data.load_samples(config.DataConfig(path=str(path)), seed=0)
    assert [x.label for x in s] == [1, 0]


def test_bad_label_names_the_row(tmp_path):
    path = _jsonl(
        tmp_path, [{"text": "a", "label": 0}, {"text": "b", "label": "maybe"}]
    )
    with pytest.raises(ValueError, match="row 1: label 'maybe'"):
        data.load_samples(config.DataConfig(path=path), seed=0)
    path = _jsonl(tmp_path, [{"text": "a"}])
    with pytest.raises(ValueError, match="row 0 has no 'label' key"):
        data.load_samples(config.DataConfig(path=path), seed=0)


def test_duplicate_ids_are_refused(tmp_path):
    rows = [
        {"id": "x", "text": "a", "label": 0},
        {"id": "y", "text": "b", "label": 1},
        {"id": "x", "text": "c", "label": 1},
    ]
    with pytest.raises(ValueError, match="row 2: duplicate id 'x'"):
        data.load_samples(config.DataConfig(path=_jsonl(tmp_path, rows)), 0)


def test_limit_is_a_seeded_subsample(tmp_path):
    path = _jsonl(
        tmp_path, [{"text": str(i), "label": i % 2} for i in range(20)]
    )
    src = config.DataConfig(path=path, limit=5)
    a = [x.id for x in data.load_samples(src, seed=1)]
    assert a == [x.id for x in data.load_samples(src, seed=1)]
    assert len(a) == 5 and a == sorted(a, key=int)
    assert a != [x.id for x in data.load_samples(src, seed=2)]


def test_load_hf_dataset(monkeypatch):
    calls = []

    def fake(name, cfg_name, split):
        calls.append((name, cfg_name, split))
        return [{"inputs": "x", "labels": "high-stakes", "ids": 9}]

    monkeypatch.setattr(datasets, "load_dataset", fake)
    src = config.DataConfig(
        hf="org/ds",
        hf_config="training",
        text="inputs",
        label="labels",
        id="ids",
        label_map={"high-stakes": 1, "low-stakes": 0},
    )
    s = data.load_samples(src, seed=0)
    assert calls == [("org/ds", "training", "train")]
    assert (s[0].id, s[0].label) == ("9", 1)


def test_split_is_stratified_and_group_disjoint():
    labels = [i % 2 for i in range(200)]
    groups = [f"g{i % 50}" for i in range(200)]
    parts = data.split(labels, groups, (0.6, 0.2, 0.2), seed=0)
    assert sorted(i for idx in parts.values() for i in idx) == list(range(200))
    where = {}
    for name, idx in parts.items():
        assert {labels[i] for i in idx} == {0, 1}
        for i in idx:
            assert where.setdefault(groups[i], name) == name
    assert abs(len(parts["train"]) / 200 - 0.6) < 0.05
    assert abs(sum(labels[i] for i in parts["train"]) / 100 - 0.6) < 0.05
    assert parts == data.split(labels, groups, (0.6, 0.2, 0.2), seed=0)


def test_split_missing_class_names_the_split():
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="split 'val' has n_pos=0"):
        data.split(labels, [None] * 9, (0.7, 0.15, 0.15), seed=0)


TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _tok(template: str | None = None):
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = template
    return tok


def _src(**kw) -> config.DataConfig:
    return config.DataConfig(path="unused.jsonl", **kw)


def _sample(**kw) -> data.Sample:
    fields = dict(
        id="0",
        text=None,
        messages=None,
        response=None,
        response_ids=None,
        label=1,
        group=None,
        raw={},
    )
    return data.Sample(**{**fields, **kw})


def test_raw_windows():
    tok = _tok()
    p = tok("Hello world").input_ids
    r = tok(" yes sir", add_special_tokens=False).input_ids
    src = _src(format="raw", response="r")
    s = _sample(text="Hello world", response=" yes sir")
    assert data.render(s, src, "prompt", tok).span == (0, len(p))
    resp = data.render(s, src, "response", tok)
    assert resp.ids == p + r and resp.span == (len(p), len(p) + len(r))
    assert data.render(s, src, "all", tok).span == (0, len(p) + len(r))
    assert data.prompt_format_of(src, tok) == fingerprint.RAW


def test_chat_windows_and_prompt_format():
    tok = _tok(TEMPLATE)
    src = _src(system="be brief")
    s = _sample(text="hi")
    enc = data.render(s, src, "prompt", tok)
    want = "<|system|>\nbe brief\n<|user|>\nhi\n<|assistant|>\n"
    assert tok.decode(enc.ids) == want
    assert enc.span == (0, len(enc.ids))
    assert data.prompt_format_of(src, tok) == fingerprint.prompt_format(
        TEMPLATE
    )
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    turn = data.render(
        _sample(messages=msgs), _src(messages="m"), "last_turn", tok
    )
    assert tok.decode(turn.ids[turn.span[0] :]) == "hello there\n"


def test_response_ids_are_reused_exactly():
    tok = _tok()
    s = _sample(text="Q", response="ignored", response_ids=[11, 12, 13])
    enc = data.render(s, _src(format="raw", response="r"), "response", tok)
    assert enc.ids[enc.span[0] :] == [11, 12, 13]


def test_chat_render_has_no_double_bos():
    tok = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    enc = data.render(_sample(text="hi"), _src(), "prompt", tok)
    assert enc.ids[0] == tok.bos_token_id
    assert enc.ids[1] != tok.bos_token_id


def test_chat_without_template_says_use_raw():
    tok = _tok()
    with pytest.raises(ValueError, match=r"set data\.format: raw"):
        data.render(_sample(text="hi"), _src(), "prompt", tok)
    with pytest.raises(ValueError, match=r"set data\.format: raw"):
        data.prompt_format_of(_src(), tok)


def test_drops_are_counted_never_empty():
    tok = _tok(TEMPLATE)
    long_prompt = _sample(text="word " * 50, response="yes")
    no_response = _sample(text="q", response="")
    ok = _sample(text="q", response="yes")
    src = _src(response="r", max_length=16)
    encoded, drops = data.encode_all(
        [long_prompt, no_response, ok], src, "response", tok
    )
    assert drops == {"empty_window": 1, "empty_response": 1}
    assert len(encoded) == 1
    start, end = encoded[0].span
    assert 0 <= start < end == len(encoded[0].ids) <= 16


def test_max_length_clips_span():
    tok = _tok()
    s = _sample(text="one two three four five six")
    enc = data.render(s, _src(format="raw", max_length=3), "prompt", tok)
    assert len(enc.ids) == 3 and enc.span == (0, 3)


def test_last_turn_drops():
    tok = _tok(TEMPLATE)
    user_last = [{"role": "user", "content": "hi"}]
    with pytest.raises(data.Drop, match="not_assistant_last"):
        data.render(
            _sample(messages=user_last), _src(messages="m"), "last_turn", tok
        )
    thinking = TEMPLATE.replace(
        "<|assistant|>\n{% endif %}", "<|assistant|>\n<think>{% endif %}"
    )
    msgs = [*user_last, {"role": "assistant", "content": "ok"}]
    with pytest.raises(data.Drop, match="last_turn_prefix_mismatch"):
        data.render(
            _sample(messages=msgs),
            _src(messages="m"),
            "last_turn",
            _tok(thinking),
        )


def test_all_rows_dropped_raises():
    with pytest.raises(ValueError, match="every row was dropped"):
        data.encode_all(
            [_sample(text="q", response="")],
            _src(format="raw", response="r"),
            "response",
            _tok(),
        )
