"""Builds the datasets the example configs read, under configs/examples/data/.

Usage, from the repo root:

    python configs/examples/prepare_data.py            # every dataset
    python configs/examples/prepare_data.py harmful    # one group

Groups:
    harmful: AdvBench harmful instructions (MIT) vs Alpaca instructions
        (CC BY-NC 4.0), a JailbreakBench out-of-distribution set (MIT), a
        benign-only Alpaca set, and 200-row harmful-only and harmless-only
        subsets for steering and generation.
    truth: Geometry of Truth city statements and their negations
        (github.com/saprmarks/geometry-of-truth).
    keyword: a synthetic control where the label is whether "dog" appears.

The sources are downloaded, not redistributed; check their licenses before
using the data for anything beyond research. Needs `datasets` (installed
with `sondekit[hf]`). Output is deterministic.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import random
import sys
import urllib.request

import datasets

OUT = pathlib.Path(__file__).parent / "data"
_ADVBENCH = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
_TRUTH = (
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/"
    "datasets/{}.csv"
)


def _csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode())))


def _write(name: str, rows: list[dict]) -> list[dict]:
    rows = [{"id": f"{name}-{i}", **r} for i, r in enumerate(rows)]
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    positives = sum(r.get("label", 0) for r in rows)
    print(f"{name}: {len(rows)} rows, {positives} positive")
    return rows


def harmful() -> None:
    rng = random.Random(0)
    adv = [r["goal"] for r in _csv(_ADVBENCH)]
    alpaca = [
        r["instruction"]
        for r in datasets.load_dataset("tatsu-lab/alpaca", split="train")
        if not r["input"]
    ]
    rows = [{"text": t, "label": 1} for t in adv]
    rows += [{"text": t, "label": 0} for t in rng.sample(alpaca, len(adv))]
    rng.shuffle(rows)
    rows = _write("harmful", rows)
    _write(
        "alpaca_benign",
        [{"text": t, "label": 0} for t in rng.sample(alpaca, 500)],
    )
    jbb = datasets.load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    _write(
        "jbb",
        [{"text": r["Goal"], "label": 1} for r in jbb["harmful"]]
        + [{"text": r["Goal"], "label": 0} for r in jbb["benign"]],
    )
    for name, label in (("harmful_only200", 1), ("harmless200", 0)):
        subset = [r for r in rows if r["label"] == label][:200]
        _write(name, [{"text": r["text"], "label": label} for r in subset])


def truth() -> None:
    for name in ("cities", "neg_cities"):
        _write(
            name,
            [
                {
                    "text": r["statement"],
                    "label": int(r["label"]),
                    "city": r["city"],
                }
                for r in _csv(_TRUTH.format(name))
            ],
        )


def keyword() -> None:
    rng = random.Random(0)
    subjects = ["dog", "cat", "bird", "horse", "fish", "rabbit"]
    verbs = [
        "ran across",
        "slept near",
        "looked at",
        "jumped over",
        "waited by",
        "played in",
    ]
    places = [
        "the park",
        "the house",
        "the river",
        "the garden",
        "the road",
        "the field",
    ]
    rows = []
    for _ in range(2000):
        s = rng.choice(subjects)
        text = f"The {s} {rng.choice(verbs)} {rng.choice(places)}."
        rows.append({"text": text, "label": int(s == "dog")})
    _write("keyword_dog", rows)


GROUPS = {"harmful": harmful, "truth": truth, "keyword": keyword}


def main(argv: list[str]) -> int:
    unknown = set(argv) - set(GROUPS)
    if unknown:
        print(f"unknown group(s) {sorted(unknown)}; choose from {list(GROUPS)}")
        return 2
    OUT.mkdir(exist_ok=True)
    for name in argv or GROUPS:
        GROUPS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
