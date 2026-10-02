from __future__ import annotations

import pathlib

from sonde import config

QUICKSTART = config.RECIPES_DIR / "data" / "quickstart.jsonl"


def test_quickstart_data_resolves_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config.load("quickstart")
    paths = [cfg.data.path if cfg.data else None, cfg.score[0].path]
    assert all(p is not None and pathlib.Path(p).is_file() for p in paths)
