"""Extraction on disk: run.json, safetensors shards, manifest.json."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil

import safetensors
import safetensors.torch
import torch

FORMAT = 1


def _write_json(path: pathlib.Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def config_hash(sections: dict) -> str:
    """sha256 of the JSON-dumped sections, first 16 hex chars."""
    blob = json.dumps(sections, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def prepare(out_dir: pathlib.Path, h: str, overwrite: bool) -> str:
    """Decides what an extraction into `out_dir` must do. Loads no model.

    Args:
        out_dir: The extraction directory.
        h: `config_hash` of the config that would write it.
        overwrite: Delete `out_dir` first.

    Returns:
        `"done"` (finished manifest), `"resume"` (run.json only) or
        `"fresh"` (run.json just written).

    Raises:
        ValueError: If run.json or the manifest has another hash.
    """
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    for name, state in (("manifest.json", "done"), ("run.json", "resume")):
        path = out_dir / name
        if path.exists():
            found = json.loads(path.read_text())["config_hash"]
            if found != h:
                raise ValueError(
                    f"{path} was written by another config (hash {found}, "
                    f"now {h}); set output.overwrite: true or "
                    "change name"
                )
            return state
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "run.json", {"config_hash": h})
    return "fresh"


class ShardWriter:
    """Buffers per-sample features and writes them as ordered shards.

    A shard closes once it holds `shard_size` samples or `shard_bytes`
    bytes, checked after each batch. Complete shards found in `out_dir`
    count toward `done`, so a resumed run continues at sample `done`.
    """

    def __init__(
        self,
        out_dir: pathlib.Path,
        keep: str,
        shard_size: int,
        shard_bytes: int,
    ):
        self.out_dir = out_dir
        self.keep = keep
        self.shard_size = shard_size
        self.shard_bytes = shard_bytes
        self.shards = sorted(
            p.name for p in out_dir.glob("shard_*.safetensors")
        )
        self.done = 0
        for name in self.shards:
            with safetensors.safe_open(out_dir / name, "pt") as f:
                self.done += int(f.metadata()["n"])
        self._buf: dict[int, list[torch.Tensor]] = {}
        self._n = 0
        self._bytes = 0

    def add(self, features: dict[int, list[torch.Tensor]]) -> None:
        """Appends one batch: block -> per-sample [H] or [T, H] rows."""
        for block, rows in features.items():
            self._buf.setdefault(block, []).extend(rows)
            self._bytes += sum(t.nbytes for t in rows)
        self._n += len(next(iter(features.values())))
        if self._n >= self.shard_size or self._bytes >= self.shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._n:
            return
        join = torch.stack if self.keep == "pooled" else torch.cat
        tensors = {f"B{b}": join(rows) for b, rows in self._buf.items()}
        if self.keep == "tokens":
            rows = next(iter(self._buf.values()))
            lengths = torch.tensor([0] + [len(t) for t in rows])
            tensors["offsets"] = lengths.cumsum(0)
        name = f"shard_{len(self.shards):05d}.safetensors"
        tmp = self.out_dir / f"{name}.tmp"
        safetensors.torch.save_file(tensors, tmp, {"n": str(self._n)})
        os.replace(tmp, self.out_dir / name)
        self.shards.append(name)
        self._buf, self._n, self._bytes = {}, 0, 0

    def close(self) -> list[str]:
        """Flushes the last shard; returns all shard file names in order."""
        self._flush()
        return self.shards


def write_manifest(out_dir: pathlib.Path, manifest: dict) -> None:
    """Atomically writes manifest.json with `format` added."""
    _write_json(out_dir / "manifest.json", {"format": FORMAT, **manifest})


def read_manifest(out_dir: pathlib.Path) -> dict:
    """Reads manifest.json.

    Raises:
        ValueError: On a `format` other than 1.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"{out_dir}/manifest.json has format {manifest.get('format')!r}; "
            f"this sonde reads format {FORMAT}"
        )
    return manifest


def read_layer(
    out_dir: pathlib.Path, block: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reads one block across all shards.

    Args:
        out_dir: A finished extraction directory.
        block: A block listed in the manifest.

    Returns:
        `(X, offsets)`: pooled X [n, H] with offsets None, or tokens X
        [sum T, H] with offsets [n + 1] (row i is X[offsets[i]:offsets[i+1]]).

    Raises:
        ValueError: If `block` was not extracted.
    """
    # ponytail: one tokens-mode block must fit in RAM; stream by shard if a
    # dataset outgrows it.
    manifest = read_manifest(out_dir)
    if block not in manifest["blocks"]:
        raise ValueError(f"block {block} not in {manifest['blocks']}")
    xs, offsets = [], [torch.zeros(1, dtype=torch.long)]
    for name in manifest["shards"]:
        with safetensors.safe_open(out_dir / name, "pt") as f:
            xs.append(f.get_tensor(f"B{block}"))
            if manifest["keep"] == "tokens":
                offsets.append(f.get_tensor("offsets")[1:] + offsets[-1][-1])
    if manifest["keep"] == "pooled":
        return torch.cat(xs), None
    return torch.cat(xs), torch.cat(offsets)
