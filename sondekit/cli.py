"""sondekit run CFG|RECIPE [-o key=value ...] [--dry-run]."""

from __future__ import annotations

import argparse
import logging
import sys

import transformers
import yaml

from sondekit import config
from sondekit import data
from sondekit import extract
from sondekit import runner


def _disk_estimate(cfg: config.RunConfig) -> int:
    """Bytes extract will write; reads the tokenizer and config.json only."""
    model, src, ex = cfg.model, cfg.data, cfg.extract
    if model is None or src is None or ex is None:
        raise ValueError("extract needs the model, data and extract sections")
    tok = transformers.AutoTokenizer.from_pretrained(
        model.name, revision=model.revision
    )
    hf = transformers.AutoConfig.from_pretrained(
        model.name, revision=model.revision
    )
    encoded, _ = data.encode_all(
        data.load_samples(src, cfg.seed), src, ex.window, tok
    )
    blocks = config.resolve_blocks(ex.layers, hf.num_hidden_layers)
    return extract.disk_bytes(
        encoded, ex.keep, hf.hidden_size, len(blocks), model.dtype
    )


def main(argv: list[str] | None = None) -> int:
    """Console entry point; returns the exit code."""
    parser = argparse.ArgumentParser(prog="sondekit")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a config or bundled recipe")
    run.add_argument("config", help="path to a .yaml, or a recipe name")
    run.add_argument(
        "-o",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="e.g. -o probe.lr=0.01",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved config and disk estimate",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        cfg = config.load(args.config, args.overrides or ())
        if args.dry_run:
            print(
                yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
                end="",
            )
            if "extract" in cfg.steps:
                print(
                    f"# extract disk estimate: "
                    f"{_disk_estimate(cfg) / 2**20:.1f} MiB"
                )
            return 0
        print(runner.run(cfg))
    # User mistakes (bad config, missing file, unreadable data, gated model)
    # get one line; anything else is a bug and keeps its traceback.
    except (ValueError, OSError, ImportError) as e:
        print(f"sondekit: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
