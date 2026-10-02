"""sonde: train probes on LLM activations and export a portable artifact.

`sonde.probe` and `sonde.fingerprint` import only numpy and the stdlib, so
this module must not import anything heavier.
"""

from __future__ import annotations

import importlib
import typing

__version__ = "0.2.0"

_LAZY = {
    "run": ("sonde.runner", "run"),
    "load_config": ("sonde.config", "load"),
}


def __getattr__(name: str) -> typing.Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'sonde' has no attribute {name!r}")
    module, attr = _LAZY[name]
    return getattr(importlib.import_module(module), attr)
