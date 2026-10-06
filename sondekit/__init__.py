"""sondekit: train probes on LLM activations and export a portable artifact.

`sondekit.probe` and `sondekit.fingerprint` import only numpy and the stdlib, so
this module must not import anything heavier.
"""

from __future__ import annotations

import importlib
import typing

__version__ = "0.2.0"

_LAZY = {
    "run": ("sondekit.runner", "run"),
    "load_config": ("sondekit.config", "load"),
}


def __getattr__(name: str) -> typing.Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'sondekit' has no attribute {name!r}")
    module, attr = _LAZY[name]
    try:
        return getattr(importlib.import_module(module), attr)
    except ImportError as e:
        raise ImportError(
            f"sondekit.{name} needs the training extras: "
            "pip install 'sondekit[hf]' (or 'sondekit[vllm]')"
        ) from e
