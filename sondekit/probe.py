"""The portable probe artifact sondekit exports and mechanica serves.

numpy and the stdlib only, so vLLM workers can import it. The artifact is
data, never code: `np.savez` out, `np.load(allow_pickle=False)` in.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

import numpy as np

FORMAT = 1
WINDOWS = frozenset({"prompt", "response", "all", "last_turn"})
POOLINGS = {
    "linear": frozenset({"mean", "last", "max", "rolling_mean"}),
    "attention": frozenset({"attention"}),
}


def last_turn_start(tokenizer, messages: list[dict]) -> int:
    """Token index where a conversation's last message starts.

    The fit-time side of the `last_turn` window; mechanica computes the same
    boundary at serve time. Renders then tokenizes, because
    `apply_chat_template(tokenize=True)` returns a dict in transformers 5.
    `tokenizer` is duck-typed so this module imports nothing.

    Args:
        tokenizer: A Hugging Face tokenizer with a chat template.
        messages: The conversation; its last message is the turn to read.

    Returns:
        len(tokens of messages[:-1] rendered with the generation prompt).
    """
    text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _vector(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"probe {name} must be a nonempty 1-D array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"probe {name} must be finite")
    return value


@dataclasses.dataclass
class Probe:
    """A trained probe: `sigmoid(pool(acts @ w + bias))` at one block.

    Attributes:
        name: Probe name; the key in `load_dir`.
        kind: "linear" or "attention".
        w: [H] float32 score (linear) or value (attention) direction.
        bias: Added to the pooled logit.
        block: Reads the residual stream output by decoder block `block`.
        window: Token window read, one of WINDOWS.
        pooling: mean | last | max | rolling_mean (linear); "attention".
        threshold: Flag threshold on the `pooled_score` scale, in [0, 1].
        model: Model name the probe was fitted on.
        engine: "<backend>==<ver>[+<lib>==<ver-or-sha>]".
        model_fingerprint: `fingerprint.checkpoint_fingerprint` at fit time.
        prompt_format: `fingerprint.prompt_format(...)` or "raw".
        q: [H] float32 attention query; None for linear probes.
        rolling_window: Window length iff pooling == "rolling_mean".
        adapters: LoRA adapters the probe was validated on.
        escalate_threshold: mechanica's review band; sondekit never sets it.
        metrics: Eval card of measured values.
    """

    name: str
    kind: str
    w: np.ndarray
    bias: float
    block: int
    window: str
    pooling: str
    threshold: float
    model: str
    engine: str
    model_fingerprint: str | None
    prompt_format: str | None
    q: np.ndarray | None = None
    rolling_window: int | None = None
    adapters: tuple[str, ...] = ()
    escalate_threshold: float | None = None
    metrics: dict | None = None

    def __post_init__(self) -> None:
        self.w = _vector(self.w, "w")
        self.bias = float(self.bias)
        if not np.isfinite(self.bias):
            raise ValueError("probe bias must be finite")
        self.threshold = float(self.threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("probe threshold must be finite and in [0, 1]")
        if type(self.block) is not int or self.block < 0:
            raise ValueError("probe block must be a nonnegative integer")
        for attr in ("name", "model", "engine"):
            if not _nonempty_str(getattr(self, attr)):
                raise ValueError(f"probe {attr} must be a nonempty string")
        for attr in ("model_fingerprint", "prompt_format"):
            value = getattr(self, attr)
            if value is not None and not _nonempty_str(value):
                raise ValueError(f"probe {attr} must be nonempty or None")
        if self.window not in WINDOWS:
            raise ValueError(f"probe window must be one of {sorted(WINDOWS)}")
        if self.kind not in POOLINGS:
            raise ValueError(f"probe kind must be one of {sorted(POOLINGS)}")
        if (self.q is None) != (self.kind == "linear"):
            raise ValueError("probe q is required iff kind is attention")
        if self.q is not None:
            self.q = _vector(self.q, "q")
            if self.q.shape != self.w.shape:
                raise ValueError("probe q and w must have the same shape")
        if self.pooling not in POOLINGS[self.kind]:
            raise ValueError(
                f"{self.kind} probe pooling must be one of "
                f"{sorted(POOLINGS[self.kind])}"
            )
        if self.pooling == "rolling_mean":
            if type(self.rolling_window) is not int or self.rolling_window < 1:
                raise ValueError("rolling_mean needs rolling_window >= 1")
        elif self.rolling_window is not None:
            raise ValueError("rolling_window is set only for rolling_mean")
        self.adapters = tuple(sorted(self.adapters))
        if not all(_nonempty_str(adapter) for adapter in self.adapters):
            raise ValueError("probe adapters must be nonempty strings")
        if self.escalate_threshold is not None:
            self.escalate_threshold = float(self.escalate_threshold)
            if not 0.0 <= self.escalate_threshold <= self.threshold:
                raise ValueError(
                    "probe escalate_threshold must be between 0 and threshold"
                )

    def save(self, path: str) -> str:
        """Writes the probe atomically.

        Args:
            path: Target path; ".npz" is appended if missing.

        Returns:
            The path written.
        """
        meta = {k: v for k, v in vars(self).items() if k not in ("w", "q")}
        meta["format"] = FORMAT
        arrays = {"w": self.w}
        if self.q is not None:
            arrays["q"] = self.q
        final = path if path.endswith(".npz") else f"{path}.npz"
        tmp = f"{final}.tmp"
        with open(tmp, "wb") as handle:
            np.savez(
                handle, allow_pickle=False, meta=json.dumps(meta), **arrays
            )
        os.replace(tmp, final)
        return final

    @classmethod
    def load(cls, path: str) -> Probe:
        """Reads a probe written by `save`.

        Args:
            path: An `.npz` probe artifact.

        Returns:
            The validated probe.

        Raises:
            ValueError: On an unknown format or meta that does not build a
                Probe, such as a legacy mechanica artifact.
        """
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            arrays = {k: archive[k] for k in ("w", "q") if k in archive}
        if meta.pop("format", None) != FORMAT:
            raise ValueError(f"{path}: not a format-{FORMAT} sondekit probe")
        try:
            return cls(**meta, **arrays)
        except TypeError as e:
            raise ValueError(f"{path}: not a sondekit probe ({e})") from e


def load_dir(directory: str) -> dict[str, Probe]:
    """Loads every `*.npz` in a directory (not recursive), in filename order.

    Args:
        directory: A probe directory.

    Returns:
        Probes keyed by name.

    Raises:
        ValueError: On a duplicate probe name or an empty directory.
    """
    probes: dict[str, Probe] = {}
    for path in sorted(pathlib.Path(directory).glob("*.npz")):
        probe = Probe.load(str(path))
        if probe.name in probes:
            raise ValueError(f"duplicate probe {probe.name!r} in {directory}")
        probes[probe.name] = probe
    if not probes:
        raise ValueError(f"no probes in {directory}")
    return probes
