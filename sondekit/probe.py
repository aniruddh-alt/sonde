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


def _sigmoid(z) -> np.ndarray:
    # float64: probe logits reach +-40, and float32 rounds sigmoid(17) to 1.0,
    # tying the most confident rows at the threshold.
    z = np.clip(np.asarray(z, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _pool(
    logits: np.ndarray, pooling: str, rolling_window: int | None
) -> float:
    """Reduces per-token logits [T] to one logit."""
    if pooling == "mean":
        return float(logits.mean())
    if pooling == "last":
        return float(logits[-1])
    if pooling == "max":
        return float(logits.max())
    if rolling_window is None or logits.size < rolling_window:
        return float(logits.mean())
    windows = np.lib.stride_tricks.sliding_window_view(logits, rolling_window)
    return float(windows.mean(axis=1).max())


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

    def logits(self, acts) -> np.ndarray:
        """Per-token logits of a linear probe.

        Args:
            acts: [T, H] or [H] activations at `block`.

        Returns:
            [T] float32 `acts @ w + bias`.

        Raises:
            ValueError: For an attention probe, which has no per-token logit.
        """
        if self.kind != "linear":
            raise ValueError("logits() is defined for linear probes only")
        acts = np.atleast_2d(np.asarray(acts, dtype=np.float32))
        return acts @ self.w + self.bias

    def pooled_score_from_logits(self, logits) -> float:
        """Pools logits that already include the bias, then applies sigmoid.

        The path mechanica's instream scorer uses, on `logits()` output.

        Args:
            logits: [T] per-token logits.

        Returns:
            The probe score in [0, 1].

        Raises:
            ValueError: For an attention probe.
        """
        if self.kind != "linear":
            raise ValueError("attention probes pool activations, not logits")
        logits = np.atleast_1d(np.asarray(logits, dtype=np.float32))
        return float(_sigmoid(_pool(logits, self.pooling, self.rolling_window)))

    def pooled_logit(self, acts) -> float:
        """The pooled logit, before the sigmoid.

        RL rewards use this (r' = r - lambda * logit) because the sigmoid
        saturates.

        Args:
            acts: [T, H] activations over the window, or one [H] row.

        Returns:
            The window's logit; pooling acts on logits.
        """
        if self.q is None:
            return _pool(self.logits(acts), self.pooling, self.rolling_window)
        acts = np.atleast_2d(np.asarray(acts, dtype=np.float32))
        attn = acts @ self.q
        attn = np.exp(attn - attn.max())
        return float(attn @ (acts @ self.w) / attn.sum() + self.bias)

    def pooled_score(self, acts) -> float:
        """The score the threshold was calibrated on.

        Args:
            acts: [T, H] activations over the window, or one [H] row.

        Returns:
            sigmoid(pooled_logit(acts)), in [0, 1].
        """
        return float(_sigmoid(self.pooled_logit(acts)))

    def flag(self, acts) -> bool:
        """Whether `pooled_score(acts) >= threshold`."""
        return self.pooled_score(acts) >= self.threshold

    def escalates(self, score: float) -> bool:
        """Whether a score is in the review band: uncertain, not blocked."""
        if self.escalate_threshold is None:
            return False
        return self.escalate_threshold <= score < self.threshold

    def serves(
        self,
        fingerprint: str | None,
        adapter: str | None,
        prompt_format: str | None = "unchecked",
    ) -> str | None:
        """Why this probe must not score the running model, or None if it may.

        Fail-closed: a probe or server that cannot state its checkpoint or
        prompt format is refused.

        Args:
            fingerprint: Serving checkpoint fingerprint.
            adapter: The request's LoRA adapter, or None for the base model.
            prompt_format: Served prompt format; "unchecked" skips the check.

        Returns:
            A refusal reason, or None when the probe may serve.
        """
        if self.model_fingerprint is None:
            return "probe carries no checkpoint fingerprint; retrain it"
        if fingerprint is None:
            return "serving checkpoint could not be fingerprinted"
        if fingerprint != self.model_fingerprint:
            return (
                f"checkpoint mismatch (probe={self.model_fingerprint}, "
                f"serving={fingerprint}): fine-tuning invalidates a probe"
            )
        if adapter is not None and adapter not in self.adapters:
            return f"adapter {adapter!r} is not one this probe was validated on"
        if prompt_format == "unchecked":
            return None
        if self.prompt_format is None:
            return "probe does not record its prompt format; retrain it"
        if prompt_format is None:
            return "served prompt format could not be determined"
        if prompt_format != self.prompt_format:
            return (
                f"prompt format mismatch (probe={self.prompt_format}, "
                f"serving={prompt_format})"
            )
        return None

    def vllm_aux_layer(self) -> int:
        """The vLLM `extract_hidden_states` aux id that reads `block`."""
        return self.block + 1

    def backend(self) -> str:
        """The backend token of `engine`, e.g. "vllm" or "hf"."""
        return self.engine.split("==", 1)[0]

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
