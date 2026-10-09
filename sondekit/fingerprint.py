"""Identify the checkpoint and prompt format a probe was trained on.

`Probe.model` is a *name*. A LoRA-tuned or continued-pretrained model serves under the same name, so
a name match says nothing about whether the weights the probe reads are the ones it was fitted on —
and fine-tuning is the axis that actually breaks probes (published: 43–54% big-drop rate for
LoRA/QLoRA, 57% for privacy probes, against ~2% for quantization).

Stdlib only: this is imported on the CPU-only SDK path.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

RAW = "raw"          # extraction applied no chat template — valid only for completion deployments

# Weight containers worth hashing a manifest of, plus configs, because a fine-tune can change
# `config.json` (e.g. rope scaling) without changing any weight file name.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
_CONFIG_NAMES = ("config.json", "generation_config.json", "adapter_config.json")


def prompt_format(template: str | None, chat_template_kwargs: dict | None = None) -> str:
    """Digest the chat template a probe was extracted under, or `RAW` when there was none.

    This is the axis that fails silently and totally: a probe fitted on raw text scores raw benign
    prompts 0.002 and the same text templated 0.999999, while its eval card still reads FPR 0.000.
    A digest is enough — the template either matches what extraction used or the score means nothing.

    `chat_template_kwargs` (e.g. Qwen3's `enable_thinking`) change the rendered prompt, so non-empty
    kwargs are part of the digest; empty kwargs give the template-only digest.
    """
    if template is None:
        return RAW
    if chat_template_kwargs:
        template += "\0" + json.dumps(chat_template_kwargs, sort_keys=True)
    return "tmpl:" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:32]


def artifact_digest(path: str | pathlib.Path) -> str:
    """Digest of a probe artifact's bytes — which build of a probe produced a score.

    Stamped on every score row and read back by `mechanica calibrate`, so two builds of a
    same-named probe cannot be pooled into one threshold. Truncated identically on both sides;
    that is the reason this lives here rather than being spelled out at each call site.
    """
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def _local_manifest(root: pathlib.Path) -> list[list]:
    """Sorted (name, size) for every weight and config file — cheap, and a fine-tune changes it.

    # ponytail: not a content hash — sha256 over multi-GB checkpoints at startup is too slow to be
    # non-invasive. Name plus size catches retraining and re-quantization, not weights edited in
    # place at identical size; add a canary tensor read if that case ever matters.
    """
    return [[str(path.relative_to(root)), path.stat().st_size]
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and (path.suffix in _WEIGHT_SUFFIXES or path.name in _CONFIG_NAMES)]


def normalize_revision(value) -> str | None:
    """The revision an operator *asked for*, out of whatever the engine hands back.

    vLLM 0.29 hands back HF's `ResolvedRevision`, which carries it on `.initial`; the guard in
    `checkpoint_fingerprint` says why digesting the object itself is silently wrong.
    """
    return getattr(value, "initial", value)


def checkpoint_fingerprint(
    model: str, revision: str | None = None, quantization: str | None = None
) -> str:
    """Short digest of what determines the weights a probe will read.

    Local directories fingerprint by file manifest, which is the case that matters — a fine-tuned
    checkpoint is almost always a local path. Hub ids fall back to name plus revision, which cannot
    tell that a repo was re-uploaded under `revision=None`. `mode` is part of the digest input so the
    two can never collide.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    # Both go into `json.dumps` below, which serializes a `str` *subclass* as its string value — so
    # an engine object here is digested as whatever it happens to stringify to, silently, for every
    # probe. That is how vLLM 0.29 unbound every probe from its own server: HF's `ResolvedRevision`
    # (`huggingface_hub/_revision.py`) subclasses `str` with the value `initial or "main"` and keeps
    # the commit hash on `.resolved`, so training digested `{"revision": null}` while serving
    # digested `{"revision": "main"}` — two valid-looking strings, two different digests.
    # `type(...) is not str`, not `isinstance`: a str subclass is precisely the failure here, and
    # isinstance is the one check it passes. Every caller passes a plain str or None.
    for field, value in (("revision", revision), ("quantization", quantization)):
        if value is not None and type(value) is not str:
            raise ValueError(
                f"{field} must be an exact str or None, got {type(value).__name__}; a str "
                "subclass digests as its string value and would silently shift every digest")
    root = pathlib.Path(model)
    if root.is_dir():
        payload = {"mode": "local", "manifest": _local_manifest(root)}
        if not payload["manifest"]:
            raise ValueError(f"no weight or config files found under {model}")
    else:
        payload = {"mode": "hub", "model": model, "revision": revision}
    payload["quantization"] = quantization
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{payload['mode']}:{digest[:32]}"
