"""Run configuration: one pydantic model per YAML section, checked at load."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Literal

import pydantic
import yaml

RECIPES_DIR = pathlib.Path(__file__).parent / "recipes"

_NEEDS = {
    "extract": ("model", "data", "extract"),
    "train": ("data", "probe"),
    "score": ("model", "data", "score"),
    "generate": ("model", "data", "generate"),
    "steer": ("model", "data", "generate", "steer"),
}


class _Section(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


class ModelConfig(_Section):
    name: str
    revision: str | None = None
    dtype: str = "bfloat16"
    backend: Literal["hf", "vllm"] = "hf"
    vllm: dict = pydantic.Field(default_factory=dict)


class DataConfig(_Section):
    path: str | None = None
    hf: str | None = None
    hf_config: str | None = None
    hf_split: str = "train"
    text: str | None = "text"
    messages: str | None = None
    response: str | None = None
    label: str | None = "label"
    id: str | None = "id"
    group: str | None = None
    label_map: dict | None = None
    limit: int | None = pydantic.Field(default=None, ge=1)
    format: Literal["chat", "raw"] = "chat"
    system: str | None = None
    chat_template_kwargs: dict = pydantic.Field(default_factory=dict)
    max_length: int = pydantic.Field(default=2048, ge=1)
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)

    @pydantic.model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.messages is not None and "text" not in self.model_fields_set:
            self.text = None
        if (self.path is None) == (self.hf is None):
            raise ValueError("data: set exactly one of path / hf")
        if (self.text is None) == (self.messages is None):
            raise ValueError("data: set exactly one of text / messages")
        if self.format == "raw" and self.messages is not None:
            raise ValueError("data.format: raw needs text rows, not messages")
        if min(self.split) < 0 or abs(sum(self.split) - 1) > 1e-6:
            raise ValueError(f"data.split {self.split} must be >= 0, sum to 1")
        return self


class Every(_Section):
    every: int = pydantic.Field(ge=1)


class ExtractConfig(_Section):
    layers: Literal["all"] | list[pydantic.NonNegativeInt] | Every = "all"
    window: Literal["prompt", "response", "all", "last_turn"] = "prompt"
    keep: Literal["pooled", "tokens"] = "pooled"
    batch_size: int = pydantic.Field(default=16, ge=1)
    shard_size: int = pydantic.Field(default=4096, ge=1)
    shard_bytes: int = pydantic.Field(default=2**31, ge=1)

    @pydantic.field_validator("layers")
    @classmethod
    def _dedupe(cls, v):
        return sorted(set(v)) if isinstance(v, list) else v


class ProbeConfig(_Section):
    kind: Literal["linear", "attention"] = "linear"
    pooling: Literal["mean", "last", "max", "rolling_mean", "attention"] = (
        "mean"
    )
    rolling_window: int | None = None
    init: Literal["random", "diff_means"] = "random"
    epochs: int = pydantic.Field(default=20, ge=0)
    lr: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = pydantic.Field(default=256, ge=1)
    patience: int = pydantic.Field(default=3, ge=0)
    max_fpr: float | None = 0.01
    select: Literal["auroc", "recall_at_fpr", "group_auroc"] = "recall_at_fpr"

    @pydantic.model_validator(mode="after")
    def _check(self) -> ProbeConfig:
        if (self.kind == "attention") != (self.pooling == "attention"):
            raise ValueError(
                "probe: kind attention goes with pooling attention, and only"
            )
        if (self.pooling == "rolling_mean") != (
            self.rolling_window is not None
        ):
            raise ValueError(
                "probe.rolling_window is set iff pooling is rolling_mean"
            )
        if self.rolling_window is not None and self.rolling_window < 1:
            raise ValueError("probe.rolling_window must be >= 1")
        if self.select == "recall_at_fpr" and self.max_fpr is None:
            raise ValueError(
                "probe.select: recall_at_fpr needs probe.max_fpr; set "
                "select: auroc to run without one"
            )
        if self.init == "diff_means" and (
            self.kind != "linear" or self.epochs != 0
        ):
            raise ValueError(
                "probe.init: diff_means needs kind: linear and epochs: 0"
            )
        return self


class ScoreEntry(_Section):
    name: str
    probe: str | None = None
    path: str | None = None
    hf: str | None = None
    hf_config: str | None = None
    hf_split: str | None = None
    text: str | None = None
    messages: str | None = None
    response: str | None = None
    label: str | None = None
    id: str | None = None
    group: str | None = None
    label_map: dict | None = None
    limit: int | None = None
    format: Literal["chat", "raw"] | None = None
    system: str | None = None
    chat_template_kwargs: dict | None = None
    max_length: int | None = None
    split: tuple[float, float, float] | None = None


class GenerateConfig(_Section):
    max_tokens: int = pydantic.Field(default=256, ge=1)
    temperature: float = pydantic.Field(default=0.7, ge=0)
    top_p: float = 1.0


class SteerConfig(_Section):
    probe: str | None = None
    mode: Literal["add", "ablate"] = "add"
    strengths: list[float] = pydantic.Field(default_factory=lambda: [0.0])


class OutputConfig(_Section):
    dir: str = "runs"
    overwrite: bool = False


class RunConfig(_Section):
    name: str
    seed: int = 0
    model: ModelConfig | None = None
    data: DataConfig | None = None
    extract: ExtractConfig | None = None
    probe: ProbeConfig | None = None
    score: list[ScoreEntry] = pydantic.Field(default_factory=list)
    generate: GenerateConfig | None = None
    steer: SteerConfig | None = None
    output: OutputConfig = pydantic.Field(default_factory=OutputConfig)
    steps: list[Literal["generate", "extract", "train", "score", "steer"]] = (
        pydantic.Field(default_factory=lambda: ["extract", "train"])
    )

    @pydantic.model_validator(mode="after")
    def _check(self) -> RunConfig:
        for step in self.steps:
            for section in _NEEDS[step]:
                if not getattr(self, section):
                    raise ValueError(
                        f"steps has {step!r}, which needs {section}"
                    )
        ext, probe = self.extract, self.probe
        if ext and ext.keep == "pooled":
            if probe is None and "extract" in self.steps:
                raise ValueError("extract.keep: pooled needs a probe section")
            if probe and (
                probe.kind != "linear" or probe.pooling not in ("mean", "last")
            ):
                raise ValueError(
                    "extract.keep: pooled needs probe.kind: linear with "
                    "probe.pooling mean or last; use extract.keep: tokens"
                )
        if (
            ext
            and ext.window == "response"
            and self.data
            and (self.data.response is None)
        ):
            raise ValueError("extract.window: response needs data.response")
        if (
            probe
            and probe.select == "group_auroc"
            and (self.data is None or self.data.group is None)
        ):
            raise ValueError("probe.select: group_auroc needs data.group")
        for entry in self.score:
            score_data(self, entry)
        return self


def apply_override(raw: dict, item: str) -> None:
    """Sets one `a.b=v` override on a raw config dict, in place.

    Args:
        raw: The YAML mapping before validation.
        item: `dotted.key=value`; the value is parsed with `yaml.safe_load`.

    Raises:
        ValueError: If `item` has no `=` or an empty key.
    """
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise ValueError(f"override {item!r} is not key=value")
    *parents, leaf = key.split(".")
    node = raw
    for part in parents:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[leaf] = yaml.safe_load(value)


def load(source: str, overrides: Sequence[str] = ()) -> RunConfig:
    """Reads and validates a run config.

    Args:
        source: A `.yaml`/`.yml` path, or the bare name of a bundled recipe.
        overrides: `a.b=v` strings applied before validation.

    Returns:
        The validated config.

    Raises:
        ValueError: On an unknown recipe name (the message lists them).
        pydantic.ValidationError: On any schema or cross-field violation.
    """
    path = pathlib.Path(source)
    bundled = path.suffix not in (".yaml", ".yml")
    if bundled:
        path = RECIPES_DIR / f"{source}.yaml"
        if not path.is_file():
            names = sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))
            raise ValueError(f"unknown recipe {source!r}; available: {names}")
    raw = yaml.safe_load(path.read_text()) or {}
    if bundled:
        # So `sondekit run quickstart` works from any cwd. A path missing from
        # RECIPES_DIR stays cwd-relative, as in user configs.
        for src in [raw.get("data") or {}, *(raw.get("score") or [])]:
            rel = src.get("path")
            if rel and (RECIPES_DIR / rel).is_file():
                src["path"] = str(RECIPES_DIR / rel)
    for item in overrides:
        apply_override(raw, item)
    return RunConfig.model_validate(raw)


def score_data(cfg: RunConfig, entry: ScoreEntry) -> DataConfig:
    """Merges a score entry over `cfg.data`.

    Setting either of `path`/`hf` (or `text`/`messages`) in the entry
    replaces both inherited keys of that pair.

    Args:
        cfg: The run config; `cfg.data` is the base.
        entry: Keys set here win.

    Returns:
        A validated data block for this entry.
    """
    over = entry.model_dump(
        exclude_unset=True, exclude={"name", "probe", "split"}
    )
    base = cfg.data.model_dump(exclude_unset=True) if cfg.data else {}
    for pair in (("path", "hf"), ("text", "messages")):
        if any(k in over for k in pair):
            base = {k: v for k, v in base.items() if k not in pair}
    return DataConfig.model_validate({**base, **over})


def resolve_blocks(
    spec: Literal["all"] | list[int] | Every, num_layers: int
) -> list[int]:
    """Expands `extract.layers` against a model's depth.

    Args:
        spec: `"all"`, a list of block indices, or `Every(every=k)`.
        num_layers: Decoder blocks in the model.

    Returns:
        Sorted, deduplicated block indices.

    Raises:
        ValueError: If any index is outside `[0, num_layers)`.
    """
    if spec == "all":
        return list(range(num_layers))
    if isinstance(spec, Every):
        return list(range(0, num_layers, spec.every))
    bad = [b for b in spec if b >= num_layers]
    if bad:
        raise ValueError(
            f"extract.layers {bad} out of range: model has {num_layers} blocks"
        )
    return sorted(spec)


def run_dir(cfg: RunConfig) -> pathlib.Path:
    """Returns `<output.dir>/<name>`."""
    return pathlib.Path(cfg.output.dir) / cfg.name
