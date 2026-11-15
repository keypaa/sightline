"""Typed configuration for a Sightline run.

The dataclasses are the single source of defaults. A run resolves them into a
plain JSON-compatible dict (see :func:`resolve_config`) and writes that dict
into its run record. Values that depend on the checkpoint are left ``None``
until the model is loaded; they are never guessed here.
"""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ALLOWED_SEEDS = (0, 1, 2)
ALLOWED_DTYPES = ("bfloat16", "float16", "float32")
ALLOWED_ATTENTION = ("sdpa", "flash_attention_2")
ALLOWED_PROJECTOR_INIT = ("random", "embedding_alignment")


class ConfigError(ValueError):
    """Raised when a configuration value is invalid or unknown."""


def _check_dtype(name: str, value: str) -> None:
    if value not in ALLOWED_DTYPES:
        raise ConfigError(f"{name} must be one of {ALLOWED_DTYPES}, got {value!r}")


@dataclass(frozen=True)
class PathsConfig:
    artifacts: Path = Path("artifacts")
    data: Path = Path("data")
    experiments: Path = Path("experiments")
    reports: Path = Path("reports")


@dataclass(frozen=True)
class ReaderConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    revision: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    dtype: str = "bfloat16"
    attn_backend: str = "sdpa"
    eval_mode: bool = True

    def __post_init__(self) -> None:
        _check_dtype("reader.dtype", self.dtype)
        if self.attn_backend not in ALLOWED_ATTENTION:
            raise ConfigError(
                f"reader.attn_backend must be one of {ALLOWED_ATTENTION}, got {self.attn_backend!r}"
            )


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str = "moonshotai/Kimi-K3"
    revision: str = "f831ab66814297da540d832a5235f8e904f29d06"
    dtype: str = "bfloat16"
    canvas: int = 448

    def __post_init__(self) -> None:
        _check_dtype("encoder.dtype", self.dtype)
        if self.canvas <= 0:
            raise ConfigError(f"encoder.canvas must be positive, got {self.canvas}")


@dataclass(frozen=True)
class ProjectorConfig:
    # Width is derived from the reader's hidden size at load time. A non-None
    # value is an explicit override, not a default.
    hidden_width: int | None = None
    dtype: str = "float32"
    init: str = "random"

    def __post_init__(self) -> None:
        _check_dtype("projector.dtype", self.dtype)
        if self.init not in ALLOWED_PROJECTOR_INIT:
            raise ConfigError(
                f"projector.init must be one of {ALLOWED_PROJECTOR_INIT}, got {self.init!r}"
            )
        if self.hidden_width is not None and self.hidden_width <= 0:
            raise ConfigError(f"projector.hidden_width must be positive, got {self.hidden_width}")


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    global_batch: int = 128
    micro_batch: int = 8
    lr: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    max_steps: int = 2048

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ConfigError("training.seeds must not be empty")
        for seed in self.seeds:
            if seed not in ALLOWED_SEEDS:
                raise ConfigError(f"seed {seed} is not allowed; use {ALLOWED_SEEDS} (never 42)")
        if self.micro_batch < 1:
            raise ConfigError(f"training.micro_batch must be >= 1, got {self.micro_batch}")
        if self.global_batch < self.micro_batch:
            raise ConfigError(
                f"training.global_batch ({self.global_batch}) must be >= "
                f"micro_batch ({self.micro_batch})"
            )
        if self.global_batch % self.micro_batch:
            raise ConfigError(
                f"training.global_batch ({self.global_batch}) must be divisible by "
                f"micro_batch ({self.micro_batch})"
            )
        if self.lr <= 0:
            raise ConfigError(f"training.lr must be positive, got {self.lr}")
        if self.warmup_steps < 0:
            raise ConfigError(f"training.warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.max_steps <= 0:
            raise ConfigError(f"training.max_steps must be positive, got {self.max_steps}")
        if self.grad_clip <= 0:
            raise ConfigError(f"training.grad_clip must be positive, got {self.grad_clip}")
        if len(self.betas) != 2 or not all(0.0 <= b < 1.0 for b in self.betas):
            raise ConfigError(f"training.betas must be two values in [0, 1), got {self.betas}")


@dataclass(frozen=True)
class PrecisionConfig:
    frozen: str = "bfloat16"
    projector: str = "float32"
    autocast: str = "bfloat16"
    optimizer_state: str = "float32"

    def __post_init__(self) -> None:
        _check_dtype("precision.frozen", self.frozen)
        _check_dtype("precision.projector", self.projector)
        _check_dtype("precision.autocast", self.autocast)
        _check_dtype("precision.optimizer_state", self.optimizer_state)


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return resolve_config(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _build(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown field(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        target = hints[f.name]
        value = data[f.name]
        if is_dataclass(target):
            value = _build(target, value)
        elif target is Path:
            value = Path(value)
        elif typing.get_origin(target) is tuple:
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def resolve_config(cfg: Config) -> dict[str, Any]:
    """Return a JSON-compatible dict with a runtime section for derived values."""

    resolved = asdict(cfg)
    resolved["paths"] = {key: str(value) for key, value in resolved["paths"].items()}
    resolved["runtime"] = {
        "n_visual_tokens": None,
        "reader_hidden_size": None,
        "projector_hidden_width": None,
        "derived_at_runtime": True,
    }
    return resolved
