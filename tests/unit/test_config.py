from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sightline.config import Config, ConfigError, TrainingConfig, resolve_config


def test_defaults_match_decisions() -> None:
    cfg = Config()
    assert cfg.encoder.canvas == 448
    assert cfg.training.seeds == (0, 1, 2)
    assert cfg.training.global_batch == 128
    assert cfg.training.lr == pytest.approx(3e-4)
    assert cfg.training.warmup_steps == 50
    assert cfg.training.weight_decay == pytest.approx(0.1)
    assert cfg.training.betas == (0.9, 0.95)
    assert cfg.training.grad_clip == pytest.approx(1.0)
    assert cfg.reader.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert cfg.encoder.revision == "f831ab66814297da540d832a5235f8e904f29d06"
    assert cfg.reader.attn_backend == "sdpa"
    assert cfg.precision.frozen == "bfloat16"
    assert cfg.precision.projector == "float32"
    assert cfg.precision.optimizer_state == "float32"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_allowed_seeds(seed: int) -> None:
    assert TrainingConfig(seeds=(seed,)).seeds == (seed,)


def test_seed_42_is_rejected() -> None:
    with pytest.raises(ConfigError):
        TrainingConfig(seeds=(42,))


def test_unknown_seed_is_rejected() -> None:
    with pytest.raises(ConfigError):
        TrainingConfig(seeds=(3,))


def test_micro_batch_must_divide_global_batch() -> None:
    with pytest.raises(ConfigError):
        TrainingConfig(global_batch=10, micro_batch=4)


def test_resolved_config_is_json_serializable() -> None:
    resolved = resolve_config(Config())
    text = json.dumps(resolved)
    for key in ("paths", "reader", "encoder", "projector", "training", "precision", "runtime"):
        assert key in resolved, key
    assert json.loads(text)["encoder"]["canvas"] == 448


def test_from_dict_rejects_unknown_field() -> None:
    with pytest.raises(ConfigError):
        Config.from_dict({"bogus": 1})


def test_from_dict_roundtrip() -> None:
    cfg = Config()
    data = json.loads(json.dumps(resolve_config(cfg)))
    data.pop("runtime")
    again = Config.from_dict(data)
    assert again.encoder.canvas == 448
    assert again.training.seeds == (0, 1, 2)


def test_no_seed_42_in_source() -> None:
    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"seed\s*[:=]\s*42\b")
    for sub in ("src", "experiments"):
        for path in (root / sub).rglob("*.py"):
            assert not pattern.search(path.read_text()), path
