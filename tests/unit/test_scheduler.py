from __future__ import annotations

import pytest

from sightline.scheduler import warmup_cosine_lr


def test_warmup_ramps_from_zero_to_base_lr() -> None:
    assert warmup_cosine_lr(0, 100, 10, 1e-3) == pytest.approx(1e-4)
    assert warmup_cosine_lr(9, 100, 10, 1e-3) == pytest.approx(1e-3)


def test_cosine_decay_after_warmup() -> None:
    assert warmup_cosine_lr(10, 100, 10, 1e-3) == pytest.approx(1e-3)
    assert warmup_cosine_lr(100, 100, 10, 1e-3) == pytest.approx(0.0, abs=1e-9)


def test_schedule_is_monotonically_decreasing_after_warmup() -> None:
    values = [warmup_cosine_lr(step, 100, 10, 1e-3) for step in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
