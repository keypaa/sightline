"""Warmup then cosine learning-rate schedule."""

from __future__ import annotations

import math


def warmup_cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
