"""Deterministic sampling, shared by the fetch and probe stages."""

from __future__ import annotations

import random
from typing import Any


def sample_subset(items: list[Any], n: int, seed: int = 0) -> list[Any]:
    """Take n items at random, reproducibly.

    The M1 splits are ordered by answer, so ``items[:n]`` is a biased sample.
    A seeded shuffle keeps the mix and makes a re-run identical.
    """

    if n <= 0 or n >= len(items):
        return list(items)
    return random.Random(seed).sample(list(items), n)
