"""Evaluation and controls (implemented in WP9 and WP10)."""

from __future__ import annotations

from typing import Protocol


class Evaluator(Protocol):
    def evaluate(self) -> dict:
        """Return per-sample paired results, including blank, shuffled and blind controls."""
        ...
