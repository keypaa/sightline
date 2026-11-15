"""Answer-only objective (implemented in WP8)."""

from __future__ import annotations

from typing import Protocol

import torch


class Objective(Protocol):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return the mean cross-entropy over supervised answer tokens."""
        ...
