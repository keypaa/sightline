"""Training loop (implemented in WP8)."""

from __future__ import annotations

from typing import Protocol


class Trainer(Protocol):
    def train(self) -> None:
        """Run the training loop and write metrics to the run record."""
        ...
