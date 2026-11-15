"""Projector interface (implemented in WP7).

Only the projector trains. Its output width equals the reader's hidden size.
"""

from __future__ import annotations

from typing import Protocol

import torch


class Projector(Protocol):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map encoder features to reader-embedding width."""
        ...
