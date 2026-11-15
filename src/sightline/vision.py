"""Vision encoder interface (implemented in WP5).

The encoder is frozen. It maps a fixed 448x448 canvas to merged patch tokens;
the token count is derived from the checkpoint config, never hard-coded.
"""

from __future__ import annotations

from typing import Protocol

import torch


class VisionEncoder(Protocol):
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return merged patch features of shape ``(batch, tokens, width)``."""
        ...
