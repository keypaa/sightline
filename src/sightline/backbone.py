"""Reader adapter interface (implemented in WP2, WP3 and WP4).

One implementation per reader family. The adapter owns tokenization, splicing
visual embeddings into ``inputs_embeds``, position ids, the attention mask and
the generation path. Qwen3 is the M0 reader.
"""

from __future__ import annotations

from typing import Protocol

import torch


class ReaderAdapter(Protocol):
    def compose(
        self,
        input_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        spans: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(inputs_embeds, attention_mask, position_ids, labels)``."""
        ...
