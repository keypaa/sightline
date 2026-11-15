"""Reader adapter: splice, positions and masks.

The splice replaces the embedding rows at placeholder positions with projected
visual vectors and returns ``inputs_embeds``. The embedding lookup still runs
over the full ``input_ids``; the placeholder rows are overwritten, so the value
looked up at those positions is never used.

Positions and attention masks support batches with different text lengths and
right padding. The same helpers are used for generation (WP3).
"""

from __future__ import annotations

from typing import Protocol

import torch


def splice(
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    placeholder_id: int,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """Return ``inputs_embeds`` with placeholder positions replaced.

    ``input_ids`` is ``(batch, seq)``; ``visual_embeds`` is
    ``(batch, n_visual, hidden)`` and must match the per-sample placeholder
    count; ``embedding_weight`` is the reader's ``(vocab, hidden)`` table.
    """

    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2-D, got {tuple(input_ids.shape)}")
    if visual_embeds.dim() != 3:
        raise ValueError(f"visual_embeds must be 3-D, got {tuple(visual_embeds.shape)}")
    if input_ids.shape[0] != visual_embeds.shape[0]:
        raise ValueError(
            f"batch mismatch: input_ids has {input_ids.shape[0]}, "
            f"visual_embeds has {visual_embeds.shape[0]}"
        )
    if visual_embeds.shape[2] != embedding_weight.shape[1]:
        raise ValueError(
            f"width mismatch: visual_embeds has {visual_embeds.shape[2]}, "
            f"embedding_weight has {embedding_weight.shape[1]}"
        )

    inputs_embeds = embedding_weight[input_ids]
    n_visual = visual_embeds.shape[1]
    for sample in range(input_ids.shape[0]):
        positions = (input_ids[sample] == placeholder_id).nonzero(as_tuple=True)[0]
        if positions.numel() != n_visual:
            raise ValueError(
                f"sample {sample}: {positions.numel()} placeholder positions "
                f"but {n_visual} visual tokens"
            )
        inputs_embeds[sample, positions] = visual_embeds[sample].to(inputs_embeds.dtype)
    return inputs_embeds


def make_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Return a ``(batch, seq)`` mask with 1 at real tokens and 0 at padding."""

    return (input_ids != pad_token_id).long()


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return ``(batch, seq)`` 1-D positions; padding gets position 0."""

    return (attention_mask.cumsum(dim=1) - 1).clamp(min=0)


class ReaderAdapter(Protocol):
    def compose(
        self,
        input_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        spans: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(inputs_embeds, attention_mask, position_ids, labels)``."""
        ...
