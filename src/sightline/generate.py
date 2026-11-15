"""Greedy generation from ids or spliced ``inputs_embeds``.

``generate(inputs_embeds=...)`` uses the provided embeddings once and the KV
cache for the rest; it returns only the new tokens. ``generate_new_tokens``
normalises the ids path to return only the new tokens too.
"""

from __future__ import annotations

from typing import Protocol

import torch


class GenerationModel(Protocol):
    def generate(self, *args, **kwargs) -> torch.Tensor: ...


def generate_new_tokens(
    model: GenerationModel,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None = None,
    pad_token_id: int | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Greedy-decode and return only the newly generated tokens."""

    if inputs_embeds is not None:
        return model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    if input_ids is None:
        raise ValueError("provide input_ids or inputs_embeds")

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=use_cache,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return generated[:, input_ids.shape[1] :]
