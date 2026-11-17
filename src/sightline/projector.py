"""Baseten-style projector.

LayerNorm(input_width) -> Linear(input_width, hidden_width) -> GELU ->
Linear(hidden_width, reader_width). Only this module trains; the reader and
encoder are frozen.

The hidden width defaults to the merged-token width (4096 for MoonViT-V2,
4608 for K2.6). The output width is the reader's hidden size. The parameter
count is derived from the module, not hard-coded.
"""

from __future__ import annotations

import torch
from torch import nn


class Projector(nn.Module):
    def __init__(
        self,
        input_width: int,
        reader_width: int,
        hidden_width: int | None = None,
        *,
        init: str = "random",
    ) -> None:
        super().__init__()
        if init not in ("random", "embedding_alignment"):
            raise ValueError(f"init must be 'random' or 'embedding_alignment', got {init!r}")
        if init == "embedding_alignment":
            raise NotImplementedError(
                "embedding alignment init is a later ablation; see DECISIONS 0022"
            )

        self.input_width = input_width
        self.reader_width = reader_width
        self.hidden_width = input_width if hidden_width is None else hidden_width
        self.init = init
        self.norm = nn.LayerNorm(input_width)
        self.fc1 = nn.Linear(input_width, self.hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_width, reader_width)
        self.register_buffer("output_scale", torch.tensor(1.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            features = features.reshape(features.shape[0], features.shape[1], -1)
        return self.fc2(self.act(self.fc1(self.norm(features)))) * self.output_scale

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def norm_ratio(self, features: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
        """Mean projector-output norm over mean embedding-row norm."""

        output = self(features)
        embedding_norm = embedding_weight.norm(dim=-1).float().mean()
        return output.float().norm(dim=-1).mean() / embedding_norm


def calibrate_output_scale(
    projector: Projector, reference: torch.Tensor, features: torch.Tensor
) -> float:
    """Scale the projector output so its RMS matches a reference tensor's RMS.

    A randomly initialised projector spliced into the image-pad positions produces
    embeddings far larger than the reader's own, which drives the frozen reader
    into an extreme, high-loss region. Setting the output RMS to the reference's
    RMS puts the run in a sane region without hard-coding a constant: both scales
    come from the weights and data in front of us.
    """

    with torch.no_grad():
        produced = projector(features.to(next(projector.parameters()).dtype))
        rms_out = produced.float().pow(2).mean().sqrt().clamp_min(1e-12)
        rms_ref = reference.float().pow(2).mean().sqrt().clamp_min(1e-12)
    scale = float(rms_ref / rms_out)
    projector.output_scale.fill_(scale)
    return scale
