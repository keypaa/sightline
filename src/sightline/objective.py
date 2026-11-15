"""Answer-only objective.

Cross-entropy over supervised answer tokens; positions with label ``-100`` are
ignored. A prompt-only sample has no supervised positions and contributes a
zero loss (with a gradient path, so ``backward`` is still valid).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def answer_only_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0 or (labels != IGNORE_INDEX).sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
