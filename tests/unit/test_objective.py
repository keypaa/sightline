from __future__ import annotations

import pytest
import torch

from sightline.objective import answer_only_loss


def test_answer_only_loss_ignores_prompt() -> None:
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[-100, 1, 2, -100]])
    loss = answer_only_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[0, 1:3], labels[0, 1:3])
    assert loss == pytest.approx(expected.item())


def test_prompt_only_sample_has_zero_loss() -> None:
    logits = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.full((1, 4), -100)
    loss = answer_only_loss(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_all_supervised_matches_cross_entropy() -> None:
    logits = torch.randn(2, 3, 5)
    labels = torch.randint(0, 5, (2, 3))
    loss = answer_only_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
    assert loss == pytest.approx(expected.item())
