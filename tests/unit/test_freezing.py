from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, Qwen3Config

from sightline.backbone import splice
from sightline.freezing import assert_frozen, freeze, parameter_hashes, trainable_parameters

VOCAB = 128
PLACEHOLDER = 5


def tiny_model() -> AutoModelForCausalLM:
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


def tiny_ids() -> torch.Tensor:
    ids = torch.randint(0, VOCAB, (1, 6), generator=torch.Generator().manual_seed(0))
    ids[0, 2] = PLACEHOLDER
    return ids


def test_freeze_sets_eval_and_requires_grad_false() -> None:
    model = tiny_model()
    model.train()
    freeze(model)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_only_projector_has_gradients_and_changes() -> None:
    model = tiny_model()
    projector = nn.Linear(32, 32)
    freeze(model)
    frozen_before = parameter_hashes(model)
    assert trainable_parameters(model) == []
    assert len(trainable_parameters(projector)) == 2

    optimizer = torch.optim.SGD(projector.parameters(), lr=0.1)
    before = projector.weight.detach().clone()

    for _ in range(20):
        optimizer.zero_grad()
        ids = tiny_ids()
        embed = model.get_input_embeddings()
        visual = projector(embed(ids[:, 2:3]))
        spliced = splice(ids, visual, PLACEHOLDER, embed.weight)
        logits = model(inputs_embeds=spliced).logits
        loss = torch.nn.functional.cross_entropy(logits[:, -1], ids[:, -1])
        loss.backward()
        assert projector.weight.grad is not None
        assert all(p.grad is None for p in model.parameters())
        optimizer.step()

    assert not torch.equal(before, projector.weight.detach())
    assert_frozen(model, frozen_before)