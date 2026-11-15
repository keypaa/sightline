from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from sightline.backbone import make_attention_mask, make_position_ids, splice

# A tiny Qwen3 model exercises the splice, mask and position code without
# loading the 1.7B reader. The real reader is covered in tests/conformance/.
VOCAB = 128
PLACEHOLDER = 5
PAD = 6


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


def tiny_ids(length: int = 8, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (1, length), generator=generator)
    ids[0, 2] = PLACEHOLDER
    return ids


def test_golden_splice_tiny_fp32() -> None:
    model = tiny_model()
    ids = tiny_ids()
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3])
    spliced = splice(ids, visual, PLACEHOLDER, embed.weight)
    plain = model(input_ids=ids).logits
    got = model(inputs_embeds=spliced).logits
    assert torch.equal(plain, got)


def test_logit_change_tiny() -> None:
    model = tiny_model()
    ids = tiny_ids()
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3])
    ids2 = ids.repeat(2, 1)
    visual2 = visual.repeat(2, 1, 1).clone()
    visual2[0, 0] += 1.0
    perturbed = model(inputs_embeds=splice(ids2, visual2, PLACEHOLDER, embed.weight)).logits
    baseline = model(
        inputs_embeds=splice(ids2, visual.repeat(2, 1, 1), PLACEHOLDER, embed.weight)
    ).logits
    assert not torch.allclose(perturbed[0, 3:], baseline[0, 3:])
    torch.testing.assert_close(perturbed[1], baseline[1], rtol=0, atol=1e-6)


def test_placeholder_count_mismatch_raises() -> None:
    model = tiny_model()
    ids = tiny_ids()
    embed = model.get_input_embeddings()

    wrong = embed(ids[:, 2:4])
    with pytest.raises(ValueError, match="placeholder"):
        splice(ids, wrong, PLACEHOLDER, embed.weight)

    ids2 = ids.repeat(2, 1)
    ids2[1, 3] = PLACEHOLDER
    with pytest.raises(ValueError, match="sample 1"):
        splice(ids2, embed(ids2[:, 2:3]), PLACEHOLDER, embed.weight)


def test_padding_invariance_tiny() -> None:
    model = tiny_model()
    ids_a = tiny_ids(seed=0)
    ids_b = tiny_ids(length=6, seed=1)
    max_len = max(ids_a.shape[1], ids_b.shape[1])
    batch = torch.full((2, max_len), PAD, dtype=torch.long)
    batch[0, : ids_a.shape[1]] = ids_a
    batch[1, : ids_b.shape[1]] = ids_b
    mask = make_attention_mask(batch, PAD)
    position_ids = make_position_ids(mask)
    embed = model.get_input_embeddings()
    visual = torch.stack([embed(batch[0, 2:3]), embed(batch[1, 2:3])], dim=0)
    batched = model(
        inputs_embeds=splice(batch, visual, PLACEHOLDER, embed.weight),
        attention_mask=mask,
        position_ids=position_ids,
    ).logits
    single_a = model(inputs_embeds=splice(ids_a, visual[0:1], PLACEHOLDER, embed.weight)).logits
    single_b = model(inputs_embeds=splice(ids_b, visual[1:2], PLACEHOLDER, embed.weight)).logits
    torch.testing.assert_close(batched[0, : ids_a.shape[1]], single_a[0], rtol=0, atol=1e-5)
    torch.testing.assert_close(batched[1, : ids_b.shape[1]], single_b[0], rtol=0, atol=1e-5)


def test_placeholder_embedding_is_not_used() -> None:
    model = tiny_model()
    ids = tiny_ids()
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3]).clone()
    with torch.no_grad():
        saved = embed.weight[PLACEHOLDER].clone()
        embed.weight[PLACEHOLDER].zero_()
        try:
            got = splice(ids, visual, PLACEHOLDER, embed.weight)
        finally:
            embed.weight[PLACEHOLDER] = saved
    torch.testing.assert_close(got[:, 2:3], visual, rtol=0, atol=0)
    torch.testing.assert_close(got[:, :2], embed(ids[:, :2]), rtol=0, atol=0)
