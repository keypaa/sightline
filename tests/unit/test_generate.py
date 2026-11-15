from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from sightline.backbone import splice
from sightline.generate import generate_new_tokens

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


def tiny_ids(length: int = 6, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (1, length), generator=generator)
    ids[0, 2] = PLACEHOLDER
    return ids


def test_inputs_embeds_generation_returns_only_new_tokens() -> None:
    model = tiny_model()
    ids = tiny_ids()
    embed = model.get_input_embeddings()
    spliced = splice(ids, embed(ids[:, 2:3]), PLACEHOLDER, embed.weight)

    from_ids = generate_new_tokens(model, input_ids=ids, max_new_tokens=5, pad_token_id=0)
    from_embeds = generate_new_tokens(
        model,
        inputs_embeds=spliced,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=5,
        pad_token_id=0,
    )
    assert from_ids.shape == (1, 5)
    assert from_embeds.shape == (1, 5)


def test_cache_matches_recompute() -> None:
    model = tiny_model()
    ids = tiny_ids()

    cached = generate_new_tokens(
        model, input_ids=ids, max_new_tokens=5, use_cache=True, pad_token_id=0
    )
    recompute = generate_new_tokens(
        model, input_ids=ids, max_new_tokens=5, use_cache=False, pad_token_id=0
    )
    assert torch.equal(cached, recompute)


def test_golden_splice_generation_matches_ids() -> None:
    model = tiny_model()
    ids = tiny_ids()
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3])
    spliced = splice(ids, visual, PLACEHOLDER, embed.weight)

    from_ids = generate_new_tokens(model, input_ids=ids, max_new_tokens=5, pad_token_id=0)
    from_embeds = generate_new_tokens(
        model,
        inputs_embeds=spliced,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=5,
        pad_token_id=0,
    )
    assert torch.equal(from_ids, from_embeds)


def test_generation_padding_invariance() -> None:
    model = tiny_model()
    ids_a = tiny_ids(length=6, seed=0)
    ids_b = tiny_ids(length=4, seed=1)
    max_len = max(ids_a.shape[1], ids_b.shape[1])
    batch = torch.full((2, max_len), 0, dtype=torch.long)
    batch[0, : ids_a.shape[1]] = ids_a
    batch[1, : ids_b.shape[1]] = ids_b
    mask = (batch != 0).long()

    batched = generate_new_tokens(
        model,
        input_ids=batch,
        attention_mask=mask,
        max_new_tokens=5,
        pad_token_id=0,
    )
    single = generate_new_tokens(model, input_ids=ids_a, max_new_tokens=5, pad_token_id=0)
    assert torch.equal(batched[0], single[0])
