from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sightline.backbone import make_attention_mask, make_position_ids, splice

MODEL_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "Qwen3-1.7B"
IMAGE_PAD = 151655
PAD = 151643
PADDING_TOLERANCE = 1e-4

pytestmark = pytest.mark.skipif(
    not (MODEL_ROOT / "config.json").exists(),
    reason="Qwen3-1.7B snapshot not downloaded; run the WP1 snapshot step",
)


@pytest.fixture(scope="module")
def reader():
    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), dtype=torch.float32)
    model.eval()
    try:
        yield tok, model
    finally:
        del model
        gc.collect()


def ids_with_placeholder(tok, text: str = "What color is the shape?") -> torch.Tensor:
    ids = tok(text, return_tensors="pt")["input_ids"]
    ids[0, 2] = IMAGE_PAD
    return ids


def test_golden_splice(reader) -> None:
    tok, model = reader
    ids = ids_with_placeholder(tok)
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3])
    spliced = splice(ids, visual, IMAGE_PAD, embed.weight)
    plain = model(input_ids=ids).logits
    got = model(inputs_embeds=spliced).logits
    assert torch.equal(plain, got)


def test_logit_change(reader) -> None:
    tok, model = reader
    ids = ids_with_placeholder(tok)
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3])
    ids2 = ids.repeat(2, 1)
    visual2 = visual.repeat(2, 1, 1).clone()
    visual2[0, 0] += 1.0
    perturbed = model(inputs_embeds=splice(ids2, visual2, IMAGE_PAD, embed.weight)).logits
    baseline = model(
        inputs_embeds=splice(ids2, visual.repeat(2, 1, 1), IMAGE_PAD, embed.weight)
    ).logits
    assert not torch.allclose(perturbed[0, 3:], baseline[0, 3:])
    torch.testing.assert_close(perturbed[1], baseline[1], rtol=0, atol=1e-6)


def test_placeholder_count(reader) -> None:
    tok, model = reader
    ids = ids_with_placeholder(tok)
    embed = model.get_input_embeddings()
    with pytest.raises(ValueError, match="placeholder"):
        splice(ids, embed(ids[:, 2:4]), IMAGE_PAD, embed.weight)

    ids2 = ids.repeat(2, 1)
    ids2[1, 3] = IMAGE_PAD
    with pytest.raises(ValueError, match="sample 1"):
        splice(ids2, embed(ids2[:, 2:3]), IMAGE_PAD, embed.weight)


def test_padding_invariance(reader) -> None:
    tok, model = reader
    ids_a = ids_with_placeholder(tok, "What color is the shape?")
    ids_b = ids_with_placeholder(tok, "What shape is it?")
    max_len = max(ids_a.shape[1], ids_b.shape[1])
    batch = torch.full((2, max_len), PAD, dtype=torch.long)
    batch[0, : ids_a.shape[1]] = ids_a
    batch[1, : ids_b.shape[1]] = ids_b
    mask = make_attention_mask(batch, PAD)
    position_ids = make_position_ids(mask)
    embed = model.get_input_embeddings()
    visual = torch.stack([embed(batch[0, 2:3]), embed(batch[1, 2:3])], dim=0)
    batched = model(
        inputs_embeds=splice(batch, visual, IMAGE_PAD, embed.weight),
        attention_mask=mask,
        position_ids=position_ids,
    ).logits
    single_a = model(inputs_embeds=splice(ids_a, visual[0:1], IMAGE_PAD, embed.weight)).logits
    single_b = model(inputs_embeds=splice(ids_b, visual[1:2], IMAGE_PAD, embed.weight)).logits
    torch.testing.assert_close(
        batched[0, : ids_a.shape[1]], single_a[0], rtol=0, atol=PADDING_TOLERANCE
    )
    torch.testing.assert_close(
        batched[1, : ids_b.shape[1]], single_b[0], rtol=0, atol=PADDING_TOLERANCE
    )


def test_placeholder_embedding_is_not_used(reader) -> None:
    tok, model = reader
    ids = ids_with_placeholder(tok)
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3]).clone()
    with torch.no_grad():
        saved = embed.weight[IMAGE_PAD].clone()
        embed.weight[IMAGE_PAD].zero_()
        try:
            got = splice(ids, visual, IMAGE_PAD, embed.weight)
        finally:
            embed.weight[IMAGE_PAD] = saved
    torch.testing.assert_close(got[:, 2:3], visual, rtol=0, atol=0)
    torch.testing.assert_close(got[:, :2], embed(ids[:, :2]), rtol=0, atol=0)
