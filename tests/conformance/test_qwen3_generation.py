from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sightline.backbone import splice
from sightline.generate import generate_new_tokens

MODEL_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "Qwen3-1.7B"
IMAGE_PAD = 151655

pytestmark = [
    pytest.mark.skipif(
        not (MODEL_ROOT / "config.json").exists(),
        reason="Qwen3-1.7B snapshot not downloaded; run the WP1 snapshot step",
    ),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def reader():
    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), dtype=torch.float32)
    model.eval()
    return tok, model


def prompt_ids(tok) -> torch.Tensor:
    text = tok.apply_chat_template(
        [{"role": "user", "content": "<|image_pad|>What color is the shape?"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    return tok(text, return_tensors="pt")["input_ids"]


def test_golden_splice_generation(reader) -> None:
    tok, model = reader
    ids = prompt_ids(tok)
    positions = (ids[0] == IMAGE_PAD).nonzero(as_tuple=True)[0]
    assert len(positions) == 1

    embed = model.get_input_embeddings()
    visual = embed(ids[:, positions])
    spliced = splice(ids, visual, IMAGE_PAD, embed.weight)

    from_ids = generate_new_tokens(
        model,
        input_ids=ids,
        max_new_tokens=8,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    from_embeds = generate_new_tokens(
        model,
        inputs_embeds=spliced,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=8,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    assert torch.equal(from_ids, from_embeds)


def test_cache_matches_recompute(reader) -> None:
    tok, model = reader
    ids = prompt_ids(tok)

    cached = generate_new_tokens(
        model,
        input_ids=ids,
        max_new_tokens=8,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    recompute = generate_new_tokens(
        model,
        input_ids=ids,
        max_new_tokens=8,
        use_cache=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    assert torch.equal(cached, recompute)


def long_prompt_ids(tok) -> torch.Tensor:
    text = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": "<|image_pad|>" * 64 + "What color is the shape?",
            }
        ],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    return tok(text, return_tensors="pt")["input_ids"]


def test_chunked_prefill_matches_single(reader) -> None:
    tok, model = reader
    ids = long_prompt_ids(tok)
    half = ids.shape[1] // 2
    with torch.no_grad():
        single = model(input_ids=ids, use_cache=True)
        first = model(input_ids=ids[:, :half], use_cache=True)
        second = model(
            input_ids=ids[:, half:],
            past_key_values=first.past_key_values,
            use_cache=True,
        )
    torch.testing.assert_close(
        single.logits[:, -1], second.logits[:, -1], rtol=0, atol=1e-4
    )
