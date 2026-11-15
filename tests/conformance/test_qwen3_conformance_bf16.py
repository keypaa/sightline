from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sightline.backbone import splice

MODEL_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "Qwen3-1.7B"
IMAGE_PAD = 151655

# DECISIONS/0014: the calibrated bf16 golden-splice difference is 0.0 on this
# reader; 1e-3 is a floor for kernel and hardware variation.
BF16_GOLDEN_SPLICE_TOLERANCE = 1e-3

pytestmark = pytest.mark.skipif(
    not (MODEL_ROOT / "config.json").exists(),
    reason="Qwen3-1.7B snapshot not downloaded; run the WP1 snapshot step",
)


@pytest.fixture(scope="module")
def reader():
    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), dtype=torch.bfloat16)
    model.eval()
    try:
        yield tok, model
    finally:
        del model
        gc.collect()


def test_golden_splice_bf16(reader) -> None:
    tok, model = reader
    ids = tok("What color is the shape?", return_tensors="pt")["input_ids"]
    ids[0, 2] = IMAGE_PAD
    embed = model.get_input_embeddings()
    visual = embed(ids[:, 2:3])
    spliced = splice(ids, visual, IMAGE_PAD, embed.weight)
    plain = model(input_ids=ids).logits
    got = model(inputs_embeds=spliced).logits
    diff = (plain.float() - got.float()).abs().max().item()
    assert diff <= BF16_GOLDEN_SPLICE_TOLERANCE, diff
