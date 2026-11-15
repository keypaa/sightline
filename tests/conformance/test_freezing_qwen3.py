from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from sightline.backbone import splice
from sightline.freezing import assert_frozen, freeze, parameter_hashes
from sightline.vision_k3_encoder import K3VisionEncoder

MODEL_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "Qwen3-1.7B"
K3_WEIGHTS = Path(__file__).resolve().parents[2] / "artifacts" / "k3_vision_f831ab66.safetensors"
IMAGE_PAD = 151655

pytestmark = [
    pytest.mark.skipif(
        not (MODEL_ROOT / "config.json").exists() or not K3_WEIGHTS.exists(),
        reason="Qwen3-1.7B or the K3 tower is not present",
    ),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def frozen():
    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), dtype=torch.float32)
    encoder = K3VisionEncoder(dtype=torch.float32)
    freeze(model)
    freeze(encoder.model)
    return tok, model, encoder


def test_only_projector_gradients_and_frozen_hashes(frozen) -> None:
    tok, model, encoder = frozen
    model_before = parameter_hashes(model)
    encoder_before = parameter_hashes(encoder.model)

    projector = torch.nn.Linear(4096, 2048)
    image = Image.new("RGB", (448, 448), (128, 64, 32))
    features = encoder.encode([image])
    visual = projector(features)

    ids = tok("What color is the shape?", return_tensors="pt")["input_ids"]
    padding = torch.full((1, 256), IMAGE_PAD, dtype=torch.long)
    ids = torch.cat([ids[:, :2], padding, ids[:, 2:]], dim=1)

    embed = model.get_input_embeddings()
    spliced = splice(ids, visual, IMAGE_PAD, embed.weight)
    logits = model(inputs_embeds=spliced).logits
    loss = torch.nn.functional.cross_entropy(logits[:, -1], ids[:, -1])
    loss.backward()

    assert projector.weight.grad is not None
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in encoder.model.parameters())
    assert_frozen(model, model_before)
    assert_frozen(encoder.model, encoder_before)
