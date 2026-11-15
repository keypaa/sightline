from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from sightline.projector import Projector
from sightline.vision_k3_encoder import K3VisionEncoder

K3_WEIGHTS = Path(__file__).resolve().parents[2] / "artifacts" / "k3_vision_f831ab66.safetensors"

pytestmark = [
    pytest.mark.skipif(
        not K3_WEIGHTS.exists(),
        reason="K3 tower not present; run the WP5 extraction",
    ),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def encoder() -> K3VisionEncoder:
    return K3VisionEncoder()


def test_projector_maps_k3_features_to_reader_width(encoder: K3VisionEncoder) -> None:
    image = Image.new("RGB", (448, 448), (128, 64, 32))
    features = encoder.encode([image])
    assert tuple(features.shape) == (1, 256, 4096)

    projector = Projector(input_width=4096, reader_width=2048)
    out = projector(features)
    assert tuple(out.shape) == (1, 256, 2048)
    assert torch.isfinite(out).all()
