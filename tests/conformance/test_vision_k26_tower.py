from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sightline.vision_k25_encoder import K26VisionEncoder

# The eager K2.6 tower is CPU-bound (about 17 s per 448x448 image on this
# machine). The default is the 20-image acceptance run; set
# SIGHTLINE_VISION_IMAGE_COUNT to a smaller number for a quick check.
IMAGE_COUNT = int(os.environ.get("SIGHTLINE_VISION_IMAGE_COUNT", "20"))

pytestmark = [
    pytest.mark.skipif(
        not (
            Path(__file__).resolve().parents[2] / "artifacts" / "k26_vision_7eb5002f.safetensors"
        ).exists(),
        reason="extracted K2.6 tower not present; run WP6 extraction",
    ),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def encoder() -> K26VisionEncoder:
    return K26VisionEncoder()


def make_images(count: int) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    images: list[Image.Image] = []
    for i in range(count):
        if i % 3 == 0:
            array = np.full(
                (448, 448, 3),
                (i * 10 % 256, 255 - i * 5 % 256, i * 20 % 256),
                dtype=np.uint8,
            )
        elif i % 3 == 1:
            ramp = np.linspace(0, 255, 448, dtype=np.uint8)
            array = np.stack([np.tile(ramp, (448, 1))] * 3, axis=-1)
        else:
            array = rng.integers(0, 256, (448, 448, 3), dtype=np.uint8)
        images.append(Image.fromarray(array))
    return images


def test_processor_448(encoder: K26VisionEncoder) -> None:
    image = Image.new("RGB", (448, 448), (128, 64, 32))
    pixel_values, grid_thws = encoder.preprocess([image])
    assert tuple(pixel_values.shape) == (1024, 3, 14, 14)
    assert grid_thws.tolist() == [[1, 32, 32]]
    assert pixel_values.min() >= -1.0
    assert pixel_values.max() <= 1.0


def test_tower_output_shape(encoder: K26VisionEncoder) -> None:
    out = encoder.encode(make_images(1))
    assert tuple(out.shape) == (1, 256, 4608)
    assert torch.isfinite(out).all()


def test_token_count_is_derived(encoder: K26VisionEncoder) -> None:
    _, grid_thws = encoder.preprocess(make_images(1))
    assert encoder.n_visual_tokens(grid_thws) == (32 // 2) * (32 // 2) == 256


def test_interpolation_mode_is_bicubic(encoder: K26VisionEncoder) -> None:
    assert encoder.model.patch_embed.pos_emb.interpolation_mode == "bicubic"


def test_tower_uses_eager_attention(encoder: K26VisionEncoder) -> None:
    assert encoder.model.encoder.blocks[0].attn_implementation == "eager"


def test_image_batch(encoder: K26VisionEncoder) -> None:
    images = make_images(IMAGE_COUNT)
    got = encoder.encode(images)
    assert tuple(got.shape) == (IMAGE_COUNT, 256, 4608)
    assert torch.isfinite(got).all()

    indices = sorted({0, min(7, IMAGE_COUNT - 1), IMAGE_COUNT - 1})
    for index in indices:
        single = encoder.encode([images[index]])
        torch.testing.assert_close(single[0], got[index], rtol=0, atol=0)
