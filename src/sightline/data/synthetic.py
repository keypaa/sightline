"""Minimal deterministic synthetic data for the WP10 overfit.

32 scenes: 8 colors x 4 shapes. The question is constant
("What color is the shape?") and the answer is the color, so the text alone has
8 balanced answers and the model must use the image. Regenerating from the same
seed reproduces the manifest hash.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import PIL
from PIL import Image, ImageDraw

GENERATOR_VERSION = "wp9-1"
DEFAULT_SIZE = 448
QUESTION = "What color is the shape?"

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
}
SHAPES = ("circle", "square", "triangle", "diamond")


@dataclass(frozen=True)
class Example:
    index: int
    image: Image.Image
    image_sha256: str
    question: str
    answer: str
    color: str
    shape: str
    count: int
    template_id: int
    split: str
    seed: int


def render(color_rgb: tuple[int, int, int], shape: str, size: int = DEFAULT_SIZE) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    cx = cy = size // 2
    radius = size // 4
    if shape == "circle":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color_rgb)
    elif shape == "square":
        draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=color_rgb)
    elif shape == "triangle":
        draw.polygon(
            [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
            fill=color_rgb,
        )
    elif shape == "diamond":
        draw.polygon(
            [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)],
            fill=color_rgb,
        )
    else:
        raise ValueError(f"unknown shape: {shape!r}")
    return image


def generate_dataset(n: int = 32, seed: int = 0, size: int = DEFAULT_SIZE) -> list[Example]:
    expected = len(COLORS) * len(SHAPES)
    if n != expected:
        raise ValueError(f"n must be {expected} for the balanced overfit set, got {n}")

    pairs = [(color, shape) for color in COLORS for shape in SHAPES]
    random.Random(seed).shuffle(pairs)

    examples = []
    for index, (color, shape) in enumerate(pairs):
        image = render(COLORS[color], shape, size)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        examples.append(
            Example(
                index=index,
                image=image,
                image_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                question=QUESTION,
                answer=color,
                color=color,
                shape=shape,
                count=1,
                template_id=0,
                split="train",
                seed=seed,
            )
        )
    return examples


def manifest_dict(examples: list[Example], seed: int) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "pillow_version": PIL.__version__,
        "seed": seed,
        "n": len(examples),
        "examples": [
            {
                "index": example.index,
                "image_sha256": example.image_sha256,
                "question": example.question,
                "answer": example.answer,
                "color": example.color,
                "shape": example.shape,
                "count": example.count,
                "template_id": example.template_id,
                "split": example.split,
                "seed": example.seed,
            }
            for example in examples
        ],
    }


def manifest_hash(examples: list[Example], seed: int) -> str:
    payload = json.dumps(
        manifest_dict(examples, seed), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def save_manifest(examples: list[Example], seed: int, path: str | Path) -> str:
    data = manifest_dict(examples, seed)
    digest = manifest_hash(examples, seed)
    data["manifest_sha256"] = digest
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return digest


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
