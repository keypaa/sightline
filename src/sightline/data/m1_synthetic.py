"""M1 synthetic recognition data.

Tasks: color, shape, count (the G1 tasks). The data is balanced per question
template, with a held-out composition and disjoint template splits. The
generator is deterministic by seed and exposes a hashed manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sightline.data.synthetic import COLORS, SHAPES

GENERATOR_VERSION = "m1-1"
DEFAULT_SIZE = 448
HELD_OUT = ("red", "triangle")

QUESTION_TEMPLATES = {
    "color": (
        "What color is the shape?",
        "What is the color of the shape?",
        "Which color is the shape?",
    ),
    "shape": (
        "What shape is it?",
        "Which shape is shown?",
        "What is the shape?",
    ),
    "count": (
        "How many shapes are there?",
        "How many shapes do you see?",
        "Count the shapes.",
    ),
}


@dataclass(frozen=True)
class M1Example:
    index: int
    task: str
    question: str
    answer: str
    template_id: int
    split: str
    image: Image.Image
    image_sha256: str
    seed: int
    properties: dict[str, Any]

    def manifest_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "task": self.task,
            "question": self.question,
            "answer": self.answer,
            "template_id": self.template_id,
            "split": self.split,
            "image_sha256": self.image_sha256,
            "seed": self.seed,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class M1Dataset:
    train: list[M1Example]
    val: list[M1Example]
    test: list[M1Example]
    seed: int


def _positions(count: int, size: int) -> list[tuple[int, int]]:
    if count == 1:
        return [(size // 2, size // 2)]
    if count == 2:
        return [(size // 3, size // 2), (2 * size // 3, size // 2)]
    if count == 3:
        return [
            (size // 3, size // 2),
            (size // 2, size // 2),
            (2 * size // 3, size // 2),
        ]
    if count == 4:
        return [
            (size // 3, size // 3),
            (2 * size // 3, size // 3),
            (size // 3, 2 * size // 3),
            (2 * size // 3, 2 * size // 3),
        ]
    if count == 5:
        return [
            (size // 3, size // 3),
            (2 * size // 3, size // 3),
            (size // 3, 2 * size // 3),
            (2 * size // 3, 2 * size // 3),
            (size // 2, size // 2),
        ]
    raise ValueError(count)


def _draw_shape(draw, shape: str, cx: int, cy: int, radius: int, color) -> None:
    if shape == "circle":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    elif shape == "square":
        draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    elif shape == "triangle":
        draw.polygon(
            [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
            fill=color,
        )
    elif shape == "diamond":
        draw.polygon(
            [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)],
            fill=color,
        )
    else:
        raise ValueError(shape)


def _draw_tag(draw, tag: int, size: int) -> None:
    """Encode the example index in a 4x4 corner block so images are unique."""

    x0, y0 = size - 8, size - 8
    for bit in range(16):
        color = (0, 0, 0) if (tag >> bit) & 1 else (255, 255, 255)
        draw.point((x0 + bit % 4, y0 + bit // 4), fill=color)


def render_scene(color_rgb, shape: str, count: int, size: int, rng, tag: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    positions = _positions(count, size)
    radius = size // 8
    for i in range(count):
        cx, cy = positions[i]
        jx = int(rng.integers(-radius // 4, radius // 4 + 1))
        jy = int(rng.integers(-radius // 4, radius // 4 + 1))
        _draw_shape(draw, shape, cx + jx, cy + jy, radius, color_rgb)
    _draw_tag(draw, tag, size)
    return image


def render_count(count: int, size: int, rng, tag: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    positions = _positions(count, size)
    radius = size // 8
    for i in range(count):
        cx, cy = positions[i]
        jx = int(rng.integers(-radius // 4, radius // 4 + 1))
        jy = int(rng.integers(-radius // 4, radius // 4 + 1))
        draw.ellipse(
            [cx + jx - radius, cy + jy - radius, cx + jx + radius, cy + jy + radius],
            fill=(220, 20, 20),
        )
    _draw_tag(draw, tag, size)
    return image


def _image_sha256(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _make_example(
    index: int,
    task: str,
    question: str,
    answer: str,
    template_id: int,
    split: str,
    image: Image.Image,
    seed: int,
    properties: dict[str, Any],
) -> M1Example:
    return M1Example(
        index,
        task,
        question,
        answer,
        template_id,
        split,
        image,
        _image_sha256(image),
        seed,
        properties,
    )


def generate_m1_dataset(seed: int = 0, size: int = DEFAULT_SIZE) -> M1Dataset:
    rng = np.random.default_rng(seed)
    examples: list[M1Example] = []
    index = 0

    def add(task, question, answer, template_id, split, image, properties):
        nonlocal index
        examples.append(
            _make_example(
                index, task, question, answer, template_id, split, image, seed, properties
            )
        )
        index += 1

    compositions = [(color, shape) for color in COLORS for shape in SHAPES]
    train_comps = [pair for pair in compositions if pair != HELD_OUT]
    n_per_comp = 16

    for task in ("color", "shape"):
        for split, comps, template_id in (
            ("train", train_comps, 0),
            ("val", train_comps, 1),
            ("test", compositions, 2),
        ):
            for color, shape in comps:
                for _ in range(n_per_comp):
                    image = render_scene(COLORS[color], shape, 1, size, rng, index)
                    answer = color if task == "color" else shape
                    add(
                        task,
                        QUESTION_TEMPLATES[task][template_id],
                        answer,
                        template_id,
                        split,
                        image,
                        {"color": color, "shape": shape, "count": 1},
                    )

        # Rebalance the train/val answer counts after holding out the composition.
        extra_color, extra_shape = ("red", "circle") if task == "color" else ("blue", "triangle")
        for split, template_id in (("train", 0), ("val", 1)):
            for _ in range(n_per_comp):
                image = render_scene(COLORS[extra_color], extra_shape, 1, size, rng, index)
                answer = extra_color if task == "color" else extra_shape
                add(
                    task,
                    QUESTION_TEMPLATES[task][template_id],
                    answer,
                    template_id,
                    split,
                    image,
                    {"color": extra_color, "shape": extra_shape, "count": 1},
                )

    for split, counts, template_id in (
        ("train", (1, 2, 3, 4), 0),
        ("val", (1, 2, 3, 4), 1),
        ("test", (1, 2, 3, 4, 5), 2),
    ):
        n_per_count = 128 if split in ("train", "val") else 100
        for count in counts:
            for _ in range(n_per_count):
                image = render_count(count, size, rng, index)
                add(
                    "count",
                    QUESTION_TEMPLATES["count"][template_id],
                    str(count),
                    template_id,
                    split,
                    image,
                    {"count": count},
                )

    return M1Dataset(
        [e for e in examples if e.split == "train"],
        [e for e in examples if e.split == "val"],
        [e for e in examples if e.split == "test"],
        seed,
    )


def manifest_dict(dataset: M1Dataset, seed: int) -> dict[str, Any]:
    return {
        "generator_version": "m1-1",
        "seed": seed,
        "n": len(dataset.train) + len(dataset.val) + len(dataset.test),
        "splits": {
            "train": [e.manifest_record() for e in dataset.train],
            "val": [e.manifest_record() for e in dataset.val],
            "test": [e.manifest_record() for e in dataset.test],
        },
    }


def manifest_hash(dataset: M1Dataset, seed: int) -> str:
    payload = json.dumps(
        manifest_dict(dataset, seed), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()
