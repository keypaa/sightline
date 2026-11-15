from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sightline.data.synthetic import (
    COLORS,
    SHAPES,
    generate_dataset,
    manifest_hash,
    save_manifest,
)


def test_generate_is_deterministic() -> None:
    first = generate_dataset(seed=0)
    second = generate_dataset(seed=0)
    assert manifest_hash(first, 0) == manifest_hash(second, 0)
    assert [e.image_sha256 for e in first] == [e.image_sha256 for e in second]


def test_different_seed_changes_manifest() -> None:
    first = generate_dataset(seed=0)
    second = generate_dataset(seed=1)
    assert manifest_hash(first, 0) != manifest_hash(second, 1)


def test_balanced_answers() -> None:
    examples = generate_dataset(seed=0)
    counts = Counter(example.answer for example in examples)
    assert len(counts) == len(COLORS)
    assert set(counts.values()) == {len(examples) // len(COLORS)}


def test_question_does_not_identify_answer() -> None:
    examples = generate_dataset(seed=0)
    question = examples[0].question
    answers = {example.answer for example in examples if example.question == question}
    assert len(answers) == len(COLORS)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    examples = generate_dataset(seed=0)
    path = tmp_path / "manifest.json"
    digest = save_manifest(examples, seed=0, path=path)
    loaded = json.loads(path.read_text())
    assert digest == manifest_hash(examples, 0)
    assert loaded["manifest_sha256"] == digest
    assert loaded["n"] == 32
    assert len(loaded["examples"]) == 32
    assert "pillow_version" in loaded


def test_images_are_unique() -> None:
    examples = generate_dataset(seed=0)
    assert len({example.image_sha256 for example in examples}) == len(examples)


def test_image_size_and_labels() -> None:
    examples = generate_dataset(seed=0)
    assert all(example.image.size == (448, 448) for example in examples)
    assert all(example.answer in COLORS for example in examples)
    assert all(example.shape in SHAPES for example in examples)
    assert all(example.count == 1 for example in examples)


def test_question_is_constant_and_answer_varies() -> None:
    examples = generate_dataset(seed=0)
    assert len({example.question for example in examples}) == 1
    assert len({example.answer for example in examples}) == len(COLORS)
