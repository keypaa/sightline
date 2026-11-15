from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from sightline.data.m1_synthetic import generate_m1_dataset, manifest_hash


@pytest.fixture(scope="session")
def dataset():
    return generate_m1_dataset(seed=0)


def test_500_held_out_per_task(dataset) -> None:
    for task in ("color", "shape", "count"):
        assert sum(e.task == task for e in dataset.test) >= 500


def test_balanced_answers(dataset) -> None:
    counts = defaultdict(Counter)
    for example in dataset.train + dataset.val + dataset.test:
        counts[(example.task, example.template_id)][example.answer] += 1
    for answer_counts in counts.values():
        assert max(answer_counts.values()) - min(answer_counts.values()) <= 1, answer_counts


def test_held_out_compositions(dataset) -> None:
    train_pairs = {
        (e.properties.get("color"), e.properties.get("shape"))
        for e in dataset.train
        if e.task in ("color", "shape")
    }
    test_pairs = {
        (e.properties.get("color"), e.properties.get("shape"))
        for e in dataset.test
        if e.task in ("color", "shape")
    }
    assert ("red", "triangle") not in train_pairs
    assert ("red", "triangle") in test_pairs
    assert 5 not in {e.properties.get("count") for e in dataset.train if e.task == "count"}
    assert 5 in {e.properties.get("count") for e in dataset.test if e.task == "count"}


def test_matched_pairs(dataset) -> None:
    by_question = defaultdict(set)
    for example in dataset.train + dataset.val + dataset.test:
        by_question[example.question].add(example.answer)
    assert any(len(answers) > 1 for answers in by_question.values())


def test_manifest_hash(dataset) -> None:
    fresh = generate_m1_dataset(seed=0)
    assert manifest_hash(dataset, 0) == manifest_hash(fresh, 0)


def test_no_split_overlap(dataset) -> None:
    train_hashes = {e.image_sha256 for e in dataset.train}
    val_hashes = {e.image_sha256 for e in dataset.val}
    test_hashes = {e.image_sha256 for e in dataset.test}
    assert not (train_hashes & val_hashes)
    assert not (train_hashes & test_hashes)
    assert not (val_hashes & test_hashes)
    train_templates = {e.template_id for e in dataset.train}
    val_templates = {e.template_id for e in dataset.val}
    test_templates = {e.template_id for e in dataset.test}
    assert not (train_templates & val_templates)
    assert not (train_templates & test_templates)
    assert not (val_templates & test_templates)
