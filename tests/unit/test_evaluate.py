from __future__ import annotations

import pytest

from sightline.evaluate import (
    EvalExample,
    evaluate_controls,
    exact_match_accuracy,
    mean_loss_gap,
)

COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]
EXAMPLES = [EvalExample(index=i, answer=color, image=f"img{i}") for i, color in enumerate(COLORS)]


def test_exact_match_accuracy() -> None:
    generate = lambda image, condition: image
    image_for = lambda example, condition: example.answer
    assert exact_match_accuracy(generate, EXAMPLES, image_for) == 1.0


def test_exact_match_accuracy_partial() -> None:
    generate = lambda image, condition: "red" if image == "red" else "wrong"
    image_for = lambda example, condition: example.answer
    assert exact_match_accuracy(generate, EXAMPLES, image_for) == pytest.approx(1 / 8)


def test_controls_collapse_to_chance() -> None:
    def generate(image, condition):
        return image if condition == "real" else "red"

    def image_for(example, condition):
        if condition == "real":
            return example.answer
        return condition

    result = evaluate_controls(generate, EXAMPLES, image_for)
    assert result["real"] == 1.0
    assert result["blank"] == pytest.approx(1 / 8)
    assert result["shuffled"] == pytest.approx(1 / 8)
    assert result["swapped"] == pytest.approx(1 / 8)
    assert result["vision_minus_blind"] == pytest.approx(1 - 1 / 8)


def test_mean_loss_gap() -> None:
    loss = lambda features: {"real": 0.1, "blank": 1.0}[features]
    feature_for = lambda example, condition: condition
    assert mean_loss_gap(loss, EXAMPLES, feature_for) == pytest.approx(0.9)