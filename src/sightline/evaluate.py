"""Evaluation helpers: exact match, controls and loss gap.

The functions are generic: the caller supplies a ``generate_fn`` / ``loss_fn`` /
feature lookup, so the same logic is testable without the 1.7B reader and can
be used on Modal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalExample:
    index: int
    answer: str
    image: object


def exact_match_accuracy(
    generate_fn: Callable[[object, str], str],
    examples: Sequence[EvalExample],
    image_for: Callable[[EvalExample, str], object],
    condition: str = "real",
) -> float:
    if not examples:
        return 0.0
    correct = sum(
        generate_fn(image_for(example, condition), condition) == example.answer
        for example in examples
    )
    return correct / len(examples)


def evaluate_controls(
    generate_fn: Callable[[object, str], str],
    examples: Sequence[EvalExample],
    image_for: Callable[[EvalExample, str], object],
) -> dict[str, float]:
    real = exact_match_accuracy(generate_fn, examples, image_for, "real")
    blank = exact_match_accuracy(generate_fn, examples, image_for, "blank")
    shuffled = exact_match_accuracy(generate_fn, examples, image_for, "shuffled")
    swapped = exact_match_accuracy(generate_fn, examples, image_for, "swapped")
    return {
        "real": real,
        "blank": blank,
        "shuffled": shuffled,
        "swapped": swapped,
        "vision_minus_blind": real - blank,
    }


def mean_loss_gap(
    loss_fn: Callable[[object], float],
    examples: Sequence[EvalExample],
    feature_for: Callable[[EvalExample, str], object],
) -> float:
    if not examples:
        return 0.0
    gaps = [
        loss_fn(feature_for(example, "blank")) - loss_fn(feature_for(example, "real"))
        for example in examples
    ]
    return sum(gaps) / len(gaps)