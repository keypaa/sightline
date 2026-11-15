"""Evaluation helpers: exact match, controls and loss gap.

The functions are generic: the caller supplies a ``generate_fn`` / ``loss_fn`` /
feature lookup, so the same logic is testable without the 1.7B reader and can
be used on Modal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
import torch


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


def mcnemar(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Two-sided exact McNemar test on paired correctness booleans."""

    a = torch.as_tensor(a, dtype=torch.bool)
    b = torch.as_tensor(b, dtype=torch.bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return {"b": n01, "c": n10, "p": 1.0}
    k = min(n01, n10)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0**n)
    return {"b": n01, "c": n10, "p": min(p, 1.0)}


def paired_bootstrap(differences, n: int = 10000, seed: int = 0) -> dict:
    """Bootstrap the mean of paired differences (95% percentile interval)."""

    x = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        sample = rng.choice(x, size=x.size, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def g1_report(per_task: dict[str, dict[str, list[bool]]]) -> dict[str, dict]:
    """Per-task correct/blank/mismatched/no-tokens report with McNemar and bootstrap."""

    results = {}
    for task, data in per_task.items():
        correct = torch.as_tensor(data["correct"], dtype=torch.bool)
        controls = {
            name: torch.as_tensor(data[name], dtype=torch.bool)
            for name in ("blank", "mismatched", "no_tokens")
        }
        rates = {name: float(c.float().mean()) for name, c in controls.items()}
        best_name = max(rates, key=rates.get)
        best = controls[best_name]
        correct_rate = float(correct.float().mean())
        results[task] = {
            "correct": correct_rate,
            "controls": rates,
            "best_control": best_name,
            "vision_minus_blind": correct_rate - rates[best_name],
            "mcnemar": mcnemar(correct, best),
            "bootstrap": paired_bootstrap(
                (correct.float() - best.float()).numpy(), n=10000, seed=0
            ),
        }
    return results


def g1_pass(results: dict[str, dict]) -> bool:
    """G1 pass for one seed: at least 2 of 3 tasks with the frozen criteria."""

    passing = [
        task
        for task, r in results.items()
        if r["vision_minus_blind"] >= 0.30
        and r["mcnemar"]["p"] < 0.01
        and r["bootstrap"]["lo"] > 0.0
    ]
    return len(passing) >= 2
