"""Evaluation helpers: exact match, controls and loss gap.

The functions are generic: the caller supplies a ``generate_fn`` / ``loss_fn`` /
feature lookup, so the same logic is testable without the 1.7B reader and can
be used on Modal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import comb, sqrt

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


def normalise_answer(text: str) -> str:
    """The exact-match normalisation: case, surrounding space, trailing punctuation."""

    return text.strip().lower().strip(" .!")


def first_line_answer(text: str) -> str:
    """Answer from the first non-blank line, for models that keep talking.

    A reader that answers correctly and then emits markdown scores zero under
    exact match. This skips leading blank lines — a generation prompt that ends
    in a newline makes them common — and applies the same normalisation.

    It deliberately does not search for the target string anywhere in the
    output: `'not red'` contains `red` and must stay wrong. A lenient parser
    that finds substrings would score negated answers as correct.
    """

    for line in text.splitlines():
        if line.strip():
            return normalise_answer(line)
    return ""


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


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used to test a control condition against its chance rate: if the chance rate
    lies inside the interval, the control is indistinguishable from guessing.
    """

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    half = z * sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def g1_report(
    per_task: dict[str, dict[str, list[bool]]], chance: dict[str, float]
) -> dict[str, dict]:
    """Per-task correct/blank/mismatched/no-tokens report with McNemar and bootstrap.

    ``chance`` maps task to its guessing rate (1 over the number of answer options)
    and is required: DECISIONS 0032 freezes the controls against chance as a G1
    criterion, so it is checked here rather than by the caller.

    The check is one-sided. A control that scores far *below* chance is a collapsed
    control and passes; only a control significantly *above* chance fails, because
    that means the blind path carries real information. The untrained-projector
    condition is reported as a diagnostic only; it is not one of the controls that
    the pass criteria compare against.
    """

    results = {}
    for task, data in per_task.items():
        if task not in chance:
            raise KeyError(f"no chance rate for task {task!r}")
        correct = torch.as_tensor(data["correct"], dtype=torch.bool)
        controls = {
            name: torch.as_tensor(data[name], dtype=torch.bool)
            for name in ("blank", "mismatched", "no_tokens")
        }
        rates = {name: float(c.float().mean()) for name, c in controls.items()}
        intervals = {name: wilson_interval(int(c.sum()), len(c)) for name, c in controls.items()}
        within = {name: intervals[name][0] <= chance[task] for name in controls}
        best_name = max(rates, key=rates.get)
        best = controls[best_name]
        correct_rate = float(correct.float().mean())
        results[task] = {
            "correct": correct_rate,
            "chance": chance[task],
            "controls": rates,
            "control_intervals": {name: list(iv) for name, iv in intervals.items()},
            "control_within_chance": within,
            "controls_within_chance": all(within.values()),
            "best_control": best_name,
            "vision_minus_blind": correct_rate - rates[best_name],
            "mcnemar": mcnemar(correct, best),
            "bootstrap": paired_bootstrap(
                (correct.float() - best.float()).numpy(), n=10000, seed=0
            ),
        }
        if "untrained" in data:
            results[task]["untrained"] = float(
                torch.as_tensor(data["untrained"], dtype=torch.bool).float().mean()
            )
    return results


def g1_pass(results: dict[str, dict]) -> bool:
    """G1 pass for one seed: at least 2 of 3 tasks with the frozen criteria."""

    passing = [
        task
        for task, r in results.items()
        if r["vision_minus_blind"] >= 0.30
        and r["mcnemar"]["p"] < 0.01
        and r["bootstrap"]["lo"] > 0.0
        and r["controls_within_chance"]
    ]
    return len(passing) >= 2
