from __future__ import annotations

import pytest
import torch

from sightline.evaluate import g1_pass, g1_report, mcnemar, paired_bootstrap


def test_mcnemar_exact() -> None:
    a = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    b = torch.tensor([0, 0, 0, 0], dtype=torch.bool)
    result = mcnemar(a, b)
    assert result["b"] == 0
    assert result["c"] == 3
    assert result["p"] == pytest.approx(0.25)


def test_paired_bootstrap_contains_mean() -> None:
    result = paired_bootstrap([1, 2, 3, 4, 5], n=2000, seed=0)
    assert result["lo"] <= 3.0 <= result["hi"]


def test_g1_report_and_pass() -> None:
    per_task = {
        "color": {
            "correct": [True] * 450 + [False] * 50,
            "blank": [False] * 500,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
        "shape": {
            "correct": [True] * 450 + [False] * 50,
            "blank": [False] * 500,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
        "count": {
            "correct": [True] * 200 + [False] * 300,
            "blank": [False] * 500,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
    }
    report = g1_report(per_task)
    assert report["color"]["vision_minus_blind"] == pytest.approx(0.9)
    assert report["count"]["vision_minus_blind"] == pytest.approx(0.4)
    assert report["color"]["mcnemar"]["p"] < 0.01
    assert g1_pass(report) is True


def test_g1_pass_false_when_only_one_task() -> None:
    per_task = {
        "color": {
            "correct": [True] * 450 + [False] * 50,
            "blank": [False] * 500,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
        "shape": {
            "correct": [False] * 500,
            "blank": [False] * 500,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
        "count": {
            "correct": [False] * 500,
            "blank": [False] * 500,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
    }
    report = g1_report(per_task)
    assert g1_pass(report) is False
