from __future__ import annotations

import pytest
import torch

from sightline.evaluate import g1_pass, g1_report, mcnemar, paired_bootstrap, wilson_interval

CHANCE = {"color": 1 / 8, "shape": 1 / 4, "count": 1 / 5}


def test_wilson_interval_brackets_the_observed_rate() -> None:
    lo, hi = wilson_interval(83, 500)
    assert lo < 0.166 < hi
    assert lo <= 83 / 500 <= hi


def test_wilson_interval_is_wide_at_small_n() -> None:
    lo_small, hi_small = wilson_interval(1, 10)
    lo_big, hi_big = wilson_interval(50, 500)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_first_line_drops_trailing_markdown() -> None:
    from sightline.evaluate import first_line_answer

    assert first_line_answer('diamond\n```json\n{\n  "') == "diamond"


def test_first_line_rejects_a_negated_answer() -> None:
    from sightline.evaluate import first_line_answer

    assert first_line_answer("not red") == "not red"


def test_first_line_handles_an_empty_generation() -> None:
    from sightline.evaluate import first_line_answer

    assert first_line_answer("") == ""
    assert first_line_answer("   \n\n") == ""


def test_first_line_skips_leading_blank_lines() -> None:
    from sightline.evaluate import first_line_answer

    assert first_line_answer("\n\nred") == "red"
    assert first_line_answer("  \n  red  \nmore") == "red"


def test_control_below_chance_passes() -> None:
    lo, _ = wilson_interval(0, 500)
    assert lo <= CHANCE["color"]


def test_control_above_chance_fails_g1() -> None:
    per_task = {
        "color": {
            "correct": [True] * 450 + [False] * 50,
            "blank": [True] * 300 + [False] * 200,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
        "shape": {
            "correct": [True] * 450 + [False] * 50,
            "blank": [True] * 300 + [False] * 200,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
        "count": {
            "correct": [True] * 450 + [False] * 50,
            "blank": [True] * 300 + [False] * 200,
            "mismatched": [False] * 500,
            "no_tokens": [False] * 500,
        },
    }

    report = g1_report(per_task, CHANCE)

    assert report["color"]["chance"] == CHANCE["color"]
    assert report["color"]["control_within_chance"]["blank"] is False
    assert report["color"]["controls_within_chance"] is False
    assert report["color"]["vision_minus_blind"] == pytest.approx(0.30)
    assert g1_pass(report) is False


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
    report = g1_report(per_task, CHANCE)
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
    report = g1_report(per_task, CHANCE)
    assert g1_pass(report) is False
