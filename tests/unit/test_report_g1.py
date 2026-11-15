from __future__ import annotations

import json
from pathlib import Path

from sightline.report_g1 import build_report, load_g1_event, summarise


def passing_report(correct: float = 0.9) -> dict:
    return {
        task: {
            "correct": correct,
            "chance": 1 / 8,
            "controls": {"blank": 0.0, "mismatched": 0.0, "no_tokens": 0.0},
            "control_intervals": {"blank": [0.0, 0.01]},
            "control_within_chance": {"blank": True, "mismatched": True, "no_tokens": True},
            "controls_within_chance": True,
            "best_control": "blank",
            "vision_minus_blind": correct,
            "mcnemar": {"b": 0, "c": 450, "p": 1e-10},
            "bootstrap": {"mean": correct, "lo": correct - 0.05, "hi": correct + 0.05},
        }
        for task in ("color", "shape", "count")
    }


def write_seed(root: Path, seed: int, report: dict) -> None:
    directory = root / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "m1b_probe", "seed": seed}) + "\n")
        handle.write(
            json.dumps({"event": "g1", "seed": seed, "g1_pass": True, "report": report}) + "\n"
        )
        handle.write(json.dumps({"event": "m1b", "seed": seed}) + "\n")


def test_load_g1_event_takes_the_last_one(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"event": "g1", "seed": 0, "report": {"old": True}},
                {"event": "m1b", "seed": 0},
                {"event": "g1", "seed": 0, "report": {"new": True}},
            )
        )
        + "\n"
    )

    assert load_g1_event(path)["report"] == {"new": True}


def test_load_g1_event_missing_file(tmp_path: Path) -> None:
    assert load_g1_event(tmp_path / "nope.jsonl") is None


def test_summarise_passes_when_every_seed_passes() -> None:
    seeds = [0, 1, 2]
    events = {seed: {"report": passing_report()} for seed in seeds}

    summary = summarise(events, seeds)

    assert summary["g1_pass"] is True
    assert summary["missing"] == []
    assert all(summary["per_seed"][seed]["pass"] for seed in seeds)


def test_summarise_fails_when_a_seed_is_missing() -> None:
    events = {0: {"report": passing_report()}, 1: {"report": passing_report()}}

    summary = summarise(events, [0, 1, 2])

    assert summary["g1_pass"] is False
    assert summary["missing"] == [2]


def test_summarise_fails_when_a_seed_is_below_threshold() -> None:
    seeds = [0, 1, 2]
    events = {seed: {"report": passing_report()} for seed in seeds}
    events[1] = {"report": passing_report(correct=0.05)}

    summary = summarise(events, seeds)

    assert summary["g1_pass"] is False
    assert summary["per_seed"][1]["pass"] is False


def test_build_report_renders_markdown_from_run_records(tmp_path: Path) -> None:
    for seed in (0, 1, 2):
        write_seed(tmp_path, seed, passing_report())

    markdown, summary = build_report(tmp_path, [0, 1, 2])

    assert summary["g1_pass"] is True
    assert "**G1: PASS**" in markdown
    assert "| 0 | pass |" in markdown
    assert "seed0/metrics.jsonl" in markdown
    assert "| 0 | color | 0.900 |" in markdown


def test_build_report_renders_missing_seeds_honestly(tmp_path: Path) -> None:
    write_seed(tmp_path, 0, passing_report())

    markdown, summary = build_report(tmp_path, [0, 1, 2])

    assert summary["g1_pass"] is False
    assert "**G1: FAIL or incomplete**" in markdown
    assert "Seeds with no G1 event yet: 1, 2" in markdown
