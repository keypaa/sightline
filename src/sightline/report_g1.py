"""G1 report from the M1b run records.

Reads the ``g1`` metric event that ``tools/run_m1.py`` appends to each seed's
``metrics.jsonl`` and renders ``reports/g1.md``: per-seed verdicts, a per-task
table with the frozen thresholds applied, and the overall G1 verdict.

The thresholds are not restated here; they live in ``sightline.evaluate`` and
``DECISIONS/0032``, so the report cannot drift from the code that produced it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sightline.evaluate import g1_pass

TASKS = ("color", "shape", "count")


def load_g1_event(path: Path) -> dict[str, Any] | None:
    """Return the last ``g1`` event in a metrics file, or None."""

    if not path.exists():
        return None
    found = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "g1":
                found = record
    return found


def load_seed(root: Path, seed: int) -> dict[str, Any] | None:
    """Read the last ``g1`` event for a seed from ``<root>/seed<seed>/metrics.jsonl``."""

    return load_g1_event(root / f"seed{seed}" / "metrics.jsonl")


def summarise(events: dict[int, dict[str, Any] | None], seeds: Sequence[int]) -> dict[str, Any]:
    """Per-seed verdicts plus the overall G1 verdict.

    Overall G1 passes only when every seed in ``seeds`` has a recorded event and
    every one of them passes, which is how DECISIONS 0032 states "all 3 seeds
    positive".
    """

    per_seed = {}
    for seed in seeds:
        event = events.get(seed)
        if event is None:
            per_seed[seed] = {"status": "missing"}
            continue
        report = event.get("report", {})
        per_seed[seed] = {
            "status": "recorded",
            "pass": g1_pass(report),
            "tasks": {
                task: {
                    "correct": report[task]["correct"],
                    "chance": report[task]["chance"],
                    "controls": report[task]["controls"],
                    "vision_minus_blind": report[task]["vision_minus_blind"],
                    "mcnemar_p": report[task]["mcnemar"]["p"],
                    "bootstrap_lo": report[task]["bootstrap"]["lo"],
                    "controls_within_chance": report[task]["controls_within_chance"],
                    "untrained": report[task].get("untrained"),
                }
                for task in TASKS
                if task in report
            },
        }

    recorded = [seed for seed in seeds if per_seed[seed]["status"] == "recorded"]
    overall = (
        bool(recorded)
        and len(recorded) == len(list(seeds))
        and all(per_seed[seed]["pass"] for seed in seeds)
    )
    return {
        "seeds": list(seeds),
        "per_seed": per_seed,
        "missing": [seed for seed in seeds if per_seed[seed]["status"] == "missing"],
        "g1_pass": overall,
    }


def render_markdown(summary: dict[str, Any], sources: dict[int, Path]) -> str:
    lines = [
        "# G1 report: M1b trained projector on held-out M1 data",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Thresholds are frozen in `DECISIONS/0032` and implemented in",
        "`sightline.evaluate`: correct minus the best of blank/mismatched/no-tokens",
        ">= 0.30 on at least 2 of 3 tasks, McNemar p < 0.01, bootstrap 95% lower",
        "bound > 0, controls not significantly above chance, and every seed passing.",
        "",
        "## Verdict",
        "",
        f"**G1: {'PASS' if summary['g1_pass'] else 'FAIL or incomplete'}**",
        "",
        "| seed | verdict | source |",
        "| --- | --- | --- |",
    ]
    for seed in summary["seeds"]:
        entry = summary["per_seed"][seed]
        verdict = (
            entry.get("status")
            if entry["status"] == "missing"
            else ("pass" if entry["pass"] else "fail")
        )
        source = sources.get(seed)
        lines.append(
            f"| {seed} | {verdict} | `{source}` |"
            if source
            else f"| {seed} | {verdict} | missing |"
        )

    if summary["missing"]:
        lines += ["", f"Seeds with no G1 event yet: {', '.join(map(str, summary['missing']))}."]

    lines += ["", "## Per task", ""]
    header = (
        "| seed | task | correct | chance | blank | mismatched | no-tokens | "
        "untrained | vision - best control | McNemar p | bootstrap 95% |"
    )
    lines += [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for seed in summary["seeds"]:
        entry = summary["per_seed"][seed]
        if entry["status"] != "recorded":
            continue
        for task, row in entry["tasks"].items():
            controls = row["controls"]
            untrained = "-" if row["untrained"] is None else f"{row['untrained']:.3f}"
            lines.append(
                f"| {seed} | {task} | {row['correct']:.3f} | {row['chance']:.3f} | "
                f"{controls['blank']:.3f} | {controls['mismatched']:.3f} | "
                f"{controls['no_tokens']:.3f} | {untrained} | "
                f"{row['vision_minus_blind']:+.3f} | {row['mcnemar_p']:.2e} | "
                f"{row['bootstrap_lo']:+.3f} |"
            )

    lines += [
        "",
        "The untrained-projector column is a diagnostic; it is not one of the",
        "controls the pass criteria minimise over.",
        "",
    ]
    return "\n".join(lines)


def build_report(root: Path, seeds: Sequence[int]) -> tuple[str, dict[int, Path]]:
    events = {seed: load_seed(root, seed) for seed in seeds}
    summary = summarise(events, seeds)
    sources = {
        seed: root / f"seed{seed}" / "metrics.jsonl" for seed in seeds if events[seed] is not None
    }
    return render_markdown(summary, sources), summary
