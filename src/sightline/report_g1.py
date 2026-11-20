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


def load_g1_event(
    path: Path, split: str | None = None, parser: str | None = None
) -> dict[str, Any] | None:
    """Return the ``g1`` event matching the requested split and parser, or None.

    A run directory accumulates one event per evaluation, so "the last one"
    silently reports whichever evaluation ran most recently — a val run after a
    test run, or a first-line re-score after the exact run that the G1 verdict
    rests on. Both fields are therefore matched explicitly.

    Events written before ``parser`` was logged have no such field; they are
    used only when no event carries one at all.
    """

    if not path.exists():
        return None
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "g1":
                events.append(record)
    if not events:
        return None

    candidates = events
    if split is not None:
        matching = [r for r in candidates if r.get("eval_split") == split]
        if not any("eval_split" in r for r in candidates):
            pass  # legacy records carry no split; fall through to the last one
        elif not matching:
            return None
        else:
            candidates = matching
    if parser is not None:
        matching = [r for r in candidates if r.get("parser") == parser]
        if not any("parser" in r for r in candidates):
            pass  # legacy records carry no parser
        elif not matching:
            return None
        else:
            candidates = matching
    return candidates[-1]


def load_seed(
    root: Path, seed: int, split: str | None = None, parser: str | None = None
) -> dict[str, Any] | None:
    """Read a seed's ``g1`` event from ``<root>/seed<seed>/metrics.jsonl``."""

    return load_g1_event(root / f"seed{seed}" / "metrics.jsonl", split, parser)


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


def render_markdown(
    summary: dict[str, Any],
    sources: dict[int, Path],
    split: str | None = None,
    parser: str | None = None,
) -> str:
    lines = [
        "# G1 report: M1b trained projector on held-out M1 data",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Evaluation split: **{split}**" if split else "Evaluation split: last recorded",
        "",
        f"Answer parser: **{parser}**" if parser else "Answer parser: last recorded",
        "",
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


def build_report(
    root: Path, seeds: Sequence[int], split: str | None = None, parser: str | None = None
) -> tuple[str, dict[int, Path]]:
    events = {seed: load_seed(root, seed, split, parser) for seed in seeds}
    summary = summarise(events, seeds)
    sources = {
        seed: root / f"seed{seed}" / "metrics.jsonl" for seed in seeds if events[seed] is not None
    }
    return render_markdown(summary, sources, split, parser), summary
