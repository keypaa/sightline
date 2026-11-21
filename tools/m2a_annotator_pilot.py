#!/usr/bin/env python3
"""Local annotator pilot: measure parse rate, grounded rate and throughput.

Frozen bar (DECISIONS/0040): parse rate >= 0.95 and grounded rate >= 0.90.

Run this on the box serving the annotator with vLLM; the only network call is to
the OpenAI-compatible endpoint at ``--host``. The sample comes from the clean COCO
fetch at ``experiments/m2a_iter20k/fetched.jsonl``; if that file is absent the run
stops rather than inventing images. One row per sampled image is written to
``pilot.jsonl``, carrying the ``grounded`` flag computed here from the row's
``expected`` answer with the shared ``answer_forms`` rule.

COCO fetch records carry no ground-truth answer. ``expected`` is read from the
record when present; without it a row is marked ungrounded, so a grounded rate of
zero means the bar was *not measurable*, not that the annotator was caught lying.
The report says so explicitly and the exit code is 1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sightline.m2a.annotate import annotate_one, prompt_hash
from sightline.m2a.local_transport import LocalVLLM
from sightline.m2a.prompt import answer_forms
from sightline.m2a.provenance import annotator_fields
from sightline.m2a.sample import sample_subset
from sightline.record import git_state

PARSE_BAR = 0.95
GROUNDED_BAR = 0.90
SEED = 0
DEFAULT_FETCHED = Path("experiments/m2a_iter20k/fetched.jsonl")
DEFAULT_OUT = Path("experiments/m2a_pilot")
DEFAULT_REPORT = Path("reports/m2a-annotator-pilot-local.md")


def grounded_in(messages: list[dict[str, Any]] | None, expected: str) -> bool:
    """True when an answer form appears in any assistant message, lower-cased.

    This is the rule the CLI-era pilot uses; it lives here so the pilot has one
    copy and the annotator returns no half-computed ``grounded`` field.
    """

    assistant = [
        str(message["content"]).lower()
        for message in (messages or [])
        if message.get("role") == "assistant" and "content" in message
    ]
    return any(form in text for text in assistant for form in answer_forms(expected))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The two frozen rates and mean token cost, over every row."""

    n = len(rows)
    parsed = sum(1 for row in rows if row.get("parsed"))
    grounded = sum(1 for row in rows if row.get("grounded"))
    tokens_in = [row["tokens_in"] for row in rows if row.get("tokens_in") is not None]
    tokens_out = [row["tokens_out"] for row in rows if row.get("tokens_out") is not None]
    return {
        "n": n,
        "parse_rate": parsed / n if n else 0.0,
        "grounded_rate": grounded / n if n else 0.0,
        "tokens_in_mean": sum(tokens_in) / len(tokens_in) if tokens_in else None,
        "tokens_out_mean": sum(tokens_out) / len(tokens_out) if tokens_out else None,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_report(path: Path, summary: dict[str, Any]) -> None:
    """Write the human-readable run record; the metrics stay in pilot.jsonl."""

    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# M2a annotator pilot (local vLLM)",
        "",
        f"- model: `{summary['model']}` at revision `{summary['revision']}`",
        f"- engine: vllm {summary['engine_version']}, quantisation {summary['quantization']}",
        f"- prompt_sha256: `{summary['prompt_sha256']}`",
        f"- chat_template_sha256: `{summary['chat_template_sha256']}`",
        f"- seed: {summary['seed']}, images: {summary['n']}",
        "",
        "| measure | observed | frozen bar | passes |",
        "| --- | --- | --- | --- |",
        f"| parse rate | {fmt(summary['parse_rate'])} | {PARSE_BAR:.2f} | "
        f"{'yes' if summary['passes_parse_bar'] else 'no'} |",
        f"| grounded rate | {fmt(summary['grounded_rate'])} | {GROUNDED_BAR:.2f} | "
        f"{'yes' if summary['passes_grounded_bar'] else 'no'} |",
        "",
        f"- tokens in / conversation: {fmt(summary['tokens_in_mean'])}",
        f"- tokens out / conversation: {fmt(summary['tokens_out_mean'])}",
        f"- conversations/hour: {fmt(summary['conversations_per_hour'])}",
        f"- wall seconds: {summary['wall_seconds']}",
        f"- rows with a recorded `expected`: {summary['n_with_expected']} of {summary['n']}",
        f"- gate: {'PASS' if summary['passes'] else 'FAIL'}",
        "",
        "Grounding is computed by `grounded_in`: an answer form from `answer_forms(expected)`",
        "must appear in an assistant message, lower-cased. The prompt hash identifies the",
        "annotator contract (DECISIONS/0040).",
        "",
        "- [U] COCO fetch records carry no `expected` answer. When none of the sampled rows",
        "  has one, the grounded bar is not measurable: the rates are recorded honestly but",
        "  the gate fails. Supply `expected` on the sampled records before treating this",
        "  number as an annotator measurement.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=30, help="images to annotate")
    parser.add_argument("--host", default="http://localhost:8000/v1", help="vLLM base URL")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8", help="annotator model id")
    parser.add_argument("--revision", required=True, help="model revision, pinned in the row")
    parser.add_argument("--quantization", required=True, help="weight quantisation")
    parser.add_argument("--engine-version", required=True, help="vLLM version")
    parser.add_argument(
        "--chat-template-sha256", required=True, help="hash of the served chat template"
    )
    parser.add_argument("--fetched", type=Path, default=DEFAULT_FETCHED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    if not args.fetched.exists():
        print(f"no fetched corpus at {args.fetched}; refusing to invent pilot images", flush=True)
        return 2
    records = read_jsonl(args.fetched)
    if not records:
        print(f"fetched corpus at {args.fetched} is empty; nothing to pilot", flush=True)
        return 2

    picked = sample_subset(records, args.images, seed=SEED)
    annotator = annotator_fields(
        model=args.model,
        revision=args.revision,
        quantization=args.quantization,
        engine="vllm",
        engine_version=args.engine_version,
        prompt_sha256=prompt_hash(),
        chat_template_sha256=args.chat_template_sha256,
        seed=SEED,
        order_seed=SEED,
        temperature=0.0,
    )
    call = LocalVLLM(args.model, base_url=args.host, temperature=0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with (args.out / "pilot.jsonl").open("w", encoding="utf-8") as handle:
        for record in picked:
            expected = record.get("expected")
            row = annotate_one(args.model, Path(record["path"]), call, annotator=annotator)
            row["id"] = record.get("id")
            row["file_name"] = record.get("file_name")
            row["expected"] = expected
            row["grounded"] = (
                grounded_in(row["messages"], str(expected)) if expected is not None else False
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
    elapsed = time.perf_counter() - started

    summary = summarise(rows)
    summary.update(
        model=args.model,
        revision=args.revision,
        quantization=args.quantization,
        engine_version=args.engine_version,
        prompt_sha256=prompt_hash(),
        chat_template_sha256=args.chat_template_sha256,
        seed=SEED,
        host=args.host,
        wall_seconds=round(elapsed, 3),
        conversations_per_hour=round(len(rows) * 3600 / elapsed, 1) if elapsed > 0 else None,
        n_with_expected=sum(1 for row in rows if row.get("expected") is not None),
    )
    summary["passes_parse_bar"] = summary["parse_rate"] >= PARSE_BAR
    summary["grounding_measurable"] = summary["n_with_expected"] > 0
    summary["passes_grounded_bar"] = (
        summary["grounding_measurable"] and summary["grounded_rate"] >= GROUNDED_BAR
    )
    summary["passes"] = summary["passes_parse_bar"] and summary["passes_grounded_bar"]

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run_record = {
        "kind": "m2a-annotator-pilot",
        "git": git_state(Path(__file__).resolve().parents[1]),
        "config": {
            "model": args.model,
            "revision": args.revision,
            "quantization": args.quantization,
            "engine_version": args.engine_version,
            "host": args.host,
            "images": args.images,
            "seed": SEED,
        },
        "pilot_jsonl": str(args.out / "pilot.jsonl"),
        "summary": summary,
    }
    (args.out / "run_record.json").write_text(json.dumps(run_record, indent=2) + "\n")
    write_report(args.report, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
