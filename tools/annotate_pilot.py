#!/usr/bin/env python3
"""Annotator pilot: measure the contract, the token cost and the grounding.

M2a needs multi-turn image QA in Baseten's shape. Before committing to tens of
thousands of conversations, this measures three things on a small sample:

- **contract**: does the annotator return parseable JSON in the requested shape?
- **cost**: how many tokens does one conversation actually consume?
- **grounding**: are the assistant answers true of the image?

It runs on any machine with the ``opencode`` CLI; no GPU is needed. The sample
is drawn from the M1 synthetic set, which has known answers, so grounding can be
checked automatically. That is a limitation: M1 images are rendered, not
photographs, so this measures the *contract and cost*, not M2a data quality.

    python tools/annotate_pilot.py --n 30 --model kilo/nvidia/nemotron-3.5-super-vl:free
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from sightline.data.m1_synthetic import generate_m1_dataset

NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def answer_forms(answer: str) -> list[str]:
    """Acceptable spellings of an answer.

    Count answers are digits in the dataset but words in conversation, so both
    forms count as grounded. Checking only the digit scored every correct
    `'Two.'` as a miss.
    """

    forms = [answer]
    if answer.isdigit() and int(answer) in NUMBER_WORDS:
        forms.append(NUMBER_WORDS[int(answer)])
    return [f for f in forms if f]


PROMPT = """Look at the attached image and write a short conversation about it.

Rules:
- 2 or 3 user turns.
- Every user turn asks something that can be answered from the image alone.
- Every assistant turn answers in a few words.
- Do not mention that this is a test or an instruction.

Reply with ONLY a JSON object, no prose and no code fence, in exactly this shape:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}"""


class RateLimited(RuntimeError):
    """The provider refused the call; the caller should stop and resume later."""


def call_model(
    model: str, image_path: Path, prompt: str, attempts: int = 3
) -> tuple[str, dict | None]:
    """Run one annotation and return (raw text, token usage).

    Raises ``RateLimited`` when the provider refuses rather than returning text, so
    the caller can stop cleanly and resume from the last saved record. Retries
    with a short backoff cover transient failures.
    """

    result = None
    for attempt in range(attempts):
        result = subprocess.run(
            [
                "opencode",
                "run",
                "--model",
                model,
                "--file",
                str(image_path),
                "--format",
                "json",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode == 0:
            break
        if attempt == attempts - 1:
            raise RateLimited(
                f"annotator failed after {attempts} attempts: "
                f"{(result.stderr or result.stdout or '').strip()[:300]}"
            )
    assert result is not None
    texts: list[str] = []
    tokens: dict | None = None
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            texts.append(event["part"]["text"])
        elif event.get("type") == "step_finish":
            tokens = event["part"].get("tokens")
    return "\n".join(texts).strip(), tokens


def parse_messages(raw: str) -> list[dict] | None:
    """Extract the JSON object, tolerating code fences and surrounding prose."""

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    return [m for m in messages if isinstance(m, dict) and "content" in m]


def load_done(path: Path) -> dict[int, dict]:
    """Records already written, keyed by index, so a restart resumes."""

    if not path.exists():
        return {}
    done = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[int(record["index"])] = record
    return done


def summarise(records: list[dict], model: str) -> dict:
    n = len(records)
    parsed = [r for r in records if r.get("parsed")]
    grounded = [r for r in parsed if r.get("grounded")]
    tokens_in = [r["tokens_in"] for r in records if r.get("tokens_in")]
    return {
        "model": model,
        "n": n,
        "parse_rate": round(len(parsed) / n, 3) if n else 0.0,
        "grounded_of_parsed": round(len(grounded) / len(parsed), 3) if parsed else 0.0,
        "mean_turns": round(sum(r.get("turns", 0) for r in parsed) / len(parsed), 2)
        if parsed
        else 0.0,
        "tokens_in_mean": round(sum(tokens_in) / len(tokens_in)) if tokens_in else None,
        "tokens_in_total": sum(tokens_in) if tokens_in else None,
        "projected_40k_tokens_in": (sum(tokens_in) * 40000 // n) if tokens_in else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="images to annotate")
    parser.add_argument(
        "--model", default="kilo/nvidia/nemotron-3.5-super-vl:free", help="annotator model"
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/m2a_pilot"))
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="annotate the first image N times and report the token counts, to test determinism",
    )
    args = parser.parse_args()

    dataset = generate_m1_dataset(seed=0)
    per_task = max(1, args.n // 3)
    picked = []
    for task in ("color", "shape", "count"):
        examples = [e for e in dataset.train if e.task == task]
        step = max(1, len(examples) // per_task)
        picked += examples[::step][:per_task]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "annotations.jsonl"
    records = load_done(path)
    image_dir = Path(tempfile.mkdtemp(prefix="m2a_pilot_"))
    print(f"resuming: {len(records)} of {len(picked)} already done", flush=True)

    import hashlib

    try:
        with path.open("a", encoding="utf-8") as handle:
            for i, example in enumerate(picked):
                if i in records:
                    continue
                image_path = image_dir / f"{i:03d}.png"
                example.image.save(image_path)
                raw, tokens = call_model(args.model, image_path, PROMPT)
                messages = parse_messages(raw)
                assistant = [m["content"] for m in (messages or []) if m.get("role") == "assistant"]
                record = {
                    "index": i,
                    "task": example.task,
                    "expected": example.answer,
                    "image_size": list(example.image.size),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()[:16],
                    "parsed": messages is not None,
                    "turns": len(messages) // 2 if messages else 0,
                    "grounded": any(
                        form in str(a).lower()
                        for a in assistant
                        for form in answer_forms(example.answer)
                    ),
                    "tokens_in": (tokens or {}).get("input"),
                    "tokens_out": (tokens or {}).get("output"),
                    "raw": raw[:400],
                }
                records[i] = record
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(json.dumps(record), flush=True)
    except RateLimited as exc:
        print(f"STOPPED: {exc}", flush=True)
        print("partial results are on disk; re-run the same command to resume", flush=True)
        summary = summarise([records[k] for k in sorted(records)], args.model)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        return 2
    except KeyboardInterrupt:
        print("interrupted; partial results are on disk", flush=True)
        summary = summarise([records[k] for k in sorted(records)], args.model)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        return 130

    repeat_counts = []
    if args.repeat > 1 and 0 in records:
        repeat_image = image_dir / "000.png"
        for _ in range(args.repeat):
            _, tokens = call_model(args.model, repeat_image, PROMPT)
            repeat_counts.append((tokens or {}).get("input"))

    summary = summarise([records[k] for k in sorted(records)], args.model)
    if repeat_counts:
        summary["token_repeat"] = repeat_counts
        summary["token_repeat_deterministic"] = len(set(repeat_counts)) == 1
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
