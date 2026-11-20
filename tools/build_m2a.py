#!/usr/bin/env python3
"""Build the M2a corpus stage by stage, resumably.

Each stage reads the previous stage's JSONL and writes its own, so a crash costs
one stage and any stage can be re-run alone.

    python tools/build_m2a.py index     --annotations instances_train2017.json
    python tools/build_m2a.py filter
    python tools/build_m2a.py fetch     --cache artifacts/m2a_images
    python tools/build_m2a.py annotate  --model kilo/nvidia/nemotron-3.5-super-vl:free
    python tools/build_m2a.py manifest

Every stage prints how many records came in, how many were kept and how many were
dropped, so a licence drop is a number rather than an absence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sightline.m2a.coco_index import read_annotations
from sightline.m2a.images import content_hash, dedupe, dhash, fetch
from sightline.m2a.licence import classify
from sightline.m2a.manifest import manifest, splits

COCO_IMAGE_URL = "http://images.cocodataset.org/train2017/"
DEFAULT_ROOT = Path("experiments/m2a_build")
DEFAULT_ANNOTATOR = "kilo/nvidia/nemotron-3.5-super-vl:free"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def report(stage: str, incoming: int, kept: int, dropped: int, **extra) -> None:
    print(f"[{stage}] in={incoming} kept={kept} dropped={dropped} {extra}", flush=True)


def cmd_index(args: argparse.Namespace) -> int:
    records, table = read_annotations(Path(args.annotations))
    rows = [{**r, "url": COCO_IMAGE_URL + r["file_name"]} for r in records]
    write_jsonl(args.root / "index.jsonl", rows)
    print(f"[index] images={len(rows)} licences={len(table)}")
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    incoming = read_jsonl(args.root / "index.jsonl")
    kept, dropped = classify(incoming)
    write_jsonl(args.root / "filtered.jsonl", kept)
    write_jsonl(args.root / "dropped.jsonl", dropped)
    report("filter", len(incoming), len(kept), len(dropped))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    incoming = read_jsonl(args.root / "filtered.jsonl")
    cache = Path(args.cache)
    kept, dropped = [], []
    for record in incoming:
        try:
            path = fetch(record, cache, _get)
            kept.append({**record, "path": str(path), "sha256": content_hash(path)})
        except Exception as exc:
            dropped.append({**record, "drop_reason": f"fetch-failed: {type(exc).__name__}"})
    hashes = {i: (r["sha256"], dhash(Path(r["path"]))) for i, r in enumerate(kept)}
    keyed = [{**r, "id": i} for i, r in enumerate(kept)]
    deduped, dupes = dedupe(keyed, hashes)
    write_jsonl(args.root / "fetched.jsonl", deduped)
    report("fetch", len(incoming), len(deduped), len(dropped) + len(dupes), duplicates=len(dupes))
    return 0


def _get(url: str) -> bytes:
    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def cmd_annotate(args: argparse.Namespace) -> int:
    from sightline.m2a.annotate import annotate_one

    incoming = read_jsonl(args.root / "fetched.jsonl")
    done = {row["id"] for row in read_jsonl(args.root / "conversations.jsonl")}
    pending = [r for r in incoming if r["id"] not in done]

    out = args.root / "conversations.jsonl"
    kept = 0
    for record in pending:
        result = annotate_one(args.model, Path(record["path"]), _call)
        row = {k: v for k, v in record.items() if k != "path"}
        row.update(result)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        kept += 1
        if kept % 20 == 0:
            print(f"[annotate] {kept}/{len(pending)}", flush=True)
    report("annotate", len(incoming), kept, 0, already_done=len(done))
    return 0


def _call(model: str, image_path: Path, prompt: str):
    """Annotate through the opencode CLI, returning text and token usage."""

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
    texts, tokens = [], None
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


def cmd_manifest(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.root / "conversations.jsonl")
    result = manifest(args.root / "conversations.jsonl")
    for name, part in splits(rows, seed=args.seed).items():
        write_jsonl(args.root / f"split_{name}.jsonl", part)
        result.setdefault("splits", {})[name] = len(part)
    (args.root / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index")
    p.add_argument("--annotations", required=True)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("filter")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("fetch")
    p.add_argument("--cache", default="artifacts/m2a_images")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("annotate")
    p.add_argument("--model", default=DEFAULT_ANNOTATOR)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("manifest")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_manifest)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
