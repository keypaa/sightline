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
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sightline.m2a.coco_index import read_annotations
from sightline.m2a.images import content_hash, dedupe, dhash, fetch
from sightline.m2a.licence import classify
from sightline.m2a.manifest import manifest, splits, usable_records
from sightline.m2a.sample import sample_subset

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
    if args.limit and args.limit < len(incoming):
        incoming = sample_subset(list(incoming), args.limit, seed=args.seed)
        print(
            f"[fetch] sampled {len(incoming)} of {len(read_jsonl(args.root / 'filtered.jsonl'))}",
            flush=True,
        )
    cache = Path(args.cache)
    total = len(incoming)
    done = 0

    def work(record):
        try:
            path = fetch(record, cache, _get)
            return {**record, "path": str(path), "sha256": content_hash(path)}, None
        except Exception as exc:
            return record, f"fetch-failed: {type(exc).__name__}"

    kept, dropped = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, failure in pool.map(work, incoming):
            if failure:
                dropped.append({**row, "drop_reason": failure})
            else:
                kept.append(row)
            done += 1
            if done % 500 == 0:
                print(f"[fetch] {done}/{total}", flush=True)

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
    from sightline.m2a.annotate import annotate_one, pending_records
    from sightline.m2a.transport import cli_call, retry_loop

    incoming = read_jsonl(args.root / "fetched.jsonl")
    done_rows = read_jsonl(args.root / "conversations.jsonl")
    pending_all = pending_records(incoming, done_rows)
    pending = pending_all[: args.limit] if args.limit else pending_all

    out = args.root / "conversations.jsonl"
    lock = threading.Lock()
    finished = 0
    unparsed = 0

    def work(record):
        result = annotate_one(args.model, Path(record["path"]), cli_call)
        row = {k: v for k, v in record.items() if k != "path"}
        row.update(result)
        return row

    def on_row(row):
        nonlocal finished, unparsed
        with lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            finished += 1
            unparsed += int(not row["parsed"])
            if finished % 50 == 0:
                print(f"[annotate] {finished}/{len(pending)} unparsed={unparsed}", flush=True)

    completed, waits = retry_loop(
        pending,
        work,
        on_row,
        workers=args.workers,
        empty_limit=args.empty_limit,
        wait_seconds=args.wait_seconds,
        max_waits=args.max_waits,
    )

    report(
        "annotate",
        len(incoming),
        completed,
        unparsed,
        already_done=len(incoming) - len(pending_all),
        attempted=len(pending),
        waits=waits,
        workers=args.workers,
        empty_limit=args.empty_limit,
    )
    if completed < len(pending):
        print("[annotate] partial results kept; re-run this stage to resume", flush=True)
        return 1
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.root / "conversations.jsonl")
    result = manifest(args.root / "conversations.jsonl")
    for name, part in splits(usable_records(rows), seed=args.seed).items():
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
    p.add_argument("--workers", type=int, default=1, help="concurrent downloads")
    p.add_argument("--limit", type=int, default=0, help="keep only N images, sampled with --seed")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("annotate")
    p.add_argument("--model", default=DEFAULT_ANNOTATOR)
    p.add_argument("--workers", type=int, default=1, help="concurrent opencode calls")
    p.add_argument("--limit", type=int, default=0, help="annotate at most N pending images")
    p.add_argument(
        "--empty-limit",
        type=int,
        default=10,
        help="stop after this many consecutive responses with no token usage (DECISIONS 0044)",
    )
    p.add_argument(
        "--wait-seconds",
        type=int,
        default=3600,
        help="seconds to wait after a 429 before resuming (DECISIONS 0045)",
    )
    p.add_argument(
        "--max-waits",
        type=int,
        default=120,
        help="maximum quota windows to wait before stopping (DECISIONS 0045)",
    )
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("manifest")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_manifest)

    args = parser.parse_args()
    args.seed = getattr(args, "seed", 0)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
