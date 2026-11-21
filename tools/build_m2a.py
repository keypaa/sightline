#!/usr/bin/env python3
"""Build the M2a corpus stage by stage, resumably.

Each stage reads the previous stage's JSONL and writes its own, so a crash costs
one stage and any stage can be re-run alone.

    python tools/build_m2a.py index     --annotations instances_train2017.json
    python tools/build_m2a.py filter
    python tools/build_m2a.py fetch     --cache artifacts/m2a_images --write-checksums --seed 0
    python tools/build_m2a.py annotate  --revision REV --quantization fp8 --engine-version VER
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
DEFAULT_ANNOTATOR = "Qwen/Qwen3.6-27B-FP8"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
ANNOTATE_ENGINE = "vllm"


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
            return record, f"fetch-failed: {type(exc).__name__}: {exc}"

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
    keyed = [{**r, "id": i, "order_seed": args.seed} for i, r in enumerate(kept)]
    deduped, dupes = dedupe(keyed, hashes)
    if getattr(args, "write_checksums", False):
        # Persist the observed hash so a later re-fetch can detect a changed file.
        deduped = [{**r, "expected_sha256": r["sha256"]} for r in deduped]
    write_jsonl(args.root / "fetched.jsonl", deduped)
    write_jsonl(args.root / "fetch_dropped.jsonl", dropped)
    report(
        "fetch",
        len(incoming),
        len(deduped),
        len(dropped) + len(dupes),
        duplicates=len(dupes),
        bad_images=len(dropped),
    )
    return 0


def _get(url: str) -> bytes:
    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def cmd_annotate(args: argparse.Namespace) -> int:
    from sightline.m2a.annotate import annotate_one, pending_records, prompt_hash
    from sightline.m2a.local_transport import LocalVLLM
    from sightline.m2a.order import apply_order, read_order
    from sightline.m2a.provenance import annotator_fields
    from sightline.m2a.transport import retry_loop

    incoming = read_jsonl(args.root / "fetched.jsonl")
    order_path = args.root / "order.jsonl"
    if order_path.exists():
        # Resume must follow the recorded corpus order, not the order rows happened
        # to be completed in; filter pending only after applying it.
        incoming = apply_order(incoming, read_order(order_path))

    done_rows = read_jsonl(args.root / "conversations.jsonl")
    pending_all = pending_records(incoming, done_rows)
    pending = pending_all[: args.limit] if args.limit else pending_all

    # fetch pins order_seed on each kept row. Read it back rather than assuming the
    # annotate run used the same seed that generated order.jsonl.
    order_seed = incoming[0].get("order_seed", args.seed) if incoming else args.seed

    # Built once: every row in this run carries the same annotator identity, and the
    # record cannot shadow it when the result is merged in.
    annotator = annotator_fields(
        model=args.model,
        revision=args.revision,
        quantization=args.quantization,
        engine=ANNOTATE_ENGINE,
        engine_version=args.engine_version,
        prompt_sha256=prompt_hash(),
        chat_template_sha256=args.chat_template_sha256,
        seed=args.seed,
        order_seed=order_seed,
        temperature=args.temperature,
    )
    # The call-time model wins in LocalVLLM, so pass the same string annotate_one
    # records; the POST payload and the row provenance cannot disagree.
    call = LocalVLLM(args.model, base_url=args.base_url, temperature=args.temperature)

    out = args.root / "conversations.jsonl"
    lock = threading.Lock()
    finished = 0
    unparsed = 0
    failures: list[dict] = []

    def work(record):
        result = annotate_one(args.model, Path(record["path"]), call, annotator=annotator)
        row = {k: v for k, v in record.items() if k != "path"}
        row.update(result)
        return row

    def on_row(row):
        nonlocal finished, unparsed
        with lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if not row["parsed"]:
                # Unparsed rows stay in conversations.jsonl so resume retries them;
                # this file is the per-run audit of what came back unusable.
                failures.append(row)
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
    write_jsonl(args.root / "annotate_failures.jsonl", failures)

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
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for both the --limit sample and the recorded order_seed",
    )
    p.add_argument(
        "--write-checksums",
        action="store_true",
        help="write expected_sha256 into fetched.jsonl so a later re-fetch can verify it",
    )
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("annotate")
    p.add_argument("--model", default=DEFAULT_ANNOTATOR)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM chat endpoint")
    p.add_argument("--revision", default="", help="pinned model revision; empty means unrecorded")
    p.add_argument("--quantization", default="", help="e.g. fp8; empty means unrecorded")
    p.add_argument("--engine-version", default="", help="vLLM version; empty means unrecorded")
    p.add_argument(
        "--chat-template-sha256",
        default="",
        help="hash of the template the server applies; empty means unrecorded",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0, help="annotator seed recorded on each row")
    p.add_argument("--workers", type=int, default=1, help="concurrent annotation calls")
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
