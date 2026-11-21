"""Decontaminate the M2a image corpus against MMMU-Pro and BLINK.

FineVision's 66-benchmark dedup covers neither benchmark [V — DECISIONS 0046],
so a COCO-only corpus carries no decontamination claim until it is checked here.
This module reports contaminated rows; it never deletes or edits corpus files.

Benchmark manifest format
-------------------------
A JSON object mapping benchmark name to a list of image sha256 values::

    {"MMMU-Pro": ["<sha256>", ...], "BLINK": ["<sha256>", ...]}

or a JSONL file with one such object per line (objects are merged). Only hashes
are stored: the images themselves are supplied separately, as a mapping from
sha256 to an on-disk path, when similarity matching is requested.

Detection has two layers:

- **Exact** sha256 membership is the baseline. It needs no image files and no
  model, so it always runs and reports ``similarity: 1.0``.
- **Similarity** is added only when a callable is injected. The callable has
  signature ``(Path, Path) -> float`` and is expected to be an SSCD cosine
  scorer; a result at or above ``SSCD_THRESHOLD`` is a hit. The exact SSCD model
  is not available locally, so tests pass a stub and the CLI requires the owner
  to supply the real callable.

[U] MMMU-Pro and BLINK images, and the revision they come from, are not pinned
yet (DECISIONS 0046). ``benchmark_images`` is the injection point for that set.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SSCD_THRESHOLD = 0.95
"""Cosine threshold fixed by DECISIONS 0046 for SSCD decontamination."""

REQUIRED_BENCHMARKS = ("MMMU-Pro", "BLINK")
"""Both benchmarks must be present; a missing one is a failure, not a clean run."""

SscdSimilarity = Callable[[Path, Path], float]


def load_benchmark_hashes(path: Path) -> dict[str, set[str]]:
    """Read a benchmark manifest as ``{name: {sha256, ...}}``.

    JSON and JSONL are both accepted so a manifest can be assembled incrementally
    one benchmark at a time without rewriting a single large object.
    """

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    try:
        objects = [json.loads(stripped)]
    except json.JSONDecodeError:
        objects = [json.loads(line) for line in text.splitlines() if line.strip()]

    result: dict[str, set[str]] = {}
    for obj in objects:
        for name, hashes in obj.items():
            result.setdefault(name, set()).update(hashes)
    return result


def find_contamination(
    records: list[dict[str, Any]],
    benchmark: Mapping[str, set[str]],
    *,
    sscd_similarity: SscdSimilarity | None = None,
    benchmark_images: Mapping[str, Path] | None = None,
    threshold: float = SSCD_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return one row per (record, benchmark) whose image is contaminated.

    Exact sha256 matches are reported first with ``similarity: 1.0``. When a
    similarity callable is supplied, the best SSCD score against that
    benchmark's images is reported for records with no exact match, provided it
    reaches ``threshold``. ``benchmark_images`` resolves the manifest's hashes to
    files; it is required for similarity because SSCD needs pixels, not hashes.
    """

    if sscd_similarity is not None and benchmark_images is None:
        raise ValueError("similarity requires benchmark_images (sha256 -> Path)")

    rows: list[dict[str, Any]] = []
    for record in records:
        sha = record["sha256"]
        record_path = Path(record["path"]) if record.get("path") else None
        for name in sorted(benchmark):
            hashes = benchmark[name]
            if sha in hashes:
                rows.append(
                    {"id": record["id"], "benchmark": name, "sha256": sha, "similarity": 1.0}
                )
                continue
            if sscd_similarity is None or record_path is None:
                continue
            best = 0.0
            for bench_sha in sorted(hashes):
                image_path = benchmark_images.get(bench_sha) if benchmark_images else None
                if image_path is None:
                    continue
                best = max(best, sscd_similarity(record_path, image_path))
            if best >= threshold:
                rows.append(
                    {"id": record["id"], "benchmark": name, "sha256": sha, "similarity": best}
                )
    return rows


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    n_records: int,
    *,
    checked_benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Write the hit rows as JSONL and return the run summary.

    ``checked_benchmarks`` makes the report auditable: passing the names that
    were checked keeps a zero-count benchmark visible, so a clean run proves both
    benchmarks were examined rather than one being silently skipped. Without it,
    the counts describe only the rows actually found.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    counts = Counter(row["benchmark"] for row in rows)
    if checked_benchmarks is None:
        per_benchmark = dict(counts)
    else:
        per_benchmark = {name: counts.get(name, 0) for name in checked_benchmarks}
    return {
        "n_records": n_records,
        "n_contaminated": len(rows),
        "per_benchmark": per_benchmark,
    }
