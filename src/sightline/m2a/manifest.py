"""Split the corpus without leakage, and describe it so the build is checkable.

Splitting is by hash, not by position: the M1 splits are ordered by answer and
taking the first N of an ordered file is a biased sample (docs/measurement-pitfalls.md).
The manifest carries the input hash so a build can be tied to the bytes it came
from, plus the token projection that decides whether a run fits the daily cap.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

TOKENS_PER_CONVERSATION = 3179
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def splits(records: list[dict[str, Any]], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    """Shuffle deterministically and cut into train/val/test without overlap."""

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop failed rows so they cannot enter a split or a token projection."""

    return [row for row in records if row.get("parsed") is not False]


def _cross_split_hash_collisions(records: list[dict[str, Any]], seed: int) -> int:
    """Count records whose hash lands in more than one split: a leakage signal."""

    hash_to_splits: dict[str, set[str]] = {}
    for name, part in splits(records, seed=seed).items():
        for row in part:
            digest = row.get("sha256")
            if digest is not None:
                hash_to_splits.setdefault(digest, set()).add(name)

    return sum(
        1
        for row in records
        if row.get("sha256") is not None and len(hash_to_splits[row["sha256"]]) > 1
    )


def manifest(path: Path, tokens_per_conversation: int = TOKENS_PER_CONVERSATION) -> dict[str, Any]:
    """Describe a conversation file: size, provenance, coverage, and token budget."""

    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    usable = usable_records(rows)

    licences: dict[str, int] = {}
    sources: dict[str, int] = {}
    annotators: dict[str, int] = {}
    categories: dict[str, int] = {}
    for row in usable:
        name = row.get("license_name") or "missing"
        licences[name] = licences.get(name, 0) + 1
        source = row.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
        annotator = row.get("model") or "missing"
        annotators[annotator] = annotators.get(annotator, 0) + 1
        category = row.get("category")
        if category is not None:
            categories[category] = categories.get(category, 0) + 1

    # The order seed is run-level provenance duplicated on each row; report the
    # first recorded value rather than trusting a caller-supplied default.
    order_seed: int | None = None
    for row in rows:
        recorded = row.get("order_seed")
        if recorded is not None:
            order_seed = recorded
            break

    # Collisions must describe the same split the build writes. build_m2a drives
    # splits from --seed, which is the value it records as order_seed on each row.
    split_seed = order_seed if order_seed is not None else 0

    result: dict[str, Any] = {
        "n_conversations": len(usable),
        "n_unparsed": len(rows) - len(usable),
        "input_sha256": file_sha256(path),
        "licences": dict(sorted(licences.items())),
        "sources": dict(sorted(sources.items())),
        "source_count": len(sources),
        "annotators": dict(sorted(annotators.items())),
        "order_seed": order_seed,
        "n_cross_split_hash_collisions": _cross_split_hash_collisions(usable, split_seed),
        "tokens_per_conversation": tokens_per_conversation,
        "tokens_projected": len(usable) * tokens_per_conversation,
    }
    if categories:
        result["category_coverage"] = dict(sorted(categories.items()))
    return result
