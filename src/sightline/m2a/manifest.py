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


def manifest(path: Path, tokens_per_conversation: int = TOKENS_PER_CONVERSATION) -> dict[str, Any]:
    """Describe a conversation file: size, provenance, and the token budget."""

    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    licences: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in rows:
        name = row.get("license_name") or "missing"
        licences[name] = licences.get(name, 0) + 1
        source = row.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    return {
        "n_conversations": len(rows),
        "input_sha256": file_sha256(path),
        "licences": dict(sorted(licences.items())),
        "sources": dict(sorted(sources.items())),
        "tokens_per_conversation": tokens_per_conversation,
        "tokens_projected": len(rows) * tokens_per_conversation,
    }