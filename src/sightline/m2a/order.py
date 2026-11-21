"""Record the corpus permutation once, so every build reuses the same order."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def corpus_order(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return the records in a total, reproducible seeded permutation."""

    return random.Random(seed).sample(list(records), len(records))


def write_order(path: Path, records: list[dict[str, Any]], seed: int) -> None:
    """Pin the permutation as position/id rows so later runs never re-shuffle.

    The seed is part of the interface for callers that build and record an order
    together; the value used for a run is audited in the manifest, while this
    file is the record of the exact permutation itself.
    """

    with open(path, "w", encoding="utf-8") as handle:
        for position, record in enumerate(records):
            handle.write(json.dumps({"position": position, "id": record["id"]}) + "\n")


def read_order(path: Path) -> list[dict[str, Any]]:
    """Load a recorded permutation back into position/id rows."""

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_order(records: list[dict[str, Any]], order: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records in the recorded order, rejecting id mismatches."""

    positions = {entry["id"]: entry["position"] for entry in order}
    if {record["id"] for record in records} != set(positions):
        raise ValueError("record ids and order ids differ")
    return sorted(records, key=lambda record: positions[record["id"]])
