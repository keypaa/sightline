"""Identity fields that let every corpus row be traced to what produced it.

A claim id is a content hash rather than a sequence so that inserting or
removing a claim cannot renumber the ones after it — verdicts, blind results and
human decisions all join on it.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

CLAIM_ID_LENGTH = 16


def claim_id(image_id: int, turn: int, claim: str) -> str:
    """A short, stable id for one claim."""

    digest = hashlib.sha256(f"{image_id}:{turn}:{claim}".encode()).hexdigest()
    return digest[:CLAIM_ID_LENGTH]


def new_run_id() -> str:
    """A unique id for one annotate invocation, sortable by creation time."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    """The current time, timezone-aware, second precision."""

    return datetime.now(UTC).isoformat(timespec="seconds")
