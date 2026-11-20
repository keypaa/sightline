"""Decide which images may enter the corpus, and record why the rest did not.

COCO's dataset-level CC BY 4.0 covers the annotations; each photograph keeps its
own Flickr licence, recorded per image in the annotations file. So the gate reads
the *name* from the file rather than assuming the dataset card's headline.

The policy is: drop anything whose name carries ``nc`` (non-commercial) or ``nd``
(no-derivatives), and drop a missing name. A caller may pass an explicit
allow-list, but the NC/ND checks run first, so a loose allow-list cannot wave
through a non-commercial image.
"""

from __future__ import annotations

import re
from typing import Any


def _normalise(name: str | None) -> str:
    """Case-fold and collapse separators, so 'CC-BY 4.0' and 'cc by 4.0' match."""

    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def drop_reason(name: str | None) -> str | None:
    """Why this licence is unusable, or None when it passes the policy.

    COCO's own names are the test: ``Attribution-NonCommercial License``,
    ``Attribution-NonCommercial-NoDerivs License``, ``Attribution-NoDerivs
    License``. Matching only a bare ``nc`` or ``nd`` token misses all of them,
    because the words embed the letters.
    """

    if not name:
        return "missing-licence"
    squashed = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    compact = squashed.replace(" ", "")
    tokens = set(squashed.split())
    if "noncommercial" in compact or "nc" in tokens:
        return "non-commercial"
    if "noderivs" in compact or "noderivatives" in compact or "nd" in tokens:
        return "no-derivatives"
    return None


def classify(
    records: list[dict[str, Any]], allow: set[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into kept and dropped, annotating each drop with a reason.

    When ``allow`` is given, a name must be in it to be kept — but the NC/ND
    checks are applied first, so the policy cannot be bypassed.
    """

    allowed = {_normalise(name) for name in allow} if allow is not None else None

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        reason = drop_reason(record.get("license_name"))
        if reason is None and allowed is not None:
            if _normalise(record.get("license_name")) not in allowed:
                reason = "not-in-allow-list"
        if reason is None:
            kept.append(record)
        else:
            dropped.append({**record, "drop_reason": reason})
    return kept, dropped
